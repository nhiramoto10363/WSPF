#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle 粒子フィルタ (WSPF-oracle) — 厳密な事前分布/提案分布補正 (R1-5)

WSPF-A/B が近似で置き換えている量:
  - 母集団勾配 ∇L(θ)         (WSPF-A は EMA で推定, WSPF-B は marginalize)
  - 勾配ノイズ共分散 Σ(θ)     (WSPF-A は標本 Σ̂, WSPF-B はスカラー s̄)
を **真値** で与えたときの厳密なガウス補正を計算する。これにより

  厳密補正(Oracle) / Method A 近似 / Method B 近似 / 補正なし PF

を同一条件(同じデータ・初期粒子・η・σ_cd・σ0)で比較でき、
「ガウス補正そのものの妥当性」と「近似による誤差」を切り分けられる(R1-5)。

真値 (∇L, Σ) はベンチマーク側が `oracle_stats_fn` として供給する
(回帰課題では真パラメータ θ* と生成モデルから MC 推定する)。filters/ は
生成モデルを知らずに厳密補正を評価できるよう、注入された関数を呼ぶだけにする。

厳密補正 (eq:exact_logR):
    V_p = η² Σ + σ²_cd I_d,   V_q = σ²_cd I_d,   v = ε − Δμ,  Δμ = η(ĝ − ∇L)
    log R = ½(log|V_q| − log|V_p|) + ½ εᵀ V_q⁻¹ ε − ½ vᵀ V_p⁻¹ v
"""

import time

import numpy as np
from .base import (
    normalize_logweights,
    effective_sample_size,
    systematic_resample,
    weight_diagnostics,
    ensemble_spread_trace,
)


# ================================================================
# 厳密補正項 (dense d×d, 真の Σ を使用)
# ================================================================
def compute_correction_oracle(epsilon, delta_mu, Sigma, eta, sigma_sys_sq, d):
    """
    真の勾配ノイズ共分散 Σ を用いた厳密なガウス補正。

    Parameters
    ----------
    epsilon : ndarray, shape (N, d)
        concept drift ノイズの実現値 ε
    delta_mu : ndarray, shape (N, d)
        平均シフト Δμ = η(ĝ − ∇L)  (真の ∇L を使用)
    Sigma : ndarray, shape (N, d, d)
        真のバッチ勾配ノイズ共分散 Cov(ĝ_batch)
    eta : float
        学習率
    sigma_sys_sq : float
        concept drift ノイズ分散 σ²_cd (= V_q の等方スケール c)
    d : int
        パラメータ次元数

    Returns
    -------
    log_correction : ndarray, shape (N,)
        厳密な log R^(i)
    nonfinite_count : int
        非有限値を中立値 0 でガードした粒子数(通常 0)
    """
    N = epsilon.shape[0]
    c = sigma_sys_sq
    v = epsilon - delta_mu  # (N, d)

    eye = np.eye(d)
    Vp = eta ** 2 * Sigma + c * eye  # (N, d, d)

    sign, logdet_p = np.linalg.slogdet(Vp)  # (N,), (N,)
    logdet_q = d * np.log(c)

    # vᵀ V_p⁻¹ v  (粒子ごとに dense solve)
    sol = np.linalg.solve(Vp, v[:, :, None])[:, :, 0]  # (N, d)
    quad_p = np.sum(v * sol, axis=1)  # (N,)

    quad_q = np.sum(epsilon ** 2, axis=1) / c  # εᵀ V_q⁻¹ ε

    log_correction_raw = 0.5 * (logdet_q - logdet_p) + 0.5 * quad_q - 0.5 * quad_p

    finite = np.isfinite(log_correction_raw) & (sign > 0)
    log_correction = np.where(finite, log_correction_raw, 0.0)
    nonfinite_count = int(np.sum(~finite))

    return log_correction, nonfinite_count


# ================================================================
# OraclePF: 厳密補正付き粒子フィルタ
# ================================================================
class OraclePF:
    """
    厳密補正 (Oracle) 付き粒子フィルタ。

    システムモデル:
        θ_t = θ_{t-1} − η ĝ(θ; B_t) + ε_t,  ε_t ~ N(0, σ_sys² I)

    重み更新 (SIS):
        log w_t^(i) = log w_{t-1}^(i) + log p(B_t | θ_t^(i)) + log R^(i)
    ここで log R^(i) は真の (∇L, Σ) を用いた厳密なガウス補正。
    """

    def __init__(
        self,
        n_particles,
        param_dim,
        eta=0.05,
        sigma_sys=0.02,
        prior_mean=0.0,
        prior_std=1.0,
        ess_resample_ratio=0.5,
        grad_clip_norm=None,
        seed=None,
    ):
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.sigma_sys_sq = sigma_sys ** 2
        self.ess_resample_ratio = ess_resample_ratio
        self.grad_clip_norm = grad_clip_norm

        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        self.history = {
            "mean": [], "std": [], "ess": [], "ll_mean": [],
            "log_correction_mean": [],
            "entropy": [], "max_weight": [], "spread_trace": [],
            "unique_particles": [], "resampled": [],
            "logcorr_nonfinite_count": [],
            "t_step": [], "t_grad": [], "t_loglik": [],
            "t_correction": [], "t_weight": [], "t_resample": [],
            "sample_grad_evals": [],
        }
        self.grad_eval_kind = "per_sample"
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    def step(self, X, y, per_sample_grad_fn, loglik_fn, oracle_stats_fn):
        """
        1 ステップの更新（厳密補正付き）。

        Parameters
        ----------
        per_sample_grad_fn : callable
            (particles[N, d], X, y) -> (N, B, d)  実ミニバッチの各サンプル勾配
        loglik_fn : callable
            (particles[N, d], X, y) -> (N,)
        oracle_stats_fn : callable
            (particles[N, d], X, y) -> (grad_L(N, d), Sigma(N, d, d))
            更新前粒子 θ_{t-1} における真の母集団勾配とバッチ勾配ノイズ共分散。
            ベンチマーク側が真パラメータ・生成モデルから供給する。
        """
        _t0 = time.perf_counter()

        per_grads = per_sample_grad_fn(self.particles, X, y)  # (N, B, d)
        g_hat = per_grads.mean(axis=1)  # (N, d)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(per_grads.shape[0] * per_grads.shape[1])
        self.sample_grad_evals_total += n_sample_grads

        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        # 真の母集団勾配 ∇L と共分散 Σ を更新前粒子で評価
        grad_L, Sigma = oracle_stats_fn(self.particles, X, y)

        # SGD 更新 + concept drift ノイズ
        epsilon = self.rng.normal(0.0, self.sigma_sys, size=self.particles.shape)
        self.particles = self.particles - self.eta * g_hat + epsilon

        # 厳密補正
        delta_mu = self.eta * (g_hat - grad_L)
        log_correction, nonfinite_count = compute_correction_oracle(
            epsilon, delta_mu, Sigma, self.eta, self.sigma_sys_sq, self.param_dim
        )
        _t_corr = time.perf_counter()

        ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()

        log_prev = np.log(np.maximum(self.weights, 1e-300))
        self.weights = normalize_logweights(log_prev + ll + log_correction)

        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-15))
        ll_mean = float((self.weights * ll).sum())

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        _t_wt = time.perf_counter()

        if ess < self.ess_resample_ratio * self.N:
            idx = systematic_resample(self.weights, self.rng)
            n_unique = int(np.unique(idx).size)
            self.particles = self.particles[idx]
            self.weights = np.ones(self.N) / self.N
            resampled = True
        else:
            n_unique = self.N
            resampled = False
        _t_rs = time.perf_counter()

        self.history["mean"].append(mean.copy())
        self.history["std"].append(std.copy())
        self.history["ess"].append(ess)
        self.history["ll_mean"].append(ll_mean)
        self.history["log_correction_mean"].append(float(np.mean(log_correction)))
        self.history["entropy"].append(entropy)
        self.history["max_weight"].append(max_weight)
        self.history["spread_trace"].append(spread_trace)
        self.history["unique_particles"].append(n_unique)
        self.history["resampled"].append(resampled)
        self.history["logcorr_nonfinite_count"].append(nonfinite_count)
        self.history["t_grad"].append(_t_grad - _t0)
        self.history["t_correction"].append(_t_corr - _t_grad)
        self.history["t_loglik"].append(_t_ll - _t_corr)
        self.history["t_weight"].append(_t_wt - _t_ll)
        self.history["t_resample"].append(_t_rs - _t_wt)
        self.history["t_step"].append(_t_rs - _t0)
        self.history["sample_grad_evals"].append(n_sample_grads)

        return mean

    def get_history(self):
        return {k: np.array(v) for k, v in self.history.items()}
