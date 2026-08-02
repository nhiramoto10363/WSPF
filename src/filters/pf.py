#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汎用粒子フィルタ (PF) 実装

SGDダイナミクスを用いたオンライン機械学習用
"""

import time

import numpy as np
from .base import (
    normalize_logweights,
    effective_sample_size,
    systematic_resample,
    weight_diagnostics,
    ensemble_spread_trace,
    init_phi,
    propagate_phi,
    phi_to_sigma,
    weighted_sigma_mean,
)


class ParticleFilter:
    """
    SGDダイナミクスを用いた粒子フィルタ

    システムモデル:
        θ_t = θ_{t-1} - η * grad(NLL) + w_t,  w_t ~ N(0, σ_sys² I)

    観測更新（重み付け）:
        w_t^i ∝ p(y_t | X_t, θ_t^i)
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
        seed=None,
        adaptive_obs=False,
        tau_phi=0.05,
        phi_init_mean=0.0,
        phi_init_std=0.5,
        phi_seed=None,
    ):
        """
        Parameters
        ----------
        n_particles : int
            粒子数
        param_dim : int
            パラメータの総次元数
        eta : float
            学習率
        sigma_sys : float
            システムノイズの標準偏差
        prior_mean : float
            事前分布の平均
        prior_std : float
            事前分布の標準偏差
        ess_resample_ratio : float
            リサンプリングを行うESSの閾値（粒子数に対する比率）
        seed : int, optional
            乱数シード
        adaptive_obs : bool
            True なら観測ノイズ φ_t = log σ_t² を粒子別状態として推定する
            (φ_t 拡張)。False なら従来と完全に同一挙動。
        tau_phi : float
            φ ランダムウォークの標準偏差 τ
        phi_init_mean : float
            φ_0 の初期平均 (通常 2 log σ_ref)
        phi_init_std : float
            φ_0 の初期標準偏差
        phi_seed : int, optional
            φ 専用 rng のシード (θ 系の乱数列と分離するため独立に持つ)
        """
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.ess_resample_ratio = ess_resample_ratio

        self.rng = np.random.default_rng(seed)

        # 粒子の初期化
        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        # --- φ_t 拡張: 粒子別観測ノイズ状態 ---
        self.adaptive_obs = bool(adaptive_obs)
        self.tau_phi = float(tau_phi)
        if self.adaptive_obs:
            self.phi_rng = np.random.default_rng(phi_seed)
            self.phi = init_phi(self.phi_rng, self.N,
                                phi_init_mean, phi_init_std)
        else:
            self.phi_rng = None
            self.phi = None

        # 履歴
        self.history = {
            "mean": [],
            "std": [],
            "ess": [],
            "ll_mean": [],
            # --- 縮退診断量 (R1-8, R1-9, R2-2) ---
            "entropy": [],           # 重みエントロピー −Σ w log w
            "max_weight": [],        # 最大正規化重み max_i w_i
            "spread_trace": [],      # パラメータ空間の ensemble spread (tr Cov)
            "unique_particles": [],  # リサンプリング後のユニーク祖先数
            "resampled": [],         # このステップでリサンプリングしたか
            # --- 計算コスト計測 (R1-11) ---
            "t_step": [],            # 1更新の総 wall-clock [s]
            "t_grad": [],            # 勾配評価の時間 [s]
            "t_loglik": [],          # 対数尤度評価の時間 [s]
            "t_correction": [],      # 伝播/補正項の時間 [s]
            "t_weight": [],          # 重み正規化+診断の時間 [s]
            "t_resample": [],        # リサンプリングの時間 [s]
            "sample_grad_evals": [], # このステップの勾配評価数(PFはバッチ勾配=N)
            # --- φ_t 拡張診断 (adaptive_obs 時のみ追記) ---
            "sigma_hat_mean": [],    # 重み付き平均 σ̂_t = Σ w_i exp(φ_i/2)
            "phi_std": [],           # 粒子間 φ の std (φ 縮退の監視)
        }
        # 勾配評価の種別と累積カウント (R1-11)
        self.grad_eval_kind = "batch"   # PF はバッチ平均勾配
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    @property
    def obs_sigma_particles(self):
        """粒子別観測ノイズ std σ^(i) = exp(φ^(i)/2)。非適応時は None。

        runner が評価 (混合 NLL / モーメント一致 std) に使用する。
        """
        if not self.adaptive_obs:
            return None
        return phi_to_sigma(self.phi)

    def step(self, X, y, grad_fn, loglik_fn, loglik_sigma_fn=None):
        """
        1ステップの更新

        Parameters
        ----------
        X : ndarray, shape (batch_size, input_dim)
            入力データ
        y : ndarray, shape (batch_size,) or (batch_size, output_dim)
            ラベル
        grad_fn : callable
            勾配計算関数: grad_fn(particles, X, y) -> (N, param_dim)
        loglik_fn : callable
            対数尤度計算関数: loglik_fn(particles, X, y) -> (N,)

        Returns
        -------
        mean : ndarray, shape (param_dim,)
            パラメータの推定値（重み付き平均）
        """
        _t0 = time.perf_counter()

        # 1) SGDステップ + ノイズ
        grad = grad_fn(self.particles, X, y)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(grad.shape[0])  # バッチ勾配: N 個
        self.sample_grad_evals_total += n_sample_grads

        self.particles = (
            self.particles
            - self.eta * grad
            + self.rng.normal(0.0, self.sigma_sys, size=self.particles.shape)
        )

        # 1b) φ 遷移 (adaptive_obs 時のみ; φ 専用 rng を消費するため
        #     θ 側の乱数列は非適応時と完全一致する)
        if self.adaptive_obs:
            self.phi = propagate_phi(self.phi, self.tau_phi, self.phi_rng)
        _t_corr = time.perf_counter()

        # 2) 尤度による重み更新（SIS の逐次累積: 前ステップの重みを保持）
        #    リサンプリングしないステップの重み情報を捨てないよう、WSPF と
        #    同じく log w_t = log w_{t-1} + log p(B_t | θ_t) を累積する。
        #    adaptive_obs 時は粒子別 σ^(i)=exp(φ^(i)/2) の尤度を使う。
        if self.adaptive_obs:
            if loglik_sigma_fn is None:
                raise ValueError(
                    "adaptive_obs=True には loglik_sigma_fn が必要です")
            ll = loglik_sigma_fn(self.particles, X, y, phi_to_sigma(self.phi))
        else:
            ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        self.weights = normalize_logweights(log_prev + ll)

        # 3) 推定値の計算（リサンプリング前）
        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-15))

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        ll_mean = float((self.weights * ll).sum())
        sigma_hat_mean = (weighted_sigma_mean(self.weights, self.phi)
                          if self.adaptive_obs else None)
        _t_wt = time.perf_counter()

        # 4) ESSが低い場合はリサンプリング
        if ess < self.ess_resample_ratio * self.N:
            idx = systematic_resample(self.weights, self.rng)
            n_unique = int(np.unique(idx).size)
            self.particles = self.particles[idx]
            if self.adaptive_obs:
                self.phi = self.phi[idx]   # φ も同一祖先で引き継ぐ
            self.weights = np.ones(self.N) / self.N
            resampled = True
        else:
            n_unique = self.N
            resampled = False
        _t_rs = time.perf_counter()

        # φ 診断 (adaptive_obs 時のみ; リサンプリング前の重みで σ̂ を評価
        # したいので weights リセット前に計算済みの値を使う)
        if self.adaptive_obs:
            self.history["sigma_hat_mean"].append(sigma_hat_mean)
            self.history["phi_std"].append(float(np.std(self.phi)))

        # 履歴に保存
        self.history["mean"].append(mean.copy())
        self.history["std"].append(std.copy())
        self.history["ess"].append(ess)
        self.history["ll_mean"].append(ll_mean)
        self.history["entropy"].append(entropy)
        self.history["max_weight"].append(max_weight)
        self.history["spread_trace"].append(spread_trace)
        self.history["unique_particles"].append(n_unique)
        self.history["resampled"].append(resampled)
        # 計算コスト (R1-11)
        self.history["t_grad"].append(_t_grad - _t0)
        self.history["t_correction"].append(_t_corr - _t_grad)
        self.history["t_loglik"].append(_t_ll - _t_corr)
        self.history["t_weight"].append(_t_wt - _t_ll)
        self.history["t_resample"].append(_t_rs - _t_wt)
        self.history["t_step"].append(_t_rs - _t0)
        self.history["sample_grad_evals"].append(n_sample_grads)

        return mean

    def run(self, X_list, y_list, grad_fn, loglik_fn):
        """
        全時刻での推定を実行

        Parameters
        ----------
        X_list : list of ndarray
            各時刻の入力データ
        y_list : list of ndarray
            各時刻のラベル
        grad_fn : callable
            勾配計算関数
        loglik_fn : callable
            対数尤度計算関数

        Returns
        -------
        means : ndarray, shape (T, param_dim)
            各時刻のパラメータ推定値
        """
        T = len(X_list)
        means = np.empty((T, self.param_dim))

        for t in range(T):
            means[t] = self.step(X_list[t], y_list[t], grad_fn, loglik_fn)

        return means

    def get_history(self):
        """履歴をndarrayで返す"""
        return {k: np.array(v) for k, v in self.history.items()}
