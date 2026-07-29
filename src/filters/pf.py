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
        }
        # 勾配評価の種別と累積カウント (R1-11)
        self.grad_eval_kind = "batch"   # PF はバッチ平均勾配
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    def step(self, X, y, grad_fn, loglik_fn):
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
        _t_corr = time.perf_counter()

        # 2) 尤度による重み更新
        ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()
        self.weights = normalize_logweights(ll)

        # 3) 推定値の計算（リサンプリング前）
        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-15))

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        ll_mean = float((self.weights * ll).sum())
        _t_wt = time.perf_counter()

        # 4) ESSが低い場合はリサンプリング
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
