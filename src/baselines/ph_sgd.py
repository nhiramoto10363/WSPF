#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PH-SGD : Page-Hinkley でドリフト検知して θ をリセットする SGD ベースライン
(drift-detector-triggered reset / adaptive learning の代表, R2-4)

再現性のため定義を明示する(修正方針 R2-4):
  - Page-Hinkley が監視する量 : 予測誤差スカラー err のストリーム(平均レベル)
  - δ (delta)                : 許容する緩やかなドリフト(小さいほど敏感)
  - 閾値 λ (lambda_)          : PH_T = m_T − min m_t がこれを超えたら検知
  - forgetting factor α       : m_t の指数的忘却(1 に近いほど長期記憶)
  - 検出時のリセット対象      : 点推定 θ
  - リセット後の初期値        : θ ~ N(0, prior_std² I)
  - 検出回数                  : self.n_resets

粒子数 N は持たない(R2-4)。
"""

import numpy as np


class PageHinkley:
    """
    Page-Hinkley 検定(平均の上昇方向の変化を検知)。

    誤差平均の逐次推定 x̄_t を追跡し、累積偏差 m_T の最小値からの乖離
    PH_T = m_T − min m_t が λ を超えたら変化検知。

    Parameters
    ----------
    delta : float
        許容する平均の緩やかなドリフト(小さいほど敏感)
    lambda_ : float
        検知閾値
    alpha : float
        忘却係数(1 に近いほど長期記憶)
    """

    def __init__(self, delta=0.005, lambda_=5.0, alpha=0.9999):
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.n = 0
        self.x_mean = 0.0
        self.m_t = 0.0
        self.min_m = 0.0

    def update(self, x):
        x = float(x)
        self.n += 1
        self.x_mean += (x - self.x_mean) / self.n
        self.m_t = self.alpha * self.m_t + (x - self.x_mean - self.delta)
        self.min_m = min(self.min_m, self.m_t)
        ph = self.m_t - self.min_m
        if ph > self.lambda_:
            self.reset()
            return True
        return False


class PHSGD:
    """
    Page-Hinkley ドリフト検知 + θ リセット付きオンライン SGD。

    Parameters
    ----------
    param_dim : int
    eta : float                学習率
    prior_std : float          事前分布の標準偏差(初期化・リセットに使用) σ0
    grad_fn : callable         grad_fn(theta[1, d], X, y) -> (1, d) バッチ平均勾配
    seed : int
    grad_clip_norm : float or None
    ph_delta, ph_lambda, ph_alpha : float
        Page-Hinkley のハイパーパラメータ(選択区間だけで選ぶ)
    """

    def __init__(self, param_dim, eta, prior_std, grad_fn,
                 seed=0, grad_clip_norm=None,
                 ph_delta=0.005, ph_lambda=5.0, ph_alpha=0.9999):
        self.param_dim = param_dim
        self.eta = eta
        self.prior_std = prior_std
        self.grad_fn = grad_fn
        self.grad_clip_norm = grad_clip_norm
        self.detector = PageHinkley(delta=ph_delta, lambda_=ph_lambda,
                                    alpha=ph_alpha)
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.normal(0.0, prior_std, size=param_dim)
        self.n_resets = 0

    def predict_theta(self):
        return self.theta

    def train(self, X, y):
        g = self.grad_fn(self.theta.reshape(1, -1), X, y).reshape(-1)
        if self.grad_clip_norm is not None:
            nrm = np.linalg.norm(g)
            if nrm > self.grad_clip_norm:
                g = g * (self.grad_clip_norm / (nrm + 1e-12))
        self.theta = self.theta - self.eta * g

    def observe_error(self, err):
        """誤差スカラーを検知器へ。ドリフト検知時は θ をリセットして True。"""
        if self.detector.update(err):
            self.theta = self.rng.normal(0.0, self.prior_std,
                                         size=self.param_dim)
            self.n_resets += 1
            return True
        return False
