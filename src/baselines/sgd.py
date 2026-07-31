#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オンライン SGD ベースライン (点推定, R2-4 の基準)

粒子を持たず、単一のパラメータ点 θ を各バッチのバッチ平均勾配で更新する。
PF / WSPF と同一のデータ・評価窓・勾配関数(grad_fn)で比較できるよう、
共通の最小インタフェース(predict_theta / train / observe_error / n_resets)を
提供する。粒子数 N を持たない(R2-4)。

  θ_t = θ_{t-1} − η · ĝ(θ_{t-1}; B_t)          # バッチ平均勾配 = 平均(mean)

注) ミニバッチ対数尤度は「合計」、SGD 勾配は「平均」で扱う(査読 Minor 対応)。
"""

import numpy as np


class OnlineSGD:
    """
    素のオンライン SGD (検知器なし・窓なし・粒子なし)。

    Parameters
    ----------
    param_dim : int
        パラメータ次元数 d
    eta : float
        学習率
    prior_std : float
        初期化 θ ~ N(0, prior_std² I) の標準偏差 σ0
    grad_fn : callable
        grad_fn(theta[1, d], X, y) -> (1, d)  バッチ平均勾配
    seed : int
        乱数シード(初期化用)
    grad_clip_norm : float or None
        勾配クリッピングのノルム上限
    """

    def __init__(self, param_dim, eta, prior_std, grad_fn,
                 seed=0, grad_clip_norm=None):
        self.param_dim = param_dim
        self.eta = eta
        self.prior_std = prior_std
        self.grad_fn = grad_fn
        self.grad_clip_norm = grad_clip_norm
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.normal(0.0, prior_std, size=param_dim)
        self.n_resets = 0  # 検知器なし(比較表の体裁用)

    def predict_theta(self):
        return self.theta

    def _clip(self, g):
        if self.grad_clip_norm is not None:
            nrm = np.linalg.norm(g)
            if nrm > self.grad_clip_norm:
                g = g * (self.grad_clip_norm / (nrm + 1e-12))
        return g

    def train(self, X, y):
        g = self.grad_fn(self.theta.reshape(1, -1), X, y).reshape(-1)
        g = self._clip(g)
        self.theta = self.theta - self.eta * g

    def observe_error(self, err):
        return False  # 検知器なし
