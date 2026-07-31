#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドリフト検知リセット付き SGD — R2-4 ベースライン

オンライン SGD で点推定 θ を更新し、予測誤差ストリームをドリフト検知器
(ADWIN / Page-Hinkley)に投入する。ドリフトを検知したら θ を事前分布から
再サンプルして学習を再スタートする(drift-detector-triggered reset /
adaptive sliding window の代表的挙動)。

WSPF / PF と同一のデータ・評価窓・勾配関数で比較できるよう、状態と更新を
最小限のインタフェースで提供する。
"""

from collections import deque

import numpy as np


class WindowSGD:
    """
    スライディング窓 SGD (adaptive sliding window の代表的挙動, R2-4)。

    直近 W バッチのみを保持し、各ステップで窓内バッチを 1 パス学習することで
    古い概念を忘却し、最近のデータに適応する(検知器不要)。
    """

    def __init__(self, param_dim, eta, prior_std, grad_fn, window=5,
                 seed=0, grad_clip_norm=None):
        self.param_dim = param_dim
        self.eta = eta
        self.grad_fn = grad_fn
        self.grad_clip_norm = grad_clip_norm
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.normal(0.0, prior_std, size=param_dim)
        self.buffer = deque(maxlen=window)
        self.window = window
        self.n_resets = 0  # 窓法は reset しない(比較表の体裁用)

    def predict_theta(self):
        return self.theta

    def _sgd(self, X, y):
        g = self.grad_fn(self.theta.reshape(1, -1), X, y).reshape(-1)
        if self.grad_clip_norm is not None:
            nrm = np.linalg.norm(g)
            if nrm > self.grad_clip_norm:
                g = g * (self.grad_clip_norm / (nrm + 1e-12))
        self.theta = self.theta - self.eta * g

    def train(self, X, y):
        self.buffer.append((np.asarray(X), np.asarray(y)))
        for Xb, yb in self.buffer:      # 窓内を 1 パス
            self._sgd(Xb, yb)

    def observe_error(self, err):
        return False  # 検知器なし


class DriftResetSGD:
    def __init__(self, param_dim, eta, prior_std, grad_fn, detector,
                 seed=0, grad_clip_norm=None):
        """
        Parameters
        ----------
        param_dim : int
        eta : float                学習率
        prior_std : float          事前分布の標準偏差(初期化・リセットに使用)
        grad_fn : callable         grad_fn(theta[1,d], X, y) -> (1, d) バッチ勾配
        detector : object          update(err)->bool を持つドリフト検知器
        grad_clip_norm : float or None
        """
        self.param_dim = param_dim
        self.eta = eta
        self.prior_std = prior_std
        self.grad_fn = grad_fn
        self.detector = detector
        self.grad_clip_norm = grad_clip_norm
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
