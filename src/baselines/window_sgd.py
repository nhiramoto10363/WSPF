#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window-SGD : スライディング窓 SGD ベースライン
(adaptive sliding window の代表, R2-4)

修正方針 R2-4 の指摘への対応:
  通常の SGD も直近ミニバッチだけで更新するため、単に「最近のデータを使う」
  だけでは差別化にならない。そこで本ベースラインを次のように明確に定義する。

  - 直近 W 個のミニバッチ(サンプル)をバッファに保存
  - 各ステップでバッファ内を K 回パス学習する
      (K=1 なら「窓を 1 パス」、K>1 なら窓内で複数回反復)

探索対象は W, K, eta(いずれも選択区間だけで選ぶ)。粒子数 N は持たない。
"""

from collections import deque

import numpy as np


class WindowSGD:
    """
    スライディング窓 SGD。

    Parameters
    ----------
    param_dim : int
    eta : float                学習率
    prior_std : float          事前分布の標準偏差(初期化に使用) σ0
    grad_fn : callable         grad_fn(theta[1, d], X, y) -> (1, d) バッチ平均勾配
    window : int               保持する直近バッチ数 W
    n_passes : int             各ステップでバッファを学習するパス数 K
    seed : int
    grad_clip_norm : float or None
    """

    def __init__(self, param_dim, eta, prior_std, grad_fn,
                 window=5, n_passes=1, seed=0, grad_clip_norm=None):
        self.param_dim = param_dim
        self.eta = eta
        self.grad_fn = grad_fn
        self.grad_clip_norm = grad_clip_norm
        self.window = window
        self.n_passes = n_passes
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.normal(0.0, prior_std, size=param_dim)
        self.buffer = deque(maxlen=window)
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
        for _ in range(self.n_passes):          # 窓内を K パス
            for Xb, yb in self.buffer:
                self._sgd(Xb, yb)

    def observe_error(self, err):
        return False  # 検知器なし
