#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Window-SGD: a sliding-window SGD baseline.

Plain SGD already updates from the most recent mini-batch only, so "use recent
data" alone would not distinguish this baseline. It is therefore defined as:
keep the last W mini-batches in a buffer, and take K passes over the buffer at
each step. W, K and eta are all selected on the selection window.
"""

from collections import deque

import numpy as np


class WindowSGD:
    """Sliding-window SGD.

    Parameters
    ----------
    param_dim : int
    eta : float                Learning rate.
    prior_std : float          Std of the initialization.
    grad_fn : callable         grad_fn(theta[1, d], X, y) -> (1, d).
    window : int               Number of recent batches kept, W.
    n_passes : int             Passes over the buffer per step, K.
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
        self.n_resets = 0

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
        for _ in range(self.n_passes):
            for Xb, yb in self.buffer:
                self._sgd(Xb, yb)

    def observe_error(self, err):
        return False
