#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online SGD baseline: a single parameter point, no particles.

    theta_t = theta_{t-1} - eta * ghat(theta_{t-1}; B_t)

The mini-batch log-likelihood is a sum, while the SGD gradient is a mean.
"""

import numpy as np


class OnlineSGD:
    """Plain online SGD (no drift detector, no window, no particles).

    Parameters
    ----------
    param_dim : int
        Parameter dimension d.
    eta : float
        Learning rate.
    prior_std : float
        Std of the initialization theta ~ N(0, prior_std^2 I).
    grad_fn : callable
        grad_fn(theta[1, d], X, y) -> (1, d), the batch-mean gradient.
    seed : int
    grad_clip_norm : float or None
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
        self.n_resets = 0

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
        return False
