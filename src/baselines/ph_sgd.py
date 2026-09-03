#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PH-SGD: online SGD that resets theta when Page-Hinkley detects drift.

The detector watches the stream of scalar prediction errors. On detection,
theta is re-drawn from N(0, prior_std^2 I) and n_resets is incremented.
"""

import numpy as np


class PageHinkley:
    """Page-Hinkley test for an upward shift in the mean.

    Tracks the running mean of the monitored scalar and flags a change when
    PH_T = m_T - min_t m_t exceeds lambda_.

    Parameters
    ----------
    delta : float
        Tolerated slow drift in the mean; smaller is more sensitive.
    lambda_ : float
        Detection threshold.
    alpha : float
        Forgetting factor; closer to 1 means longer memory.
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
    """Online SGD with Page-Hinkley drift detection and reset.

    Parameters
    ----------
    param_dim : int
    eta : float                Learning rate.
    prior_std : float          Std used for initialization and for resets.
    grad_fn : callable         grad_fn(theta[1, d], X, y) -> (1, d).
    seed : int
    grad_clip_norm : float or None
    ph_delta, ph_lambda, ph_alpha : float
        Page-Hinkley hyper-parameters, selected on the selection window.
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
        """Feed one error scalar; reset theta and return True on detection."""
        if self.detector.update(err):
            self.theta = self.rng.normal(0.0, self.prior_std,
                                         size=self.param_dim)
            self.n_resets += 1
            return True
        return False
