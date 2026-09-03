#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Particle filter with SGD dynamics and likelihood-only weighting.

This is the uncorrected baseline the paper compares against.
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
    """Particle filter whose transition is an SGD step.

    System model:
        theta_t = theta_{t-1} - eta * grad(NLL) + w_t,  w_t ~ N(0, sigma_sys^2 I)

    Weight update:
        w_t^i  propto  p(y_t | X_t, theta_t^i)
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
        param_dim : int
        eta : float
            Learning rate.
        sigma_sys : float
            System-noise std.
        prior_mean, prior_std : float
            Initialization of the particles.
        ess_resample_ratio : float
            Resample when ESS falls below this fraction of N.
        seed : int, optional
        """
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.ess_resample_ratio = ess_resample_ratio

        self.rng = np.random.default_rng(seed)

        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        self.history = {
            "mean": [],
            "std": [],
            "ess": [],
            "ll_mean": [],
            # Spread of the log-likelihood across particles: the denominator
            # of the diagnostic R = rho * sqrt(d/2) / sd(loglik).
            "ll_std": [],
            # Degeneracy diagnostics
            "entropy": [],           # -sum w log w
            "max_weight": [],
            "spread_trace": [],      # ensemble spread, tr Cov
            "unique_particles": [],  # unique ancestors after resampling
            "resampled": [],
            # Timing, in seconds
            "t_step": [],
            "t_grad": [],
            "t_loglik": [],
            "t_correction": [],
            "t_weight": [],
            "t_resample": [],
            "sample_grad_evals": [],
        }
        self.grad_eval_kind = "batch"
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    def step(self, X, y, grad_fn, loglik_fn):
        """Advance one step and return the weighted-mean parameter.

        Parameters
        ----------
        X : ndarray, shape (batch_size, input_dim)
        y : ndarray, shape (batch_size,) or (batch_size, output_dim)
        grad_fn : callable
            grad_fn(particles, X, y) -> (N, param_dim)
        loglik_fn : callable
            loglik_fn(particles, X, y) -> (N,)

        Returns
        -------
        mean : ndarray, shape (param_dim,)
        """
        _t0 = time.perf_counter()

        # 1) SGD move plus system noise
        grad = grad_fn(self.particles, X, y)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(grad.shape[0])
        self.sample_grad_evals_total += n_sample_grads

        self.particles = (
            self.particles
            - self.eta * grad
            + self.rng.normal(0.0, self.sigma_sys, size=self.particles.shape)
        )
        _t_corr = time.perf_counter()

        # 2) Likelihood weighting, accumulated as in SIS:
        #    log w_t = log w_{t-1} + log p(B_t | theta_t), so that weight
        #    information from steps without resampling is not discarded.
        ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        self.weights = normalize_logweights(log_prev + ll)

        # 3) Estimate before resampling
        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-15))

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        ll_mean = float((self.weights * ll).sum())
        # Unweighted spread: the scatter of the particles drawn from the
        # proposal. Non-finite values are dropped first.
        _ll_finite = ll[np.isfinite(ll)]
        ll_std = float(np.std(_ll_finite)) if _ll_finite.size else float("nan")
        _t_wt = time.perf_counter()

        # 4) Resample on low ESS
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
        self.history["ll_std"].append(ll_std)
        self.history["entropy"].append(entropy)
        self.history["max_weight"].append(max_weight)
        self.history["spread_trace"].append(spread_trace)
        self.history["unique_particles"].append(n_unique)
        self.history["resampled"].append(resampled)
        self.history["t_grad"].append(_t_grad - _t0)
        self.history["t_correction"].append(_t_corr - _t_grad)
        self.history["t_loglik"].append(_t_ll - _t_corr)
        self.history["t_weight"].append(_t_wt - _t_ll)
        self.history["t_resample"].append(_t_rs - _t_wt)
        self.history["t_step"].append(_t_rs - _t0)
        self.history["sample_grad_evals"].append(n_sample_grads)

        return mean

    def run(self, X_list, y_list, grad_fn, loglik_fn):
        """Run over a whole stream and return the per-step estimates."""
        T = len(X_list)
        means = np.empty((T, self.param_dim))

        for t in range(T):
            means[t] = self.step(X_list[t], y_list[t], grad_fn, loglik_fn)

        return means

    def get_history(self):
        """Return the recorded history as arrays."""
        return {k: np.array(v) for k, v in self.history.items()}
