#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Oracle particle filter: the exact prior-proposal correction.

WSPF-A and WSPF-B approximate two quantities: the population gradient
grad L(theta) (an EMA in Method A, marginalized in Method B) and the
gradient-noise covariance Sigma(theta) (a sample estimate in Method A, a
scalar in Method B). This filter is given both exactly, so that

    oracle / Method A / Method B / uncorrected PF

can be compared under identical conditions (same data, same initial
particles, same eta, sigma_cd and sigma_0). That separates the validity of
the Gaussian correction itself from the error introduced by approximating it.

The true (grad L, Sigma) are supplied by the benchmark as `oracle_stats_fn`
(for the regression task they are Monte-Carlo estimates from the true
parameters and the generating model), so this module never needs to know the
generating model.

Exact correction:
    V_p = eta^2 Sigma + sigma_cd^2 I_d,   V_q = sigma_cd^2 I_d
    v = eps - dmu,   dmu = eta (ghat - grad L)
    log R = 0.5(log|V_q| - log|V_p|) + 0.5 eps^T V_q^-1 eps - 0.5 v^T V_p^-1 v
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


# ================================================================
# Exact correction (dense d-by-d, using the true Sigma)
# ================================================================
def compute_correction_oracle(epsilon, delta_mu, Sigma, eta, sigma_sys_sq, d):
    """Exact Gaussian correction given the true gradient-noise covariance.

    Parameters
    ----------
    epsilon : ndarray, shape (N, d)
        Realized concept-drift noise.
    delta_mu : ndarray, shape (N, d)
        Mean shift eta (ghat - grad L), using the true grad L.
    Sigma : ndarray, shape (N, d, d)
        True covariance of the batch gradient.
    eta : float
    sigma_sys_sq : float
        Concept-drift noise variance, the isotropic scale of V_q.
    d : int

    Returns
    -------
    log_correction : ndarray, shape (N,)
    nonfinite_count : int
    """
    N = epsilon.shape[0]
    c = sigma_sys_sq
    v = epsilon - delta_mu  # (N, d)

    eye = np.eye(d)
    Vp = eta ** 2 * Sigma + c * eye  # (N, d, d)

    sign, logdet_p = np.linalg.slogdet(Vp)  # (N,), (N,)
    logdet_q = d * np.log(c)

    # v^T V_p^-1 v, one dense solve per particle
    sol = np.linalg.solve(Vp, v[:, :, None])[:, :, 0]  # (N, d)
    quad_p = np.sum(v * sol, axis=1)  # (N,)

    quad_q = np.sum(epsilon ** 2, axis=1) / c

    log_correction_raw = 0.5 * (logdet_q - logdet_p) + 0.5 * quad_q - 0.5 * quad_p

    finite = np.isfinite(log_correction_raw) & (sign > 0)
    log_correction = np.where(finite, log_correction_raw, 0.0)
    nonfinite_count = int(np.sum(~finite))

    return log_correction, nonfinite_count


# ================================================================
# Filter
# ================================================================
class OraclePF:
    """Particle filter with the exact correction.

    System model:
        theta_t = theta_{t-1} - eta ghat(theta; B_t) + eps_t,
        eps_t ~ N(0, sigma_sys^2 I)

    Weight update (SIS):
        log w_t^i = log w_{t-1}^i + log p(B_t | theta_t^i) + log R^i,
    where log R^i uses the true (grad L, Sigma).
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
        grad_clip_norm=None,
        seed=None,
    ):
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.sigma_sys_sq = sigma_sys ** 2
        self.ess_resample_ratio = ess_resample_ratio
        self.grad_clip_norm = grad_clip_norm

        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        self.history = {
            "mean": [], "std": [], "ess": [], "ll_mean": [],
            # Denominator of the diagnostic R.
            "ll_std": [],
            "log_correction_mean": [],
            # Spread of the correction; theory predicts rho sqrt(d/2).
            "log_correction_std": [],
            "entropy": [], "max_weight": [], "spread_trace": [],
            "unique_particles": [], "resampled": [],
            "logcorr_nonfinite_count": [],
            "t_step": [], "t_grad": [], "t_loglik": [],
            "t_correction": [], "t_weight": [], "t_resample": [],
            "sample_grad_evals": [],
        }
        self.grad_eval_kind = "per_sample"
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    def step(self, X, y, per_sample_grad_fn, loglik_fn, oracle_stats_fn):
        """Advance one step with the exact correction.

        Parameters
        ----------
        per_sample_grad_fn : callable
            (particles[N, d], X, y) -> (N, B, d)
        loglik_fn : callable
            (particles[N, d], X, y) -> (N,)
        oracle_stats_fn : callable
            (particles[N, d], X, y) -> (grad_L(N, d), Sigma(N, d, d)),
            evaluated at the pre-update particles theta_{t-1} and supplied by
            the benchmark from the true parameters and generating model.
        """
        _t0 = time.perf_counter()

        per_grads = per_sample_grad_fn(self.particles, X, y)  # (N, B, d)
        g_hat = per_grads.mean(axis=1)  # (N, d)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(per_grads.shape[0] * per_grads.shape[1])
        self.sample_grad_evals_total += n_sample_grads

        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        grad_L, Sigma = oracle_stats_fn(self.particles, X, y)

        epsilon = self.rng.normal(0.0, self.sigma_sys, size=self.particles.shape)
        self.particles = self.particles - self.eta * g_hat + epsilon

        delta_mu = self.eta * (g_hat - grad_L)
        log_correction, nonfinite_count = compute_correction_oracle(
            epsilon, delta_mu, Sigma, self.eta, self.sigma_sys_sq, self.param_dim
        )
        _t_corr = time.perf_counter()

        ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()

        log_prev = np.log(np.maximum(self.weights, 1e-300))
        self.weights = normalize_logweights(log_prev + ll + log_correction)

        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(axis=0)
        std = np.sqrt(np.maximum(var, 1e-15))
        ll_mean = float((self.weights * ll).sum())
        _ll_finite = ll[np.isfinite(ll)]
        ll_std = float(np.std(_ll_finite)) if _ll_finite.size else float("nan")

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        _t_wt = time.perf_counter()

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
        self.history["log_correction_mean"].append(float(np.mean(log_correction)))
        self.history["log_correction_std"].append(
            float(np.std(log_correction[np.isfinite(log_correction)]))
            if np.any(np.isfinite(log_correction)) else float("nan"))
        self.history["entropy"].append(entropy)
        self.history["max_weight"].append(max_weight)
        self.history["spread_trace"].append(spread_trace)
        self.history["unique_particles"].append(n_unique)
        self.history["resampled"].append(resampled)
        self.history["logcorr_nonfinite_count"].append(nonfinite_count)
        self.history["t_grad"].append(_t_grad - _t0)
        self.history["t_correction"].append(_t_corr - _t_grad)
        self.history["t_loglik"].append(_t_ll - _t_corr)
        self.history["t_weight"].append(_t_wt - _t_ll)
        self.history["t_resample"].append(_t_rs - _t_wt)
        self.history["t_step"].append(_t_rs - _t0)
        self.history["sample_grad_evals"].append(n_sample_grads)

        return mean

    def get_history(self):
        return {k: np.array(v) for k, v in self.history.items()}
