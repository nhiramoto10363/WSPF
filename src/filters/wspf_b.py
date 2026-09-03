#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weighted SGD Particle Filter, Method B (WSPF-B).

Method B applies the prior-proposal correction through a scalar
approximation, so it never needs an estimate of the full-data gradient.

    prior transition: p(theta_t | theta_{t-1})
                      = N(theta_t | theta_{t-1} - eta grad L,
                          eta^2 Sigma(theta) + Q_cd)
    proposal:         q(theta_t | theta_{t-1}, B_t)
                      = N(theta_t | theta_{t-1} - eta ghat, Q_cd)
    correction:       R_t = p(theta_t | theta_{t-1})
                            / q(theta_t | theta_{t-1}, B_t)

    E_xi[log R_t] = -(d/2) log(1/(1-rho))
                    + rho ||eps||^2 / (2 sigma_cd^2)
                    - (d/2) rho

    rho = eta^2 sbar / (eta^2 sbar + sigma_cd^2)   signal-to-drift ratio
    sbar = tr(Sigma_hat) / d                       scalar batch-gradient noise

The three terms are a volume penalty (the prior is wider than the proposal),
a bonus for particles whose realized drift noise is large, and the offset
that makes the expectation come out right.
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

# Cap on rho, keeping the volume penalty finite (Algorithm 2 in the paper).
RHO_CLIP = 0.999

# ================================================================
# Correction term
# ================================================================
def compute_gradient_noise_variance(per_sample_grads):
    """Estimate the scalar noise variance of the *batch* gradient.

        ghat = (1/B) sum_j g_j
        Var(ghat) = C(theta) / B,  C(theta) = E[(g_j - grad L)(g_j - grad L)^T]
        sbar = tr(Var(ghat)) / d

    The unbiased estimator used here is

        shat = (1 / (d B (B-1))) sum_j ||g_j - ghat||^2,

    which equals tr(Sigma_hat)/d for
    Sigma_hat = (1/(B(B-1))) sum_j (g_j - ghat)(g_j - ghat)^T, so the scalar
    estimator of WSPF-B is consistent with the matrix one of WSPF-A.

    Parameters
    ----------
    per_sample_grads : ndarray, shape (N, B, d)
        Per-particle, per-sample NLL gradients.

    Returns
    -------
    g_hat : ndarray, shape (N, d)
    s_bar : ndarray, shape (N,)
    deviations : ndarray, shape (N, B, d)
        g_j - ghat, reused by WSPF-A to build Sigma_hat.
    """
    N, B, d = per_sample_grads.shape

    g_hat = per_sample_grads.mean(axis=1)  # (N, d)

    deviations = per_sample_grads - g_hat[:, np.newaxis, :]  # (N, B, d)

    s_bar = np.sum(deviations ** 2, axis=(1, 2)) / (B * (B - 1) * d)  # (N,)

    return g_hat, s_bar, deviations


def compute_correction_method_b(epsilon, eta, s_bar, sigma_sys_sq, d):
    """Scalar (Method B) importance-weight correction.

    Parameters
    ----------
    epsilon : ndarray, shape (N, d)
        Realized concept-drift noise.
    eta : float
    s_bar : ndarray, shape (N,)
        Scalar batch-gradient noise variance per particle.
    sigma_sys_sq : float
        System-noise variance sigma_cd^2.
    d : int

    Returns
    -------
    log_correction : ndarray, shape (N,)
    rho : ndarray, shape (N,)
    nonfinite_count : int
    """
    eta_sq = eta ** 2
    s_bar_safe = np.maximum(s_bar, 1e-30)
    rho_raw = eta_sq * s_bar_safe / (eta_sq * s_bar_safe + sigma_sys_sq)
    rho = np.minimum(rho_raw, RHO_CLIP)

    eps_norm_sq = np.sum(epsilon ** 2, axis=1)  # (N,)

    # All three terms use the clipped rho.
    # Volume penalty: -(d/2) log(1/(1-rho)) = (d/2) log(1-rho)
    term1 = 0.5 * d * np.log1p(-rho)

    # Bonus for the realized noise
    term2 = rho * eps_norm_sq / (2.0 * sigma_sys_sq)

    # Expectation offset
    term3 = -0.5 * d * rho

    log_correction_raw = term1 + term2 + term3

    # The rho clip already bounds the volume penalty, so only non-finite
    # values are guarded here (a hard +-d/2 clamp would deform the
    # correction and would fire constantly at small sigma_cd).
    finite = np.isfinite(log_correction_raw)
    log_correction = np.where(finite, log_correction_raw, 0.0)
    nonfinite_count = int(np.sum(~finite))

    return log_correction, rho, nonfinite_count


# ================================================================
# Filter
# ================================================================
class WSPF_B:
    """Weighted SGD particle filter, Method B.

    System model:
        theta_t = theta_{t-1} - eta ghat(theta; B_t) + eps_t,
        eps_t ~ N(0, sigma_sys^2 I)

    Weight update (SIS):
        log w_t^i = log w_{t-1}^i + log p(B_t | theta_t^i) + log R_t^i

    Systematic resampling with weight reset whenever ESS falls below the
    threshold.
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
        """
        Parameters
        ----------
        n_particles : int
        param_dim : int
        eta : float
            Learning rate.
        sigma_sys : float
            System-noise std, sigma_cd.
        prior_mean, prior_std : float
            Initialization of the particles.
        ess_resample_ratio : float
            Resample when ESS falls below this fraction of N.
        grad_clip_norm : float or None
        seed : int, optional
        """
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
            "mean": [],
            "std": [],
            "ess": [],
            "ll_mean": [],
            # Denominator of the diagnostic R = rho sqrt(d/2) / sd(loglik).
            "ll_std": [],
            "rho_mean": [],
            "rho_max": [],
            "log_correction_mean": [],
            # Spread of the correction across particles; theory predicts
            # rho sqrt(d/2), since ||eps||^2 / sigma^2 ~ chi^2_d.
            "log_correction_std": [],
            # Degeneracy diagnostics
            "entropy": [],
            "max_weight": [],
            "spread_trace": [],
            "unique_particles": [],
            "resampled": [],
            # Method B specific
            "rho": [],               # per-step, per-particle rho, shape (N,)
            "rho_clip_count": [],    # particles that hit RHO_CLIP
            "logcorr_nonfinite_count": [],
            # Timing, in seconds
            "t_step": [],
            "t_grad": [],
            "t_loglik": [],
            "t_correction": [],
            "t_weight": [],
            "t_resample": [],
            "sample_grad_evals": [],   # N * B per step
        }
        self.grad_eval_kind = "per_sample"
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    def step(self, X, y, per_sample_grad_fn, loglik_fn):
        """Advance one step and return the weighted-mean parameter.

        Parameters
        ----------
        X : ndarray, shape (B, input_dim)
        y : ndarray, shape (B,) or (B, output_dim)
        per_sample_grad_fn : callable
            per_sample_grad_fn(particles, X, y) -> (N, B, param_dim)
        loglik_fn : callable
            loglik_fn(particles, X, y) -> (N,)

        Returns
        -------
        mean : ndarray, shape (param_dim,)
        """
        _t0 = time.perf_counter()

        # 1) Per-sample gradients -> batch mean and noise variance
        per_grads = per_sample_grad_fn(self.particles, X, y)  # (N, B, d)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(per_grads.shape[0] * per_grads.shape[1])  # N*B
        self.sample_grad_evals_total += n_sample_grads

        g_hat, s_bar, _ = compute_gradient_noise_variance(per_grads)

        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        # 2) SGD move plus concept-drift noise
        epsilon = self.rng.normal(
            0.0, self.sigma_sys, size=self.particles.shape
        )
        self.particles = self.particles - self.eta * g_hat + epsilon

        # 3) Correction
        log_correction, rho, logcorr_nonfinite_count = compute_correction_method_b(
            epsilon, self.eta, s_bar, self.sigma_sys_sq, self.param_dim
        )
        _t_corr = time.perf_counter()

        # 4) Log-likelihood
        ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()

        # 5) Accumulated weights: log w_t = log w_{t-1} + increments
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        log_weights = log_prev + ll + log_correction
        self.weights = normalize_logweights(log_weights)

        # 6) Weighted estimate
        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(
            axis=0
        )
        std = np.sqrt(np.maximum(var, 1e-15))
        ll_mean = float((self.weights * ll).sum())
        _ll_finite = ll[np.isfinite(ll)]
        ll_std = float(np.std(_ll_finite)) if _ll_finite.size else float("nan")

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        rho_clip_count = int(np.sum(rho >= RHO_CLIP))
        _t_wt = time.perf_counter()

        # 7) Resample on low ESS
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
        self.history["rho_mean"].append(float(np.mean(rho)))
        self.history["rho_max"].append(float(np.max(rho)))
        self.history["log_correction_mean"].append(
            float(np.mean(log_correction))
        )
        self.history["log_correction_std"].append(
            float(np.std(log_correction[np.isfinite(log_correction)]))
            if np.any(np.isfinite(log_correction)) else float("nan"))
        self.history["entropy"].append(entropy)
        self.history["max_weight"].append(max_weight)
        self.history["spread_trace"].append(spread_trace)
        self.history["unique_particles"].append(n_unique)
        self.history["resampled"].append(resampled)
        self.history["rho"].append(rho.copy())
        self.history["rho_clip_count"].append(rho_clip_count)
        self.history["logcorr_nonfinite_count"].append(logcorr_nonfinite_count)
        self.history["t_grad"].append(_t_grad - _t0)
        self.history["t_correction"].append(_t_corr - _t_grad)
        self.history["t_loglik"].append(_t_ll - _t_corr)
        self.history["t_weight"].append(_t_wt - _t_ll)
        self.history["t_resample"].append(_t_rs - _t_wt)
        self.history["t_step"].append(_t_rs - _t0)
        self.history["sample_grad_evals"].append(n_sample_grads)

        return mean

    def run(self, X_list, y_list, per_sample_grad_fn, loglik_fn):
        """Run over a whole stream and return the per-step estimates."""
        T = len(X_list)
        means = np.empty((T, self.param_dim))
        for t in range(T):
            means[t] = self.step(
                X_list[t], y_list[t], per_sample_grad_fn, loglik_fn
            )
        return means

    def get_history(self):
        """Return the recorded history as arrays."""
        return {k: np.array(v) for k, v in self.history.items()}
