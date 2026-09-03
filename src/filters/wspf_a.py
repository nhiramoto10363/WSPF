#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weighted SGD Particle Filter, Method A (WSPF-A, matrix form).

Method A keeps the gradient-noise covariance as a full matrix, so the
correction sees its anisotropy. Where Method B collapses the noise to a
scalar and marginalizes xi, Method A estimates the full-data gradient with an
EMA and uses xi_hat = ghat - gtilde explicitly.

    gtilde_t = m_{t-1} / (1 - beta^{t-1})              EMA estimate of grad L
    xi_hat_t = ghat_t - gtilde_t,   dmu_hat = eta xi_hat
    Sigma_hat_t = (1/(B(B-1))) sum_j (g_j - ghat)(g_j - ghat)^T
    Vp_hat = eta^2 Sigma_hat + Q_cd,  Q_cd = sigma_cd^2 I_d = Vq

    log R_A = 0.5 log(|Vq| / |Vp_hat|) + 0.5 eps^T Vq^-1 eps
              - 0.5 (eps - dmu_hat)^T Vp_hat^-1 (eps - dmu_hat)

    m_t = beta m_{t-1} + (1 - beta) ghat_t

Vp_hat = c I_d + U U^T (c = sigma_cd^2, U = sqrt(alpha) W^T,
alpha = eta^2/(B(B-1)), W the B-by-d deviation matrix) is low rank, so the
Woodbury identity and the matrix determinant lemma avoid inverting a d-by-d
matrix:

    M = I_B + c^-1 (alpha W W^T)                       B-by-B, SPD, eigs >= 1
    log|Vp_hat| = d log c + log|M|
    (eps - dmu_hat)^T Vp_hat^-1 (eps - dmu_hat)
        = c^-1 (||v||^2 - c^-1 p^T M^-1 p),   p = sqrt(alpha) W v

This costs O(dB^2 + B^3) per particle instead of O(d^3). Because the
eigenvalues of M are at least one, its Cholesky factor is stable; jitter is
added only on failure, and the conditioning is monitored.
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
from .wspf_b import compute_gradient_noise_variance, RHO_CLIP


# ================================================================
# Correction term
# ================================================================
def compute_correction_method_a(epsilon, xi_hat, deviations, eta,
                                sigma_sys_sq, d):
    """Method A correction, using the full covariance Sigma_hat.

    Parameters
    ----------
    epsilon : ndarray, shape (N, d)
        Realized concept-drift noise.
    xi_hat : ndarray, shape (N, d)
        Estimated mini-batch noise, ghat - gtilde.
    deviations : ndarray, shape (N, B, d)
        g_j - ghat, used to build Sigma_hat.
    eta : float
    sigma_sys_sq : float
        Concept-drift noise variance, the isotropic scale of Vq.
    d : int

    Returns
    -------
    log_correction : ndarray, shape (N,)
    rho : ndarray, shape (N,)
        Scalar-equivalent signal-to-drift ratio, for diagnostics only.
    logcorr_nonfinite_count : int
    cond_M : ndarray, shape (N,)
        Conditioning of M, monitored for numerical stability.
    jitter_count : int
        Particles that needed a jitter fallback in the Cholesky.
    """
    N, B, dd = deviations.shape
    c = sigma_sys_sq
    alpha = eta ** 2 / (B * (B - 1))

    v = epsilon - eta * xi_hat  # (N, d)

    # G = alpha W W^T, M = I_B + c^-1 G  (SPD, eigenvalues >= 1)
    G = alpha * np.einsum("nbd,ncd->nbc", deviations, deviations)
    M = G / c
    diag = np.arange(B)
    M[:, diag, diag] += 1.0

    # p = sqrt(alpha) W v
    p = np.sqrt(alpha) * np.einsum("nbd,nd->nb", deviations, v)

    # Try the fast batched Cholesky; on failure fall back per particle and
    # add jitter only to the particles that need it.
    jitter_count = 0
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        L = np.empty_like(M)
        for i in range(N):
            try:
                L[i] = np.linalg.cholesky(M[i])
            except np.linalg.LinAlgError:
                jit = 1e-8 * np.mean(np.diagonal(M[i]))
                M[i, diag, diag] += jit
                L[i] = np.linalg.cholesky(M[i])
                jitter_count += 1

    # log|M| = 2 sum log diag(L); the diagonal is reused for conditioning.
    Ldiag = np.diagonal(L, axis1=1, axis2=2)  # (N, B)
    logdet_M = 2.0 * np.sum(np.log(Ldiag), axis=1)  # (N,)

    # Solve M^-1 p by reusing the same factor L (forward/back substitution)
    # rather than calling solve(M, ...) again.
    u = np.linalg.solve(L, p[:, :, None])                      # (N, B, 1)
    z = np.linalg.solve(np.swapaxes(L, -1, -2), u)[:, :, 0]    # (N, B)
    pMp = np.sum(p * z, axis=1)  # p^T M^-1 p

    v_norm_sq = np.sum(v ** 2, axis=1)  # (N,)
    quad_p = (v_norm_sq - pMp / c) / c

    eps_norm_sq = np.sum(epsilon ** 2, axis=1)  # (N,)
    quad_q = eps_norm_sq / c

    # log R_A = 0.5(d log c - log|Vp_hat|) + 0.5 quad_q - 0.5 quad_p
    #         = -0.5 log|M| + 0.5 quad_q - 0.5 quad_p   (d log c cancels)
    log_correction_raw = -0.5 * logdet_M + 0.5 * quad_q - 0.5 * quad_p

    # Scalar-equivalent rho for diagnostics, with sbar = tr(Sigma_hat)/d.
    s_bar = np.sum(deviations ** 2, axis=(1, 2)) / (B * (B - 1) * d)
    rho = eta ** 2 * s_bar / (eta ** 2 * s_bar + c)

    # Cheap proxy for the condition number (squared ratio of Cholesky
    # diagonals). It reuses L, so it adds no SVD cost to the timings.
    cond_M = (Ldiag.max(axis=1) /
              np.maximum(Ldiag.min(axis=1), 1e-300)) ** 2  # (N,)

    finite = np.isfinite(log_correction_raw)
    log_correction = np.where(finite, log_correction_raw, 0.0)
    nonfinite_count = int(np.sum(~finite))

    return log_correction, rho, nonfinite_count, cond_M, jitter_count


# ================================================================
# Filter
# ================================================================
class WSPF_A:
    """Weighted SGD particle filter, Method A (EMA plug-in).

    Compared with Method B, each particle carries an EMA state used to
    estimate the full-data gradient, the mini-batch noise enters the
    correction explicitly, and the EMA state is duplicated on resampling.

    System model:
        theta_t = theta_{t-1} - eta ghat(theta; B_t) + eps_t,
        eps_t ~ N(0, sigma_sys^2 I)

    Weight update (SIS):
        log w_t^i = log w_{t-1}^i + log p(B_t | theta_t^i) + log R_A^i
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
        beta=0.9,
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
        beta : float
            EMA forgetting factor in [0, 1).
        seed : int, optional
        """
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.sigma_sys_sq = sigma_sys ** 2
        self.ess_resample_ratio = ess_resample_ratio
        self.grad_clip_norm = grad_clip_norm
        self.beta = beta

        self.rng = np.random.default_rng(seed)

        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        # Per-particle EMA of the batch gradient
        self.ema_m = np.zeros((self.N, param_dim))
        self.t_step = 0

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
            "rho": [],
            "rho_clip_count": [],
            "logcorr_nonfinite_count": [],
            # Conditioning of the low-rank solve
            "cond_M_mean": [],
            "cond_M_max": [],
            "jitter_count": [],
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
        self.t_step += 1

        _t0 = time.perf_counter()

        # 1) Per-sample gradients -> batch mean and deviations
        per_grads = per_sample_grad_fn(self.particles, X, y)  # (N, B, d)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(per_grads.shape[0] * per_grads.shape[1])  # N*B
        self.sample_grad_evals_total += n_sample_grads

        g_hat, _s_bar, deviations = compute_gradient_noise_variance(per_grads)

        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        # 2) EMA estimate of the full-data gradient, formed before seeing B_t
        if self.t_step == 1:
            # No EMA yet, so xi_hat = 0 and the step matches Method B.
            g_tilde = g_hat.copy()
        else:
            bias_correction = 1.0 - self.beta ** (self.t_step - 1)
            g_tilde = self.ema_m / bias_correction  # (N, d)

        # 3) Estimated mini-batch noise
        xi_hat = g_hat - g_tilde  # (N, d)

        # 4) SGD move plus concept-drift noise
        epsilon = self.rng.normal(
            0.0, self.sigma_sys, size=self.particles.shape
        )
        self.particles = self.particles - self.eta * g_hat + epsilon

        # 5) Correction (the Woodbury/Cholesky/logdet work happens here)
        (log_correction, rho, logcorr_nonfinite_count,
         cond_M, jitter_count) = compute_correction_method_a(
            epsilon, xi_hat, deviations, self.eta,
            self.sigma_sys_sq, self.param_dim,
        )
        _t_corr = time.perf_counter()

        # 6) Log-likelihood
        ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()

        # 7) Accumulated weights
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        log_weights = log_prev + ll + log_correction
        self.weights = normalize_logweights(log_weights)

        # 8) Weighted estimate
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

        # 9) EMA update with the current batch gradient
        self.ema_m = self.beta * self.ema_m + (1.0 - self.beta) * g_hat

        # 10) Resample on low ESS, duplicating the EMA state as well
        if ess < self.ess_resample_ratio * self.N:
            idx = systematic_resample(self.weights, self.rng)
            n_unique = int(np.unique(idx).size)
            self.particles = self.particles[idx]
            self.ema_m = self.ema_m[idx]
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
        self.history["cond_M_mean"].append(float(np.mean(cond_M)))
        self.history["cond_M_max"].append(float(np.max(cond_M)))
        self.history["jitter_count"].append(jitter_count)
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
