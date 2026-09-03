#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Applicability diagnostic: when does the correction matter?

As a function of the realized drift noise eps ~ N(0, sigma_cd^2 I_d), the
correction of WSPF-B is

    log R = (d/2) log(1-rho) + rho ||eps||^2 / (2 sigma_cd^2) - d rho / 2,
    rho = eta^2 sbar / (eta^2 sbar + sigma_cd^2).

The first and third terms are constant across particles, so all of the
particle-to-particle variation comes from the second one. Since
||eps||^2 / sigma_cd^2 ~ chi^2_d (mean d, variance 2d),

    E[log R] = (d/2) log(1-rho),     sd(log R) = rho sqrt(d/2).      (A)

Weights update as log w <- log w + loglik + log R, so the correction can only
change their ordering when sd(log R) is non-negligible against the spread of
the log-likelihood across particles. That motivates the diagnostic

    R_diag = rho sqrt(d/2) / sd(loglik).                             (B)

With R_diag >> 1 the weights are dominated by the correction; with
R_diag << 1 the likelihood dominates and the correction is invisible.

Too much spread in the log-weights degenerates them instead. The variance of
log R is rho^2 d / 2, so keeping a usable effective sample size requires

    N >~ exp(rho^2 d / 4)   <=>   rho^2 d <~ 4 log N.                (C)

(B) and (C) are the two quantities that predict, before running anything,
whether the correction will pay off on a given benchmark. This module
computes them from a run's history, which must carry rho_mean (or rho),
ll_std, and optionally log_correction_std.
"""

from __future__ import annotations

import numpy as np


def _series(history, key):
    """Extract a 1-D float series from the history, or an empty array."""
    v = history.get(key)
    if v is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(v, dtype=np.float64).ravel()


def _rho_series(history):
    """One representative rho per step.

    Prefers the scalar series rho_mean; otherwise averages the (T, N) array
    rho over particles.
    """
    rm = _series(history, "rho_mean")
    if rm.size:
        return rm
    rho = history.get("rho")
    if rho is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(rho, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return np.nanmean(arr.reshape(arr.shape[0], -1), axis=1)


def correction_spread_theory(rho, param_dim):
    """sd(log R) = rho sqrt(d/2), equation (A).

    Returns a scalar for scalar input, otherwise an array of the same shape.
    """
    r = np.asarray(rho, dtype=np.float64)
    out = r * np.sqrt(param_dim / 2.0)
    return float(out) if out.ndim == 0 else out


def required_particles(rho, param_dim):
    """Required particle count N >~ exp(rho^2 d / 4), equation (C).

    Overflow is returned as inf, which reads correctly: no practical particle
    count suffices in that regime.
    """
    r = np.asarray(rho, dtype=np.float64)
    expo = r ** 2 * param_dim / 4.0
    with np.errstate(over="ignore"):
        out = np.exp(np.minimum(expo, 709.0))
    out = np.where(expo > 709.0, np.inf, out)
    return float(out) if out.ndim == 0 else out


def max_supported_rho(param_dim, n_particles):
    """Largest rho that N particles support, sqrt(4 log N / d) from (C).

    rho lies in [0, 1), so a bound above one is clipped to one, meaning N is
    large enough that this constraint does not bind.
    """
    if param_dim <= 0 or n_particles <= 1:
        return float("nan")
    val = np.sqrt(4.0 * np.log(float(n_particles)) / float(param_dim))
    return float(min(val, 1.0))


def diagnose(history, param_dim, n_particles, mask=None):
    """Compute the diagnostic quantities for one run.

    Parameters
    ----------
    history : dict
        Output of a filter's get_history(); uses rho_mean or rho, ll_std, and
        log_correction_std when present.
    param_dim : int
    n_particles : int
    mask : array-like[bool] | None
        Steps to aggregate over (e.g. the reporting window). Applied only
        when its length matches the series; None uses every step.

    Returns
    -------
    dict
        rho_mean            mean rho over the selected steps
        ll_std              mean sd(loglik) over the selected steps
        sd_logR_theory      rho_bar sqrt(d/2)                        (A)
        sd_logR_measured    mean measured sd(log R), or NaN
        R_diag              sd_logR_theory / ll_std                  (B)
        R_diag_measured     the same ratio from the measured spread
        required_N          exp(rho_bar^2 d / 4)                     (C)
        rho_max_supported   sqrt(4 log N / d)
        satisfies_condition whether rho_bar^2 d <= 4 log N
        param_dim, n_particles, n_steps
    """
    rho = _rho_series(history)
    ll_std = _series(history, "ll_std")
    lc_std = _series(history, "log_correction_std")

    def _apply(a):
        if mask is None:
            return a
        m = np.asarray(mask, dtype=bool)
        return a[m] if a.size == m.size else a

    rho, ll_std, lc_std = _apply(rho), _apply(ll_std), _apply(lc_std)

    def _nanmean(a):
        if a.size == 0 or not np.any(np.isfinite(a)):
            return float("nan")
        return float(np.nanmean(a))

    rho_bar = _nanmean(rho)
    ll_bar = _nanmean(ll_std)
    lc_bar = _nanmean(lc_std)

    sd_theory = (correction_spread_theory(rho_bar, param_dim)
                 if np.isfinite(rho_bar) else float("nan"))

    def _ratio(num):
        if not np.isfinite(num) or not np.isfinite(ll_bar) or ll_bar <= 0.0:
            return float("nan")
        return float(num / ll_bar)

    req_n = (required_particles(rho_bar, param_dim)
             if np.isfinite(rho_bar) else float("nan"))
    rho_cap = max_supported_rho(param_dim, n_particles)
    satisfies = (bool(rho_bar ** 2 * param_dim <= 4.0 * np.log(n_particles))
                 if np.isfinite(rho_bar) and n_particles > 1 else None)

    return {
        "param_dim": int(param_dim),
        "n_particles": int(n_particles),
        "n_steps": int(rho.size if rho.size else ll_std.size),
        "rho_mean": rho_bar,
        "ll_std": ll_bar,
        "sd_logR_theory": sd_theory,
        "sd_logR_measured": lc_bar,
        "R_diag": _ratio(sd_theory),
        "R_diag_measured": _ratio(lc_bar),
        "required_N": req_n,
        "rho_max_supported": rho_cap,
        "satisfies_condition": satisfies,
    }
