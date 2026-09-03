#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities: activations, resampling and weight diagnostics."""

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numba extension module 'numba_dpex.*' failed to load",
)

import numpy as np
from numba import njit, prange


# ================================================================
# Activations
# ================================================================
def sigmoid(z):
    """Numerically stable logistic function."""
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def softplus(x):
    """Stable log(1 + exp(x))."""
    return np.logaddexp(0.0, x)


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float64)


def tanh_derivative(x):
    """1 - tanh(x)^2."""
    t = np.tanh(x)
    return 1.0 - t ** 2


# ================================================================
# Resampling
# ================================================================
@njit(cache=True)
def _systematic_resample_njit(weights, u):
    """Systematic resampling given one uniform draw u in [0, 1)."""
    N = weights.shape[0]
    indices = np.empty(N, dtype=np.int64)
    cumsum = np.cumsum(weights)
    j = 0
    for i in range(N):
        pos = (u + i) / N
        while j < N - 1 and cumsum[j] < pos:
            j += 1
        indices[i] = j
    return indices


def systematic_resample(weights, rng):
    """Systematic resampling.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Normalized weights.
    rng : numpy.random.Generator

    Returns
    -------
    indices : ndarray, shape (N,)
    """
    u = rng.random()
    return _systematic_resample_njit(weights, u)


@njit(cache=True)
def normalize_logweights(ll):
    """Normalize log-weights to weights summing to one.

    Falls back to uniform weights when the normalizer underflows.
    """
    N = ll.shape[0]
    m = ll[0]
    for i in range(1, N):
        if ll[i] > m:
            m = ll[i]

    w = np.empty(N)
    s = 0.0
    for i in range(N):
        w[i] = np.exp(ll[i] - m)
        s += w[i]

    if s == 0.0 or not np.isfinite(s):
        inv_n = 1.0 / N
        for i in range(N):
            w[i] = inv_n
    else:
        for i in range(N):
            w[i] /= s

    return w


@njit(cache=True)
def effective_sample_size(w):
    """Effective sample size 1 / sum_i w_i^2 for normalized weights w."""
    ss = 0.0
    for i in range(w.shape[0]):
        ss += w[i] * w[i]
    return 1.0 / ss


# ================================================================
# Degeneracy diagnostics
# ================================================================
def weight_diagnostics(w):
    """Return (ESS, Shannon entropy, max weight) for normalized weights.

    The entropy uses the convention 0 log 0 = 0.
    """
    ess = float(effective_sample_size(w))
    nz = w > 0.0
    entropy = float(-(w[nz] * np.log(w[nz])).sum())
    max_weight = float(w.max())
    return ess, entropy, max_weight


def ensemble_spread_trace(particles):
    """Unweighted ensemble spread: trace of the sample covariance,
    i.e. sum over dimensions of Var(particles[:, dim])."""
    return float(particles.var(axis=0).sum())
