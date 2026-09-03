#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequential weight accumulation, log w_t = log w_{t-1} + increments.

Confirms that PF and WSPF accumulate weights as SIS prescribes, discarding
nothing on steps without resampling.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters.base import normalize_logweights
from src.filters.pf import ParticleFilter


def test_normalize_logweights_basic():
    """normalize_logweights returns a normalized probability vector."""
    ll = np.array([-1.0, -2.0, -3.0, -100.0])
    w = normalize_logweights(ll)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0.0)
    # The largest log-weight carries the largest weight.
    assert np.argmax(w) == 0


def test_normalize_logweights_degenerate():
    """All -inf falls back to uniform weights."""
    ll = np.full(5, -np.inf)
    w = normalize_logweights(ll)
    assert np.allclose(w, 1.0 / 5)


def test_pf_sis_accumulation_without_resample():
    """With resampling disabled, weights accumulate over both steps."""
    N, d = 32, 3
    pf = ParticleFilter(
        n_particles=N, param_dim=d, eta=0.0, sigma_sys=0.0,
        prior_std=0.5, ess_resample_ratio=0.0, seed=0,
    )
    # eta=0 and sigma_sys=0 freeze the particles, so both steps evaluate the
    # log-likelihood at the same points.
    particles0 = pf.particles.copy()

    # Deterministic log-likelihood, one fixed value per particle
    fixed_ll = np.linspace(-1.0, 1.0, N)

    def grad_fn(p, X, y):
        return np.zeros_like(p)

    def loglik_fn(p, X, y):
        return fixed_ll.copy()

    pf.step(None, None, grad_fn, loglik_fn)
    pf.step(None, None, grad_fn, loglik_fn)

    # Expected: log w_2 proportional to 2 * fixed_ll, since the constant
    # log w_0 = -log N cancels in the normalization.
    expected = normalize_logweights(2.0 * fixed_ll)
    assert np.allclose(pf.weights, expected, atol=1e-10)
    # The particles have not moved.
    assert np.allclose(pf.particles, particles0)


def test_pf_resample_resets_weights():
    """With ess_resample_ratio=1.0 every step resamples and resets weights."""
    N, d = 16, 2
    pf = ParticleFilter(
        n_particles=N, param_dim=d, eta=0.0, sigma_sys=0.0,
        prior_std=0.5, ess_resample_ratio=1.0, seed=1,
    )

    def grad_fn(p, X, y):
        return np.zeros_like(p)

    def loglik_fn(p, X, y):
        return np.linspace(-2.0, 2.0, N)

    pf.step(None, None, grad_fn, loglik_fn)
    # Weights are reset to uniform after resampling.
    assert np.allclose(pf.weights, 1.0 / N)
    assert pf.get_history()["resampled"][-1]


if __name__ == "__main__":
    test_normalize_logweights_basic()
    test_normalize_logweights_degenerate()
    test_pf_sis_accumulation_without_resample()
    test_pf_resample_resets_weights()
    print("test_weight_accumulation: all tests passed")
