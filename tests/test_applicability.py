#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the applicability diagnostic.

The theory rests on the fact that the particle-to-particle spread of log R
comes from its second term, rho ||eps||^2 / (2 sigma^2), and that
||eps||^2 / sigma^2 ~ chi^2_d, giving sd(log R) = rho sqrt(d/2). These tests
check that

  1. the closed form is implemented correctly (correction_spread_theory),
  2. it agrees with the actual WSPF-B correction under Monte Carlo, and
  3. the particle budget N >~ exp(rho^2 d / 4) is consistent with the
     condition rho^2 d <~ 4 log N.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.applicability import (  # noqa: E402
    correction_spread_theory, diagnose, max_supported_rho, required_particles,
)
from src.filters.wspf_b import compute_correction_method_b  # noqa: E402


# ======================================================================
# (A) sd(log R) = ρ √(d/2)
# ======================================================================
def test_correction_spread_closed_form():
    assert correction_spread_theory(0.5, 8) == pytest.approx(0.5 * 2.0)
    assert correction_spread_theory(0.0, 100) == pytest.approx(0.0)


def test_correction_spread_scales_with_sqrt_d():
    """Quadrupling d doubles sd(log R)."""
    a = correction_spread_theory(0.3, 25)
    b = correction_spread_theory(0.3, 100)
    assert b == pytest.approx(2.0 * a)


def test_correction_spread_vectorized():
    out = correction_spread_theory([0.1, 0.2], 50)
    assert isinstance(out, np.ndarray) and out.shape == (2,)
    assert out[1] == pytest.approx(2.0 * out[0])


@pytest.mark.parametrize("d,rho_target", [(25, 0.3), (100, 0.5), (10, 0.8)])
def test_theory_matches_wspf_b_correction_by_monte_carlo(d, rho_target):
    """The measured spread of the WSPF-B correction equals rho sqrt(d/2).

    sbar is solved for so that rho = eta^2 sbar / (eta^2 sbar + sigma^2) hits
    rho_target, then many eps are drawn and the empirical sd is compared with
    the theoretical value.
    """
    sigma_sys = 0.1
    sigma_sys_sq = sigma_sys ** 2
    eta = 0.05
    # sbar that puts rho at rho_target
    s_bar_val = rho_target * sigma_sys_sq / ((1.0 - rho_target) * eta ** 2)

    n = 40000
    rng = np.random.default_rng(0)
    epsilon = rng.normal(0.0, sigma_sys, size=(n, d))
    s_bar = np.full(n, s_bar_val)

    log_corr, rho, nonfinite = compute_correction_method_b(
        epsilon, eta, s_bar, sigma_sys_sq, d)
    assert nonfinite == 0
    assert rho[0] == pytest.approx(rho_target, rel=1e-9)

    measured = float(np.std(log_corr))
    theory = correction_spread_theory(rho_target, d)
    # The Monte-Carlo error is about sd / sqrt(2n), so 3% is a tight bound.
    assert measured == pytest.approx(theory, rel=0.03)


def test_theory_matches_mean_of_log_correction():
    """E[log R] = (d/2) log(1 - rho), the first and third terms cancelling."""
    d, rho_target, sigma_sys, eta = 25, 0.4, 0.1, 0.05
    sigma_sys_sq = sigma_sys ** 2
    s_bar_val = rho_target * sigma_sys_sq / ((1.0 - rho_target) * eta ** 2)
    rng = np.random.default_rng(1)
    epsilon = rng.normal(0.0, sigma_sys, size=(40000, d))
    log_corr, _, _ = compute_correction_method_b(
        epsilon, eta, np.full(40000, s_bar_val), sigma_sys_sq, d)
    expected = 0.5 * d * np.log1p(-rho_target)
    assert float(np.mean(log_corr)) == pytest.approx(expected, abs=0.02)


# ======================================================================
# (C) N ≳ exp(ρ² d / 4)
# ======================================================================
def test_required_particles_matches_condition():
    """required_particles(rho, d) satisfies rho^2 d = 4 log N exactly."""
    rho, d = 0.4, 60
    n = required_particles(rho, d)
    assert rho ** 2 * d == pytest.approx(4.0 * np.log(n))


def test_required_particles_is_one_at_zero_rho():
    assert required_particles(0.0, 1000) == pytest.approx(1.0)


def test_required_particles_overflows_to_inf():
    """A very large rho^2 d returns inf: no practical N suffices."""
    assert np.isinf(required_particles(0.99, 100000))


def test_max_supported_rho_is_inverse_of_required_particles():
    d, n = 50, 500
    rho_cap = max_supported_rho(d, n)
    assert rho_cap < 1.0
    assert required_particles(rho_cap, d) == pytest.approx(n, rel=1e-9)


def test_max_supported_rho_clipped_at_one():
    """With N large enough the rho constraint does not bind, and is clipped to 1."""
    assert max_supported_rho(4, 10000) == pytest.approx(1.0)


def test_max_supported_rho_degenerate_inputs():
    assert np.isnan(max_supported_rho(0, 100))
    assert np.isnan(max_supported_rho(10, 1))


# ======================================================================
# diagnose
# ======================================================================
def _history(rho, ll_std, log_corr_std=None, t=10):
    h = {"rho_mean": np.full(t, rho), "ll_std": np.full(t, ll_std)}
    if log_corr_std is not None:
        h["log_correction_std"] = np.full(t, log_corr_std)
    return h


def test_diagnose_computes_ratio():
    d, n, rho, ll = 32, 100, 0.25, 0.5
    out = diagnose(_history(rho, ll), d, n)
    assert out["rho_mean"] == pytest.approx(rho)
    assert out["sd_logR_theory"] == pytest.approx(rho * np.sqrt(d / 2))
    assert out["R_diag"] == pytest.approx(rho * np.sqrt(d / 2) / ll)
    assert out["param_dim"] == d and out["n_particles"] == n


def test_diagnose_flags_condition_violation():
    """High dimension and high rho violate rho^2 d <~ 4 log N."""
    ok = diagnose(_history(0.1, 1.0), param_dim=25, n_particles=100)
    ng = diagnose(_history(0.9, 1.0), param_dim=833, n_particles=100)
    assert ok["satisfies_condition"] is True
    assert ng["satisfies_condition"] is False
    assert ng["required_N"] > ng["n_particles"]


def test_diagnose_uses_measured_spread_when_available():
    out = diagnose(_history(0.25, 0.5, log_corr_std=2.0), 32, 100)
    assert out["sd_logR_measured"] == pytest.approx(2.0)
    assert out["R_diag_measured"] == pytest.approx(2.0 / 0.5)


def test_diagnose_falls_back_to_rho_matrix():
    """Without rho_mean, the (T, N) rho array is averaged per step."""
    rho = np.tile(np.array([0.1, 0.3]), (6, 1))       # (T=6, N=2)
    out = diagnose({"rho": rho, "ll_std": np.full(6, 1.0)}, 18, 50)
    assert out["rho_mean"] == pytest.approx(0.2)


def test_diagnose_applies_mask():
    h = {"rho_mean": np.array([0.1, 0.1, 0.9, 0.9]),
         "ll_std": np.ones(4)}
    mask = np.array([True, True, False, False])
    assert diagnose(h, 25, 100, mask=mask)["rho_mean"] == pytest.approx(0.1)


def test_diagnose_handles_missing_rho():
    """A history without rho, as for PF, yields NaN rather than an error."""
    out = diagnose({"ll_std": np.ones(5)}, 25, 100)
    assert np.isnan(out["rho_mean"])
    assert np.isnan(out["R_diag"])
    assert out["satisfies_condition"] is None


def test_diagnose_handles_zero_loglik_spread():
    """With sd(loglik) = 0 the ratio is undefined and comes back NaN."""
    out = diagnose(_history(0.3, 0.0), 25, 100)
    assert np.isnan(out["R_diag"])
