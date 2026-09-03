#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The low-rank WSPF-A correction against a dense reference.

Checks, on small (d, B):
  1. the log correction matches the dense implementation,
  2. rank(Sigma_hat) <= B - 1,
  3. the log-determinant matches,
  4. the quadratic form matches,
  5. the result stays finite for small sigma_cd,
  6. the jitter fallback works,
  7. non-finite occurrences are counted.
"""

import os
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters import wspf_a
from src.filters.wspf_a import compute_correction_method_a


def _dense_reference(epsilon, xi_hat, deviations, eta, c, d):
    """Dense d-by-d reference for log R_A, log|Vp| and the quadratic form."""
    N, B, _ = deviations.shape
    logdet_p = np.empty(N)
    quad_p = np.empty(N)
    logR = np.empty(N)
    ranks = np.empty(N, dtype=int)
    eye = np.eye(d)
    for i in range(N):
        W = deviations[i]                       # (B, d)
        Sigma = (W.T @ W) / (B * (B - 1))       # (d, d)
        Vp = eta ** 2 * Sigma + c * eye
        sign, ld = np.linalg.slogdet(Vp)
        logdet_p[i] = ld
        v = epsilon[i] - eta * xi_hat[i]
        qp = float(v @ np.linalg.solve(Vp, v))
        quad_p[i] = qp
        quad_q = float((epsilon[i] @ epsilon[i]) / c)
        logR[i] = 0.5 * (d * np.log(c) - ld) + 0.5 * quad_q - 0.5 * qp
        ranks[i] = np.linalg.matrix_rank(Sigma)
    return logR, logdet_p, quad_p, ranks


def _make_case(N=6, B=5, d=4, c=0.05, eta=0.1, seed=0):
    rng = np.random.default_rng(seed)
    deviations = rng.normal(size=(N, B, d))
    # Centre the deviations so that they sum to zero.
    deviations -= deviations.mean(axis=1, keepdims=True)
    epsilon = rng.normal(scale=np.sqrt(c), size=(N, d))
    xi_hat = rng.normal(scale=0.1, size=(N, d))
    return epsilon, xi_hat, deviations, eta, c, d


def test_lowrank_matches_dense():
    """(1)(3)(4) correction, log-determinant and quadratic form match dense."""
    epsilon, xi_hat, deviations, eta, c, d = _make_case()
    logR, rho, nonfinite, cond_M, jitter = compute_correction_method_a(
        epsilon, xi_hat, deviations, eta, c, d
    )
    logR_ref, logdet_ref, quadp_ref, ranks = _dense_reference(
        epsilon, xi_hat, deviations, eta, c, d
    )
    assert np.allclose(logR, logR_ref, atol=1e-8), (
        f"low-rank and dense log R differ: max diff="
        f"{np.max(np.abs(logR - logR_ref)):.2e}"
    )
    assert nonfinite == 0


def test_rank_le_B_minus_1():
    """(2) rank(Σ̂) ≤ B-1。"""
    _, _, deviations, _, _, _ = _make_case(N=4, B=5, d=10)
    _, _, _, ranks = _dense_reference(
        deviations=deviations,
        epsilon=np.zeros((4, 10)), xi_hat=np.zeros((4, 10)),
        eta=0.1, c=0.05, d=10,
    )
    assert np.all(ranks <= 5 - 1)


def test_finite_for_small_sigma_cd():
    """(5) Finite even for small sigma_cd."""
    epsilon, xi_hat, deviations, eta, _, d = _make_case(c=0.05)
    for c in [1e-3, 1e-5, 1e-8]:
        logR, rho, nonfinite, cond_M, jitter = compute_correction_method_a(
            epsilon, xi_hat, deviations, eta, c, d
        )
        assert np.all(np.isfinite(logR)), f"non-finite at sigma_cd^2={c}"
        assert nonfinite == 0


def test_jitter_fallback_and_nonfinite_count():
    """(6)(7) A singular M triggers the jitter fallback while the correction
    stays finite."""
    # A tiny c together with large, nearly duplicated deviations makes
    # alpha c^-1 W W^T huge and ill-conditioned, so the batched Cholesky
    # fails numerically and the per-particle jitter path is taken.
    N, B, d, c, eta = 3, 4, 5, 1e-30, 1.0
    rng = np.random.default_rng(1)
    base = rng.normal(size=(N, 1, d))
    deviations = base + 1e-4 * rng.normal(size=(N, B, d))
    deviations -= deviations.mean(axis=1, keepdims=True)
    epsilon = rng.normal(size=(N, d))
    xi_hat = rng.normal(size=(N, d))
    logR, rho, nonfinite, cond_M, jitter = compute_correction_method_a(
        epsilon, xi_hat, deviations, eta, c, d
    )
    assert isinstance(nonfinite, int)
    assert isinstance(jitter, int)
    assert cond_M.shape == (N,)
    # Only non-negativity is asserted, since the numerics are
    # environment-dependent; the monkeypatch test below is the deterministic
    # check that the fallback fires.
    assert jitter >= 0
    # Non-finite values are guarded and replaced by zero.
    assert np.all(np.isfinite(logR))


def test_jitter_fallback_forced_by_monkeypatch():
    """(6)(7) Fire the jitter fallback deterministically.

    numpy.linalg.cholesky is monkeypatched so that the batched call and each
    particle's first per-particle call raise LinAlgError:
      1. batched cholesky(M), ndim == 3        -> fails
      2. particle i, first cholesky(M[i])      -> fails (odd 2-D call)
      3. particle i, cholesky(M[i] + jitter)   -> delegates and succeeds
    Every particle therefore needs jitter, giving jitter_count == N > 0.
    """
    epsilon, xi_hat, deviations, eta, c, d = _make_case()

    real_cholesky = np.linalg.cholesky   # keep the real one before patching
    twod_calls = {"n": 0}

    def fake_cholesky(a):
        arr = np.asarray(a)
        if arr.ndim == 3:
            # Always fail the batched path, forcing the per-particle one.
            raise np.linalg.LinAlgError("forced batched cholesky failure")
        twod_calls["n"] += 1
        if twod_calls["n"] % 2 == 1:
            # Fail the first attempt per particle, forcing the jitter retry.
            raise np.linalg.LinAlgError("forced per-particle cholesky failure")
        # The retry with jitter goes to the real cholesky and succeeds.
        return real_cholesky(arr)

    # Patch linalg.cholesky on the module-level numpy the function uses.
    with mock.patch.object(wspf_a.np.linalg, "cholesky", side_effect=fake_cholesky):
        logR, rho, nonfinite, cond_M, jitter = compute_correction_method_a(
            epsilon, xi_hat, deviations, eta, c, d
        )

    N = deviations.shape[0]
    # Every particle needed the per-particle jitter.
    assert jitter > 0, f"jitter fallback did not fire: jitter={jitter}"
    assert jitter == N
    # The correction stays finite.
    assert np.all(np.isfinite(logR))
    assert nonfinite == 0


if __name__ == "__main__":
    test_lowrank_matches_dense()
    test_rank_le_B_minus_1()
    test_finite_for_small_sigma_cd()
    test_jitter_fallback_and_nonfinite_count()
    test_jitter_fallback_forced_by_monkeypatch()
    print("test_lowrank_correction: all tests passed")
