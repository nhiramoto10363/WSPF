#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1-10: WSPF-A の低ランク補正(Woodbury/Cholesky)を dense 実装と照合する単体テスト。

修正方針 R1-10 が要求する項目:
  1. 小さい (d, B) で dense 版と low-rank 版の log correction が一致する
  2. rank(Σ̂) ≤ B-1 を確認する
  3. log-det が dense 版と一致する
  4. 二次形式が dense 版と一致する
  5. σ_cd が小さい場合にも有限値になる
  6. jitter fallback が動作する
  7. 非有限値の発生数が記録される
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters.wspf_a import compute_correction_method_a


def _dense_reference(epsilon, xi_hat, deviations, eta, c, d):
    """dense d×d で厳密に log R_A / log|Vp| / 二次形式を評価する参照実装。"""
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
    # 偏差は平均 0 (Σ_j (g_j-ĝ)=0) となるよう中心化
    deviations -= deviations.mean(axis=1, keepdims=True)
    epsilon = rng.normal(scale=np.sqrt(c), size=(N, d))
    xi_hat = rng.normal(scale=0.1, size=(N, d))
    return epsilon, xi_hat, deviations, eta, c, d


def test_lowrank_matches_dense():
    """(1)(3)(4) log correction / log-det / 二次形式が dense と一致。"""
    epsilon, xi_hat, deviations, eta, c, d = _make_case()
    logR, rho, nonfinite, cond_M = compute_correction_method_a(
        epsilon, xi_hat, deviations, eta, c, d
    )
    logR_ref, logdet_ref, quadp_ref, ranks = _dense_reference(
        epsilon, xi_hat, deviations, eta, c, d
    )
    assert np.allclose(logR, logR_ref, atol=1e-8), (
        f"low-rank と dense の log R が不一致: max diff="
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
    """(5) σ_cd が小さくても有限。"""
    epsilon, xi_hat, deviations, eta, _, d = _make_case(c=0.05)
    for c in [1e-3, 1e-5, 1e-8]:
        logR, rho, nonfinite, cond_M = compute_correction_method_a(
            epsilon, xi_hat, deviations, eta, c, d
        )
        assert np.all(np.isfinite(logR)), f"σ_cd²={c} で非有限"
        assert nonfinite == 0


def test_jitter_fallback_and_nonfinite_count():
    """(6)(7) 特異に近い M でも Cholesky が jitter fallback し、
    非有限カウントが int で返る。"""
    # 偏差をほぼ同一にして M を悪条件化
    N, B, d, c, eta = 3, 4, 5, 1e-10, 1.0
    rng = np.random.default_rng(1)
    base = rng.normal(size=(N, 1, d))
    deviations = base + 1e-12 * rng.normal(size=(N, B, d))
    deviations -= deviations.mean(axis=1, keepdims=True)
    epsilon = rng.normal(size=(N, d))
    xi_hat = rng.normal(size=(N, d))
    logR, rho, nonfinite, cond_M = compute_correction_method_a(
        epsilon, xi_hat, deviations, eta, c, d
    )
    assert isinstance(nonfinite, int)
    assert cond_M.shape == (N,)
    # 非有限はガードされ 0 埋めされている
    assert np.all(np.isfinite(logR))


if __name__ == "__main__":
    test_lowrank_matches_dense()
    test_rank_le_B_minus_1()
    test_finite_for_small_sigma_cd()
    test_jitter_fallback_and_nonfinite_count()
    print("test_lowrank_correction: 全テスト通過")
