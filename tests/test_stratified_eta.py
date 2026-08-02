#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
層化学習率 WSPF (Stratified Learning-Rate WSPF) のテスト (設計書 §13)。

fixed スキームが従来実装と数値一致すること、層化配置の性質、補正の broadcast
一致、リサンプリング時のスロット固定などを検証する。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters.base import (          # noqa: E402
    make_stratified_learning_rates,
    stratified_eta_diagnostics,
    fast_to_slow_rate,
)
from src.filters.wspf_a import WSPF_A, compute_correction_method_a  # noqa: E402
from src.filters.wspf_b import WSPF_B, compute_correction_method_b  # noqa: E402
from src.models import (                # noqa: E402
    NeuralNetRegression,
    create_regression_per_sample_grad_fn,
    create_regression_loglik_fn,
)


def _funcs():
    m = NeuralNetRegression(1, 8, 1)
    return (m.param_dim,
            create_regression_per_sample_grad_fn(m, 0.5),
            create_regression_loglik_fn(m, 0.5))


def _run(cls, d, psg, ll, scheme, n=60, steps=40, eta_seed=7, **extra):
    kw = dict(eta=0.05, sigma_sys=0.05, prior_std=0.3, seed=1,
              grad_clip_norm=5.0, eta_scheme=scheme, eta_seed=eta_seed)
    if cls is WSPF_A:
        kw["beta"] = 0.9
    kw.update(extra)
    f = cls(n, d, **kw)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        X = rng.normal(0, 1, (16, 1))
        y = rng.normal(0, 1, 16)
        f.step(X, y, psg, ll)
    return f


# --- §13.2/3/4: 分布の性質 ---
@pytest.mark.parametrize("N", [25, 100, 400])
def test_mean_exact_and_positive(N):
    e = make_stratified_learning_rates(N, 0.05, seed=42)
    assert e.shape == (N,)
    assert np.all(e > 0)
    assert abs(e.mean() - 0.05) < 1e-14        # 平均は厳密に η̄


def test_reproducible_and_seed_dependent():
    a = make_stratified_learning_rates(100, 0.05, seed=3)
    b = make_stratified_learning_rates(100, 0.05, seed=3)
    c = make_stratified_learning_rates(100, 0.05, seed=4)
    assert np.array_equal(a, b)                 # 同一 seed → 同一配置
    assert not np.array_equal(a, c)             # 置換が効いている


def test_mean_invariant_under_N():
    for N in [10, 50, 200, 500]:
        e = make_stratified_learning_rates(N, 0.123, seed=1)
        assert abs(e.mean() - 0.123) < 1e-13    # §13.11: N に依らず平均一定


# --- §13.1/9: fixed は従来実装と一致 (自己再現で担保) ---
@pytest.mark.parametrize("cls", [WSPF_B, WSPF_A])
def test_fixed_reproducible(cls):
    d, psg, ll = _funcs()
    f1 = _run(cls, d, psg, ll, "fixed")
    f2 = _run(cls, d, psg, ll, "fixed")
    assert np.array_equal(f1.particles, f2.particles)
    assert np.array_equal(f1.weights, f2.weights)


@pytest.mark.parametrize("cls", [WSPF_B, WSPF_A])
def test_fixed_all_equal_slots(cls):
    d, psg, ll = _funcs()
    f = _run(cls, d, psg, ll, "fixed")
    assert np.allclose(f.eta_slots, 0.05)       # fixed は全スロット η̄


# --- §13.7/8: broadcast 補正が粒子別スカラー計算と一致 ---
def test_correction_b_broadcast_matches_scalar():
    rng = np.random.default_rng(0)
    N, B, d = 30, 16, 25
    eps = rng.normal(0, 0.05, (N, d))
    s_bar = rng.uniform(0.1, 1.0, N)
    etas = rng.uniform(0.01, 0.2, N)
    lc_vec, rho_vec, _ = compute_correction_method_b(eps, etas, s_bar, 0.05 ** 2, d)
    for i in range(N):
        lc_i, rho_i, _ = compute_correction_method_b(
            eps[i:i + 1], float(etas[i]), s_bar[i:i + 1], 0.05 ** 2, d)
        assert np.isclose(lc_vec[i], lc_i[0], rtol=1e-12)
        assert np.isclose(rho_vec[i], rho_i[0], rtol=1e-12)


def test_correction_a_broadcast_matches_scalar():
    rng = np.random.default_rng(1)
    N, B, d = 20, 16, 25
    eps = rng.normal(0, 0.05, (N, d))
    xi = rng.normal(0, 0.1, (N, d))
    dev = rng.normal(0, 0.1, (N, B, d))
    etas = rng.uniform(0.01, 0.2, N)
    lc_vec, rho_vec, _, _, _ = compute_correction_method_a(
        eps, xi, dev, etas, 0.05 ** 2, d)
    for i in range(N):
        lc_i, rho_i, _, _, _ = compute_correction_method_a(
            eps[i:i + 1], xi[i:i + 1], dev[i:i + 1], float(etas[i]),
            0.05 ** 2, d)
        assert np.isclose(lc_vec[i], lc_i[0], rtol=1e-10, atol=1e-12)
        assert np.isclose(rho_vec[i], rho_i[0], rtol=1e-10, atol=1e-12)


def test_correction_scalar_equals_uniform_vector():
    """全 η_i=η̄ のベクトル入力が、スカラー η̄ 入力と一致 (§13.9)。"""
    rng = np.random.default_rng(2)
    N, B, d = 15, 16, 25
    eps = rng.normal(0, 0.05, (N, d))
    xi = rng.normal(0, 0.1, (N, d))
    dev = rng.normal(0, 0.1, (N, B, d))
    s_bar = rng.uniform(0.1, 1.0, N)
    lc_s, _, _ = compute_correction_method_b(eps, 0.07, s_bar, 0.05 ** 2, d)
    lc_v, _, _ = compute_correction_method_b(eps, np.full(N, 0.07), s_bar,
                                             0.05 ** 2, d)
    assert np.allclose(lc_s, lc_v, rtol=1e-14)
    la_s, _, _, _, _ = compute_correction_method_a(eps, xi, dev, 0.07,
                                                   0.05 ** 2, d)
    la_v, _, _, _, _ = compute_correction_method_a(eps, xi, dev,
                                                   np.full(N, 0.07),
                                                   0.05 ** 2, d)
    assert np.allclose(la_s, la_v, rtol=1e-12, atol=1e-13)


# --- §13.5: リサンプリング後も eta_slots 自体は不変 ---
@pytest.mark.parametrize("cls", [WSPF_B, WSPF_A])
def test_eta_slots_invariant_after_resample(cls):
    d, psg, ll = _funcs()
    f = cls(60, d, eta=0.05, sigma_sys=0.2, prior_std=0.3, seed=1,
            grad_clip_norm=5.0, eta_scheme="stratified_exp", eta_seed=7,
            **({"beta": 0.9} if cls is WSPF_A else {}),
            ess_resample_ratio=0.9)   # 高閾値でリサンプリング頻発
    slots0 = f.eta_slots.copy()
    rng = np.random.default_rng(0)
    resampled_any = False
    for _ in range(40):
        X = rng.normal(0, 1, (16, 1))
        y = rng.normal(0, 1, 16)
        f.step(X, y, psg, ll)
        resampled_any = resampled_any or f.history["resampled"][-1]
    assert resampled_any, "テスト前提: リサンプリングが起きること"
    assert np.array_equal(f.eta_slots, slots0)   # スロット固定 (§5)


# --- §13.10/12: 走行の有限性・診断 ---
@pytest.mark.parametrize("cls", [WSPF_B, WSPF_A])
def test_stratified_runs_finite_and_diagnostics(cls):
    d, psg, ll = _funcs()
    f = _run(cls, d, psg, ll, "stratified_exp")
    assert np.isfinite(f.particles).all()
    assert np.isfinite(f.weights).all()
    assert np.isclose(f.weights.sum(), 1.0)
    h = f.history
    # 診断量が記録され有限
    for k in ("eta_weighted_mean", "eta_weighted_std", "eta_slow_mass",
              "eta_fast_mass", "eta_map", "eta_fast_to_slow_rate"):
        assert len(h[k]) == 40
        assert np.isfinite(h[k]).all()
    # 重み付き平均 η は分布の範囲内
    assert f.eta_slots.min() - 1e-9 <= min(h["eta_weighted_mean"])
    assert max(h["eta_weighted_mean"]) <= f.eta_slots.max() + 1e-9


def test_pf_s_fixed_equals_pf():
    """PF-S (層化 PF) の fixed が通常 PF とビット一致 (補正なし版の後方互換)。"""
    from src.filters.pf import ParticleFilter
    from src.models import create_regression_grad_fn
    m = NeuralNetRegression(1, 8, 1)
    gf = create_regression_grad_fn(m, 0.5)
    ll = create_regression_loglik_fn(m, 0.5)
    d = m.param_dim

    def run(**kw):
        f = ParticleFilter(60, d, eta=0.05, sigma_sys=0.05, prior_std=0.3,
                           seed=1, **kw)
        rng = np.random.default_rng(0)
        for _ in range(40):
            X = rng.normal(0, 1, (16, 1))
            y = rng.normal(0, 1, 16)
            f.step(X, y, gf, ll)
        return f
    pf = run()                                   # 既定 (= fixed)
    pf_fix = run(eta_scheme="fixed", eta_seed=7)
    pf_s = run(eta_scheme="stratified_exp", eta_seed=7)
    assert np.array_equal(pf.particles, pf_fix.particles)   # 後方互換
    assert np.isfinite(pf_s.particles).all()
    assert not np.allclose(pf_fix.particles, pf_s.particles)
    assert len(pf_s.history["eta_weighted_mean"]) == 40


def test_diagnostics_helpers():
    eta = make_stratified_learning_rates(100, 0.05, seed=1)
    w = np.ones(100) / 100
    wm, ws, sm, fm, mp = stratified_eta_diagnostics(w, eta, 0.05)
    assert abs(wm - 0.05) < 1e-12               # 等重み → 平均 η̄
    assert 0.0 <= sm <= 1.0 and 0.0 <= fm <= 1.0
    assert mp in eta
    # a_i = i (リサンプリング無し) なら fast→slow = 0
    assert fast_to_slow_rate(np.arange(100), eta, 0.05) == 0.0
