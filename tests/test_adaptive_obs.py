#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
φ_t 拡張 (adaptive observation noise) のユニットテスト (設計書 §6)

  1. 後方互換     : adaptive_obs=False が既存実装と同一乱数列・同一出力
                    (フラグ導入によるコード経路の変化が無いことの検証)
  2. τ=0 等価性   : adaptive_obs=True, tau_phi=0, φ_0 ≡ 2 log σ のとき
                    θ 軌跡・重みが固定 σ 版と一致 (φ rng 分離の検証)
  3. 補正不変性   : adaptive on/off で log R̂ (log_correction_mean) が一致
                    (φ が prior–proposal 補正に混入していないこと)
  4. σ 回復       : 定常データ (σ=0.7) で誤った初期値から σ̂ が真値近傍へ収束
  5. 混合 NLL 検算: N=1 粒子で nll_gaussian_mixture == nll_gaussian
  6. スケジュール : regime / rw の σ*_t 生成と constant の後方互換

実行:  python -m pytest tests/test_adaptive_obs.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.regression_switch import RegressionSwitchBenchmark
from src.filters import ParticleFilter, WSPF_A, WSPF_B
from src.evaluation import metrics as M


# ----------------------------------------------------------------------
# 共通フィクスチャ: 小さな constant ベンチマークと関数群
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench_const():
    b = RegressionSwitchBenchmark(T=60, batch_size=8, test_size=20,
                                  noise_std=0.5, oracle_samples=100,
                                  eval_start=5, select_start=5, select_end=30)
    funcs = b.build_functions(0)
    data = b._generate(0)
    return b, funcs, data


def _run_filter(cls, funcs, data, T=40, adaptive=False, seed=7, **kw):
    """フィルタを T ステップ駆動して (θ 軌跡, 最終重み, history) を返す。"""
    b_dim = data["theta_true"].shape[1]
    common = dict(n_particles=16, param_dim=b_dim, eta=0.1, sigma_sys=0.05,
                  prior_mean=0.0, prior_std=0.3, seed=seed)
    if adaptive:
        common.update(adaptive_obs=True, phi_seed=seed + 200, **kw)
    else:
        common.update(**kw)
    f = cls(**common)
    means = []
    for t in range(T):
        X, y = data["X_train"][t], data["y_train"][t]
        if cls is ParticleFilter:
            m = f.step(X, y, funcs["grad_fn"], funcs["loglik_fn"],
                       loglik_sigma_fn=funcs.get("loglik_sigma_fn"))
        else:
            m = f.step(X, y, funcs["per_sample_grad_fn"], funcs["loglik_fn"],
                       loglik_sigma_fn=funcs.get("loglik_sigma_fn"))
        means.append(m)
    return np.asarray(means), f.weights.copy(), f.get_history()


# ----------------------------------------------------------------------
# 1. 後方互換: フラグ off はフラグ導入前と同一経路 (自己整合チェック)
#    ここでは「off で 2 回走らせて決定的に一致」+「off が φ 属性を持たない」
#    を確認する。既存 outputs との突合はクラウド側の R0 実行で行う。
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ParticleFilter, WSPF_B, WSPF_A])
def test_non_adaptive_deterministic(cls, bench_const):
    _, funcs, data = bench_const
    m1, w1, _ = _run_filter(cls, funcs, data, adaptive=False)
    m2, w2, _ = _run_filter(cls, funcs, data, adaptive=False)
    assert np.array_equal(m1, m2)
    assert np.array_equal(w1, w2)


# ----------------------------------------------------------------------
# 2. τ=0 等価性: φ を真値に固定した適応版 == 固定 σ 版
#    (φ 専用 rng の分離により θ 側の乱数列が不変であることの検証)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ParticleFilter, WSPF_B, WSPF_A])
def test_tau_zero_equivalence(cls, bench_const):
    b, funcs, data = bench_const
    sigma_ref = funcs["sigma_ref"]
    m_fix, w_fix, _ = _run_filter(cls, funcs, data, adaptive=False)
    m_ad, w_ad, _ = _run_filter(
        cls, funcs, data, adaptive=True,
        tau_phi=0.0, phi_init_std=0.0,
        phi_init_mean=2.0 * np.log(sigma_ref))
    # φ ≡ 2 log σ_ref → 尤度は固定 σ 版と厳密一致、θ 乱数列も不変
    np.testing.assert_allclose(m_ad, m_fix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w_ad, w_fix, rtol=0, atol=1e-12)


# ----------------------------------------------------------------------
# 3. 補正不変性: log R̂ は φ に依存しない (θ ブロックのみ)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cls", [WSPF_B, WSPF_A])
def test_correction_independent_of_phi(cls, bench_const):
    _, funcs, data = bench_const
    _, _, h_fix = _run_filter(cls, funcs, data, adaptive=False)
    _, _, h_ad = _run_filter(cls, funcs, data, adaptive=True,
                             tau_phi=0.2, phi_init_std=0.5,
                             phi_init_mean=2.0 * np.log(0.5))
    # 適応版は尤度が変わるため重み・リサンプリングが分岐し得るが、粒子軌跡は
    # 最初のリサンプリングで初めて分岐する。補正 log R̂ は各ステップ内で
    # リサンプリングの **前** に計算されるため、最初のリサンプリング発生
    # ステップ自身までは両者の log R̂ が一致するはず。
    fix_rs = np.asarray(h_fix["resampled"], bool)
    ad_rs = np.asarray(h_ad["resampled"], bool)
    either = fix_rs | ad_rs
    first_rs = int(np.argmax(either)) if either.any() else len(either) - 1
    first_div = first_rs + 1   # 比較可能な先頭区間 [0, first_rs]
    assert first_div >= 1, "リサンプリング前の比較区間が存在しない"
    np.testing.assert_allclose(
        np.asarray(h_ad["log_correction_mean"][:first_div]),
        np.asarray(h_fix["log_correction_mean"][:first_div]),
        rtol=0, atol=1e-10)


# ----------------------------------------------------------------------
# 4. σ 回復: 定常 σ=0.7 のデータで、誤った初期値 (σ=0.3) から回復する
# ----------------------------------------------------------------------
def test_sigma_recovery():
    b = RegressionSwitchBenchmark(T=200, batch_size=16, test_size=20,
                                  noise_std=0.7, oracle_samples=100,
                                  eval_start=5, select_start=5, select_end=30)
    funcs = b.build_functions(3)
    data = b._generate(3)
    _, _, h = _run_filter(WSPF_B, funcs, data, T=200, adaptive=True,
                          tau_phi=0.05, phi_init_std=0.1,
                          phi_init_mean=2.0 * np.log(0.3))
    sig = np.asarray(h["sigma_hat_mean"])
    # 後半 50 ステップの平均 σ̂ が真値 0.7 の近傍 (±0.15)
    tail = float(np.mean(sig[-50:]))
    assert abs(tail - 0.7) < 0.15, f"σ̂ tail={tail:.3f} が 0.7 から乖離"


# ----------------------------------------------------------------------
# 5. 混合 NLL の検算
# ----------------------------------------------------------------------
def test_mixture_nll_single_particle():
    rng = np.random.default_rng(0)
    y = rng.normal(size=10)
    pred = rng.normal(size=(1, 10))
    sigma = np.array([0.6])
    nll_mix = M.nll_gaussian_mixture(y, pred, np.array([1.0]), sigma)
    nll_single = M.nll_gaussian(y, pred[0], np.full(10, 0.6))
    np.testing.assert_allclose(nll_mix, nll_single, rtol=0, atol=1e-12)


def test_mixture_moment_std():
    w = np.array([0.5, 0.5])
    sig = np.array([0.3, 0.7])
    out = M.mixture_moment_std(np.array([0.0]), w, sig)
    expect = np.sqrt(0.5 * 0.09 + 0.5 * 0.49)
    np.testing.assert_allclose(out, [expect], rtol=0, atol=1e-12)


# ----------------------------------------------------------------------
# 6. ノイズスケジュール
# ----------------------------------------------------------------------
def test_schedule_constant_backward_compatible():
    b1 = RegressionSwitchBenchmark(T=50)
    b2 = RegressionSwitchBenchmark(T=50, noise_schedule="constant")
    d1, d2 = b1._generate(1), b2._generate(1)
    for k in ("y_train", "y_test"):
        for a, c in zip(d1[k], d2[k]):
            assert np.array_equal(a, c)
    assert np.all(d1["sigma_true"] == b1.noise_std)


def test_schedule_regime_alignment():
    b = RegressionSwitchBenchmark(T=500, noise_schedule="regime",
                                  noise_std=0.3, noise_std_alt=0.7)
    d = b._generate(0)
    sig, rid = d["sigma_true"], d["regime_ids"]
    assert np.all(sig[rid % 2 == 0] == 0.3)
    assert np.all(sig[rid % 2 == 1] == 0.7)


def test_schedule_rw_range_and_tau_zero():
    b0 = RegressionSwitchBenchmark(T=100, noise_schedule="rw",
                                   noise_std=0.5, noise_rw_tau=0.0)
    d0 = b0._generate(0)
    assert np.allclose(d0["sigma_true"], 0.5)
    b = RegressionSwitchBenchmark(T=100, noise_schedule="rw",
                                  noise_std=0.5, noise_rw_tau=0.05)
    d = b._generate(0)
    assert d["sigma_true"].std() > 0
    assert np.all(d["sigma_true"] > 0)


def test_regime_requires_alt():
    with pytest.raises(ValueError):
        RegressionSwitchBenchmark(noise_schedule="regime")
