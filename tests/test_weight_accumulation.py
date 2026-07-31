#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIS の逐次重み累積 (log w_t = log w_{t-1} + increments) を検証する単体テスト。

リサンプリングしないステップの重み情報を捨てず、
PF/WSPF が SIS に忠実に重みを累積することを確認する
(pf.py / wspf_b.py の重み更新の中心的性質)。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.filters.base import normalize_logweights
from src.filters.pf import ParticleFilter


def test_normalize_logweights_basic():
    """normalize_logweights が正規化された確率ベクトルを返す。"""
    ll = np.array([-1.0, -2.0, -3.0, -100.0])
    w = normalize_logweights(ll)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0.0)
    # 最尤の要素が最大重み
    assert np.argmax(w) == 0


def test_normalize_logweights_degenerate():
    """全て -inf でも一様重みにフォールバックする(縮退ガード)。"""
    ll = np.full(5, -np.inf)
    w = normalize_logweights(ll)
    assert np.allclose(w, 1.0 / 5)


def test_pf_sis_accumulation_without_resample():
    """リサンプリングを無効化(ess_resample_ratio=0)したとき、
    2ステップ後の重みが 2 ステップ分の対数尤度の累積正規化に一致する。"""
    N, d = 32, 3
    pf = ParticleFilter(
        n_particles=N, param_dim=d, eta=0.0, sigma_sys=0.0,
        prior_std=0.5, ess_resample_ratio=0.0, seed=0,
    )
    # eta=0, sigma_sys=0 なので粒子は動かない → ll は各ステップ同じ粒子で評価される
    particles0 = pf.particles.copy()

    # 決定論的な対数尤度関数(粒子ごとに固定値)
    fixed_ll = np.linspace(-1.0, 1.0, N)

    def grad_fn(p, X, y):
        return np.zeros_like(p)

    def loglik_fn(p, X, y):
        return fixed_ll.copy()

    pf.step(None, None, grad_fn, loglik_fn)
    pf.step(None, None, grad_fn, loglik_fn)

    # 期待値: log w_2 ∝ 2 * fixed_ll (初期 log w_0 = -log N は定数なので相殺)
    expected = normalize_logweights(2.0 * fixed_ll)
    assert np.allclose(pf.weights, expected, atol=1e-10)
    # 粒子は動いていない
    assert np.allclose(pf.particles, particles0)


def test_pf_resample_resets_weights():
    """ess_resample_ratio=1.0 で毎ステップ リサンプリングし、重みが一様化される。"""
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
    # リサンプリング後は重みリセット(一様)
    assert np.allclose(pf.weights, 1.0 / N)
    assert pf.get_history()["resampled"][-1]


if __name__ == "__main__":
    test_normalize_logweights_basic()
    test_normalize_logweights_degenerate()
    test_pf_sis_accumulation_without_resample()
    test_pf_resample_resets_weights()
    print("test_weight_accumulation: 全テスト通過")
