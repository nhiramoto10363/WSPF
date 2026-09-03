#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the paired statistics and the mixture NLL.

  - holm_adjust: monotonicity, the cap at one, being no more conservative
    than Bonferroni, and the handling of NaN;
  - all_pairs_compare: the number of pairs and their symmetry, the direction
    of "better", and the comparison family Holm is applied over;
  - nll_gaussian_mixture: coincides with the single-Gaussian NLL in the
    degenerate cases, and departs from it once the particles spread out.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import (  # noqa: E402
    nll_gaussian, nll_gaussian_mixture, prediction_std_with_noise,
    weighted_moments,
)
from src.evaluation.statistics import (  # noqa: E402
    all_pairs_compare, holm_adjust, paired_compare,
)


# ======================================================================
# holm_adjust
# ======================================================================
def test_holm_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(25)
    adj = holm_adjust(p)
    assert np.all(adj <= 1.0)
    assert np.all(adj >= p - 1e-12)          # adjustment never lowers p
    # The ranking of the original p-values is preserved.
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_holm_is_less_conservative_than_bonferroni():
    p = np.array([0.001, 0.02, 0.04])
    adj = holm_adjust(p)
    bonf = np.minimum(1.0, p * p.size)
    assert np.all(adj <= bonf + 1e-12)
    assert np.any(adj < bonf - 1e-12)        # at least one strictly smaller


def test_holm_smallest_p_matches_bonferroni():
    """The smallest p is multiplied by m, the first Holm step."""
    p = np.array([0.001, 0.02, 0.04])
    assert holm_adjust(p)[0] == pytest.approx(0.003)


def test_holm_excludes_nan_from_family():
    """NaN p-values leave the family and come back as NaN."""
    p = [0.01, float("nan"), 0.02]
    adj = holm_adjust(p)
    assert np.isnan(adj[1])
    # Two valid comparisons, so the smallest p doubles rather than triples.
    assert adj[0] == pytest.approx(0.02)


def test_holm_all_nan():
    assert np.all(np.isnan(holm_adjust([float("nan")] * 3)))


# ======================================================================
# paired_compare / all_pairs_compare
# ======================================================================
def test_paired_compare_returns_both_tests():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 10)
    b = a + rng.normal(0.5, 0.1, 10)
    c = paired_compare(list(a), list(b))
    for key in ("mean_diff", "std_diff", "t", "p", "wilcoxon_stat",
                "wilcoxon_p", "n"):
        assert key in c, f"{key} missing"
    assert c["n"] == 10
    assert 0.0 <= c["p"] <= 1.0
    assert 0.0 <= c["wilcoxon_p"] <= 1.0
    assert c["mean_diff"] < 0.0                # a is the smaller


def test_all_pairs_covers_every_pair_once():
    d = {"A": [1.0, 2.0, 3.0], "B": [2.0, 2.5, 4.0], "C": [0.5, 1.0, 2.0]}
    rows = all_pairs_compare(d)
    assert len(rows) == 3                      # 3C2
    seen = {(r["a"], r["b"]) for r in rows}
    assert seen == {("A", "B"), ("A", "C"), ("B", "C")}


def test_all_pairs_respects_method_order():
    d = {"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}
    rows = all_pairs_compare(d, methods=["C", "A"])
    assert len(rows) == 1
    assert (rows[0]["a"], rows[0]["b"]) == ("C", "A")


def test_all_pairs_better_direction():
    """lower_is_better flips the direction of "better"."""
    d = {"lo": [1.0, 1.0, 1.0, 1.5], "hi": [2.0, 2.0, 2.0, 2.5]}
    assert all_pairs_compare(d, lower_is_better=True)[0]["better"] == "lo"
    assert all_pairs_compare(d, lower_is_better=False)[0]["better"] == "hi"


def test_all_pairs_holm_family_is_the_pair_set():
    """Holm treats the set of pairs as one family."""
    rng = np.random.default_rng(2)
    d = {name: rng.normal(i, 1.0, 12) for i, name in enumerate("ABCD")}
    rows = all_pairs_compare(d)
    assert len(rows) == 6
    raw = np.array([r["paired_t_p"] for r in rows])
    adj = np.array([r["paired_t_p_holm"] for r in rows])
    assert np.all(adj >= raw - 1e-12)
    assert np.nanmin(adj) == pytest.approx(min(1.0, 6 * np.nanmin(raw)))


def test_all_pairs_skips_missing_methods():
    d = {"A": [1.0, 2.0], "B": [2.0, 3.0]}
    rows = all_pairs_compare(d, methods=["A", "B", "missing"])
    assert len(rows) == 1


# ======================================================================
# nll_gaussian_mixture
# ======================================================================
def test_mixture_equals_single_gaussian_for_point_estimator():
    """With one particle the mixture NLL equals the Gaussian NLL."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=32)
    preds = rng.normal(size=(1, 32))
    sigma = 0.5
    mix = nll_gaussian_mixture(y, preds, [1.0], sigma)
    mean, var = weighted_moments(preds, [1.0])
    single = nll_gaussian(y, mean, prediction_std_with_noise(var, sigma))
    assert mix == pytest.approx(single, rel=1e-12)


def test_mixture_equals_single_gaussian_when_particles_identical():
    """Identical predictions degenerate the mixture to a single Gaussian."""
    rng = np.random.default_rng(4)
    y = rng.normal(size=16)
    row = rng.normal(size=16)
    preds = np.tile(row, (25, 1))
    w = rng.random(25)
    sigma = 0.7
    mix = nll_gaussian_mixture(y, preds, w, sigma)
    mean, var = weighted_moments(preds, w)
    single = nll_gaussian(y, mean, prediction_std_with_noise(var, sigma))
    assert mix == pytest.approx(single, rel=1e-10)


def test_mixture_differs_when_particles_are_spread():
    """For multimodal particles the two NLLs disagree."""
    y = np.array([0.0, 0.0])
    preds = np.array([[-5.0, -5.0], [5.0, 5.0]])   # bimodal, with y in the valley
    w = np.array([0.5, 0.5])
    sigma = 0.5
    mix = nll_gaussian_mixture(y, preds, w, sigma)
    mean, var = weighted_moments(preds, w)
    single = nll_gaussian(y, mean, prediction_std_with_noise(var, sigma))
    # Under the mixture y is far from both modes and the NLL is large; the
    # single Gaussian, centred at zero and wide, explains y far too well.
    assert mix > single + 10.0


def test_mixture_ignores_zero_weight_particles():
    """Zero-weight particles contribute nothing to the mixture."""
    rng = np.random.default_rng(5)
    y = rng.normal(size=8)
    good = rng.normal(size=(3, 8))
    bad = np.full((2, 8), 1e3)                    # wildly off particles
    preds = np.vstack([good, bad])
    w = np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])
    mix_all = nll_gaussian_mixture(y, preds, w, 0.5)
    mix_good = nll_gaussian_mixture(y, good, np.full(3, 1 / 3), 0.5)
    assert mix_all == pytest.approx(mix_good, rel=1e-10)


def test_mixture_is_finite_under_weight_degeneracy():
    """Weights collapsed onto one particle still give a finite value."""
    rng = np.random.default_rng(6)
    y = rng.normal(size=8)
    preds = rng.normal(size=(50, 8))
    w = np.zeros(50)
    w[7] = 1.0
    assert np.isfinite(nll_gaussian_mixture(y, preds, w, 0.5))
