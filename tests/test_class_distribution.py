# -*- coding: utf-8 -*-
"""Tests for the per-regime class distribution.

Pins down the interval split, the proportions and the region assignment in
src/evaluation/class_distribution.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.class_distribution import (  # noqa: E402
    regime_bounds, class_distribution_rows, _region_of,
)


# ======================================================================
# Regime bounds
# ======================================================================
def test_regime_bounds_basic():
    assert regime_bounds(1500, [300, 600, 900, 1200]) == [
        (0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]


def test_regime_bounds_covers_stream_without_gaps():
    """The intervals are contiguous and cover the stream exactly."""
    n = 52848
    cps = [14352, 19500, 33240, 38682, 39510]
    bounds = regime_bounds(n, cps)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (s0, e0), (s1, _) in zip(bounds, bounds[1:]):
        assert e0 == s1
    assert sum(e - s for s, e in bounds) == n


def test_regime_bounds_ignores_out_of_range_and_duplicates():
    assert regime_bounds(100, [0, 50, 50, 100, 200, -5]) == [(0, 50), (50, 100)]


def test_regime_bounds_no_change_points():
    assert regime_bounds(10, []) == [(0, 10)]


# ======================================================================
# Region assignment
# ======================================================================
def test_region_of():
    assert _region_of(0, 300, 600) == "selection"
    assert _region_of(600, 900, 600) == "report"
    assert _region_of(300, 900, 600) == "mixed"
    assert _region_of(0, 300, None) == ""


# ======================================================================
# Class distribution
# ======================================================================
def test_fractions_sum_to_one_per_regime():
    """The class proportions sum to one within every regime."""
    rng = np.random.default_rng(0)
    C, n = 6, 1200
    y = rng.integers(0, C, size=n)
    rows = class_distribution_rows(y, [300, 600, 900], C)
    for r in rows:
        total = sum(r[f"class_{c}_frac"] for c in range(C))
        assert total == pytest.approx(1.0, abs=1e-12)


def test_counts_match_and_sum_to_n():
    rng = np.random.default_rng(1)
    C, n = 4, 500
    y = rng.integers(0, C, size=n)
    rows = class_distribution_rows(y, [123, 400], C)
    regimes = [r for r in rows if r["regime"] != "all"]
    for r in regimes:
        assert sum(r[f"class_{c}_count"] for c in range(C)) == r["n_samples"]
    # The "all" row equals the sum over regimes.
    allrow = [r for r in rows if r["regime"] == "all"][0]
    for c in range(C):
        assert allrow[f"class_{c}_count"] == sum(
            r[f"class_{c}_count"] for r in regimes)
    assert allrow["n_samples"] == n


def test_label_flip_is_visible():
    """The label reversal that defines the email stream is visible."""
    # regime 1 is 2/3 class 0, regime 2 is 2/3 class 1
    y = np.array([0] * 200 + [1] * 100 + [1] * 200 + [0] * 100)
    rows = class_distribution_rows(y, [300], 2, report_start=300)
    r1, r2 = rows[0], rows[1]
    assert r1["class_0_frac"] == pytest.approx(2 / 3)
    assert r1["class_1_frac"] == pytest.approx(1 / 3)
    assert r2["class_0_frac"] == pytest.approx(1 / 3)
    assert r2["class_1_frac"] == pytest.approx(2 / 3)
    assert r1["region"] == "selection"
    assert r2["region"] == "report"


def test_balanced_stream_is_uniform():
    """A perfectly balanced stream gives 1/C in every regime."""
    C = 6
    y = np.tile(np.arange(C), 1000)
    rows = class_distribution_rows(y, [1200, 3000], C)
    for r in rows:
        for c in range(C):
            assert r[f"class_{c}_frac"] == pytest.approx(1.0 / C, abs=1e-12)


def test_binary_float_labels_accepted():
    """The email labels are floats (0.0 / 1.0) and must still work."""
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    rows = class_distribution_rows(y, [3], 2)
    assert rows[0]["class_0_count"] == 1
    assert rows[0]["class_1_count"] == 2
    assert rows[1]["class_0_count"] == 1
    assert rows[1]["class_1_count"] == 2


def test_class_names_attached_once():
    y = np.array([0, 1, 0, 1])
    rows = class_distribution_rows(y, [2], 2, class_names=["ham", "spam"])
    assert rows[0]["class_names"] == "ham|spam"
    assert "class_names" not in rows[1]
