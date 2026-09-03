#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the grid-edge detection in _is_boundary.

The baselines' own hyper-parameters (window, n_passes, ph_*) must be checked
for grid edges just like the filters'. Leaving them out would let Window-SGD
and PH-SGD be selected from effectively narrower grids without a warning,
biasing the comparison against them; these tests guard against that
regression.
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "scripts")))

from _common import _is_boundary  # noqa: E402

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs"))


def _grid(name):
    with open(os.path.join(_CONFIG_DIR, f"{name}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["grid"]


# ======================================================================
# Filter hyper-parameters
# ======================================================================
def test_detects_filter_hp_boundaries():
    grid = {"eta": [0.1, 0.2, 0.3], "sigma_sys": [0.01, 0.05],
            "prior_std": [0.1, 1.0], "beta": [0.8, 0.9]}
    hits = _is_boundary({"eta": 0.1, "sigma_sys": 0.05,
                         "prior_std": 0.1, "beta": 0.9}, grid)
    assert "eta(lower)" in hits
    assert "sigma_sys(upper)" in hits
    assert "prior_std(lower)" in hits
    assert "beta(upper)" in hits


def test_interior_point_reports_nothing():
    grid = {"eta": [0.1, 0.2, 0.3]}
    assert _is_boundary({"eta": 0.2}, grid) == []


# ======================================================================
# Baseline-specific hyper-parameters
# ======================================================================
def test_detects_baseline_hp_boundaries():
    """window, n_passes and ph_* are checked for edges as well."""
    grid = {"window": [3, 5, 10], "n_passes": [1, 2],
            "ph_delta": [0.005, 0.01], "ph_lambda": [2.0, 5.0]}
    hits = _is_boundary({"window": 3, "n_passes": 2,
                         "ph_delta": 0.005, "ph_lambda": 2.0}, grid)
    assert "window(lower)" in hits
    assert "n_passes(upper)" in hits
    assert "ph_delta(lower)" in hits
    assert "ph_lambda(lower)" in hits


def test_regression_of_job4141_selection():
    """The earlier Window-SGD and PH-SGD choices sat on the edges of the old grid."""
    old_grid = {"eta": [0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                "prior_std": [0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
                "window": [3, 5, 10], "n_passes": [1, 2],
                "ph_delta": [0.005, 0.01], "ph_lambda": [2.0, 5.0]}
    win = {"eta": 0.1, "prior_std": 0.2, "window": 3, "n_passes": 2}
    ph = {"eta": 0.05, "prior_std": 0.5, "ph_delta": 0.005, "ph_lambda": 2.0}
    assert _is_boundary(win, old_grid), "should be detected as an edge on the old grid"
    assert _is_boundary(ph, old_grid), "should be detected as an edge on the old grid"


# ======================================================================
# A single-point axis is not an edge
# ======================================================================
def test_single_valued_axis_is_not_a_boundary():
    """An axis with one candidate cannot be extended, so it raises no warning."""
    assert _is_boundary({"ph_alpha": 0.9999}, {"ph_alpha": [0.9999]}) == []


def test_axis_absent_from_grid_is_skipped():
    assert _is_boundary({"window": 3}, {"eta": [0.1, 0.2]}) == []


# ======================================================================
# With the current configs, every selected value is interior
# ======================================================================
_JOB4141_SELECTION = {
    "PF": {"eta": 0.05, "sigma_sys": 0.05, "prior_std": 0.1},
    "WSPF-A": {"eta": 0.05, "sigma_sys": 0.05, "prior_std": 0.75,
               "beta": 0.95},
    "WSPF-B": {"eta": 0.01, "sigma_sys": 0.1, "prior_std": 1.0},
    "SGD": {"eta": 0.05, "prior_std": 0.5},
    "PH-SGD": {"eta": 0.05, "prior_std": 0.5, "ph_delta": 0.005,
               "ph_lambda": 2.0, "ph_alpha": 0.9999},
    "Window-SGD": {"eta": 0.1, "prior_std": 0.2, "window": 3, "n_passes": 2},
}


def test_extended_regression_grid_makes_job4141_selection_interior():
    """After the extensions, none of the selected values sits on an edge."""
    grid = _grid("regression")
    for method, params in _JOB4141_SELECTION.items():
        assert _is_boundary(params, grid) == [], f"{method} still on an edge"


_BENCHMARKS = ("regression", "email", "insects")


def test_baseline_grids_are_shared_across_benchmarks():
    """The window and PH grids are identical across benchmarks.

    A narrower grid for the baselines alone would compare Window-SGD and
    PH-SGD under a handicap.
    """
    ref = _grid("regression")
    for name in _BENCHMARKS[1:]:
        grid = _grid(name)
        for key in ("window", "n_passes", "ph_delta", "ph_lambda", "ph_alpha"):
            assert ref[key] == grid[key], f"{key} differs between regression and {name}"


def test_prior_std_extends_above_one_everywhere():
    """Every benchmark can search prior_std above 1.0."""
    for name in _BENCHMARKS:
        assert max(_grid(name)["prior_std"]) > 1.0, f"{name}: prior_std capped at 1.0"


def test_every_tunable_axis_has_room_to_expand():
    """Apart from single-point axes, every axis has at least two candidates."""
    for name in _BENCHMARKS:
        grid = _grid(name)
        for key in ("eta", "sigma_sys", "prior_std", "beta",
                    "window", "n_passes", "ph_delta", "ph_lambda", "ph_alpha"):
            assert len(set(grid[key])) >= 2, f"{name}.{key} has a single value"
