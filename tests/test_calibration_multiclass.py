# -*- coding: utf-8 -*-
"""Tests for the multiclass calibration metrics.

classification_analysis was originally binary-only and ravelled probs before
passing them to brier_ece. With C = 6 the probabilities are (B, C), so
ravelling pairs the first B probability values with the B labels and returns
meaningless numbers without raising (a six-class Brier score of 8.33 was
observed, against a range of [0, 2]).

Only assertions on the ranges and on known values can catch that kind of
silent breakage, so they are pinned down here.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from src.evaluation.metrics import (  # noqa: E402
    brier_ece, brier_ece_multiclass, brier_multiclass,
)


def _softmax_probs(rng, n, C):
    logits = rng.normal(size=(n, C))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ======================================================================
# Ranges: the direct test for silent breakage
# ======================================================================
@pytest.mark.parametrize("C", [3, 6, 10])
def test_multiclass_brier_in_range(C):
    """The multiclass Brier score lies in [0, 2]; the old path returned 8.33."""
    rng = np.random.default_rng(0)
    n = 500
    probs = _softmax_probs(rng, n, C)
    y = rng.integers(0, C, size=n)
    brier, ece, _, _ = brier_ece_multiclass(probs, y, C)
    assert 0.0 <= brier <= 2.0, f"Brier={brier} outside [0, 2]"
    assert 0.0 <= ece <= 1.0, f"ECE={ece} outside [0, 1]"


def test_multiclass_brier_matches_brier_multiclass():
    """The Brier score of brier_ece_multiclass matches brier_multiclass."""
    rng = np.random.default_rng(1)
    C, n = 6, 400
    probs = _softmax_probs(rng, n, C)
    y = rng.integers(0, C, size=n)
    brier, _, _, _ = brier_ece_multiclass(probs, y, C)
    assert brier == pytest.approx(brier_multiclass(probs, y, C), rel=1e-12)


# ======================================================================
# Known values
# ======================================================================
def test_multiclass_perfect_prediction():
    """Perfect, fully confident predictions give Brier = 0 and ECE = 0."""
    C, n = 6, 120
    rng = np.random.default_rng(2)
    y = rng.integers(0, C, size=n)
    probs = np.zeros((n, C))
    probs[np.arange(n), y] = 1.0
    brier, ece, rel_x, rel_y = brier_ece_multiclass(probs, y, C)
    assert brier == pytest.approx(0.0, abs=1e-12)
    assert ece == pytest.approx(0.0, abs=1e-12)
    assert rel_x[-1] == pytest.approx(1.0)
    assert rel_y[-1] == pytest.approx(1.0)


def test_multiclass_uniform_prediction():
    """Uniform predictions: confidence is 1/C and accuracy is empirical.

    Brier = mean Σ_c (1/C − onehot)² = (C−1)/C²·1 + ... = 1 − 1/C
    ECE   = |1/C − accuracy|
    """
    C, n = 4, 800
    rng = np.random.default_rng(3)
    y = rng.integers(0, C, size=n)
    probs = np.full((n, C), 1.0 / C)
    brier, ece, _, _ = brier_ece_multiclass(probs, y, C)
    assert brier == pytest.approx(1.0 - 1.0 / C, rel=1e-12)
    # Every argmax ties, and np.argmax returns 0, so accuracy = P(y == 0).
    acc = float((y == 0).mean())
    assert ece == pytest.approx(abs(1.0 / C - acc), rel=1e-12)


def test_multiclass_ece_is_top_label():
    """The ECE is top-label: confidence is the max probability, correctness the argmax."""
    C = 3
    probs = np.array([
        [0.9, 0.05, 0.05],   # conf 0.9, correct
        [0.9, 0.05, 0.05],   # conf 0.9, wrong
        [0.4, 0.35, 0.25],   # conf 0.4, correct
        [0.4, 0.35, 0.25],   # conf 0.4, wrong
    ])
    y = np.array([0, 1, 0, 1])
    _, ece, _, _ = brier_ece_multiclass(probs, y, C)
    # bin [0.4, 0.5): conf 0.4, acc 0.5 -> |0.4 - 0.5| * (2/4) = 0.05
    # bin [0.9, 1.0): conf 0.9, acc 0.5 -> |0.9 - 0.5| * (2/4) = 0.20
    assert ece == pytest.approx(0.05 + 0.20, rel=1e-12)


# ======================================================================
# Input validation: a wrong shape must not pass silently
# ======================================================================
def test_multiclass_rejects_wrong_shape():
    """Anything but (B, C) raises, so the old silent failure cannot return."""
    C, n = 6, 50
    rng = np.random.default_rng(4)
    y = rng.integers(0, C, size=n)
    flat = _softmax_probs(rng, n, C).ravel()      # the shape the old path produced
    with pytest.raises(ValueError):
        brier_ece_multiclass(flat, y, C)


def test_multiclass_rejects_length_mismatch():
    C, n = 6, 50
    rng = np.random.default_rng(5)
    probs = _softmax_probs(rng, n, C)
    with pytest.raises(ValueError):
        brier_ece_multiclass(probs, np.zeros(n - 1, dtype=int), C)


# ======================================================================
# The binary path must keep the published email numbers unchanged
# ======================================================================
def test_binary_path_unchanged_by_dispatch():
    """The binary path of classification_analysis matches brier_ece exactly.

    Adding the multiclass branch must not move the binary results by a single
    bit, or the published email numbers would no longer reproduce.
    """
    from calibration_report import classification_analysis

    rng = np.random.default_rng(6)
    n = 300
    probs = rng.uniform(size=n)
    y = (rng.uniform(size=n) < probs).astype(np.float64)

    results = {"PF": [{"predictions": {"probs": probs, "y": y}}]}

    # Both n_classes=None (email has no such attribute) and 2 take the
    # binary path.
    for n_classes in (None, 2):
        rows, rel = classification_analysis({}, results, n_classes=n_classes)
        got_brier = [r for r in rows if r["metric"] == "brier"][0]["mean"]
        got_ece = [r for r in rows if r["metric"] == "ece"][0]["mean"]
        exp_brier, exp_ece, exp_x, exp_y = brier_ece(probs, y, n_bins=10)
        assert got_brier == pytest.approx(exp_brier, rel=1e-15)
        assert got_ece == pytest.approx(exp_ece, rel=1e-15)
        np.testing.assert_allclose(rel["PF"][0], exp_x, rtol=1e-15)
        np.testing.assert_allclose(rel["PF"][1], exp_y, rtol=1e-15)


def test_classification_analysis_multiclass_end_to_end():
    """classification_analysis handles multiclass probs of shape (B, C)."""
    from calibration_report import classification_analysis

    rng = np.random.default_rng(7)
    C, n = 6, 250
    probs = _softmax_probs(rng, n, C)
    y = rng.integers(0, C, size=n).astype(np.float64)
    results = {"PF": [{"predictions": {"probs": probs, "y": y}}]}

    rows, rel = classification_analysis({}, results, n_classes=C)
    brier = [r for r in rows if r["metric"] == "brier"][0]["mean"]
    ece = [r for r in rows if r["metric"] == "ece"][0]["mean"]
    assert 0.0 <= brier <= 2.0
    assert 0.0 <= ece <= 1.0
    assert brier == pytest.approx(brier_multiclass(probs, y.astype(int), C))
    # The reliability-diagram coordinates are in range too.
    assert np.all((rel["PF"][0] >= 0.0) & (rel["PF"][0] <= 1.0))
    assert np.all((rel["PF"][1] >= 0.0) & (rel["PF"][1] <= 1.0))


def test_multiclass_raises_on_flattened_probs_via_analysis():
    """Flattened probs under a multiclass setting raise, as the old bug did not."""
    from calibration_report import classification_analysis

    rng = np.random.default_rng(8)
    C, n = 6, 100
    probs = _softmax_probs(rng, n, C).ravel()
    y = rng.integers(0, C, size=n).astype(np.float64)
    results = {"PF": [{"predictions": {"probs": probs, "y": y}}]}
    with pytest.raises(ValueError):
        classification_analysis({}, results, n_classes=C)
