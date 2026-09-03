#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The email protocol: prequential test-then-train on the same block.

Checks that no observation is trained on before it is evaluated, that each is
evaluated exactly once, that straddling blocks are flagged correctly, and
that they are excluded from the switch-aligned aggregate.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.email import EmailBenchmark, DRIFT_POINTS

_ARFF = os.path.join(os.path.dirname(__file__), "..", "data", "email_data.arff")
_HAS_DATA = os.path.exists(_ARFF)
_skip = pytest.mark.skipif(not _HAS_DATA, reason="email data not available")


@_skip
def _steps():
    b = EmailBenchmark()
    return list(b.stream(0))


@_skip
def test_test_before_train():
    for s in _steps():
        assert s.test_before_train is True


@_skip
def test_each_observation_evaluated_once():
    """Concatenating every test block leaves no duplicates."""
    steps = _steps()
    all_test = np.concatenate([s.test_indices for s in steps])
    assert len(all_test) == len(np.unique(all_test)), "test observations repeat"


@_skip
def test_not_trained_before_evaluated():
    """No test observation of step t was trained on at an earlier step."""
    steps = _steps()
    trained = set()
    for s in steps:
        # The step evaluates before it trains, so nothing here may already
        # be in `trained`.
        assert set(s.test_indices.tolist()).isdisjoint(trained), (
            f"step {s.step_index}: an already-trained observation is being evaluated")
        trained.update(s.train_indices.tolist())


@_skip
def test_straddle_flag_correct():
    """straddles_switch is True exactly when the block crosses a drift point."""
    for s in _steps():
        lo, hi = int(s.test_indices.min()), int(s.test_indices.max()) + 1
        expected = any(lo < d < hi for d in DRIFT_POINTS)
        assert s.straddles_switch == expected, (
            f"step {s.step_index}: straddle flag disagrees (range [{lo},{hi}))")


@_skip
def test_straddle_excluded_from_switch_aligned():
    """After excluding straddling blocks, none of the rest crosses a drift point."""
    steps = _steps()
    non_straddle = [s for s in steps if not s.straddles_switch]
    assert len(non_straddle) > 0
    for s in non_straddle:
        lo, hi = int(s.test_indices.min()), int(s.test_indices.max()) + 1
        assert not any(lo < d < hi for d in DRIFT_POINTS)
    # There must be straddling blocks at all, or the test proves nothing.
    assert any(s.straddles_switch for s in steps)


if __name__ == "__main__":
    if not _HAS_DATA:
        print("test_email_protocol: skipped, email data not available")
        sys.exit(0)
    test_test_before_train()
    test_each_observation_evaluated_once()
    test_not_trained_before_evaluated()
    test_straddle_flag_correct()
    test_straddle_excluded_from_switch_aligned()
    print("test_email_protocol: all tests passed")
