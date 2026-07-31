#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1-12: Email の時系列プロトコル(prequential test-then-train, 同一ブロック)を検証。

修正方針 R1-12 の必須テスト:
  - 評価前にその観測が学習されていない
  - 各観測の評価回数が 1 回
  - straddle 判定が正しい
  - switch-aligned 集計から straddle が除外される
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks.email import EmailBenchmark, DRIFT_POINTS

_ARFF = os.path.join(os.path.dirname(__file__), "..", "Email", "email_data.arff")
_HAS_DATA = os.path.exists(_ARFF)
_skip = pytest.mark.skipif(not _HAS_DATA, reason="Email データが無い環境")


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
    """全 test_indices を連結して重複が無い(=各観測の評価回数 1)。"""
    steps = _steps()
    all_test = np.concatenate([s.test_indices for s in steps])
    assert len(all_test) == len(np.unique(all_test)), "test 観測が重複している"


@_skip
def test_not_trained_before_evaluated():
    """step t の test 観測が、それ以前のステップで学習されていない。"""
    steps = _steps()
    trained = set()
    for s in steps:
        # このステップは「評価 → 学習」の順。評価時点で trained に含まれてはいけない。
        assert set(s.test_indices.tolist()).isdisjoint(trained), (
            f"step {s.step_index}: 評価前に学習済みの観測が含まれる")
        trained.update(s.train_indices.tolist())


@_skip
def test_straddle_flag_correct():
    """straddles_switch はブロックがドリフト点をまたぐときのみ True。"""
    for s in _steps():
        lo, hi = int(s.test_indices.min()), int(s.test_indices.max()) + 1
        expected = any(lo < d < hi for d in DRIFT_POINTS)
        assert s.straddles_switch == expected, (
            f"step {s.step_index}: straddle 判定不一致 (range [{lo},{hi}))")


@_skip
def test_straddle_excluded_from_switch_aligned():
    """switch-aligned 集計(straddle 除外)で、残るブロックはどれも
    ドリフト点をまたがない。"""
    steps = _steps()
    non_straddle = [s for s in steps if not s.straddles_switch]
    assert len(non_straddle) > 0
    for s in non_straddle:
        lo, hi = int(s.test_indices.min()), int(s.test_indices.max()) + 1
        assert not any(lo < d < hi for d in DRIFT_POINTS)
    # straddle が実在すること(テスト自体の有効性確認)
    assert any(s.straddles_switch for s in steps)


if __name__ == "__main__":
    if not _HAS_DATA:
        print("test_email_protocol: Email データが無いためスキップ")
        sys.exit(0)
    test_test_before_train()
    test_each_observation_evaluated_once()
    test_not_trained_before_evaluated()
    test_straddle_flag_correct()
    test_straddle_excluded_from_switch_aligned()
    print("test_email_protocol: 全テスト通過")
