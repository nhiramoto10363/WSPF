#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1-13: PCA 学習・選択区間のリーク除去を検証する。

  - Email の PCA は pca_fit_end より前の観測だけで主成分方向を学習する
    (将来のサンプルを見ない)。fit 区間を変えると早期サンプルの射影が変わる
    ことで、fit が prefix に限定されていることを確認する。
  - ストリームは因果順(train インデックスが単調非減少・非重複)である。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ARFF = os.path.join(os.path.dirname(__file__), "..", "Email", "email_data.arff")
_HAS_DATA = os.path.exists(_ARFF)
_skip = pytest.mark.skipif(not _HAS_DATA, reason="Email データが無い環境")


@_skip
def test_pca_fit_restricted_to_prefix():
    """PCA の fit 区間を変えると早期サンプルの射影が変わる
    (=fit が pca_fit_end までの prefix に限定されている証拠)。"""
    from src.benchmarks.loaders import EmailDataLoader

    early = EmailDataLoader(_ARFF, n_components=20, seed=42, pca_fit_end=300)
    full = EmailDataLoader(_ARFF, n_components=20, seed=42, pca_fit_end=None)

    # 同じ生データの先頭 300 行でも、fit 区間が異なれば主成分方向が変わり
    # 射影は一致しないはず(符号・スケールの差では説明できない全体差)。
    a = early.X[:300]
    b = full.X[:300]
    assert a.shape == b.shape
    assert not np.allclose(np.abs(a), np.abs(b)), (
        "fit 区間を変えても射影が不変 → PCA が prefix に限定されていない疑い")


@_skip
def test_stream_is_causal():
    """train インデックスが単調非減少・ブロック非重複(因果順)。"""
    from src.benchmarks.email import EmailBenchmark

    steps = list(EmailBenchmark().stream(0))
    last_end = -1
    for s in steps:
        start = int(s.train_indices.min())
        end = int(s.train_indices.max())
        assert start > last_end, f"step {s.step_index}: ブロックが重複/逆行"
        last_end = end


@_skip
def test_pca_dim_and_shape():
    from src.benchmarks.loaders import EmailDataLoader

    ld = EmailDataLoader(_ARFF, n_components=50, seed=42, pca_fit_end=600)
    assert ld.X.shape[1] == 50
    assert ld.pca_fit_end == 600


if __name__ == "__main__":
    if not _HAS_DATA:
        print("test_no_leakage: Email データが無いためスキップ")
        sys.exit(0)
    test_pca_fit_restricted_to_prefix()
    test_stream_is_causal()
    test_pca_dim_and_shape()
    print("test_no_leakage: 全テスト通過")
