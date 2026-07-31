#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク: Email (elist) 二値分類 (Email)

Katakis et al. (2010) の Email データセット。1,500 サンプル, 5 期間 × 300,
4 コンセプトドリフト (index 300, 600, 900, 1200)。二値分類 (interesting/junk)。

R1-12 プロトコル: prequential test-then-train を **同一ブロック** で行う。
各ステップでブロック B_t = [pos:pos+B] を
  1. 更新前の推定 θ_{t-1} で評価し、
  2. 同じブロック B_t で学習し、
  3. 次のブロックへ進む (pos += B, 非重複)。
これにより各観測は学習に使われる前に厳密に 1 回だけ評価され、
test ブロックはステップ間で重複しない
(Each observation is evaluated exactly once before it is used for training,
and test blocks do not overlap across steps)。
なお同一ブロックを評価後に学習へ使う設計なので、「評価と学習が一切重ならない」
という意味ではない点に注意する。

R1-12 straddle: ブロックがドリフト点をまたぐ (block 内で概念が切り替わる)
場合は straddles_switch=True とし、集計側(region_mask)で全ての報告指標・
switch整合解析から除外する(Straddling blocks are excluded from all reported
metrics and switch-aligned analyses)。regime_id は 5 期間 (0..4) をブロック
先頭で判定する。

リークフリー PCA: pca_fit_end (既定 600) より前のみで主成分方向を学習する。
"""

from __future__ import annotations

import os

import numpy as np

from src.models import (
    NeuralNetModel,
    create_nn_grad_fn,
    create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.loaders import EmailDataLoader

# リポジトリ既定のデータ位置
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_ARFF = os.path.join(_REPO_ROOT, "Email", "email_data.arff")

DRIFT_POINTS = [300, 600, 900, 1200]


class EmailBenchmark(Benchmark):
    """Email 二値分類ベンチマーク。"""

    name = "email"
    task_type = "classification"
    switch_points = [300, 600, 900, 1200]

    def __init__(self, arff_path=_DEFAULT_ARFF, pca_dim=50, pca_fit_end=600,
                 hidden_dim=16, batch_size=16, test_size=32,
                 grad_clip_norm=5.0, seed=42, report_start=600):
        self.arff_path = arff_path
        self.pca_dim = int(pca_dim)
        self.pca_fit_end = int(pca_fit_end)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.loader_seed = int(seed)
        # 報告区間の開始インデックス。これより前(選択区間)は最終評価に
        # 集計しない(PCA fit と同様のリーク除去, R1-13)。
        self.report_start = int(report_start)

        # ローダー読込 (データ欠損時はここで例外。import は失敗しない)
        self.loader = EmailDataLoader(
            arff_path, n_components=self.pca_dim, seed=self.loader_seed,
            pca_fit_end=self.pca_fit_end)
        self.input_dim = self.loader.input_dim

        self.model = NeuralNetModel(self.input_dim, self.hidden_dim,
                                    output_dim=1, activation="tanh")
        self.param_dim = self.model.param_dim

    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        clip = self.grad_clip_norm

        raw_grad = create_nn_grad_fn(model)
        per_sample_grad_fn = create_nn_per_sample_grad_fn(model)
        loglik_fn = create_nn_loglik_fn(model)

        def grad_fn(theta, X, y):
            g = raw_grad(theta, X, y)
            if clip is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                scale = np.minimum(1.0, clip / (norms + 1e-12))
                g = g * scale
            return g

        def predict_fn(particles, X):
            # sigmoid 出力 (確率) を返す
            output, _, _ = model.forward(particles, X)
            return output.squeeze(-1)   # (N, B)

        return {
            "grad_fn": grad_fn,
            "per_sample_grad_fn": per_sample_grad_fn,
            "loglik_fn": loglik_fn,
            "predict_fn": predict_fn,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _period_of(index: int) -> int:
        """サンプルインデックスが属する 5 期間 (0..4) を返す。"""
        p = 0
        for d in DRIFT_POINTS:
            if index >= d:
                p += 1
        return p

    def stream(self, seed: int):
        X, y = self.loader.X, self.loader.y
        n = len(X)
        B = self.batch_size
        pos = 0
        step_index = 0
        while pos + B <= n:
            blk = np.arange(pos, pos + B)     # 同一ブロック B_t (非重複)

            # ブロックがドリフト点をまたぐか (block 内で概念が切り替わる)
            straddles = any(pos < d < pos + B for d in DRIFT_POINTS)
            regime_id = self._period_of(pos)

            # 選択区間(report_start より前)と報告区間を分離(R1-13)
            in_rep = (pos >= self.report_start)

            # 評価も学習も同一ブロック: θ_{t-1} で評価 → B_t で学習
            yield StreamStep(
                step_index=step_index,
                X_train=X[blk],
                y_train=y[blk],
                X_test=X[blk],
                y_test=y[blk],
                train_indices=blk,
                test_indices=blk,
                test_before_train=True,
                is_selection_step=(not in_rep),
                is_report_step=in_rep,
                straddles_switch=straddles,
                is_switch_step=straddles,
                regime_id=regime_id,
            )
            pos += B
            step_index += 1
