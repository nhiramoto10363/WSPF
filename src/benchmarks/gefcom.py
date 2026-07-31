#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク: GEFCom2014-S 太陽光発電予測 (GEFCom)

実世界の点予測回帰タスク。既知のスイッチ点はなく、漸進的な季節ドリフトを持つ
(regression sim / INSECTS の abrupt と相補的)。

プロトコル: **重複しない同一ブロック** prequential test-then-train。
各ブロック B_t = [pos:pos+B] を、更新前モデル θ_{t-1} で予測(=one-step-ahead
forecast)してから同じブロックで学習し、pos += B で前進する。旧実装は
test 窓(TEST=32)を train 窓(B=16)より後ろに重ねて取り、隣接ブロックが
重複していたが、説明の明快さを優先して Email と同じ非重複プロトコルに統一した。

選択区間(select_end_ts より前)と報告区間を、タイムスタンプで分離する
(グリッド探索・最終評価とも同じ全ストリームを連続処理し、集計だけ分ける, R1-13)。

R2-1 修正方針: param_dim を大きくして WSPF-A のランク B 近似の効果を検証する
ため、hidden_dim の既定を 64 とする
(param_dim = (14+1)*64 + (64+1) = 1025 > 833)。
"""

from __future__ import annotations

import os

import numpy as np

from src.models import (
    NeuralNetRegression,
    create_regression_grad_fn,
    create_regression_loglik_fn,
    create_regression_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.loaders import GefcomSolarLoader

# リポジトリ既定のデータ位置 (src/benchmarks/gefcom.py から 3 つ上が WSPF/)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_PREDICTORS = os.path.join(
    _REPO_ROOT, "GEFCom2014", "Solar", "predictors15.csv")


class GefcomBenchmark(Benchmark):
    """GEFCom2014-S 太陽光発電予測ベンチマーク。"""

    name = "gefcom"
    task_type = "regression"
    switch_points = []   # 既知スイッチ点なし (漸進的季節ドリフト)

    def __init__(self, predictors_path=_DEFAULT_PREDICTORS, zone=1,
                 train_path=None, select_end_ts="2012-10-01",
                 hidden_dim=64, batch_size=16, test_size=32,
                 noise_std=0.1, grad_clip_norm=5.0):
        self.predictors_path = predictors_path
        self.zone = int(zone)
        self.train_path = train_path
        self.select_end_ts = select_end_ts
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.noise_std = float(noise_std)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)

        # ローダー読込 (データ欠損時はここで例外。import は失敗しない)
        self.loader = GefcomSolarLoader(
            predictors_path, zone=self.zone, train_path=train_path,
            select_end_ts=select_end_ts)
        self.loader.set_noise_std(self.noise_std)
        self.n_features = self.loader.n_features

        self.model = NeuralNetRegression(self.n_features, self.hidden_dim,
                                         output_dim=1, activation="tanh")
        self.param_dim = self.model.param_dim

    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        noise_std = self.noise_std
        clip = self.grad_clip_norm

        raw_grad = create_regression_grad_fn(model, noise_std)
        per_sample_grad_fn = create_regression_per_sample_grad_fn(
            model, noise_std)
        loglik_fn = create_regression_loglik_fn(model, noise_std)

        def grad_fn(theta, X, y):
            g = raw_grad(theta, X, y)
            if clip is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                scale = np.minimum(1.0, clip / (norms + 1e-12))
                g = g * scale
            return g

        def predict_fn(particles, X):
            output, _, _ = model.forward(particles, X)
            return output.squeeze(-1)   # (N, B)

        return {
            "grad_fn": grad_fn,
            "per_sample_grad_fn": per_sample_grad_fn,
            "loglik_fn": loglik_fn,
            "predict_fn": predict_fn,
            "obs_sigma": noise_std,
        }

    # ------------------------------------------------------------------
    def stream(self, seed: int):
        X, y = self.loader.X, self.loader.y
        n = len(X)
        B = self.batch_size
        # select_mask[i] = True なら i は選択区間(ts < select_end_ts)
        sel_mask = np.asarray(getattr(self.loader, "select_mask",
                                      np.zeros(n, bool)))
        pos = 0
        step_index = 0
        while pos + B <= n:
            blk = np.arange(pos, pos + B)     # 同一ブロック(非重複)
            # ブロック全体が選択区間 / 報告区間のどちらに入るか
            in_sel = bool(sel_mask[blk].all())
            in_rep = bool((~sel_mask[blk]).all())
            yield StreamStep(
                step_index=step_index,
                X_train=X[blk],
                y_train=y[blk],
                X_test=X[blk],
                y_test=y[blk],
                train_indices=blk,
                test_indices=blk,
                test_before_train=True,
                is_selection_step=in_sel,
                is_report_step=in_rep,
                straddles_switch=False,
                is_switch_step=False,
                regime_id=None,
            )
            pos += B
            step_index += 1
