#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク: GEFCom2014-P 電力価格予測 (gefcom_price)

実世界の点予測回帰タスク。Solar と役割を分担する相補ペア:
  - Solar (gefcom): 漸進的季節ドリフト・大 d (R2-1: rank-B 近似の検証)
  - Price (gefcom_price): 急峻な価格スパイク・確率評価が主・中庸 d (d=321)

負荷 → 価格写像の時間変動により σ_cd の内部最適が立つ (SGD 単独で追い切れない)。
既知スイッチ点はなく、switch-aligned 解析は行わない (スパイク・エピソード別誤差で報告)。

プロトコル: Solar と同一の **重複しない同一ブロック** prequential test-then-train。
各ブロック B_t = [pos:pos+B] を更新前モデル θ_{t-1} で予測してから同じブロックで
学習し、pos += B で前進する。選択区間 (select_end_ts より前) と報告区間を
タイムスタンプで分離する (R1-13)。

モデルは既存 NeuralNetRegression をそのまま使用 (input 8, hidden 32 →
param_dim = (8+1)*32 + (32+1) = 321)。観測モデルは初版ガウス (D1 の asinh 変換で
裾を制御)。Student-t は D1 破綻時のフォールバック。
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
from src.benchmarks.loaders import GefcomPriceLoader

# リポジトリ既定のデータ位置 (src/benchmarks/gefcom_price.py から 2 つ上が WSPF/)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DATA = os.path.join(
    _REPO_ROOT, "GEFCom2014", "Price", "price15.csv")


class GefcomPriceBenchmark(Benchmark):
    """GEFCom2014-P 電力価格予測ベンチマーク (単一ゾーン)。"""

    name = "gefcom_price"
    task_type = "regression"
    switch_points = []   # 既知スイッチ点なし (Solar と同じ)

    def __init__(self, data_path=_DEFAULT_DATA, select_end_ts="2012-01-01",
                 hidden_dim=32, batch_size=16, test_size=32,
                 noise_std=0.1, grad_clip_norm=5.0,
                 target_transform="asinh"):
        self.data_path = data_path
        self.select_end_ts = select_end_ts
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.noise_std = float(noise_std)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.target_transform = str(target_transform)

        # ローダー読込 (データ欠損時はここで例外。import は失敗しない)
        self.loader = GefcomPriceLoader(
            data_path, select_end_ts=select_end_ts,
            target_transform=target_transform)
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
        # select_mask[i] = True なら i は選択区間 (ts < select_end_ts)
        sel_mask = np.asarray(getattr(self.loader, "select_mask",
                                      np.zeros(n, bool)))
        pos = 0
        step_index = 0
        while pos + B <= n:
            blk = np.arange(pos, pos + B)     # 同一ブロック (非重複)
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
