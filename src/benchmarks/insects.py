#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク: INSECTS (abrupt, balanced) 多クラス分類

Souza et al. (2020) の実世界ストリーム。33 実数値特徴 + 6 クラス(蚊 3 種 ×
雌雄)。温度制御による既知の急激な概念スイッチ点を持つ。約 52,848 サンプル。

プロトコル(R1-12 test-then-train, R1-13 リーク除去):
  各ステップで train ブロック B_t = X[pos:pos+B](B サンプル)と、その直後の
  test 窓 X[pos+B:pos+B+T](T サンプル)を取り出す。
    1. 更新前の θ_{t-1}(粒子群)で test 窓を評価し、
    2. train ブロック B_t で更新し、
    3. pos += B(非重複ブロック)。
  test 窓は train ブロックより先(未学習の未来)にあり、評価は必ず学習前に
  行われるためリークはない。標準化は選択区間末尾(scale_fit_end)までで fit。

選択/報告区間(regression の教訓: 選択区間は必ず切替点を含める):
  実データ単一ストリームのためシード分離は使えないが、prequential な時間分割
  なので email と同じくリーク上の問題はない。既定では先頭 select_end_step
  ステップ(= select_end_step*batch_size サンプル)を選択区間とし、残りを報告
  区間とする。既定 select_end_step=1250(≈ サンプル 20000)は最初の 2 スイッチ
  (14352, 19500)を含み、報告区間には残り 3 スイッチ(33240, 38682, 39510)が
  入る。

★切替点定数はデータ添付の README / 原論文 Table と照合すること★
  (バリアント取り違えは switch-aligned 解析を全て無効にする)
"""

from __future__ import annotations

import os

import numpy as np

from src.models import (
    MulticlassNeuralNetModel,
    create_mc_grad_fn,
    create_mc_loglik_fn,
    create_mc_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.loaders import InsectsDataLoader
from src.benchmarks.loaders.insects_loader import CHANGE_POINTS_ABRUPT_BALANCED

# リポジトリ既定のデータ位置
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CSV = os.path.join(
    _REPO_ROOT, "INSECTS", "INSECTS-abrupt_balanced_norm.csv")


class InsectsBenchmark(Benchmark):
    """INSECTS (abrupt, balanced) 多クラス分類ベンチマーク。"""

    name = "insects"
    task_type = "classification"
    # スイッチ点(サンプルインデックス)。__init__ で loader の値に上書きする。
    switch_points = list(CHANGE_POINTS_ABRUPT_BALANCED)

    def __init__(self, csv_path=_DEFAULT_CSV, hidden_dim=32, batch_size=16,
                 test_size=32, grad_clip_norm=5.0, seed=42,
                 select_end_step=1250, change_points=None):
        self.csv_path = csv_path
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.grad_clip_norm = (None if grad_clip_norm is None
                               else float(grad_clip_norm))
        self.loader_seed = int(seed)
        # 選択区間の終端(バッチステップ単位)。これ以降を報告区間とする。
        self.select_end_step = int(select_end_step)
        # 標準化リーク除去: 選択区間末尾(サンプル単位)までで fit(R1-13)。
        scale_fit_end = self.select_end_step * self.batch_size

        # ローダー読込(データ欠損時はここで例外。import は失敗しない)
        self.loader = InsectsDataLoader(
            csv_path, scale_fit_end=scale_fit_end,
            change_points=change_points, seed=self.loader_seed)
        self.input_dim = self.loader.n_features        # 33
        self.n_classes = self.loader.n_classes         # 6
        self.switch_points = list(self.loader.change_points)  # サンプル単位

        self.model = MulticlassNeuralNetModel(
            self.input_dim, self.hidden_dim, self.n_classes, activation="tanh")
        self.param_dim = self.model.param_dim

    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        clip = self.grad_clip_norm

        raw_grad = create_mc_grad_fn(model)
        per_sample_grad_fn = create_mc_per_sample_grad_fn(model)
        loglik_fn = create_mc_loglik_fn(model)

        def grad_fn(theta, X, y):
            g = raw_grad(theta, X, y)
            if clip is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                scale = np.minimum(1.0, clip / (norms + 1e-12))
                g = g * scale
            return g

        def predict_fn(particles, X):
            # 多クラス: 粒子ごとのクラス確率 (N, B, C) を返す。
            # runner 側で重み平均 → argmax する。
            probs, _, _ = model.forward(particles, X)
            return probs   # (N, B, C)

        return {
            "grad_fn": grad_fn,
            "per_sample_grad_fn": per_sample_grad_fn,
            "loglik_fn": loglik_fn,
            "predict_fn": predict_fn,
        }

    # ------------------------------------------------------------------
    def _regime_of(self, index: int) -> int:
        """サンプルインデックスが属するレジーム (0..) を返す。"""
        r = 0
        for cp in self.switch_points:
            if index >= cp:
                r += 1
        return r

    def stream(self, seed: int):
        X, y = self.loader.X, self.loader.y
        n = len(X)
        B = self.batch_size
        T = self.test_size
        cps = self.switch_points
        pos = 0
        step_index = 0
        while pos + B + T <= n:
            tr = np.arange(pos, pos + B)                 # train ブロック
            te = np.arange(pos + B, pos + B + T)         # test 窓(先読み)
            te0, te1 = pos + B, pos + B + T

            # test 窓が概念切替点をまたぐ(窓内で概念が変わる)か
            straddles = any(te0 < cp < te1 for cp in cps)
            regime_id = self._regime_of(te0)

            # 選択区間(step < select_end_step)と報告区間を分離。
            # 選択区間は最初の 2 スイッチを含む(regression の教訓)。
            in_sel = (step_index < self.select_end_step)

            yield StreamStep(
                step_index=step_index,
                X_train=X[tr],
                y_train=y[tr],
                X_test=X[te],
                y_test=y[te],
                train_indices=tr,
                test_indices=te,
                test_before_train=True,
                is_selection_step=in_sel,
                is_report_step=(not in_sel),
                straddles_switch=straddles,
                is_switch_step=straddles,
                regime_id=regime_id,
            )
            pos += B
            step_index += 1
