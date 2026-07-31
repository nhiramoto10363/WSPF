#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク(実験課題)共通プロトコル

修正方針の指摘に従い、`model` ではなく `benchmark` と呼ぶ
(Regression / GEFCom / Email はモデルではなく実験課題・データセット)。

各ベンチマークは
  1. モデル依存の関数群(勾配・対数尤度・予測)を `build_functions(seed)` で供給し、
  2. データストリームを `stream(seed)` で StreamStep 列として供給する。

これにより evaluation/runner.py は、ベンチマークの中身を知らずに
PF / WSPF-A / WSPF-B / Oracle / SGD 系を同一ループで駆動できる。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ======================================================================
# ストリームの 1 ステップ (R1-12: prequential test-then-train プロトコル)
# ======================================================================
@dataclass
class StreamStep:
    """
    ストリームの 1 ブロック分の学習・評価データとメタ情報。

    prequential test-then-train:
        まず theta_{t-1} で test ブロックを評価し、その後 train ブロックで更新する。
        (各観測は「学習前に 1 回評価 → その後 1 回学習」となる)

    Attributes
    ----------
    step_index : int
        ストリーム内の 0 始まりのステップ番号。
    X_train, y_train : ndarray
        このステップで学習に使うミニバッチ (B サンプル)。
    X_test, y_test : ndarray
        このステップで(学習前に)評価に使うブロック。
    train_indices, test_indices : ndarray
        元データにおけるグローバルインデックス(リーク検査・再現用, R1-12/R1-13)。
    test_before_train : bool
        評価が学習より前かどうか(prequential では True)。
    straddles_switch : bool
        この test ブロックが概念切替点をまたぐか。
        True のブロックは全体集計には含めてよいが、
        stable / post-switch 分析からは除外する(R1-12)。
    regime_id : Optional[int]
        現在のレジーム識別子(切替のあるベンチマークのみ)。
    """

    step_index: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    test_indices: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    test_before_train: bool = True
    straddles_switch: bool = False
    regime_id: Optional[int] = None


# ======================================================================
# ベンチマーク基底クラス
# ======================================================================
class Benchmark(ABC):
    """
    実験課題の共通インタフェース。

    サブクラスが実装/公開すべきもの:
      - name : str                       課題名 ("regression" / "gefcom" / "email")
      - task_type : str                  "regression" | "classification"
      - param_dim : int                  パラメータ次元数 d
      - switch_points : list[int]        既知の概念切替ステップ(なければ [])
      - build_functions(seed) -> dict    下記の関数群を返す
      - stream(seed) -> Iterator[StreamStep]

    build_functions() が返す dict のキー:
      - "grad_fn"            : grad_fn(theta[N, d], X, y) -> (N, d)
                               バッチ平均勾配。N=1 で baselines にも使える。
      - "per_sample_grad_fn" : (particles[N, d], X, y) -> (N, B, d)
                               各サンプル勾配(WSPF-A/B が要求)。
      - "loglik_fn"          : (particles[N, d], X, y) -> (N,)
                               各粒子のブロック対数尤度(ミニバッチは合計)。
      - "predict_fn"         : (particles[N, d], X) -> 予測(回帰は関数値, 分類はlogit/確率)
      - 回帰のみ:
          "obs_sigma"        : 観測ノイズ標準偏差 σ_obs (予測区間 y=f+ε に使用, R2-5)
      - Oracle 用(回帰のみ, R1-5):
          "oracle_grad_fn"   : (theta[N, d], X, y) -> (N, d)  真の母集団勾配 ∇L
          "oracle_noise_cov_fn" : (theta[N, d], X, y, batch_size) -> ...
                               真の勾配ノイズ共分散(スカラー s または行列)。
    """

    name: str = "base"
    task_type: str = "regression"
    param_dim: int = 0
    switch_points: list = []

    @abstractmethod
    def build_functions(self, seed: int) -> dict:
        """モデル依存の関数群を返す(上記キー)。"""
        raise NotImplementedError

    @abstractmethod
    def stream(self, seed: int):
        """StreamStep のイテレータを返す。"""
        raise NotImplementedError

    # -- 便利メソッド -------------------------------------------------
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def near_switch(self, step_index: int, offset: int) -> bool:
        """step_index が切替点から offset ステップ以内かどうか(post-switch分析用)。"""
        return any(0 <= step_index - sp <= offset for sp in self.switch_points)
