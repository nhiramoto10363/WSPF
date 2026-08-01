#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク: 回帰レジームスイッチ (Regression regime-switch)

真の生成分布が既知の合成回帰タスク。θ* が 2 レジーム [θ1, θ2, θ1, θ2, θ1] を
switch_times=[100,200,300,400] で切り替え、レジーム内は微小ランダムウォーク
(within_regime_drift) で緩やかに漂う。データ生成ロジックは旧
`experiments/regression_regime_switch.py::generate_regression_regime_data` と同一。

生成分布が既知なので、各ステップ・各粒子の真の母集団勾配 ∇L(θ) と勾配ノイズ
共分散 Σ(θ)=Cov(ĝ_batch)=C(θ)/B を大標本モンテカルロで高精度推定できる
(旧 `oracle_regression.py::oracle_grad_stats` と同一)。これを WSPF-oracle が
厳密補正式にそのまま代入する。
"""

from __future__ import annotations

import numpy as np

from src.models import (
    NeuralNetRegression,
    create_regression_grad_fn,
    create_regression_loglik_fn,
    create_regression_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep

# オラクル MC の既定設定 (oracle_regression.py に準拠)
ORACLE_SAMPLES = 10000     # 各ステップの MC 標本数 (∇L, Σ 推定)
ORACLE_CHUNK = 2500        # メモリ節約のためのチャンク分割


def oracle_grad_stats(model, particles, theta_star, noise_std, M, B,
                      rng, chunk=ORACLE_CHUNK):
    """
    時刻 t の真の分布 (x~N(0,1), y=f(θ*;x)+N(0,σ²)) から大標本を引き、
    各粒子 θ^i について ∇L(θ^i)=E[∇ℓ] と Σ(θ^i)=Cov(ĝ_batch)=C(θ^i)/B を
    高精度推定する。チャンク処理で (N,M,d) の一括確保を避ける。

    旧 `experiments/oracle_regression.py::oracle_grad_stats` と同一実装。

    Returns
    -------
    grad_L : ndarray (N, d)   真の母集団勾配 ∇L
    Sigma  : ndarray (N, d, d) 勾配ノイズ共分散 Cov(ĝ_batch) = C/B
    """
    N, d = particles.shape
    ps_grad_fn = create_regression_per_sample_grad_fn(model, noise_std)
    sum_g = np.zeros((N, d))
    sum_ggT = np.zeros((N, d, d))
    done = 0
    while done < M:
        m = min(chunk, M - done)
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = ps_grad_fn(particles, X, y)            # (N, m, d)
        sum_g += g.sum(axis=1)
        sum_ggT += np.einsum("nmd,nme->nde", g, g)
        done += m
    grad_L = sum_g / M                              # ∇L  (N,d)
    # C = (Σggᵀ − M·mean meanᵀ)/(M−1) → per-sample 勾配共分散
    C = (sum_ggT - M * np.einsum("nd,ne->nde", grad_L, grad_L)) / (M - 1)
    Sigma = C / B                                   # Cov(ĝ_batch) = C/B
    return grad_L, Sigma


class RegressionSwitchBenchmark(Benchmark):
    """回帰レジームスイッチ・ベンチマーク。"""

    name = "regression"
    task_type = "regression"
    switch_points = [100, 200, 300, 400]

    def __init__(self, T=500, batch_size=16, test_size=200, noise_std=0.5,
                 hidden_dim=8, input_dim=1, within_regime_drift=0.0005,
                 grad_clip_norm=5.0, oracle_samples=ORACLE_SAMPLES,
                 eval_start=50, select_start=50, select_end=150):
        self.T = int(T)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.noise_std = float(noise_std)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.within_regime_drift = float(within_regime_drift)
        # None なら勾配クリッピング無効(Oracle/R1-6 の整合用)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.oracle_samples = int(oracle_samples)
        # 報告区間(最終評価): step >= eval_start。step < eval_start はウォームアップ
        # として連続処理はするがスコア集計しない。R1-13。
        self.eval_start = int(eval_start)
        # 選択区間(HP選択のスコア集計対象): [select_start, select_end)。
        # 最初のレジームチェンジ(t=100)を中心に前後50ステップ=[50,150) とし、
        # 定常のみで η が過小選択される問題を回避する(切替を含む窓で選ぶ)。
        self.select_start = int(select_start)
        self.select_end = int(select_end)

        # モデルは 1 つ生成して使い回す (param_dim をここで確定)
        self.model = NeuralNetRegression(self.input_dim, self.hidden_dim,
                                         output_dim=1, activation="tanh")
        self.param_dim = self.model.param_dim

        # switch_points は generate 後に上書き確認する (T=500 の既定と一致)
        self.switch_points = [self.T // 5 * (i + 1) for i in range(4)]

        # seed ごとの生成データキャッシュ
        self._cache = {}
        # build_functions / stream が最後に扱った seed の真パラメータ
        self.theta_true = None

    # ------------------------------------------------------------------
    # データ生成 (generate_regression_regime_data と同一)
    # ------------------------------------------------------------------
    def _generate(self, seed):
        if seed in self._cache:
            return self._cache[seed]

        rng = np.random.default_rng(seed)
        model = self.model
        param_dim = model.param_dim
        T = self.T

        theta_star_1 = rng.normal(0.0, 0.8, size=param_dim)
        theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)
        while np.linalg.norm(theta_star_1 - theta_star_2) < 1.0:
            theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)

        regime_length = T // 5
        switch_times = [regime_length * (i + 1) for i in range(4)]
        regime_thetas = [theta_star_1, theta_star_2, theta_star_1,
                         theta_star_2, theta_star_1]

        theta_true = np.empty((T, param_dim))
        regime_ids = np.empty(T, dtype=int)
        for t in range(T):
            regime_idx = 0
            for st in switch_times:
                if t >= st:
                    regime_idx += 1
            regime_ids[t] = regime_idx
            if t == 0 or t in switch_times:
                theta_true[t] = regime_thetas[regime_idx].copy()
            else:
                theta_true[t] = theta_true[t - 1] + rng.normal(
                    0.0, self.within_regime_drift, size=param_dim)

        X_train, y_train, X_test, y_test = [], [], [], []
        for t in range(T):
            theta_t = theta_true[t: t + 1]
            X = rng.normal(0.0, 1.0, size=(self.batch_size, model.input_dim))
            output, _, _ = model.forward(theta_t, X)
            y = output.squeeze() + rng.normal(0.0, self.noise_std,
                                              size=self.batch_size)
            X_train.append(X)
            y_train.append(y)

            Xte = rng.normal(0.0, 1.0, size=(self.test_size, model.input_dim))
            output_te, _, _ = model.forward(theta_t, Xte)
            yte = output_te.squeeze() + rng.normal(0.0, self.noise_std,
                                                   size=self.test_size)
            X_test.append(Xte)
            y_test.append(yte)

        data = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "theta_true": theta_true, "regime_ids": regime_ids,
            "switch_times": switch_times,
        }
        self._cache[seed] = data
        return data

    # ------------------------------------------------------------------
    # モデル依存の関数群
    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        noise_std = self.noise_std

        # oracle_stats_fn がステップ別の真パラメータを引けるよう、
        # ここで対象 seed の生成データを確定させる。
        data = self._generate(seed)
        self.theta_true = data["theta_true"]

        raw_grad = create_regression_grad_fn(model, noise_std)
        per_sample_grad_fn = create_regression_per_sample_grad_fn(
            model, noise_std)
        loglik_fn = create_regression_loglik_fn(model, noise_std)
        clip = self.grad_clip_norm

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
            # runner は f["oracle_stats_fn"](step_index, rng) で
            # ステップ別クロージャ (particles, X, y) -> (grad_L, Sigma) を得る。
            "oracle_stats_fn": self.oracle_stats_fn_for_step,
        }

    # ------------------------------------------------------------------
    # オラクル勾配統計 (ステップ別)
    # ------------------------------------------------------------------
    def oracle_stats_fn_for_step(self, step_index: int, rng):
        """
        指定ステップの真パラメータ θ*_t を用いたオラクル統計クロージャを返す。

        runner 側の使い方::

            f = benchmark.build_functions(seed)   # theta_true を確定
            for step in benchmark.stream(seed):
                osf = f["oracle_stats_fn"](step.step_index, oracle_rng)
                grad_L, Sigma = osf(particles, step.X_train, step.y_train)

        Returns
        -------
        closure : (particles, X, y) -> (grad_L (N,d), Sigma (N,d,d))
        """
        if self.theta_true is None:
            raise RuntimeError(
                "theta_true が未確定です。先に build_functions(seed) または "
                "stream(seed) を呼んでください。")
        theta_star_t = self.theta_true[step_index]

        def _closure(particles, X, y):
            # X, y は真分布から MC 再サンプルするため実バッチは使わない
            return oracle_grad_stats(
                self.model, particles, theta_star_t, self.noise_std,
                self.oracle_samples, self.batch_size, rng)

        return _closure

    # ------------------------------------------------------------------
    # ストリーム
    # ------------------------------------------------------------------
    def stream(self, seed: int):
        data = self._generate(seed)
        self.theta_true = data["theta_true"]
        switch_times = set(data["switch_times"])
        regime_ids = data["regime_ids"]

        for t in range(self.T):
            yield StreamStep(
                step_index=t,
                X_train=data["X_train"][t],
                y_train=data["y_train"][t],
                X_test=data["X_test"][t],
                y_test=data["y_test"][t],
                train_indices=np.empty(0, int),
                test_indices=np.empty(0, int),
                test_before_train=True,
                regime_id=int(regime_ids[t]),
                # 選択区間 [select_start, select_end)(切替を含む窓)と
                # 報告区間 t>=eval_start を分離(R1-13)。選択と報告は別シードで
                # 実行されるためステップ範囲が重なってもリークにはならない。
                is_selection_step=(self.select_start <= t < self.select_end),
                is_report_step=(t >= self.eval_start),
                # 合成タスクでは train/test を θ*_t から独立生成するため
                # test ブロックは切替点を「またがない」→ straddles_switch=False。
                # 切替時点そのものは is_switch_step で示す(全体集計から除外しない)。
                straddles_switch=False,
                is_switch_step=(t in switch_times),
            )
