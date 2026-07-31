#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts 共通ヘルパ

  - config(yaml) の読込
  - config からベンチマーク構築 (benchmark と呼ぶ; model ではない)
  - グリッドサーチ(選択区間・端点自動拡張の警告つき)
  - 主要指標の集計(回帰=MSE最小化, 分類=F1最大化)

各スクリプトはこの薄い層を介して src.evaluation.runner を駆動する。
"""

from __future__ import annotations

import inspect
import itertools
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks import get_benchmark  # noqa: E402
from src.evaluation import run_method, run_seeds  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")

# 各ベンチマークのコンストラクタが受け付ける data キー
_BENCH_CLASSES = None


def _bench_class(name):
    from src.benchmarks import (
        RegressionSwitchBenchmark, GefcomBenchmark, EmailBenchmark)
    return {
        "regression": RegressionSwitchBenchmark,
        "gefcom": GefcomBenchmark,
        "email": EmailBenchmark,
    }[name]


# ======================================================================
# config
# ======================================================================
def load_config(name_or_path):
    """config 名(regression/gefcom/email)またはパスから dict を読む。"""
    if os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(_CONFIG_DIR, f"{name_or_path}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_benchmark(cfg, **overrides):
    """config の data セクションからベンチマークを構築する。"""
    name = cfg["benchmark"]
    cls = _bench_class(name)
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    data = dict(cfg.get("data", {}))
    data.update(overrides)
    # 相対パスはリポジトリルート基準に解決
    for k in ("predictors_path", "train_path", "arff_path"):
        v = data.get(k)
        if isinstance(v, str) and not os.path.isabs(v):
            data[k] = os.path.join(_REPO_ROOT, v)
    kwargs = {k: v for k, v in data.items() if k in accepted and v is not None}
    return get_benchmark(name, **kwargs)


# ======================================================================
# 主要指標の集計
# ======================================================================
def primary_score(result, cfg, exclude_straddle=True):
    """1 実行結果からスカラー主要指標を返す(小さいほど良い方向に符号統一)。

    回帰: 平均 MSE(小さいほど良い) → そのまま
    分類: 平均 F1(大きいほど良い) → 符号反転して「小さいほど良い」に統一
    """
    metrics = result["metrics"]
    mask = np.ones_like(next(iter(metrics.values())), dtype=bool)
    if exclude_straddle and "straddle_mask" in result:
        mask &= ~result["straddle_mask"]
    if cfg["task_type"] == "regression":
        v = np.asarray(metrics["mse"])[mask]
        return float(np.nanmean(v))
    else:
        v = np.asarray(metrics["f1"])[mask]
        return -float(np.nanmean(v))


# ======================================================================
# グリッドサーチ (選択区間: selection シード)
# ======================================================================
def _param_grid(method, grid):
    """手法ごとに探索するパラメータ格子(直積)を生成する。"""
    etas = grid["eta"]
    if method in ("SGD", "PH-SGD", "Window-SGD"):
        for eta, ps in itertools.product(etas, grid["prior_std"]):
            yield {"eta": eta, "prior_std": ps}
    elif method == "WSPF-A":
        for eta, ss, ps, beta in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"], grid["beta"]):
            yield {"eta": eta, "sigma_sys": ss, "prior_std": ps, "beta": beta}
    else:  # PF, WSPF-B, Oracle
        for eta, ss, ps in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"]):
            yield {"eta": eta, "sigma_sys": ss, "prior_std": ps}


def _is_boundary(best, grid):
    """best の各パラメータがグリッド端点にあるか(自動拡張の警告用, 修正方針11)。"""
    hits = []
    for key, gkey in (("eta", "eta"), ("sigma_sys", "sigma_sys"),
                      ("prior_std", "prior_std"), ("beta", "beta")):
        if key in best and gkey in grid:
            g = grid[gkey]
            if best[key] in (min(g), max(g)):
                hits.append(key)
    return hits


def grid_search(method, cfg, n_particles, selection_seeds, emit=print):
    """選択シード平均で最良パラメータを返す。端点採択時は警告する。"""
    grid = cfg["grid"]
    best, best_score = None, np.inf
    for params in _param_grid(method, grid):
        scores = []
        for s in selection_seeds:
            bench = build_benchmark(cfg)
            r = run_method(method, bench, n_particles, params, seed=s,
                           collect_diagnostics=False)
            scores.append(primary_score(r, cfg))
        sc = float(np.nanmean(scores))
        if sc < best_score:
            best_score, best = sc, dict(params)
    hits = _is_boundary(best, grid)
    if hits:
        emit(f"  [警告] {method}: 最良点がグリッド端 {hits} に到達。"
             f"該当方向へ幾何級数的に1段階拡張して再実行を推奨(最大2回)。")
    emit(f"  {method}: best={best} (score={best_score:.4f})")
    return best, best_score


# ======================================================================
# HP 入出力
# ======================================================================
def hp_path(cfg):
    return os.path.join(_REPO_ROOT, cfg["output_dir"], "selected_params.json")


def resolve_seeds(cfg, kind):
    """kind='selection'|'evaluation' のシード列。"""
    return list(cfg["seeds"][kind])


def load_selected(cfg):
    """grid_search.py が保存した selected_params.json を読む。"""
    import json
    path = hp_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} が無い。先に scripts/grid_search.py を実行してください。")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_params(selected, method, n_particles):
    """selected から (method, N) の最良パラメータを取り出す。"""
    if method in ("PF", "WSPF-A", "WSPF-B"):
        return selected["by_n_particles"][str(n_particles)][method]
    return selected["no_n"][method]
