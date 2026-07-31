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
    ev = cfg.get("eval", {})
    # 選択/報告区間の境界を config(eval セクション)からベンチマークへ渡す
    if name == "regression" and "eval_start" in ev:
        data.setdefault("eval_start", ev["eval_start"])
    if name == "email":
        rs = ev.get("report_start", data.get("report_start"))
        if rs is not None:
            data["report_start"] = rs
    # 相対パスはリポジトリルート基準に解決
    for k in ("predictors_path", "train_path", "arff_path"):
        v = data.get(k)
        if isinstance(v, str) and not os.path.isabs(v):
            data[k] = os.path.join(_REPO_ROOT, v)
    # config 由来の None(train_path: null 等)は除外するが、
    # 明示的な overrides は None でも尊重する(例: grad_clip_norm=None)。
    kwargs = {k: v for k, v in data.items() if k in accepted and v is not None}
    for k, v in overrides.items():
        if k in accepted:
            kwargs[k] = v
    return get_benchmark(name, **kwargs)


# ======================================================================
# 主要指標の集計
# ======================================================================
def region_mask(result, region):
    """result から集計対象マスクを返す。

    region='selection' → 選択区間(HP選択)、'report' → 報告区間(最終評価)、
    'all' → 全ステップ。いずれも straddle(切替をまたぐブロック)は除外する。

    選択/報告区間が空のときは **黙って全区間へ戻さず例外を投げる**
    (リーク防止の意図と逆になるため, 修正方針 C1)。設定ミス(eval_start が
    T 以上、report_start がストリーム長を超える 等)を早期に検出する。
    """
    n = len(next(iter(result["metrics"].values())))
    if region == "selection":
        mask = np.asarray(result.get("selection_mask", np.zeros(n, bool)))
    elif region == "report":
        mask = np.asarray(result.get("report_mask", np.zeros(n, bool)))
    elif region == "all":
        mask = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"unknown region: {region!r}")
    if region in ("selection", "report") and not mask.any():
        raise ValueError(
            f"region '{region}' にサンプルがありません(空マスク)。"
            f"eval_start / report_start とストリーム長の設定を確認してください。")
    mask = mask & ~np.asarray(result.get("straddle_mask", np.zeros(n, bool)))
    return mask


def primary_score(result, cfg, region="selection"):
    """1 実行結果からスカラー主要指標を返す(小さいほど良い方向に符号統一)。

    回帰: 平均 MSE(小さいほど良い) → そのまま
    分類: 平均 F1(大きいほど良い) → 符号反転して「小さいほど良い」に統一
    region: 'selection'(HP選択, 既定) / 'report'(最終評価) / 'all'
    """
    metrics = result["metrics"]
    mask = region_mask(result, region)
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
    """手法ごとに探索するパラメータ格子(直積)を生成する。

    ベースラインの固有HPも選択区間で探索する(R2-4):
      PH-SGD    : ph_delta, ph_lambda, ph_alpha (Page-Hinkley)
      Window-SGD: window(W), n_passes(K)
    config の grid にキーが無ければ単一既定値で1点だけ探索する。
    """
    etas = grid["eta"]
    if method == "SGD":
        for eta, ps in itertools.product(etas, grid["prior_std"]):
            yield {"eta": eta, "prior_std": ps}
    elif method == "PH-SGD":
        deltas = grid.get("ph_delta", [0.005])
        lambdas = grid.get("ph_lambda", [5.0])
        alphas = grid.get("ph_alpha", [0.9999])
        for eta, ps, dl, lm, al in itertools.product(
                etas, grid["prior_std"], deltas, lambdas, alphas):
            yield {"eta": eta, "prior_std": ps,
                   "ph_delta": dl, "ph_lambda": lm, "ph_alpha": al}
    elif method == "Window-SGD":
        windows = grid.get("window", [5])
        passes = grid.get("n_passes", [1])
        for eta, ps, w, k in itertools.product(
                etas, grid["prior_std"], windows, passes):
            yield {"eta": eta, "prior_std": ps, "window": w, "n_passes": k}
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


def grid_search(method, cfg, n_particles, selection_seeds, emit=print,
                contexts=None):
    """選択区間スコアで最良パラメータを返す。端点採択時は警告する。

    contexts: ベンチマーク構築の override 辞書のリスト(例: GEFCom の
    [{'zone':1,'noise_std':..}, {'zone':2,..}, ..])。複数指定すると
    その **全コンテキスト×選択シードの平均** 選択区間スコアで選ぶ
    (GEFCom の 3zone 平均による共通 HP 選択, C3)。既定は単一 [{}]。
    """
    grid = cfg["grid"]
    contexts = contexts or [{}]
    best, best_score = None, np.inf
    for params in _param_grid(method, grid):
        scores = []
        for ctx in contexts:
            for s in selection_seeds:
                bench = build_benchmark(cfg, **ctx)
                r = run_method(method, bench, n_particles, params, seed=s,
                               collect_diagnostics=False)
                scores.append(primary_score(r, cfg, region="selection"))
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


def estimate_obs_noise(cfg, zone=None, eta=0.1, prior_std=0.1):
    """GEFCom の観測ノイズ σ_obs を **選択区間だけ** から推定する(P3/R1-13)。

    選択区間を短い OnlineSGD で 1 パス学習しながら、学習前予測の残差
    (y − ŷ) を選択区間の後半で集め、その標準偏差を σ_obs とする。
    将来(報告区間)のデータは一切見ない。回帰ベンチマーク用。
    """
    from src.baselines import OnlineSGD

    overrides = {}
    if zone is not None:
        overrides["zone"] = zone
    bench = build_benchmark(cfg, **overrides)
    funcs = bench.build_functions(0)
    grad_fn, predict_fn = funcs["grad_fn"], funcs["predict_fn"]
    sgd = OnlineSGD(bench.param_dim, eta, prior_std, grad_fn, seed=0,
                    grad_clip_norm=getattr(bench, "grad_clip_norm", None))

    steps = [s for s in bench.stream(0) if s.is_selection_step]
    if not steps:
        return float(getattr(bench, "noise_std", 0.1))
    warmup = len(steps) // 2          # 前半はウォームアップ、後半で残差収集
    resid = []
    for i, s in enumerate(steps):
        theta = sgd.predict_theta().reshape(1, -1)
        pred = np.asarray(predict_fn(theta, s.X_test)).ravel()
        if i >= warmup:
            resid.append(np.asarray(s.y_test, float).ravel() - pred)
        sgd.train(s.X_train, s.y_train)
    if not resid:
        return float(getattr(bench, "noise_std", 0.1))
    sigma = float(np.std(np.concatenate(resid)))
    return max(sigma, 1e-3)
