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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks import get_benchmark  # noqa: E402
from src.evaluation import run_method, run_seeds  # noqa: E402
from src.evaluation import resolve_workers, _init_worker  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")

# 各ベンチマークのコンストラクタが受け付ける data キー
_BENCH_CLASSES = None


def _bench_class(name):
    from src.benchmarks import (
        RegressionSwitchBenchmark, GefcomBenchmark, GefcomPriceBenchmark,
        EmailBenchmark, InsectsBenchmark)
    return {
        "regression": RegressionSwitchBenchmark,
        "gefcom": GefcomBenchmark,
        "gefcom_price": GefcomPriceBenchmark,
        "email": EmailBenchmark,
        "insects": InsectsBenchmark,
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
    if name == "regression":
        if "eval_start" in ev:
            data.setdefault("eval_start", ev["eval_start"])
        if "select_start" in ev:
            data.setdefault("select_start", ev["select_start"])
        if "select_end" in ev:
            data.setdefault("select_end", ev["select_end"])
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
    # straddle 除外を先に適用してから空判定する(除外後に空になる設定も検出)。
    mask = mask & ~np.asarray(result.get("straddle_mask", np.zeros(n, bool)))
    if region in ("selection", "report") and not mask.any():
        raise ValueError(
            f"region '{region}' にサンプルがありません(straddle除外後に空)。"
            f"eval_start / report_start とストリーム長の設定を確認してください。")
    return mask


def masked_history(history, mask):
    """フィルタ履歴を time 軸マスクで切り出す(診断量を report 区間に限定, F2)。

    先頭軸長がマスク長と一致する配列だけをマスクし、それ以外(スカラー等)は
    そのまま返す。rho のような (T, N) 配列は (n_report, N) になる。
    """
    mask = np.asarray(mask, dtype=bool)
    out = {}
    for key, value in history.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == mask.size:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def primary_score(result, cfg, region="selection"):
    """1 実行結果からスカラー主要指標を返す(小さいほど良い方向に符号統一)。

    回帰: 平均 MSE(小さいほど良い) → そのまま
    分類: 平均 F1(大きいほど良い) → 符号反転して「小さいほど良い」に統一
    region: 'selection'(HP選択, 既定) / 'report'(最終評価) / 'all'

    φ_t 拡張 (設計書 §5.3): cfg["eval"]["select_metric"] = "nll" のとき、
    **選択区間に限り** 平均 NLL で選ぶ(σ_obs / τ_φ は MSE では識別できない
    ため)。既定 "mse" で完全後方互換。報告区間の集計は従来通り。
    """
    metrics = result["metrics"]
    mask = region_mask(result, region)
    if cfg["task_type"] == "regression":
        select_metric = cfg.get("eval", {}).get("select_metric", "mse")
        key = "nll" if (region == "selection"
                        and select_metric == "nll") else "mse"
        v = np.asarray(metrics[key])[mask]
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
    # φ_t 拡張 (設計書 §5.3): grid に sigma_obs があれば、固定 σ 手法
    # (非 -N)の探索軸に加える(真値注入の廃止)。無ければ従来通り
    # 軸なし(= benchmark の obs_sigma を使用)。
    sigma_obs_axis = grid.get("sigma_obs", [None])
    tau_phi_axis = grid.get("tau_phi", [0.05])

    def _with_sigma_obs(base, so):
        if so is not None:
            base["sigma_obs"] = so
        return base

    if method == "SGD":
        for eta, ps, so in itertools.product(
                etas, grid["prior_std"], sigma_obs_axis):
            yield _with_sigma_obs({"eta": eta, "prior_std": ps}, so)
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
        for eta, ss, ps, beta, so in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"], grid["beta"],
                sigma_obs_axis):
            yield _with_sigma_obs(
                {"eta": eta, "sigma_sys": ss, "prior_std": ps, "beta": beta},
                so)
    elif method == "WSPF-A-N":
        # -N 変種: sigma_obs 軸は持たず (φ が σ を推定)、tau_phi 軸を持つ
        for eta, ss, ps, beta, tp in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"], grid["beta"],
                tau_phi_axis):
            yield {"eta": eta, "sigma_sys": ss, "prior_std": ps,
                   "beta": beta, "tau_phi": tp}
    elif method in ("PF-N", "WSPF-B-N"):
        for eta, ss, ps, tp in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"], tau_phi_axis):
            yield {"eta": eta, "sigma_sys": ss, "prior_std": ps,
                   "tau_phi": tp}
    else:  # PF, WSPF-B, Oracle
        for eta, ss, ps, so in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"], sigma_obs_axis):
            yield _with_sigma_obs(
                {"eta": eta, "sigma_sys": ss, "prior_std": ps}, so)


def _is_boundary(best, grid):
    """best の各パラメータがグリッド端点にあるか(自動拡張の警告用, 修正方針11)。"""
    hits = []
    for key, gkey in (("eta", "eta"), ("sigma_sys", "sigma_sys"),
                      ("prior_std", "prior_std"), ("beta", "beta"),
                      ("sigma_obs", "sigma_obs"), ("tau_phi", "tau_phi")):
        if key in best and gkey in grid:
            g = grid[gkey]
            if best[key] in (min(g), max(g)):
                hits.append(key)
    return hits


def _grid_eval_job(args):
    """グリッド 1 ジョブ((params, ctx, seed) の 1 評価)。プロセス並列で実行。

    モジュールレベル関数なので ProcessPoolExecutor でピクル可能。
    戻り値は選択区間のスカラースコアのみ(重い result dict はプロセス間を
    渡らない)。
    """
    method, cfg, ctx, n_particles, params, seed = args
    bench = build_benchmark(cfg, **ctx)
    r = run_method(method, bench, n_particles, params, seed=seed,
                   collect_diagnostics=False)
    return primary_score(r, cfg, region="selection")


def grid_search(method, cfg, n_particles, selection_seeds, emit=print,
                contexts=None, n_workers=None):
    """選択区間スコアで最良パラメータを返す。端点採択時は警告する。

    contexts: ベンチマーク構築の override 辞書のリスト(例: GEFCom の
    [{'zone':1,'noise_std':..}, {'zone':2,..}, ..])。複数指定すると
    その **全コンテキスト×選択シードの平均** 選択区間スコアで選ぶ
    (GEFCom の 3zone 平均による共通 HP 選択, C3)。既定は単一 [{}]。

    (候補 × context × seed)を 1 ジョブとして **プロセス並列** で評価する
    (旧版と同方針)。並列数は n_workers か環境変数 WSPF_NUM_WORKERS / NCPUS。
    候補ごとに全ジョブ平均で選ぶため、並列でも結果は逐次と一致する。
    """
    grid = cfg["grid"]
    contexts = contexts or [{}]
    candidates = list(_param_grid(method, grid))

    # ジョブ列を平坦化: (candidate_index, job_args)
    jobs = []
    for ci, params in enumerate(candidates):
        for ctx in contexts:
            for s in selection_seeds:
                jobs.append((ci, (method, cfg, ctx, n_particles, params, s)))

    workers = resolve_workers(len(jobs), n_workers)
    scores_by_cand = defaultdict(list)
    if workers <= 1:
        for ci, arg in jobs:
            scores_by_cand[ci].append(_grid_eval_job(arg))
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker) as ex:
                for (ci, _), sc in zip(jobs,
                                       ex.map(_grid_eval_job,
                                              [a for _, a in jobs])):
                    scores_by_cand[ci].append(sc)
        except Exception as e:
            emit(f"  [警告] {method}: 並列グリッド失敗({e!r})→逐次実行")
            scores_by_cand = defaultdict(list)
            for ci, arg in jobs:
                scores_by_cand[ci].append(_grid_eval_job(arg))

    best, best_score = None, np.inf
    for ci, params in enumerate(candidates):
        sc = float(np.nanmean(scores_by_cand[ci]))
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
    if method in ("PF", "WSPF-A", "WSPF-B",
                  "PF-N", "WSPF-A-N", "WSPF-B-N"):
        return selected["by_n_particles"][str(n_particles)][method]
    return selected["no_n"][method]


def benchmark_contexts(cfg, selected):
    """ベンチマーク構築の override 辞書リストを返す(補助実験の共通化)。

    GEFCom-Solar は全 zone をループし、grid_search が保存した zone 別観測ノイズ
    (selected["gefcom_noise"])を必ず使う。GEFCom-Price は単一ゾーンなので
    単一コンテキスト [{"noise_std": selected["gefcom_price_noise"]}] を返す。
    いずれも未保存なら再現性のため例外にする(デフォルト値へ黙ってフォールバック
    しない, M1)。他ベンチマークは [{}]。GEFCom-Solar のコンテキストには識別用に
    "zone" が入る(出力行の zone 列に使う)。
    """
    bench = cfg["benchmark"]
    if bench == "gefcom_price":
        noise = selected.get("gefcom_price_noise")
        if noise is None:
            raise KeyError(
                "gefcom_price_noise が selected_params にありません。"
                "先に scripts/grid_search.py --benchmark gefcom_price を"
                "実行してください。")
        return [{"noise_std": float(noise)}]
    if bench != "gefcom":
        return [{}]
    noise = selected.get("gefcom_noise")
    if not noise:
        raise KeyError(
            "gefcom_noise が selected_params にありません。"
            "先に scripts/grid_search.py --benchmark gefcom を実行してください。")
    zones = cfg.get("data", {}).get("zones", [1])
    ctxs = []
    for z in zones:
        if str(z) not in noise:
            raise KeyError(f"GEFCom zone {z} の noise_std が selected_params に"
                           f"ありません。grid_search を再実行してください。")
        ctxs.append({"zone": z, "noise_std": noise[str(z)]})
    return ctxs


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
