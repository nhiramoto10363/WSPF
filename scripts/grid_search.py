#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グリッドサーチ (選択区間のみ, R1-13 リーク除去)

R2-2 の方針:
  主分析は N=100(main) の HP を固定して N をスイープする(N別に再チューニング
  しない)。したがって既定では **N=main のみ** で探索する。
  N別最適化は Supplement 用の補足分析なので、必要なときだけ --tune-all-n で
  全 N を探索する(その結果は run_n_sweep.py の tuned-per-N モードで利用)。

  for method in [PF, WSPF-A, WSPF-B]:
      N=main (既定) または全 N (--tune-all-n) で選択シード平均の最良を選ぶ
  for method in [SGD, PH-SGD, Window-SGD (, Oracle*)]:
      N=main で最良パラメータを選ぶ            (*Oracle は Regression のみ)

結果は outputs/<benchmark>/selected_params.json に保存する。
最適点がグリッド端に達したら警告する(端点自動拡張規則, 修正方針11)。

使い方:
    python scripts/grid_search.py --benchmark regression
    python scripts/grid_search.py --benchmark regression --tune-all-n   # Supplement
"""

import argparse
import json
import os

from _common import (load_config, resolve_seeds, grid_search, hp_path,
                     estimate_obs_noise)

# φ_t 拡張: -N (noise-adaptive) 変種を含む。実際に探索するのは
# config の methods に載っているものだけ (下の _filter_methods)。
FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B", "PF-N", "WSPF-A-N", "WSPF-B-N"]


def _filter_methods(cfg):
    """config の methods にある粒子フィルタ手法を FILTER_METHODS 順で返す。

    後方互換: 旧 config (methods 未指定の呼び出し経路は無い) でも
    [PF, WSPF-A, WSPF-B] がそのまま選ばれる。
    """
    wanted = set(cfg.get("methods", []))
    ms = [m for m in FILTER_METHODS if m in wanted]
    return ms if ms else ["PF", "WSPF-A", "WSPF-B"]


def _gefcom_contexts(cfg):
    """GEFCom の zone 別コンテキストと観測ノイズを構築する(C2/C3)。

    観測ノイズ σ_obs は各 zone の選択区間から **一度だけ** 推定して固定し、
    grid_search と run_main/calibration の両方で同じ値を使えるよう保存する。
    HP は全 zone の選択区間平均で共通選択する(共通 HP)。
    """
    zones = cfg.get("data", {}).get("zones", [1])
    noise_by_zone = {}
    contexts = []
    for z in zones:
        sigma = estimate_obs_noise(cfg, zone=z)
        noise_by_zone[str(z)] = sigma
        contexts.append({"zone": z, "noise_std": sigma})
        print(f"  [zone {z}] 選択区間から推定した σ_obs = {sigma:.4f}")
    return contexts, noise_by_zone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / gefcom / email または config パス")
    ap.add_argument("--tune-all-n", action="store_true",
                    help="全 N で個別に探索(Supplement 用の N別最適化)。"
                         "既定は N=main のみ(R2-2 主分析)。")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    sel = resolve_seeds(cfg, "selection")
    n_main = cfg["n_particles"]["main"]
    n_sweep = cfg["n_particles"]["sweep"]

    result = {"by_n_particles": {}, "no_n": {}}

    # GEFCom: zone 別に σ_obs を推定・保存し、全 zone 平均で共通 HP を選ぶ(C2/C3)
    contexts = None
    if cfg["benchmark"] == "gefcom":
        print("[gefcom: 選択区間から σ_obs を推定]")
        contexts, noise_by_zone = _gefcom_contexts(cfg)
        result["gefcom_noise"] = noise_by_zone
    # GEFCom-Price: 単一ゾーン。選択区間から σ_obs を推定・保存する
    # (estimate_obs_noise は zone=None でそのまま動く)。
    elif cfg["benchmark"] == "gefcom_price":
        print("[gefcom_price: 選択区間から σ_obs を推定]")
        sigma = estimate_obs_noise(cfg)
        print(f"  推定した σ_obs = {sigma:.4f}")
        contexts = [{"noise_std": sigma}]
        result["gefcom_price_noise"] = sigma

    # 粒子フィルタ: 既定は N=main のみ(主分析)。--tune-all-n で全 N(Supplement)。
    target_ns = n_sweep if args.tune_all_n else [n_main]
    fmethods = _filter_methods(cfg)
    for n in target_ns:
        print(f"[N={n}]")
        result["by_n_particles"][str(n)] = {}
        for m in fmethods:
            best, score = grid_search(m, cfg, n, sel, contexts=contexts)
            result["by_n_particles"][str(n)][m] = best

    # 点推定ベースライン(N を持たない): N=main で選択。
    # Oracle は独立グリッドサーチ不要(run_oracle.py が WSPF-A の HP を共有する)。
    base_methods = [m for m in cfg["methods"]
                    if m in ("SGD", "PH-SGD", "Window-SGD")]
    print("[baselines (no N)]")
    for m in base_methods:
        best, score = grid_search(m, cfg, n_main, sel, contexts=contexts)
        result["no_n"][m] = best

    out = hp_path(cfg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
