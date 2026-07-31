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

from _common import (load_config, resolve_seeds, grid_search, hp_path)

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]


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

    # 粒子フィルタ: 既定は N=main のみ(主分析)。--tune-all-n で全 N(Supplement)。
    target_ns = n_sweep if args.tune_all_n else [n_main]
    for n in target_ns:
        print(f"[N={n}]")
        result["by_n_particles"][str(n)] = {}
        for m in FILTER_METHODS:
            best, score = grid_search(m, cfg, n, sel)
            result["by_n_particles"][str(n)][m] = best

    # 点推定ベースライン(N を持たない): N=main で選択
    base_methods = [m for m in cfg["methods"]
                    if m in ("SGD", "PH-SGD", "Window-SGD")]
    if cfg.get("oracle"):
        base_methods = base_methods + ["Oracle"]
    print("[baselines (no N)]")
    for m in base_methods:
        best, score = grid_search(m, cfg, n_main, sel)
        result["no_n"][m] = best

    out = hp_path(cfg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
