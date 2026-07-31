#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バッチサイズ別 手法性能スイープ (R1-6)

査読対応(R1-6): 勾配ノイズ解析(analyze_gradient_noise.py)に対応する形で、
ミニバッチサイズ B を変えたときの各手法の予測性能を測る。

  batch_sizes = [8, 16, 32, 64]
  methods     = [PF, WSPF-A, WSPF-B]

HP は **固定**(N=main の選択済み HP をそのまま流用し、B ごとには再チューニング
しない)。これは B の効果を HP 再最適化の交絡なしで解釈するための設計判断で、
査読で要求された「バッチサイズに対する手法性能」を素直に読むためである。

報告区間(region='report', straddle 除外)の MSE を評価シードで集計し、
mean/std を出力する。

使い方(先に grid_search.py で selected_params.json を作成しておく):
    python scripts/grid_search.py --benchmark regression
    python scripts/run_batch_size_sweep.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, load_selected, get_params, resolve_seeds,
                     build_benchmark, region_mask)
from src.evaluation import run_seeds, write_table, mean_std

BATCH_SIZES = [8, 16, 32, 64]
METHODS = ["PF", "WSPF-A", "WSPF-B"]


def _report_mse(result):
    """1 実行結果から報告区間の平均 MSE を返す。"""
    mse = np.asarray(result["metrics"]["mse"])
    mask = region_mask(result, "report")
    return float(np.nanmean(mse[mask]))


def sweep(cfg):
    """(method, B) ごとの報告区間 MSE (mean, std) 行のリストを返す。"""
    selected = load_selected(cfg)
    n_main = cfg["n_particles"]["main"]
    eval_seeds = resolve_seeds(cfg, "evaluation")

    rows = []
    for method in METHODS:
        # 固定 HP: N=main の選択済みパラメータを全 B で共用(B 別再チューニング無し)
        params = get_params(selected, method, n_main)
        for B in BATCH_SIZES:
            bench = build_benchmark(cfg, batch_size=B)
            results = run_seeds(method, bench, n_main, params, eval_seeds,
                                collect_diagnostics=False)
            mses = [_report_mse(r) for r in results]
            m, s = mean_std(mses)
            rows.append({"method": method, "batch_size": B,
                         "mse_mean": m, "mse_std": s})
            print(f"{method:8s} B={B:3d} MSE={m:.5f} ± {s:.5f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="regression",
                    help="regression / gefcom / email または config パス")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    rows = sweep(cfg)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "batch_size_sweep")
    os.makedirs(out_dir, exist_ok=True)
    write_table(rows, os.path.join(out_dir, "metrics"))
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
