#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主結果の実行 (N=main, 評価シード全体)

各手法を、選択済みハイパーパラメータ・評価シード(10個)で実行し、
指標・診断量・実行成果物を outputs/<benchmark>/ に保存する。

  粒子フィルタ (PF/WSPF-A/WSPF-B) : N=main の最良 HP
  点推定 (SGD/PH-SGD/Window-SGD)  : 選択済み HP
  Oracle                          : Regression のみ(run_oracle.py 参照)

使い方:
    python scripts/run_main.py --benchmark gefcom
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params)
from src.evaluation import run_seeds, save_run_dir, mean_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n_main = cfg["n_particles"]["main"]
    methods = [m for m in cfg["methods"] if m != "NoChange"]

    rows = []
    diagnostics = {}
    for m in methods:
        params = get_params(selected, m, n_main)
        bench = build_benchmark(cfg)
        results = run_seeds(m, bench, n_main, params, eval_seeds)
        # 評価シード横断で主要指標を集計(straddle 除外)
        key = "mse" if cfg["task_type"] == "regression" else "f1"
        vals = []
        for r in results:
            mask = ~r.get("straddle_mask",
                          np.zeros_like(r["metrics"][key], bool))
            vals.append(np.nanmean(np.asarray(r["metrics"][key])[mask]))
        mu, sd = mean_std(vals)
        rows.append({"method": m, "N": n_main, "metric": key,
                     "mean": mu, "std": sd, "n_seeds": len(eval_seeds)})
        diagnostics[m] = {"per_seed": vals}
        print(f"{m:12s} {key}={mu:.4f} ± {sd:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "main")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics=diagnostics)
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
