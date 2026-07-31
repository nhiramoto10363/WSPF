#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N スイープ (R2-2, 有限粒子挙動)

主分析: 各手法とも **N=100(main) で選択した HP を固定** し、
        N ∈ {25,50,100,200,400} を実行する。
        (N ごとの再チューニングはしない。それは補足分析として Supplement へ)

指標(MSE/F1)に加えて診断量(ESS/重み縮退/リサンプリング頻度)と
実行時間も集計できるよう、履歴を保存する。

使い方:
    python scripts/run_n_sweep.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params)
from src.evaluation import (run_seeds, save_run_dir, mean_std,
                            summarize_history, timing_report)

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--tuned-per-n", action="store_true",
                    help="各 N でチューニングした HP を使う(Supplement)。"
                         "既定は N=main の HP を固定(R2-2 主分析)。"
                         "先に grid_search.py --tune-all-n が必要。")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n_main = cfg["n_particles"]["main"]
    n_sweep = cfg["n_particles"]["sweep"]
    key = "mse" if cfg["task_type"] == "regression" else "f1"

    rows = []
    for m in FILTER_METHODS:
        # 主分析: N=main の HP を固定して全 N で使う
        params_fixed = get_params(selected, m, n_main)
        for n in n_sweep:
            # Supplement モードでは各 N のチューニング済み HP を使う
            params = get_params(selected, m, n) if args.tuned_per_n else params_fixed
            bench = build_benchmark(cfg)
            results = run_seeds(m, bench, n, params, eval_seeds)
            vals, ess_frac, resamp = [], [], []
            for r in results:
                mask = ~r.get("straddle_mask",
                              np.zeros_like(r["metrics"][key], bool))
                vals.append(np.nanmean(np.asarray(r["metrics"][key])[mask]))
                if r.get("history"):
                    d = summarize_history(r["history"])
                    ess_frac.append(d["pre_resample"].get("ess", np.nan) / n)
                    resamp.append(d["post_resample"].get("resample_rate", np.nan))
            mu, sd = mean_std(vals)
            rows.append({"method": m, "N": n, "metric": key,
                         "mean": mu, "std": sd,
                         "ess_over_N": float(np.nanmean(ess_frac)) if ess_frac else None,
                         "resample_rate": float(np.nanmean(resamp)) if resamp else None})
            print(f"{m:8s} N={n:4d}  {key}={mu:.4f}±{sd:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "n_sweep")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}  (N別再チューニングは Supplement)")


if __name__ == "__main__":
    main()
