#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q_cd (σ_cd) スイープ (R1-8)

Regression + GEFCom, N=100。他の共有パラメータは固定して σ_cd のみ変える。
保存: MSE, ESS/N, リサンプリング頻度, ρ の分位点・P(ρ>0.9)・P(ρ>0.99),
      クリッピング回数, 非有限補正回数。

使い方:
    python scripts/run_qcd_sweep.py --benchmark regression \
        --sigma-cd 0.01 0.05 0.1 0.15 0.2
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params)
from src.evaluation import (run_seeds, save_run_dir, mean_std,
                            summarize_history, rho_report)

METHODS = ["WSPF-A", "WSPF-B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--sigma-cd", type=float, nargs="+", default=None,
                    help="スイープする σ_cd 候補(既定: config grid.sigma_sys)")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n = cfg["n_particles"]["main"]
    sigmas = args.sigma_cd or cfg["grid"]["sigma_sys"]

    rows = []
    for m in METHODS:
        base = get_params(selected, m, n)
        for sc in sigmas:
            params = dict(base)
            params["sigma_sys"] = sc          # σ_cd のみ変える
            bench = build_benchmark(cfg)
            results = run_seeds(m, bench, n, params, eval_seeds)
            mses, ess, resamp, rhoq, p90, p99, clip, nonf = ([] for _ in range(8))
            for r in results:
                mask = ~r.get("straddle_mask",
                              np.zeros_like(r["metrics"]["mse"], bool))
                mses.append(np.nanmean(np.asarray(r["metrics"]["mse"])[mask]))
                if r.get("history"):
                    d = summarize_history(r["history"])
                    ess.append(d["pre_resample"].get("ess", np.nan) / n)
                    resamp.append(d["post_resample"].get("resample_rate", np.nan))
                    rr = rho_report(r["history"]) or {}
                    p90.append(rr.get("p_gt_0.9", np.nan))
                    p99.append(rr.get("p_gt_0.99", np.nan))
                    clip.append(rr.get("rho_clip_count", np.nan))
                    nonf.append(rr.get("logcorr_nonfinite_count", np.nan))
            mu, sd = mean_std(mses)
            rows.append({
                "method": m, "sigma_cd": sc, "mse_mean": mu, "mse_std": sd,
                "ess_over_N": float(np.nanmean(ess)) if ess else None,
                "resample_rate": float(np.nanmean(resamp)) if resamp else None,
                "P_rho_gt_0.9": float(np.nanmean(p90)) if p90 else None,
                "P_rho_gt_0.99": float(np.nanmean(p99)) if p99 else None,
                "clip_count": float(np.nanmean(clip)) if clip else None,
                "nonfinite": float(np.nanmean(nonf)) if nonf else None,
            })
            print(f"{m:7s} σ_cd={sc:<6} MSE={mu:.4f}±{sd:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "qcd_sweep")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
