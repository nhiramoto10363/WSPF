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
                     load_selected, get_params, region_mask)
from src.evaluation import (run_seeds, save_run_dir, mean_std,
                            summarize_history, rho_report)

# PF も含める(R1-8: PF の性能・ESS・リサンプリング頻度を σ_cd 依存で報告)。
# PF は補正/ρ を持たないため ρ 系・clip 系の列は None になる。
METHODS = ["PF", "WSPF-A", "WSPF-B"]


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
            (mses, ess, resamp, q50, q90v, q99v, rmax,
             p90, p99, clip, nonf) = ([] for _ in range(11))
            for r in results:
                mask = region_mask(r, "report")
                mses.append(np.nanmean(np.asarray(r["metrics"]["mse"])[mask]))
                if r.get("history"):
                    d = summarize_history(r["history"])
                    ess.append(d["pre_resample"].get("ess", np.nan) / n)
                    resamp.append(d["post_resample"].get("resample_rate", np.nan))
                    rr = rho_report(r["history"]) or {}
                    q50.append(rr.get("q50", np.nan))
                    q90v.append(rr.get("q90", np.nan))
                    q99v.append(rr.get("q99", np.nan))
                    rmax.append(rr.get("max", np.nan))
                    p90.append(rr.get("p_gt_0.9", np.nan))
                    p99.append(rr.get("p_gt_0.99", np.nan))
                    clip.append(rr.get("rho_clip_count", np.nan))
                    nonf.append(rr.get("logcorr_nonfinite_count", np.nan))
            mu, sd = mean_std(mses)

            def _mean(x):
                return float(np.nanmean(x)) if x and np.any(np.isfinite(x)) else None

            # clip 命名の区別(R1-8): WSPF-B は補正 clipping の作動回数、
            # WSPF-A は診断上の ρ≥0.999 到達回数(実際の補正 clip ではない)。
            clip_key = "rho_ge_0.999_count" if m == "WSPF-A" else "rho_clip_count"
            row = {
                "method": m, "sigma_cd": sc, "mse_mean": mu, "mse_std": sd,
                "ess_over_N": _mean(ess), "resample_rate": _mean(resamp),
                "rho_q50": _mean(q50), "rho_q90": _mean(q90v),
                "rho_q99": _mean(q99v), "rho_max": _mean(rmax),
                "P_rho_gt_0.9": _mean(p90), "P_rho_gt_0.99": _mean(p99),
                clip_key: _mean(clip), "nonfinite": _mean(nonf),
            }
            rows.append(row)
            print(f"{m:7s} σ_cd={sc:<6} MSE={mu:.4f}±{sd:.4f} "
                  f"ESS/N={row['ess_over_N'] if row['ess_over_N'] is None else round(row['ess_over_N'],2)}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "qcd_sweep")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
