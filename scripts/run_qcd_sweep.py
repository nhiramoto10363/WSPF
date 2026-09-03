#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep over the concept-drift noise sigma_cd.

At N=100, every other shared parameter is held fixed and only sigma_cd
varies. The output records the primary metric, ESS/N, the resampling rate,
quantiles of rho with P(rho > 0.9) and P(rho > 0.99), the number of clipped
corrections and the number of non-finite ones.

Usage:
    python scripts/run_qcd_sweep.py --benchmark regression \
        --sigma-cd 0.01 0.05 0.1 0.15 0.2
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask, benchmark_contexts,
                     masked_history)
from src.evaluation import (run_seeds, save_run_dir, mean_std,
                            summarize_history, rho_report)

# PF is included so that its performance, ESS and resampling rate are also
# reported against sigma_cd; having no correction, its rho and clip columns
# come out as None.
METHODS = ["PF", "WSPF-A", "WSPF-B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--sigma-cd", type=float, nargs="+", default=None,
                    help="sigma_cd values to sweep "
                         "(default: grid.sigma_sys from the config)")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n = cfg["n_particles"]["main"]
    sigmas = args.sigma_cd or cfg["grid"]["sigma_sys"]

    # The primary metric follows task_type, and the column names follow the
    # metric: mse_mean/mse_std for regression, f1_mean/f1_std otherwise.
    metric_key = "mse" if cfg["task_type"] == "regression" else "f1"
    mean_col, std_col = f"{metric_key}_mean", f"{metric_key}_std"

    contexts = benchmark_contexts(cfg, selected)   # single context, [{}]
    rows = []
    for ctx in contexts:
      zone = ctx.get("zone")
      for m in METHODS:
        base = get_params(selected, m, n)
        for sc in sigmas:
            params = dict(base)
            params["sigma_sys"] = sc          # vary sigma_cd only
            bench = build_benchmark(cfg, **ctx)
            results = run_seeds(m, bench, n, params, eval_seeds)
            (mses, ess, resamp, q50, q90v, q99v, rmax,
             p90, p99, clip, cliprate, nonf) = ([] for _ in range(12))
            for r in results:
                mask = region_mask(r, "report")
                mses.append(np.nanmean(np.asarray(r["metrics"][metric_key])[mask]))
                if r.get("history"):
                    # Restrict the diagnostics to the reporting window too.
                    hr = masked_history(r["history"], mask)
                    d = summarize_history(hr)
                    ess.append(d["pre_resample"].get("ess", np.nan) / n)
                    resamp.append(d["post_resample"].get("resample_rate", np.nan))
                    rr = rho_report(hr) or {}
                    q50.append(rr.get("q50", np.nan))
                    q90v.append(rr.get("q90", np.nan))
                    q99v.append(rr.get("q99", np.nan))
                    rmax.append(rr.get("max", np.nan))
                    p90.append(rr.get("p_gt_0.9", np.nan))
                    p99.append(rr.get("p_gt_0.99", np.nan))
                    cnt = rr.get("rho_clip_count", np.nan)
                    clip.append(cnt)
                    # Clip *rate* = count / (particles x reporting steps).
                    denom = n * max(int(mask.sum()), 1)
                    cliprate.append(cnt / denom if np.isfinite(cnt) else np.nan)
                    nonf.append(rr.get("logcorr_nonfinite_count", np.nan))
            mu, sd = mean_std(mses)

            def _mean(x):
                return float(np.nanmean(x)) if x and np.any(np.isfinite(x)) else None

            # The two are named apart: for WSPF-B the counter is an actual
            # clipping of the correction, while for WSPF-A it only records
            # rho reaching 0.999 as a diagnostic.
            if m == "WSPF-A":
                count_key, rate_key = ("rho_ge_0.999_count",
                                       "wspf_a_rho_threshold_exceedance_rate")
            else:  # WSPF-B; PF leaves these None
                count_key, rate_key = ("clip_count", "wspf_b_actual_clip_rate")
            row = {
                "method": m, "zone": zone, "sigma_cd": sc,
                mean_col: mu, std_col: sd,
                "ess_over_N": _mean(ess), "resample_rate": _mean(resamp),
                "rho_q50": _mean(q50), "rho_q90": _mean(q90v),
                "rho_q99": _mean(q99v), "rho_max": _mean(rmax),
                "P_rho_gt_0.9": _mean(p90), "P_rho_gt_0.99": _mean(p99),
                count_key: _mean(clip), rate_key: _mean(cliprate),
                "nonfinite": _mean(nonf),
            }
            rows.append(row)
            ztag = f"[zone {zone}] " if zone is not None else ""
            print(f"{ztag}{m:7s} σ_cd={sc:<6} {metric_key.upper()}={mu:.4f}±{sd:.4f} "
                  f"ESS/N={row['ess_over_N'] if row['ess_over_N'] is None else round(row['ess_over_N'],2)}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "qcd_sweep")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
