#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Oracle experiment (regression only).

At N=100, under matched settings and common random numbers, the exact
correction (oracle), the Method A approximation, the Method B approximation
and the uncorrected PF are compared under identical conditions. That
separates the validity of the Gaussian correction itself from the error
introduced by approximating it.

Common random numbers: all four methods share the data stream (same data
seed) and the same initial particles and drift-noise sequence (a shared
filter_seed). Resampling can still occur at different times for different
methods, so this is not perfect CRN, but sharing the initialization and the
noise sequence sharpens the comparison considerably.

Gradient clipping is disabled here (grad_clip_norm=None), because clipping
the mean gradient while the population statistics remain unclipped would be
inconsistent with the exact Gaussian correction. The shared hyper-parameters
are those of WSPF-A, so the WSPF-A-to-oracle gap measures the cost of the
approximation.

Usage:
    python scripts/run_oracle.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask)
from src.evaluation import (run_seeds, save_run_dir, mean_std,
                            all_pairs_compare)

METHODS = ["PF", "Oracle", "WSPF-A", "WSPF-B"]
SHARED_KEYS = ("eta", "sigma_sys", "prior_std")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="regression")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["task_type"] != "regression" or not cfg.get("oracle"):
        raise SystemExit(
            "the oracle applies only to regression benchmarks with "
            "oracle: true")

    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n = cfg["n_particles"]["main"]

    # Shared hyper-parameters: those of WSPF-A.
    a = get_params(selected, "WSPF-A", n)
    shared = {k: a[k] for k in SHARED_KEYS if k in a}

    rows, per_seed = [], {}
    for m in METHODS:
        params = dict(shared)
        if m == "WSPF-A":
            params["beta"] = a.get("beta", 0.9)
        # Clipping off, to test the exact correction.
        bench = build_benchmark(cfg, grad_clip_norm=None)
        # Common random numbers: one filter_seed shared by every method.
        # Seeds run in parallel processes, which matters here because the
        # oracle statistics are Monte Carlo and expensive.
        fseeds = [s + 1 for s in eval_seeds]
        results = run_seeds(m, bench, n, params, eval_seeds, filter_seeds=fseeds)
        vals = [np.nanmean(np.asarray(r["metrics"]["mse"])[region_mask(r, "report")])
                for r in results]
        per_seed[m] = vals
        mu, sd = mean_std(vals)
        rows.append({"method": m, "metric": "mse", "mean": mu, "std": sd})
        print(f"{m:8s} MSE={mu:.4f}±{sd:.4f}")

    # Paired t-test and Wilcoxon for every pair in the mechanism table, with
    # Holm-adjusted p-values in their own columns.
    pairs = all_pairs_compare(per_seed, methods=METHODS, lower_is_better=True)
    print(f"--- paired tests (mse, {len(pairs)} pairs, Holm-adjusted) ---")
    for pr in pairs:
        rows.append({"method": f"{pr['a']}_vs_{pr['b']}", "metric": "mse_paired",
                     "mean_difference": pr["mean_diff"],
                     "std_difference": pr["std_diff"],
                     "paired_t_p": pr["paired_t_p"],
                     "wilcoxon_p": pr["wilcoxon_p"],
                     "paired_t_p_holm": pr["paired_t_p_holm"],
                     "wilcoxon_p_holm": pr["wilcoxon_p_holm"],
                     "better": pr["better"], "n_seeds": pr["n_seeds"]})
        print(f"  {pr['a']:8s}−{pr['b']:8s} Δ={pr['mean_diff']:+.4f}  "
              f"t-p={pr['paired_t_p']:.3g} (Holm {pr['paired_t_p_holm']:.3g})  "
              f"W-p={pr['wilcoxon_p']:.3g} (Holm {pr['wilcoxon_p_holm']:.3g})")
    gap = float(np.mean(np.array(per_seed["WSPF-A"]) - np.array(per_seed["Oracle"])))
    print(f"WSPF-A minus Oracle (cost of the approximation) = {gap:.4f}")
    rows.append({"method": "WSPF-A_minus_Oracle", "metric": "gap",
                 "mean": gap, "std": None})

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "oracle")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={"per_seed": per_seed})
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
