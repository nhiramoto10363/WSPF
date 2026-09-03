#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictive performance against the mini-batch size B.

The counterpart of the gradient-noise analysis: B varies over
[8, 16, 32, 64] for PF, WSPF-A and WSPF-B.

The hyper-parameters are held fixed at the values selected for N=main rather
than re-tuned per B, so that the effect of B can be read without the
confound of re-optimization.

Reported MSE is aggregated over the reporting window (straddling blocks
excluded) across the evaluation seeds, as mean and std.

Usage (run grid_search.py first, to produce selected_params.json):
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
    """Mean MSE over the reporting window of one run."""
    mse = np.asarray(result["metrics"]["mse"])
    mask = region_mask(result, "report")
    return float(np.nanmean(mse[mask]))


def sweep(cfg):
    """One row of (mean, std) reporting-window MSE per (method, B)."""
    selected = load_selected(cfg)
    n_main = cfg["n_particles"]["main"]
    eval_seeds = resolve_seeds(cfg, "evaluation")

    rows = []
    for method in METHODS:
        # Fixed hyper-parameters, shared across every B.
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
                    help="regression / email / insects, or a config path")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    rows = sweep(cfg)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "batch_size_sweep")
    os.makedirs(out_dir, exist_ok=True)
    write_table(rows, os.path.join(out_dir, "metrics"))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
