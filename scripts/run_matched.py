#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched-hyper-parameter comparison.

N is fixed at 100 and only eta, sigma_cd and sigma_0 are shared; the beta of
WSPF-A stays at its own selected value rather than being taken from another
method. Each of the three filters supplies the shared setting in turn, giving
a 3x3 table.

Usage:
    python scripts/run_matched.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask, benchmark_contexts)
from src.evaluation import run_seeds, save_run_dir, mean_std

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]
SHARED_KEYS = ("eta", "sigma_sys", "prior_std")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n = cfg["n_particles"]["main"]
    key = "mse" if cfg["task_type"] == "regression" else "f1"

    contexts = benchmark_contexts(cfg, selected)   # single context, [{}]
    rows = []
    for ctx in contexts:
        zone = ctx.get("zone")
        for fixed in FILTER_METHODS:
            base = get_params(selected, fixed, n)
            shared = {k: base[k] for k in SHARED_KEYS if k in base}
            for m in FILTER_METHODS:
                params = dict(shared)
                if m == "WSPF-A":  # beta stays at its own selected value
                    params["beta"] = get_params(selected, "WSPF-A", n).get("beta", 0.9)
                bench = build_benchmark(cfg, **ctx)
                results = run_seeds(m, bench, n, params, eval_seeds)
                vals = [np.nanmean(np.asarray(r["metrics"][key])[
                            region_mask(r, "report")]) for r in results]
                mu, sd = mean_std(vals)
                rows.append({"shared_from": fixed, "method": m, "zone": zone,
                             "metric": key, "mean": mu, "std": sd})
                ztag = f"[zone {zone}] " if zone is not None else ""
                print(f"{ztag}shared={fixed:7s} → {m:7s}  {key}={mu:.4f}±{sd:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "matched_hp")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"saved: {out_dir}  (3x3 table)")


if __name__ == "__main__":
    main()
