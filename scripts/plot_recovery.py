#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-switch recovery curves.

Aggregation follows src.evaluation.recovery_curve: within each seed the
metric at lag tau is averaged across switch points, then averaged across
seeds, with a band of +-1 SE (or SD).

Scope. This script takes the switch points from the run's switch_mask, i.e.
the straddling blocks. That is correct for regression, where switches align
with the step grid and all of them fall inside the reporting window, but
*wrong for the classification benchmarks*:

  - the email drift points are sample indices [300, 600, 900, 1200] with a
    block width of 16, and sample 1200 lands exactly on a block boundary, so
    it never straddles and would be missed;
  - conversely sample 300 lies in the selection window (reporting starts at
    600) yet would be picked up, mixing selection data into a figure that
    reports final results.

Use scripts/plot_recovery_classification.py for those, which plots the curve
that calibration_report builds from the benchmark's switch_points.

  - regression: y is the test MSE, on a log scale.
  - insects: y is the test error, 1 - accuracy, on a linear scale, since an
    error rate does not span orders of magnitude. --metric switches to the
    macro-F1 error.

Usage:
    python3 scripts/plot_recovery.py --benchmark regression
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.evaluation import recovery_curve   # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directory prefix -> display name, in plotting order
METHOD_ORDER = [
    ("sgd", "SGD"),
    ("ph_sgd", "PH-SGD"),
    ("window_sgd", "Window-SGD"),
    ("pf", "PF"),
    ("wspf_a", "WSPF-A"),
    ("wspf_b", "WSPF-B"),
]
# Colours per method
COLORS = {
    "SGD": "#7f7f7f",         # gray
    "PH-SGD": "#ff7f0e",      # orange
    "Window-SGD": "#9467bd",  # purple
    "PF": "#1f77b4",          # blue
    "WSPF-A": "#2ca02c",      # green
    "WSPF-B": "#d62728",      # crimson
}


def _runs_for(benchmark, prefix):
    """All per-seed run directories for one method prefix."""
    base = os.path.join(_REPO_ROOT, "outputs", benchmark, "main", "runs")
    dirs = sorted(glob.glob(os.path.join(base, f"{prefix}_seed*")))
    # Run directories carrying a zone (name contains _zone) are out of scope.
    return [d for d in dirs if "_zone" not in os.path.basename(d)]


def _series_and_switches(run_dir, metric_key, invert):
    """Return (metric series, switch points) for one run; invert gives 1 - metric."""
    z = np.load(os.path.join(run_dir, "metrics.npz"))
    if metric_key not in z.files:
        raise KeyError(f"{metric_key} not in {run_dir} "
                       f"(available: {[k for k in z.files if k.startswith('metric_')]})")
    ts = np.asarray(z[metric_key], dtype=np.float64)
    if invert:
        ts = 1.0 - ts
    sw = np.where(np.asarray(z["switch_mask"], dtype=bool))[0].tolist()
    return ts, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / insects (anything with main/runs)")
    ap.add_argument("--metric", default=None,
                    help="regression: mse. classification: accuracy "
                         "(1 - acc, the default) or macro_f1")
    ap.add_argument("--max-lag", type=int, default=11,
                    help="number of lags (default 11, i.e. 0..10)")
    ap.add_argument("--logy", dest="logy", action="store_true", default=None)
    ap.add_argument("--no-logy", dest="logy", action="store_false")
    ap.add_argument("--band", choices=["sem", "sd"], default="sem",
                    help="band: sem = +-1 standard error (default), "
                         "sd = +-1 standard deviation")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    # task_type comes from the config
    cfg_path = os.path.join(_REPO_ROOT, "outputs", args.benchmark,
                            "main", "config.json")
    import json
    cfg = json.load(open(cfg_path))
    is_reg = cfg.get("task_type") == "regression"

    # Metric key and y-axis settings
    if is_reg:
        metric_key = "metric_" + (args.metric or "mse")
        invert = False
        ylabel = "Test MSE (log scale)" if (args.logy is not False) \
            else "Test MSE"
        logy = True if args.logy is None else args.logy
        err_name = "MSE"
    else:
        base_metric = args.metric or "accuracy"
        metric_key = "metric_" + base_metric
        invert = True                       # error rate = 1 - metric
        ylabel = f"Test error (1 − {base_metric})"
        logy = False if args.logy is None else args.logy
        err_name = f"1−{base_metric}"

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    switch_points_ref = None
    plotted = 0
    curve_lo, curve_hi = np.inf, -np.inf   # ylim follows the curves, not the band
    tau = np.arange(args.max_lag)
    for prefix, disp in METHOD_ORDER:
        runs = _runs_for(args.benchmark, prefix)
        if not runs:
            print(f"  [skip] {disp}: no runs found")
            continue
        mse_ts, switches = [], None
        for d in runs:
            ts, sw = _series_and_switches(d, metric_key, invert)
            mse_ts.append(ts)
            switches = sw                     # identical across seeds
        if switch_points_ref is None:
            switch_points_ref = switches
        rec = recovery_curve(mse_ts, switches, max_lag=args.max_lag)
        curve, sd = rec["curve"], rec["std"]
        n_seed = len(mse_ts)
        band = sd / np.sqrt(max(n_seed, 1)) if args.band == "sem" else sd
        c = COLORS.get(disp, None)
        ax.plot(tau, curve, "-o", color=c, label=disp, lw=2, ms=5, zorder=3)
        ax.fill_between(tau, curve - band, curve + band, color=c, alpha=0.18,
                        lw=0, zorder=1)
        curve_lo = min(curve_lo, float(np.nanmin(curve)))
        curve_hi = max(curve_hi, float(np.nanmax(curve)))
        plotted += 1
        print(f"  {disp}: τ0={curve[0]:.3f}  τ{args.max_lag-1}={curve[-1]:.3f}"
              f"  (n_seed={n_seed})")

    if plotted == 0:
        raise SystemExit("no runs to plot")

    # Fit ylim to the curves, so a wide band cannot flatten the axis.
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(curve_lo * 0.6, curve_hi * 1.6)
    else:
        pad = 0.06 * (curve_hi - curve_lo)
        ax.set_ylim(curve_lo - pad, curve_hi + pad)
    ax.set_xlabel(r"$\tau$ (steps after regime switch)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Post-Switch Recovery ({args.benchmark}, "
                 f"{'log ' if logy else ''}first {args.max_lag-1} steps)",
                 fontsize=14)
    ax.set_xlim(0, args.max_lag - 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=11, framealpha=0.95)
    n_sw = len(switch_points_ref) if switch_points_ref else 0
    band_lab = "±1 SE" if args.band == "sem" else "±1 SD"
    fig.text(0.99, 0.01,
             f"band = {band_lab} across seeds; averaged over {n_sw} switches, "
             f"10 seeds; metric={err_name}",
             ha="right", va="bottom", fontsize=8, color="#666")
    fig.tight_layout()

    out = args.outfile or os.path.join(
        _REPO_ROOT, "outputs", args.benchmark, "figures",
        "recovery_post_switch.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}  (switches={switch_points_ref})")


if __name__ == "__main__":
    main()
