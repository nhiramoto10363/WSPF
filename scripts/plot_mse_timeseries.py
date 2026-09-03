#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test MSE over time on the regime-switching regression benchmark.

Reads outputs/regression/main/runs/*/metrics.npz only, so the figure is
guaranteed to agree with the numbers in the tables.

Usage:
    python3 scripts/plot_mse_timeseries.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Colours per method.
STYLES = {
    "SGD":        dict(color="#888888", marker="x", ls="-",  label="SGD"),
    "PH-SGD":     dict(color="#B8860B", marker="v", ls="-",  label="PH-SGD"),
    "Window-SGD": dict(color="#6A3D9A", marker="s", ls="-",  label="Window-SGD"),
    "PF":         dict(color="#2196F3", marker="o", ls="-",  label="PF (uncorrected)"),
    "WSPF-A":     dict(color="#4CAF50", marker="D", ls="-",  label="WSPF-A"),
    "WSPF-B":     dict(color="#E91E63", marker="^", ls="-",  label="WSPF-B"),
}
ORDER = ["SGD", "PH-SGD", "Window-SGD", "PF", "WSPF-A", "WSPF-B"]

# Run directories are named after the method, lower-cased with hyphens
# replaced by underscores (WSPF-A -> wspf_a).
_RUN_PREFIX = {m: m.lower().replace("-", "_") for m in
               ("SGD", "PH-SGD", "Window-SGD", "PF", "WSPF-A", "WSPF-B")}


def load_mse_timeseries(bench="regression"):
    """Return {method: seed-averaged MSE series} and the switch points."""
    root = os.path.join(_REPO_ROOT, "outputs", bench, "main", "runs")
    out = {}
    for m in ORDER:
        pat = os.path.join(root, f"{_RUN_PREFIX[m]}_seed*", "metrics.npz")
        paths = sorted(glob.glob(pat))
        if not paths:
            print(f"[note] {m}: no run matches {pat}; skipping")
            continue
        curves = []
        for p in paths:
            d = np.load(p, allow_pickle=True)
            curves.append(np.asarray(d["metric_mse"], dtype=float))
        out[m] = np.nanmean(np.vstack(curves), axis=0)
        print(f"  {m:11s} {len(paths)} seeds, T={out[m].size}")

    cfg = os.path.join(_REPO_ROOT, "outputs", bench, "main", "config.json")
    with open(cfg, encoding="utf-8") as f:
        switches = json.load(f)["data"].get("switch_points", [])
    return out, switches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default=os.path.join(
        _REPO_ROOT, "outputs", "figures",
        "regression_regime_switch_timeseries_N100.png"))
    ap.add_argument("--interval", type=int, default=25,
                    help="bin width for the plotted average (default 25)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    series, switches = load_mse_timeseries()
    if not series:
        raise SystemExit("no MSE series found; run run_main.py first")

    T = len(next(iter(series.values())))
    w = args.interval

    # Average within bins rather than subsampling. The switch points fall
    # exactly on the subsampling grid and recovery takes 6-9 steps, so
    # subsampling would keep the spikes at full height and hide the recovery.
    # Binning lifts the bin containing a switch while still showing the level
    # of the stable stretches and the differences between methods.
    n_bins = int(np.ceil(T / w))
    bin_centers = np.array([(b * w + min((b + 1) * w, T)) / 2.0
                            for b in range(n_bins)])

    def binned(v):
        return np.array([np.nanmean(v[b * w:min((b + 1) * w, T)])
                         for b in range(n_bins)])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for m in ORDER:
        if m not in series:
            continue
        ax.plot(bin_centers, binned(series[m]), markersize=4.5, linewidth=1.3,
                **STYLES[m])

    for s in switches:
        ax.axvline(s, color="0.55", ls=":", lw=1.0, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Time step $t$", fontsize=11)
    ax.set_ylabel("Test MSE", fontsize=11)
    ax.set_xlim(0, T)
    ax.tick_params(labelsize=10)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    fig.savefig(args.outfile, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {args.outfile}")

    # Also print the reporting-window mean, to check against the tables.
    print("\nmean MSE over the reporting window (t >= 50):")
    for m in ORDER:
        if m in series:
            print(f"  {m:11s} {np.nanmean(series[m][50:]):.4f}")


if __name__ == "__main__":
    main()
