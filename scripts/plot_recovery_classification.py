#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-switch recovery curves for the classification benchmarks.

Reads the recovery_acc and recovery_f1 rows (mean and std per lag) that
calibration_report.py wrote to calibration/calibration_report.csv, so
nothing is re-run and the figure necessarily matches the tables.

Why not plot_recovery.py: that script derives switch points from the run's
switch_mask, i.e. the straddling blocks, which is wrong here. The email
drift points are sample indices [300, 600, 900, 1200] with a block width of
16, so sample 1200 lands exactly on a block boundary and never straddles,
while sample 300 lies inside the selection window yet would be picked up.
calibration_report uses the benchmark's switch_points directly and drops any
drift point whose lag 0 falls outside the reporting window.

Usage:
    python3 scripts/plot_recovery_classification.py --benchmark email --metric f1
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STYLES = {
    "SGD":        dict(color="#888888", marker="x", label="SGD"),
    "PH-SGD":     dict(color="#B8860B", marker="v", label="PH-SGD"),
    "Window-SGD": dict(color="#6A3D9A", marker="s", label="Window-SGD"),
    "PF":         dict(color="#2196F3", marker="o", label="PF (uncorrected)"),
    "WSPF-A":     dict(color="#4CAF50", marker="D", label="WSPF-A"),
    "WSPF-B":     dict(color="#E91E63", marker="^", label="WSPF-B"),
}
ORDER = ["SGD", "PH-SGD", "Window-SGD", "PF", "WSPF-A", "WSPF-B"]

_YLABEL = {"acc": "Report-region accuracy", "f1": "Report-region $F_1$"}


def load_curves(bench, metric):
    path = os.path.join(_REPO_ROOT, "outputs", bench, "calibration",
                        "calibration_report.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run scripts/calibration_report.py first.")
    key = f"recovery_{metric}"
    acc = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("metric") != key:
                continue
            lag = int(r["level"].split("_")[1])
            acc.setdefault(r["method"], {})[lag] = (
                float(r["mean"]), float(r["std"]))
    out = {}
    for m, d in acc.items():
        lags = np.array(sorted(d))
        out[m] = (lags,
                  np.array([d[int(l)][0] for l in lags]),
                  np.array([d[int(l)][1] for l in lags]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="email", choices=["email", "insects"])
    ap.add_argument("--metric", default="acc", choices=["acc", "f1"])
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    curves = load_curves(args.benchmark, args.metric)
    if not curves:
        raise SystemExit("no recovery-curve rows found")

    default_name = ("figure_email_switch_recovery.png" if args.benchmark == "email"
                    else f"figure_{args.benchmark}_switch_recovery.png")
    out = args.outfile or os.path.join(_REPO_ROOT, "outputs", "figures",
                                       default_name)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    n_seeds = None
    for m in ORDER:
        if m not in curves:
            continue
        lags, mu, sd = curves[m]
        # The band is +-1 SE across seeds. Switch events are averaged within
        # a seed first, so they are never treated as independent samples.
        se = sd / np.sqrt(10.0)
        ax.plot(lags, mu, linewidth=1.5, markersize=5, **STYLES[m])
        ax.fill_between(lags, mu - se, mu + se, color=STYLES[m]["color"],
                        alpha=0.15, linewidth=0)
        n_seeds = 10

    ax.set_xlabel(r"$\tau$ (blocks after concept switch)", fontsize=11)
    ax.set_ylabel(_YLABEL[args.metric], fontsize=11)
    ax.set_xticks(sorted({int(l) for m in curves for l in curves[m][0]}))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.text(0.5, -0.20,
            f"band = $\\pm 1$ SE across {n_seeds} seeds; switch events are "
            "averaged within each seed before aggregation",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="0.35")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {out}")
    for m in ORDER:
        if m in curves:
            lags, mu, _ = curves[m]
            print(f"  {m:11s} lag0={mu[0]:.4f}  lag{int(lags[-1])}={mu[-1]:.4f}"
                  f"  area={np.mean(mu):.4f}")


if __name__ == "__main__":
    main()
