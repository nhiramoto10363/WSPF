#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot of the fitted function shortly after a switch.

The figure is about *where the ensemble sits*, not how wide it is: a few
steps after a switch, the uncorrected PF is still pulled towards the old
regime by the most recent mini-batch, while the corrected ensemble has moved
to the neighbourhood of the true function. (Particle spread itself is
dimension-dependent, and on this benchmark WSPF-A is in fact narrower than
PF, so the width is not the point.)

Drawing the particle cloud requires the particles themselves, so seed 0 is
re-run up to the target step with
run_method(..., max_steps=t+1, return_estimator=True). The regression
benchmark is light enough for this to take seconds, and the hyper-parameters
come from outputs/regression/selected_params.json, so the setting matches
the tables.

Usage:
    python3 scripts/plot_snapshot_func.py
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from _common import (load_config, build_benchmark, load_selected,  # noqa: E402
                     get_params)
from src.evaluation import run_method  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_METHODS = ["PF", "WSPF-A", "WSPF-B"]
COLORS = {"PF": "#2196F3", "WSPF-A": "#4CAF50", "WSPF-B": "#E91E63"}
TITLES = {"PF": "PF (uncorrected)", "WSPF-A": "WSPF-A", "WSPF-B": "WSPF-B"}


def _snapshot_data(bench, seed, t):
    """True parameters and training mini-batch at step t.

    _generate caches the generated data per seed; theta_true has shape
    (T, param_dim).
    """
    d = bench._generate(seed)
    theta_star = np.asarray(d["theta_true"])[t]
    X_tr = np.asarray(d["X_train"])[t]
    y_tr = np.asarray(d["y_train"])[t]
    return theta_star, X_tr, y_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=305, help="snapshot step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="WSPF-A",
                    choices=ALL_METHODS + ["all"],
                    help="method to draw; the default is a single panel "
                         "(WSPF-A), and 'all' gives three side by side")
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config("regression")
    selected = load_selected(cfg)
    n_main = cfg["n_particles"]["main"]
    bench = build_benchmark(cfg)
    model = bench.model

    out = args.outfile or os.path.join(
        _REPO_ROOT, "outputs", "figures",
        f"snapshot_func_t{args.t}_N{n_main}.png")
    # Keep the filename used by the paper.

    panels = ALL_METHODS if args.method == "all" else [args.method]

    # --- Run each method up to t and collect its particles ---
    states = {}
    for m in panels + ["SGD"]:
        params = get_params(selected, m, n_main)
        r = run_method(m, bench, n_main, params, args.seed,
                       collect_diagnostics=False,
                       max_steps=args.t + 1, return_estimator=True)
        est = r["estimator"]
        if m == "SGD":
            theta = np.asarray(getattr(est, "theta", getattr(est, "particles", None)))
            states[m] = theta.reshape(-1)
        else:
            states[m] = (np.asarray(est.particles), np.asarray(est.weights))
        print(f"  {m:8s} collected")

    theta_star, X_tr, y_tr = _snapshot_data(bench, args.seed, args.t)

    # --- Evaluate on the x grid ---
    x_grid = np.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    y_true = model.forward(theta_star.reshape(1, -1), x_grid)[0].squeeze()
    y_sgd = model.forward(states["SGD"].reshape(1, -1), x_grid)[0].squeeze()

    if len(panels) == 1:
        fig, ax0 = plt.subplots(figsize=(6.6, 4.6))
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, m in zip(axes, panels):
        particles, weights = states[m]
        preds = model.forward(particles, x_grid)[0].squeeze(-1)   # (N, G)
        w = weights / weights.sum()
        y_mean = (w[:, None] * preds).sum(axis=0)
        y_var = (w[:, None] * (preds - y_mean) ** 2).sum(axis=0)
        y_std = np.sqrt(np.maximum(y_var, 1e-15))
        c = COLORS[m]

        for i in range(preds.shape[0]):
            ax.plot(x_grid.squeeze(), preds[i], color=c, alpha=0.06, lw=0.5,
                    zorder=1)
        ax.fill_between(x_grid.squeeze(), y_mean - 2 * y_std, y_mean + 2 * y_std,
                        color=c, alpha=0.12, zorder=2)
        ax.fill_between(x_grid.squeeze(), y_mean - y_std, y_mean + y_std,
                        color=c, alpha=0.22, zorder=2)
        ax.plot(x_grid.squeeze(), y_true, "k-", lw=2.2, zorder=5)
        ax.plot(x_grid.squeeze(), y_mean, color=c, lw=2.0, zorder=4)
        ax.plot(x_grid.squeeze(), y_sgd, color="#888888", lw=1.5, ls="--",
                zorder=3)
        ax.scatter(X_tr.squeeze(), y_tr, s=16, c="black", alpha=0.55,
                   zorder=6)

        if len(panels) > 1:
            ax.set_title(TITLES[m], fontsize=14)
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_xlim(-2, 2)
        ax.grid(True, alpha=0.28)
        ax.tick_params(labelsize=10)

        rmse = float(np.sqrt(np.mean((y_mean - y_true) ** 2)))
        print(f"  {m:8s} ensemble-mean RMSE vs true = {rmse:.4f}"
              f"   mean band width (1sd) = {float(np.mean(2*y_std)):.4f}")

    axes[0].set_ylabel("$y$", fontsize=12)

    handles = [
        Line2D([0], [0], color="k", lw=2.2, label="True function"),
        Line2D([0], [0], color="0.4", lw=2.0,
               label="Ensemble mean (per-panel color)"),
        Patch(facecolor="0.4", alpha=0.30, label=r"$\pm 1\sigma$"),
        Patch(facecolor="0.4", alpha=0.15, label=r"$\pm 2\sigma$"),
        Line2D([0], [0], color="#888888", lw=1.5, ls="--", label="SGD"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="black",
               markeredgecolor="black", markersize=6, alpha=0.55,
               label="mini-batch $B_t$"),
    ]
    if len(panels) == 1:
        handles.insert(1, Line2D([0], [0], color=COLORS[panels[0]], lw=2.0,
                                 label="Ensemble mean"))
        handles.pop(2)   # the per-panel colour note is moot in one panel
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, -0.13))
    else:
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
                   fontsize=11, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
