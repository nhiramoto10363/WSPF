#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Effective sample size against the particle count.

One panel per benchmark (d = 25, 833, 1286), each plotting the *absolute*
ESS (ess_over_N times N) against N on log-log axes.

What the figure shows:
  - the uncorrected PF has ESS proportional to N, a line of slope one, as a
    particle filter should;
  - the corrected WSPF-A and WSPF-B flatten as the dimension grows, and at
    d = 833 and d = 1286 a sixteen-fold increase in N still leaves
    ESS near one particle;
  - that is the direct consequence of the condition N >~ exp(rho^2 d / 4):
    raising N does not rescue the correction.

Reference lines mark ESS = N (ideal) and ESS = 1 (complete degeneracy).
Output goes to outputs/figures/ by default.

Usage:
    python3 scripts/plot_ess_sweep.py
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

# (benchmark, panel title), ordered by dimension
BENCHES = [
    ("regression", "Regression", 25),
    ("email", "Email", 833),
    ("insects", "INSECTS", 1286),
]

# Per-method style; PF is the reference, in black.
STYLES = {
    "PF": dict(color="black", marker="o", ls="-", label="PF (uncorrected)"),
    "WSPF-A": dict(color="#C1272D", marker="s", ls="-", label="WSPF-A"),
    "WSPF-B": dict(color="#1F5FA9", marker="^", ls="-", label="WSPF-B"),
}
ORDER = ["PF", "WSPF-A", "WSPF-B"]


def load_ess(benchmark):
    """Build {method: (N values, absolute ESS)} from the n_sweep rows."""
    path = os.path.join(_REPO_ROOT, "outputs", benchmark, "summary_all.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run scripts/run_n_sweep.py first.")
    acc = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("experiment") != "n_sweep":
                continue
            v = r.get("ess_over_N")
            if not v:
                continue
            m, N = r["method"], int(r["N"])
            # One row per (method, N) is expected; a duplicate does not
            # overwrite the first.
            acc.setdefault(m, {}).setdefault(N, float(v) * N)
    out = {}
    for m, d in acc.items():
        Ns = np.array(sorted(d), dtype=float)
        out[m] = (Ns, np.array([d[int(n)] for n in Ns], dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default=os.path.join(_REPO_ROOT, "outputs",
                                                      "figures",
                                                      "ess_sweep.png"),
                    help="output path (default: outputs/figures/ess_sweep.png)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)

    for ax, (bench, title, d) in zip(axes, BENCHES):
        data = load_ess(bench)
        Nall = None
        for m in ORDER:
            if m not in data:
                print(f"[note] {bench}: no n_sweep rows for {m}; skipping")
                continue
            Ns, ess = data[m]
            Nall = Ns if Nall is None else Nall
            ax.plot(Ns, ess, markersize=6, linewidth=1.8, **STYLES[m])

        if Nall is not None:
            # Reference lines: ESS = N (ideal) and ESS = 1 (degenerate)
            ax.plot(Nall, Nall, color="0.6", ls=":", lw=1.2, zorder=0)
            ax.axhline(1.0, color="0.6", ls="--", lw=1.2, zorder=0)
            ax.text(Nall[-1], Nall[-1] * 1.12, r"$\mathrm{ESS}=N$",
                    color="0.45", ha="right", va="bottom", fontsize=9)
            # Label the ESS = 1 line below it: in the high-dimensional
            # panels the WSPF points sit near y = 1.1-3 and would collide.
            ax.text(Nall[-1], 0.80, r"$\mathrm{ESS}=1$",
                    color="0.45", ha="right", va="center", fontsize=9)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title}  ($d={d}$)", fontsize=12)
        ax.set_xlabel("Number of particles $N$", fontsize=11)
        ax.set_xticks([25, 50, 100, 200, 400])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(labelsize=10)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)

    axes[0].set_ylabel("Effective sample size", fontsize=11)
    axes[0].set_ylim(0.7, 700)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    out = args.outfile
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved: {out}")

    # Print the plotted values, so they can be checked against the tables.
    for bench, title, d in BENCHES:
        data = load_ess(bench)
        print(f"\n[{bench}] d={d}  absolute ESS")
        for m in ORDER:
            if m not in data:
                continue
            Ns, ess = data[m]
            cells = "  ".join(f"N={int(n)}:{e:7.2f}" for n, e in zip(Ns, ess))
            print(f"  {m:8s} {cells}")


if __name__ == "__main__":
    main()
