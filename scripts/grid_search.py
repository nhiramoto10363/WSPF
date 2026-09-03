#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid search over the selection window only.

The main analysis fixes the hyper-parameters selected at N=100 and then
sweeps N, rather than re-tuning per N, so by default only N=main is searched.
Per-N tuning is a supplementary analysis, enabled with --tune-all-n and used
by the tuned-per-N mode of run_n_sweep.py.

Filters (PF, WSPF-A, WSPF-B) and the point-estimate baselines are all scored
by their mean over the selection seeds; the baselines are selected at N=main.
The oracle needs no search of its own, since run_oracle.py shares the
hyper-parameters of WSPF-A.

Results are written to outputs/<benchmark>/selected_params.json. A warning is
printed when the selected point sits on an edge of the grid.

Usage:
    python scripts/grid_search.py --benchmark regression
    python scripts/grid_search.py --benchmark regression --tune-all-n
"""

import argparse
import json
import os

from _common import load_config, resolve_seeds, grid_search, hp_path

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / email / insects, or a config path")
    ap.add_argument("--tune-all-n", action="store_true",
                    help="search every N separately (supplementary "
                         "per-N tuning); the default searches N=main only")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    sel = resolve_seeds(cfg, "selection")
    n_main = cfg["n_particles"]["main"]
    n_sweep = cfg["n_particles"]["sweep"]

    result = {"by_n_particles": {}, "no_n": {}}
    contexts = None          # all three benchmarks run single-context

    # Filters: N=main by default, every N with --tune-all-n.
    target_ns = n_sweep if args.tune_all_n else [n_main]
    for n in target_ns:
        print(f"[N={n}]")
        result["by_n_particles"][str(n)] = {}
        for m in FILTER_METHODS:
            best, score = grid_search(m, cfg, n, sel, contexts=contexts)
            result["by_n_particles"][str(n)][m] = best

    # Point-estimate baselines have no N; they are selected at N=main.
    base_methods = [m for m in cfg["methods"]
                    if m in ("SGD", "PH-SGD", "Window-SGD")]
    print("[baselines (no N)]")
    for m in base_methods:
        best, score = grid_search(m, cfg, n_main, sel, contexts=contexts)
        result["no_n"][m] = best

    out = hp_path(cfg)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
