#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the results into the tables used in the paper.

Collects outputs/<benchmark>/*/metrics.csv into a summary table (methods by
metrics) in csv, txt and tex, and generates the selected-hyper-parameter
table.

It also builds the two-tier tables:
  table_mechanism : PF / WSPF-B / WSPF-A / Oracle, the correction mechanism
  table_context   : the above plus SGD / Window-SGD / PH-SGD, for context
Both carry mean +- SD of the primary metric, the NLL and the mixture NLL,
together with the all-pairs paired tests.

Usage:
    python scripts/summarize_results.py --benchmark regression
"""

import argparse
import csv
import glob
import os

from _common import load_config, load_selected
from src.evaluation import write_table

# Row order of the two-tier tables; only methods that exist are emitted.
_MECHANISM_ORDER = ["PF", "WSPF-B", "WSPF-A", "Oracle"]
_CONTEXT_EXTRA = ["SGD", "Window-SGD", "PH-SGD"]


def _read_metrics_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _perf_index(rows):
    """Fold the performance rows into {(method, metric): (mean, std)}."""
    out = {}
    for r in rows:
        if r.get("kind") != "performance":
            continue
        m, met = r.get("method"), r.get("metric")
        if m and met:
            out[(m, met)] = (_num(r.get("mean")), _num(r.get("std")))
    return out


def _build_tier_tables(root, cfg):
    """Build the mechanism and context tables from the metrics.csv files.

    run_oracle.py writes to its own directory and runs under different
    conditions - clipping disabled, hyper-parameters shared - so the oracle
    row carries its provenance in the source column rather than hiding the
    difference inside the table.
    """
    main_rows = []
    main_csv = os.path.join(root, "main", "metrics.csv")
    if os.path.exists(main_csv):
        main_rows = _read_metrics_csv(main_csv)
    oracle_rows = []
    oracle_csv = os.path.join(root, "oracle", "metrics.csv")
    if os.path.exists(oracle_csv):
        oracle_rows = _read_metrics_csv(oracle_csv)
    if not main_rows and not oracle_rows:
        return None, None

    primary = "mse" if cfg["task_type"] == "regression" else "f1"
    metrics = [primary, "nll", "nll_mixture"]
    perf = _perf_index(main_rows)

    # run_oracle.py rows have no kind and are {method, metric, mean, std}.
    oracle_perf = {}
    for r in oracle_rows:
        if r.get("metric") == "mse" and _num(r.get("mean")) is not None:
            oracle_perf[r["method"]] = (_num(r["mean"]), _num(r.get("std")))

    def _rows_for(order):
        out = []
        for m in order:
            have_main = any((m, met) in perf for met in metrics)
            if have_main:
                row = {"method": m, "source": "main"}
                for met in metrics:
                    mu, sd = perf.get((m, met), (None, None))
                    row[f"{met}_mean"] = mu
                    row[f"{met}_std"] = sd
                out.append(row)
            elif m in oracle_perf:
                mu, sd = oracle_perf[m]
                row = {"method": m,
                       "source": "oracle (no clipping; shared HPs)"}
                for met in metrics:
                    row[f"{met}_mean"] = mu if met == primary else None
                    row[f"{met}_std"] = sd if met == primary else None
                out.append(row)
        return out

    mech = _rows_for(_MECHANISM_ORDER)
    ctx = _rows_for(_MECHANISM_ORDER + _CONTEXT_EXTRA)
    return (mech or None), (ctx or None)


def _paired_rows(root):
    """Merge the paired-test rows of main/ and oracle/ into one table."""
    out = []
    for sub in ("main", "oracle"):
        path = os.path.join(root, sub, "metrics.csv")
        if not os.path.exists(path):
            continue
        for r in _read_metrics_csv(path):
            metric = r.get("metric", "")
            if r.get("kind") == "paired" or metric.endswith("_paired"):
                out.append({"experiment": sub, **r})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        cfg["output_dir"])

    # (1) Collect the metrics.csv of every sub-experiment
    all_rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.csv"))):
        sub = os.path.basename(os.path.dirname(path))
        for row in _read_metrics_csv(path):
            row = dict(row)
            row["experiment"] = sub
            all_rows.append(row)
    if all_rows:
        write_table(all_rows, os.path.join(root, "summary_all"))
        print(f"collected {len(all_rows)} rows -> {root}/summary_all.*")
    else:
        print(f"[note] no {root}/*/metrics.csv found; run the run_* scripts first.")

    # (2) The two-tier tables and the all-pairs paired tests
    mech, ctx = _build_tier_tables(root, cfg)
    if mech:
        write_table(mech, os.path.join(root, "table_mechanism"))
        print(f"mechanism table ({len(mech)} methods): {root}/table_mechanism.*")
    if ctx:
        write_table(ctx, os.path.join(root, "table_context"))
        print(f"context table ({len(ctx)} methods): {root}/table_context.*")
    if not mech and not ctx:
        print("[note] no main/metrics.csv, so the two-tier tables are skipped")
    paired = _paired_rows(root)
    if paired:
        write_table(paired, os.path.join(root, "table_paired_tests"))
        print(f"paired tests ({len(paired)} comparisons): "
              f"{root}/table_paired_tests.*")

    # (3) Selected-hyper-parameter table
    try:
        selected = load_selected(cfg)
        hp_rows = []
        for n, per_m in selected.get("by_n_particles", {}).items():
            for m, params in per_m.items():
                hp_rows.append({"method": m, "N": n, **(params or {})})
        for m, params in selected.get("no_n", {}).items():
            hp_rows.append({"method": m, "N": "-", **(params or {})})
        if hp_rows:
            write_table(hp_rows, os.path.join(root, "selected_hp_table"))
            print(f"hyper-parameters: {root}/selected_hp_table.*")
    except FileNotFoundError:
        print("[note] selected_params.json missing; grid_search has not run")


if __name__ == "__main__":
    main()
