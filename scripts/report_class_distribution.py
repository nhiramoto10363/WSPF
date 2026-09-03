#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report the class distribution of each regime.

Applies to the classification benchmarks (email, binary; insects, six
classes). Regimes are delimited by the concept-drift points, and the table
gives the class proportions of each one together with whether it falls in the
selection or the reporting window.

Since email is a semisynthetic stream whose topic labels reverse at the drift
points, the swap in class proportions between adjacent regimes is visible in
the table.

Usage:
    python3 scripts/report_class_distribution.py --benchmark email
    python3 scripts/report_class_distribution.py --benchmark insects
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import load_config, build_benchmark          # noqa: E402
from src.evaluation import (class_distribution_rows,      # noqa: E402
                            format_class_distribution, write_table)


def _report_start_samples(cfg, bench):
    """Start of the reporting window as a *sample* index, or None.

    For email, report_start is already in samples; for insects,
    select_end_step counts batch steps and is multiplied by batch_size.
    """
    name = cfg["benchmark"]
    if name == "email":
        rs = cfg.get("eval", {}).get("report_start",
                                     cfg.get("data", {}).get("report_start"))
        return None if rs is None else int(rs)
    if name == "insects":
        step = getattr(bench, "select_end_step", None)
        if step is None:
            return None
        return int(step) * int(getattr(bench, "batch_size", 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="email / insects, or a config path")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["task_type"] != "classification":
        raise SystemExit(
            f"class distributions apply to classification benchmarks only; "
            f"{cfg['benchmark']} has task_type={cfg['task_type']}")

    bench = build_benchmark(cfg)
    y = bench.loader.y
    n_classes = int(getattr(bench, "n_classes", 2))
    change_points = list(getattr(bench, "switch_points", []))
    class_names = getattr(getattr(bench, "loader", None), "class_names", None)
    report_start = _report_start_samples(cfg, bench)

    rows = class_distribution_rows(
        y, change_points, n_classes,
        benchmark=cfg["benchmark"], report_start=report_start,
        class_names=class_names)

    print(f"===== {cfg['benchmark']}: class distribution by regime =====")
    print(f"  n_samples={len(y)}  n_classes={n_classes}")
    print(f"  change points (samples): {change_points}")
    print(f"  reporting window starts at sample: {report_start}")
    if class_names is not None:
        print(f"  classes: {list(class_names)}")
    format_class_distribution(rows, n_classes)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "class_distribution")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "class_distribution")
    write_table(rows, base)
    print(f"\nsaved: {base}.{{csv,txt,tex}}")


if __name__ == "__main__":
    main()
