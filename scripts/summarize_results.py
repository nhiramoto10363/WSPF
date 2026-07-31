#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
結果の要約 (表・図の再現用集約)

outputs/<benchmark>/*/metrics.csv を集めて、論文用のまとめ表
(手法 × 指標)を csv/txt/tex で出力する。選択されたハイパーパラメータ表も
自動生成する(査読 Minor 対応)。

使い方:
    python scripts/summarize_results.py --benchmark regression
"""

import argparse
import csv
import glob
import os

from _common import load_config, load_selected
from src.evaluation import write_table


def _read_metrics_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        cfg["output_dir"])

    # (1) 各サブ実験の metrics.csv を集約
    all_rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.csv"))):
        sub = os.path.basename(os.path.dirname(path))
        for row in _read_metrics_csv(path):
            row = dict(row)
            row["experiment"] = sub
            all_rows.append(row)
    if all_rows:
        write_table(all_rows, os.path.join(root, "summary_all"))
        print(f"集約: {len(all_rows)} 行 → {root}/summary_all.*")
    else:
        print(f"[注意] {root}/*/metrics.csv が見つかりません。先に run_* を実行してください。")

    # (2) 選択ハイパーパラメータ表の自動生成
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
            print(f"HP表: {root}/selected_hp_table.*")
    except FileNotFoundError:
        print("[注意] selected_params.json 未生成(grid_search 未実行)。")


if __name__ == "__main__":
    main()
