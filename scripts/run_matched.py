#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matched ハイパーパラメータ比較 (R1-7)

N=100 固定。共有するのは η, σ_cd(sigma_sys), σ0(prior_std) のみ。
WSPF-A の β は各手法自身の選択値を使う(混入させない)。

  for fixed_method in [PF, WSPF-A, WSPF-B]:      # 3 通りの共有 HP ソース
      shared = {η, σ_cd, σ0} = best of fixed_method at N=100
      for method in [PF, WSPF-A, WSPF-B]:
          run(method, shared)                    # → 3×3 の結果表

使い方:
    python scripts/run_matched.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params)
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

    rows = []
    for fixed in FILTER_METHODS:
        base = get_params(selected, fixed, n)
        shared = {k: base[k] for k in SHARED_KEYS if k in base}
        for m in FILTER_METHODS:
            params = dict(shared)
            if m == "WSPF-A":  # β は WSPF-A 自身の選択値
                params["beta"] = get_params(selected, "WSPF-A", n).get("beta", 0.9)
            bench = build_benchmark(cfg)
            results = run_seeds(m, bench, n, params, eval_seeds)
            vals = [np.nanmean(np.asarray(r["metrics"][key])[
                        ~r.get("straddle_mask",
                               np.zeros_like(r["metrics"][key], bool))])
                    for r in results]
            mu, sd = mean_std(vals)
            rows.append({"shared_from": fixed, "method": m,
                         "metric": key, "mean": mu, "std": sd})
            print(f"shared={fixed:7s} → {m:7s}  {key}={mu:.4f}±{sd:.4f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "matched_hp")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}  (3×3 表)")


if __name__ == "__main__":
    main()
