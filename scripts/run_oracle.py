#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle 実験 (R1-5, Regression のみ)

N=100 固定・matched 条件・共通乱数で
    厳密補正(Oracle) / Method A 近似 / Method B 近似 / 補正なし PF
を同一条件(同じデータ・初期粒子・η・σ_cd・σ0)で比較する。
これにより「ガウス補正そのものの妥当性」と「近似による誤差」を切り分ける。

共有 HP は WSPF-A の (η, σ_cd, σ0) を用いる(WSPF-A ↔ Oracle の差 =
近似による推定誤差の寄与)。

使い方:
    python scripts/run_oracle.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params)
from src.evaluation import run_seeds, save_run_dir, mean_std, paired_compare

METHODS = ["PF", "Oracle", "WSPF-A", "WSPF-B"]
SHARED_KEYS = ("eta", "sigma_sys", "prior_std")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="regression")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["task_type"] != "regression" or not cfg.get("oracle"):
        raise SystemExit("Oracle は回帰(oracle:true)ベンチマークのみ対応です。")

    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n = cfg["n_particles"]["main"]

    # 共有 HP: WSPF-A の (η, σ_cd, σ0)
    a = get_params(selected, "WSPF-A", n)
    shared = {k: a[k] for k in SHARED_KEYS if k in a}

    rows, per_seed = [], {}
    for m in METHODS:
        params = dict(shared)
        if m == "WSPF-A":
            params["beta"] = a.get("beta", 0.9)
        bench = build_benchmark(cfg)
        results = run_seeds(m, bench, n, params, eval_seeds)
        vals = [np.nanmean(np.asarray(r["metrics"]["mse"])[
                    ~r.get("straddle_mask",
                           np.zeros_like(r["metrics"]["mse"], bool))])
                for r in results]
        per_seed[m] = vals
        mu, sd = mean_std(vals)
        rows.append({"method": m, "metric": "mse", "mean": mu, "std": sd})
        print(f"{m:8s} MSE={mu:.4f}±{sd:.4f}")

    # PF に対する対応のある検定 & WSPF-A − Oracle ギャップ(推定誤差寄与)
    for m in ("Oracle", "WSPF-A", "WSPF-B"):
        cmp = paired_compare(per_seed[m], per_seed["PF"])
        rows.append({"method": f"{m}_vs_PF", "metric": "paired",
                     "mean": cmp.get("mean_diff"), "std": cmp.get("p")})
    gap = float(np.mean(np.array(per_seed["WSPF-A"]) - np.array(per_seed["Oracle"])))
    print(f"WSPF-A − Oracle ギャップ(近似による推定誤差寄与) = {gap:.4f}")
    rows.append({"method": "WSPF-A_minus_Oracle", "metric": "gap",
                 "mean": gap, "std": None})

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "oracle")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={"per_seed": per_seed})
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
