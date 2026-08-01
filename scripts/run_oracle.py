#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle 実験 (R1-5, Regression のみ)

N=100 固定・matched 条件・**共通乱数(CRN)** で
    厳密補正(Oracle) / Method A 近似 / Method B 近似 / 補正なし PF
を同一条件で比較する。これにより「ガウス補正そのものの妥当性」と
「近似による誤差」を切り分ける。

CRN(修正方針 P4): 4 手法で
  - 同じデータストリーム(同一データ seed)
  - 同じ初期粒子・同じドリフトノイズ系列(共通フィルタシード filter_seed)
を共有する。リサンプリング時点は手法ごとに異なりうるため完全な CRN では
ないが、最低限 初期粒子とドリフトノイズ系列を共有して比較を明確化する。

勾配クリッピング(P4): 厳密なガウス補正を検証するため、Oracle 比較では
grad_clip_norm=None で **無効化** する(クリップ後平均勾配とクリップ前
母集団統計の不整合を避ける)。共有 HP は WSPF-A の (η, σ_cd, σ0) を用いる
(WSPF-A ↔ Oracle の差 = 近似による推定誤差の寄与)。

使い方:
    python scripts/run_oracle.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask)
from src.evaluation import (run_seeds, save_run_dir, mean_std, paired_compare)

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
        # P4: 勾配クリッピング無効化(厳密補正の検証)
        bench = build_benchmark(cfg, grad_clip_norm=None)
        # CRN: 全手法で同一の filter_seed(初期粒子・ドリフトノイズ共有)。
        # seed 間はプロセス並列(Oracle は MC が重いので特に効く)。
        fseeds = [s + 1 for s in eval_seeds]
        results = run_seeds(m, bench, n, params, eval_seeds, filter_seeds=fseeds)
        vals = [np.nanmean(np.asarray(r["metrics"]["mse"])[region_mask(r, "report")])
                for r in results]
        per_seed[m] = vals
        mu, sd = mean_std(vals)
        rows.append({"method": m, "metric": "mse", "mean": mu, "std": sd})
        print(f"{m:8s} MSE={mu:.4f}±{sd:.4f}")

    # PF に対する対応のある検定 & WSPF-A − Oracle ギャップ(推定誤差寄与)
    # p値は std ではなく専用列に格納する(較正レポートと形式を統一)。
    for m in ("Oracle", "WSPF-A", "WSPF-B"):
        cmp = paired_compare(per_seed[m], per_seed["PF"])
        diffs = np.asarray(per_seed[m], float) - np.asarray(per_seed["PF"], float)
        rows.append({"method": f"{m}_vs_PF", "metric": "mse_paired",
                     "mean_difference": cmp.get("mean_diff"),
                     "std_difference": float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0,
                     "paired_t_p": cmp.get("p")})
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
