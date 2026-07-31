#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N スイープ (R2-2, 有限粒子挙動)

主分析: 各手法とも **N=100(main) で選択した HP を固定** し、
        N ∈ {25,50,100,200,400} を実行する。
        (N ごとの再チューニングはしない。それは補足分析として Supplement へ)

指標(MSE/F1)に加えて診断量(ESS/重み縮退/リサンプリング頻度)と
実行時間も集計できるよう、履歴を保存する。

使い方:
    python scripts/run_n_sweep.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask, benchmark_contexts,
                     masked_history)
from src.evaluation import (run_seeds, save_run_dir, mean_std,
                            summarize_history, timing_report)

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--tuned-per-n", action="store_true",
                    help="各 N でチューニングした HP を使う(Supplement)。"
                         "既定は N=main の HP を固定(R2-2 主分析)。"
                         "先に grid_search.py --tune-all-n が必要。")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n_main = cfg["n_particles"]["main"]
    n_sweep = cfg["n_particles"]["sweep"]
    key = "mse" if cfg["task_type"] == "regression" else "f1"

    contexts = benchmark_contexts(cfg, selected)   # GEFCom は 3zone+保存noise
    rows = []
    for ctx in contexts:
        zone = ctx.get("zone")
        for m in FILTER_METHODS:
            # 主分析: N=main の HP を固定して全 N で使う
            params_fixed = get_params(selected, m, n_main)
            for n in n_sweep:
                # Supplement モードでは各 N のチューニング済み HP を使う
                params = get_params(selected, m, n) if args.tuned_per_n else params_fixed
                bench = build_benchmark(cfg, **ctx)
                results = run_seeds(m, bench, n, params, eval_seeds)
                # R2-2: weight degeneracy + runtime も report 区間で集計する。
                acc = {k: [] for k in (
                    "val", "ess_over_N", "entropy_norm", "max_weight",
                    "spread", "unique_all", "unique_resampled",
                    "resample_rate", "t_step_ms")}
                for r in results:
                    mask = region_mask(r, "report")   # straddle 除外済み
                    acc["val"].append(np.nanmean(np.asarray(r["metrics"][key])[mask]))
                    if r.get("history"):
                        hr = masked_history(r["history"], mask)  # report 区間に限定
                        d = summarize_history(hr)
                        pre, post = d["pre_resample"], d["post_resample"]
                        acc["ess_over_N"].append(pre.get("ess", np.nan) / n)
                        # エントロピーは N 依存を除くため logN で正規化
                        acc["entropy_norm"].append(pre.get("entropy", np.nan) / np.log(n))
                        acc["max_weight"].append(pre.get("max_weight", np.nan))
                        acc["spread"].append(pre.get("spread_trace", np.nan))
                        # ユニーク祖先率の分母は実際の N を使う。summarize_history は
                        # N を max(unique) で推定するため、毎ステップ縮退する高次元
                        # 条件では分母を過小評価し率を過大評価する。ここは n が既知
                        # なので明示的に n で割る。
                        unique = np.asarray(hr.get("unique_particles", []), dtype=float)
                        resampled = np.asarray(hr.get("resampled", []), dtype=bool)
                        if unique.size:
                            acc["unique_all"].append(float(np.mean(unique) / n))
                            if resampled.size == unique.size and resampled.any():
                                acc["unique_resampled"].append(
                                    float(np.mean(unique[resampled]) / n))
                            else:
                                acc["unique_resampled"].append(np.nan)
                        acc["resample_rate"].append(post.get("resample_rate", np.nan))
                        acc["t_step_ms"].append(
                            timing_report(hr).get("t_step", {}).get("mean_ms", np.nan))
                mu, sd = mean_std(acc["val"])

                def _m(k):
                    v = acc[k]
                    return float(np.nanmean(v)) if v and np.any(np.isfinite(v)) else None

                rows.append({"method": m, "N": n, "zone": zone, "metric": key,
                             "mean": mu, "std": sd,
                             "ess_over_N": _m("ess_over_N"),
                             "entropy_norm": _m("entropy_norm"),
                             "max_weight": _m("max_weight"),
                             "particle_spread": _m("spread"),
                             "unique_ancestor_rate_all": _m("unique_all"),
                             "unique_ancestor_rate_resampled": _m("unique_resampled"),
                             "resample_rate": _m("resample_rate"),
                             "t_step_ms": _m("t_step_ms")})
                ztag = f"[zone {zone}] " if zone is not None else ""
                print(f"{ztag}{m:8s} N={n:4d}  {key}={mu:.4f}±{sd:.4f} "
                      f"ESS/N={_m('ess_over_N')}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "n_sweep")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}  (N別再チューニングは Supplement)")


if __name__ == "__main__":
    main()
