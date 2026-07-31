#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
計算コスト・スケーリング測定 (R1-11, R2-1)

  - オンライン更新1回当たりの実行時間(t_step/t_grad/t_correction/... のms要約)
  - サンプルごとの勾配評価回数
  - runtime vs N, runtime vs stream length(累積実行時間)
  - low-rank 補正の計時(WSPF-A の t_correction)

R2-1 の stream-length スケーリングは config.eval.stream_length_checkpoints
(例: 500/1000/2000/全ステップ)の累積 t_step を報告する。

使い方:
    python scripts/benchmark_compute.py --benchmark gefcom
"""

import argparse
import os

import numpy as np

from _common import (load_config, build_benchmark, load_selected, get_params)
from src.evaluation import (run_method, save_run_dir, timing_report)

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    n_sweep = cfg["n_particles"]["sweep"]
    n_main = cfg["n_particles"]["main"]
    checkpoints = cfg.get("eval", {}).get("stream_length_checkpoints", [-1])

    rows = []
    # (a) runtime vs N
    for m in FILTER_METHODS:
        params = get_params(selected, m, n_main)
        for n in n_sweep:
            bench = build_benchmark(cfg)
            r = run_method(m, bench, n, params, seed=0, collect_diagnostics=True)
            if not r.get("history"):
                continue
            tr = timing_report(r["history"])
            t_step = tr.get("t_step", {}).get("mean_ms", np.nan)
            sge = int(np.nanmean(r["history"].get("sample_grad_evals",
                                                  [np.nan])))
            rows.append({"method": m, "N": n, "t_step_ms": t_step,
                         "t_correction_ms": tr.get("t_correction", {}).get("mean_ms"),
                         "sample_grad_evals_per_step": sge})
            print(f"{m:7s} N={n:4d} t_step={t_step:.2f}ms grad_evals/step={sge}")

    # (b) runtime vs stream length (累積 t_step, R2-1)
    for m in FILTER_METHODS:
        params = get_params(selected, m, n_main)
        bench = build_benchmark(cfg)
        r = run_method(m, bench, n_main, params, seed=0, collect_diagnostics=True)
        if not r.get("history"):
            continue
        t_step_series = np.asarray(r["history"]["t_step"])
        cum = np.cumsum(t_step_series)
        for c in checkpoints:
            idx = (len(cum) - 1) if c == -1 else min(c, len(cum)) - 1
            if idx < 0:
                continue
            rows.append({"method": m, "N": n_main, "stream_len": (
                "all" if c == -1 else c),
                "cumulative_runtime_s": float(cum[idx])})

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "compute_cost")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
