#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
計算コスト・スケーリング測定 (R1-11, R2-1)

  - オンライン更新1回当たりの実行時間(t_step/t_grad/t_correction/... のms要約)
  - サンプルごとの勾配評価回数
  - runtime vs N, runtime vs stream length(累積実行時間)
  - low-rank 補正の計時(WSPF-A の t_correction)
  - ピークメモリ (tracemalloc) と WSPF-A の低ランクバッファの静的見積り (R1-11)
  - パラメータ次元 d スイープ: hidden_dim を変えて runtime/memory vs d を測る (R2-1)

R2-1 の stream-length スケーリングは config.eval.stream_length_checkpoints
(例: 500/1000/2000/全ステップ)の累積 t_step を報告する。

使い方:
    python scripts/benchmark_compute.py --benchmark gefcom
"""

import argparse
import os
import tracemalloc

import numpy as np

from _common import (load_config, build_benchmark, load_selected, get_params)
from src.evaluation import (run_method, save_run_dir, timing_report,
                            write_table)

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]

# 次元スイープで走査する hidden_dim(ベンチマーク別)
DIM_SWEEP = {
    "regression": [8, 16, 32, 64],
    "gefcom": [16, 32, 64, 128],
}
# メモリ/次元スイープ計測時のストリーム上限(回帰は T を縮めて高速化)
MEM_CAP_T = 80
DIM_CAP_STEPS = 40   # gefcom は全ストリームだが先頭 ~40 ステップのみ計時


# ======================================================================
# 補助
# ======================================================================
def _peak_mem_mb(fn):
    """fn() を tracemalloc で包み (戻り値, ピーク MB) を返す。"""
    tracemalloc.start()
    try:
        out = fn()
    finally:
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return out, peak / 1e6


def _capped_timing(history, cap=DIM_CAP_STEPS, warmup=5):
    """history の先頭区間 [warmup:cap] から t_step/t_correction の平均 ms を返す。"""
    out = {}
    for k in ("t_step", "t_correction"):
        arr = np.asarray(history.get(k, []), dtype=np.float64)
        arr = arr[warmup:cap] if arr.size > warmup else arr
        out[k] = 1e3 * float(np.nanmean(arr)) if arr.size else float("nan")
    return out


def _static_lowrank_mb(method, benchmark, n_particles):
    """WSPF-A の EMA バッファ ema_m(N×d)の静的メモリ見積り MB。他は空。"""
    if method != "WSPF-A":
        return ""
    d = int(getattr(benchmark, "param_dim", 0))
    return float(n_particles * d * 8) / 1e6   # float64 = 8 byte


# ======================================================================
# メイン
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    n_sweep = cfg["n_particles"]["sweep"]
    n_main = cfg["n_particles"]["main"]
    checkpoints = cfg.get("eval", {}).get("stream_length_checkpoints", [-1])
    bench_name = cfg["benchmark"]

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

    # (c) ピークメモリ (R1-11): 各フィルタ手法を N=main・小ストリームで1回計測。
    for m in FILTER_METHODS:
        params = get_params(selected, m, n_main)
        # 回帰は T を縮めて高速化(gefcom は T を受け付けないので無視される)。
        bench = build_benchmark(cfg, T=MEM_CAP_T)

        def _run():
            return run_method(m, bench, n_main, params, seed=0,
                              collect_diagnostics=True)

        r, peak_mb = _peak_mem_mb(_run)
        rows.append({"method": m, "N": n_main, "section": "peak_mem",
                     "peak_mem_MB": peak_mb,
                     "static_lowrank_MB": _static_lowrank_mb(m, bench, n_main)})
        print(f"{m:7s} N={n_main:4d} peak_mem={peak_mb:.2f} MB")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "compute_cost")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"保存: {out_dir}")

    # (d) パラメータ次元 d スイープ (R2-1): hidden_dim を変えて WSPF-A/B を計時。
    dim_rows = dim_sweep(cfg, bench_name, selected, n_main)
    if dim_rows:
        dim_base = os.path.join(out_dir, "dim_sweep")
        write_table(dim_rows, dim_base, formats=("csv", "txt", "tex"))
        print(f"保存: {dim_base}.{{csv,txt,tex}}  ({len(dim_rows)} 行)")


def dim_sweep(cfg, bench_name, selected, n_main):
    """hidden_dim を走査し WSPF-A/WSPF-B の t_step/t_correction/peak_mem を測る。"""
    hidden_dims = DIM_SWEEP.get(bench_name)
    if not hidden_dims:
        print(f"[注意] 次元スイープ未定義のベンチマーク: {bench_name}")
        return []
    rows = []
    for h in hidden_dims:
        for m in ("WSPF-A", "WSPF-B"):
            params = get_params(selected, m, n_main)
            # 回帰は T を縮める(gefcom は無視される→全ストリームだが先頭のみ計時)。
            bench = build_benchmark(cfg, hidden_dim=h, T=MEM_CAP_T)
            d = int(bench.param_dim)

            def _run():
                return run_method(m, bench, n_main, params, seed=0,
                                  collect_diagnostics=True)

            r, peak_mb = _peak_mem_mb(_run)
            hist = r.get("history") or {}
            if bench_name == "gefcom":
                tm = _capped_timing(hist)
            else:
                tr = timing_report(hist)
                tm = {"t_step": tr.get("t_step", {}).get("mean_ms", np.nan),
                      "t_correction": tr.get("t_correction", {}).get("mean_ms",
                                                                     np.nan)}
            rows.append({"benchmark": bench_name, "method": m,
                         "hidden_dim": h, "param_dim": d,
                         "t_step_ms": tm["t_step"],
                         "t_correction_ms": tm["t_correction"],
                         "peak_mem_MB": peak_mb})
            print(f"{m:7s} h={h:4d} d={d:5d} t_step={tm['t_step']:.2f}ms "
                  f"peak_mem={peak_mb:.2f}MB")
    return rows


if __name__ == "__main__":
    main()
