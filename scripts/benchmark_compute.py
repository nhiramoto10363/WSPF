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
# メモリ/次元スイープ計測時のストリーム上限。max_steps でループ自体を打ち切る
# ため、gefcom を含む全ベンチマークがこのステップ数で確実に停止する。
MEM_CAP_T = 80
DIM_CAP_STEPS = 40   # 次元スイープは先頭 DIM_CAP_STEPS ステップで停止して計時


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


def _capped_timing(history, warmup=5):
    """max_steps でループ自体が既に打ち切られているので、短い history をそのまま
    timing_report で集計する(先頭 warmup ステップのみ除外)。"""
    tr = timing_report(history or {}, warmup=warmup)
    return {"t_step": tr.get("t_step", {}).get("mean_ms", np.nan),
            "t_correction": tr.get("t_correction", {}).get("mean_ms",
                                                           np.nan)}


def _static_lowrank_mb(method, benchmark, n_particles):
    """WSPF-A の低ランク補正(Woodbury 経路)が確保する float64 バッファの静的
    メモリ見積り MB を内訳付きで返す。他手法は空 dict。

    含めるバッファ(いずれも float64 = 8 byte):
      - EMA 状態 ema_m            : N×d
      - 粒子配列 particles        : N×d
      - サンプル毎の勾配偏差 W     : N×B×d
      - B×B 小行列 (M / G)        : N×B×B
      - solve 用の一時バッファ     : N×B×d
    """
    if method != "WSPF-A":
        return {}
    d = int(getattr(benchmark, "param_dim", 0))
    B = int(getattr(benchmark, "batch_size", 16))
    N = int(n_particles)
    bytes_f64 = 8

    ema_mb = float(N * d * bytes_f64) / 1e6            # ema_m: N×d
    particles_mb = float(N * d * bytes_f64) / 1e6      # particles: N×d
    deviations_mb = float(N * B * d * bytes_f64) / 1e6  # W: N×B×d
    bxb_mb = float(N * B * B * bytes_f64) / 1e6        # M / G: N×B×B
    solve_mb = float(N * B * d * bytes_f64) / 1e6      # solve 一時: N×B×d
    total_mb = ema_mb + particles_mb + deviations_mb + bxb_mb + solve_mb
    return {
        "static_ema_MB": ema_mb,
        "static_particles_MB": particles_mb,
        "static_deviations_MB": deviations_mb,
        "static_BxB_MB": bxb_mb,
        "static_solve_MB": solve_mb,
        "static_total_MB": total_mb,
    }


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
        # T=MEM_CAP_T は回帰の回帰用に残す(gefcom は無視)が、実際の打ち切りは
        # max_steps で保証する(gefcom も先頭 MEM_CAP_T ステップで停止)。
        bench = build_benchmark(cfg, T=MEM_CAP_T)

        def _run():
            return run_method(m, bench, n_main, params, seed=0,
                              collect_diagnostics=True, max_steps=MEM_CAP_T)

        r, peak_mb = _peak_mem_mb(_run)
        row = {"method": m, "N": n_main, "section": "peak_mem",
               "peak_mem_MB": peak_mb}
        row.update(_static_lowrank_mb(m, bench, n_main))  # 内訳を行に展開
        rows.append(row)
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
            # T=MEM_CAP_T は回帰用に残すが、実際の打ち切りは max_steps で保証する
            # (gefcom も先頭 DIM_CAP_STEPS ステップでループを停止)。
            bench = build_benchmark(cfg, hidden_dim=h, T=MEM_CAP_T)
            d = int(bench.param_dim)

            def _run():
                return run_method(m, bench, n_main, params, seed=0,
                                  collect_diagnostics=True,
                                  max_steps=DIM_CAP_STEPS)

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
