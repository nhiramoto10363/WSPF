#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute cost and scaling.

  - wall-clock time of one online update (t_step, t_grad, t_correction, ...,
    summarized in milliseconds)
  - number of per-sample gradient evaluations
  - runtime against N, and against stream length (cumulative)
  - the cost of the low-rank correction (t_correction of WSPF-A)
  - peak memory (tracemalloc), plus a static estimate of the low-rank
    buffers of WSPF-A
  - a sweep over the parameter dimension d, varying hidden_dim

The stream-length scaling reports the cumulative t_step at the checkpoints in
config.eval.stream_length_checkpoints (e.g. 500 / 1000 / 2000 / all steps).

Usage:
    python scripts/benchmark_compute.py --benchmark regression
"""

import argparse
import os
import tracemalloc

import numpy as np

from _common import load_config, build_benchmark, load_selected, get_params
from src.evaluation import (run_method, save_run_dir, timing_report,
                            write_table)

FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]

# hidden_dim values visited by the dimension sweep, per benchmark
DIM_SWEEP = {
    "regression": [8, 16, 32, 64],
}
# Stream cap for the memory and dimension measurements. max_steps truncates
# the loop itself, so every benchmark stops after this many steps.
MEM_CAP_T = 80
DIM_CAP_STEPS = 40   # the dimension sweep is timed over this many steps


# ======================================================================
# Helpers
# ======================================================================
def _peak_mem_mb(fn):
    """Run fn() under tracemalloc and return (result, peak MB)."""
    tracemalloc.start()
    try:
        out = fn()
    finally:
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return out, peak / 1e6


def _static_lowrank_mb(method, benchmark, n_particles):
    """Static size, in MB, of the float64 buffers of the WSPF-A correction.

    Empty for the other methods. The buffers counted (all float64, 8 bytes):
      - the EMA state ema_m           N x d
      - the particles                 N x d
      - the per-sample deviations W   N x B x d
      - the B-by-B matrices (M, G)    N x B x B
      - the temporary of the solve    N x B x d
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
    solve_mb = float(N * B * d * bytes_f64) / 1e6      # solve temporary
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
# Main
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
            rows.append({"method": m, "N": n, "zone": None, "t_step_ms": t_step,
                         "t_correction_ms": tr.get("t_correction", {}).get("mean_ms"),
                         "sample_grad_evals_per_step": sge})
            print(f"{m:7s} N={n:4d} t_step={t_step:.2f}ms grad_evals/step={sge}")

    # (b) runtime vs stream length (cumulative t_step)
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
            rows.append({"method": m, "N": n_main, "zone": None, "stream_len": (
                "all" if c == -1 else c),
                "cumulative_runtime_s": float(cum[idx])})

    # (c) Peak memory: one run per filter at N=main over a short stream.
    for m in FILTER_METHODS:
        params = get_params(selected, m, n_main)
        # T=MEM_CAP_T applies to regression; the real-data benchmarks ignore
        # it, and max_steps guarantees the truncation either way.
        bench = build_benchmark(cfg, T=MEM_CAP_T)

        def _run():
            return run_method(m, bench, n_main, params, seed=0,
                              collect_diagnostics=True, max_steps=MEM_CAP_T)

        r, peak_mb = _peak_mem_mb(_run)
        # tracemalloc only tracks Python allocations, not the true RSS peak
        # of NumPy, which the column name makes explicit.
        row = {"method": m, "N": n_main, "zone": None, "section": "peak_mem",
               "peak_python_traced_mem_MB": peak_mb}
        row.update(_static_lowrank_mb(m, bench, n_main))  # expand the breakdown
        rows.append(row)
        print(f"{m:7s} N={n_main:4d} peak_python_traced_mem={peak_mb:.2f} MB")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "compute_cost")
    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=rows, diagnostics={})
    print(f"saved: {out_dir}")

    # (d) Dimension sweep: time WSPF-A and WSPF-B against hidden_dim.
    dim_rows = dim_sweep(cfg, bench_name, selected, n_main)
    if dim_rows:
        dim_base = os.path.join(out_dir, "dim_sweep")
        write_table(dim_rows, dim_base, formats=("csv", "txt", "tex"))
        print(f"saved: {dim_base}.{{csv,txt,tex}}  ({len(dim_rows)} rows)")


def dim_sweep(cfg, bench_name, selected, n_main):
    """Measure t_step, t_correction and peak memory against hidden_dim."""
    hidden_dims = DIM_SWEEP.get(bench_name)
    if not hidden_dims:
        print(f"[note] no dimension sweep defined for {bench_name}")
        return []
    rows = []
    for h in hidden_dims:
        for m in ("WSPF-A", "WSPF-B"):
            params = get_params(selected, m, n_main)
            # As above, max_steps is what actually truncates the loop.
            bench = build_benchmark(cfg, hidden_dim=h, T=MEM_CAP_T)
            d = int(bench.param_dim)

            def _run():
                return run_method(m, bench, n_main, params, seed=0,
                                  collect_diagnostics=True,
                                  max_steps=DIM_CAP_STEPS)

            r, peak_mb = _peak_mem_mb(_run)
            hist = r.get("history") or {}
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
