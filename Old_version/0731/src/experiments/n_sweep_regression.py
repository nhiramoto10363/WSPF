#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子数 N スイープ — Regression (R2-2)

N∈{25,50,100,200,400} で PF/WSPF-A/WSPF-B を走らせ、MSE・ESS(と ESS/N)・
リサンプリング頻度・ユニーク祖先数・1更新あたり実行時間の N 依存性を報告する。

HP は N に依らず固定(タスク4のマッチド設定 = PF 最良構成)し、N 依存性のみを
分離する。各 N × 複数シードで集計。

出力:
  outputs/n_sweep/
    - n_sweep_regression.txt / .csv
    - n_sweep_mse.png / n_sweep_ess.png / n_sweep_resample.png / n_sweep_time.png
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiments.regression_regime_switch import (
    run_single, SEEDS, T, EVAL_START,
)
from src.experiments.matched_hp_regression import build_matched_hp

N_GRID = [25, 50, 100, 200, 400]
PARTICLE_METHODS = ["PF", "WSPF-A", "WSPF-B"]
ALL_METHODS = ["SGD"] + PARTICLE_METHODS

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "n_sweep",
)


def _ev(a):
    return np.asarray(a)[EVAL_START:]


def _nsweep_job(args):
    """1 (N, seed) ジョブ。histories から eval-region 集計値のみ返す(軽量)。"""
    N, seed, bp, bb, ba, best_sgd = args
    res = run_single(seed, N, bp, bb, ba, best_sgd=best_sgd)
    out = {"N": N, "mse": {}, "ess": {}, "ess_ratio": {}, "resample": {},
           "unique": {}, "t_step": {}}
    out["mse"]["SGD"] = float(_ev(res["mse"]["SGD"]).mean())
    for m in PARTICLE_METHODS:
        h = res["histories"][m]
        out["mse"][m] = float(_ev(res["mse"][m]).mean())
        ess = _ev(h["ess"])
        out["ess"][m] = float(ess.mean())
        out["ess_ratio"][m] = float(ess.mean() / N)
        out["resample"][m] = float(_ev(h["resampled"]).mean())
        out["unique"][m] = float(_ev(h["unique_particles"]).mean())
        out["t_step"][m] = float(_ev(h["t_step"]).mean())
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched, beta, src, best_sgd = build_matched_hp()[100]
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 72)
    emit("Particle-count (N) sweep — Regression (R2-2)")
    emit(f"  N grid={N_GRID}, T={T}, seeds={len(SEEDS)}, EVAL_START={EVAL_START}")
    emit(f"  fixed HP (η,σcd,σ0)={matched} [{src}], WSPF-A β={beta}")
    emit("=" * 72)

    jobs = []
    for N in N_GRID:
        bp = dict(matched); bb = dict(matched); ba = {**matched, "beta": beta}
        for seed in SEEDS:
            jobs.append((N, seed, bp, bb, ba, best_sgd))

    # 集計コンテナ
    agg = {N: {"mse": {m: [] for m in ALL_METHODS},
               "ess": {m: [] for m in PARTICLE_METHODS},
               "ess_ratio": {m: [] for m in PARTICLE_METHODS},
               "resample": {m: [] for m in PARTICLE_METHODS},
               "unique": {m: [] for m in PARTICLE_METHODS},
               "t_step": {m: [] for m in PARTICLE_METHODS}} for N in N_GRID}

    n_workers = min(os.cpu_count() or 1, 48)
    emit(f"  workers={n_workers}, jobs={len(jobs)}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_nsweep_job, jobs):
            N = r["N"]
            agg[N]["mse"]["SGD"].append(r["mse"]["SGD"])
            for m in PARTICLE_METHODS:
                agg[N]["mse"][m].append(r["mse"][m])
                for key in ("ess", "ess_ratio", "resample", "unique", "t_step"):
                    agg[N][key][m].append(r[key][m])
    emit(f"  elapsed {time.time()-t0:.0f}s")

    def mean(N, key, m):
        return float(np.mean(agg[N][key][m]))

    # ---- 表 ----
    emit(f"\n{'='*72}\n  MSE vs N (eval-region mean over seeds)\n{'='*72}")
    emit(f"  {'N':>5s} " + " ".join(f"{m:>9s}" for m in ALL_METHODS))
    for N in N_GRID:
        emit(f"  {N:>5d} " + " ".join(f"{mean(N,'mse',m):>9.4f}" for m in ALL_METHODS))

    emit(f"\n{'='*72}\n  ESS/N・リサンプル頻度・ユニーク数・時間 vs N\n{'='*72}")
    for m in PARTICLE_METHODS:
        emit(f"\n  [{m}]")
        emit(f"  {'N':>5s} {'ESS':>9s} {'ESS/N':>7s} {'resample%':>10s} "
             f"{'unique':>8s} {'t_step[ms]':>11s}")
        for N in N_GRID:
            emit(f"  {N:>5d} {mean(N,'ess',m):>9.2f} {mean(N,'ess_ratio',m):>7.3f} "
                 f"{100*mean(N,'resample',m):>9.1f}% {mean(N,'unique',m):>8.1f} "
                 f"{1e3*mean(N,'t_step',m):>11.3f}")

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "n_sweep_regression.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,method,mse,ess,ess_ratio,resample_freq,unique,t_step_ms\n")
        for N in N_GRID:
            f.write(f"{N},SGD,{mean(N,'mse','SGD'):.6f},,,,,\n")
            for m in PARTICLE_METHODS:
                f.write(f"{N},{m},{mean(N,'mse',m):.6f},{mean(N,'ess',m):.6f},"
                        f"{mean(N,'ess_ratio',m):.6f},{mean(N,'resample',m):.6f},"
                        f"{mean(N,'unique',m):.6f},{1e3*mean(N,'t_step',m):.6f}\n")

    # ---- プロット ----
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    Ns = np.array(N_GRID)

    def _plot(key, ylabel, title, fname, methods, logy=False, scale=1.0,
              transform=None):
        plt.figure(figsize=(7, 5))
        for m in methods:
            ys = [ (transform(N, m) if transform else scale*mean(N, key, m))
                   for N in N_GRID]
            plt.plot(Ns, ys, "o-", color=colors[m], label=m, linewidth=1.6)
        plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.xticks(Ns, [str(n) for n in N_GRID])
        plt.xlabel("number of particles N")
        plt.ylabel(ylabel); plt.title(title)
        plt.grid(True, alpha=0.3, which="both"); plt.legend()
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        return p

    p1 = _plot("mse", "Test MSE (eval region)", "MSE vs N (Regression)",
               "n_sweep_mse.png", ALL_METHODS)
    p2 = _plot("ess_ratio", "ESS / N", "ESS ratio vs N",
               "n_sweep_ess.png", PARTICLE_METHODS)
    p3 = _plot("resample", "resample frequency", "Resampling frequency vs N",
               "n_sweep_resample.png", PARTICLE_METHODS)
    p4 = _plot("t_step", "time per update [ms]", "Wall-clock per update vs N",
               "n_sweep_time.png", PARTICLE_METHODS, logy=True, scale=1e3)

    txt_path = os.path.join(OUTPUT_DIR, "n_sweep_regression.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    for p in (p1, p2, p3, p4):
        emit(f"Saved: {p}")


if __name__ == "__main__":
    main()
