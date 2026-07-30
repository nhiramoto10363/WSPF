#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子数 N スイープ — INSECTS (R2-2, 補助)

n_sweep_email.py の鏡像。N∈{25,50,100,200,400} で insects 実験を走らせ、
macro-F1・ESS(ESS/N)・リサンプリング頻度・1 更新あたり実行時間の N 依存性を
報告する (報告区間 = [REPORT_START,end))。各メソッドは自身のグリッド best HP
を使用 (insects_multiseed と同じ方針)。F1 のシード分散を抑えるため少数シードで
平均する。

注) N 依存の ESS/resample/time はシード非依存に近く、機構の N スケーリングを
    示す。d=1286 のため大きい N は計算コストが高い (クラウド前提)。

出力:
  outputs/n_sweep/
    - insects_n_sweep.txt / .csv / insects_n_sweep.png
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.experiments.insects_experiment as IE
from src.data.insects_loader import InsectsDataLoader
from src.models.neural_net_multiclass import MulticlassNeuralNetModel

N_GRID = [25, 50, 100, 200, 400]
SEEDS = [0, 1, 2]
PARTICLE_METHODS = ["PF", "WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "n_sweep",
)


def _job(args):
    """1 (N, seed) ジョブ。報告区間で集計した軽量値を返す。"""
    N, seed, bp, bb, ba, best_sgd = args
    loader = InsectsDataLoader(
        IE.DATA_PATH, scale_fit_end=IE.SCALE_FIT_END, seed=IE.SEED)
    model = MulticlassNeuralNetModel(
        loader.n_features, IE.HIDDEN_DIM, loader.n_classes)

    (result_rows, acc, f1, ll, sample_positions, eval_mask,
     diagnostics) = IE.run_experiment(
        N, model, loader, bp, bb, ba, best_sgd,
        seed=seed, collect_diagnostics=True, verbose=False)

    by_m = {r["method"]: r for r in result_rows}
    out = {"N": N, "f1": {}, "ess": {}, "ess_ratio": {},
           "resample": {}, "t_step": {}}
    for m in PARTICLE_METHODS:
        h = diagnostics[m]
        out["f1"][m] = float(by_m[m]["macro_f1"])
        ess = np.asarray(h["ess"])[eval_mask]
        out["ess"][m] = float(ess.mean())
        out["ess_ratio"][m] = float(ess.mean() / N)
        out["resample"][m] = float(np.asarray(h["resampled"])[eval_mask].mean())
        out["t_step"][m] = float(np.asarray(h["t_step"])[eval_mask].mean())
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bp, bb, ba, best_sgd = IE.load_grid_search_params()[100]
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Particle-count (N) sweep — INSECTS (R2-2)")
    emit(f"  N grid={N_GRID}, seeds={SEEDS}")
    emit(f"  PF={bp}  WSPF-A={ba}  WSPF-B={bb}")
    emit(f"  (leak-free: scale fit + select [0,{IE.SELECT_END}), "
         f"report [{IE.REPORT_START},end))")
    emit("=" * 72)

    jobs = [(N, s, bp, bb, ba, best_sgd) for N in N_GRID for s in SEEDS]
    agg = {N: {k: {m: [] for m in PARTICLE_METHODS}
               for k in ("f1", "ess", "ess_ratio", "resample", "t_step")}
           for N in N_GRID}
    n_workers = min(os.cpu_count() or 1, len(jobs))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_job, jobs):
            N = r["N"]
            for k in ("f1", "ess", "ess_ratio", "resample", "t_step"):
                for m in PARTICLE_METHODS:
                    agg[N][k][m].append(r[k][m])
    emit(f"  elapsed {time.time() - t0:.0f}s")

    def mean(N, k, m):
        return float(np.mean(agg[N][k][m]))

    for m in PARTICLE_METHODS:
        emit(f"\n  [{m}]")
        emit(f"  {'N':>5s} {'macroF1':>8s} {'ESS':>9s} {'ESS/N':>7s} "
             f"{'resample%':>10s} {'t_step[ms]':>11s}")
        for N in N_GRID:
            emit(f"  {N:>5d} {mean(N, 'f1', m):>8.4f} "
                 f"{mean(N, 'ess', m):>9.2f} "
                 f"{mean(N, 'ess_ratio', m):>7.3f} "
                 f"{100 * mean(N, 'resample', m):>9.1f}% "
                 f"{1e3 * mean(N, 't_step', m):>11.3f}")

    csv_path = os.path.join(OUTPUT_DIR, "insects_n_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,method,macro_f1,ess,ess_ratio,resample_freq,t_step_ms\n")
        for N in N_GRID:
            for m in PARTICLE_METHODS:
                f.write(f"{N},{m},{mean(N, 'f1', m):.6f},"
                        f"{mean(N, 'ess', m):.6f},"
                        f"{mean(N, 'ess_ratio', m):.6f},"
                        f"{mean(N, 'resample', m):.6f},"
                        f"{1e3 * mean(N, 't_step', m):.6f}\n")

    # プロット: macro-F1 vs N と t_step vs N
    colors = {"PF": "#0072B2", "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    Ns = np.array(N_GRID)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in PARTICLE_METHODS:
        a1.plot(Ns, [mean(N, "f1", m) for N in N_GRID], "o-",
                color=colors[m], label=m, linewidth=1.6)
        a2.plot(Ns, [1e3 * mean(N, "t_step", m) for N in N_GRID], "o-",
                color=colors[m], label=m, linewidth=1.6)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks(Ns)
        a.set_xticklabels([str(n) for n in N_GRID])
        a.set_xlabel("number of particles N")
        a.grid(True, alpha=0.3, which="both")
        a.legend()
    a1.set_ylabel("macro-F1 (report region)")
    a1.set_title("macro-F1 vs N (INSECTS)")
    a2.set_yscale("log")
    a2.set_ylabel("time per update [ms]")
    a2.set_title("Wall-clock per update vs N (d=1286)")
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "insects_n_sweep.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "insects_n_sweep.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {png}")


if __name__ == "__main__":
    main()
