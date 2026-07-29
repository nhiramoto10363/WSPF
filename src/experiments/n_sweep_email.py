#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子数 N スイープ — Email (R2-2, 補助)

N∈{25,50,100,200,400} で email 実験を走らせ、F1・ESS(ESS/N)・リサンプリング
頻度・1更新あたり実行時間の N 依存性を報告する(報告区間=期3-5)。
各メソッドは自身のグリッド best HP を使用(email_multiseed と同じ方針)。
F1 のシード分散を抑えるため少数シードで平均する。

注) email グリッドはリーク除去後に再実行が前提(HP は暫定)。N 依存の
    ESS/resample/time はシード非依存に近く、機構の N スケーリングを示す。

出力:
  outputs/n_sweep/
    - n_sweep_email.txt / .csv / n_sweep_email.png
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

import src.experiments.email_binary_experiment as EB
from src.experiments.email_multiseed import load_hp
from src.data import EmailDataLoader
from src.models.neural_net import (
    NeuralNetModel, create_nn_grad_fn, create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)

N_GRID = [25, 50, 100, 200, 400]
SEEDS = [0, 1, 2]
PARTICLE_METHODS = ["PF", "WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "n_sweep",
)


def _job(args):
    """1 (N, seed) ジョブ。報告区間で集計した軽量値を返す。"""
    N, seed, bp, bb, ba, best_sgd = args
    loader = EmailDataLoader(EB.DATA_PATH, n_components=EB.PCA_DIM,
                             seed=EB.SEED, pca_fit_end=EB.PCA_FIT_END)
    model = NeuralNetModel(input_dim=loader.input_dim, hidden_dim=EB.HIDDEN_DIM,
                           output_dim=1, activation="tanh")
    param_dim = model.param_dim
    grad_raw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)

    def grad_fn(p, X, y):
        return EB.clip_gradients(grad_raw(p, X, y), EB.MAX_GRAD_NORM)

    (result_rows, acc, f1, ll, methods, eval_start,
     sample_pos, diagnostics) = EB.run_experiment(
        n_particles=N, model=model, loader=loader, grad_fn=grad_fn,
        loglik_fn=loglik_fn, ps_grad_fn=ps_grad_fn,
        best_pf=bp, best_wspf_b=bb, best_wspf_a=ba,
        sgd_eta=best_sgd["eta"], sgd_prior=best_sgd["prior_std"], param_dim=param_dim,
        seed=seed)

    by_m = {r["method"]: r for r in result_rows}
    mask = (np.asarray(sample_pos) + EB.BATCH_SIZE) >= EB.REPORT_START
    out = {"N": N, "f1": {}, "ess": {}, "ess_ratio": {},
           "resample": {}, "t_step": {}}
    for m in PARTICLE_METHODS:
        h = diagnostics[m]
        out["f1"][m] = float(by_m[m]["f1"])
        ess = np.asarray(h["ess"])[mask]
        out["ess"][m] = float(ess.mean())
        out["ess_ratio"][m] = float(ess.mean() / N)
        out["resample"][m] = float(np.asarray(h["resampled"])[mask].mean())
        out["t_step"][m] = float(np.asarray(h["t_step"])[mask].mean())
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bp, bb, ba, best_sgd, src = load_hp(100)
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 72)
    emit("Particle-count (N) sweep — Email (R2-2)")
    emit(f"  N grid={N_GRID}, seeds={SEEDS}, HP src={src}")
    emit(f"  PF={bp}  WSPF-A={ba}  WSPF-B={bb}")
    emit(f"  (leak-free: PCA fit 期1-2, report 期3-5)")
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
    emit(f"  elapsed {time.time()-t0:.0f}s")

    def mean(N, k, m):
        return float(np.mean(agg[N][k][m]))

    for m in PARTICLE_METHODS:
        emit(f"\n  [{m}]")
        emit(f"  {'N':>5s} {'F1':>8s} {'ESS':>9s} {'ESS/N':>7s} "
             f"{'resample%':>10s} {'t_step[ms]':>11s}")
        for N in N_GRID:
            emit(f"  {N:>5d} {mean(N,'f1',m):>8.4f} {mean(N,'ess',m):>9.2f} "
                 f"{mean(N,'ess_ratio',m):>7.3f} {100*mean(N,'resample',m):>9.1f}% "
                 f"{1e3*mean(N,'t_step',m):>11.3f}")

    csv_path = os.path.join(OUTPUT_DIR, "n_sweep_email.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,method,f1,ess,ess_ratio,resample_freq,t_step_ms\n")
        for N in N_GRID:
            for m in PARTICLE_METHODS:
                f.write(f"{N},{m},{mean(N,'f1',m):.6f},{mean(N,'ess',m):.6f},"
                        f"{mean(N,'ess_ratio',m):.6f},{mean(N,'resample',m):.6f},"
                        f"{1e3*mean(N,'t_step',m):.6f}\n")

    # プロット: F1 vs N と t_step vs N
    colors = {"PF": "#0072B2", "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    Ns = np.array(N_GRID)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in PARTICLE_METHODS:
        a1.plot(Ns, [mean(N, "f1", m) for N in N_GRID], "o-",
                color=colors[m], label=m, linewidth=1.6)
        a2.plot(Ns, [1e3*mean(N, "t_step", m) for N in N_GRID], "o-",
                color=colors[m], label=m, linewidth=1.6)
    for a in (a1, a2):
        a.set_xscale("log"); a.set_xticks(Ns)
        a.set_xticklabels([str(n) for n in N_GRID])
        a.set_xlabel("number of particles N"); a.grid(True, alpha=0.3, which="both")
        a.legend()
    a1.set_ylabel("F1 (report region)"); a1.set_title("F1 vs N (Email)")
    a2.set_yscale("log")
    a2.set_ylabel("time per update [ms]"); a2.set_title("Wall-clock per update vs N")
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "n_sweep_email.png")
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "n_sweep_email.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {png}")


if __name__ == "__main__":
    main()
