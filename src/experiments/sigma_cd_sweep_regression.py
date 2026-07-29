#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
σ_cd 感度スイープ — Regression (R1-8)

concept-drift ノイズ σ_cd を全メソッド共通でスイープし、各メソッドの
MSE を σ_cd の関数として曲線化する。併せて WSPF 系の ρ 分布と
クリッピング頻度(ρ≥0.999 到達数, log_correction クランプ発動数)を
σ_cd の関数として報告する。

ρ = η²ŝ / (η²ŝ + σ_cd²) なので、σ_cd が小さいほど ρ→1(補正大・クリップ
発動)、大きいほど ρ→0(補正小)。この感度が補正の効きどころとロバスト性
を示す(R1-8)。

固定ハイパラ: (η, σ0) = PF 最良構成(タスク4のマッチド設定と整合)、
WSPF-A の β = グリッド best。σ_cd のみをスイープ変数として全メソッドに
同一値を与える。

出力:
  outputs/sigma_cd_sweep/
    - sigma_cd_sweep_regression.txt / .csv
    - sigma_cd_sweep_mse.png
    - sigma_cd_sweep_rho.png
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

# σ_cd スイープ(グリッド [0.01..0.2] を含み、ρ 飽和域を見るため下方に拡張)
SIGMA_CD_GRID = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
N_PARTICLES = 100
METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]
RHO_METHODS = ["WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "sigma_cd_sweep",
)


def _eval_slice(arr):
    return np.asarray(arr)[EVAL_START:]


def run_one(args):
    """(sigma_cd, seed) を 1 ジョブとして実行。"""
    sigma_cd, seed, n_p, matched, beta = args
    best_pf = {**matched, "sigma_sys": sigma_cd}
    best_wspf_b = {**matched, "sigma_sys": sigma_cd}
    best_wspf_a = {**matched, "sigma_sys": sigma_cd, "beta": beta}
    res = run_single(seed, n_p, best_pf, best_wspf_b, best_wspf_a)

    out = {"sigma_cd": sigma_cd, "seed": seed, "mse": {}}
    for m in METHODS:
        out["mse"][m] = float(_eval_slice(res["mse"][m]).mean())
    # ρ / クリップ統計(eval region)
    out["rho"] = {}
    for m in RHO_METHODS:
        h = res["histories"][m]
        rho_eval = np.asarray(h["rho"])[EVAL_START:]        # (T_eval, N)
        out["rho"][m] = {
            "rho_mean": float(rho_eval.mean()),
            "rho_max": float(rho_eval.max()),
            "rho_clip_count": int(np.asarray(h["rho_clip_count"])[EVAL_START:].sum()),
            "logcorr_clip_count": int(np.asarray(h["logcorr_clip_count"])[EVAL_START:].sum()),
            "n_particle_steps": int(rho_eval.size),
        }
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched, beta, src = build_matched_hp()[N_PARTICLES]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("σ_cd sensitivity sweep — Regression (R1-8)")
    emit(f"  fixed (η,σ0) = PF best {matched} [{src}], WSPF-A β={beta}")
    emit(f"  σ_cd grid = {SIGMA_CD_GRID}")
    emit(f"  N={N_PARTICLES}, T={T}, seeds={len(SEEDS)}, EVAL_START={EVAL_START}")
    emit("=" * 72)

    jobs = [(sc, seed, N_PARTICLES, matched, beta)
            for sc in SIGMA_CD_GRID for seed in SEEDS]
    n_workers = min(os.cpu_count() or 1, 48)
    emit(f"  workers={n_workers}, jobs={len(jobs)}")

    # 集計コンテナ
    mse_by_sc = {sc: {m: [] for m in METHODS} for sc in SIGMA_CD_GRID}
    rho_by_sc = {sc: {m: {"rho_mean": [], "rho_max": [],
                          "rho_clip_count": [], "logcorr_clip_count": [],
                          "n_particle_steps": []}
                      for m in RHO_METHODS} for sc in SIGMA_CD_GRID}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(run_one, a) for a in jobs]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            sc = r["sigma_cd"]
            for m in METHODS:
                mse_by_sc[sc][m].append(r["mse"][m])
            for m in RHO_METHODS:
                for k, v in r["rho"][m].items():
                    rho_by_sc[sc][m][k].append(v)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                emit(f"  [{done}/{len(jobs)}] done")
    emit(f"  elapsed {time.time()-t0:.0f}s")

    # ---- 表 ----
    emit(f"\n{'='*72}\n  MSE vs σ_cd (eval-region mean over seeds)\n{'='*72}")
    header = f"  {'σ_cd':>7s} " + " ".join(f"{m:>9s}" for m in METHODS)
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    mse_curve = {m: [] for m in METHODS}
    for sc in SIGMA_CD_GRID:
        row = f"  {sc:>7.3f} "
        for m in METHODS:
            mean = float(np.mean(mse_by_sc[sc][m]))
            mse_curve[m].append(mean)
            row += f"{mean:>9.4f} "
        emit(row)

    emit(f"\n{'='*72}\n  ρ 分布とクリップ頻度 vs σ_cd (WSPF, eval-region)\n{'='*72}")
    # ρ≥.999(ρクリップ到達数)と ρclip%(その割合)を隣接させ、
    # log_correction 非有限ガード数(nonfinite)は別列にして誤読を防ぐ。
    emit(f"  {'σ_cd':>7s} {'method':<8s} {'ρ_mean':>8s} {'ρ_max':>8s} "
         f"{'ρ≥.999':>9s} {'ρclip%':>8s} {'nonfinite':>10s}")
    rho_curve = {m: {"rho_mean": [], "rho_clip_frac": []} for m in RHO_METHODS}
    for sc in SIGMA_CD_GRID:
        for m in RHO_METHODS:
            rm = float(np.mean(rho_by_sc[sc][m]["rho_mean"]))
            rx = float(np.max(rho_by_sc[sc][m]["rho_max"]))
            clip = int(np.sum(rho_by_sc[sc][m]["rho_clip_count"]))
            lclip = int(np.sum(rho_by_sc[sc][m]["logcorr_clip_count"]))
            nps = int(np.sum(rho_by_sc[sc][m]["n_particle_steps"]))
            frac = 100.0 * clip / nps if nps else 0.0
            rho_curve[m]["rho_mean"].append(rm)
            rho_curve[m]["rho_clip_frac"].append(frac)
            emit(f"  {sc:>7.3f} {m:<8s} {rm:>8.4f} {rx:>8.4f} "
                 f"{clip:>9d} {frac:>7.2f}% {lclip:>10d}")

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_regression.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sigma_cd,metric,method,value\n")
        for i, sc in enumerate(SIGMA_CD_GRID):
            for m in METHODS:
                f.write(f"{sc},mse,{m},{mse_curve[m][i]:.6f}\n")
            for m in RHO_METHODS:
                f.write(f"{sc},rho_mean,{m},{rho_curve[m]['rho_mean'][i]:.6f}\n")
                f.write(f"{sc},rho_clip_frac_pct,{m},"
                        f"{rho_curve[m]['rho_clip_frac'][i]:.6f}\n")

    # ---- プロット ----
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    x = np.array(SIGMA_CD_GRID)

    plt.figure(figsize=(7, 5))
    for m in METHODS:
        plt.plot(x, mse_curve[m], "o-", color=colors[m], label=m, linewidth=1.6)
    plt.xscale("log")
    plt.xlabel(r"$\sigma_{\mathrm{cd}}$")
    plt.ylabel("Test MSE (eval region)")
    plt.title(r"MSE sensitivity to $\sigma_{\mathrm{cd}}$ (Regression)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    mse_png = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_mse.png")
    plt.savefig(mse_png, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 5))
    for m in RHO_METHODS:
        ax1.plot(x, rho_curve[m]["rho_mean"], "o-", color=colors[m],
                 label=f"{m} ρ_mean", linewidth=1.6)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\sigma_{\mathrm{cd}}$")
    ax1.set_ylabel(r"mean $\rho$")
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    for m in RHO_METHODS:
        ax2.plot(x, rho_curve[m]["rho_clip_frac"], "s--", color=colors[m],
                 alpha=0.6, label=f"{m} clip%")
    ax2.set_ylabel(r"$\rho\geq0.999$ clip fraction [%]")
    ax1.grid(True, alpha=0.3, which="both")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="best")
    plt.title(r"$\rho$ distribution & clipping vs $\sigma_{\mathrm{cd}}$")
    plt.tight_layout()
    rho_png = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_rho.png")
    plt.savefig(rho_png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_regression.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {mse_png}")
    emit(f"Saved: {rho_png}")


if __name__ == "__main__":
    main()
