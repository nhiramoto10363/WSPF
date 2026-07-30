#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
σ_obs 感度スイープ — GEFCom2014 Solar (R1-8 の実データ回帰側)

観測ノイズ SD σ_obs は回帰の観測モデル p(y|x,θ)=N(f_θ(x), σ_obs²) を規定し、
グリッドでは選択区間 SGD パイロットの残差 RMSE で 1 度だけ推定して固定する。
本スクリプトはその推定値まわりで σ_obs を乗数的にスイープし、報告区間 MSE
とメソッド順位が σ_obs の選択に頑健かを確認する (σ_obs 感度チェック)。
併せて WSPF 系の平均 ρ を σ_obs の関数として報告する (σ_obs が大きいほど
尤度が平坦になり ρ・補正の効き方が変わる)。

固定ハイパラ: (η, σcd, σ0) = PF 最良構成 (matched 設定と整合)、WSPF-A の
β = グリッド best。σ_obs のみをスイープ変数として全メソッド共通に与える。

出力:
  outputs/sigma_obs_sweep/
    - sigma_obs_sweep_gefcom.txt / .csv
    - sigma_obs_sweep_mse.png
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

import src.experiments.gefcom_experiment as GE
from src.experiments.gefcom_matched_hp import build_matched_hp
from src.data.gefcom_solar_loader import GefcomSolarLoader

# σ_obs をグリッド推定値まわりで乗数スイープ (1.0 = 推定値そのもの)
SIGMA_OBS_FACTORS = [0.5, 0.7, 1.0, 1.4, 2.0]
ZONE = 1                 # 選択ゾーン (σ_obs はここで推定された)
SEEDS = [0, 1, 2]
METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]
RHO_METHODS = ["WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "sigma_obs_sweep",
)

_G = {}


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _G["matched"] = build_matched_hp()
    _G["loader"] = GefcomSolarLoader(
        GE.PREDICTORS_PATH, zone=ZONE, train_path=GE.TRAIN_PATH,
        select_end_ts=GE.SELECT_END_TS)


def _run_job(args):
    sigma_obs, seed = args
    matched, beta, _, best_sgd, _base = _G["matched"]
    loader = _G["loader"]

    best_pf = matched
    best_wspf_b = dict(matched)
    best_wspf_a = {**matched, "beta": beta}
    (rows, mse, mae, window_ts, eval_mask, diag) = GE.run_experiment(
        loader, best_pf, best_wspf_b, best_wspf_a, best_sgd,
        sigma_obs, seed=seed, collect_diagnostics=True, verbose=False)

    out = {"sigma_obs": sigma_obs, "seed": seed, "mse": {}, "rho": {}}
    for r in rows:
        out["mse"][r["method"]] = float(r["mse"])
    for m in RHO_METHODS:
        h = diag[m]
        if "rho" in h:
            rho = np.asarray(h["rho"])
            rep = rho[eval_mask] if rho.shape[0] == eval_mask.shape[0] else rho
            out["rho"][m] = float(np.mean(rep))
        else:
            out["rho"][m] = float("nan")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched, beta, src, best_sgd, base_sigma = build_matched_hp()
    grid = [round(f * base_sigma, 5) for f in SIGMA_OBS_FACTORS]

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("σ_obs sensitivity sweep — GEFCom2014 Solar (R1-8)")
    emit(f"  fixed (η,σcd,σ0) = PF best {matched} [{src}], WSPF-A β={beta}")
    emit(f"  σ_obs base (grid estimate) = {base_sigma:.4f}")
    emit(f"  factors {SIGMA_OBS_FACTORS} → σ_obs grid = {grid}")
    emit(f"  zone={ZONE}, seeds={SEEDS}")
    emit("=" * 72)

    jobs = [(so, s) for so in grid for s in SEEDS]
    n_workers = min(os.cpu_count() or 1, len(jobs))
    emit(f"  workers={n_workers}, jobs={len(jobs)}")

    mse_by_so = {so: {m: [] for m in METHODS} for so in grid}
    rho_by_so = {so: {m: [] for m in RHO_METHODS} for so in grid}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as ex:
        for r in ex.map(_run_job, jobs):
            so = r["sigma_obs"]
            for m in METHODS:
                mse_by_so[so][m].append(r["mse"][m])
            for m in RHO_METHODS:
                rho_by_so[so][m].append(r["rho"][m])
    emit(f"  elapsed {time.time() - t0:.0f}s")

    # ---- 表 ----
    emit(f"\n{'=' * 72}\n  MSE vs σ_obs (report-region mean over seeds)\n"
         f"{'=' * 72}")
    header = (f"  {'factor':>7s} {'σ_obs':>8s} " +
              " ".join(f"{m:>9s}" for m in METHODS))
    emit(header)
    emit("  " + "-" * (len(header) - 2))
    mse_curve = {m: [] for m in METHODS}
    for f, so in zip(SIGMA_OBS_FACTORS, grid):
        row = f"  {f:>7.2f} {so:>8.4f} "
        for m in METHODS:
            mean = float(np.mean(mse_by_so[so][m]))
            mse_curve[m].append(mean)
            row += f"{mean:>9.5f} "
        emit(row)

    emit(f"\n  平均 ρ vs σ_obs (WSPF, report-region):")
    emit(f"  {'σ_obs':>8s} " + " ".join(f"{m:>9s}" for m in RHO_METHODS))
    rho_curve = {m: [] for m in RHO_METHODS}
    for so in grid:
        row = f"  {so:>8.4f} "
        for m in RHO_METHODS:
            rm = float(np.nanmean(rho_by_so[so][m]))
            rho_curve[m].append(rm)
            row += f"{rm:>9.4f} "
        emit(row)

    # ---- 感度サマリ (順位が σ_obs 全域で不変か) ----
    emit(f"\n  感度サマリ:")
    for m in ("WSPF-A", "WSPF-B"):
        wins = sum(1 for i in range(len(grid))
                   if mse_curve[m][i] <= mse_curve["PF"][i])
        emit(f"    {m} ≤ PF (MSE) を満たす σ_obs 点: {wins}/{len(grid)}")
    best_idx = {m: int(np.argmin(mse_curve[m])) for m in METHODS}
    emit(f"    各メソッド MSE 最小の σ_obs: "
         + ", ".join(f"{m}={grid[best_idx[m]]}" for m in METHODS))

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "sigma_obs_sweep_gefcom.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sigma_obs,factor,metric,method,value\n")
        for i, (fa, so) in enumerate(zip(SIGMA_OBS_FACTORS, grid)):
            for m in METHODS:
                f.write(f"{so},{fa},mse,{m},{mse_curve[m][i]:.6f}\n")
            for m in RHO_METHODS:
                f.write(f"{so},{fa},rho_mean,{m},{rho_curve[m][i]:.6f}\n")

    # ---- プロット ----
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    x = np.array(grid)
    plt.figure(figsize=(7, 5))
    for m in METHODS:
        plt.plot(x, mse_curve[m], "o-", color=colors[m], label=m, linewidth=1.6)
    plt.axvline(base_sigma, color="gray", ls="--", lw=0.9,
                label="grid estimate")
    plt.xlabel(r"$\sigma_{\mathrm{obs}}$")
    plt.ylabel("Test MSE (report region)")
    plt.title(r"MSE sensitivity to $\sigma_{\mathrm{obs}}$ "
              f"(GEFCom Solar, zone {ZONE})")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    mse_png = os.path.join(OUTPUT_DIR, "sigma_obs_sweep_mse.png")
    plt.savefig(mse_png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "sigma_obs_sweep_gefcom.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {mse_png}")


if __name__ == "__main__":
    main()
