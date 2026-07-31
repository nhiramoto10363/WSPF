#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
σ_cd 感度スイープ — GEFCom2014 Solar (R1-7 の主図・表)

sigma_obs_sweep_gefcom.py (構造) + sigma_cd_sweep_regression.py (出力形式) の
実データ回帰版。concept-drift ノイズ σ_cd を全メソッド共通でスイープし、
各メソッドの報告区間 MSE を σ_cd の関数として曲線化する。σ_cd は補正スケール
ρ = η²ŝ / (η²ŝ + σ_cd²) を規定するため、σ_cd が小さいほど ρ→1 (補正大・退化端)、
大きいほど ρ→0 (補正消失・全手法収束)。両端の漸近が見えて初めて「曲線」の
主張になる。

共有固定 HP (grid JSON から strict 読込):
  η = PF/B/SGD の共通 best (=0.05), σ0 = 0.1, β = 0.95, σ_obs = grid 推定値。
  ★WSPF-A の own-best η は 0.01 だが、matched 条件として共通 best η=0.05 に
    統一している (A に不利側で揃えている)。この点はヘッダに明記する。★
σ_cd のみをスイープ変数として全メソッドに同一値を与える。SGD は σ_cd 非依存
なので参照線として 1 度分だけ集計する。

出力 (outputs/sigma_cd_sweep/sigma_cd_sweep_gefcom.{txt,csv,png}):
- 表: σ_cd ごとの各手法 MSE mean±std、improvement over PF (%) と paired-t /
  Wilcoxon の p (A vs PF, B vs PF)、ρ の mean/max、ESS mean、resample%、
  非有限カウント
- CSV: sigma_cd, seed, method, mse, mae, rho_mean, rho_max, ess_mean,
  resample_frac, nonfinite (生値)
- 図: 上段 MSE vs σ_cd (x 対数, PF best 0.001 / WSPF best 0.005 に縦点線,
  SGD 水平線), 下段 ρ̄ vs σ_cd

符号規約: improvement over PF (%) = 100·(MSE_PF − MSE_alt)/MSE_PF。正 = 改善。

事前に grid_search_gefcom.py を実行しておくこと (strict loader)。
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
from scipy import stats

import src.experiments.gefcom_experiment as GE
from src.experiments.gefcom_matched_hp import build_matched_hp
from src.data.gefcom_solar_loader import GefcomSolarLoader

# 3 領域を写す: PF best 0.001 (ρ→1 退化端), WSPF best ~0.005 (交差予想),
# 大 σ_cd (ρ→0 補正消失・収束確認)
SIGMA_CD_GRID = [0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1]
PF_BEST_SIGMA = 0.001
WSPF_BEST_SIGMA = 0.005
ZONE = 1
SEEDS = list(range(10))
METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]
FILTER_METHODS = ["PF", "WSPF-A", "WSPF-B"]
RHO_METHODS = ["WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "sigma_cd_sweep",
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
    sigma_cd, seed = args
    matched, beta, _, best_sgd, noise_std = _G["matched"]
    loader = _G["loader"]

    best_pf = {**matched, "sigma_sys": sigma_cd}
    best_wspf_b = {**matched, "sigma_sys": sigma_cd}
    best_wspf_a = {**matched, "sigma_sys": sigma_cd, "beta": beta}
    (rows, mse, mae, window_ts, eval_mask, diag) = GE.run_experiment(
        loader, best_pf, best_wspf_b, best_wspf_a, best_sgd,
        noise_std, seed=seed, collect_diagnostics=True, verbose=False)

    by_m = {r["method"]: r for r in rows}
    out = {"sigma_cd": sigma_cd, "seed": seed, "mse": {}, "mae": {},
           "diag": {}}
    for m in METHODS:
        out["mse"][m] = float(by_m[m]["mse"])
        out["mae"][m] = float(by_m[m]["mae"])
    for m in FILTER_METHODS:
        h = diag[m]
        ess = np.asarray(h["ess"])[eval_mask]
        res = np.asarray(h["resampled"])[eval_mask]
        d = {"ess_mean": float(ess.mean()),
             "resample_frac": float(res.mean()),
             "rho_mean": float("nan"), "rho_max": float("nan"),
             "nonfinite": 0}
        if "rho" in h:
            rho = np.asarray(h["rho"])
            rep = rho[eval_mask] if rho.shape[0] == eval_mask.shape[0] else rho
            d["rho_mean"] = float(np.mean(rep))
            d["rho_max"] = float(np.max(rep))
        if "logcorr_nonfinite_count" in h:
            nf = np.asarray(h["logcorr_nonfinite_count"])[eval_mask]
            d["nonfinite"] = int(nf.sum())
        out["diag"][m] = d
    return out


def _improve_and_test(pf_vals, alt_vals):
    """improvement over PF (%) と対応検定 (正=改善)。"""
    pf = np.asarray(pf_vals, dtype=np.float64)
    alt = np.asarray(alt_vals, dtype=np.float64)
    imp = float(100.0 * (pf - alt).mean() / pf.mean()) if pf.mean() else 0.0
    d = pf - alt
    if len(d) >= 2 and np.any(d != 0):
        tp = float(stats.ttest_rel(pf, alt).pvalue)
        try:
            wp = float(stats.wilcoxon(pf, alt).pvalue)
        except ValueError:
            wp = float("nan")
    else:
        tp = wp = float("nan")
    return imp, tp, wp


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched, beta, src, best_sgd, noise_std = build_matched_hp()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("σ_cd sensitivity sweep — GEFCom2014 Solar (R1-7 main figure)")
    emit(f"  shared fixed HP: η={matched['eta']}, σ0={matched['prior_std']}, "
         f"β={beta}, σ_obs={noise_std:.4f}  [{src}]")
    emit(f"  ★matched 条件: WSPF-A own-best η=0.01 ではなく共通 best η="
         f"{matched['eta']} に統一 (A に不利側)★")
    emit(f"  σ_cd grid = {SIGMA_CD_GRID} "
         f"(PF best={PF_BEST_SIGMA}, WSPF best≈{WSPF_BEST_SIGMA})")
    emit(f"  zone={ZONE}, seeds={SEEDS} (SGD は σ_cd 非依存の参照)")
    emit("  改善符号: improvement over PF (%) = 100·(MSE_PF−MSE_alt)/MSE_PF, 正=改善")
    emit("=" * 72)

    jobs = [(sc, s) for sc in SIGMA_CD_GRID for s in SEEDS]
    n_workers = min(int(os.environ.get("NCPUS", os.cpu_count() or 1)),
                    len(jobs))
    emit(f"  workers={n_workers}, jobs={len(jobs)}")

    # 集計コンテナ
    mse = {sc: {m: [] for m in METHODS} for sc in SIGMA_CD_GRID}
    mae = {sc: {m: [] for m in METHODS} for sc in SIGMA_CD_GRID}
    diag = {sc: {m: {"ess_mean": [], "resample_frac": [], "rho_mean": [],
                     "rho_max": [], "nonfinite": []}
                 for m in FILTER_METHODS} for sc in SIGMA_CD_GRID}
    csv_raw = []

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as ex:
        for r in ex.map(_run_job, jobs):
            sc = r["sigma_cd"]
            for m in METHODS:
                mse[sc][m].append(r["mse"][m])
                mae[sc][m].append(r["mae"][m])
            for m in FILTER_METHODS:
                for k in diag[sc][m]:
                    diag[sc][m][k].append(r["diag"][m][k])
            for m in METHODS:
                dd = r["diag"].get(m, {})
                csv_raw.append(
                    f"{sc},{r['seed']},{m},{r['mse'][m]:.6f},{r['mae'][m]:.6f},"
                    f"{dd.get('rho_mean', float('nan'))},"
                    f"{dd.get('rho_max', float('nan'))},"
                    f"{dd.get('ess_mean', float('nan'))},"
                    f"{dd.get('resample_frac', float('nan'))},"
                    f"{dd.get('nonfinite', '')}")
    emit(f"  elapsed {time.time() - t0:.0f}s")

    # SGD 参照 (σ_cd 非依存: 全 σ_cd で同値。seed 平均)
    sgd_vals = np.array(mse[SIGMA_CD_GRID[0]]["SGD"])
    emit(f"\n  SGD 参照 (σ_cd 非依存): MSE = {sgd_vals.mean():.5f}±"
         f"{sgd_vals.std():.5f}")

    # ---- MSE 表 + 検定 ----
    emit(f"\n{'=' * 72}\n  MSE vs σ_cd + improvement over PF (%) と対応検定\n"
         f"{'=' * 72}")
    emit(f"  {'σ_cd':>7s} {'PF':>16s} {'WSPF-A':>16s} {'impA%':>7s} "
         f"{'pA(t)':>8s} {'WSPF-B':>16s} {'impB%':>7s} {'pB(t)':>8s}")
    mse_curve = {m: [] for m in METHODS}
    for sc in SIGMA_CD_GRID:
        for m in METHODS:
            mse_curve[m].append(float(np.mean(mse[sc][m])))
        pf = np.array(mse[sc]["PF"])
        a = np.array(mse[sc]["WSPF-A"])
        b = np.array(mse[sc]["WSPF-B"])
        impA, pAt, pAw = _improve_and_test(pf, a)
        impB, pBt, pBw = _improve_and_test(pf, b)
        emit(f"  {sc:>7.3f} {pf.mean():>8.5f}±{pf.std():.5f} "
             f"{a.mean():>8.5f}±{a.std():.5f} {impA:>+6.2f} {pAt:>8.3g} "
             f"{b.mean():>8.5f}±{b.std():.5f} {impB:>+6.2f} {pBt:>8.3g}")

    # ---- ρ / ESS / resample 表 ----
    emit(f"\n{'=' * 72}\n  ρ / ESS / resample vs σ_cd (WSPF, report region)\n"
         f"{'=' * 72}")
    emit(f"  {'σ_cd':>7s} {'method':<8s} {'ρ_mean':>8s} {'ρ_max':>8s} "
         f"{'ESS':>8s} {'resamp%':>8s} {'nonfin':>7s}")
    rho_curve = {m: [] for m in RHO_METHODS}
    for sc in SIGMA_CD_GRID:
        for m in FILTER_METHODS:
            ess = float(np.mean(diag[sc][m]["ess_mean"]))
            rf = 100.0 * float(np.mean(diag[sc][m]["resample_frac"]))
            nf = int(np.sum(diag[sc][m]["nonfinite"]))
            # PF は補正を持たないため ρ は N/A
            has_rho = m in RHO_METHODS and np.any(
                np.isfinite(diag[sc][m]["rho_mean"]))
            if has_rho:
                rm = float(np.nanmean(diag[sc][m]["rho_mean"]))
                rx = float(np.nanmax(diag[sc][m]["rho_max"]))
                rho_curve[m].append(rm)
                emit(f"  {sc:>7.3f} {m:<8s} {rm:>8.4f} {rx:>8.4f} "
                     f"{ess:>8.2f} {rf:>7.2f}% {nf:>7d}")
            else:
                emit(f"  {sc:>7.3f} {m:<8s} {'-':>8s} {'-':>8s} "
                     f"{ess:>8.2f} {rf:>7.2f}% {nf:>7d}")

    # ---- CSV (生値) ----
    csv_path = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_gefcom.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("sigma_cd,seed,method,mse,mae,rho_mean,rho_max,"
                "ess_mean,resample_frac,nonfinite\n")
        f.write("\n".join(csv_raw) + "\n")

    # ---- 図: 上段 MSE, 下段 ρ ----
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    x = np.array(SIGMA_CD_GRID)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)
    for m in FILTER_METHODS:
        ax1.plot(x, mse_curve[m], "o-", color=colors[m], label=m, linewidth=1.7)
    ax1.axhline(sgd_vals.mean(), color=colors["SGD"], ls="-.", lw=1.3,
                label="SGD (σ_cd-indep.)")
    ax1.axvline(PF_BEST_SIGMA, color=colors["PF"], ls=":", lw=1.0)
    ax1.axvline(WSPF_BEST_SIGMA, color=colors["WSPF-B"], ls=":", lw=1.0)
    ax1.text(PF_BEST_SIGMA, ax1.get_ylim()[1], " PF best", color=colors["PF"],
             fontsize=7, va="top")
    ax1.text(WSPF_BEST_SIGMA, ax1.get_ylim()[1], " WSPF best",
             color=colors["WSPF-B"], fontsize=7, va="top")
    ax1.set_xscale("log")
    ax1.set_ylabel("Test MSE (report region)")
    ax1.set_title(f"MSE sensitivity to $\\sigma_{{cd}}$ (GEFCom Solar zone {ZONE})")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()

    for m in RHO_METHODS:
        ax2.plot(x, rho_curve[m], "o-", color=colors[m], label=f"{m} ρ̄",
                 linewidth=1.7)
    ax2.axvline(PF_BEST_SIGMA, color=colors["PF"], ls=":", lw=1.0)
    ax2.axvline(WSPF_BEST_SIGMA, color=colors["WSPF-B"], ls=":", lw=1.0)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel(r"$\sigma_{\mathrm{cd}}$")
    ax2.set_ylabel(r"mean $\rho$ (report region)")
    ax2.set_title(r"$\rho$ (correction scale) vs $\sigma_{\mathrm{cd}}$")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_gefcom.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "sigma_cd_sweep_gefcom.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {png}")


if __name__ == "__main__":
    main()
