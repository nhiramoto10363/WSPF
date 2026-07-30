#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マッチド・ハイパーパラメータ実験 — GEFCom2014 Solar (R1-7 の実データ回帰側)

matched_hp_regression.py / matched_hp_email.py の GEFCom 版。PF / WSPF-A /
WSPF-B に **同一の (η, σcd, σ0)** を与えて比較し、性能差が「ハイパラ最適化の
差」ではなく「補正項そのものの効果」であることを分離する。共通ハイパラ =
グリッドの PF 最良構成 (best_pf) を全メソッドに適用 (WSPF-A の β は補正固有
なのでグリッド best_wspf_a[β] を使用)。

gefcom_experiment.run_experiment を再利用し、10 シード × 3 ゾーンで報告区間
MSE/MAE を集計、ゾーンごとに PF に対する対応検定を報告する (σ_obs は
グリッドで推定した値を固定)。

出力:
  outputs/matched_hp/
    - gefcom_matched_hp.txt / .csv

事前に grid_search_gefcom.py を実行しておくこと (strict loader)。
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

import src.experiments.gefcom_experiment as GE
from src.data.gefcom_solar_loader import GefcomSolarLoader

SEEDS_MULTI = list(range(10))
MAX_WORKERS = int(os.environ.get("NCPUS", os.cpu_count() or 1))
METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "matched_hp",
)

_G = {}


def build_matched_hp():
    """PF 最良構成を共通ハイパラとして構築する (strict)。

    Returns
    -------
    matched : dict  {eta, sigma_sys, prior_std}  (PF best)
    beta : float                                  (WSPF-A best β)
    src : str
    best_sgd : dict                               (SGD 独自 best, matched 対象外)
    noise_std : float                             (σ_obs, グリッド推定値)
    """
    best_pf, best_wspf_b, best_wspf_a, best_sgd, noise_std = \
        GE.load_grid_search_params()  # strict: 未実行・欠損は例外
    beta = best_wspf_a["beta"]
    matched = {"eta": best_pf["eta"],
               "sigma_sys": best_pf["sigma_sys"],
               "prior_std": best_pf["prior_std"]}
    return matched, beta, "grid best_pf", best_sgd, noise_std


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _G["matched"] = build_matched_hp()
    _G["loaders"] = {}


def _run_job(args):
    zone, seed = args
    matched, beta, _, best_sgd, noise_std = _G["matched"]
    if zone not in _G["loaders"]:
        _G["loaders"][zone] = GefcomSolarLoader(
            GE.PREDICTORS_PATH, zone=zone, train_path=GE.TRAIN_PATH,
            select_end_ts=GE.SELECT_END_TS)
    loader = _G["loaders"][zone]

    best_pf = matched
    best_wspf_b = dict(matched)
    best_wspf_a = {**matched, "beta": beta}
    rows, *_ = GE.run_experiment(
        loader, best_pf, best_wspf_b, best_wspf_a, best_sgd,
        noise_std, seed=seed, collect_diagnostics=False, verbose=False)
    return zone, seed, {r["method"]: r for r in rows}


def paired(a_pf, a_alt):
    """PF と代替の per-seed report-MSE 差に対する対応検定。"""
    a_pf = np.asarray(a_pf, dtype=np.float64)
    a_alt = np.asarray(a_alt, dtype=np.float64)
    d = a_pf - a_alt  # 正なら alt の MSE が小 (改善)
    imp = float(100.0 * d.mean() / a_pf.mean()) if a_pf.mean() != 0 else 0.0
    if len(d) >= 2 and np.any(d != 0):
        tp = float(stats.ttest_rel(a_pf, a_alt).pvalue)
        try:
            wp = float(stats.wilcoxon(a_pf, a_alt).pvalue)
        except ValueError:
            wp = float("nan")
    else:
        tp = wp = float("nan")
    return imp, tp, wp


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    matched, beta, src, best_sgd, noise_std = build_matched_hp()
    emit("=" * 72)
    emit("Matched-HP experiment — GEFCom2014 Solar (R1-7)")
    emit("  共通ハイパラ = PF 最良構成を全メソッドに適用 (leak-free pipeline)")
    emit(f"  N={GE.N_PARTICLES}, zones={GE.ZONES}, seeds={SEEDS_MULTI}")
    emit(f"  matched (η,σcd,σ0)={matched} [{src}], WSPF-A β={beta}")
    emit(f"  SGD(独立)={best_sgd}, σ_obs={noise_std:.4f}")
    emit(f"  (HP selected on zone {GE.ZONES[0]}, report = ts >= "
         f"{GE.SELECT_END_TS})")
    emit("=" * 72)

    jobs = [(z, s) for z in GE.ZONES for s in SEEDS_MULTI]
    t0 = time.time()
    per_zone_seed = {z: {} for z in GE.ZONES}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             initializer=_init_worker) as ex:
        for zone, seed, rows in ex.map(_run_job, jobs):
            per_zone_seed[zone][seed] = rows
            emit(f"  zone {zone} seed {seed} done ({time.time() - t0:.0f}s)")

    csv_rows = [("zone", "method", "mse_mean", "mse_std", "mae_mean",
                 "improve_vs_pf_pct", "paired_t_p", "wilcoxon_p")]
    for zone in GE.ZONES:
        per_seed = per_zone_seed[zone]
        emit(f"\n===== Zone {zone} — Matched-HP (report region) =====")
        emit(f"  {'Method':<10s} {'MSE mean±std':>20s} "
             f"{'MAE mean':>10s} {'vs PF %':>9s} {'paired-t p':>11s} "
             f"{'Wilcoxon p':>11s}")
        pf_mse = [per_seed[s]["PF"]["mse"] for s in SEEDS_MULTI]
        for m in METHODS:
            v_mse = np.array([per_seed[s][m]["mse"] for s in SEEDS_MULTI])
            v_mae = np.array([per_seed[s][m]["mae"] for s in SEEDS_MULTI])
            if m in ("WSPF-A", "WSPF-B"):
                imp, tp, wp = paired(pf_mse, v_mse)
                emit(f"  {m:<10s} {v_mse.mean():>11.5f}±{v_mse.std():.5f} "
                     f"{v_mae.mean():>10.5f} {imp:>8.2f}% "
                     f"{tp:>11.4g} {wp:>11.4g}")
                csv_rows.append((zone, m, f"{v_mse.mean():.6f}",
                                 f"{v_mse.std():.6f}", f"{v_mae.mean():.6f}",
                                 f"{imp:.4f}", f"{tp:.6g}", f"{wp:.6g}"))
            else:
                mark = "  (baseline)" if m == "PF" else ""
                emit(f"  {m:<10s} {v_mse.mean():>11.5f}±{v_mse.std():.5f} "
                     f"{v_mae.mean():>10.5f} {'-':>9s} {'-':>11s} "
                     f"{'-':>11s}{mark}")
                csv_rows.append((zone, m, f"{v_mse.mean():.6f}",
                                 f"{v_mse.std():.6f}", f"{v_mae.mean():.6f}",
                                 "", "", ""))

    txt_path = os.path.join(OUTPUT_DIR, "gefcom_matched_hp.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out_lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "gefcom_matched_hp.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        for row in csv_rows:
            fp.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
