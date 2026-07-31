#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEFCom2014 Solar 複数シード実験 (R1-14 / R2-1)

- 10 シード × 3 ゾーン。シードは初期粒子・リサンプリング・SGD 初期化
  のみに効く (ストリーム・標準化・σ_obs は固定)。
- 対応検定: ゾーンごとに vs PF / vs SGD / A-vs-B (insects_multiseed と
  同じ拡充版)。
- 並列化: (zone, seed) 単位。
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

OUTPUT_DIR = GE.OUTPUT_DIR
TESTED = ["SGD", "PF", "WSPF-A", "WSPF-B"]
PAIRS = [("WSPF-A", "PF"), ("WSPF-B", "PF"), ("SGD", "PF"),
         ("WSPF-A", "SGD"), ("WSPF-B", "SGD"), ("WSPF-A", "WSPF-B")]

_G = {}


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _G["hp"] = GE.load_grid_search_params()
    _G["loaders"] = {}


def _run_job(args):
    zone, seed = args
    if zone not in _G["loaders"]:
        _G["loaders"][zone] = GefcomSolarLoader(
            GE.PREDICTORS_PATH, zone=zone, train_path=GE.TRAIN_PATH,
            select_end_ts=GE.SELECT_END_TS)
    loader = _G["loaders"][zone]
    best_pf, best_wspf_b, best_wspf_a, best_sgd, noise_std = _G["hp"]
    rows, *_ = GE.run_experiment(
        loader, best_pf, best_wspf_b, best_wspf_a, best_sgd,
        noise_std, seed=seed, collect_diagnostics=False, verbose=False)
    return zone, seed, {r["method"]: r for r in rows}


def _paired_tests(emit, per_seed, metric_key):
    emit(f"\n  対応のある検定 ({metric_key}, "
         f"{len(SEEDS_MULTI)} seeds):")
    emit(f"  {'comparison':<22s} {'mean(1)':>9s} {'mean(2)':>9s} "
         f"{'Δ':>10s} {'paired-t p':>11s} {'Wilcoxon p':>11s}")
    for m1, m2 in PAIRS:
        a = np.array([per_seed[s][m1][metric_key] for s in SEEDS_MULTI])
        b = np.array([per_seed[s][m2][metric_key] for s in SEEDS_MULTI])
        tt = stats.ttest_rel(a, b)
        try:
            wp = stats.wilcoxon(a, b).pvalue
        except ValueError:
            wp = float("nan")
        emit(f"  {m1 + ' vs ' + m2:<22s} {a.mean():>9.5f} "
             f"{b.mean():>9.5f} {(a - b).mean():>+10.5f} "
             f"{tt.pvalue:>11.4g} {wp:>11.4g}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    _init_worker()
    best_pf, best_wspf_b, best_wspf_a, best_sgd, noise_std = _G["hp"]
    emit("=" * 72)
    emit("GEFCom2014 Solar multi-seed experiment (R1-14/R2-1)")
    emit(f"  N={GE.N_PARTICLES}, zones={GE.ZONES}, seeds={SEEDS_MULTI}")
    emit(f"  SGD={best_sgd}  PF={best_pf}")
    emit(f"  WSPF-B={best_wspf_b}")
    emit(f"  WSPF-A={best_wspf_a}")
    emit(f"  noise_std={noise_std:.4f}")
    emit("=" * 72)

    jobs = [(z, s) for z in GE.ZONES for s in SEEDS_MULTI]
    t0 = time.time()
    per_zone_seed = {z: {} for z in GE.ZONES}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             initializer=_init_worker) as ex:
        for zone, seed, rows in ex.map(_run_job, jobs):
            per_zone_seed[zone][seed] = rows
            emit(f"  zone {zone} seed {seed} done "
                 f"({time.time() - t0:.0f}s)")

    csv_rows = []
    for zone in GE.ZONES:
        per_seed = per_zone_seed[zone]
        emit(f"\n===== Zone {zone} =====")
        emit(f"  {'Method':<10s} {'MSE mean±std':>20s} "
             f"{'MAE mean±std':>20s}")
        for m in TESTED:
            v_mse = np.array([per_seed[s][m]["mse"]
                              for s in SEEDS_MULTI])
            v_mae = np.array([per_seed[s][m]["mae"]
                              for s in SEEDS_MULTI])
            emit(f"  {m:<10s} "
                 f"{v_mse.mean():>11.5f}±{v_mse.std():.5f} "
                 f"{v_mae.mean():>11.5f}±{v_mae.std():.5f}")
            for s in SEEDS_MULTI:
                csv_rows.append(
                    f"{zone},{s},{m},{per_seed[s][m]['mse']:.6f},"
                    f"{per_seed[s][m]['mae']:.6f}")
        _paired_tests(emit, per_seed, "mse")

    csv_path = os.path.join(OUTPUT_DIR, "gefcom_multiseed.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("zone,seed,method,mse,mae\n")
        fp.write("\n".join(csv_rows) + "\n")

    txt_path = os.path.join(OUTPUT_DIR, "gefcom_multiseed.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out_lines) + "\n")
    print(f"\n  saved: {txt_path}")
    print(f"  saved: {csv_path}")
    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
