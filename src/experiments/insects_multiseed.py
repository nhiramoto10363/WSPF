#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSECTS 複数シード実験 (R1-14 / R2-1)

email_multiseed.py の鏡像 + 検定の拡充:
- vs PF に加えて vs SGD と WSPF-A vs WSPF-B の対応検定も出力
  (email 側にも同じ拡充を予定; 本スクリプトが先行実装)

シードは初期粒子・リサンプリング・SGD 初期化にのみ効く
(データストリームと標準化 fit は固定)。
選択シード 42 と評価シード 0-9 は disjoint。

並列化: シード単位で ProcessPoolExecutor。
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

import src.experiments.insects_experiment as IE
from src.data.insects_loader import InsectsDataLoader
from src.models.neural_net_multiclass import MulticlassNeuralNetModel

SEEDS_MULTI = list(range(10))
N_PARTICLES = 100
MAX_WORKERS = min(len(SEEDS_MULTI),
                  int(os.environ.get("NCPUS", os.cpu_count() or 1)))

OUTPUT_DIR = IE.OUTPUT_DIR
METRICS = [("accuracy", "acc"), ("macro_f1", "f1"), ("loglik", "loglik")]
TESTED = ["SGD", "PF", "WSPF-A", "WSPF-B"]

_G = {}


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    loader = InsectsDataLoader(
        IE.DATA_PATH, scale_fit_end=IE.SCALE_FIT_END, seed=IE.SEED)
    model = MulticlassNeuralNetModel(
        loader.n_features, IE.HIDDEN_DIM, loader.n_classes)
    _G["loader"], _G["model"] = loader, model
    _G["hp"] = IE.load_grid_search_params()[N_PARTICLES]


def _run_seed(seed):
    best_pf, best_wspf_b, best_wspf_a, best_sgd = _G["hp"]
    rows, *_ = IE.run_experiment(
        N_PARTICLES, _G["model"], _G["loader"],
        best_pf, best_wspf_b, best_wspf_a, best_sgd,
        seed=seed, collect_diagnostics=False, verbose=False)
    return seed, {r["method"]: r for r in rows}


def _paired_tests(emit, per_seed, metric_key, pairs):
    emit(f"\n  対応のある検定 ({metric_key}, {len(SEEDS_MULTI)} seeds):")
    emit(f"  {'comparison':<22s} {'mean(1)':>9s} {'mean(2)':>9s} "
         f"{'Δ':>9s} {'paired-t p':>11s} {'Wilcoxon p':>11s}")
    for m1, m2 in pairs:
        a = np.array([per_seed[s][m1][metric_key] for s in SEEDS_MULTI])
        b = np.array([per_seed[s][m2][metric_key] for s in SEEDS_MULTI])
        d = a - b
        tt = stats.ttest_rel(a, b)
        try:
            wc = stats.wilcoxon(a, b)
            wp = wc.pvalue
        except ValueError:
            wp = float("nan")
        emit(f"  {m1 + ' vs ' + m2:<22s} {a.mean():>9.4f} "
             f"{b.mean():>9.4f} {d.mean():>+9.4f} "
             f"{tt.pvalue:>11.4g} {wp:>11.4g}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_lines = []

    def emit(s=""):
        print(s)
        out_lines.append(s)

    _init_worker()  # HP 表示用 (strict check もここで走る)
    best_pf, best_wspf_b, best_wspf_a, best_sgd = _G["hp"]

    emit("=" * 72)
    emit(f"INSECTS multi-seed experiment (R1-14/R2-1)")
    emit(f"  N={N_PARTICLES}, seeds={SEEDS_MULTI}")
    emit(f"  SGD={best_sgd}")
    emit(f"  PF={best_pf}")
    emit(f"  WSPF-B={best_wspf_b}")
    emit(f"  WSPF-A={best_wspf_a}")
    emit(f"  (leak-free: scale fit + select [0,{IE.SELECT_END}), "
         f"report [{IE.REPORT_START},end))")
    emit("=" * 72)

    t0 = time.time()
    per_seed = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             initializer=_init_worker) as ex:
        for seed, rows in ex.map(_run_seed, SEEDS_MULTI):
            per_seed[seed] = rows
            emit(f"  seed {seed} done ({time.time() - t0:.0f}s)")

    emit(f"\n  {'Method':<10s} {'Acc mean±std':>16s} "
         f"{'macroF1 mean±std':>18s} {'LogLik mean±std':>18s}")
    for m in ["NoChange"] + TESTED:
        vals = {mk: np.array([per_seed[s][m][mk] for s in SEEDS_MULTI])
                for mk, _ in METRICS}
        emit(f"  {m:<10s} "
             f"{vals['accuracy'].mean():>8.4f}±{vals['accuracy'].std():.4f} "
             f"{vals['macro_f1'].mean():>9.4f}±{vals['macro_f1'].std():.4f} "
             f"{vals['loglik'].mean():>9.4f}±{vals['loglik'].std():.4f}")

    pairs = [("WSPF-A", "PF"), ("WSPF-B", "PF"),
             ("SGD", "PF"),
             ("WSPF-A", "SGD"), ("WSPF-B", "SGD"),
             ("WSPF-A", "WSPF-B")]
    for metric_key, _ in METRICS:
        _paired_tests(emit, per_seed, metric_key, pairs)

    # CSV (シードごとの生値)
    csv_path = os.path.join(OUTPUT_DIR, "insects_multiseed.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("seed,method,accuracy,macro_f1,loglik\n")
        for s in SEEDS_MULTI:
            for m in ["NoChange"] + TESTED:
                r = per_seed[s][m]
                fp.write(f"{s},{m},{r['accuracy']:.6f},"
                         f"{r['macro_f1']:.6f},{r['loglik']:.6f}\n")

    txt_path = os.path.join(OUTPUT_DIR, "insects_multiseed.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out_lines) + "\n")
    print(f"\n  saved: {txt_path}")
    print(f"  saved: {csv_path}")
    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
