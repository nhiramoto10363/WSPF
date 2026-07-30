#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データストリーム系ベースライン比較 — INSECTS (R2-4)

baselines_email.py の多クラス鏡像。概念ドリフト学習の代表的ベースラインを
追加する:
  - PH-SGD     : Page-Hinkley でドリフト検知し θ をリセット(drift-detector reset)
  - Window-SGD : 直近 W バッチのみで学習(adaptive sliding window)

既存の SGD / PF / WSPF-A / WSPF-B / NoChange (insects_experiment.run_experiment)
と同一データ・同一評価窓 (リークフリー: 標準化 fit + HP 選択 [0,SELECT_END)、
報告 [REPORT_START,end)) で比較し、複数シードで macro-F1 / Accuracy を集計、
PF に対する対応のある検定を報告する。

出力:
  outputs/baselines/
    - insects_baselines.txt / .csv
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

import src.experiments.insects_experiment as IE
from src.data.insects_loader import InsectsDataLoader
from src.models.neural_net_multiclass import (
    MulticlassNeuralNetModel, create_mc_grad_fn,
)
from src.baselines import PageHinkley, DriftResetSGD, WindowSGD

SEEDS_MULTI = list(range(10))
N_PARTICLES = 100
BASE_METHODS = ["PH-SGD", "Window-SGD"]
WINDOW = 5
FILTER_METHODS = IE.METHODS  # ["NoChange","SGD","PF","WSPF-A","WSPF-B"]
ALL_METHODS = FILTER_METHODS + BASE_METHODS

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "baselines",
)


def _run_baseline(learner, loader, model, n_classes):
    """1 ベースライン学習器を insects の窓プロトコルで走らせ報告区間 F1/Acc。"""
    n = loader.n_samples
    B, TE = IE.BATCH_SIZE, IE.TEST_SIZE
    f1s, accs = [], []
    pos = 0
    while pos + B + TE <= n:
        X_tr, y_tr = loader.X[pos:pos + B], loader.y[pos:pos + B]
        X_te = loader.X[pos + B:pos + B + TE]
        y_te = loader.y[pos + B:pos + B + TE]
        theta = learner.predict_theta()
        pred = model.predict(theta, X_te)[0]
        acc = float(np.mean(pred == y_te))
        # 報告区間のみ集計 (test window が REPORT_START 以降に始まるもの)
        if pos + B >= IE.REPORT_START:
            f1s.append(IE.macro_f1(pred, y_te, n_classes))
            accs.append(acc)
        # ドリフト信号 = テスト誤分類率(因果的: 学習前の θ で評価)
        learner.observe_error(1.0 - acc)
        learner.train(X_tr, y_tr)
        pos += B
    return float(np.mean(f1s)), float(np.mean(accs)), learner.n_resets


def _seed_job(args):
    seed, bp, bb, ba, best_sgd = args
    loader = InsectsDataLoader(
        IE.DATA_PATH, scale_fit_end=IE.SCALE_FIT_END, seed=IE.SEED)
    model = MulticlassNeuralNetModel(
        loader.n_features, IE.HIDDEN_DIM, loader.n_classes)
    param_dim = model.param_dim
    n_classes = loader.n_classes
    grad_raw = create_mc_grad_fn(model)

    def grad_fn(p, X, y):
        return IE.clip_gradients(grad_raw(p, X, y), IE.MAX_GRAD_NORM)

    # 既存手法(run_experiment)
    (result_rows, *_rest) = IE.run_experiment(
        N_PARTICLES, model, loader, bp, bb, ba, best_sgd,
        seed=seed, collect_diagnostics=False, verbose=False)
    out = {r["method"]: {"f1": r["macro_f1"], "acc": r["accuracy"]}
           for r in result_rows}

    # ベースライン(SGD 系学習器なので SGD 独自 HP を使用)
    learners = {
        "PH-SGD": DriftResetSGD(
            param_dim=param_dim, eta=best_sgd["eta"],
            prior_std=best_sgd["prior_std"],
            grad_fn=grad_fn, detector=PageHinkley(delta=0.01, lambda_=2.0),
            seed=seed + 20, grad_clip_norm=IE.MAX_GRAD_NORM),
        "Window-SGD": WindowSGD(
            param_dim=param_dim, eta=best_sgd["eta"],
            prior_std=best_sgd["prior_std"],
            grad_fn=grad_fn, window=WINDOW, seed=seed + 20,
            grad_clip_norm=IE.MAX_GRAD_NORM),
    }
    for name, learner in learners.items():
        f1, acc, nres = _run_baseline(learner, loader, model, n_classes)
        out[name] = {"f1": f1, "acc": acc, "resets": nres}
    return seed, out


def paired(pf, alt):
    pf = np.asarray(pf)
    alt = np.asarray(alt)
    d = alt - pf
    if len(d) >= 2 and np.any(d != 0):
        _, tp = stats.ttest_rel(alt, pf)
        try:
            _, wp = stats.wilcoxon(alt, pf)
        except ValueError:
            wp = float("nan")
    else:
        tp = wp = float("nan")
    return float(alt.mean()), float(d.mean()), float(tp), float(wp)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bp, bb, ba, best_sgd = IE.load_grid_search_params()[N_PARTICLES]
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Data-stream baselines — INSECTS (R2-4)")
    emit(f"  N={N_PARTICLES}, seeds={SEEDS_MULTI}")
    emit(f"  baselines: PH-SGD(drift-reset), Window-SGD(W={WINDOW}) "
         f"(SGD HP η={best_sgd['eta']}, σ0={best_sgd['prior_std']})")
    emit(f"  (leak-free: scale fit + select [0,{IE.SELECT_END}), "
         f"report [{IE.REPORT_START},end))")
    emit("=" * 72)

    per = {m: {"f1": [], "acc": []} for m in ALL_METHODS}
    resets = {m: [] for m in BASE_METHODS}
    jobs = [(s, bp, bb, ba, best_sgd) for s in SEEDS_MULTI]
    with ProcessPoolExecutor(
            max_workers=min(os.cpu_count() or 1, len(jobs))) as ex:
        for seed, out in ex.map(_seed_job, jobs):
            for m in ALL_METHODS:
                if m in out:
                    per[m]["f1"].append(out[m]["f1"])
                    per[m]["acc"].append(out[m]["acc"])
            for m in BASE_METHODS:
                resets[m].append(out[m].get("resets", 0))

    emit(f"\n  {'Method':<11s} {'macroF1 mean±std':>18s} "
         f"{'Acc mean±std':>16s} {'resets':>8s}")
    emit("  " + "-" * 56)
    for m in ALL_METHODS:
        f = np.asarray(per[m]["f1"])
        a = np.asarray(per[m]["acc"])
        rr = f"{np.mean(resets[m]):.1f}" if m in BASE_METHODS else "-"
        emit(f"  {m:<11s} {f.mean():>9.4f}±{f.std():<7.4f} "
             f"{a.mean():>8.4f}±{a.std():<6.4f} {rr:>8s}")

    emit(f"\n  対応のある検定 macro-F1 (vs PF, {len(SEEDS_MULTI)} seeds):")
    emit(f"  {'method':<11s} {'mean(alt)':>10s} {'Δ vs PF':>9s} "
         f"{'paired-t p':>11s} {'Wilcoxon p':>11s}")
    csv_rows = [("method", "f1_mean", "f1_std", "acc_mean", "delta_f1_vs_pf",
                 "paired_t_p", "wilcoxon_p", "resets")]
    for m in ALL_METHODS:
        f = np.asarray(per[m]["f1"])
        a = np.asarray(per[m]["acc"])
        if m != "PF":
            mean_alt, dd, tp, wp = paired(per["PF"]["f1"], per[m]["f1"])
            emit(f"  {m:<11s} {mean_alt:>10.4f} {dd:>+9.4f} "
                 f"{tp:>11.4g} {wp:>11.4g}")
            csv_rows.append((m, f"{f.mean():.6f}", f"{f.std():.6f}",
                             f"{a.mean():.6f}", f"{dd:.6f}", f"{tp:.6g}",
                             f"{wp:.6g}",
                             f"{np.mean(resets[m]):.2f}"
                             if m in BASE_METHODS else ""))
        else:
            csv_rows.append((m, f"{f.mean():.6f}", f"{f.std():.6f}",
                             f"{a.mean():.6f}", "", "", "", ""))

    txt_path = os.path.join(OUTPUT_DIR, "insects_baselines.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "insects_baselines.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        for row in csv_rows:
            fp.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
