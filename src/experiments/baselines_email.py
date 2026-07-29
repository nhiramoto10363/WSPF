#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データストリーム系ベースライン比較 — Email (R2-4)

概念ドリフト学習の代表的ベースラインを追加する:
  - PH-SGD     : Page-Hinkley でドリフト検知し θ をリセット(drift-detector reset)
  - Window-SGD : 直近 W バッチのみで学習(adaptive sliding window)

(ADWIN も src/baselines に実装済みだが、本 email ストリームでは高学習率 SGD が
 概念ドリフトを速やかに吸収し誤差平均がほぼ定常なため ADWIN はほとんど発火しない。
 参考としてコードは同梱、主要比較には発火する上記2手法を用いる。)

既存の SGD / PF / WSPF-A / WSPF-B / NoChange(run_experiment)と同一データ・
同一評価窓(リーク除去: PCA 期1-2 学習、報告 期3-5)で比較し、複数シードで
F1/Acc を集計、PF に対する対応のある検定を報告する。

出力:
  outputs/baselines/
    - baselines_email.txt / .csv
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

import src.experiments.email_binary_experiment as EB
from src.experiments.email_multiseed import load_hp
from src.data import EmailDataLoader
from src.models.neural_net import (
    NeuralNetModel, create_nn_grad_fn, create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)
from src.baselines import PageHinkley, DriftResetSGD, WindowSGD

SEEDS_MULTI = list(range(10))
N_PARTICLES = 100
BASE_METHODS = ["PH-SGD", "Window-SGD"]
WINDOW = 5
FILTER_METHODS = ["NoChange", "SGD", "PF", "WSPF-A", "WSPF-B"]
ALL_METHODS = FILTER_METHODS + BASE_METHODS

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "baselines",
)


def _run_baseline(learner, loader, model):
    """1 ベースライン学習器を email の窓プロトコルで走らせ報告区間 F1/Acc。"""
    n = loader.n_samples
    B, TE = EB.BATCH_SIZE, EB.TEST_SIZE
    f1s, accs = [], []
    pos = 0
    while pos + B + TE <= n:
        X_tr, y_tr = loader.X[pos:pos + B], loader.y[pos:pos + B]
        X_te, y_te = loader.X[pos + B:pos + B + TE], loader.y[pos + B:pos + B + TE]
        theta = learner.predict_theta()
        acc = EB.compute_accuracy(model, theta, X_te, y_te)
        # 報告区間のみ集計(期3-5)
        if pos + B >= EB.REPORT_START:
            f1s.append(EB.compute_f1(model, theta, X_te, y_te))
            accs.append(acc)
        # ドリフト信号 = テスト誤分類率(因果的: 学習前の θ で評価)
        learner.observe_error(1.0 - acc)
        learner.train(X_tr, y_tr)
        pos += B
    return float(np.mean(f1s)), float(np.mean(accs)), learner.n_resets


def _seed_job(args):
    seed, bp, bb, ba = args
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

    # 既存手法(run_experiment)
    (result_rows, *_rest) = EB.run_experiment(
        n_particles=N_PARTICLES, model=model, loader=loader, grad_fn=grad_fn,
        loglik_fn=loglik_fn, ps_grad_fn=ps_grad_fn,
        best_pf=bp, best_wspf_b=bb, best_wspf_a=ba,
        sgd_eta=bp["eta"], sgd_prior=bp["prior_std"], param_dim=param_dim,
        seed=seed)
    out = {r["method"]: {"f1": r["f1"], "acc": r["accuracy"]}
           for r in result_rows}

    # ベースライン(SGD の HP を流用)
    learners = {
        "PH-SGD": DriftResetSGD(
            param_dim=param_dim, eta=bp["eta"], prior_std=bp["prior_std"],
            grad_fn=grad_fn, detector=PageHinkley(delta=0.01, lambda_=2.0),
            seed=seed + 20, grad_clip_norm=EB.MAX_GRAD_NORM),
        "Window-SGD": WindowSGD(
            param_dim=param_dim, eta=bp["eta"], prior_std=bp["prior_std"],
            grad_fn=grad_fn, window=WINDOW, seed=seed + 20,
            grad_clip_norm=EB.MAX_GRAD_NORM),
    }
    for name, learner in learners.items():
        f1, acc, nres = _run_baseline(learner, loader, model)
        out[name] = {"f1": f1, "acc": acc, "resets": nres}
    return seed, out


def paired(pf, alt):
    pf = np.asarray(pf); alt = np.asarray(alt)
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
    bp, bb, ba, src = load_hp(N_PARTICLES)
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 72)
    emit("Data-stream baselines — Email (R2-4)")
    emit(f"  N={N_PARTICLES}, seeds={SEEDS_MULTI}, HP src={src}")
    emit(f"  baselines: PH-SGD(drift-reset), Window-SGD(W={WINDOW}) (SGD HP={bp})")
    emit("=" * 72)

    per = {m: {"f1": [], "acc": []} for m in ALL_METHODS}
    resets = {m: [] for m in BASE_METHODS}
    jobs = [(s, bp, bb, ba) for s in SEEDS_MULTI]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(jobs))) as ex:
        for seed, out in ex.map(_seed_job, jobs):
            for m in ALL_METHODS:
                if m in out:
                    per[m]["f1"].append(out[m]["f1"])
                    per[m]["acc"].append(out[m]["acc"])
            for m in BASE_METHODS:
                resets[m].append(out[m].get("resets", 0))

    emit(f"\n  {'Method':<11s} {'F1 mean±std':>16s} {'Acc mean±std':>16s} {'resets':>8s}")
    emit("  " + "-" * 54)
    for m in ALL_METHODS:
        f = np.asarray(per[m]["f1"]); a = np.asarray(per[m]["acc"])
        rr = f"{np.mean(resets[m]):.1f}" if m in BASE_METHODS else "-"
        emit(f"  {m:<11s} {f.mean():>8.4f}±{f.std():<6.4f} "
             f"{a.mean():>8.4f}±{a.std():<6.4f} {rr:>8s}")

    emit(f"\n  対応のある検定 F1 (vs PF, {len(SEEDS_MULTI)} seeds):")
    emit(f"  {'method':<11s} {'mean(alt)':>10s} {'Δ vs PF':>9s} "
         f"{'paired-t p':>11s} {'Wilcoxon p':>11s}")
    csv_rows = [("method", "f1_mean", "f1_std", "acc_mean", "delta_f1_vs_pf",
                 "paired_t_p", "wilcoxon_p", "resets")]
    for m in ALL_METHODS:
        f = np.asarray(per[m]["f1"]); a = np.asarray(per[m]["acc"])
        if m != "PF":
            mean_alt, dd, tp, wp = paired(per["PF"]["f1"], per[m]["f1"])
            emit(f"  {m:<11s} {mean_alt:>10.4f} {dd:>+9.4f} "
                 f"{tp:>11.4g} {wp:>11.4g}")
            csv_rows.append((m, f"{f.mean():.6f}", f"{f.std():.6f}",
                             f"{a.mean():.6f}", f"{dd:.6f}", f"{tp:.6g}",
                             f"{wp:.6g}",
                             f"{np.mean(resets[m]):.2f}" if m in BASE_METHODS else ""))
        else:
            csv_rows.append((m, f"{f.mean():.6f}", f"{f.std():.6f}",
                             f"{a.mean():.6f}", "", "", "", ""))

    txt_path = os.path.join(OUTPUT_DIR, "baselines_email.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "baselines_email.csv")
    with open(csv_path, "w", encoding="utf-8") as fp:
        for row in csv_rows:
            fp.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
