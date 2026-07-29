#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
較正・不確実性・回復のレポート (R2-5)

R2-5 は「適応・較正・不確実性の質」を分離した評価を求める。本スクリプトは
既存ログを活用して以下を集約する:

  Regression:
    - 予測区間被覆率(名目 90%)の名目 vs 経験値(較正)
    - 事後多様性(粒子群 spread trace, ユニーク祖先数)
    - post-switch 回復曲線(スイッチからの経過ステップ別 MSE)+ 各ラグの対応検定
  Email:
    - Brier スコア(報告区間の予測確率)
    - 信頼性図(reliability)と ECE(expected calibration error)
    - 事後多様性(report 区間の spread trace, ユニーク数)

出力:
  outputs/calibration/
    - calibration_report.txt / .csv
    - calibration_reliability_email.png
    - calibration_recovery_regression.png
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import src.experiments.regression_regime_switch as RR
from src.experiments.regression_regime_switch import (
    run_single, SEEDS, T, EVAL_START, N_POST_SWITCH, COVERAGE_Z,
)
from src.experiments.matched_hp_regression import build_matched_hp
import src.experiments.email_binary_experiment as EB
from src.experiments.email_multiseed import load_hp

REG_METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]
PARTICLE_METHODS = ["PF", "WSPF-A", "WSPF-B"]
EMAIL_METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]
NOMINAL = 0.90
N_PARTICLES = 100

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "calibration",
)


def _ev(a):
    return np.asarray(a)[EVAL_START:]


# ================================================================
# Regression
# ================================================================
def _reg_job(args):
    seed, bp, bb, ba = args
    res = run_single(seed, N_PARTICLES, bp, bb, ba)
    out = {"cov": {}, "spread": {}, "unique": {}, "mse_ts": {},
           "switch": res["switch_times"]}
    for m in REG_METHODS:
        out["cov"][m] = float(_ev(res["coverage"][m]).mean())
        out["mse_ts"][m] = np.asarray(res["mse"][m])
    for m in PARTICLE_METHODS:
        h = res["histories"][m]
        out["spread"][m] = float(_ev(h["spread_trace"]).mean())
        out["unique"][m] = float(_ev(h["unique_particles"]).mean())
    return out


def regression_calibration(emit):
    matched, beta, src = build_matched_hp()[N_PARTICLES]
    bp = dict(matched); bb = dict(matched); ba = {**matched, "beta": beta}
    jobs = [(s, bp, bb, ba) for s in SEEDS]
    results = []
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(jobs))) as ex:
        for r in ex.map(_reg_job, jobs):
            results.append(r)
    switch = results[0]["switch"]

    emit(f"\n{'='*72}\n  Regression 較正・多様性 (R2-5, matched HP)\n{'='*72}")
    emit(f"  予測区間被覆率(名目 {NOMINAL:.0%}, eval region):")
    emit(f"  {'method':<8s} {'empirical cov':>14s} {'|cov−nominal|':>14s} "
         f"{'spread(tr)':>12s} {'unique':>8s}")
    cov_rows = {}
    for m in REG_METHODS:
        cov = np.mean([r["cov"][m] for r in results])
        sp = np.mean([r["spread"][m] for r in results]) if m in PARTICLE_METHODS else float("nan")
        un = np.mean([r["unique"][m] for r in results]) if m in PARTICLE_METHODS else float("nan")
        cov_rows[m] = (cov, sp, un)
        sp_s = f"{sp:>12.4f}" if m in PARTICLE_METHODS else f"{'-':>12s}"
        un_s = f"{un:>8.1f}" if m in PARTICLE_METHODS else f"{'-':>8s}"
        emit(f"  {m:<8s} {cov:>13.4f} {abs(cov-NOMINAL):>14.4f} {sp_s} {un_s}")

    # post-switch 回復曲線: スイッチからの経過ステップ別 MSE(全スイッチ平均)
    L = N_POST_SWITCH
    recov = {m: np.zeros((len(results), L)) for m in REG_METHODS}
    for si, r in enumerate(results):
        for m in REG_METHODS:
            ts = r["mse_ts"][m]
            lags = []
            for lag in range(L):
                vals = [ts[st + lag] for st in switch if st + lag < T]
                lags.append(np.mean(vals))
            recov[m][si] = lags
    emit(f"\n  post-switch 回復曲線 MSE(lag=スイッチ後ステップ, seed平均)+ 対応検定 vs PF:")
    header = "  lag " + " ".join(f"{m:>8s}" for m in REG_METHODS) + \
             "   pA(WSPF-A) pB(WSPF-B)"
    emit(header)
    recov_curve = {m: recov[m].mean(axis=0) for m in REG_METHODS}
    for lag in range(L):
        row = f"  {lag:>3d} " + " ".join(f"{recov_curve[m][lag]:>8.4f}"
                                         for m in REG_METHODS)
        pf_l = recov["PF"][:, lag]
        pa = stats.ttest_rel(recov["WSPF-A"][:, lag], pf_l)[1]
        pb = stats.ttest_rel(recov["WSPF-B"][:, lag], pf_l)[1]
        row += f"   {pa:>9.3g} {pb:>9.3g}"
        emit(row)

    return cov_rows, recov_curve


# ================================================================
# Email
# ================================================================
def _brier_ece(probs, labels, n_bins=10):
    brier = float(np.mean((probs - labels) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    rel_x, rel_y = [], []
    n = len(probs)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
        rel_x.append(conf); rel_y.append(acc)
    return brier, float(ece), np.array(rel_x), np.array(rel_y)


def email_calibration(emit):
    bp, bb, ba, src = load_hp(N_PARTICLES)
    seeds = list(range(10))
    jobs = [(s, bp, bb, ba) for s in seeds]
    # _email_seed_job は (seed, {method:{f1,acc}}) を返すため、較正には
    # 予測確率が必要 → run_experiment を直接呼ぶ専用 worker を使う
    pooled = {m: {"p": [], "y": []} for m in EMAIL_METHODS}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, len(jobs))) as ex:
        for probs, labels in ex.map(_email_calib_job, jobs):
            for m in EMAIL_METHODS:
                pooled[m]["p"].append(probs[m])
                pooled[m]["y"].append(labels)

    emit(f"\n{'='*72}\n  Email 較正 (R2-5, {len(seeds)} seeds, report 区間)\n{'='*72}")
    emit(f"  {'method':<8s} {'Brier':>9s} {'ECE':>9s}")
    reliab = {}
    rows = {}
    for m in EMAIL_METHODS:
        p = np.concatenate(pooled[m]["p"]); y = np.concatenate(pooled[m]["y"])
        brier, ece, rx, ry = _brier_ece(p, y)
        reliab[m] = (rx, ry)
        rows[m] = (brier, ece)
        emit(f"  {m:<8s} {brier:>9.4f} {ece:>9.4f}")
    return rows, reliab


def _email_calib_job(args):
    seed, bp, bb, ba = args
    from src.data import EmailDataLoader
    from src.models.neural_net import (
        NeuralNetModel, create_nn_grad_fn, create_nn_loglik_fn,
        create_nn_per_sample_grad_fn)
    loader = EmailDataLoader(EB.DATA_PATH, n_components=EB.PCA_DIM,
                             seed=EB.SEED, pca_fit_end=EB.PCA_FIT_END)
    model = NeuralNetModel(input_dim=loader.input_dim, hidden_dim=EB.HIDDEN_DIM,
                           output_dim=1, activation="tanh")
    pd_ = model.param_dim
    graw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps = create_nn_per_sample_grad_fn(model)

    def gfn(p, X, y):
        return EB.clip_gradients(graw(p, X, y), EB.MAX_GRAD_NORM)

    (*_head, diagnostics) = EB.run_experiment(
        n_particles=N_PARTICLES, model=model, loader=loader, grad_fn=gfn,
        loglik_fn=loglik_fn, ps_grad_fn=ps, best_pf=bp, best_wspf_b=bb,
        best_wspf_a=ba, sgd_eta=bp["eta"], sgd_prior=bp["prior_std"],
        param_dim=pd_, seed=seed)
    calib = diagnostics["_calib"]
    return calib["probs"], calib["labels"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 72)
    emit("Calibration / uncertainty / recovery report (R2-5)")
    emit("=" * 72)

    cov_rows, recov_curve = regression_calibration(emit)
    email_rows, reliab = email_calibration(emit)

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "calibration_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("task,method,metric,value\n")
        for m, (cov, sp, un) in cov_rows.items():
            f.write(f"regression,{m},interval_coverage,{cov:.6f}\n")
            if not np.isnan(sp):
                f.write(f"regression,{m},spread_trace,{sp:.6f}\n")
                f.write(f"regression,{m},unique_particles,{un:.6f}\n")
        for m, (brier, ece) in email_rows.items():
            f.write(f"email,{m},brier,{brier:.6f}\n")
            f.write(f"email,{m},ece,{ece:.6f}\n")

    # ---- 信頼性図(email) ----
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    for m in EMAIL_METHODS:
        rx, ry = reliab[m]
        plt.plot(rx, ry, "o-", color=colors[m], label=m, linewidth=1.4)
    plt.xlabel("mean predicted probability"); plt.ylabel("empirical frequency")
    plt.title("Reliability diagram (Email, report region)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    rel_png = os.path.join(OUTPUT_DIR, "calibration_reliability_email.png")
    plt.savefig(rel_png, dpi=150, bbox_inches="tight"); plt.close()

    # ---- 回復曲線(regression) ----
    plt.figure(figsize=(7, 5))
    for m in REG_METHODS:
        plt.plot(range(N_POST_SWITCH), recov_curve[m], "o-",
                 color=colors[m], label=m, linewidth=1.5)
    plt.xlabel("steps since concept switch"); plt.ylabel("MSE")
    plt.title("Post-switch recovery (Regression)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    rec_png = os.path.join(OUTPUT_DIR, "calibration_recovery_regression.png")
    plt.savefig(rec_png, dpi=150, bbox_inches="tight"); plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "calibration_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {rel_png}")
    emit(f"Saved: {rec_png}")


if __name__ == "__main__":
    main()
