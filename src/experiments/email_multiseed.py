#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email 複数シード実験 (R1-14)

現行の email 実験は単一シード(初期粒子・リサンプリング・SGD 初期化に
一つの seed)だった。R1-14 に応え、シードを変えた複数回(既定 10)実行に
変更し、各メソッドの Acc/F1/LogLik をシード間で集計、PF に対する
WSPF-A / WSPF-B の対応のある検定(paired t + Wilcoxon)を報告する。

データストリームと PCA fit(期1-2)は固定なので、seed は初期粒子・
リサンプリング・SGD 初期化のみに効く(リーク除去パイプラインを維持)。
報告は評価区間(期3-5, sample>=REPORT_START)のみ(run_experiment に準拠)。

注) HP はグリッド結果の各メソッド best を用いる(無い/旧キーの場合は
    フォールバック)。email グリッドはリーク除去後に再実行が前提なので、
    確定値はグリッド再実行後に更新すること。

出力:
  outputs/email_binary/
    - email_multiseed.txt / .csv
"""

import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

import src.experiments.email_binary_experiment as EB
from src.data import EmailDataLoader
from src.models.neural_net import (
    NeuralNetModel, create_nn_grad_fn, create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)

SEEDS_MULTI = list(range(10))
N_PARTICLES = 100
METHODS = ["NoChange", "SGD", "PF", "WSPF-A", "WSPF-B"]
PARTICLE_METHODS = ["PF", "WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "email_binary",
)


def load_hp(n_p):
    """グリッド結果から各メソッド best HP を厳格に取得する。

    無言フォールバック(旧キー best_cpf への回避や PF 構成の代入)は誤設定の
    温床なので廃止。グリッド未実行・旧キー・キー欠損は明示的に失敗させる。
    """
    gs = EB.load_grid_search_params()
    if gs is None or str(n_p) not in gs:
        raise RuntimeError(
            f"email グリッド結果が見つかりません(N={n_p})。先に "
            "grid_search_email.py を実行してください。")
    e = gs[str(n_p)]
    for k in ("best_pf", "best_wspf_b", "best_wspf_a"):
        if k not in e:
            raise KeyError(
                f"email グリッド JSON に '{k}' がありません(旧キー best_cpf 等の"
                "可能性)。グリッドを再生成してください。")
    ba = e["best_wspf_a"]
    if "beta" not in ba:
        raise KeyError("best_wspf_a に 'beta' がありません。")
    return e["best_pf"], e["best_wspf_b"], ba, "grid"


def _seed_job(args):
    """1 シードの実験(worker 内で loader/model を再構築)。"""
    seed, n_p, bp, bb, ba = args
    loader = EmailDataLoader(EB.DATA_PATH, n_components=EB.PCA_DIM,
                             seed=EB.SEED, pca_fit_end=EB.PCA_FIT_END)
    model = NeuralNetModel(input_dim=loader.input_dim, hidden_dim=EB.HIDDEN_DIM,
                           output_dim=1, activation="tanh")
    param_dim = model.param_dim
    grad_fn_raw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)

    def grad_fn(particles, X, y):
        return EB.clip_gradients(grad_fn_raw(particles, X, y), EB.MAX_GRAD_NORM)

    (result_rows, acc, f1, ll, methods, eval_start,
     sample_pos, diagnostics) = EB.run_experiment(
        n_particles=n_p, model=model, loader=loader,
        grad_fn=grad_fn, loglik_fn=loglik_fn, ps_grad_fn=ps_grad_fn,
        best_pf=bp, best_wspf_b=bb, best_wspf_a=ba,
        sgd_eta=bp["eta"], sgd_prior=bp["prior_std"], param_dim=param_dim,
        seed=seed)
    return seed, {r["method"]: {"acc": r["accuracy"], "f1": r["f1"],
                                "loglik": r["loglik"]} for r in result_rows}


def paired_report(pf_vals, alt_vals):
    pf_vals = np.asarray(pf_vals); alt_vals = np.asarray(alt_vals)
    diff = alt_vals - pf_vals   # 正なら alt が PF より高い(F1/Acc は高い方が良い)
    out = {"mean_pf": float(pf_vals.mean()), "mean_alt": float(alt_vals.mean()),
           "mean_diff": float(diff.mean())}
    if len(diff) >= 2 and np.any(diff != 0):
        t, tp = stats.ttest_rel(alt_vals, pf_vals)
        out["t_p"] = float(tp)
        try:
            w, wp = stats.wilcoxon(alt_vals, pf_vals)
            out["w_p"] = float(wp)
        except ValueError:
            out["w_p"] = float("nan")
    else:
        out["t_p"] = out["w_p"] = float("nan")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bp, bb, ba, src = load_hp(N_PARTICLES)
    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 72)
    emit("Email multi-seed experiment (R1-14)")
    emit(f"  N={N_PARTICLES}, seeds={SEEDS_MULTI}")
    emit(f"  HP source={src}  PF={bp}")
    emit(f"    WSPF-B={bb}")
    emit(f"    WSPF-A={ba}")
    emit(f"  (leak-free: PCA fit 期1-2, report 期3-5)")
    emit("=" * 72)

    jobs = [(s, N_PARTICLES, bp, bb, ba) for s in SEEDS_MULTI]
    per_seed = {m: {"acc": [], "f1": [], "loglik": []} for m in METHODS}
    n_workers = min(os.cpu_count() or 1, len(SEEDS_MULTI))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for seed, res in ex.map(_seed_job, jobs):
            for m in METHODS:
                if m in res:
                    for k in ("acc", "f1", "loglik"):
                        per_seed[m][k].append(res[m][k])

    # ---- 集計表 ----
    emit(f"\n  {'Method':<10s} {'Acc mean±std':>18s} {'F1 mean±std':>18s} "
         f"{'LogLik mean±std':>20s}")
    emit("  " + "-" * 68)
    for m in METHODS:
        a = np.asarray(per_seed[m]["acc"]); f = np.asarray(per_seed[m]["f1"])
        l = np.asarray(per_seed[m]["loglik"])
        emit(f"  {m:<10s} {a.mean():>8.4f}±{a.std():<7.4f} "
             f"{f.mean():>8.4f}±{f.std():<7.4f} "
             f"{l.mean():>9.4f}±{l.std():<8.4f}")

    # ---- 対応検定(PF vs WSPF-A/B) ----
    emit(f"\n  対応のある検定 (vs PF, {len(SEEDS_MULTI)} seeds):")
    emit(f"  {'metric':<8s} {'method':<8s} {'mean(alt)':>10s} {'mean(PF)':>10s} "
         f"{'Δ':>9s} {'paired-t p':>11s} {'Wilcoxon p':>11s}")
    csv_rows = [("metric", "method", "mean_alt", "mean_pf", "delta",
                 "paired_t_p", "wilcoxon_p")]
    for metric in ("f1", "acc", "loglik"):
        for m in ("WSPF-A", "WSPF-B"):
            r = paired_report(per_seed["PF"][metric], per_seed[m][metric])
            emit(f"  {metric:<8s} {m:<8s} {r['mean_alt']:>10.4f} "
                 f"{r['mean_pf']:>10.4f} {r['mean_diff']:>+9.4f} "
                 f"{r['t_p']:>11.4g} {r['w_p']:>11.4g}")
            csv_rows.append((metric, m, f"{r['mean_alt']:.6f}",
                             f"{r['mean_pf']:.6f}", f"{r['mean_diff']:.6f}",
                             f"{r['t_p']:.6g}", f"{r['w_p']:.6g}"))

    txt_path = os.path.join(OUTPUT_DIR, "email_multiseed.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "email_multiseed.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
