#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マッチド・ハイパーパラメータ実験 — Email (R1-7)

PF / WSPF-A / WSPF-B に **同一の (η, σcd, σ0)** を与えて比較する。
共通ハイパラ = グリッドサーチの PF 最良構成 (best_pf) を全メソッドに適用
(WSPF-A の β は補正固有なのでグリッド best_wspf_a[β] を使用)。

リーク除去(R1-13)後のパイプラインをそのまま使う(PCA は期1-2 で学習、
報告は期3-5 のみ)。email_binary_experiment.run_experiment を再利用。

注) 本スクリプトは単一シードの点比較(F1/Acc/LogLik)。対応のある
    有意差検定は複数シード化(タスク8, R1-14)後に別途行う。

出力:
  outputs/matched_hp/
    - matched_hp_email.txt / .csv
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

import src.experiments.email_binary_experiment as EB
from src.data import EmailDataLoader
from src.models.neural_net import (
    NeuralNetModel,
    create_nn_grad_fn,
    create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "matched_hp",
)


def build_matched_hp():
    """PF 最良構成を共通ハイパラとして各粒子数ぶん構築する。"""
    gs = EB.load_grid_search_params()
    # 厳格化(無言フォールバック廃止): 未実行・N 欠損・旧/欠損キーは明示的に失敗。
    if gs is None:
        raise RuntimeError(
            "email グリッド結果が見つかりません。先に "
            "grid_search_email.py を実行してください。")
    out = {}
    for n_p in EB.N_PARTICLES_LIST:
        if str(n_p) not in gs:
            raise KeyError(f"email グリッド JSON に N={n_p} がありません。")
        entry = gs[str(n_p)]
        for k in ("best_pf", "best_wspf_a"):
            if k not in entry:
                raise KeyError(
                    f"email グリッド JSON に '{k}' がありません(旧キーの可能性)。"
                    "グリッドを再生成してください。")
        if "beta" not in entry["best_wspf_a"]:
            raise KeyError("best_wspf_a に 'beta' がありません。")
        best_pf = entry["best_pf"]
        beta = entry["best_wspf_a"]["beta"]
        matched = {"eta": best_pf["eta"], "sigma_sys": best_pf["sigma_sys"],
                   "prior_std": best_pf["prior_std"]}
        out[n_p] = (matched, beta, "grid best_pf")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Matched-HP experiment — Email (R1-7)")
    emit("  共通ハイパラ = PF 最良構成を全メソッドに適用 (leak-free pipeline)")
    emit("=" * 72)

    # -------- データ・モデル(email_binary と同一設定) --------
    loader = EmailDataLoader(
        EB.DATA_PATH, n_components=EB.PCA_DIM, seed=EB.SEED,
        pca_fit_end=EB.PCA_FIT_END,
    )
    emit(f"  samples={loader.n_samples}, features={loader.input_dim}, "
         f"PCA_FIT_END={EB.PCA_FIT_END}, REPORT_START={EB.REPORT_START}")

    model = NeuralNetModel(input_dim=loader.input_dim, hidden_dim=EB.HIDDEN_DIM,
                           output_dim=1, activation="tanh")
    param_dim = model.param_dim
    grad_fn_raw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)

    def grad_fn(particles, X, y):
        g = grad_fn_raw(particles, X, y)
        return EB.clip_gradients(g, EB.MAX_GRAD_NORM)

    matched_by_n = build_matched_hp()

    csv_rows = [("n_particles", "method", "accuracy", "f1", "loglik",
                 "f1_delta_vs_pf")]
    for n_p in EB.N_PARTICLES_LIST:
        matched, beta, src = matched_by_n[n_p]
        emit(f"\n  N={n_p}: (η,σcd,σ0)={matched} [{src}], WSPF-A β={beta}")

        best_pf = matched
        best_wspf_b = dict(matched)
        best_wspf_a = {**matched, "beta": beta}

        (result_rows, acc, f1, ll, methods, eval_start,
         sample_pos, diagnostics) = EB.run_experiment(
            n_particles=n_p, model=model, loader=loader,
            grad_fn=grad_fn, loglik_fn=loglik_fn, ps_grad_fn=ps_grad_fn,
            best_pf=best_pf, best_wspf_b=best_wspf_b, best_wspf_a=best_wspf_a,
            sgd_eta=matched["eta"], sgd_prior=matched["prior_std"],
            param_dim=param_dim,
        )

        by_method = {r["method"]: r for r in result_rows}
        pf_f1 = by_method["PF"]["f1"]
        emit(f"\n  === Matched-HP results (N={n_p}, reported on periods 3-5) ===")
        emit(f"  {'Method':<10s} {'Accuracy':>10s} {'F1':>10s} "
             f"{'Log-Lik':>10s} {'F1 vs PF':>10s}")
        emit(f"  {'-'*54}")
        for r in result_rows:
            dstr = "-"
            if r["method"] in ("WSPF-A", "WSPF-B"):
                dstr = f"{r['f1']-pf_f1:+.4f}"
            emit(f"  {r['method']:<10s} {r['accuracy']:>10.4f} "
                 f"{r['f1']:>10.4f} {r['loglik']:>10.4f} {dstr:>10s}")
            csv_rows.append((n_p, r["method"], f"{r['accuracy']:.6f}",
                             f"{r['f1']:.6f}", f"{r['loglik']:.6f}",
                             f"{r['f1']-pf_f1:.6f}"
                             if r["method"] in ("WSPF-A", "WSPF-B") else ""))

    txt_path = os.path.join(OUTPUT_DIR, "matched_hp_email.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "matched_hp_email.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
