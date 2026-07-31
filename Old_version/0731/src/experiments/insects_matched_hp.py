#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マッチド・ハイパーパラメータ実験 — INSECTS (R1-7 の実データ側)

matched_hp_email.py の多クラス鏡像。PF / WSPF-A / WSPF-B に **同一の
(η, σcd, σ0)** を与えて比較する。共通ハイパラ = グリッドサーチの PF 最良
構成 (best_pf) を全メソッドに適用 (WSPF-A の β は補正固有なのでグリッド
best_wspf_a[β] を使用)。

リークフリー・パイプラインをそのまま使う (標準化 fit と HP 選択は
[0, SELECT_END)、報告は test window が REPORT_START 以降)。
insects_experiment.run_experiment を再利用する。

注) 本スクリプトは単一シードの点比較 (Accuracy/macro-F1/LogLik)。対応の
    ある有意差検定は複数シード側 (insects_multiseed.py, R1-14) で行う。

出力:
  outputs/matched_hp/
    - insects_matched_hp.txt / .csv

事前に grid_search_insects.py を実行しておくこと (strict loader)。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import src.experiments.insects_experiment as IE
from src.data.insects_loader import InsectsDataLoader
from src.models.neural_net_multiclass import MulticlassNeuralNetModel

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "matched_hp",
)


def build_matched_hp():
    """PF 最良構成を共通ハイパラとして各粒子数ぶん構築する。"""
    hp_by_n = IE.load_grid_search_params()  # strict: 未実行・欠損は例外
    out = {}
    for n_p in IE.N_PARTICLES_LIST:
        best_pf, best_wspf_b, best_wspf_a, best_sgd = hp_by_n[n_p]
        beta = best_wspf_a["beta"]
        matched = {"eta": best_pf["eta"],
                   "sigma_sys": best_pf["sigma_sys"],
                   "prior_std": best_pf["prior_std"]}
        out[n_p] = (matched, beta, "grid best_pf", best_sgd)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Matched-HP experiment — INSECTS (R1-7)")
    emit("  共通ハイパラ = PF 最良構成を全メソッドに適用 (leak-free pipeline)")
    emit("=" * 72)

    loader = InsectsDataLoader(
        IE.DATA_PATH, scale_fit_end=IE.SCALE_FIT_END, seed=IE.SEED)
    loader.print_regime_class_distribution(emit)
    model = MulticlassNeuralNetModel(
        loader.n_features, IE.HIDDEN_DIM, loader.n_classes)
    emit(f"\n  model: input={loader.n_features}, hidden={IE.HIDDEN_DIM}, "
         f"classes={loader.n_classes}, d={model.param_dim}")
    emit(f"  leak-free: scale fit + HP select [0,{IE.SELECT_END}), "
         f"report [{IE.REPORT_START},end)")

    matched_by_n = build_matched_hp()

    csv_rows = [("n_particles", "method", "accuracy", "macro_f1", "loglik",
                 "f1_delta_vs_pf")]
    for n_p in IE.N_PARTICLES_LIST:
        matched, beta, src, best_sgd = matched_by_n[n_p]
        emit(f"\n  N={n_p}: (η,σcd,σ0)={matched} [{src}], WSPF-A β={beta}, "
             f"SGD(独立)η={best_sgd['eta']}")

        best_pf = matched
        best_wspf_b = dict(matched)
        best_wspf_a = {**matched, "beta": beta}

        (result_rows, acc, f1, ll, sample_positions, eval_mask,
         diagnostics) = IE.run_experiment(
            n_p, model, loader, best_pf, best_wspf_b, best_wspf_a,
            best_sgd, seed=IE.SEED, collect_diagnostics=False)

        by_method = {r["method"]: r for r in result_rows}
        pf_f1 = by_method["PF"]["macro_f1"]
        emit(f"\n  === Matched-HP results (N={n_p}, "
             f"reported on report region) ===")
        emit(f"  {'Method':<10s} {'Accuracy':>10s} {'macro-F1':>10s} "
             f"{'Log-Lik':>10s} {'F1 vs PF':>10s}")
        emit(f"  {'-' * 54}")
        for r in result_rows:
            dstr = "-"
            if r["method"] in ("WSPF-A", "WSPF-B"):
                dstr = f"{r['macro_f1'] - pf_f1:+.4f}"
            emit(f"  {r['method']:<10s} {r['accuracy']:>10.4f} "
                 f"{r['macro_f1']:>10.4f} {r['loglik']:>10.4f} {dstr:>10s}")
            csv_rows.append((n_p, r["method"], f"{r['accuracy']:.6f}",
                             f"{r['macro_f1']:.6f}", f"{r['loglik']:.6f}",
                             f"{r['macro_f1'] - pf_f1:.6f}"
                             if r["method"] in ("WSPF-A", "WSPF-B") else ""))

    txt_path = os.path.join(OUTPUT_DIR, "insects_matched_hp.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "insects_matched_hp.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
