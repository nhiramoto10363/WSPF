#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
較正レポート — INSECTS 多クラス (R2-5)

calibration_report.py (email, 二値 Brier/ECE) の多クラス一般化。
insects_experiment.py が保存済みの npz (報告区間の予測確率ベクトルと真ラベル)
を後処理するのみで、実験の再実行は不要。

指標:
  - 多クラス Brier スコア (true-class / full vector):
      BS = (1/n) Σ_i Σ_c (p_ic − onehot_ic)²   … 範囲 [0, 2]
    参考に true-class Brier  (1/n)Σ_i (1 − p_i,true)²  も併記。
  - ECE (expected calibration error, top-label/confidence ベース):
      confidence = max_c p_ic, prediction = argmax, correct = (pred==y)
      信頼度で 10 分割し Σ_b (n_b/n)|conf_b − acc_b|
  - 信頼性図 (confidence vs empirical accuracy)

npz 側 (calib_probs_{method} = (n,C), calib_labels = (n,)) は SEED=42 の
単一実行分。複数シード集約が必要なら insects_experiment を各シードで走らせて
npz を増やすこと (本スクリプトは全 results_N*_seed*.npz をプールする)。

出力:
  outputs/calibration/
    - insects_calibration.txt / .csv
    - insects_reliability.png
"""

import sys
import os
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.experiments.insects_experiment as IE

METHODS = IE.FILTER_METHODS  # ["SGD","PF","WSPF-A","WSPF-B"]
N_BINS = 10

INSECTS_DIR = IE.OUTPUT_DIR
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "calibration",
)


def load_npz_pool(n_particles=100):
    """results_N{n}_seed*.npz を全て読み、method ごとに probs/labels をプール。"""
    pattern = os.path.join(INSECTS_DIR, f"results_N{n_particles}_seed*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(
            f"npz が見つかりません({pattern})。先に "
            "insects_experiment.py を実行してください。")
    pooled = {m: {"p": [], "y": []} for m in METHODS}
    n_classes = None
    for path in paths:
        d = np.load(path, allow_pickle=True)
        labels = np.asarray(d["calib_labels"], dtype=np.int64)
        if labels.size == 0:
            continue
        for m in METHODS:
            key = m.replace("-", "_")
            probs = np.asarray(d[f"calib_probs_{key}"], dtype=np.float64)
            if probs.size == 0:
                continue
            pooled[m]["p"].append(probs)
            pooled[m]["y"].append(labels)
            if n_classes is None:
                n_classes = probs.shape[1]
    return pooled, paths, n_classes


def multiclass_brier(probs, labels, n_classes):
    """full-vector Brier と true-class Brier を返す。"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    bs_full = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    p_true = probs[np.arange(len(labels)), labels]
    bs_true = float(np.mean((1.0 - p_true) ** 2))
    return bs_full, bs_true


def confidence_ece(probs, labels, n_bins=N_BINS):
    """top-label(confidence) ベースの ECE と信頼性図の点列を返す。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    rel_x, rel_y, rel_w = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        c = conf[mask].mean()
        a = correct[mask].mean()
        ece += (mask.sum() / n) * abs(c - a)
        rel_x.append(c)
        rel_y.append(a)
        rel_w.append(mask.sum() / n)
    return float(ece), np.array(rel_x), np.array(rel_y), np.array(rel_w)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pooled, paths, n_classes = load_npz_pool(100)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Calibration report — INSECTS multiclass (R2-5)")
    emit(f"  classes={n_classes}, npz pooled={len(paths)} "
         f"(report region only)")
    for p in paths:
        emit(f"    - {os.path.basename(p)}")
    emit("=" * 72)

    emit(f"\n  {'method':<8s} {'Brier(full)':>12s} {'Brier(true)':>12s} "
         f"{'ECE':>9s} {'n':>9s}")
    emit("  " + "-" * 52)
    rows = {}
    reliab = {}
    csv_rows = [("method", "brier_full", "brier_true", "ece", "n")]
    for m in METHODS:
        if not pooled[m]["p"]:
            emit(f"  {m:<8s} (no probs)")
            continue
        p = np.concatenate(pooled[m]["p"], axis=0)
        y = np.concatenate(pooled[m]["y"], axis=0)
        bs_full, bs_true = multiclass_brier(p, y, n_classes)
        ece, rx, ry, rw = confidence_ece(p, y)
        rows[m] = (bs_full, bs_true, ece)
        reliab[m] = (rx, ry)
        emit(f"  {m:<8s} {bs_full:>12.4f} {bs_true:>12.4f} "
             f"{ece:>9.4f} {len(y):>9d}")
        csv_rows.append((m, f"{bs_full:.6f}", f"{bs_true:.6f}",
                         f"{ece:.6f}", str(len(y))))

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "insects_calibration.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---- 信頼性図 ----
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    for m in METHODS:
        if m not in reliab:
            continue
        rx, ry = reliab[m]
        plt.plot(rx, ry, "o-", color=colors.get(m), label=m, linewidth=1.4)
    plt.xlabel("confidence (max predicted probability)")
    plt.ylabel("empirical accuracy")
    plt.title("Reliability diagram (INSECTS, report region)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    rel_png = os.path.join(OUTPUT_DIR, "insects_reliability.png")
    plt.savefig(rel_png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "insects_calibration.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {rel_png}")


if __name__ == "__main__":
    main()
