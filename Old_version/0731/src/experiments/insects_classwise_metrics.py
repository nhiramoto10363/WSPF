#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
クラス別指標レポート — INSECTS 多クラス (R1-minor)

再実行不要の純粋な後処理。insects_experiment.py が保存した
results_N100_seed42.npz の較正用確率ベクトル
calib_probs_{SGD,PF,WSPF_A,WSPF_B} (shape (n_report_samples, 6)) と
calib_labels (n_report_samples,) から全て計算する (seed 42, 記述統計)。

手法ごとに:
  1. pred = argmax(probs) → 6×6 混同行列 (報告区間プール)
  2. per-class precision / recall / F1 (6 クラス)
  3. プール macro-F1、balanced accuracy (= mean recall)、micro accuracy
  4. 整合性チェック: micro accuracy == acc_{m}[eval_mask] の窓平均
     (32 サンプル窓が均一なので厳密一致するはず。ズレたら probs 記録
      タイミングのバグ検出)
  5. window 平均 macro-F1 (選択・検定に使用) と プール macro-F1 (記述用) の
     対比を明示。乖離は 32 サンプル窓内のラベル時間クラスタリングによる
     窓内クラス不均衡に起因する (docstring / 論文で併記方針)。

出力:
  outputs/insects/
    - insects_classwise_metrics.txt / .csv
    - insects_confusion_matrices.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.experiments.insects_experiment as IE

METHODS = IE.FILTER_METHODS  # ["SGD","PF","WSPF-A","WSPF-B"]

# データ CSV のラベルは '0'..'5'。実クラス名 (蚊 3 種 × 雌雄) との対応が
# README で確定したらここを書き換えれば論文の表がそのまま書ける。
CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]

NPZ_PATH = os.path.join(IE.OUTPUT_DIR, "results_N100_seed42.npz")
OUTPUT_DIR = IE.OUTPUT_DIR


def confusion_matrix(pred, y, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    """混同行列 (true×pred) から per-class precision/recall/F1。"""
    n = cm.shape[0]
    prec, rec, f1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * prec[c] * rec[c] / (prec[c] + rec[c])
                 if (prec[c] + rec[c]) > 0 else 0.0)
    return prec, rec, f1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(NPZ_PATH):
        raise RuntimeError(
            f"npz が見つかりません({NPZ_PATH})。先に "
            "insects_experiment.py を実行してください。")
    d = np.load(NPZ_PATH, allow_pickle=True)
    labels = np.asarray(d["calib_labels"], dtype=np.int64)
    eval_mask = np.asarray(d["eval_mask"], dtype=bool)
    n_classes = len(CLASS_NAMES)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Class-wise metrics — INSECTS (R1-minor, seed 42, descriptive)")
    emit(f"  classes={n_classes} {CLASS_NAMES}, report samples={len(labels)}")
    emit("=" * 72)

    cms = {}
    csv_rows = [("method", "class", "precision", "recall", "f1", "support")]
    summary_rows = [("method", "pooled_macro_f1", "window_avg_macro_f1",
                     "balanced_acc", "micro_acc", "window_avg_acc")]

    for m in METHODS:
        key = m.replace("-", "_")
        probs = np.asarray(d[f"calib_probs_{key}"], dtype=np.float64)
        if probs.size == 0:
            emit(f"\n  [{m}] no probs")
            continue
        pred = probs.argmax(axis=1)
        cm = confusion_matrix(pred, labels, n_classes)
        cms[m] = cm
        prec, rec, f1 = per_class_prf(cm)
        support = cm.sum(axis=1)

        pooled_macro_f1 = float(np.mean(f1))
        balanced_acc = float(np.mean(rec))          # = mean recall
        micro_acc = float(np.trace(cm) / cm.sum())

        # per-step 配列から窓平均 (選択・検定に使う量)
        win_f1 = float(np.asarray(d[f"f1_{key}"])[eval_mask].mean())
        win_acc = float(np.asarray(d[f"acc_{key}"])[eval_mask].mean())

        # --- 整合性チェック: micro accuracy == 窓平均 accuracy ---
        # 各報告窓は 32 サンプル均一なので厳密一致するはず。
        assert np.isclose(micro_acc, win_acc, atol=1e-6), (
            f"{m}: micro_acc={micro_acc:.6f} != window-avg acc="
            f"{win_acc:.6f} — probs 記録タイミングのバグの可能性")

        emit(f"\n  ===== {m} =====")
        emit(f"  {'class':<7s} {'prec':>8s} {'recall':>8s} {'F1':>8s} "
             f"{'support':>9s}")
        for c in range(n_classes):
            emit(f"  {CLASS_NAMES[c]:<7s} {prec[c]:>8.4f} {rec[c]:>8.4f} "
                 f"{f1[c]:>8.4f} {support[c]:>9d}")
            csv_rows.append((m, CLASS_NAMES[c], f"{prec[c]:.6f}",
                             f"{rec[c]:.6f}", f"{f1[c]:.6f}", str(int(support[c]))))
        emit(f"  pooled macro-F1 = {pooled_macro_f1:.4f}  "
             f"(window-avg macro-F1 = {win_f1:.4f})")
        emit(f"  balanced acc (mean recall) = {balanced_acc:.4f}  "
             f"micro acc = {micro_acc:.4f}  (window-avg acc = {win_acc:.4f}, "
             f"整合 OK)")
        summary_rows.append((m, f"{pooled_macro_f1:.6f}", f"{win_f1:.6f}",
                             f"{balanced_acc:.6f}", f"{micro_acc:.6f}",
                             f"{win_acc:.6f}"))

    emit(f"\n  注) pooled macro-F1 と window-avg macro-F1 の乖離は、32 サンプル")
    emit(f"      窓内のラベル時間クラスタリング(窓内クラス不均衡)による。")
    emit(f"      選択・検定は window-avg、記述は pooled を併記する方針。")

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "insects_classwise_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
        f.write("\n")
        for row in summary_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---- 混同行列ヒートマップ (4 手法 2×2, 行正規化) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, m in zip(axes.ravel(), METHODS):
        if m not in cms:
            ax.axis("off")
            continue
        cm = cms[m]
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{m} (row-normalized)")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black",
                        fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("INSECTS confusion matrices (report region, seed 42)")
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "insects_confusion_matrices.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "insects_classwise_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {png}")


if __name__ == "__main__":
    main()
