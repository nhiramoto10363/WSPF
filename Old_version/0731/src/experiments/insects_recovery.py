#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Switch-aligned recovery 図 — INSECTS (旧 Fig. 5 の実データ多クラス版)

insects_experiment.py が保存済みの npz (per-step の Accuracy / macro-F1 と
change points) を後処理するのみ。実験の再実行は不要。

各既知スイッチ点 cp について、スイッチ直後の train window を lag=0 として
per-step 指標を整合させ、スイッチで平均した「回復曲線」を描く。整合対象は
リークフリー評価と整合させるため報告区間のスイッチ (cp >= REPORT_START) の
4 点に限定する (選択区間内の 14352 は除外)。WSPF-A/B が PF/SGD よりスイッチ後
の劣化が浅く回復が速いことを可視化する。

複数の results_N{n}_seed*.npz があればシード平均する (データ・スイッチ点は
固定なので per-step 指標を平均できる)。

出力:
  outputs/recovery/
    - insects_recovery.txt / .csv
    - insects_switch_recovery.png
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

METHODS = IE.METHODS  # ["NoChange","SGD","PF","WSPF-A","WSPF-B"]
PRE = 2            # スイッチ前に表示するステップ数
LAGS = 20         # スイッチ後に整合するステップ数
BATCH_SIZE = IE.BATCH_SIZE

INSECTS_DIR = IE.OUTPUT_DIR
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "recovery",
)


def load_perstep_pool(n_particles=100):
    """results_N{n}_seed*.npz をシード平均して per-step 指標を返す。"""
    pattern = os.path.join(INSECTS_DIR, f"results_N{n_particles}_seed*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(
            f"npz が見つかりません({pattern})。先に "
            "insects_experiment.py を実行してください。")
    acc = {m: [] for m in METHODS}
    f1 = {m: [] for m in METHODS}
    sample_positions = None
    change_points = None
    for path in paths:
        d = np.load(path, allow_pickle=True)
        if sample_positions is None:
            sample_positions = np.asarray(d["sample_positions"])
            change_points = np.asarray(d["change_points"])
        for m in METHODS:
            key = m.replace("-", "_")
            acc[m].append(np.asarray(d[f"acc_{key}"]))
            f1[m].append(np.asarray(d[f"f1_{key}"]))
    acc = {m: np.mean(acc[m], axis=0) for m in METHODS}
    f1 = {m: np.mean(f1[m], axis=0) for m in METHODS}
    return acc, f1, sample_positions, change_points, paths


def align_recovery(series, sample_positions, change_points):
    """
    各スイッチを lag=0 に整合し、lag∈[-PRE, LAGS) の per-step 指標を平均。

    Returns
    -------
    lags : ndarray            (PRE+LAGS,)  lag 値 (負=スイッチ前)
    curve : ndarray           (PRE+LAGS,)  スイッチ平均した指標
    used : list of int        整合に使えたスイッチ点
    """
    n_steps = len(sample_positions)
    lags = np.arange(-PRE, LAGS)
    per_switch = []
    used = []
    for cp in change_points:
        s0 = int(np.searchsorted(sample_positions, cp))  # スイッチ後最初のstep
        if s0 - PRE < 0 or s0 + LAGS > n_steps:
            continue
        per_switch.append(series[s0 - PRE: s0 + LAGS])
        used.append(int(cp))
    if not per_switch:
        return lags, np.full(len(lags), np.nan), used
    return lags, np.mean(per_switch, axis=0), used


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    acc, f1, sample_positions, change_points, paths = load_perstep_pool(100)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # 報告区間のスイッチのみ整合対象にする (選択区間 [0,REPORT_START) の
    # スイッチ 14352 はリークフリー評価の対象外なので除外)。
    report_switches = np.asarray(
        [int(c) for c in change_points if int(c) >= IE.REPORT_START])

    emit("=" * 72)
    emit("Switch-aligned recovery — INSECTS (旧 Fig. 5, multiclass)")
    emit(f"  seeds pooled={len(paths)}, "
         f"all change_points={[int(c) for c in change_points]}")
    emit(f"  report-region switches (>= {IE.REPORT_START}): "
         f"{[int(c) for c in report_switches]}")
    emit(f"  lag window: [-{PRE}, {LAGS}) steps (1 step = {BATCH_SIZE} samples)")
    emit("=" * 72)

    curves_f1 = {}
    curves_acc = {}
    used = None
    for m in METHODS:
        lags, cf1, u = align_recovery(f1[m], sample_positions, report_switches)
        _, cacc, _ = align_recovery(acc[m], sample_positions, report_switches)
        curves_f1[m] = cf1
        curves_acc[m] = cacc
        used = u

    emit(f"\n  整合に使ったスイッチ ({len(used)}): {used}")
    emit(f"\n  macro-F1 recovery (lag=スイッチ後ステップ, スイッチ平均):")
    header = "  lag  " + " ".join(f"{m:>9s}" for m in METHODS)
    emit(header)
    for i, lag in enumerate(lags):
        emit(f"  {lag:>3d}  " +
             " ".join(f"{curves_f1[m][i]:>9.4f}" for m in METHODS))

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "insects_recovery.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("lag," + ",".join(f"f1_{m}" for m in METHODS) + "," +
                ",".join(f"acc_{m}" for m in METHODS) + "\n")
        for i, lag in enumerate(lags):
            f.write(f"{lag}," +
                    ",".join(f"{curves_f1[m][i]:.6f}" for m in METHODS) + "," +
                    ",".join(f"{curves_acc[m][i]:.6f}" for m in METHODS) + "\n")

    # ---- 図 ----
    colors = {"NoChange": "#000000", "SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in METHODS:
        a1.plot(lags, curves_f1[m], "o-", ms=3, color=colors.get(m),
                label=m, linewidth=1.4)
        a2.plot(lags, curves_acc[m], "o-", ms=3, color=colors.get(m),
                label=m, linewidth=1.4)
    for a, ylab, title in [
            (a1, "macro-F1", "macro-F1 recovery"),
            (a2, "accuracy", "accuracy recovery")]:
        a.axvline(0.0, color="red", ls=":", lw=0.9, label="switch")
        a.set_xlabel("steps since concept switch")
        a.set_ylabel(ylab)
        a.set_title(f"{title} (INSECTS, {len(used)} switches avg)")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, "insects_switch_recovery.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "insects_recovery.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {png}")


if __name__ == "__main__":
    main()
