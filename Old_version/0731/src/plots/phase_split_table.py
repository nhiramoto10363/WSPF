#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email binary 実験の期間分解テーブルを CSV 出力する。

各手法について、ウォームアップ後の評価ステップを
  - 安定期 (stable):     スイッチ直後の遷移ウィンドウ以外
  - スイッチ後 (post):   各コンセプトドリフト直後の数ステップ
に分割し、対数尤度 (ll) と精度 (acc) の平均を集計する。

出力 CSV のレイアウト（表示テーブルに対応）:
    period,metric,SGD,PF,WSPF-A,WSPF-B
    stable,ll,...
    stable,acc,...
    post_switch,ll,...
    post_switch,acc,...
"""

import os
import sys
import csv

import numpy as np

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

# ================================================================
# 設定（email_binary_experiment.py と整合）
# ================================================================
INPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "email_binary")
OUTPUT_DIR = INPUT_DIR

BATCH_SIZE = 16
REPORT_START = 600                 # リーク除去後の報告区間開始(期3-5)
# コンセプトドリフト点(サンプル位置)。リーク除去後は報告区間(≥600)に入る
# スイッチのみ有効。300 は warm-up、600 は報告区間の開始点(直前の安定期が
# 報告区間外)なので除外し、実質のスイッチイベントは 900, 1200 の 2 点。
# 論文の「four switch events で平均」は「two events」に要修正(Table 6 / Fig 5)。
ALL_SWITCHES = [300, 600, 900, 1200]
SWITCHES = [sw for sw in ALL_SWITCHES if sw > REPORT_START]  # -> [900, 1200]
POST_SWITCH_STEPS = 5              # スイッチ後を「遷移期」とみなすステップ数

# npz のキー接頭辞 -> 論文での手法名
METHODS = [
    ("sgd", "SGD"),
    ("pf", "PF"),
    ("wspf_a", "WSPF-A"),
    ("wspf_b", "WSPF-B"),
]


def compute_phase_table(npz_path):
    """1 つの結果 npz から期間分解テーブル（dict）を計算する。"""
    d = np.load(npz_path, allow_pickle=True)

    eval_start = int(d["eval_start"])
    pos = d["sample_positions"]
    n_steps = len(pos)

    # テストウィンドウの開始位置（学習バッチの直後）
    test_start = pos + BATCH_SIZE

    # スイッチ後の遷移ウィンドウに入るステップを post-switch とする
    post = np.zeros(n_steps, dtype=bool)
    for sw in SWITCHES:
        post |= (test_start >= sw) & (test_start < sw + POST_SWITCH_STEPS * BATCH_SIZE)

    valid = np.arange(n_steps) >= eval_start          # ウォームアップ除外
    stable_mask = valid & ~post
    post_mask = valid & post

    table = {}  # table[(period, metric)][display_name] = value
    for period, mask in [("stable", stable_mask), ("post_switch", post_mask)]:
        for metric in ["ll", "acc"]:
            row = {}
            for key, name in METHODS:
                arr = d[f"{key}_{metric}"]
                row[name] = float(arr[mask].mean())
            table[(period, metric)] = row

    n_post = int(post_mask.sum())
    n_stable = int(stable_mask.sum())
    return table, eval_start, n_stable, n_post


def write_csv(table, out_path):
    """期間分解テーブルを CSV に書き出す。"""
    display_names = [name for _, name in METHODS]
    with open(out_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["period", "metric"] + display_names)
        for period in ["stable", "post_switch"]:
            for metric in ["ll", "acc"]:
                row = table[(period, metric)]
                writer.writerow(
                    [period, metric] + [f"{row[name]:.6f}" for name in display_names]
                )


def main():
    n_particles = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    npz_path = os.path.join(INPUT_DIR, f"results_N{n_particles}_seed{seed}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"結果ファイルが見つかりません: {npz_path}")

    table, eval_start, n_stable, n_post = compute_phase_table(npz_path)

    out_path = os.path.join(
        OUTPUT_DIR, f"email_binary_phase_split_N{n_particles}_seed{seed}.csv"
    )
    write_csv(table, out_path)

    # コンソールにも要約を表示
    print(f"入力: {os.path.relpath(npz_path, _PROJECT_ROOT)}")
    print(f"ウォームアップ終了ステップ (eval_start): {eval_start}")
    print(f"安定期ステップ数: {n_stable}, スイッチ後ステップ数: {n_post}")
    print()
    header = f"{'period':12s} {'metric':6s}" + "".join(
        f"{name:>9s}" for _, name in METHODS
    )
    print(header)
    for period in ["stable", "post_switch"]:
        for metric in ["ll", "acc"]:
            row = table[(period, metric)]
            line = f"{period:12s} {metric:6s}" + "".join(
                f"{row[name]:9.3f}" for _, name in METHODS
            )
            print(line)
    print()
    print(f"出力: {os.path.relpath(out_path, _PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
