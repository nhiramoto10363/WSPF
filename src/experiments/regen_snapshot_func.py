#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snapshot_func_t{t}_N{n}.png だけを再生成するユーティリティ。

regression_regime_switch.py のフルパイプライン（グリッドサーチ＋多シード）を
回さずに、スナップショット図のみを seed=0 で再現する。凡例レイアウトなどの
体裁変更を確認するために使う。

使い方:
  python experiments/regen_snapshot_func.py [t] [n_particles]
  （既定: t=305, n_particles=100）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.regression_regime_switch import (
    run_single,
    save_snapshot_plots,
    load_grid_search_params,
    NeuralNetRegression,
    INPUT_DIM,
    HIDDEN_DIM,
    DEFAULT_BETA,
    DEFAULT_ETA,
    DEFAULT_SIGMA_SYS,
    DEFAULT_PRIOR_STD,
)


def get_params(n_particles):
    """グリッドサーチ結果（無ければ既定値）から最良ハイパラを取得。"""
    gs = load_grid_search_params()
    key = str(n_particles)
    if gs is not None and key in gs:
        entry = gs[key]
        best_pf = entry["best_pf"]
        best_wspf_b = entry["best_wspf_b"]
        best_wspf_a = entry.get("best_wspf_a") or {**best_wspf_b, "beta": DEFAULT_BETA}
    else:
        print("  grid search 結果なし -> 既定ハイパラを使用")
        best_pf = {"eta": DEFAULT_ETA, "sigma_sys": DEFAULT_SIGMA_SYS,
                   "prior_std": DEFAULT_PRIOR_STD}
        best_wspf_b = dict(best_pf)
        best_wspf_a = {**best_pf, "beta": DEFAULT_BETA}
    return best_pf, best_wspf_b, best_wspf_a


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 305
    n_particles = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "regression_regime_switch"
    )

    best_pf, best_wspf_b, best_wspf_a = get_params(n_particles)
    print(f"Re-running seed=0, N={n_particles}, snapshot at t={t} ...")
    snap_result = run_single(
        0, n_particles, best_pf, best_wspf_b, best_wspf_a,
        snapshot_times=[t],
    )
    model = NeuralNetRegression(
        INPUT_DIM, HIDDEN_DIM, output_dim=1, activation="tanh"
    )
    paths = save_snapshot_plots(
        output_dir, model, snap_result["snapshots"],
        snap_result["switch_times"], n_particles,
    )
    print("Saved:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
