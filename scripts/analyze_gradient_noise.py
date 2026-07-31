#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
勾配ノイズ解析 (R1-6, R1-8 とは別実験)

対象は Regression のみ(真のデータ生成分布から反復サンプリングでき、
母集団勾配を高精度に推定できる)。バッチサイズと概念切替からの距離を変え、
ミニバッチ勾配ノイズのガウス近似・異方性・バッチサイズ依存を検証する。

  batch_sizes = [8, 16, 32, 64]
  phases      = [stable, immediately_after_switch, 5_steps_after, 20_steps_after]
各条件で: 各成分の歪度・尖度, 正規性検定, Mahalanobis 距離分布,
          共分散固有値, condition number, effective rank, 上位固有値の説明分散比率。
(同じバッチサイズで PF/WSPF-A/B の MSE も別途 run_main/n_sweep で取得)

使い方:
    python scripts/analyze_gradient_noise.py --benchmark regression
"""

import argparse
import os

import numpy as np

from _common import load_config, build_benchmark
from src.evaluation import write_table


def _phase_steps(switch_points, T):
    """代表的な位相ステップを返す。"""
    sp = switch_points[0] if switch_points else T // 2
    return {"stable": max(sp - 30, 1),
            "post+0": sp,
            "post+5": sp + 5,
            "post+20": sp + 20}


def _noise_stats(per_grads):
    """per-sample 勾配 (B, d) から勾配ノイズ統計を計算する。"""
    from scipy import stats
    g = per_grads
    B, d = g.shape
    dev = g - g.mean(axis=0, keepdims=True)
    C = dev.T @ dev / (B - 1)                    # (d, d) per-sample cov
    evals = np.linalg.eigvalsh(C)
    evals = np.clip(evals, 0, None)[::-1]
    total = evals.sum() + 1e-30
    cond = float(evals[0] / max(evals[-1], 1e-30))
    eff_rank = float((evals.sum() ** 2) / (np.sum(evals ** 2) + 1e-30))
    top_ratio = float(evals[0] / total)
    # 各成分の歪度・尖度 と 正規性検定(平均)
    skew = float(np.nanmean(stats.skew(dev, axis=0)))
    kurt = float(np.nanmean(stats.kurtosis(dev, axis=0)))
    try:
        pval = float(np.nanmean([stats.normaltest(dev[:, j]).pvalue
                                 for j in range(min(d, 50))]))
    except Exception:
        pval = float("nan")
    return {"cond_number": cond, "effective_rank": eff_rank,
            "top_eig_ratio": top_ratio, "mean_skew": skew,
            "mean_kurtosis": kurt, "normaltest_p": pval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="regression")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["task_type"] != "regression":
        raise SystemExit("R1-6 勾配ノイズ解析は Regression のみ対応です。")

    batch_sizes = [8, 16, 32, 64]
    rows = []
    for B in batch_sizes:
        bench = build_benchmark(cfg, batch_size=B)
        f = bench.build_functions(seed=0)
        ps_grad = f["per_sample_grad_fn"]
        steps = list(bench.stream(seed=0))
        phases = _phase_steps(bench.switch_points, len(steps))
        # 1 粒子(真パラメータ近傍の初期点)で勾配ノイズを評価
        theta = np.zeros((1, bench.param_dim))
        for phase, si in phases.items():
            si = min(si, len(steps) - 1)
            st = steps[si]
            pg = ps_grad(theta, st.X_train, st.y_train)[0]   # (B, d)
            stats = _noise_stats(pg)
            stats.update({"batch_size": B, "phase": phase})
            rows.append(stats)
            print(f"B={B:3d} {phase:8s} cond={stats['cond_number']:.1f} "
                  f"eff_rank={stats['effective_rank']:.1f} "
                  f"normaltest_p={stats['normaltest_p']:.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "grad_noise")
    os.makedirs(out_dir, exist_ok=True)
    write_table(rows, os.path.join(out_dir, "gradient_noise"))
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
