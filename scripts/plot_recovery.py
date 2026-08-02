#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-switch 回復曲線のプロット (Old_version/0729 figure1 の現行版)

Old_version/0729/figures/figure1_regime_switch_recovery_log_zoom.png を、
現在の outputs/{benchmark}/main/runs/*/metrics.npz から再現する。旧図は
SGD/PF/WSPF-B/WSPF-A の4手法だったが、現行で追加した PH-SGD / Window-SGD も
含めて全手法をプロットする。

集計は src.evaluation.recovery_curve と同順序:
  各シードで各 lag τ の指標を全スイッチ点で平均 (スイッチ横断) → シード横断で
  平均し、帯は ±1 SD (シード間)。

- regression: y = Test MSE (log スケール, 旧図と同じ)。
- 分類 (insects): y = Test error = 1 − accuracy (線形; 誤り率は桁をまたがない)。
  --metric で macro_f1 誤り (1 − macro_f1) にも切替可。

使い方:
    python scripts/plot_recovery.py --benchmark regression
    python scripts/plot_recovery.py --benchmark insects
    python scripts/plot_recovery.py --benchmark insects --metric macro_f1
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.evaluation import recovery_curve   # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ディレクトリ接頭辞 → 表示名 (プロット順)
METHOD_ORDER = [
    ("sgd", "SGD"),
    ("ph_sgd", "PH-SGD"),
    ("window_sgd", "Window-SGD"),
    ("pf", "PF"),
    ("wspf_a", "WSPF-A"),
    ("wspf_b", "WSPF-B"),
]
# 旧図の配色を踏襲しつつ、追加2手法に別色を割当
COLORS = {
    "SGD": "#7f7f7f",         # gray (旧図)
    "PH-SGD": "#ff7f0e",      # orange (新規)
    "Window-SGD": "#9467bd",  # purple (新規)
    "PF": "#1f77b4",          # blue (旧図)
    "WSPF-A": "#2ca02c",      # green (旧図)
    "WSPF-B": "#d62728",      # crimson (旧図)
}


def _runs_for(benchmark, prefix):
    """method prefix の全 seed run ディレクトリ (zone なし前提)。"""
    base = os.path.join(_REPO_ROOT, "outputs", benchmark, "main", "runs")
    dirs = sorted(glob.glob(os.path.join(base, f"{prefix}_seed*")))
    # gefcom のような zone 付きは除外 (name に _zone を含む) — regression/insects は無い
    return [d for d in dirs if "_zone" not in os.path.basename(d)]


def _series_and_switches(run_dir, metric_key, invert):
    """1 run の (指標時系列, switch点配列) を返す。invert=True は 1−metric。"""
    z = np.load(os.path.join(run_dir, "metrics.npz"))
    if metric_key not in z.files:
        raise KeyError(f"{metric_key} が {run_dir} にありません "
                       f"(利用可能: {[k for k in z.files if k.startswith('metric_')]})")
    ts = np.asarray(z[metric_key], dtype=np.float64)
    if invert:
        ts = 1.0 - ts
    sw = np.where(np.asarray(z["switch_mask"], dtype=bool))[0].tolist()
    return ts, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / insects (main/runs を持つもの)")
    ap.add_argument("--metric", default=None,
                    help="回帰: mse (既定)。分類: accuracy(既定,1−acc) / macro_f1")
    ap.add_argument("--max-lag", type=int, default=11,
                    help="τ の点数 (既定 11 = 0..10, 旧図と同じ)")
    ap.add_argument("--logy", dest="logy", action="store_true", default=None)
    ap.add_argument("--no-logy", dest="logy", action="store_false")
    ap.add_argument("--band", choices=["sem", "sd"], default="sem",
                    help="帯: sem=±1標準誤差(既定,旧図と同じtight) / sd=±1標準偏差")
    ap.add_argument("--overlay", default=None,
                    help="別ベンチの手法を破線・同色で重畳。書式 'bench:m1,m2' "
                         "(例 regression_strat:WSPF-A,WSPF-B)。ラベルは (strat) 付き")
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    # ベンチマークの task_type を config から判定
    cfg_path = os.path.join(_REPO_ROOT, "outputs", args.benchmark,
                            "main", "config.json")
    import json
    cfg = json.load(open(cfg_path))
    is_reg = cfg.get("task_type") == "regression"

    # 指標キーと y 軸設定
    if is_reg:
        metric_key = "metric_" + (args.metric or "mse")
        invert = False
        ylabel = "Test MSE (log scale)" if (args.logy is not False) \
            else "Test MSE"
        logy = True if args.logy is None else args.logy
        err_name = "MSE"
    else:
        base_metric = args.metric or "accuracy"
        metric_key = "metric_" + base_metric
        invert = True                       # 誤り率 = 1 − 指標
        ylabel = f"Test error (1 − {base_metric})"
        logy = False if args.logy is None else args.logy
        err_name = f"1−{base_metric}"

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    switch_points_ref = None
    plotted = 0
    curve_lo, curve_hi = np.inf, -np.inf   # ylim は帯でなく曲線基準で決める
    tau = np.arange(args.max_lag)
    for prefix, disp in METHOD_ORDER:
        runs = _runs_for(args.benchmark, prefix)
        if not runs:
            print(f"  [skip] {disp}: run が見つかりません")
            continue
        mse_ts, switches = [], None
        for d in runs:
            ts, sw = _series_and_switches(d, metric_key, invert)
            mse_ts.append(ts)
            switches = sw                     # 同一ベンチは seed 間で switch 一致
        if switch_points_ref is None:
            switch_points_ref = switches
        rec = recovery_curve(mse_ts, switches, max_lag=args.max_lag)
        curve, sd = rec["curve"], rec["std"]
        n_seed = len(mse_ts)
        band = sd / np.sqrt(max(n_seed, 1)) if args.band == "sem" else sd
        c = COLORS.get(disp, None)
        ax.plot(tau, curve, "-o", color=c, label=disp, lw=2, ms=5, zorder=3)
        ax.fill_between(tau, curve - band, curve + band, color=c, alpha=0.18,
                        lw=0, zorder=1)
        curve_lo = min(curve_lo, float(np.nanmin(curve)))
        curve_hi = max(curve_hi, float(np.nanmax(curve)))
        plotted += 1
        print(f"  {disp}: τ0={curve[0]:.3f}  τ{args.max_lag-1}={curve[-1]:.3f}"
              f"  (n_seed={n_seed})")

    # --- 重畳 (別ベンチの手法を破線・同色で; fixed vs stratified 比較用) ---
    if args.overlay:
        ov_bench, ov_methods = args.overlay.split(":")
        ov_set = set(m.strip() for m in ov_methods.split(","))
        disp_to_prefix = {d: p for p, d in METHOD_ORDER}
        for disp in [d for _, d in METHOD_ORDER if d in ov_set]:
            prefix = disp_to_prefix[disp]
            runs = _runs_for(ov_bench, prefix)
            if not runs:
                print(f"  [skip overlay] {disp}: {ov_bench} に run なし")
                continue
            mse_ts, switches = [], None
            for d in runs:
                ts, sw = _series_and_switches(d, metric_key, invert)
                mse_ts.append(ts)
                switches = sw
            rec = recovery_curve(mse_ts, switches, max_lag=args.max_lag)
            curve, sd = rec["curve"], rec["std"]
            band = sd / np.sqrt(max(len(mse_ts), 1)) if args.band == "sem" else sd
            c = COLORS.get(disp, None)
            ax.plot(tau, curve, "--s", color=c, label=f"{disp} (strat)",
                    lw=2, ms=4, zorder=4, markerfacecolor="white")
            ax.fill_between(tau, curve - band, curve + band, color=c,
                            alpha=0.10, lw=0, zorder=1)
            curve_lo = min(curve_lo, float(np.nanmin(curve)))
            curve_hi = max(curve_hi, float(np.nanmax(curve)))
            print(f"  {disp} (strat): τ0={curve[0]:.3f} "
                  f"τ{args.max_lag-1}={curve[-1]:.3f} (n_seed={len(mse_ts)})")

    if plotted == 0:
        raise SystemExit("プロット対象の run がありません。")

    # ylim は曲線の範囲に合わせる (帯の暴れで軸が潰れるのを防ぐ; 旧図の zoom 相当)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(curve_lo * 0.6, curve_hi * 1.6)
    else:
        pad = 0.06 * (curve_hi - curve_lo)
        ax.set_ylim(curve_lo - pad, curve_hi + pad)
    ax.set_xlabel(r"$\tau$ (steps after regime switch)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f"Post-Switch Recovery ({args.benchmark}, "
                 f"{'log ' if logy else ''}first {args.max_lag-1} steps)",
                 fontsize=14)
    ax.set_xlim(0, args.max_lag - 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=11, framealpha=0.95)
    n_sw = len(switch_points_ref) if switch_points_ref else 0
    band_lab = "±1 SE" if args.band == "sem" else "±1 SD"
    fig.text(0.99, 0.01,
             f"band = {band_lab} across seeds; averaged over {n_sw} switches, "
             f"10 seeds; metric={err_name}",
             ha="right", va="bottom", fontsize=8, color="#666")
    fig.tight_layout()

    out = args.outfile or os.path.join(
        _REPO_ROOT, "outputs", args.benchmark, "figures",
        "recovery_post_switch.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"保存: {out}  (switches={switch_points_ref})")


if __name__ == "__main__":
    main()
