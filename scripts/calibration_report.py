#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
較正・回復・月別性能レポート (R2-5)

査読対応 R2-5 の予測分布評価を 1 か所に集約する:

  回帰 (regression / gefcom):
    - カバレッジ vs 名目水準 (0.5/0.8/0.9/0.95): 報告区間の
      coverage_{lvl}/width_{lvl} をシード平均 (mean±std)。
    - 回復曲線 (regression のみ; 既知スイッチ点あり):
      各シードの報告区間 MSE 時系列を recovery_curve に渡し、
      post-switch lag 別に集計。回復面積を PF と対応比較する。
      ※ スイッチ事象は独立標本として扱わない。まずシード内で
        スイッチ横断平均を取り (recovery_curve がやる)、その後 10 シード
        にわたって paired_compare する。
    - 月別性能 (gefcom のみ; 人工スイッチ点なし): 報告区間ステップを
      タイムスタンプの月へ写像し、月別平均 MSE を報告する
      (回復曲線の代替, R2 指摘)。

  分類 (email):
    - 報告区間のサンプル単位 probs/labels をシード横断で連結し、
      Brier / ECE と信頼度図 (reliability diagram) を出す。

出力: outputs/<benchmark>/calibration/calibration_report.{csv,txt,tex}
      (tidy schema: task, method, metric, level, mean, std)
      + PNG (reliability_*.png / recovery_regression.png / monthly_gefcom.png)
      は matplotlib が使える場合のみ。失敗しても CSV は必ず出す。

使い方:
    python scripts/calibration_report.py --benchmark regression
    python scripts/calibration_report.py --benchmark gefcom
    python scripts/calibration_report.py --benchmark email
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask)
from src.evaluation import (run_seeds, mean_std, recovery_curve,
                            paired_compare, brier_ece, write_table)

# ----------------------------------------------------------------------
# matplotlib は任意。設定を済ませてから import し、失敗しても致命傷にしない。
# ----------------------------------------------------------------------
os.environ.setdefault("MPLCONFIGDIR", "/workspace/WSPF/.tmp/mpl")
_HAVE_MPL = False
try:  # pragma: no cover - 描画は環境依存
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

LEVELS = (0.5, 0.8, 0.9, 0.95)
N_POST_SWITCH = 10   # 回復曲線の lag 数 (post_switch_lag と同義)


# ======================================================================
# 共通ヘルパ
# ======================================================================
def _methods(cfg):
    """較正対象の手法 (粒子フィルタ + 点推定ベースライン; NoChange 除外)。"""
    return [m for m in cfg["methods"] if m != "NoChange"]


def _report_mean(result, key):
    """1 実行結果の報告区間(straddle 除外)での指標平均。"""
    mask = region_mask(result, "report")
    arr = np.asarray(result["metrics"].get(key, []), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    v = arr[mask]
    return float(np.nanmean(v)) if v.size else float("nan")


# ======================================================================
# 回帰: カバレッジ vs 名目
# ======================================================================
def coverage_rows(cfg, results_by_method):
    """報告区間のカバレッジ/区間幅を水準別にシード集計する。"""
    rows = []
    for m, results in results_by_method.items():
        for lvl in LEVELS:
            cov_key = f"coverage_{lvl:.2f}"
            wid_key = f"width_{lvl:.2f}"
            cov = [_report_mean(r, cov_key) for r in results]
            wid = [_report_mean(r, wid_key) for r in results]
            cmu, csd = mean_std(cov)
            wmu, wsd = mean_std(wid)
            rows.append({"task": "regression", "method": m,
                         "metric": "coverage", "level": f"{lvl:.2f}",
                         "mean": cmu, "std": csd})
            rows.append({"task": "regression", "method": m,
                         "metric": "width", "level": f"{lvl:.2f}",
                         "mean": wmu, "std": wsd})
    return rows


# ======================================================================
# 回帰(スイッチあり): 回復曲線 + PF との対応比較
# ======================================================================
def recovery_analysis(cfg, results_by_method, switch_points):
    """
    post-switch 回復曲線を計算する。

    Returns
    -------
    rows : list[dict]           # tidy 行
    curves : dict[method] -> (curve, std)  # 描画用
    """
    rows = []
    curves = {}
    if not switch_points:
        return rows, curves

    # 報告区間に完全に収まるスイッチ点のみ採用(選択区間の切替は除外)。
    ref = next(iter(results_by_method.values()))[0]
    rep_mask = region_mask(ref, "report")
    T = rep_mask.size
    valid_sw = [sp for sp in switch_points
                if 0 <= sp < T and rep_mask[sp]]
    if not valid_sw:
        return rows, curves

    # 各手法の per-seed 回復面積(lag 平均)を保持し、PF と対応比較する。
    area_by_method = {}
    for m, results in results_by_method.items():
        mse_ts = []
        for r in results:
            mask = region_mask(r, "report")
            mse = np.asarray(r["metrics"]["mse"], dtype=np.float64).copy()
            mse[~mask] = np.nan            # 報告区間のみ
            mse_ts.append(mse)
        rec = recovery_curve(mse_ts, valid_sw, max_lag=N_POST_SWITCH)
        curves[m] = (rec["curve"], rec["std"])
        # per-seed 回復面積 = lag 平均(シード内でスイッチ横断は済み)
        area = np.nanmean(rec["per_seed"], axis=1)
        area_by_method[m] = area
        amu, asd = mean_std(area)
        rows.append({"task": "regression", "method": m,
                     "metric": "recovery_area", "level": "",
                     "mean": amu, "std": asd})

    # PF との対応比較(シード横断の paired t)
    if "PF" in area_by_method:
        pf_area = area_by_method["PF"]
        for m, area in area_by_method.items():
            if m == "PF":
                continue
            cmp = paired_compare(area, pf_area)
            rows.append({"task": "regression", "method": m,
                         "metric": "recovery_area_diff_vs_PF", "level": "",
                         "mean": cmp["mean_diff"], "std": cmp["p"]})
    return rows, curves


# ======================================================================
# gefcom: 月別性能(回復曲線の代替)
# ======================================================================
def monthly_analysis(cfg, benchmark, results_by_method):
    """
    報告区間ステップを月へ写像し、月別平均 MSE をシード集計する。

    Returns
    -------
    rows : list[dict]
    monthly : dict[method] -> (months(sorted), mean_per_month, std_per_month)
    """
    rows = []
    monthly = {}
    timestamps = getattr(getattr(benchmark, "loader", None), "timestamps", None)
    if timestamps is None:
        return rows, monthly

    for m, results in results_by_method.items():
        # per-seed に month -> mean MSE を作り、月ごとにシード横断集計。
        per_seed_month = []
        for r in results:
            rep_mask = region_mask(r, "report")
            mse = np.asarray(r["metrics"]["mse"], dtype=np.float64)
            test_idx = r.get("test_indices", [])
            bucket = {}
            for step in range(len(mse)):
                if not rep_mask[step]:
                    continue
                if step >= len(test_idx):
                    continue
                idx = np.asarray(test_idx[step], dtype=int)
                if idx.size == 0:
                    continue
                gi = int(idx[0])
                if gi < 0 or gi >= len(timestamps):
                    continue
                month = timestamps[gi].month
                if np.isfinite(mse[step]):
                    bucket.setdefault(month, []).append(float(mse[step]))
            per_seed_month.append({mo: float(np.mean(v))
                                   for mo, v in bucket.items()})
        # 月ごとにシード横断集計
        all_months = sorted({mo for d in per_seed_month for mo in d})
        means, stds = [], []
        for mo in all_months:
            vals = [d[mo] for d in per_seed_month if mo in d]
            mu, sd = mean_std(vals)
            means.append(mu)
            stds.append(sd)
            rows.append({"task": "gefcom", "method": m,
                         "metric": "monthly_mse", "level": f"month_{mo:02d}",
                         "mean": mu, "std": sd})
        monthly[m] = (all_months, means, stds)
    return rows, monthly


# ======================================================================
# 分類 (email): Brier / ECE / 信頼度図
# ======================================================================
def classification_analysis(cfg, results_by_method):
    """
    報告区間のサンプル単位 probs/labels をシード横断で連結し、
    Brier / ECE と信頼度図データを返す。
    """
    rows = []
    reliability = {}
    for m, results in results_by_method.items():
        probs = np.concatenate(
            [np.asarray(r["predictions"].get("probs", []), np.float64)
             for r in results]) if results else np.empty(0)
        labels = np.concatenate(
            [np.asarray(r["predictions"].get("y", []), np.float64)
             for r in results]) if results else np.empty(0)
        if probs.size == 0 or labels.size == 0:
            rows.append({"task": "classification", "method": m,
                         "metric": "brier", "level": "",
                         "mean": float("nan"), "std": float("nan")})
            continue
        brier, ece, rel_x, rel_y = brier_ece(probs, labels, n_bins=10)
        reliability[m] = (rel_x, rel_y)
        rows.append({"task": "classification", "method": m,
                     "metric": "brier", "level": "",
                     "mean": float(brier), "std": ""})
        rows.append({"task": "classification", "method": m,
                     "metric": "ece", "level": "",
                     "mean": float(ece), "std": ""})
    return rows, reliability


# ======================================================================
# 描画(すべて任意; 失敗しても致命傷にしない)
# ======================================================================
def _plot_recovery(curves, out_dir):
    if not (_HAVE_MPL and curves):
        return
    try:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(6, 4))
        lags = np.arange(N_POST_SWITCH)
        for m, (curve, std) in curves.items():
            ax.plot(lags, curve, marker="o", label=m)
            # 帯は「シード間ばらつき(±1SD)」であって信頼区間ではない(R2 指摘)。
            ax.fill_between(lags, curve - std, curve + std, alpha=0.15)
        ax.set_xlabel("lag after switch (steps)")
        ax.set_ylabel("report-region MSE")
        ax.set_title("Post-switch recovery (band = ±1 SD across seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "recovery_regression.png"), dpi=120)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


def _plot_monthly(monthly, out_dir):
    if not (_HAVE_MPL and monthly):
        return
    try:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, (months, means, stds) in monthly.items():
            ax.errorbar(months, means, yerr=stds, marker="o", capsize=2,
                        label=m)
        ax.set_xlabel("month")
        ax.set_ylabel("report-region MSE")
        ax.set_title("GEFCom monthly performance (err = ±1 SD across seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "monthly_gefcom.png"), dpi=120)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


def _plot_reliability(reliability, out_dir):
    if not (_HAVE_MPL and reliability):
        return
    for m, (rel_x, rel_y) in reliability.items():
        try:  # pragma: no cover
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            ax.plot(rel_x, rel_y, marker="o", label=m)
            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("empirical frequency")
            ax.set_title(f"Reliability diagram: {m}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            fig.tight_layout()
            from src.evaluation import sanitize
            fig.savefig(os.path.join(out_dir, f"reliability_{sanitize(m)}.png"),
                        dpi=120)
            plt.close(fig)
        except Exception:  # noqa: BLE001
            pass


# ======================================================================
# メイン
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / gefcom / email または config パス")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n_main = cfg["n_particles"]["main"]
    methods = _methods(cfg)
    task_type = cfg["task_type"]
    bench_name = cfg["benchmark"]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "calibration")
    os.makedirs(out_dir, exist_ok=True)

    # --- 各手法を評価シードで実行(結果を使い回す)---
    # タイムスタンプ参照用に 1 つ benchmark を作っておく(gefcom 月別で使用)。
    bench_ref = build_benchmark(cfg)
    results_by_method = {}
    for m in methods:
        params = get_params(selected, m, n_main)
        bench = build_benchmark(cfg)
        results_by_method[m] = run_seeds(m, bench, n_main, params, eval_seeds)
        print(f"[run] {m}: {len(eval_seeds)} seeds done")

    rows = []
    curves, monthly, reliability = {}, {}, {}

    if task_type == "regression":
        rows += coverage_rows(cfg, results_by_method)
        if bench_name == "gefcom":
            # 人工スイッチ点なし → 月別性能を回復曲線の代わりに使う。
            mrows, monthly = monthly_analysis(cfg, bench_ref,
                                              results_by_method)
            rows += mrows
        else:
            # 既知スイッチ点あり → post-switch 回復曲線。
            switch_points = list(getattr(bench_ref, "switch_points", []))
            rrows, curves = recovery_analysis(cfg, results_by_method,
                                              switch_points)
            rows += rrows
    else:
        crows, reliability = classification_analysis(cfg, results_by_method)
        rows += crows

    # --- テーブル書き出し(必ず実行)---
    base = os.path.join(out_dir, "calibration_report")
    write_table(rows, base, formats=("csv", "txt", "tex"))
    print(f"保存: {base}.{{csv,txt,tex}}  ({len(rows)} 行)")

    # --- 図(任意)---
    _plot_recovery(curves, out_dir)
    _plot_monthly(monthly, out_dir)
    _plot_reliability(reliability, out_dir)
    if not _HAVE_MPL:
        print("[注意] matplotlib 不使用: PNG はスキップ(CSV は出力済み)。")

    # --- 概要の表示 ---
    print("\n=== 較正レポート概要 ===")
    for r in rows:
        lvl = f" level={r['level']}" if r.get("level") else ""
        sd = r.get("std", "")
        sd_s = f" ± {sd:.4f}" if isinstance(sd, float) and np.isfinite(sd) else ""
        mu = r.get("mean")
        mu_s = f"{mu:.4f}" if isinstance(mu, float) else str(mu)
        print(f"  [{r['task']}] {r['method']:11s} {r['metric']:24s}"
              f"{lvl}: {mu_s}{sd_s}")


if __name__ == "__main__":
    main()
