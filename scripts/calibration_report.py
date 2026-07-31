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
    - 報告区間のサンプル単位 probs/labels を **シードごとに** Brier / ECE を
      計算し、シード横断で mean±SD を報告する(プールしない)。さらに各手法の
      per-seed ECE / Brier を PF と対応比較する。信頼度図 (reliability diagram)
      だけは全シードをプールして描く(唯一プールする箇所)。

査読 R2-5 フォローアップ対応:
  1. 回復面積の PF 比較で p 値を std 列に入れない。mean_difference /
     std_difference / paired_t_p / wilcoxon_p を別カラムに分離する。
  2. Brier / ECE はシードごとに算出してから集計する(プール禁止; 信頼度図のみ例外)。
  3. 回復は「各 lag での対応検定」を主張するため、lag τ=0..K-1 ごとに PF 対応検定を
     行い、K 個の p 値に Holm 補正を掛ける(recovery_lag_paired 行)。
  4. GEFCom は全 zone をループし、grid_search が保存した σ_obs を使う。zone 列で
     区別し、zone2/zone3 も実際に評価する。

出力: outputs/<benchmark>/calibration/calibration_report.{csv,txt,tex}
      (tidy 上位集合スキーマ: task, method, metric, level, mean, std,
       mean_difference, std_difference, paired_t_p, paired_t_p_holm,
       wilcoxon_p, zone。行によっては一部カラムが空でよい。)
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
                            paired_compare, wilcoxon_signed, brier_ece,
                            write_table)

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

# tidy テーブルの一貫した上位集合スキーマ(行によって一部空でよい)。
# _row() を必ず経由することで CSV のカラム順を固定する。
_SCHEMA = ("task", "method", "metric", "level", "mean", "std",
           "mean_difference", "std_difference", "paired_t_p",
           "paired_t_p_holm", "wilcoxon_p", "zone")


def _row(task, method, metric, level="", mean="", std="",
         mean_difference="", std_difference="", paired_t_p="",
         paired_t_p_holm="", wilcoxon_p="", zone=""):
    """スキーマ固定の 1 行を作る(欠損カラムは空文字)。"""
    return {"task": task, "method": method, "metric": metric,
            "level": level, "mean": mean, "std": std,
            "mean_difference": mean_difference,
            "std_difference": std_difference,
            "paired_t_p": paired_t_p, "paired_t_p_holm": paired_t_p_holm,
            "wilcoxon_p": wilcoxon_p, "zone": zone}


def _paired_diffs(a, b):
    """対応する有限値ペアの差 (a-b) を返す(長さ違いは短い方に合わせる)。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask] - b[mask]


def _holm_correction(pvals):
    """
    Holm-Bonferroni 補正を p 値ベクトルに掛ける(有限値のみ対象)。

    statsmodels があればそれを使い、無ければ手実装にフォールバックする。
    手実装: 昇順に並べ (m-rank)*p を掛け、単調非減少を強制し、1 でクリップ。
    (m = 有限な検定数)。NaN の p 値は補正結果も NaN のまま残す。
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    adj = np.full(p.size, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return adj
    fp = p[finite]
    try:  # statsmodels があれば利用(guard import)
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(fp, method="holm")
        adj[finite] = p_adj
        return adj
    except Exception:  # noqa: BLE001 - Holm を手実装にフォールバック
        m = fp.size
        order = np.argsort(fp)
        running = 0.0
        for rank, k in enumerate(order):
            running = max(running, (m - rank) * float(fp[k]))
            adj[finite[k]] = min(running, 1.0)
        return adj


# ======================================================================
# 共通ヘルパ
# ======================================================================
def _methods(cfg):
    """較正対象の手法 (粒子フィルタ + 点推定ベースライン; NoChange 除外)。"""
    return [m for m in cfg["methods"] if m != "NoChange"]


def _run_methods(cfg, selected, methods, n_main, eval_seeds, **ctx):
    """全手法を評価シードで実行する。ctx はベンチマーク構築の override
    (GEFCom の zone / noise_std 等)。(results_by_method, bench_ref) を返す。"""
    results_by_method = {}
    bench_ref = None
    for m in methods:
        params = get_params(selected, m, n_main)
        bench = build_benchmark(cfg, **ctx)
        if bench_ref is None:
            bench_ref = bench
        results_by_method[m] = run_seeds(m, bench, n_main, params, eval_seeds)
        tag = f" {ctx}" if ctx else ""
        print(f"[run] {m}{tag}: {len(eval_seeds)} seeds done")
    return results_by_method, bench_ref


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
def coverage_rows(cfg, results_by_method, task="regression", zone=""):
    """報告区間のカバレッジ/区間幅/CRPS を水準別にシード集計する。

    task/zone を明示して行に付与する(GEFCom は zone 別に呼ぶ)。
    """
    rows = []
    for m, results in results_by_method.items():
        for lvl in LEVELS:
            cov_key = f"coverage_{lvl:.2f}"
            wid_key = f"width_{lvl:.2f}"
            cov = [_report_mean(r, cov_key) for r in results]
            wid = [_report_mean(r, wid_key) for r in results]
            cmu, csd = mean_std(cov)
            wmu, wsd = mean_std(wid)
            rows.append(_row(task, m, "coverage", level=f"{lvl:.2f}",
                             mean=cmu, std=csd, zone=zone))
            rows.append(_row(task, m, "width", level=f"{lvl:.2f}",
                             mean=wmu, std=wsd, zone=zone))
        crps = [_report_mean(r, "crps") for r in results]
        qmu, qsd = mean_std(crps)
        rows.append(_row(task, m, "crps", mean=qmu, std=qsd, zone=zone))
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

    # 各手法の per-seed 回復面積(lag 平均)と per-seed×lag 行列を保持する。
    # recovery_curve の per_seed は (n_seed, K): 各シードでスイッチ横断平均済み。
    area_by_method = {}
    lagmat_by_method = {}
    for m, results in results_by_method.items():
        mse_ts = []
        for r in results:
            mask = region_mask(r, "report")
            mse = np.asarray(r["metrics"]["mse"], dtype=np.float64).copy()
            mse[~mask] = np.nan            # 報告区間のみ
            mse_ts.append(mse)
        rec = recovery_curve(mse_ts, valid_sw, max_lag=N_POST_SWITCH)
        curves[m] = (rec["curve"], rec["std"])
        lagmat_by_method[m] = rec["per_seed"]           # (n_seed, K)
        # per-seed 回復面積 = lag 平均(シード内でスイッチ横断は済み)
        area = np.nanmean(rec["per_seed"], axis=1)
        area_by_method[m] = area
        amu, asd = mean_std(area)
        rows.append(_row("regression", m, "recovery_area", mean=amu, std=asd))

    # PF との対応比較(シード横断)。p 値は std 列に入れず専用カラムに分離する。
    if "PF" in area_by_method:
        pf_area = area_by_method["PF"]
        pf_lag = lagmat_by_method["PF"]
        for m in area_by_method:
            if m == "PF":
                continue
            area = area_by_method[m]
            # (a) 回復面積差: paired-t と Wilcoxon を別カラムに、std は差の SD。
            cmp = paired_compare(area, pf_area)
            wil = wilcoxon_signed(area, pf_area)
            diffs = _paired_diffs(area, pf_area)
            sd_diff = float(np.std(diffs)) if diffs.size else float("nan")
            rows.append(_row("regression", m, "recovery_area_diff_vs_PF",
                             mean_difference=cmp["mean_diff"],
                             std_difference=sd_diff,
                             paired_t_p=cmp["p"], wilcoxon_p=wil["p"]))

            # (b) lag 別対応検定 + Holm 補正。各 lag τ で長さ n_seed の対応比較。
            mmat = lagmat_by_method[m]
            K = mmat.shape[1]
            lag_md, lag_p = [], []
            for tau in range(K):
                c = paired_compare(mmat[:, tau], pf_lag[:, tau])
                lag_md.append(c["mean_diff"])
                lag_p.append(c["p"])
            lag_p_holm = _holm_correction(lag_p)
            for tau in range(K):
                rows.append(_row("regression", m, "recovery_lag_paired",
                                 level=f"lag_{tau}",
                                 mean_difference=lag_md[tau],
                                 paired_t_p=lag_p[tau],
                                 paired_t_p_holm=float(lag_p_holm[tau])))
    return rows, curves


# ======================================================================
# gefcom: 月別性能(回復曲線の代替)
# ======================================================================
def monthly_analysis(cfg, benchmark, results_by_method, zone=""):
    """
    報告区間ステップを月へ写像し、月別平均 MSE をシード集計する(zone 別)。

    Returns
    -------
    rows : list[dict]
    monthly : dict[label] -> (months(sorted), mean_per_month, std_per_month)
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
            rows.append(_row("gefcom", m, "monthly_mse",
                             level=f"month_{mo:02d}", mean=mu, std=sd,
                             zone=zone))
        label = f"{m} z{zone}" if zone != "" else m
        monthly[label] = (all_months, means, stds)
    return rows, monthly


# ======================================================================
# 分類 (email): Brier / ECE / 信頼度図
# ======================================================================
def classification_analysis(cfg, results_by_method):
    """
    報告区間のサンプル単位 probs/labels から **シードごとに** Brier / ECE を
    算出し、シード横断で mean±SD を報告する(プールしない)。さらに各手法の
    per-seed ECE / Brier を PF と対応比較する。

    信頼度図 (reliability diagram) の rel_x/rel_y だけは全シードをプールして
    描く — ここが唯一プールする箇所(ビン内サンプルを増やして図を安定させるため)。
    """
    rows = []
    reliability = {}
    per_seed_brier = {}   # method -> ndarray(per-seed brier)
    per_seed_ece = {}     # method -> ndarray(per-seed ece)

    for m, results in results_by_method.items():
        briers, eces = [], []
        pooled_probs, pooled_labels = [], []
        for r in results:
            p = np.asarray(r["predictions"].get("probs", []),
                           np.float64).ravel()
            y = np.asarray(r["predictions"].get("y", []), np.float64).ravel()
            n = min(p.size, y.size)
            if n == 0:
                continue
            p, y = p[:n], y[:n]
            b, e, _, _ = brier_ece(p, y, n_bins=10)   # ← per seed
            briers.append(b)
            eces.append(e)
            pooled_probs.append(p)
            pooled_labels.append(y)
        per_seed_brier[m] = np.asarray(briers, dtype=np.float64)
        per_seed_ece[m] = np.asarray(eces, dtype=np.float64)

        bmu, bsd = mean_std(briers)
        emu, esd = mean_std(eces)
        rows.append(_row("classification", m, "brier", mean=bmu, std=bsd))
        rows.append(_row("classification", m, "ece", mean=emu, std=esd))

        # 信頼度図は全シードをプールして 1 本(唯一のプール; コメント参照)。
        if pooled_probs:
            allp = np.concatenate(pooled_probs)
            ally = np.concatenate(pooled_labels)
            _, _, rel_x, rel_y = brier_ece(allp, ally, n_bins=10)
            reliability[m] = (rel_x, rel_y)

    # PF との per-seed 対応比較(ECE / Brier)。p 値は専用カラムへ。
    if "PF" in per_seed_ece:
        for m in results_by_method:
            if m == "PF":
                continue
            for name, store in (("ece", per_seed_ece),
                                ("brier", per_seed_brier)):
                a, b = store.get(m), store.get("PF")
                cmp = paired_compare(a, b)
                diffs = _paired_diffs(a, b)
                sd_diff = float(np.std(diffs)) if diffs.size else float("nan")
                rows.append(_row("classification", m, f"{name}_diff_vs_PF",
                                 mean_difference=cmp["mean_diff"],
                                 std_difference=sd_diff,
                                 paired_t_p=cmp["p"]))
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

    rows = []
    curves, monthly, reliability = {}, {}, {}

    if task_type == "regression" and bench_name == "gefcom":
        # GEFCom: 全 zone をループし、grid_search が保存した σ_obs を使う(C2)。
        # 人工スイッチ点なし → 月別性能を回復曲線の代わりに使う。zone 別に評価。
        noise_by_zone = selected.get("gefcom_noise", {})
        zones = cfg["data"]["zones"]
        for z in zones:
            ctx = {"zone": z}
            ns = noise_by_zone.get(str(z))
            if ns is not None:                 # 無ければベンチマーク既定へフォールバック
                ctx["noise_std"] = ns
            res, bench_ref = _run_methods(cfg, selected, methods, n_main,
                                          eval_seeds, **ctx)
            rows += coverage_rows(cfg, res, task="gefcom", zone=str(z))
            mrows, zmonthly = monthly_analysis(cfg, bench_ref, res,
                                               zone=str(z))
            rows += mrows
            monthly.update(zmonthly)
            print(f"[zone {z}] 評価完了 (σ_obs={ns})")

    elif task_type == "regression":
        # 回帰(合成スイッチ): 単一コンテキスト(zone=None)。
        res, bench_ref = _run_methods(cfg, selected, methods, n_main,
                                      eval_seeds)
        rows += coverage_rows(cfg, res, task="regression")
        # 既知スイッチ点あり → post-switch 回復曲線 + lag 別対応検定(Holm)。
        switch_points = list(getattr(bench_ref, "switch_points", []))
        rrows, curves = recovery_analysis(cfg, res, switch_points)
        rows += rrows

    else:
        # 分類(email): 単一コンテキスト(zone=None)。
        res, _ = _run_methods(cfg, selected, methods, n_main, eval_seeds)
        crows, reliability = classification_analysis(cfg, res)
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
    def _num(v):
        return v if isinstance(v, float) and np.isfinite(v) else None

    print("\n=== 較正レポート概要 ===")
    for r in rows:
        lvl = f" level={r['level']}" if r.get("level") else ""
        zn = f" zone={r['zone']}" if r.get("zone") not in ("", None) else ""
        mu = _num(r.get("mean"))
        md = _num(r.get("mean_difference"))
        sd = _num(r.get("std"))
        pt = _num(r.get("paired_t_p"))
        pth = _num(r.get("paired_t_p_holm"))
        if mu is not None:                       # 通常の平均±SD 行
            val = f"{mu:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")
        elif md is not None:                     # 差分(PF 比較)行
            val = f"Δ={md:.4f}"
            if pt is not None:
                val += f" (p={pt:.3g}"
                val += f", p_holm={pth:.3g})" if pth is not None else ")"
        else:
            val = str(r.get("mean"))
        print(f"  [{r['task']}] {r['method']:11s} {r['metric']:24s}"
              f"{lvl}{zn}: {val}")


if __name__ == "__main__":
    main()
