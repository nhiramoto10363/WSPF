#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マッチド・ハイパーパラメータ実験 — Regression (R1-7)

PF / WSPF-A / WSPF-B に **同一の (η, σcd, σ0)** を与えて比較する。
これによりメソッド間の性能差が「ハイパラ最適化の差」ではなく
「補正項そのものの効果」であることを分離する。

共通ハイパラの選び方(確定): グリッドサーチの **PF 最良構成 (best_pf)** を
全メソッドに適用する(PF に最適化された条件下でも補正が改善するかを見る、
R1-7 の最も公平で厳しい比較)。WSPF-A の β は補正固有パラメータなので
グリッドの best_wspf_a[β] を用いる。

既存の regression_regime_switch.run_single を再利用し、best_pf=best_wspf_b=
共通ハイパラ、best_wspf_a=共通ハイパラ+β として呼ぶ。

出力:
  outputs/matched_hp/
    - matched_hp_regression.txt / .csv
"""

import sys
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

import src.experiments.regression_regime_switch as RR
from src.experiments.regression_regime_switch import (
    run_single, load_grid_search_params, compute_regime_split_metrics,
    N_PARTICLES_LIST, SEEDS, T, EVAL_START, N_POST_SWITCH,
    DEFAULT_ETA, DEFAULT_SIGMA_SYS, DEFAULT_PRIOR_STD, DEFAULT_BETA,
)

METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]
PARTICLE_METHODS = ["PF", "WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "matched_hp",
)


def build_matched_hp():
    """PF 最良構成を共通ハイパラとして各粒子数ぶん厳格に構築する。

    グリッド未実行・キー欠損時は明示的に失敗させる(無言フォールバック廃止)。
    """
    grid = load_grid_search_params()
    if grid is None:
        raise RuntimeError(
            "グリッド結果が見つかりません。先に "
            "grid_search_regression_regime_switch.py を実行してください。")
    matched_by_n = {}
    for n_p in N_PARTICLES_LIST:
        if str(n_p) not in grid:
            raise KeyError(f"グリッド JSON に N={n_p} がありません。")
        entry = grid[str(n_p)]
        for k in ("best_pf", "best_wspf_a"):
            if k not in entry:
                raise KeyError(
                    f"グリッド JSON に '{k}' がありません(旧キーの可能性)。"
                    "グリッドを再生成してください。")
        if "beta" not in entry["best_wspf_a"]:
            raise KeyError("best_wspf_a に 'beta' がありません。")
        best_pf = entry["best_pf"]
        beta = entry["best_wspf_a"]["beta"]
        matched = {
            "eta": best_pf["eta"],
            "sigma_sys": best_pf["sigma_sys"],
            "prior_std": best_pf["prior_std"],
        }
        matched_by_n[n_p] = (matched, beta, "grid best_pf")
    return matched_by_n


def paired_stats(a_pf, a_alt):
    """PF と代替メソッドの per-seed MSE 差に対する対応検定。"""
    a_pf = np.asarray(a_pf, dtype=np.float64)
    a_alt = np.asarray(a_alt, dtype=np.float64)
    diff = a_pf - a_alt  # 正なら alt の方が MSE 小(改善)
    out = {
        "mean_pf": float(a_pf.mean()),
        "mean_alt": float(a_alt.mean()),
        "mean_improve_pct": float(100.0 * diff.mean() / a_pf.mean()),
        "n": int(len(diff)),
    }
    if len(diff) >= 2 and np.any(diff != 0):
        t_stat, t_p = stats.ttest_rel(a_pf, a_alt)
        out["paired_t_stat"] = float(t_stat)
        out["paired_t_p"] = float(t_p)
        try:
            w_stat, w_p = stats.wilcoxon(a_pf, a_alt)
            out["wilcoxon_stat"] = float(w_stat)
            out["wilcoxon_p"] = float(w_p)
        except ValueError:
            out["wilcoxon_stat"] = float("nan")
            out["wilcoxon_p"] = float("nan")
    else:
        out["paired_t_stat"] = out["paired_t_p"] = float("nan")
        out["wilcoxon_stat"] = out["wilcoxon_p"] = float("nan")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched_by_n = build_matched_hp()

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Matched-HP experiment — Regression (R1-7)")
    emit("  共通ハイパラ = PF 最良構成を全メソッドに適用")
    emit(f"  seeds={len(SEEDS)}, T={T}, EVAL_START={EVAL_START}")
    emit("=" * 72)
    for n_p in N_PARTICLES_LIST:
        matched, beta, src = matched_by_n[n_p]
        emit(f"  N={n_p}: (η,σcd,σ0)={matched} [{src}], WSPF-A β={beta}")

    # ---- 並列実行 ----
    task_list = [(n_p, seed) for n_p in N_PARTICLES_LIST for seed in SEEDS]
    n_workers = min(os.cpu_count() or 1, 48)
    emit(f"\n  workers={n_workers}, jobs={len(task_list)}")

    # per-seed eval-region MSE と時系列(regime-split用)
    seed_mse = {n_p: {m: [] for m in METHODS} for n_p in N_PARTICLES_LIST}
    mse_ts = {n_p: {m: [] for m in METHODS} for n_p in N_PARTICLES_LIST}
    switch_times_by_n = {}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {}
        for n_p, seed in task_list:
            matched, beta, _ = matched_by_n[n_p]
            best_pf = matched
            best_wspf_b = dict(matched)
            best_wspf_a = {**matched, "beta": beta}
            fut = ex.submit(run_single, seed, n_p,
                            best_pf, best_wspf_b, best_wspf_a)
            futs[fut] = (n_p, seed)
        done = 0
        for fut in as_completed(futs):
            n_p, seed = futs[fut]
            res = fut.result()
            for m in METHODS:
                seed_mse[n_p][m].append(float(res["mse"][m][EVAL_START:].mean()))
                mse_ts[n_p][m].append(res["mse"][m])
            switch_times_by_n.setdefault(n_p, res["switch_times"])
            done += 1
            if done % 10 == 0 or done == len(task_list):
                emit(f"  [{done}/{len(task_list)}] done")
    emit(f"  elapsed {time.time()-t0:.0f}s")

    # ---- 集計・検定 ----
    csv_rows = [("n_particles", "method", "mse_mean", "mse_std",
                 "improve_vs_pf_pct", "paired_t_p", "wilcoxon_p")]
    for n_p in N_PARTICLES_LIST:
        emit(f"\n{'='*72}\n  N = {n_p} — Matched-HP MSE (eval region)\n{'='*72}")
        emit(f"  {'Method':<8s} {'MSE mean':>10s} {'MSE std':>9s} "
             f"{'vs PF %':>9s} {'paired-t p':>11s} {'Wilcoxon p':>11s}")
        emit(f"  {'-'*62}")
        pf_vals = seed_mse[n_p]["PF"]
        for m in METHODS:
            vals = np.asarray(seed_mse[n_p][m])
            if m in ("WSPF-A", "WSPF-B"):
                ps = paired_stats(pf_vals, vals)
                imp = ps["mean_improve_pct"]
                tp, wp = ps["paired_t_p"], ps["wilcoxon_p"]
                emit(f"  {m:<8s} {vals.mean():>10.4f} {vals.std():>9.4f} "
                     f"{imp:>8.2f}% {tp:>11.4g} {wp:>11.4g}")
                csv_rows.append((n_p, m, f"{vals.mean():.6f}",
                                 f"{vals.std():.6f}", f"{imp:.4f}",
                                 f"{tp:.6g}", f"{wp:.6g}"))
            else:
                mark = "  (baseline)" if m == "PF" else ""
                emit(f"  {m:<8s} {vals.mean():>10.4f} {vals.std():>9.4f} "
                     f"{'-':>9s} {'-':>11s} {'-':>11s}{mark}")
                csv_rows.append((n_p, m, f"{vals.mean():.6f}",
                                 f"{vals.std():.6f}", "", "", ""))

        # regime-split (post-switch vs stable)
        split = compute_regime_split_metrics(
            mse_ts[n_p], switch_times_by_n[n_p], T, n_post_switch=N_POST_SWITCH)
        emit(f"\n  Regime-split MSE (post-switch first {N_POST_SWITCH} / stable):")
        emit(f"  {'Method':<8s} {'Post-switch':>20s} {'Stable':>20s}")
        for m in METHODS:
            r = split[m]
            emit(f"  {m:<8s} "
                 f"{r['post_switch_mean']:8.4f}+/-{r['post_switch_std']:.4f}   "
                 f"{r['stable_mean']:8.4f}+/-{r['stable_std']:.4f}")

    # ---- 保存 ----
    txt_path = os.path.join(OUTPUT_DIR, "matched_hp_regression.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "matched_hp_regression.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
