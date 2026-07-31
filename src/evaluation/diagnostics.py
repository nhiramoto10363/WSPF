#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子/重み診断(diagnostics) — R1-9 / R1-8 / R1-11

計測タイミングを固定して縮退診断を集約する:
  - 正規化後・リサンプリング前:  ess, entropy, max_weight, spread_trace, rho
  - リサンプリング後:            resampled, unique_particles(祖先ユニーク数)

リサンプリングしなかったステップでは unique_particles は常に N になるため、
「全ステップ平均」と「リサンプリングしたステップのみの平均」の両方を報告する
(R1-9 の指摘: 診断量の測定点と平均の取り方を明示する)。

unique 数はフィルタが保存済みの祖先インデックス由来の unique_particles を使う。
"""

from __future__ import annotations

import numpy as np


def _get(history, key):
    return np.asarray(history[key]) if key in history else None


def summarize_history(history):
    """
    フィルタ履歴から縮退診断を要約する(測定タイミングを固定)。

    Parameters
    ----------
    history : dict[str, ndarray]
        filter.get_history() の戻り値。

    Returns
    -------
    dict
        {
          "pre_resample": {ess, entropy, max_weight, spread_trace, rho}  # 全ステップ平均
          "post_resample": {
             "resample_rate", "unique_all_mean", "unique_ancestor_rate_all",
             "unique_resample_mean", "unique_ancestor_rate_resample", "n_particles"
          },
          "n_steps": int
        }
    """
    ess = _get(history, "ess")
    n_steps = int(len(ess)) if ess is not None else 0

    # --- 正規化後・リサンプリング前の量(全ステップ平均) ---
    pre = {}
    for k in ("ess", "entropy", "max_weight", "spread_trace"):
        arr = _get(history, k)
        pre[k] = float(np.mean(arr)) if arr is not None and arr.size else float("nan")

    # rho は (T, N) 配列(WSPF)か、無し(PF/Oracle)。rho_mean 列があればそれを優先。
    rho_mean_col = _get(history, "rho_mean")
    rho = _get(history, "rho")
    if rho_mean_col is not None and rho_mean_col.size:
        pre["rho"] = float(np.mean(rho_mean_col))
    elif rho is not None and rho.size:
        pre["rho"] = float(np.mean(rho))
    else:
        pre["rho"] = float("nan")

    # --- リサンプリング後の量 ---
    resampled = _get(history, "resampled")
    unique = _get(history, "unique_particles")

    # 粒子数 N の推定: unique の最頻値(リサンプリング無しステップは常に N)。
    n_particles = None
    if unique is not None and unique.size:
        n_particles = int(np.max(unique))

    post = {"n_particles": n_particles}
    if resampled is not None and resampled.size:
        resampled_bool = resampled.astype(bool)
        post["resample_rate"] = float(np.mean(resampled_bool))
    else:
        resampled_bool = None
        post["resample_rate"] = float("nan")

    if unique is not None and unique.size:
        unique = unique.astype(np.float64)
        post["unique_all_mean"] = float(np.mean(unique))
        if n_particles:
            post["unique_ancestor_rate_all"] = float(np.mean(unique) / n_particles)
        else:
            post["unique_ancestor_rate_all"] = float("nan")
        # リサンプリングしたステップのみ
        if resampled_bool is not None and resampled_bool.any():
            u_rs = unique[resampled_bool]
            post["unique_resample_mean"] = float(np.mean(u_rs))
            post["unique_ancestor_rate_resample"] = (
                float(np.mean(u_rs) / n_particles) if n_particles else float("nan")
            )
        else:
            post["unique_resample_mean"] = float("nan")
            post["unique_ancestor_rate_resample"] = float("nan")
    else:
        post["unique_all_mean"] = float("nan")
        post["unique_ancestor_rate_all"] = float("nan")
        post["unique_resample_mean"] = float("nan")
        post["unique_ancestor_rate_resample"] = float("nan")

    return {"pre_resample": pre, "post_resample": post, "n_steps": n_steps}


def rho_report(history):
    """
    R1-8: signal-to-drift 比 ρ の分布報告。

    Returns
    -------
    dict or None
        ρ を持たないフィルタ(PF/Oracle)では None。
        {
          "q50","q90","q99","max","mean",
          "p_gt_0.9","p_gt_0.99",
          "rho_clip_count","logcorr_nonfinite_count"
        }
    """
    rho = _get(history, "rho")
    if rho is None or rho.size == 0:
        return None
    flat = np.asarray(rho, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    report = {
        "q50": float(np.quantile(finite, 0.50)),
        "q90": float(np.quantile(finite, 0.90)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "p_gt_0.9": float(np.mean(finite > 0.9)),
        "p_gt_0.99": float(np.mean(finite > 0.99)),
    }
    clip = _get(history, "rho_clip_count")
    report["rho_clip_count"] = int(np.sum(clip)) if clip is not None else 0
    nf = _get(history, "logcorr_nonfinite_count")
    report["logcorr_nonfinite_count"] = int(np.sum(nf)) if nf is not None else 0
    return report


def _pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def timing_report(history, warmup=10):
    """
    R1-11: 計算コスト集計。旧 compute_cost_benchmark._summarize_timing を踏襲。

    warmup ステップを除外し、各計時キーごとに mean/median/p95 [ms] を返す。
    さらに sample_grad_evals(1 ステップあたりの勾配評価数)を報告する。

    Returns
    -------
    dict
        {
          "t_step": {"mean_ms","median_ms","p95_ms"}, ...,
          "sample_grad_evals": {"mean","total"}
        }
    """
    keys = ["t_step", "t_grad", "t_correction", "t_loglik",
            "t_weight", "t_resample"]
    out = {}
    for k in keys:
        arr = _get(history, k)
        if arr is None or arr.size == 0:
            out[k] = {"mean_ms": float("nan"), "median_ms": float("nan"),
                      "p95_ms": float("nan")}
            continue
        arr = np.asarray(arr[warmup:], dtype=np.float64)
        if arr.size == 0:
            arr = np.asarray(history[k], dtype=np.float64)
        out[k] = {
            "mean_ms": 1e3 * float(arr.mean()),
            "median_ms": 1e3 * float(np.median(arr)),
            "p95_ms": 1e3 * _pct(arr, 95),
        }
    sge = _get(history, "sample_grad_evals")
    if sge is not None and sge.size:
        sge_w = np.asarray(sge[warmup:], dtype=np.float64)
        if sge_w.size == 0:
            sge_w = np.asarray(sge, dtype=np.float64)
        out["sample_grad_evals"] = {
            "mean": float(sge_w.mean()),
            "total": int(np.sum(sge)),
        }
    else:
        out["sample_grad_evals"] = {"mean": float("nan"), "total": 0}
    return out
