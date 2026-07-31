#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計比較(statistics) — シード横断の対応検定・回復曲線

修正方針: gefcom_multiseed / calibration_report に散らばっていた
対応 t 検定・Wilcoxon 符号順位検定・post-switch 回復曲線を集約する。

scipy は関数内で遅延 import する(scipy が無い環境でもモジュール自体は
import 可能で、検定を呼ばない限り落ちない)。ゼロ分散・NaN はガードする。
"""

from __future__ import annotations

import numpy as np


def mean_std(x):
    """(平均, 母標準偏差) を返す。NaN は無視。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x))


def _clean_pair(a, b):
    """対応する有限値のペアのみ抽出。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def paired_t(a, b):
    """
    対応 t 検定(scipy.stats.ttest_rel)。

    Returns
    -------
    dict {"t", "p", "n", "mean_diff"}
        分散ゼロ・データ不足時は t=p=nan(mean_diff は算出可能なら返す)。
    """
    a, b = _clean_pair(a, b)
    n = a.size
    mean_diff = float(np.mean(a - b)) if n > 0 else float("nan")
    if n < 2:
        return {"t": float("nan"), "p": float("nan"), "n": n,
                "mean_diff": mean_diff}
    diff = a - b
    if np.allclose(np.std(diff), 0.0):
        # 差が定数: 全て同じなら差 0 → 有意でない、非零定数なら完全分離。
        p = 1.0 if np.allclose(diff, 0.0) else 0.0
        t = 0.0 if np.allclose(diff, 0.0) else np.inf * np.sign(mean_diff)
        return {"t": float(t), "p": float(p), "n": n, "mean_diff": mean_diff}
    from scipy import stats  # 遅延 import
    res = stats.ttest_rel(a, b)
    return {"t": float(res.statistic), "p": float(res.pvalue), "n": n,
            "mean_diff": mean_diff}


def wilcoxon_signed(a, b):
    """
    Wilcoxon 符号順位検定(scipy.stats.wilcoxon)。

    Returns
    -------
    dict {"stat", "p", "n", "mean_diff"}
        全差 0・データ不足時は stat=p=nan。
    """
    a, b = _clean_pair(a, b)
    n = a.size
    mean_diff = float(np.mean(a - b)) if n > 0 else float("nan")
    if n < 1:
        return {"stat": float("nan"), "p": float("nan"), "n": n,
                "mean_diff": mean_diff}
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"stat": float("nan"), "p": float("nan"), "n": n,
                "mean_diff": mean_diff}
    from scipy import stats  # 遅延 import
    try:
        res = stats.wilcoxon(a, b)
        return {"stat": float(res.statistic), "p": float(res.pvalue), "n": n,
                "mean_diff": mean_diff}
    except ValueError:
        return {"stat": float("nan"), "p": float("nan"), "n": n,
                "mean_diff": mean_diff}


def paired_compare(per_seed_a, per_seed_b):
    """
    シードごとのスカラー系列 a, b の対応比較。平均差と t/p を返す。

    Returns
    -------
    dict {"mean_a","mean_b","mean_diff","t","p","n"}
    """
    a, b = _clean_pair(per_seed_a, per_seed_b)
    tt = paired_t(a, b)
    return {
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
        "mean_diff": tt["mean_diff"],
        "t": tt["t"],
        "p": tt["p"],
        "n": tt["n"],
    }


def recovery_curve(mse_ts, switch_points, max_lag):
    """
    post-switch 回復曲線(R1-12 / calibration_report と同じ集計順序)。

    各シードについて、各 lag(0..max_lag-1)で全スイッチ点の
    mse_ts[sp + lag] を平均(スイッチ横断)し、その後シード横断で平均する。

    Parameters
    ----------
    mse_ts : list[ndarray] | ndarray
        シードごとの MSE 時系列(各 shape (T,))。単一 (T,) 配列も可。
    switch_points : list[int]
    max_lag : int

    Returns
    -------
    dict {
      "curve": ndarray (max_lag,)     # シード平均
      "std":   ndarray (max_lag,)     # シード間標準偏差
      "per_seed": ndarray (n_seed, max_lag)
    }
    """
    if isinstance(mse_ts, np.ndarray) and mse_ts.ndim == 1:
        mse_ts = [mse_ts]
    n_seed = len(mse_ts)
    per_seed = np.full((n_seed, max_lag), np.nan)
    for si, ts in enumerate(mse_ts):
        ts = np.asarray(ts, dtype=np.float64).ravel()
        T = ts.size
        for lag in range(max_lag):
            vals = [ts[sp + lag] for sp in switch_points if 0 <= sp + lag < T]
            if vals:
                per_seed[si, lag] = float(np.mean(vals))
    curve = np.nanmean(per_seed, axis=0) if n_seed else np.full(max_lag, np.nan)
    std = np.nanstd(per_seed, axis=0) if n_seed else np.full(max_lag, np.nan)
    return {"curve": curve, "std": std, "per_seed": per_seed}
