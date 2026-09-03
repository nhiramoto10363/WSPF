#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-seed statistics: paired tests and post-switch recovery curves.

scipy is imported lazily inside the functions, so the module imports even
where scipy is unavailable and only fails if a test is actually called.
Zero variance and NaNs are guarded throughout.
"""

from __future__ import annotations

import numpy as np


def mean_std(x):
    """Return (mean, population std), ignoring NaNs."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x))


def _clean_pair(a, b):
    """Keep only the pairs where both entries are finite."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def paired_t(a, b):
    """Paired t-test (scipy.stats.ttest_rel).

    Returns
    -------
    dict {"t", "p", "n", "mean_diff"}
        t and p are NaN when the variance is zero or there is too little
        data; mean_diff is still returned when it can be computed.
    """
    a, b = _clean_pair(a, b)
    n = a.size
    mean_diff = float(np.mean(a - b)) if n > 0 else float("nan")
    if n < 2:
        return {"t": float("nan"), "p": float("nan"), "n": n,
                "mean_diff": mean_diff}
    diff = a - b
    if np.allclose(np.std(diff), 0.0):
        # Constant difference: zero means no effect, non-zero means the two
        # samples are perfectly separated.
        p = 1.0 if np.allclose(diff, 0.0) else 0.0
        t = 0.0 if np.allclose(diff, 0.0) else np.inf * np.sign(mean_diff)
        return {"t": float(t), "p": float(p), "n": n, "mean_diff": mean_diff}
    from scipy import stats  # lazy import
    res = stats.ttest_rel(a, b)
    return {"t": float(res.statistic), "p": float(res.pvalue), "n": n,
            "mean_diff": mean_diff}


def _wilcoxon_method(diff):
    """Choose between the exact null distribution and the normal approximation.

    Zero differences or ties move the null distribution of the signed-rank
    statistic away from the exact one, so those cases use the normal
    approximation and the rest use the exact distribution.

    scipy has changed the criterion behind its own default (auto) between
    versions, which makes p-values depend on the scipy version for the same
    input (visible whenever a difference is exactly zero). Fixing the rule
    here keeps the results reproducible; it matches the auto behaviour of
    scipy 1.7.3, under which the published results were produced.
    """
    nz = diff[diff != 0]
    has_ties = np.unique(np.abs(nz)).size != nz.size
    return "approx" if ((diff == 0).any() or has_ties) else "exact"


def wilcoxon_signed(a, b):
    """Wilcoxon signed-rank test (scipy.stats.wilcoxon).

    The exact/approximate choice is fixed by _wilcoxon_method rather than
    left to the scipy default, so p-values do not depend on the scipy
    version.

    Returns
    -------
    dict {"stat", "p", "n", "mean_diff"}
        stat and p are NaN when all differences are zero or data is missing.
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
    from scipy import stats  # lazy import
    method = _wilcoxon_method(diff)
    try:
        try:
            res = stats.wilcoxon(a, b, method=method)   # scipy >= 1.12
        except TypeError:
            res = stats.wilcoxon(a, b, mode=method)     # scipy < 1.12
        return {"stat": float(res.statistic), "p": float(res.pvalue), "n": n,
                "mean_diff": mean_diff}
    except ValueError:
        return {"stat": float("nan"), "p": float("nan"), "n": n,
                "mean_diff": mean_diff}


def paired_compare(per_seed_a, per_seed_b):
    """Paired comparison of two per-seed scalar series.

    Every comparison in the paper reports both a paired t-test and a Wilcoxon
    signed-rank test, so both are computed here. The keys "t" and "p" refer
    to the t-test.

    Returns
    -------
    dict {"mean_a","mean_b","mean_diff","std_diff","t","p","n",
          "wilcoxon_stat","wilcoxon_p"}
    """
    a, b = _clean_pair(per_seed_a, per_seed_b)
    tt = paired_t(a, b)
    wl = wilcoxon_signed(a, b)
    diff = a - b
    return {
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
        "mean_diff": tt["mean_diff"],
        "std_diff": float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0,
        "t": tt["t"],
        "p": tt["p"],
        "n": tt["n"],
        "wilcoxon_stat": wl["stat"],
        "wilcoxon_p": wl["p"],
    }


def holm_adjust(pvalues):
    """Holm-Bonferroni adjustment of a family of p-values.

    The main tables test every pair of methods, so the family is large. Holm
    controls the family-wise error rate with more power than Bonferroni. NaN
    p-values are excluded from the family and returned as NaN, so an
    untestable comparison is pushed neither towards nor away from
    significance.

    Parameters
    ----------
    pvalues : array-like of float

    Returns
    -------
    ndarray, shape (len(pvalues),)
        Adjusted p-values, made monotone by a running maximum.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    out = np.full(p.size, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return out
    m = valid.size
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running = max(running, adj)          # enforce monotonicity
        out[idx] = min(1.0, running)
    return out


def all_pairs_compare(per_seed_by_method, methods=None, lower_is_better=True):
    """Paired t-test and Wilcoxon for *every pair* of methods.

    Parameters
    ----------
    per_seed_by_method : dict[str, list[float]]
        Method name -> per-seed scalar metric, in the same seed order for
        every method.
    methods : list[str] | None
        Which methods to compare, and in what order; defaults to the
        insertion order of per_seed_by_method.
    lower_is_better : bool
        With True (MSE, NLL) a negative mean_diff favours a; with False (F1,
        accuracy) the sign flips. Only affects the "better" column.

    Returns
    -------
    list[dict]
        {"a","b","mean_diff","std_diff","paired_t_p","wilcoxon_p",
         "paired_t_p_holm","wilcoxon_p_holm","n_seeds","better"}, with the
        p-values Holm-adjusted over all pairs as one family.
    """
    ms = list(methods) if methods is not None else list(per_seed_by_method)
    ms = [m for m in ms if m in per_seed_by_method]

    rows = []
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            c = paired_compare(per_seed_by_method[a], per_seed_by_method[b])
            md = c["mean_diff"]
            if not np.isfinite(md) or md == 0.0:
                better = "tie"
            elif (md < 0.0) == bool(lower_is_better):
                better = a
            else:
                better = b
            rows.append({
                "a": a, "b": b,
                "mean_diff": md, "std_diff": c["std_diff"],
                "paired_t_p": c["p"], "wilcoxon_p": c["wilcoxon_p"],
                "n_seeds": c["n"], "better": better,
            })

    for key in ("paired_t_p", "wilcoxon_p"):
        adj = holm_adjust([r[key] for r in rows])
        for r, v in zip(rows, adj):
            r[f"{key}_holm"] = float(v) if np.isfinite(v) else float("nan")
    return rows


def recovery_curve(mse_ts, switch_points, max_lag):
    """Post-switch recovery curve.

    Within each seed, the value at lag tau is averaged across switch points
    (mse_ts[sp + tau] for every sp); the resulting per-seed curves are then
    averaged across seeds. Switch events are therefore never treated as
    independent samples.

    Parameters
    ----------
    mse_ts : list[ndarray] | ndarray
        One MSE series of shape (T,) per seed; a single (T,) array is fine.
    switch_points : list[int]
    max_lag : int

    Returns
    -------
    dict {
      "curve": ndarray (max_lag,)      mean across seeds
      "std":   ndarray (max_lag,)      std across seeds
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
