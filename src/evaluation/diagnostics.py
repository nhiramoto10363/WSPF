#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Particle and weight diagnostics.

The measurement points are fixed:
  - after normalization, before resampling: ess, entropy, max_weight,
    spread_trace, rho
  - after resampling: resampled, unique_particles (unique ancestors)

On steps without resampling unique_particles is always N, so both the
all-step average and the average over resampled steps are reported.
"""

from __future__ import annotations

import numpy as np


def _get(history, key):
    return np.asarray(history[key]) if key in history else None


def summarize_history(history):
    """Summarize the degeneracy diagnostics of one run.

    Parameters
    ----------
    history : dict[str, ndarray]
        Output of filter.get_history().

    Returns
    -------
    dict
        {
          "pre_resample": {ess, entropy, max_weight, spread_trace, rho},
          "post_resample": {
             "resample_rate", "unique_all_mean", "unique_ancestor_rate_all",
             "unique_resample_mean", "unique_ancestor_rate_resample",
             "n_particles"
          },
          "n_steps": int
        }
    """
    ess = _get(history, "ess")
    n_steps = int(len(ess)) if ess is not None else 0

    # Quantities measured after normalization, before resampling.
    pre = {}
    for k in ("ess", "entropy", "max_weight", "spread_trace"):
        arr = _get(history, k)
        pre[k] = float(np.mean(arr)) if arr is not None and arr.size else float("nan")

    # rho is a (T, N) array for the WSPF filters and absent for PF and the
    # oracle; the scalar rho_mean series takes precedence when present.
    rho_mean_col = _get(history, "rho_mean")
    rho = _get(history, "rho")
    if rho_mean_col is not None and rho_mean_col.size:
        pre["rho"] = float(np.mean(rho_mean_col))
    elif rho is not None and rho.size:
        pre["rho"] = float(np.mean(rho))
    else:
        pre["rho"] = float("nan")

    resampled = _get(history, "resampled")
    unique = _get(history, "unique_particles")

    # N is recovered as the maximum of unique_particles, since steps without
    # resampling record exactly N.
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
    """Distribution of the signal-to-drift ratio rho.

    Returns
    -------
    dict or None
        None for filters without rho (PF and the oracle). Otherwise
        quantiles, mean, tail probabilities, and the guard counters.
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
    """Summarize the per-step timings.

    Drops the first `warmup` steps and reports mean, median and p95 in
    milliseconds for each timing key, plus the number of gradient
    evaluations per step.

    Returns
    -------
    dict
        {"t_step": {"mean_ms", "median_ms", "p95_ms"}, ...,
         "sample_grad_evals": {"mean", "total"}}
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
