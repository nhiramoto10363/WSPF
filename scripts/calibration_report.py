#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration and post-switch recovery.

Regression:
  - coverage against the nominal level (0.5 / 0.8 / 0.9 / 0.95): the
    reporting-window coverage_{lvl} and width_{lvl}, averaged over seeds;
  - recovery curves, since the switch points are known: the reporting-window
    MSE series of each seed goes through recovery_curve and is aggregated by
    post-switch lag, and the recovery area is compared against PF. Switch
    events are never treated as independent samples: recovery_curve averages
    across switches within a seed first, and the paired comparison then runs
    over the ten seeds.

Classification (email, insects):
  - Brier and ECE are computed *per seed* from the per-sample probabilities
    and labels of the reporting window, then reported as mean +- SD across
    seeds; the per-seed values are also compared against PF. Only the
    reliability diagram pools all seeds, to stabilize the bins.
  - post-switch recovery curves (accuracy and F1) as for regression. The
    drift points [300, 600, 900, 1200] do not align with the block boundaries
    at B=16, so straddling blocks are dropped and lag 0 is defined as the
    first complete post-switch block after each drift point.

p-values never share a column with a standard deviation: mean_difference,
std_difference, paired_t_p and wilcoxon_p are separate. The recovery claim is
made per lag, so a paired test runs at each lag tau = 0..K-1 and the K
p-values are Holm-corrected (the recovery_lag_paired rows).

Output: outputs/<benchmark>/calibration/calibration_report.{csv,txt,tex},
in a tidy superset schema (task, method, metric, level, mean, std,
mean_difference, std_difference, paired_t_p, paired_t_p_holm, wilcoxon_p,
zone), where a given row may leave some columns empty. PNGs
(reliability_*.png, recovery_regression.png) are written when matplotlib is
available; the CSV is always written.

Usage:
    python scripts/calibration_report.py --benchmark regression
    python scripts/calibration_report.py --benchmark email
    python scripts/calibration_report.py --benchmark insects
"""

import argparse
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask)
from src.evaluation import (run_seeds, mean_std, recovery_curve,
                            paired_compare, wilcoxon_signed, brier_ece,
                            brier_ece_multiclass, write_table)

# ----------------------------------------------------------------------
# matplotlib is optional: configure it before importing, and carry on if it
# is unavailable.
# ----------------------------------------------------------------------
# Cache directory for matplotlib. Default to .mplcache inside the repository
# so that the script also works where the home directory is not writable; an
# existing MPLCONFIGDIR wins.
_MPL_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".mplcache")
os.makedirs(_MPL_CACHE, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE)
_HAVE_MPL = False
try:  # pragma: no cover - plotting is environment-dependent
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False

LEVELS = (0.5, 0.8, 0.9, 0.95)
N_POST_SWITCH = 10   # number of lags in the recovery curve

# Superset schema of the tidy table; a row may leave some columns empty.
# Going through _row() keeps the CSV column order fixed.
_SCHEMA = ("task", "method", "metric", "level", "mean", "std",
           "mean_difference", "std_difference", "paired_t_p",
           "paired_t_p_holm", "wilcoxon_p", "zone")


def _row(task, method, metric, level="", mean="", std="",
         mean_difference="", std_difference="", paired_t_p="",
         paired_t_p_holm="", wilcoxon_p="", zone=""):
    """Build one row of the fixed schema; missing columns are empty."""
    return {"task": task, "method": method, "metric": metric,
            "level": level, "mean": mean, "std": std,
            "mean_difference": mean_difference,
            "std_difference": std_difference,
            "paired_t_p": paired_t_p, "paired_t_p_holm": paired_t_p_holm,
            "wilcoxon_p": wilcoxon_p, "zone": zone}


def _paired_diffs(a, b):
    """Differences a - b over the finite pairs, truncated to the shorter."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask] - b[mask]


def _holm_correction(pvals):
    """
    Holm-Bonferroni correction over the finite p-values.

    Uses statsmodels when available and falls back to an equivalent
    implementation: sort ascending, multiply by (m - rank), enforce
    monotonicity and clip at one, where m counts the finite tests. NaN
    p-values stay NaN.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    adj = np.full(p.size, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return adj
    fp = p[finite]
    try:  # use statsmodels when available
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(fp, method="holm")
        adj[finite] = p_adj
        return adj
    except Exception:  # noqa: BLE001 - fall back to the local implementation
        m = fp.size
        order = np.argsort(fp)
        running = 0.0
        for rank, k in enumerate(order):
            running = max(running, (m - rank) * float(fp[k]))
            adj[finite[k]] = min(running, 1.0)
        return adj


# ======================================================================
# Helpers
# ======================================================================
def _methods(cfg):
    """Methods to calibrate: the filters and the baselines, minus NoChange."""
    return [m for m in cfg["methods"] if m != "NoChange"]


def _run_methods(cfg, selected, methods, n_main, eval_seeds, **ctx):
    """Run every method on the evaluation seeds.

    ctx holds benchmark-construction overrides. Returns
    (results_by_method, bench_ref).
    """
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
    """Mean of one metric over the reporting window of a single run."""
    mask = region_mask(result, "report")
    arr = np.asarray(result["metrics"].get(key, []), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    v = arr[mask]
    return float(np.nanmean(v)) if v.size else float("nan")


# ======================================================================
# Regression: coverage against the nominal level
# ======================================================================
def coverage_rows(cfg, results_by_method, task="regression", zone=""):
    """Coverage, interval width and CRPS by level, averaged over seeds.

    task and zone are written into each row.
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
# Regression: recovery curves and the paired comparison against PF
# ======================================================================
def recovery_analysis(cfg, results_by_method, switch_points):
    """
    Compute the post-switch recovery curves.

    Returns
    -------
    rows : list[dict]
    curves : dict[method] -> (curve, std), for plotting
    """
    rows = []
    curves = {}
    if not switch_points:
        return rows, curves

    # Keep only switches that lie wholly inside the reporting window.
    ref = next(iter(results_by_method.values()))[0]
    rep_mask = region_mask(ref, "report")
    T = rep_mask.size
    valid_sw = [sp for sp in switch_points
                if 0 <= sp < T and rep_mask[sp]]
    if not valid_sw:
        return rows, curves

    # Keep the per-seed recovery area (the mean over lags) and the
    # per-seed-by-lag matrix. The per_seed array of recovery_curve is
    # (n_seed, K), already averaged across switches within each seed.
    area_by_method = {}
    lagmat_by_method = {}
    for m, results in results_by_method.items():
        mse_ts = []
        for r in results:
            mask = region_mask(r, "report")
            mse = np.asarray(r["metrics"]["mse"], dtype=np.float64).copy()
            mse[~mask] = np.nan            # reporting window only
            mse_ts.append(mse)
        rec = recovery_curve(mse_ts, valid_sw, max_lag=N_POST_SWITCH)
        curves[m] = (rec["curve"], rec["std"])
        lagmat_by_method[m] = rec["per_seed"]           # (n_seed, K)
        # Recovery area per seed: the mean over lags.
        area = np.nanmean(rec["per_seed"], axis=1)
        area_by_method[m] = area
        amu, asd = mean_std(area)
        rows.append(_row("regression", m, "recovery_area", mean=amu, std=asd))

    # Paired comparison against PF across seeds, with the p-values in their
    # own columns.
    if "PF" in area_by_method:
        pf_area = area_by_method["PF"]
        pf_lag = lagmat_by_method["PF"]
        for m in area_by_method:
            if m == "PF":
                continue
            area = area_by_method[m]
            # (a) Difference in recovery area: the t-test and the Wilcoxon
            #     test go in separate columns, and std is the SD of the
            #     difference.
            cmp = paired_compare(area, pf_area)
            wil = wilcoxon_signed(area, pf_area)
            diffs = _paired_diffs(area, pf_area)
            sd_diff = float(np.std(diffs)) if diffs.size else float("nan")
            rows.append(_row("regression", m, "recovery_area_diff_vs_PF",
                             mean_difference=cmp["mean_diff"],
                             std_difference=sd_diff,
                             paired_t_p=cmp["p"], wilcoxon_p=wil["p"]))

            # (b) A paired test at each lag over the n_seed values, with a
            #     Holm correction across lags.
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
# Classification: Brier, ECE and the reliability diagram
# ======================================================================
def classification_analysis(cfg, results_by_method, n_classes=None):
    """
    Brier and ECE are computed per seed from the per-sample probabilities
    and labels of the reporting window and reported as mean +- SD across
    seeds, never pooled; the per-seed values are also compared against PF.

    The reliability diagram is the one exception: rel_x and rel_y pool every
    seed, to put enough samples in each bin for a stable figure.

    Both the binary (email) and the multiclass (INSECTS) cases are handled.
    With n_classes None or at most two the binary path (brier_ece) is used,
    and with three or more the multiclass one (brier_ece_multiclass, with
    top-label calibration).

    The branch matters: sending multiclass probabilities down the binary path
    would ravel a (B, C) array and pair only its first B entries with the
    labels, returning meaningless values (a Brier score outside its range)
    without raising.
    """
    C = None if n_classes is None else int(n_classes)
    multiclass = C is not None and C > 2

    rows = []
    reliability = {}
    per_seed_brier = {}   # method -> ndarray(per-seed brier)
    per_seed_ece = {}     # method -> ndarray(per-seed ece)

    for m, results in results_by_method.items():
        briers, eces = [], []
        pooled_probs, pooled_labels = [], []
        for r in results:
            raw_p = np.asarray(r["predictions"].get("probs", []), np.float64)
            y = np.asarray(r["predictions"].get("y", []), np.float64).ravel()
            if multiclass:
                # Keep the (B, C) shape; do not ravel.
                if raw_p.ndim != 2 or raw_p.shape[1] != C:
                    raise ValueError(
                        f"{m}: multiclass probs must have shape (B, {C}), "
                        f"got {raw_p.shape}")
                n = min(raw_p.shape[0], y.size)
                if n == 0:
                    continue
                p, y = raw_p[:n], y[:n]
                b, e, _, _ = brier_ece_multiclass(p, y.astype(np.int64), C,
                                                  n_bins=10)   # ← per seed
            else:
                p = raw_p.ravel()
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

        # One reliability curve, pooled over seeds - the only pooling here.
        if pooled_probs:
            allp = np.concatenate(pooled_probs)   # multiclass: (sum B, C)
            ally = np.concatenate(pooled_labels)
            if multiclass:
                _, _, rel_x, rel_y = brier_ece_multiclass(
                    allp, ally.astype(np.int64), C, n_bins=10)
            else:
                _, _, rel_x, rel_y = brier_ece(allp, ally, n_bins=10)
            reliability[m] = (rel_x, rel_y)

    # Per-seed paired comparison against PF, with the p-values in their own
    # columns.
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
# Classification: post-switch recovery curves (accuracy and F1)
# ======================================================================
def _email_recovery_curve(result, switch_points, metric_key, max_lag):
    """Post-switch recovery curve for a single run.

    The email drift points do not align with the block boundaries at B=16, so:
      - straddling blocks are dropped;
      - lag 0 is the first complete post-switch block after a drift point,
        i.e. the first non-straddling block whose start (min test_indices) is
        at least d;
      - lag tau is tau blocks later, and anything outside the reporting
        window or straddling is NaN.
    A drift point whose lag 0 falls outside the reporting window is dropped
    entirely (email's d=300 lies in the selection window). Switches are
    averaged within the seed here; aggregation across seeds is the caller's
    job.

    Returns
    -------
    ndarray (max_lag,)   lag curve, averaged across switches for this seed
    """
    arr = np.asarray(result["metrics"].get(metric_key, []), dtype=np.float64)
    T = arr.size
    if T == 0:
        return np.full(max_lag, np.nan)
    report = np.asarray(result.get("report_mask", np.zeros(T, bool)))
    straddle = np.asarray(result.get("straddle_mask", np.zeros(T, bool)))
    test_idx = result.get("test_indices", [])

    def _block_start(step):
        """Data index at the start of a block (step * B)."""
        if step < len(test_idx):
            idx = np.asarray(test_idx[step], dtype=int)
            if idx.size:
                return int(idx.min())
        return step  # fallback, normally unreachable

    per_switch = []
    for d in switch_points:
        # First complete block after drift point d: not straddling, and
        # starting at or after d.
        lag0 = None
        for s in range(T):
            if straddle[s]:
                continue
            if _block_start(s) >= d:
                lag0 = s
                break
        # Drop a drift point whose lag 0 lies outside the reporting window.
        if lag0 is None or not report[lag0]:
            continue
        lags = np.full(max_lag, np.nan)
        for tau in range(max_lag):
            t = lag0 + tau
            if t >= T or not report[t] or straddle[t]:
                continue
            v = arr[t]
            if np.isfinite(v):
                lags[tau] = v
        per_switch.append(lags)
    if not per_switch:
        return np.full(max_lag, np.nan)
    return np.nanmean(np.vstack(per_switch), axis=0)


def email_recovery_analysis(cfg, results_by_method, switch_points):
    """
    Post-switch recovery curves (accuracy and F1) for classification.

    The counterpart of recovery_analysis. _email_recovery_curve is evaluated
    for each method and seed (already averaged across switches within the
    seed), then aggregated across seeds as mean +- SD per lag, giving the
    recovery_acc and recovery_f1 rows. The per-seed recovery area, the mean
    over lags, is compared against PF in the
    recovery_acc_area_diff_vs_PF and recovery_f1_area_diff_vs_PF rows.

    Returns
    -------
    rows : list[dict]
    curves : dict[metric_key] -> dict[method] -> (curve, std), for plotting
    """
    rows = []
    curves = {"accuracy": {}, "f1": {}}
    if not switch_points:
        return rows, curves
    K = int(cfg.get("eval", {}).get("post_switch_lag", N_POST_SWITCH))

    specs = (("accuracy", "recovery_acc", "recovery_acc_area_diff_vs_PF"),
             ("f1", "recovery_f1", "recovery_f1_area_diff_vs_PF"))
    for metric_key, rec_metric, area_metric in specs:
        area_by_method = {}
        for m, results in results_by_method.items():
            if results:
                per_seed = np.vstack([
                    _email_recovery_curve(r, switch_points, metric_key, K)
                    for r in results])
            else:
                per_seed = np.full((0, K), np.nan)
            if per_seed.size:
                curve = np.nanmean(per_seed, axis=0)
                std = np.nanstd(per_seed, axis=0)
                area = np.nanmean(per_seed, axis=1)   # recovery area per seed
            else:
                curve = np.full(K, np.nan)
                std = np.full(K, np.nan)
                area = np.full(0, np.nan)
            curves[metric_key][m] = (curve, std)
            area_by_method[m] = area
            for tau in range(K):
                rows.append(_row("classification", m, rec_metric,
                                 level=f"lag_{tau}",
                                 mean=float(curve[tau]), std=float(std[tau])))

        # Paired comparison of the recovery area against PF, across seeds.
        if "PF" in area_by_method:
            pf_area = area_by_method["PF"]
            for m in area_by_method:
                if m == "PF":
                    continue
                area = area_by_method[m]
                cmp = paired_compare(area, pf_area)
                wil = wilcoxon_signed(area, pf_area)
                diffs = _paired_diffs(area, pf_area)
                sd_diff = float(np.std(diffs)) if diffs.size else float("nan")
                rows.append(_row("classification", m, area_metric,
                                 mean_difference=cmp["mean_diff"],
                                 std_difference=sd_diff,
                                 paired_t_p=cmp["p"], wilcoxon_p=wil["p"]))
    return rows, curves


# ======================================================================
# Plotting: optional throughout, and never fatal
# ======================================================================
def _plot_recovery(curves, out_dir):
    if not (_HAVE_MPL and curves):
        return
    try:  # pragma: no cover
        fig, ax = plt.subplots(figsize=(6, 4))
        lags = np.arange(N_POST_SWITCH)
        for m, (curve, std) in curves.items():
            ax.plot(lags, curve, marker="o", label=m)
            # The band is the spread across seeds (+-1 SD), not a confidence
            # interval.
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


def _plot_classification_recovery(curves, out_dir, bench_name):
    """Plot the classification recovery curves (optional).

    Written as recovery_<benchmark>_<metric>.png (email or insects).
    """
    if not (_HAVE_MPL and curves):
        return
    try:  # pragma: no cover - plotting is environment-dependent
        for metric_key, per_method in curves.items():
            if not per_method:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            n_lag = len(next(iter(per_method.values()))[0])
            lags = np.arange(n_lag)
            for m, (curve, std) in per_method.items():
                curve = np.asarray(curve, dtype=np.float64)
                std = np.asarray(std, dtype=np.float64)
                ax.plot(lags, curve, marker="o", label=m)
                # The band is the spread across seeds (+-1 SD), not a
                # confidence interval.
                ax.fill_between(lags, curve - std, curve + std, alpha=0.15)
            ax.set_xlabel("lag after switch (blocks)")
            ax.set_ylabel(f"report-region {metric_key}")
            ax.set_title(f"{bench_name} post-switch recovery: {metric_key} "
                         "(band = ±1 SD across seeds)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"recovery_{bench_name}_{metric_key}.png"),
                        dpi=120)
            plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / email / insects, or a config path")
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
    curves, reliability = {}, {}
    email_curves = {}

    if task_type == "regression":
        # Synthetic switches, single context.
        res, bench_ref = _run_methods(cfg, selected, methods, n_main,
                                      eval_seeds)
        rows += coverage_rows(cfg, res, task="regression")
        # Known switch points, so recovery curves plus per-lag paired tests.
        switch_points = list(getattr(bench_ref, "switch_points", []))
        rrows, curves = recovery_analysis(cfg, res, switch_points)
        rows += rrows

    else:
        # Classification: email is binary, INSECTS multiclass. n_classes
        # comes from the benchmark; email has no such attribute and is
        # treated as binary.
        res, bench_ref = _run_methods(cfg, selected, methods, n_main, eval_seeds)
        n_classes = getattr(bench_ref, "n_classes", None)
        crows, reliability = classification_analysis(cfg, res,
                                                     n_classes=n_classes)
        rows += crows
        # Known drift points, so recovery curves plus the area comparison.
        switch_points = list(getattr(bench_ref, "switch_points", []))
        erows, email_curves = email_recovery_analysis(cfg, res, switch_points)
        rows += erows

    # --- Always write the table ---
    base = os.path.join(out_dir, "calibration_report")
    write_table(rows, base, formats=("csv", "txt", "tex"))
    print(f"saved: {base}.{{csv,txt,tex}}  ({len(rows)} rows)")

    # --- Figures (optional) ---
    _plot_recovery(curves, out_dir)
    _plot_reliability(reliability, out_dir)
    _plot_classification_recovery(email_curves, out_dir, bench_name)
    if not _HAVE_MPL:
        print("[note] matplotlib unavailable: PNGs skipped, CSV written")

    # --- Summary ---
    def _num(v):
        return v if isinstance(v, float) and np.isfinite(v) else None

    print("\n=== calibration summary ===")
    for r in rows:
        lvl = f" level={r['level']}" if r.get("level") else ""
        zn = f" zone={r['zone']}" if r.get("zone") not in ("", None) else ""
        mu = _num(r.get("mean"))
        md = _num(r.get("mean_difference"))
        sd = _num(r.get("std"))
        pt = _num(r.get("paired_t_p"))
        pth = _num(r.get("paired_t_p_holm"))
        if mu is not None:                       # ordinary mean +- SD row
            val = f"{mu:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")
        elif md is not None:                     # difference row (vs PF)
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
