#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main results: N=main over every evaluation seed.

Each method is run with its selected hyper-parameters on the ten evaluation
seeds, and the metrics, diagnostics and artifacts are written to
outputs/<benchmark>/. The filters use the best hyper-parameters at N=main,
the point-estimate baselines their own selected values; the oracle is run
separately by run_oracle.py and only for regression.

Every (method, seed) run is saved in full - metrics.npz, predictions.npz,
diagnostics.npz, indices.npz, timings.json and meta.json. Predictions are
flattened over the reporting window, with pred_step_index and pred_offsets
mapping each sample back to the step it came from.

Usage:
    python scripts/run_main.py --benchmark regression
"""

import argparse
import json
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask)
from src.evaluation import (run_seeds, save_run_dir, mean_std, sanitize,
                            write_json, summarize_history, timing_report,
                            all_pairs_compare)

_REPO = os.path.dirname(os.path.dirname(__file__))

# Metrics that carry paired tests: the primary one plus two NLLs that probe
# the predictive distribution. nll scores a single Gaussian moment-matched to
# the particles (regression), while nll_mixture scores the mixture itself,
# which is the correct predictive distribution; for classification the two
# coincide.
_TEST_METRICS = ("nll", "nll_mixture")

# Diagnostics aggregated over the reporting window. The unique-ancestor rate
# is reported both over all steps and over resampled steps only.
_DIAG_KEYS = ["ess_over_N", "weight_entropy", "max_weight",
              "particle_spread", "unique_ancestor_rate_all",
              "unique_ancestor_rate_resampled", "resample_rate"]


def _report_diag(history, mask, n_particles):
    """Diagnostics averaged over the reporting window.

    Pass a mask that already excludes straddling blocks, i.e.
    region_mask(r, "report"). The measurement points are fixed: ess, entropy,
    max_weight and spread are read after weight normalization and before
    resampling; resampled and unique after it. The unique-ancestor rate is
    reported over all steps and over resampled steps separately.
    """
    m = np.asarray(mask, bool)

    def rmean(key, sel=None):
        a = np.asarray(history.get(key, []), dtype=float)
        if a.size != m.size:
            return float("nan")
        idx = m if sel is None else (m & sel)
        return float(np.nanmean(a[idx])) if idx.any() else float("nan")

    resampled = np.asarray(history.get("resampled", []), dtype=bool)
    resampled = resampled if resampled.size == m.size else np.zeros(m.size, bool)

    return {
        "ess_over_N": rmean("ess") / n_particles,
        "weight_entropy": rmean("entropy"),
        "max_weight": rmean("max_weight"),
        "particle_spread": rmean("spread_trace"),
        "unique_ancestor_rate_all": rmean("unique_particles") / n_particles,
        "unique_ancestor_rate_resampled": (
            rmean("unique_particles", sel=resampled) / n_particles),
        "resample_rate": rmean("resampled"),
    }


def _save_per_seed(base, method, seed, zone, r, cfg):
    """Save the full artifact set of one (method, seed) run."""
    tag = f"{sanitize(method)}_seed{seed}"
    if zone is not None:
        tag += f"_zone{zone}"
    d = os.path.join(base, "runs", tag)
    os.makedirs(d, exist_ok=True)

    # Step series plus masks
    np.savez_compressed(
        os.path.join(d, "metrics.npz"),
        step_index=r["step_index"],
        report_mask=r["report_mask"], selection_mask=r["selection_mask"],
        straddle_mask=r["straddle_mask"], switch_mask=r["switch_mask"],
        regime_ids=r["regime_ids"],
        **{f"metric_{k}": v for k, v in r["metrics"].items()})
    # Predictions, targets and probabilities, per reporting sample
    np.savez_compressed(os.path.join(d, "predictions.npz"), **r["predictions"])
    # Train/test indices, for leakage checks and reproducibility
    idx = {}
    for i, (tr, te) in enumerate(zip(r["train_indices"], r["test_indices"])):
        idx[f"train_{i}"] = np.asarray(tr, int)
        idx[f"test_{i}"] = np.asarray(te, int)
    np.savez_compressed(os.path.join(d, "indices.npz"), **idx)
    # Diagnostic history (filters only)
    if r.get("history"):
        np.savez_compressed(os.path.join(d, "diagnostics.npz"), **r["history"])
        write_json(timing_report(r["history"]), os.path.join(d, "timings.json"))
    write_json({"method": method, "seed": int(seed),
                "zone": zone, "n_resets": r["n_resets"]},
               os.path.join(d, "meta.json"))


def _run_one_config(cfg, out_dir, selected, eval_seeds, n_main, methods,
                    zone=None):
    """Run every method on every seed and return the summary rows."""
    overrides = {}
    if zone is not None:
        overrides["zone"] = zone

    key = "mse" if cfg["task_type"] == "regression" else "f1"
    ztag = f"[zone {zone}] " if zone is not None else ""
    rows = []
    # metric -> {method -> per-seed reporting values}, for the paired tests
    per_seed_by_metric = {key: {}}
    for mk in _TEST_METRICS:
        per_seed_by_metric.setdefault(mk, {})
    for m in methods:
        params = get_params(selected, m, n_main)
        # run_seeds parallelizes over seeds. The benchmark object itself is
        # seed-independent, so it is built once; build_functions(seed)
        # produces the per-seed data.
        bench = build_benchmark(cfg, **overrides)
        results = run_seeds(m, bench, n_main, params, eval_seeds)
        vals, diags = [], []
        extra = {mk: [] for mk in _TEST_METRICS}
        for s, r in zip(eval_seeds, results):
            mask = region_mask(r, "report")   # straddling excluded
            vals.append(np.nanmean(np.asarray(r["metrics"][key])[mask]))
            for mk in _TEST_METRICS:
                series = r["metrics"].get(mk)
                extra[mk].append(float("nan") if series is None
                                 else np.nanmean(np.asarray(series)[mask]))
            if r.get("history"):
                # Pass the straddle-excluded mask, not the raw report_mask.
                diags.append(_report_diag(r["history"], mask, n_main))
            _save_per_seed(out_dir, m, s, zone, r, cfg)
        per_seed_by_metric[key][m] = vals
        for mk in _TEST_METRICS:
            per_seed_by_metric[mk][m] = extra[mk]
        # Report the secondary metrics too, so the tested values appear in
        # the table alongside the tests.
        for mk in _TEST_METRICS:
            if np.all(~np.isfinite(extra[mk])):
                continue
            emu, esd = mean_std(extra[mk])
            rows.append({"method": m, "N": n_main, "zone": zone,
                         "kind": "performance", "metric": mk,
                         "mean": emu, "std": esd, "n_seeds": len(eval_seeds)})
        mu, sd = mean_std(vals)
        rows.append({"method": m, "N": n_main, "zone": zone, "kind": "performance",
                     "metric": key, "mean": mu, "std": sd,
                     "n_seeds": len(eval_seeds)})
        print(f"  {ztag}{m:12s} {key}={mu:.4f} ± {sd:.4f}")
        # Diagnostics: mean and SD across seeds over the reporting window.
        for dk in _DIAG_KEYS:
            dv = [d[dk] for d in diags if np.isfinite(d.get(dk, np.nan))]
            if dv:
                dm, ds = mean_std(dv)
                rows.append({"method": m, "N": n_main, "zone": zone,
                             "kind": "diagnostic", "metric": dk,
                             "mean": dm, "std": ds, "n_seeds": len(dv)})

    # Paired t-test and Wilcoxon for every pair in the table. Each metric
    # forms its own comparison family, and Holm-adjusted p-values are
    # reported alongside the raw ones.
    for metric_key in (key, *_TEST_METRICS):
        table = per_seed_by_metric.get(metric_key, {})
        usable = {m: v for m, v in table.items()
                  if np.any(np.isfinite(np.asarray(v, float)))}
        if len(usable) < 2:
            continue
        lower_better = metric_key != "f1"
        pairs = all_pairs_compare(usable, methods=[m for m in methods
                                                   if m in usable],
                                  lower_is_better=lower_better)
        print(f"  {ztag}--- paired tests ({metric_key}, {len(pairs)} pairs, "
              f"Holm-adjusted) ---")
        for pr in pairs:
            rows.append({
                "method": f"{pr['a']}_vs_{pr['b']}", "N": n_main, "zone": zone,
                "kind": "paired", "metric": f"{metric_key}_paired",
                "mean_difference": pr["mean_diff"],
                "std_difference": pr["std_diff"],
                "paired_t_p": pr["paired_t_p"],
                "wilcoxon_p": pr["wilcoxon_p"],
                "paired_t_p_holm": pr["paired_t_p_holm"],
                "wilcoxon_p_holm": pr["wilcoxon_p_holm"],
                "better": pr["better"], "n_seeds": pr["n_seeds"],
            })
            print(f"  {ztag}{pr['a']:12s}−{pr['b']:12s} "
                  f"Δ={pr['mean_diff']:+.4f}  t-p={pr['paired_t_p']:.3g} "
                  f"(Holm {pr['paired_t_p_holm']:.3g})  "
                  f"W-p={pr['wilcoxon_p']:.3g} "
                  f"(Holm {pr['wilcoxon_p_holm']:.3g})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n_main = cfg["n_particles"]["main"]
    methods = [m for m in cfg["methods"] if m != "NoChange"]
    out_dir = os.path.join(_REPO, cfg["output_dir"], "main")

    all_rows = _run_one_config(
        cfg, out_dir, selected, eval_seeds, n_main, methods)

    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=all_rows, diagnostics={})
    print(f"saved: {out_dir}  (per-seed artifacts in {out_dir}/runs/)")


if __name__ == "__main__":
    main()
