#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Applicability diagnostic: when does the correction matter?

The quantities are derived in the docstring of
src/evaluation/applicability.py. R_diag >> 1 means the weights are dominated
by the correction and R_diag << 1 that the likelihood dominates; where the
particle-budget condition fails, the weights degenerate and the correction
brings no gain. Tabulating the measured values across benchmarks shows that
the benchmark-dependence of the gain is what the theory predicts, and that R
diagnoses it in advance.

By default the numbers come from existing run_main artifacts
(outputs/<bench>/main/runs/*/diagnostics.npz), so no new runs are needed.
--rerun instead re-runs the filters with the selected hyper-parameters and
collects the diagnostics directly, as a fallback when those logs are absent.

Usage:
    python scripts/analyze_applicability.py --benchmark regression email
"""

import argparse
import glob
import os
import re

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask)
from src.evaluation import (diagnose_applicability, mean_std, run_seeds,
                            sanitize, save_run_dir, write_table)

_REPO = os.path.dirname(os.path.dirname(__file__))

# Only methods that carry a correction: PF has no log R and no rho.
_METHODS = ["WSPF-A", "WSPF-B"]

_RUN_DIR_RE = re.compile(r"^(?P<method>.+)_seed(?P<seed>\d+)(?:_zone\d+)?$")

# Map the runs/<tag>/ directory names written by run_main back to method
# names. The tags come from output.sanitize(method), so the inverse table is
# built from that same function rather than hard-coded.
_TAG_TO_METHOD = {sanitize(m): m
                  for m in ("PF", "WSPF-A", "WSPF-B", "Oracle")}


def _load_from_outputs(cfg, methods):
    """Build {method: [diagnostic dicts]} from the run_main artifacts.

    Reads diagnostics.npz (the history) together with metrics.npz (the
    reporting and straddle masks) and aggregates over the reporting window
    with straddling blocks excluded, matching the main tables.
    """
    root = os.path.join(_REPO, cfg["output_dir"], "main", "runs")
    if not os.path.isdir(root):
        return {}

    by_method = {m: [] for m in methods}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        mt = _RUN_DIR_RE.match(os.path.basename(d))
        if not mt:
            continue
        method = _TAG_TO_METHOD.get(mt.group("method"))
        if method not in by_method:
            continue
        diag_path = os.path.join(d, "diagnostics.npz")
        met_path = os.path.join(d, "metrics.npz")
        if not (os.path.exists(diag_path) and os.path.exists(met_path)):
            continue
        with np.load(diag_path, allow_pickle=False) as z:
            history = {k: z[k] for k in z.files}
        with np.load(met_path, allow_pickle=False) as z:
            mask = z["report_mask"].astype(bool) & ~z["straddle_mask"].astype(bool)
        by_method[method].append(
            (int(mt.group("seed")), history, mask))
    return by_method


def _rerun(cfg, methods, n_main):
    """Re-run the filters with the selected hyper-parameters (--rerun)."""
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    out = {}
    for m in methods:
        params = get_params(selected, m, n_main)
        bench = build_benchmark(cfg)
        results = run_seeds(m, bench, n_main, params, eval_seeds)
        rows = []
        for s, r in zip(eval_seeds, results):
            if r.get("history"):
                rows.append((s, r["history"], region_mask(r, "report")))
        out[m] = rows
    return out


def _analyze_benchmark(name, rerun=False):
    cfg = load_config(name)
    n_main = cfg["n_particles"]["main"]
    methods = [m for m in _METHODS if m in cfg.get("methods", [])]
    if not methods:
        print(f"[{name}] no corrected method in the config; skipping")
        return [], []

    bench = build_benchmark(cfg)
    d = int(bench.param_dim)

    if rerun:
        by_method = _rerun(cfg, methods, n_main)
        source = "rerun"
    else:
        by_method = _load_from_outputs(cfg, methods)
        source = "outputs/main/runs"
        if not any(by_method.values()):
            raise SystemExit(
                f"[{name}] no diagnostics.npz under "
                f"outputs/{name}/main/runs/; run run_main.py first, or pass "
                f"--rerun.")

    print(f"\n===== {name} (d={d}, N={n_main}, source={source}) =====")
    rows, per_seed_rows = [], []
    for m in methods:
        runs = by_method.get(m, [])
        if not runs:
            print(f"  [{m}] no runs; skipping")
            continue
        diags = []
        for seed, history, mask in runs:
            dg = diagnose_applicability(history, d, n_main, mask=mask)
            diags.append(dg)
            per_seed_rows.append({"benchmark": name, "method": m, "seed": seed,
                                  **dg})

        def agg(key):
            v = [x[key] for x in diags if np.isfinite(x.get(key, np.nan))]
            return mean_std(v) if v else (float("nan"), float("nan"))

        rho_m, rho_s = agg("rho_mean")
        ll_m, ll_s = agg("ll_std")
        th_m, th_s = agg("sd_logR_theory")
        ms_m, ms_s = agg("sd_logR_measured")
        rd_m, rd_s = agg("R_diag")
        rdm_m, _ = agg("R_diag_measured")
        req_m, _ = agg("required_N")
        rho_cap = diags[0]["rho_max_supported"]
        n_ok = sum(1 for x in diags if x.get("satisfies_condition"))

        rows.append({
            "benchmark": name, "method": m, "d": d, "N": n_main,
            "rho_mean": rho_m, "rho_sd": rho_s,
            "sd_loglik": ll_m, "sd_loglik_sd": ll_s,
            "sd_logR_theory": th_m, "sd_logR_theory_sd": th_s,
            "sd_logR_measured": ms_m, "sd_logR_measured_sd": ms_s,
            "R_diag": rd_m, "R_diag_sd": rd_s,
            "R_diag_measured": rdm_m,
            "required_N": req_m,
            "rho_max_supported": rho_cap,
            "condition_holds": f"{n_ok}/{len(diags)}",
            "n_seeds": len(diags),
        })
        print(f"  {m:8s} ρ={rho_m:.4f}±{rho_s:.4f}  sd(ℓ)={ll_m:.3f}  "
              f"sd(logR): theory={th_m:.3f} measured={ms_m:.3f}")
        print(f"           R_diag={rd_m:.3f}±{rd_s:.3f}  "
              f"required N=exp(rho^2 d/4)={req_m:.3g}  "
              f"max supported rho={rho_cap:.3f}  "
              f"condition holds {n_ok}/{len(diags)}")
    return rows, per_seed_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", nargs="+", required=True,
                    help="one or more benchmark names, e.g. regression email")
    ap.add_argument("--rerun", action="store_true",
                    help="re-run with the selected hyper-parameters instead "
                         "of reading the stored logs")
    ap.add_argument("--out", default=None,
                    help="output directory (default: output_dir/"
                         "applicability)")
    args = ap.parse_args()

    all_rows, all_per_seed = [], []
    for name in args.benchmark:
        rows, per_seed = _analyze_benchmark(name, rerun=args.rerun)
        all_rows += rows
        all_per_seed += per_seed

    if not all_rows:
        raise SystemExit("no runs available to diagnose")

    out_dir = args.out or os.path.join(
        _REPO, load_config(args.benchmark[0])["output_dir"], "applicability")
    os.makedirs(out_dir, exist_ok=True)
    # Keep the selected hyper-parameters too: rho depends on eta and
    # sigma_cd, so reproducing the diagnostics needs them. Benchmarks without
    # a selection are skipped silently.
    selected = {}
    for name in args.benchmark:
        try:
            selected[name] = load_selected(load_config(name))
        except FileNotFoundError:
            pass
    save_run_dir(out_dir,
                 config={"benchmarks": args.benchmark,
                         "source": "rerun" if args.rerun else "outputs"},
                 selected_params=selected,
                 metrics_rows=all_rows, diagnostics={})
    write_table(all_rows, os.path.join(out_dir, "table_applicability"))
    write_table(all_per_seed, os.path.join(out_dir, "applicability_per_seed"))
    print(f"\nsaved: {out_dir}/table_applicability.*  "
          f"({len(all_rows)} rows, {len(all_per_seed)} per-seed rows)")


if __name__ == "__main__":
    main()
