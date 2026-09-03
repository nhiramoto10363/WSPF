#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers shared by the scripts.

  - loading a YAML config
  - constructing a benchmark from that config
  - grid search over the selection window, with a grid-edge warning
  - reducing a run to its primary score (minimize MSE, maximize F1)

The scripts drive src.evaluation.runner through this thin layer.
"""

from __future__ import annotations

import inspect
import itertools
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarks import get_benchmark  # noqa: E402
from src.evaluation import run_method, run_seeds  # noqa: E402
from src.evaluation import resolve_workers, _init_worker  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")

_BENCH_CLASSES = None


def _bench_class(name):
    from src.benchmarks import (
        RegressionSwitchBenchmark, EmailBenchmark, InsectsBenchmark)
    return {
        "regression": RegressionSwitchBenchmark,
        "email": EmailBenchmark,
        "insects": InsectsBenchmark,
    }[name]


# ======================================================================
# config
# ======================================================================
def load_config(name_or_path):
    """Load a config by name (regression / email / insects) or by path."""
    if os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(_CONFIG_DIR, f"{name_or_path}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_benchmark(cfg, **overrides):
    """Construct the benchmark from the data section of the config."""
    name = cfg["benchmark"]
    cls = _bench_class(name)
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    data = dict(cfg.get("data", {}))
    ev = cfg.get("eval", {})
    # Pass the window boundaries from the eval section to the benchmark.
    if name == "regression":
        if "eval_start" in ev:
            data.setdefault("eval_start", ev["eval_start"])
        if "select_start" in ev:
            data.setdefault("select_start", ev["select_start"])
        if "select_end" in ev:
            data.setdefault("select_end", ev["select_end"])
    if name == "email":
        rs = ev.get("report_start", data.get("report_start"))
        if rs is not None:
            data["report_start"] = rs
    # Resolve relative paths against the repository root, so data/ is found
    # from any working directory.
    for k in ("arff_path", "csv_path"):
        v = data.get(k)
        if isinstance(v, str) and not os.path.isabs(v):
            data[k] = os.path.join(_REPO_ROOT, v)
    # Drop Nones coming from the config, but honour an explicit override
    # even when it is None (e.g. grad_clip_norm=None).
    kwargs = {k: v for k, v in data.items() if k in accepted and v is not None}
    for k, v in overrides.items():
        if k in accepted:
            kwargs[k] = v
    return get_benchmark(name, **kwargs)


# ======================================================================
# Primary score
# ======================================================================
def region_mask(result, region):
    """Mask of the steps to aggregate over.

    region='selection' is the hyper-parameter selection window, 'report' the
    final evaluation window, and 'all' every step. Straddling blocks are
    excluded in all three cases.

    An empty selection or reporting window raises instead of silently
    falling back to the whole stream, which would invert the point of the
    split. That catches misconfigurations early (eval_start beyond T,
    report_start beyond the stream length, and so on).
    """
    n = len(next(iter(result["metrics"].values())))
    if region == "selection":
        mask = np.asarray(result.get("selection_mask", np.zeros(n, bool)))
    elif region == "report":
        mask = np.asarray(result.get("report_mask", np.zeros(n, bool)))
    elif region == "all":
        mask = np.ones(n, dtype=bool)
    else:
        raise ValueError(f"unknown region: {region!r}")
    # Exclude straddling blocks first, so a window that empties out only
    # after the exclusion is caught too.
    mask = mask & ~np.asarray(result.get("straddle_mask", np.zeros(n, bool)))
    if region in ("selection", "report") and not mask.any():
        raise ValueError(
            f"region '{region}' has no samples (empty after excluding "
            f"straddling blocks); check eval_start / report_start against "
            f"the stream length.")
    return mask


def masked_history(history, mask):
    """Restrict a filter history to the masked steps.

    Only arrays whose leading axis matches the mask length are indexed;
    anything else (scalars and so on) passes through. A (T, N) array such as
    rho becomes (n_selected, N).
    """
    mask = np.asarray(mask, dtype=bool)
    out = {}
    for key, value in history.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == mask.size:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def primary_score(result, cfg, region="selection"):
    """Reduce one run to a scalar score, always lower-is-better.

    Regression returns the mean MSE; classification returns the negated mean
    F1, so that both can be minimized. region is 'selection' (the default),
    'report' or 'all'.
    """
    metrics = result["metrics"]
    mask = region_mask(result, region)
    if cfg["task_type"] == "regression":
        v = np.asarray(metrics["mse"])[mask]
        return float(np.nanmean(v))
    else:
        v = np.asarray(metrics["f1"])[mask]
        return -float(np.nanmean(v))


# ======================================================================
# Grid search over the selection seeds
# ======================================================================
def _param_grid(method, grid):
    """Yield the parameter grid (a Cartesian product) for one method.

    The baselines' own hyper-parameters are searched on the same window:
    ph_delta, ph_lambda and ph_alpha for PH-SGD, and window (W) and n_passes
    (K) for Window-SGD. A key absent from the config grid contributes a
    single default value.
    """
    etas = grid["eta"]
    if method == "SGD":
        for eta, ps in itertools.product(etas, grid["prior_std"]):
            yield {"eta": eta, "prior_std": ps}
    elif method == "PH-SGD":
        deltas = grid.get("ph_delta", [0.005])
        lambdas = grid.get("ph_lambda", [5.0])
        alphas = grid.get("ph_alpha", [0.9999])
        for eta, ps, dl, lm, al in itertools.product(
                etas, grid["prior_std"], deltas, lambdas, alphas):
            yield {"eta": eta, "prior_std": ps,
                   "ph_delta": dl, "ph_lambda": lm, "ph_alpha": al}
    elif method == "Window-SGD":
        windows = grid.get("window", [5])
        passes = grid.get("n_passes", [1])
        for eta, ps, w, k in itertools.product(
                etas, grid["prior_std"], windows, passes):
            yield {"eta": eta, "prior_std": ps, "window": w, "n_passes": k}
    elif method == "WSPF-A":
        for eta, ss, ps, beta in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"], grid["beta"]):
            yield {"eta": eta, "sigma_sys": ss, "prior_std": ps, "beta": beta}
    else:  # PF, WSPF-B, Oracle
        for eta, ss, ps in itertools.product(
                etas, grid["sigma_sys"], grid["prior_std"]):
            yield {"eta": eta, "sigma_sys": ss, "prior_std": ps}


def _is_boundary(best, grid):
    """Which parameters of `best` sit on an edge of their grid.

    The baselines' own axes (window, n_passes, ph_*) are included. Missing
    them would leave Window-SGD and PH-SGD searched over effectively narrower
    grids than the filters, breaking the premise that every method follows
    the same selection protocol and biasing the comparison against the
    baselines.

    Axes with a single candidate are skipped: min equals max there, so they
    would always look like an edge.

    Returns
    -------
    list[str]  parameter names with the direction, e.g. "prior_std(upper)"
    """
    hits = []
    for key in ("eta", "sigma_sys", "prior_std", "beta",
                "window", "n_passes", "ph_delta", "ph_lambda", "ph_alpha"):
        if key not in best or key not in grid:
            continue
        g = grid[key]
        if len(set(g)) <= 1:          # single-point axis: nothing to extend
            continue
        if best[key] == min(g):
            hits.append(f"{key}(lower)")
        elif best[key] == max(g):
            hits.append(f"{key}(upper)")
    return hits


def _grid_eval_job(args):
    """Evaluate one (params, ctx, seed) grid job.

    Defined at module level so that ProcessPoolExecutor can pickle it. Only
    the scalar selection score crosses the process boundary, never the heavy
    result dict.
    """
    method, cfg, ctx, n_particles, params, seed = args
    bench = build_benchmark(cfg, **ctx)
    r = run_method(method, bench, n_particles, params, seed=seed,
                   collect_diagnostics=False)
    return primary_score(r, cfg, region="selection")


def grid_search(method, cfg, n_particles, selection_seeds, emit=print,
                contexts=None, n_workers=None):
    """Return the best parameters by selection-window score.

    contexts is a list of benchmark-construction overrides; with more than
    one, candidates are ranked by the mean score over all contexts and
    selection seeds. The three benchmarks here all run single-context.

    Each (candidate, context, seed) triple is one job, evaluated in parallel
    processes; the worker count comes from n_workers or from
    WSPF_NUM_WORKERS / NCPUS. Candidates are ranked by the mean over their
    jobs, so parallel and sequential runs agree exactly.
    """
    grid = cfg["grid"]
    contexts = contexts or [{}]
    candidates = list(_param_grid(method, grid))

    # Flatten into (candidate_index, job_args) pairs.
    jobs = []
    for ci, params in enumerate(candidates):
        for ctx in contexts:
            for s in selection_seeds:
                jobs.append((ci, (method, cfg, ctx, n_particles, params, s)))

    workers = resolve_workers(len(jobs), n_workers)
    scores_by_cand = defaultdict(list)
    if workers <= 1:
        for ci, arg in jobs:
            scores_by_cand[ci].append(_grid_eval_job(arg))
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker) as ex:
                for (ci, _), sc in zip(jobs,
                                       ex.map(_grid_eval_job,
                                              [a for _, a in jobs])):
                    scores_by_cand[ci].append(sc)
        except Exception as e:
            emit(f"  [warning] {method}: parallel grid failed ({e!r}); "
                 f"running sequentially")
            scores_by_cand = defaultdict(list)
            for ci, arg in jobs:
                scores_by_cand[ci].append(_grid_eval_job(arg))

    best, best_score = None, np.inf
    for ci, params in enumerate(candidates):
        sc = float(np.nanmean(scores_by_cand[ci]))
        if sc < best_score:
            best_score, best = sc, dict(params)
    hits = _is_boundary(best, grid)
    if hits:
        emit(f"  [warning] {method}: best point on the grid edge {hits}; "
             f"consider extending that axis by one geometric step and "
             f"re-running (at most twice).")
    emit(f"  {method}: best={best} (score={best_score:.4f})")
    return best, best_score


# ======================================================================
# Hyper-parameter I/O
# ======================================================================
def hp_path(cfg):
    return os.path.join(_REPO_ROOT, cfg["output_dir"], "selected_params.json")


def resolve_seeds(cfg, kind):
    """Seed list for kind='selection' or 'evaluation'."""
    return list(cfg["seeds"][kind])


def load_selected(cfg):
    """Read the selected_params.json written by grid_search.py."""
    import json
    path = hp_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run scripts/grid_search.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_params(selected, method, n_particles):
    """Pick the best parameters for (method, N) out of `selected`."""
    if method in ("PF", "WSPF-A", "WSPF-B"):
        return selected["by_n_particles"][str(n_particles)][method]
    return selected["no_n"][method]


def benchmark_contexts(cfg, selected):
    """Benchmark-construction overrides for the auxiliary experiments.

    None of regression, email or insects needs an override, so this always
    returns the single context [{}]. The scripts still loop over contexts, so
    a benchmark that does need them can be added without rewriting them.
    """
    return [{}]
