#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The shared run loop.

PF, WSPF-A, WSPF-B, the oracle filter and the point-estimate baselines are
all driven through the same prequential (test-then-train) loop, without any
of them knowing the benchmark. Keeping one loop keeps evaluation,
diagnostics and output consistent across tasks.

At each StreamStep the loop first scores X_test with the current estimate
(theta, or the weighted particle prediction) and only then updates on
X_train (.train for the baselines, .step for the filters). Blocks with
straddles_switch set are recorded but excluded downstream, in region_mask,
from every reported metric and switch-aligned analysis.

Filter seeds are offsets from the base seed: PF +1, WSPF-B +3, WSPF-A +5,
Oracle +7, and +10 for the baselines.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.filters import ParticleFilter, WSPF_B, WSPF_A, OraclePF
from src.baselines import OnlineSGD, PHSGD, WindowSGD

from . import metrics as M


# ======================================================================
# Process-parallel helpers (over seeds and grid candidates)
# ======================================================================
def _init_worker():
    """Worker initializer: avoid oversubscribing BLAS.

    With many worker processes, letting numpy and BLAS thread inside each one
    gives (processes x BLAS threads) of parallelism and ends up slower, so
    each worker is pinned to a single BLAS thread. Sequential runs do not set
    this and keep multithreaded BLAS.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def resolve_workers(n_jobs, requested=None):
    """Decide how many worker processes to use.

    With requested=None the value comes from WSPF_NUM_WORKERS, then NCPUS,
    then the CPU count. It is capped at the number of jobs, and a result of
    one means sequential execution, so WSPF_NUM_WORKERS=1 forces that.
    """
    if n_jobs <= 1:
        return 1
    if requested is None:
        requested = int(os.environ.get(
            "WSPF_NUM_WORKERS",
            os.environ.get("NCPUS", os.cpu_count() or 1)))
    return max(1, min(int(requested), n_jobs))


# ======================================================================
# Method groups
# ======================================================================
SGD_METHODS = {"SGD", "PH-SGD", "Window-SGD"}
FILTER_METHODS = {"PF", "WSPF-A", "WSPF-B", "Oracle"}
ALL_METHODS = SGD_METHODS | FILTER_METHODS

# Seed offsets from the base seed, for the filter and the initialization
SEED_OFFSET = {
    "PF": 1,
    "WSPF-B": 3,
    "WSPF-A": 5,
    "Oracle": 7,
    "SGD": 10,
    "PH-SGD": 10,
    "Window-SGD": 10,
}


# ======================================================================
# Estimator construction
# ======================================================================
def _build_estimator(method, benchmark, n_particles, params, seed, funcs,
                     filter_seed=None):
    """Build the estimator for one method.

    Passing filter_seed overrides the per-method offset, which is how the
    oracle comparison runs every method on common random numbers.
    """
    d = benchmark.param_dim
    eta = params["eta"]
    # sigma_sys is required by the filters only; the baselines have none.
    sigma_sys = params.get("sigma_sys", 0.0)
    prior_std = params["prior_std"]
    prior_mean = params.get("prior_mean", 0.0)
    ess_ratio = params.get("ess_resample_ratio", 0.5)
    grad_clip = getattr(benchmark, "grad_clip_norm", None)
    fseed = (seed + SEED_OFFSET[method]) if filter_seed is None else filter_seed

    if method == "PF":
        return ParticleFilter(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, seed=fseed,
        )
    if method == "WSPF-B":
        return WSPF_B(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, grad_clip_norm=grad_clip, seed=fseed,
        )
    if method == "WSPF-A":
        return WSPF_A(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, grad_clip_norm=grad_clip,
            beta=params.get("beta", 0.9), seed=fseed,
        )
    if method == "Oracle":
        return OraclePF(
            n_particles, d, eta=eta, sigma_sys=sigma_sys,
            prior_mean=prior_mean, prior_std=prior_std,
            ess_resample_ratio=ess_ratio, grad_clip_norm=grad_clip, seed=fseed,
        )
    if method == "SGD":
        return OnlineSGD(d, eta, prior_std, funcs["grad_fn"],
                         seed=fseed, grad_clip_norm=grad_clip)
    if method == "PH-SGD":
        return PHSGD(d, eta, prior_std, funcs["grad_fn"],
                     seed=fseed, grad_clip_norm=grad_clip,
                     ph_delta=params.get("ph_delta", 0.005),
                     ph_lambda=params.get("ph_lambda", 5.0),
                     ph_alpha=params.get("ph_alpha", 0.9999))
    if method == "Window-SGD":
        return WindowSGD(d, eta, prior_std, funcs["grad_fn"],
                         window=params.get("window", 5),
                         n_passes=params.get("n_passes", 1),
                         seed=fseed, grad_clip_norm=grad_clip)
    raise ValueError(f"unknown method: {method!r} (valid: {sorted(ALL_METHODS)})")


# ======================================================================
# Predictions, uniform across baselines and filters
# ======================================================================
def _predict_particles_weights(method, estimator):
    """Return (particles (N, d), weights (N,)) from any estimator."""
    if method in FILTER_METHODS:
        return estimator.particles, estimator.weights
    # A point estimate is treated as a single particle.
    theta = np.asarray(estimator.predict_theta()).reshape(1, -1)
    return theta, np.array([1.0])


# ======================================================================
# Main entry point
# ======================================================================
def run_method(method, benchmark, n_particles, params, seed,
               collect_diagnostics=True, filter_seed=None, max_steps=None,
               return_estimator=False):
    """Run one method on one seed through the prequential loop.

    Parameters
    ----------
    method : str
        {"SGD","PH-SGD","Window-SGD","PF","WSPF-A","WSPF-B","Oracle"}
    benchmark : Benchmark
    n_particles : int
        Ignored by the baselines.
    params : dict
        {"eta","sigma_sys","prior_std", plus "beta" for WSPF-A, ...}
    seed : int
        Base seed for the data and the offsets.
    collect_diagnostics : bool
        The filters always record their history; this only controls whether
        it is returned.
    return_estimator : bool
        Include the estimator as of the last step in the result, for uses
        that need the particle set itself (such as the snapshot figure,
        together with max_steps). It only adds a key to the returned dict and
        never affects the numbers.

    Returns
    -------
    dict {
      "metrics": {metric_name: (T,) ndarray},
      "straddle_mask": (T,) bool,    blocks crossing a switch
      "switch_mask": (T,) bool,      known switch steps
      "selection_mask": (T,) bool,   selection window
      "report_mask": (T,) bool,      reporting window
      "history": dict or None,
      "n_resets": int,
      "regime_ids": (T,) int,
      "step_index": (T,) int,
      "predictions": {"y", ("mean","std") | ("probs")},  per sample, reporting
      "train_indices": list[ndarray], "test_indices": list[ndarray],
    }
    """
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method: {method!r}")

    funcs = benchmark.build_functions(seed)
    grad_fn = funcs["grad_fn"]
    per_sample_grad_fn = funcs.get("per_sample_grad_fn")
    loglik_fn = funcs["loglik_fn"]
    predict_fn = funcs["predict_fn"]
    task_type = benchmark.task_type
    is_reg = task_type == "regression"
    # More than two classes means the multiclass path.
    n_classes = int(getattr(benchmark, "n_classes", 2))
    is_multiclass = (not is_reg) and n_classes > 2
    obs_sigma = float(funcs.get("obs_sigma", 0.0)) if is_reg else 0.0
    levels = (0.5, 0.8, 0.9, 0.95)

    estimator = _build_estimator(method, benchmark, n_particles, params,
                                 seed, funcs, filter_seed=filter_seed)

    # Supplies the per-step oracle statistics (regression only).
    oracle_hook = getattr(benchmark, "oracle_stats_fn_for_step", None)
    _oseed = (seed + SEED_OFFSET.get(method, 0)) if filter_seed is None else filter_seed
    oracle_rng = np.random.default_rng(_oseed + 100)

    # Metric buffers
    metric_lists = {}
    straddle = []
    switch = []
    selection = []
    report = []
    regime_ids = []
    step_indices = []
    # Per-sample predictions over the reporting window
    rep_y, rep_mean, rep_std, rep_probs = [], [], [], []
    rep_block_step, rep_block_len = [], []   # step index and length per block
    train_idx_list, test_idx_list = [], []

    def _push(name, val):
        metric_lists.setdefault(name, []).append(float(val))

    for _i, stp in enumerate(benchmark.stream(seed)):
        # max_steps truncates the loop itself, so timing runs need not walk
        # the whole stream.
        if max_steps is not None and _i >= max_steps:
            break
        Xte, yte = stp.X_test, stp.y_test
        Xtr, ytr = stp.X_train, stp.y_train
        has_test = Xte is not None and np.asarray(Xte).shape[0] > 0

        # Only reporting blocks that do not straddle a switch contribute
        # per-sample predictions, which keeps straddling blocks out of the
        # calibration metrics and reliability diagrams.
        is_reported_block = bool(stp.is_report_step) and not bool(stp.straddles_switch)

        # -------- 1) Evaluate, before training --------
        particles, weights = _predict_particles_weights(method, estimator)
        if has_test and is_multiclass:
            # Multiclass: average the per-particle probabilities, then argmax.
            yint = np.asarray(yte, dtype=np.int64).ravel()
            probs = M.weighted_prediction_proba(
                predict_fn, particles, weights, Xte)
            hard = probs.argmax(axis=1).astype(np.float64)
            _push("accuracy", M.accuracy(hard, yint))
            _push("f1", M.macro_f1(hard, yint, n_classes))       # primary metric
            _push("macro_f1", M.macro_f1(hard, yint, n_classes))
            _push("nll", M.nll_categorical(probs, yint))
            # The mixture of categoricals is again a categorical, so the
            # mixture NLL coincides with nll here.
            _push("nll_mixture", M.nll_categorical(probs, yint))
            _push("brier", M.brier_multiclass(probs, yint, n_classes))
            primary_err = 1.0 - M.accuracy(hard, yint)
            if is_reported_block:
                rep_probs.append(np.asarray(probs, np.float64))   # (B, C)
                rep_y.append(yint.astype(np.float64))
                rep_block_step.append(int(stp.step_index))
                rep_block_len.append(int(yint.size))
        elif has_test:
            # Compute the per-particle predictions once and derive both the
            # moment-matched and the mixture NLL from them.
            preds_particles = M.particle_predictions(predict_fn, particles, Xte)
            pred_mean, pred_var = M.weighted_moments(preds_particles, weights)
            yte_arr = np.asarray(yte, dtype=np.float64).ravel()
            if is_reg:
                _push("mse", M.test_mse(yte_arr, pred_mean))
                _push("mae", M.test_mae(yte_arr, pred_mean))
                pred_std = M.prediction_std_with_noise(pred_var, obs_sigma)
                _push("nll", M.nll_gaussian(yte_arr, pred_mean, pred_std))
                # For regression the mixture NLL differs from the
                # single-Gaussian nll, and it is the correct predictive
                # distribution of a particle filter.
                _push("nll_mixture", M.nll_gaussian_mixture(
                    yte_arr, preds_particles, weights, obs_sigma))
                _push("crps", M.crps_gaussian(yte_arr, pred_mean, pred_std))
                cw = M.coverage_and_width(yte_arr, pred_mean, pred_std, levels)
                for lvl in levels:
                    _push(f"coverage_{lvl:.2f}", cw[lvl]["coverage"])
                    _push(f"width_{lvl:.2f}", cw[lvl]["width"])
                primary_err = M.test_mse(yte_arr, pred_mean)
                if is_reported_block:
                    rep_y.append(yte_arr)
                    rep_mean.append(np.asarray(pred_mean, np.float64).ravel())
                    rep_std.append(np.asarray(pred_std, np.float64).ravel())
                    rep_block_step.append(int(stp.step_index))
                    rep_block_len.append(int(yte_arr.size))
            else:
                # predict_fn returns probabilities, or logits; anything
                # outside [0, 1] is passed through a sigmoid.
                probs = pred_mean
                if np.any(probs < 0.0) or np.any(probs > 1.0):
                    probs = 1.0 / (1.0 + np.exp(-np.clip(probs, -60, 60)))
                hard = (probs > 0.5).astype(np.float64)
                _push("accuracy", M.accuracy(hard, yte_arr))
                _push("f1", M.f1(hard, yte_arr))
                _push("balanced_accuracy", M.balanced_accuracy(hard, yte_arr))
                _push("nll", M.nll_bernoulli(probs, yte_arr))
                # The binary mixture is closed: sum_i w_i Bern(y; p_i)
                # = Bern(y; sum_i w_i p_i), so the mixture NLL equals nll
                # identically. It is recorded anyway to keep the columns
                # aligned with regression.
                _push("nll_mixture", M.nll_bernoulli(probs, yte_arr))
                _push("brier", float(np.mean((probs - yte_arr) ** 2)))
                primary_err = 1.0 - M.accuracy(hard, yte_arr)
                if is_reported_block:
                    rep_probs.append(np.asarray(probs, np.float64).ravel())
                    rep_y.append(yte_arr)
                    rep_block_step.append(int(stp.step_index))
                    rep_block_len.append(int(yte_arr.size))
        else:
            # Empty test block: record NaN to keep the series aligned.
            if is_reg:
                for name in ("mse", "mae", "nll", "nll_mixture", "crps"):
                    _push(name, float("nan"))
                for lvl in levels:
                    _push(f"coverage_{lvl:.2f}", float("nan"))
                    _push(f"width_{lvl:.2f}", float("nan"))
            elif is_multiclass:
                for name in ("accuracy", "f1", "macro_f1", "nll",
                             "nll_mixture", "brier"):
                    _push(name, float("nan"))
            else:
                for name in ("accuracy", "f1", "balanced_accuracy", "nll",
                             "nll_mixture", "brier"):
                    _push(name, float("nan"))
            primary_err = None

        straddle.append(bool(stp.straddles_switch))
        switch.append(bool(getattr(stp, "is_switch_step", False)))
        selection.append(bool(getattr(stp, "is_selection_step", False)))
        report.append(bool(getattr(stp, "is_report_step", True)))
        regime_ids.append(-1 if stp.regime_id is None else int(stp.regime_id))
        step_indices.append(int(stp.step_index))
        train_idx_list.append(np.asarray(stp.train_indices, int))
        test_idx_list.append(np.asarray(stp.test_indices, int))

        # -------- 2) Update (train) --------
        if method in SGD_METHODS:
            estimator.train(Xtr, ytr)
            if primary_err is not None:
                estimator.observe_error(primary_err)
        elif method == "Oracle":
            oracle_stats_fn = None
            if oracle_hook is not None:
                oracle_stats_fn = oracle_hook(stp.step_index, oracle_rng)
            if oracle_stats_fn is None:
                # No oracle wired in: fall back to grad L = ghat, Sigma = 0.
                def oracle_stats_fn(p, X, y):
                    g = per_sample_grad_fn(p, X, y).mean(axis=1)
                    d = p.shape[1]
                    return g, np.zeros((p.shape[0], d, d))
            estimator.step(Xtr, ytr, per_sample_grad_fn, loglik_fn,
                           oracle_stats_fn)
        elif method == "PF":
            estimator.step(Xtr, ytr, grad_fn, loglik_fn)
        else:  # WSPF-A / WSPF-B
            estimator.step(Xtr, ytr, per_sample_grad_fn, loglik_fn)

    metrics_out = {k: np.asarray(v, dtype=np.float64) for k, v in metric_lists.items()}
    history = None
    if collect_diagnostics and method in FILTER_METHODS:
        history = estimator.get_history()

    # Per-sample predictions over the reporting window, used by the
    # calibration and recovery analyses and saved to disk.
    def _cat(lst):
        return np.concatenate(lst) if lst else np.empty(0, np.float64)

    predictions = {"y": _cat(rep_y)}
    if is_reg:
        predictions["mean"] = _cat(rep_mean)
        predictions["std"] = _cat(rep_std)
    else:
        predictions["probs"] = _cat(rep_probs)
    # Sample-to-step mapping: the step index and length of each reporting
    # block, the block offsets into the flattened predictions, and the step
    # index of every individual sample.
    block_step = np.asarray(rep_block_step, dtype=int)
    block_len = np.asarray(rep_block_len, dtype=int)
    predictions["block_step_index"] = block_step
    predictions["block_len"] = block_len
    predictions["offsets"] = np.concatenate([[0], np.cumsum(block_len)])
    predictions["pred_step_index"] = np.repeat(block_step, block_len) \
        if block_step.size else np.empty(0, int)

    return {
        "metrics": metrics_out,
        "straddle_mask": np.asarray(straddle, dtype=bool),
        "switch_mask": np.asarray(switch, dtype=bool),
        "selection_mask": np.asarray(selection, dtype=bool),
        "report_mask": np.asarray(report, dtype=bool),
        "history": history,
        "n_resets": int(getattr(estimator, "n_resets", 0)),
        "regime_ids": np.asarray(regime_ids, dtype=int),
        "step_index": np.asarray(step_indices, dtype=int),
        "predictions": predictions,
        "train_indices": train_idx_list,
        "test_indices": test_idx_list,
        # None unless requested; adding it does not affect the numbers.
        "estimator": estimator if return_estimator else None,
    }


def run_seeds(method, benchmark, n_particles, params, seeds,
              collect_diagnostics=True, n_workers=None, filter_seeds=None):
    """Run run_method over several seeds, returning one result dict per seed.

    Seeds are independent, so the runs are executed in parallel processes;
    the worker count comes from n_workers, or from WSPF_NUM_WORKERS / NCPUS,
    and defaults to the available cores. Results keep the order of `seeds`,
    so the output is deterministic regardless of parallelism, and the loop
    falls back to sequential execution where a pool cannot be created.

    filter_seeds, if given, must match `seeds` in length and sets the
    filter seed of each run explicitly (used for common random numbers in the
    oracle comparison); None uses the per-method offsets.
    """
    seeds = list(seeds)
    fseeds = list(filter_seeds) if filter_seeds is not None else [None] * len(seeds)
    workers = resolve_workers(len(seeds), n_workers)

    def _seq():
        return [run_method(method, benchmark, n_particles, params, s,
                           collect_diagnostics=collect_diagnostics,
                           filter_seed=fs)
                for s, fs in zip(seeds, fseeds)]

    if workers <= 1:
        return _seq()
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker) as ex:
            futs = [ex.submit(run_method, method, benchmark, n_particles,
                              params, s, collect_diagnostics, fs)
                    for s, fs in zip(seeds, fseeds)]
            return [f.result() for f in futs]
    except Exception as e:  # no usable process pool
        print(f"[run_seeds] parallel execution failed ({e!r}); "
              f"falling back to sequential")
        return _seq()
