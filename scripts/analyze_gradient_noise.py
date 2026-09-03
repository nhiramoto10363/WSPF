#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sampling noise of the mini-batch mean gradient.

The object of study is the noise of the *batch-mean* gradient,
ghat(theta; B) - grad L(theta), not the per-sample gradient.

Only the regression benchmark qualifies, since it is the one whose true
generating distribution can be sampled repeatedly, giving an accurate
estimate of the population gradient. Each phase specifies the parameter that
generates the data (theta* at data_step) and the parameter at which the
gradient is evaluated (theta at eval_step) independently, so that the
Gaussianity, anisotropy and batch-size dependence of the noise can be probed
as a function of B and of the distance from a concept switch.

    same-theta phases: data_step == eval_step
    old-theta phases:  data from after the switch, evaluated at the
                       pre-switch parameters (sp - 1)

The old-theta phases reproduce what a learner actually experiences just after
a switch: still holding the old parameters while facing the new distribution.
(sp is the first switch point; indices are clipped to [0, T-1].)

For each (B, phase):
  1. estimate grad L(theta*_phase) from about 100,000 Monte-Carlo samples of
     the true distribution (x ~ N(0,1), y = f(theta*; x) + N(0, sigma^2));
  2. draw 3,000 independent mini-batches of size B from the same
     distribution and collect noise = ghat - grad L, giving (3000, d);
  3. report the mean absolute skewness, the mean kurtosis, a summary of the
     componentwise normality tests (rejection rate, median p, Fisher-combined
     p), the distribution of the Mahalanobis distance against chi^2(d), and
     the covariance spectrum (condition number, effective rank, share of the
     leading eigenvalue).

The Mahalanobis comparison uses a covariance estimated from the same sample,
so the KS statistic against chi^2(d) is a descriptive goodness-of-fit
diagnostic rather than a formal normality test.

Memory stays bounded: both the population estimate and the mini-batch draws
are chunked. WSPF_GN_POP and WSPF_GN_BATCHES shrink the sample sizes for a
smoke test.

Usage:
    python scripts/analyze_gradient_noise.py --benchmark regression
"""

import argparse
import os

import numpy as np
from scipy import stats

from _common import load_config, build_benchmark
from src.evaluation import write_table
from src.models import create_regression_per_sample_grad_fn

# Monte-Carlo sample size, number of independent mini-batches, chunk width
POP_SAMPLES = int(os.environ.get("WSPF_GN_POP", 100_000))
N_BATCHES = int(os.environ.get("WSPF_GN_BATCHES", 3000))
POP_CHUNK = 5000

BATCH_SIZES = [8, 16, 32, 64]


def _phase_specs(switch_points, T):
    """Return the (data_step, eval_step) pair of each phase.

    data_step selects the true parameters theta* that generate the data, and
    eval_step the theta at which the gradient is evaluated. The same-theta
    phases have data_step == eval_step; the old-theta phases take data from
    after a switch and evaluate at the pre-switch parameters (sp - 1). All
    indices are clipped to [0, T-1].
    """
    sp = switch_points[0] if switch_points else T // 2

    def clip(s):
        return min(max(s, 0), T - 1)

    stable = clip(sp - 30)
    post0, post5, post20 = clip(sp), clip(sp + 5), clip(sp + 20)
    old = clip(sp - 1)                 # pre-switch parameters
    return {
        # same-theta phases
        "stable":       (stable, stable),
        "post+0":       (post0,  post0),
        "post+5":       (post5,  post5),
        "post+20":      (post20, post20),
        # old-theta phases: post-switch data, pre-switch evaluation point
        "post+0_oldθ":  (post0,  old),
        "post+5_oldθ":  (post5,  old),
        "post+20_oldθ": (post20, old),
    }


def _population_grad(model, per_sample_grad, theta_star, theta_eval,
                     noise_std, rng, n_samples=POP_SAMPLES, chunk=POP_CHUNK):
    """Monte-Carlo estimate of grad L(theta_eval) under the true distribution.

    theta_eval is a single evaluation point of shape (1, d). Per-sample
    gradients (1, m, d) are squeezed to (m, d) and averaged over the whole
    sample; chunking avoids allocating (m, d) at once.
    """
    d = theta_eval.shape[1]
    sum_g = np.zeros(d)
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = per_sample_grad(theta_eval, X, y)[0]          # (m, d)
        sum_g += g.sum(axis=0)
        done += m
    return sum_g / n_samples                               # ∇L (d,)


def _minibatch_noises(model, per_sample_grad, theta_star, theta_eval,
                      noise_std, B, pop_grad, rng,
                      n_batches=N_BATCHES, chunk=POP_CHUNK):
    """Draw n_batches independent mini-batches of size B and collect the noise.

    Returns noises of shape (n_batches, d). Mini-batches are drawn in chunks,
    but only their means are kept, so memory stays bounded.
    """
    d = theta_eval.shape[1]
    noises = np.empty((n_batches, d))
    mb_per_chunk = max(1, chunk // B)     # mini-batches per chunk
    bi = 0
    while bi < n_batches:
        nb = min(mb_per_chunk, n_batches - bi)
        m = nb * B
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = per_sample_grad(theta_eval, X, y)[0]          # (m, d)
        gbar = g.reshape(nb, B, d).mean(axis=1)           # (nb, d) batch means
        noises[bi:bi + nb] = gbar - pop_grad[None, :]
        bi += nb
    return noises


def _noise_stats(noises):
    """Summary statistics of the batch-mean gradient noise, shape (n, d)."""
    n, d = noises.shape

    # The population gradient is itself a Monte-Carlo estimate, so the mean
    # of the noise is not exactly zero; the covariance and the Mahalanobis
    # distance use the centred deviations.
    dev = noises - noises.mean(axis=0, keepdims=True)

    # Mean |skewness| over components (signed values would cancel) and mean
    # kurtosis. Both are central moments, so noises and dev give the same
    # answer.
    mean_abs_skew = float(np.mean(np.abs(stats.skew(noises, axis=0))))
    mean_kurtosis = float(np.mean(stats.kurtosis(noises, axis=0)))

    # Sample covariance from the centred deviations, with a small jitter so
    # that it can be inverted.
    Sigma = np.cov(dev, rowvar=False)
    if Sigma.ndim == 0:                      # guard for d == 1
        Sigma = Sigma.reshape(1, 1)
    jitter = 1e-8 * (np.trace(Sigma) / d + 1e-30)
    Sigma_j = Sigma + jitter * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma_j)

    # Squared Mahalanobis distance against chi^2(df=d)
    md2 = np.einsum("ni,ij,nj->n", dev, Sigma_inv, dev)   # (n,)
    mean_maha2 = float(np.mean(md2))
    ks = stats.kstest(md2, "chi2", args=(d,))
    maha_ks_p = float(ks.pvalue)

    # Covariance spectrum
    evals = np.linalg.eigvalsh(Sigma_j)
    evals = np.clip(evals, 0.0, None)[::-1]
    total = evals.sum() + 1e-30
    cond = float(evals[0] / max(evals[-1], 1e-30))
    eff_rank = float((evals.sum() ** 2) / (np.sum(evals ** 2) + 1e-30))
    top_ratio = float(evals[0] / total)

    # Componentwise normality tests, summarized in a well-defined way (only
    # for n >= 20). Averaging p-values is meaningless, so the rejection rate,
    # the median and the Fisher-combined p are reported instead.
    if n >= 20:
        pvals = np.array([stats.normaltest(noises[:, j]).pvalue
                          for j in range(d)])
        frac_reject_normal = float(np.mean(pvals < 0.05))
        median_normal_p = float(np.median(pvals))
        fisher_normal_p = float(stats.combine_pvalues(pvals, method="fisher")[1])
    else:
        frac_reject_normal = float("nan")
        median_normal_p = float("nan")
        fisher_normal_p = float("nan")

    return {
        "mean_abs_skew": mean_abs_skew,
        "mean_kurtosis": mean_kurtosis,
        "mean_maha2": mean_maha2,
        "expected_maha2": float(d),
        "maha_ks_p": maha_ks_p,
        "cond_number": cond,
        "effective_rank": eff_rank,
        "top_eig_ratio": top_ratio,
        "frac_reject_normal": frac_reject_normal,
        "median_normal_p": median_normal_p,
        "fisher_normal_p": fisher_normal_p,
    }


def analyze(cfg, seed=0):
    """One row of noise statistics per (B, phase)."""
    bench = build_benchmark(cfg)
    model = bench.model
    noise_std = bench.noise_std
    d = bench.param_dim

    # Fix theta_true for this seed
    bench.build_functions(seed=seed)
    theta_true = bench.theta_true
    per_sample_grad = create_regression_per_sample_grad_fn(model, noise_std)

    phase_specs = _phase_specs(bench.switch_points, bench.T)
    rng = np.random.default_rng(12345)

    rows = []
    # (data_step, eval_step) -> grad L. It does not depend on B, so it is
    # computed once and reused.
    pop_cache = {}
    for B in BATCH_SIZES:
        for phase, (data_step, eval_step) in phase_specs.items():
            theta_star = theta_true[data_step]            # generating theta*
            theta_eval = theta_true[eval_step].reshape(1, d)   # evaluation point
            key = (data_step, eval_step)
            if key not in pop_cache:
                pop_cache[key] = _population_grad(
                    model, per_sample_grad, theta_star, theta_eval,
                    noise_std, rng)
            pop_grad = pop_cache[key]

            noises = _minibatch_noises(
                model, per_sample_grad, theta_star, theta_eval,
                noise_std, B, pop_grad, rng)
            st = _noise_stats(noises)
            st.update({"batch_size": B, "phase": phase})
            rows.append(st)
            print(f"B={B:3d} {phase:13s} "
                  f"|skew|={st['mean_abs_skew']:.3f} "
                  f"kurt={st['mean_kurtosis']:.3f} "
                  f"maha2={st['mean_maha2']:.2f}/{d} "
                  f"ks_p={st['maha_ks_p']:.3f} "
                  f"cond={st['cond_number']:.1f} "
                  f"eff_rank={st['effective_rank']:.1f} "
                  f"rej={st['frac_reject_normal']:.2f} "
                  f"fisher_p={st['fisher_normal_p']:.3f}")
    return rows


# Output columns
_COLUMNS = ["batch_size", "phase", "mean_abs_skew", "mean_kurtosis",
            "mean_maha2", "expected_maha2", "maha_ks_p", "cond_number",
            "effective_rank", "top_eig_ratio", "frac_reject_normal",
            "median_normal_p", "fisher_normal_p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="regression",
                    help="regression, or a config path (regression only)")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["task_type"] != "regression":
        raise SystemExit("the gradient-noise analysis supports regression only")

    rows = analyze(cfg, seed=0)
    rows = [{k: r[k] for k in _COLUMNS} for r in rows]   # fix the column order

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "grad_noise")
    os.makedirs(out_dir, exist_ok=True)
    write_table(rows, os.path.join(out_dir, "gradient_noise"))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
