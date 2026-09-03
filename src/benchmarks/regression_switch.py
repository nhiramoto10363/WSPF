#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark: regime-switching regression (synthetic).

The true parameters alternate between two regimes,
[theta1, theta2, theta1, theta2, theta1], switching at steps 100, 200, 300
and 400, and drift slowly within a regime via a small random walk.

Because the generating distribution is known, the true population gradient
grad L(theta) and the gradient-noise covariance
Sigma(theta) = Cov(ghat_batch) = C(theta)/B can be estimated to high accuracy
by large-sample Monte Carlo at every step and for every particle. The oracle
filter substitutes them directly into the exact correction.
"""

from __future__ import annotations

import numpy as np

from src.models import (
    NeuralNetRegression,
    create_regression_grad_fn,
    create_regression_loglik_fn,
    create_regression_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep

ORACLE_SAMPLES = 10000     # Monte-Carlo samples per step for (grad L, Sigma)
ORACLE_CHUNK = 2500        # chunk size, to bound memory


def oracle_grad_stats(model, particles, theta_star, noise_std, M, B,
                      rng, chunk=ORACLE_CHUNK):
    """Estimate the true (grad L, Sigma) at each particle by Monte Carlo.

    Draws a large sample from the true distribution at time t
    (x ~ N(0,1), y = f(theta*; x) + N(0, sigma^2)) and estimates, for each
    particle theta^i, grad L(theta^i) = E[grad l] and
    Sigma(theta^i) = Cov(ghat_batch) = C(theta^i)/B. The draw is chunked to
    avoid allocating an (N, M, d) array at once.

    Returns
    -------
    grad_L : ndarray (N, d)
    Sigma  : ndarray (N, d, d)
    """
    N, d = particles.shape
    ps_grad_fn = create_regression_per_sample_grad_fn(model, noise_std)
    sum_g = np.zeros((N, d))
    sum_ggT = np.zeros((N, d, d))
    done = 0
    while done < M:
        m = min(chunk, M - done)
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = ps_grad_fn(particles, X, y)            # (N, m, d)
        sum_g += g.sum(axis=1)
        sum_ggT += np.einsum("nmd,nme->nde", g, g)
        done += m
    grad_L = sum_g / M
    # Per-sample gradient covariance, then the batch-mean covariance.
    C = (sum_ggT - M * np.einsum("nd,ne->nde", grad_L, grad_L)) / (M - 1)
    Sigma = C / B
    return grad_L, Sigma


class RegressionSwitchBenchmark(Benchmark):
    """Regime-switching regression benchmark."""

    name = "regression"
    task_type = "regression"
    switch_points = [100, 200, 300, 400]

    def __init__(self, T=500, batch_size=16, test_size=200, noise_std=0.5,
                 hidden_dim=8, input_dim=1, within_regime_drift=0.0005,
                 grad_clip_norm=5.0, oracle_samples=ORACLE_SAMPLES,
                 eval_start=50, select_start=50, select_end=150):
        self.T = int(T)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.noise_std = float(noise_std)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.within_regime_drift = float(within_regime_drift)
        # None disables gradient clipping (used by the oracle comparison).
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.oracle_samples = int(oracle_samples)
        # Reporting window: step >= eval_start. Earlier steps are processed as
        # warm-up but not aggregated.
        self.eval_start = int(eval_start)
        # Selection window [select_start, select_end), centred on the first
        # switch at t=100 so that eta is not selected on a stationary stretch.
        self.select_start = int(select_start)
        self.select_end = int(select_end)

        # One model instance fixes param_dim.
        self.model = NeuralNetRegression(self.input_dim, self.hidden_dim,
                                         output_dim=1, activation="tanh")
        self.param_dim = self.model.param_dim

        self.switch_points = [self.T // 5 * (i + 1) for i in range(4)]

        # Generated data, cached per seed.
        self._cache = {}
        # True parameters of the seed most recently touched.
        self.theta_true = None

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def _generate(self, seed):
        if seed in self._cache:
            return self._cache[seed]

        rng = np.random.default_rng(seed)
        model = self.model
        param_dim = model.param_dim
        T = self.T

        theta_star_1 = rng.normal(0.0, 0.8, size=param_dim)
        theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)
        while np.linalg.norm(theta_star_1 - theta_star_2) < 1.0:
            theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)

        regime_length = T // 5
        switch_times = [regime_length * (i + 1) for i in range(4)]
        regime_thetas = [theta_star_1, theta_star_2, theta_star_1,
                         theta_star_2, theta_star_1]

        theta_true = np.empty((T, param_dim))
        regime_ids = np.empty(T, dtype=int)
        for t in range(T):
            regime_idx = 0
            for st in switch_times:
                if t >= st:
                    regime_idx += 1
            regime_ids[t] = regime_idx
            if t == 0 or t in switch_times:
                theta_true[t] = regime_thetas[regime_idx].copy()
            else:
                theta_true[t] = theta_true[t - 1] + rng.normal(
                    0.0, self.within_regime_drift, size=param_dim)

        X_train, y_train, X_test, y_test = [], [], [], []
        for t in range(T):
            theta_t = theta_true[t: t + 1]
            X = rng.normal(0.0, 1.0, size=(self.batch_size, model.input_dim))
            output, _, _ = model.forward(theta_t, X)
            y = output.squeeze() + rng.normal(0.0, self.noise_std,
                                              size=self.batch_size)
            X_train.append(X)
            y_train.append(y)

            Xte = rng.normal(0.0, 1.0, size=(self.test_size, model.input_dim))
            output_te, _, _ = model.forward(theta_t, Xte)
            yte = output_te.squeeze() + rng.normal(0.0, self.noise_std,
                                                   size=self.test_size)
            X_test.append(Xte)
            y_test.append(yte)

        data = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "theta_true": theta_true, "regime_ids": regime_ids,
            "switch_times": switch_times,
        }
        self._cache[seed] = data
        return data

    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        noise_std = self.noise_std

        # Fix the data for this seed so that oracle_stats_fn can look up the
        # true parameters step by step.
        data = self._generate(seed)
        self.theta_true = data["theta_true"]

        raw_grad = create_regression_grad_fn(model, noise_std)
        per_sample_grad_fn = create_regression_per_sample_grad_fn(
            model, noise_std)
        loglik_fn = create_regression_loglik_fn(model, noise_std)
        clip = self.grad_clip_norm

        def grad_fn(theta, X, y):
            g = raw_grad(theta, X, y)
            if clip is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                scale = np.minimum(1.0, clip / (norms + 1e-12))
                g = g * scale
            return g

        def predict_fn(particles, X):
            output, _, _ = model.forward(particles, X)
            return output.squeeze(-1)   # (N, B)

        return {
            "grad_fn": grad_fn,
            "per_sample_grad_fn": per_sample_grad_fn,
            "loglik_fn": loglik_fn,
            "predict_fn": predict_fn,
            "obs_sigma": noise_std,
            # The runner calls f["oracle_stats_fn"](step_index, rng) to get a
            # per-step closure (particles, X, y) -> (grad_L, Sigma).
            "oracle_stats_fn": self.oracle_stats_fn_for_step,
        }

    # ------------------------------------------------------------------
    def oracle_stats_fn_for_step(self, step_index: int, rng):
        """Return an oracle-statistics closure for one step.

        Usage in the runner::

            f = benchmark.build_functions(seed)   # fixes theta_true
            for step in benchmark.stream(seed):
                osf = f["oracle_stats_fn"](step.step_index, oracle_rng)
                grad_L, Sigma = osf(particles, step.X_train, step.y_train)

        Returns
        -------
        closure : (particles, X, y) -> (grad_L (N,d), Sigma (N,d,d))
        """
        if self.theta_true is None:
            raise RuntimeError(
                "theta_true is not set; call build_functions(seed) or "
                "stream(seed) first.")
        theta_star_t = self.theta_true[step_index]

        def _closure(particles, X, y):
            # X and y are unused: the statistics are re-sampled from the true
            # distribution rather than taken from the realized batch.
            return oracle_grad_stats(
                self.model, particles, theta_star_t, self.noise_std,
                self.oracle_samples, self.batch_size, rng)

        return _closure

    # ------------------------------------------------------------------
    def stream(self, seed: int):
        data = self._generate(seed)
        self.theta_true = data["theta_true"]
        switch_times = set(data["switch_times"])
        regime_ids = data["regime_ids"]

        for t in range(self.T):
            yield StreamStep(
                step_index=t,
                X_train=data["X_train"][t],
                y_train=data["y_train"][t],
                X_test=data["X_test"][t],
                y_test=data["y_test"][t],
                train_indices=np.empty(0, int),
                test_indices=np.empty(0, int),
                test_before_train=True,
                regime_id=int(regime_ids[t]),
                # The selection and reporting windows may overlap in step
                # range without leaking, because they are run on disjoint
                # seeds and each seed generates its own data.
                is_selection_step=(self.select_start <= t < self.select_end),
                is_report_step=(t >= self.eval_start),
                # Train and test are generated independently from theta*_t, so
                # a test block never straddles a switch; the switch step
                # itself is flagged instead, and stays in the aggregate.
                straddles_switch=False,
                is_switch_step=(t in switch_times),
            )
