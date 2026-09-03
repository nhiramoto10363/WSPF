#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark: INSECTS (abrupt, balanced) multiclass classification.

The real-world stream of Souza et al. (2020): about 52,848 samples with 33
real-valued features and 6 classes (three mosquito species by sex), whose
abrupt concept switches are induced by controlled temperature changes.

Protocol. At each step the train block B_t = X[pos:pos+B] is paired with the
test window X[pos+B:pos+B+T] that immediately follows it. The window is
scored with the pre-update particles theta_{t-1}, the model is then updated
on B_t, and pos advances by B (non-overlapping blocks). Because the test
window lies ahead of the train block, evaluation always precedes training.
Standardization is fit on samples up to the end of the selection window.

Selection and reporting. A single real stream leaves no room to separate
seeds, but the split is prequential in time, as for email. The first
select_end_step steps (select_end_step * batch_size samples) form the
selection window; the rest is the reporting window. The default of 1250 steps
(about 20,000 samples) covers the first two switches, leaving the remaining
three in the reporting window - the selection window must contain switches,
or the learning rate is selected on a stationary stretch.

Check the change points against the README shipped with the data: taking the
wrong variant invalidates every switch-aligned analysis.
"""

from __future__ import annotations

import os

import numpy as np

from src.models import (
    MulticlassNeuralNetModel,
    create_mc_grad_fn,
    create_mc_loglik_fn,
    create_mc_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.loaders import InsectsDataLoader
from src.benchmarks.loaders.insects_loader import CHANGE_POINTS_ABRUPT_BALANCED

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CSV = os.path.join(
    _REPO_ROOT, "data", "INSECTS-abrupt_balanced_norm.csv")


class InsectsBenchmark(Benchmark):
    """INSECTS (abrupt, balanced) multiclass benchmark."""

    name = "insects"
    task_type = "classification"
    # Sample indices; overwritten from the loader in __init__.
    switch_points = list(CHANGE_POINTS_ABRUPT_BALANCED)

    def __init__(self, csv_path=_DEFAULT_CSV, hidden_dim=32, batch_size=16,
                 test_size=32, grad_clip_norm=5.0, seed=42,
                 select_end_step=1250, change_points=None):
        self.csv_path = csv_path
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.grad_clip_norm = (None if grad_clip_norm is None
                               else float(grad_clip_norm))
        self.loader_seed = int(seed)
        # End of the selection window, in batch steps.
        self.select_end_step = int(select_end_step)
        # Standardization is fit up to the same point, in samples.
        scale_fit_end = self.select_end_step * self.batch_size

        # Missing data raises here rather than at import time.
        self.loader = InsectsDataLoader(
            csv_path, scale_fit_end=scale_fit_end,
            change_points=change_points, seed=self.loader_seed)
        self.input_dim = self.loader.n_features        # 33
        self.n_classes = self.loader.n_classes         # 6
        self.switch_points = list(self.loader.change_points)

        self.model = MulticlassNeuralNetModel(
            self.input_dim, self.hidden_dim, self.n_classes, activation="tanh")
        self.param_dim = self.model.param_dim

    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        clip = self.grad_clip_norm

        raw_grad = create_mc_grad_fn(model)
        per_sample_grad_fn = create_mc_per_sample_grad_fn(model)
        loglik_fn = create_mc_loglik_fn(model)

        def grad_fn(theta, X, y):
            g = raw_grad(theta, X, y)
            if clip is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                scale = np.minimum(1.0, clip / (norms + 1e-12))
                g = g * scale
            return g

        def predict_fn(particles, X):
            """Per-particle class probabilities; the runner takes the
            weighted average and then the argmax."""
            probs, _, _ = model.forward(particles, X)
            return probs   # (N, B, C)

        return {
            "grad_fn": grad_fn,
            "per_sample_grad_fn": per_sample_grad_fn,
            "loglik_fn": loglik_fn,
            "predict_fn": predict_fn,
        }

    # ------------------------------------------------------------------
    def _regime_of(self, index: int) -> int:
        """Which regime a sample index falls in."""
        r = 0
        for cp in self.switch_points:
            if index >= cp:
                r += 1
        return r

    def stream(self, seed: int):
        X, y = self.loader.X, self.loader.y
        n = len(X)
        B = self.batch_size
        T = self.test_size
        cps = self.switch_points
        pos = 0
        step_index = 0
        while pos + B + T <= n:
            tr = np.arange(pos, pos + B)                 # train block
            te = np.arange(pos + B, pos + B + T)         # look-ahead test window
            te0, te1 = pos + B, pos + B + T

            straddles = any(te0 < cp < te1 for cp in cps)
            regime_id = self._regime_of(te0)

            in_sel = (step_index < self.select_end_step)

            yield StreamStep(
                step_index=step_index,
                X_train=X[tr],
                y_train=y[tr],
                X_test=X[te],
                y_test=y[te],
                train_indices=tr,
                test_indices=te,
                test_before_train=True,
                is_selection_step=in_sel,
                is_report_step=(not in_sel),
                straddles_switch=straddles,
                is_switch_step=straddles,
                regime_id=regime_id,
            )
            pos += B
            step_index += 1
