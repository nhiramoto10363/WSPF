#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark: Email (elist) binary classification.

The dataset of Katakis et al. (2010): 1,500 samples in five periods of 300,
with concept switches at 300, 600, 900 and 1200, labelled interesting or junk.

Protocol. Prequential test-then-train on the *same* block: at each step the
block B_t = [pos:pos+B] is first scored with the pre-update estimate
theta_{t-1}, then learned from, and the position advances by B so that test
blocks never overlap. Every observation is therefore evaluated exactly once
before it is trained on. (Note that this does mean the same block is both
scored and learned from, in that order.)

A block that crosses a drift point is flagged with straddles_switch, and the
aggregation drops it from every reported metric and switch-aligned analysis.
regime_id is decided from the first sample in the block.

The PCA components are fit on samples before pca_fit_end (600 by default), so
the reporting window never reaches the feature extraction.
"""

from __future__ import annotations

import os

import numpy as np

from src.models import (
    NeuralNetModel,
    create_nn_grad_fn,
    create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)
from src.benchmarks.base import Benchmark, StreamStep
from src.benchmarks.loaders import EmailDataLoader

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_ARFF = os.path.join(_REPO_ROOT, "data", "email_data.arff")

DRIFT_POINTS = [300, 600, 900, 1200]


class EmailBenchmark(Benchmark):
    """Email binary-classification benchmark."""

    name = "email"
    task_type = "classification"
    switch_points = [300, 600, 900, 1200]

    def __init__(self, arff_path=_DEFAULT_ARFF, pca_dim=50, pca_fit_end=600,
                 hidden_dim=16, batch_size=16, test_size=32,
                 grad_clip_norm=5.0, seed=42, report_start=600):
        self.arff_path = arff_path
        self.pca_dim = int(pca_dim)
        self.pca_fit_end = int(pca_fit_end)
        self.hidden_dim = int(hidden_dim)
        self.batch_size = int(batch_size)
        self.test_size = int(test_size)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.loader_seed = int(seed)
        # Start of the reporting window; earlier steps form the selection
        # window and are excluded from the final aggregates.
        self.report_start = int(report_start)

        # Missing data raises here rather than at import time.
        self.loader = EmailDataLoader(
            arff_path, n_components=self.pca_dim, seed=self.loader_seed,
            pca_fit_end=self.pca_fit_end)
        self.input_dim = self.loader.input_dim

        self.model = NeuralNetModel(self.input_dim, self.hidden_dim,
                                    output_dim=1, activation="tanh")
        self.param_dim = self.model.param_dim

    # ------------------------------------------------------------------
    def build_functions(self, seed: int) -> dict:
        model = self.model
        clip = self.grad_clip_norm

        raw_grad = create_nn_grad_fn(model)
        per_sample_grad_fn = create_nn_per_sample_grad_fn(model)
        loglik_fn = create_nn_loglik_fn(model)

        def grad_fn(theta, X, y):
            g = raw_grad(theta, X, y)
            if clip is not None:
                norms = np.linalg.norm(g, axis=1, keepdims=True)
                scale = np.minimum(1.0, clip / (norms + 1e-12))
                g = g * scale
            return g

        def predict_fn(particles, X):
            """Class probabilities from the sigmoid output."""
            output, _, _ = model.forward(particles, X)
            return output.squeeze(-1)   # (N, B)

        return {
            "grad_fn": grad_fn,
            "per_sample_grad_fn": per_sample_grad_fn,
            "loglik_fn": loglik_fn,
            "predict_fn": predict_fn,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _period_of(index: int) -> int:
        """Which of the five periods (0..4) a sample index falls in."""
        p = 0
        for d in DRIFT_POINTS:
            if index >= d:
                p += 1
        return p

    def stream(self, seed: int):
        X, y = self.loader.X, self.loader.y
        n = len(X)
        B = self.batch_size
        pos = 0
        step_index = 0
        while pos + B <= n:
            blk = np.arange(pos, pos + B)     # non-overlapping block B_t

            straddles = any(pos < d < pos + B for d in DRIFT_POINTS)
            regime_id = self._period_of(pos)

            in_rep = (pos >= self.report_start)

            # Score with theta_{t-1}, then train on the same block.
            yield StreamStep(
                step_index=step_index,
                X_train=X[blk],
                y_train=y[blk],
                X_test=X[blk],
                y_test=y[blk],
                train_indices=blk,
                test_indices=blk,
                test_before_train=True,
                is_selection_step=(not in_rep),
                is_report_step=in_rep,
                straddles_switch=straddles,
                is_switch_step=straddles,
                regime_id=regime_id,
            )
            pos += B
            step_index += 1
