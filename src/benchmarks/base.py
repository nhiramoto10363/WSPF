#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common protocol for the benchmarks (learning tasks).

Each benchmark supplies model-dependent functions via build_functions(seed)
and a data stream via stream(seed), so that evaluation/runner.py can drive
every method through one loop without knowing the task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ======================================================================
# One step of the stream (prequential test-then-train)
# ======================================================================
@dataclass
class StreamStep:
    """Training and evaluation data for one block, plus its metadata.

    Prequential test-then-train: the test block is evaluated with
    theta_{t-1}, and only afterwards is the model updated on the train block,
    so every observation is scored once before it is learned from.

    Attributes
    ----------
    step_index : int
        Zero-based position in the stream.
    X_train, y_train : ndarray
        Mini-batch used for the update (B samples).
    X_test, y_test : ndarray
        Block used for evaluation, before the update.
    train_indices, test_indices : ndarray
        Global indices into the original data, kept for leakage checks and
        reproducibility.
    test_before_train : bool
        True under the prequential protocol.
    is_selection_step : bool
        Whether this step belongs to the selection window. Grid-search scores
        aggregate over this mask only.
    is_report_step : bool
        Whether this step belongs to the reporting window. Final scores
        aggregate over this mask only. The two windows are processed
        continuously (no re-initialization); only the aggregation differs.
    straddles_switch : bool
        Whether the test block *crosses* a concept switch. Such blocks may be
        kept in the overall aggregate but are excluded from the stable and
        post-switch analyses. When the test set is generated independently (as
        in the synthetic regression task) no block ever straddles, and the
        switch itself is flagged by is_switch_step instead.
    is_switch_step : bool
        Whether this step is a known concept switch, the anchor for the
        post-switch recovery analysis. Kept distinct from straddles_switch so
        that switch steps are not dropped from the overall aggregate.
    regime_id : Optional[int]
        Current regime, for benchmarks with switches.
    """

    step_index: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    test_indices: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    test_before_train: bool = True
    is_selection_step: bool = False
    is_report_step: bool = True
    straddles_switch: bool = False
    is_switch_step: bool = False
    regime_id: Optional[int] = None


# ======================================================================
# Base class
# ======================================================================
class Benchmark(ABC):
    """Interface every benchmark implements.

    Subclasses provide:
      - name : str                       "regression" / "email" / "insects"
      - task_type : str                  "regression" | "classification"
      - param_dim : int
      - switch_points : list[int]        known concept switches, or []
      - build_functions(seed) -> dict
      - stream(seed) -> Iterator[StreamStep]

    Keys of the dict returned by build_functions():
      - "grad_fn"            : (theta[N, d], X, y) -> (N, d), the batch-mean
                               gradient; N=1 also serves the baselines.
      - "per_sample_grad_fn" : (particles[N, d], X, y) -> (N, B, d), required
                               by WSPF-A and WSPF-B.
      - "loglik_fn"          : (particles[N, d], X, y) -> (N,), the block
                               log-likelihood (summed over the mini-batch).
      - "predict_fn"         : (particles[N, d], X) -> predictions (function
                               values for regression, logits or probabilities
                               for classification).
      - regression only:
          "obs_sigma"        : observation-noise std, used for prediction
                               intervals y = f + eps.
      - oracle filter (regression only):
          "oracle_grad_fn"      : (theta[N, d], X, y) -> (N, d), the true
                                  population gradient.
          "oracle_noise_cov_fn" : (theta[N, d], X, y, batch_size) -> the true
                                  gradient-noise covariance (scalar or matrix).
    """

    name: str = "base"
    task_type: str = "regression"
    param_dim: int = 0
    switch_points: list = []

    @abstractmethod
    def build_functions(self, seed: int) -> dict:
        """Return the model-dependent functions listed above."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, seed: int):
        """Return an iterator over StreamStep."""
        raise NotImplementedError

    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def near_switch(self, step_index: int, offset: int) -> bool:
        """Whether step_index falls within offset steps after a switch."""
        return any(0 <= step_index - sp <= offset for sp in self.switch_points)
