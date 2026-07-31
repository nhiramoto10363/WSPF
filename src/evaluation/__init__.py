#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation 層 — run-loop / metric / diagnostic / statistics / output の集約。

修正方針: 実験ファイルに散っていた評価ロジックを 5 モジュールに統合し、
全ベンチマークを **唯一の共有ランナー** run_method / run_seeds で駆動する。
"""

from __future__ import annotations

from . import metrics, diagnostics, statistics, output

from .runner import (
    run_method,
    run_seeds,
    resolve_workers,
    _init_worker,
    ALL_METHODS,
    SGD_METHODS,
    FILTER_METHODS,
    SEED_OFFSET,
)

from .metrics import (
    test_mse, test_mae, test_r2, nll_gaussian, crps_gaussian,
    coverage_and_width, weighted_prediction, prediction_std_with_noise,
    accuracy, f1, balanced_accuracy, precision, recall,
    nll_bernoulli, brier_ece,
)

from .diagnostics import summarize_history, rho_report, timing_report

from .statistics import (
    paired_t, wilcoxon_signed, mean_std, recovery_curve, paired_compare,
)

from .output import sanitize, save_run_dir, write_table, write_json

__all__ = [
    "metrics", "diagnostics", "statistics", "output",
    "run_method", "run_seeds",
    "ALL_METHODS", "SGD_METHODS", "FILTER_METHODS", "SEED_OFFSET",
    "test_mse", "test_mae", "test_r2", "nll_gaussian", "crps_gaussian",
    "coverage_and_width", "weighted_prediction", "prediction_std_with_noise",
    "accuracy", "f1", "balanced_accuracy", "precision", "recall",
    "nll_bernoulli", "brier_ece",
    "summarize_history", "rho_report", "timing_report",
    "paired_t", "wilcoxon_signed", "mean_std", "recovery_curve",
    "paired_compare",
    "sanitize", "save_run_dir", "write_table", "write_json",
]
