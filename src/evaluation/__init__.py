#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation layer: run loop, metrics, diagnostics, statistics, output.

Every benchmark is driven through the single shared runner, run_method /
run_seeds, so that the aggregation conventions cannot drift between tasks.
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
    test_mse, test_mae, test_r2, nll_gaussian, nll_gaussian_mixture,
    crps_gaussian,
    coverage_and_width, weighted_prediction, prediction_std_with_noise,
    particle_predictions, weighted_moments,
    accuracy, f1, balanced_accuracy, precision, recall,
    nll_bernoulli, brier_ece, brier_ece_multiclass, brier_multiclass,
    macro_f1, nll_categorical,
)

from .diagnostics import summarize_history, rho_report, timing_report

from .applicability import (
    diagnose as diagnose_applicability, correction_spread_theory,
    required_particles, max_supported_rho,
)

from .statistics import (
    paired_t, wilcoxon_signed, mean_std, recovery_curve, paired_compare,
    holm_adjust, all_pairs_compare,
)

from .class_distribution import (
    regime_bounds, class_distribution_rows, format_class_distribution,
)

from .output import sanitize, save_run_dir, write_table, write_json

__all__ = [
    "metrics", "diagnostics", "statistics", "output", "applicability",
    "run_method", "run_seeds",
    "ALL_METHODS", "SGD_METHODS", "FILTER_METHODS", "SEED_OFFSET",
    "test_mse", "test_mae", "test_r2", "nll_gaussian", "nll_gaussian_mixture",
    "crps_gaussian",
    "coverage_and_width", "weighted_prediction", "prediction_std_with_noise",
    "particle_predictions", "weighted_moments",
    "accuracy", "f1", "balanced_accuracy", "precision", "recall",
    "nll_bernoulli", "brier_ece", "brier_ece_multiclass", "brier_multiclass",
    "macro_f1", "nll_categorical",
    "summarize_history", "rho_report", "timing_report",
    "diagnose_applicability", "correction_spread_theory",
    "required_particles", "max_supported_rho",
    "paired_t", "wilcoxon_signed", "mean_std", "recovery_curve",
    "paired_compare", "holm_adjust", "all_pairs_compare",
    "regime_bounds", "class_distribution_rows", "format_class_distribution",
    "sanitize", "save_run_dir", "write_table", "write_json",
]
