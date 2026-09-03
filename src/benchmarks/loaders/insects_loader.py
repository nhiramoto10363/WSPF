#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loader for the INSECTS dataset (Souza et al., 2020).

Expects the abrupt (balanced) variant from the USP DS repository: 33
real-valued features and 6 classes (three mosquito species by sex), a
real-world stream whose concept switches are induced by controlled
temperature changes.

Accepted formats:
  - dense ARFF (33 numeric attributes plus a nominal class, CSV after @data)
  - headerless CSV (33 numeric columns plus a label column)

Standardization (per-feature mean and std) is fit on samples
[0, scale_fit_end) only, so the reporting window never reaches the
preprocessing.
"""

import os
import numpy as np

# Change points (sample indices) of the abrupt (balanced) variant, from
# Souza et al. (2020). Always check these against the README shipped with the
# data: taking the wrong variant invalidates every switch-aligned analysis.
CHANGE_POINTS_ABRUPT_BALANCED = [14352, 19500, 33240, 38682, 39510]


def _parse_dense_arff(path):
    """Parse a dense ARFF into (X_rows, y_labels, class_names)."""
    X_rows, y_labels = [], []
    class_names = None
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if not in_data:
                if low.startswith("@attribute") and "{" in line:
                    # class attribute: @attribute class {a,b,...}
                    inner = line[line.index("{") + 1: line.rindex("}")]
                    class_names = [c.strip().strip("'\"")
                                   for c in inner.split(",")]
                elif low.startswith("@data"):
                    in_data = True
                continue
            parts = [p.strip() for p in line.split(",")]
            X_rows.append([float(v) for v in parts[:-1]])
            y_labels.append(parts[-1].strip().strip("'\""))
    return X_rows, y_labels, class_names


def _parse_csv(path):
    X_rows, y_labels = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # Skip a header row, detected by the float conversion failing.
            try:
                X_rows.append([float(v) for v in parts[:-1]])
            except ValueError:
                continue
            y_labels.append(parts[-1].strip().strip("'\""))
    return X_rows, y_labels, None


class InsectsDataLoader:
    """
    Attributes
    ----------
    X : ndarray, shape (n_samples, 33)   standardized features
    y : ndarray, shape (n_samples,)      integer class labels in 0..C-1
    n_samples, n_features, n_classes : int
    class_names : list of str
    change_points : list of int
    scale_fit_end : int or None
    """

    def __init__(self, path, scale_fit_end=None,
                 change_points=None, seed=42):
        self.seed = seed
        self.scale_fit_end = scale_fit_end
        self.change_points = (list(change_points) if change_points is not None
                              else list(CHANGE_POINTS_ABRUPT_BALANCED))

        ext = os.path.splitext(path)[1].lower()
        if ext == ".arff":
            X_rows, y_labels, class_names = _parse_dense_arff(path)
        else:
            X_rows, y_labels, class_names = _parse_csv(path)

        X = np.asarray(X_rows, dtype=np.float64)
        if class_names is None:
            class_names = sorted(set(y_labels))
        name_to_idx = {c: i for i, c in enumerate(class_names)}
        y = np.asarray([name_to_idx[c] for c in y_labels], dtype=np.int64)

        self.class_names = class_names
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(class_names)

        # Leak-free standardization: fit on the initial segment only.
        fit_end = (self.n_samples if scale_fit_end is None
                   else int(scale_fit_end))
        mu = X[:fit_end].mean(axis=0)
        sd = X[:fit_end].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        self.scale_mean_, self.scale_std_ = mu, sd
        self.X = (X - mu) / sd
        self.y = y

        for cp in self.change_points:
            if not (0 < cp < self.n_samples):
                raise ValueError(
                    f"change point {cp} is outside the stream "
                    f"(n_samples={self.n_samples}). "
                    "Check the variant and CHANGE_POINTS.")

    def regime_bounds(self):
        """Regime intervals as a list of (start, end)."""
        edges = [0] + list(self.change_points) + [self.n_samples]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def print_regime_class_distribution(self, emit=print):
        """Print the class distribution within each regime."""
        emit(f"  INSECTS stream: n={self.n_samples}, "
             f"features={self.n_features}, classes={self.n_classes}")
        emit(f"  change points: {self.change_points}")
        emit(f"  classes: {self.class_names}")
        header = "  regime      range              " + "".join(
            f"{i:>8d}" for i in range(self.n_classes))
        emit(header)
        for r, (s, e) in enumerate(self.regime_bounds()):
            cnt = np.bincount(self.y[s:e], minlength=self.n_classes)
            frac = cnt / max(1, (e - s))
            emit(f"  {r + 1:>2d}   [{s:>6d},{e:>6d})  " +
                 "".join(f"{f:8.3f}" for f in frac))
