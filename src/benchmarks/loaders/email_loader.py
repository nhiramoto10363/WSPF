#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loader for the Email (elist) dataset of Katakis et al. (2010).

1,500 samples with 913 binary bag-of-words features (sparse ARFF), labelled
interesting (1) or junk (0). Five periods of 300 samples give four concept
switches at 300, 600, 900 and 1200, alternating between two recurring
contexts. Optional PCA (truncated SVD) reduces the dimension.
"""

import numpy as np


class EmailDataLoader:
    """Parses the sparse ARFF, optionally applies PCA, and streams batches."""

    def __init__(self, arff_path, n_components=None, seed=42, pca_fit_end=None):
        """
        Parameters
        ----------
        arff_path : str
        n_components : int or None
            PCA dimension; None keeps the original 913 features.
        seed : int
        pca_fit_end : int or None
            Fit the PCA (centering mean and components) on samples
            [0, pca_fit_end) only, then transform the whole stream. None fits
            on the whole stream, which leaks the evaluation window into the
            feature extraction; pass the end of the warm-up window (e.g. 600).
        """
        self.seed = seed
        self.pca_fit_end = pca_fit_end

        n_features, rows = _parse_sparse_arff(arff_path)

        n_samples = len(rows)
        X = np.zeros((n_samples, n_features), dtype=np.float64)
        y = np.zeros(n_samples, dtype=np.float64)

        label_idx = n_features  # the label is the last attribute

        for i, sparse_dict in enumerate(rows):
            for idx, val in sparse_dict.items():
                if idx == label_idx:
                    y[i] = val
                else:
                    X[i, idx] = val

        if n_components is not None and n_components < n_features:
            X = self._apply_pca(X, n_components, pca_fit_end)

        self.X = X
        self.y = y
        self._input_dim = self.X.shape[1]
        self._n_samples = len(self.X)

    @staticmethod
    def _apply_pca(X, n_components, fit_end=None):
        """Center and project with a truncated SVD.

        With fit_end set, the mean and the components are learned from samples
        [0, fit_end) only and then applied to the whole stream, so the
        evaluation window never reaches the feature extraction.
        """
        X_fit = X if fit_end is None else X[:fit_end]
        mean = X_fit.mean(axis=0)
        X_fit_centered = X_fit - mean
        U, S, Vt = np.linalg.svd(X_fit_centered, full_matrices=False)
        components = Vt[:n_components]  # (n_components, d)
        return (X - mean) @ components.T

    @property
    def input_dim(self):
        """Feature dimension after any PCA."""
        return self._input_dim

    @property
    def n_samples(self):
        return self._n_samples

    def get_stream(self, batch_size):
        """Yield consecutive (X_batch, y_batch) pairs in stream order."""
        n = self._n_samples
        for start in range(0, n - batch_size + 1, batch_size):
            end = start + batch_size
            yield self.X[start:end], self.y[start:end]


def _parse_sparse_arff(path):
    """Parse a sparse ARFF file.

    Returns
    -------
    n_features : int
        Number of features, excluding the label attribute.
    rows : list of dict
        One {attribute index: value} dict per sample.
    """
    n_attributes = 0
    rows = []
    in_data = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if not stripped or stripped.startswith("%"):
                continue

            lower = stripped.lower()

            if lower.startswith("@attribute"):
                n_attributes += 1
                continue

            if lower.startswith("@data"):
                in_data = True
                continue

            if lower.startswith("@"):
                continue

            if not in_data:
                continue

            # Sparse row: {idx val, idx val, ...}
            if stripped.startswith("{") and stripped.endswith("}"):
                inner = stripped[1:-1].strip()
                sparse_dict = {}
                if inner:
                    for token in inner.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        parts = token.split()
                        idx = int(parts[0])
                        val = float(parts[1])
                        sparse_dict[idx] = val
                rows.append(sparse_dict)
            else:
                # Dense row, as a fallback
                vals = stripped.split(",")
                sparse_dict = {}
                for idx, v in enumerate(vals):
                    v = v.strip()
                    if v == "?" or v == "":
                        continue
                    sparse_dict[idx] = float(v)
                rows.append(sparse_dict)

    # The last attribute is the label.
    n_features = n_attributes - 1

    return n_features, rows
