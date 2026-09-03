#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No leakage in the PCA fit or the stream order.

  - The email PCA learns its components from observations before
    pca_fit_end only. Changing the fit window changes the projection of the
    early samples, which is what shows the fit is restricted to the prefix.
  - The stream is causal: train indices are non-decreasing and blocks do not
    overlap.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_ARFF = os.path.join(os.path.dirname(__file__), "..", "data", "email_data.arff")
_HAS_DATA = os.path.exists(_ARFF)
_skip = pytest.mark.skipif(not _HAS_DATA, reason="email data not available")


@_skip
def test_pca_fit_restricted_to_prefix():
    """Changing the fit window changes the projection of early samples."""
    from src.benchmarks.loaders import EmailDataLoader

    early = EmailDataLoader(_ARFF, n_components=20, seed=42, pca_fit_end=300)
    full = EmailDataLoader(_ARFF, n_components=20, seed=42, pca_fit_end=None)

    # On the same first 300 raw rows, a different fit window gives different
    # components, so the projections must differ by more than sign or scale.
    a = early.X[:300]
    b = full.X[:300]
    assert a.shape == b.shape
    assert not np.allclose(np.abs(a), np.abs(b)), (
        "projection unchanged by the fit window; the PCA may not be "
        "restricted to the prefix")


@_skip
def test_stream_is_causal():
    """Train indices are non-decreasing and the blocks do not overlap."""
    from src.benchmarks.email import EmailBenchmark

    steps = list(EmailBenchmark().stream(0))
    last_end = -1
    for s in steps:
        start = int(s.train_indices.min())
        end = int(s.train_indices.max())
        assert start > last_end, f"step {s.step_index}: blocks overlap or go backwards"
        last_end = end


@_skip
def test_pca_dim_and_shape():
    from src.benchmarks.loaders import EmailDataLoader

    ld = EmailDataLoader(_ARFF, n_components=50, seed=42, pca_fit_end=600)
    assert ld.X.shape[1] == 50
    assert ld.pca_fit_end == 600


if __name__ == "__main__":
    if not _HAS_DATA:
        print("test_no_leakage: skipped, email data not available")
        sys.exit(0)
    test_pca_fit_restricted_to_prefix()
    test_stream_is_causal()
    test_pca_dim_and_shape()
    print("test_no_leakage: all tests passed")
