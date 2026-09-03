# -*- coding: utf-8 -*-
"""Per-regime class distributions.

The class balance is an empirical property of the stream, so it is reported
alongside the results. Regimes are delimited by the concept-drift points. For
email, the drift points reverse the topic labels (the reversed labels are
already stored in the ARFF), so the table makes the swap in class proportions
visible.
"""
from __future__ import annotations

import numpy as np


def regime_bounds(n_samples, change_points):
    """Regime intervals [(start, end), ...] in samples, half-open.

    change_points are sample indices; out-of-range and duplicate values are
    ignored.
    """
    n = int(n_samples)
    cps = sorted({int(c) for c in change_points if 0 < int(c) < n})
    edges = [0] + cps + [n]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _region_of(start, end, report_start):
    """Whether a regime lies in the selection or the reporting window.

    report_start is a sample index; an interval spanning it is "mixed".
    """
    if report_start is None:
        return ""
    r = int(report_start)
    if end <= r:
        return "selection"
    if start >= r:
        return "report"
    return "mixed"


def class_distribution_rows(y, change_points, n_classes, benchmark="",
                            report_start=None, class_names=None):
    """Return one row per regime with its class distribution.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Integer class labels (0/1 floats are accepted for email).
    change_points : sequence of int
        Concept-drift points, as sample indices.
    n_classes : int
    benchmark : str
        Identifier written into each row.
    report_start : int or None
        First sample of the reporting window, used to fill the region column.
    class_names : sequence of str or None
        Defaults to the class indices.

    Returns
    -------
    rows : list[dict]
        One row per regime plus a final row with regime="all". Each row
        carries class_{c}_count and class_{c}_frac.
    """
    y = np.asarray(y).ravel().astype(np.int64)
    C = int(n_classes)
    n = y.size
    names = list(class_names) if class_names is not None \
        else [str(c) for c in range(C)]

    def _row(label, s, e):
        seg = y[s:e]
        cnt = np.bincount(seg, minlength=C)[:C]
        total = max(1, seg.size)
        row = {
            "benchmark": benchmark,
            "regime": label,
            "start": int(s),
            "end": int(e),
            "n_samples": int(seg.size),
            "region": _region_of(s, e, report_start),
        }
        for c in range(C):
            row[f"class_{c}_count"] = int(cnt[c])
            row[f"class_{c}_frac"] = float(cnt[c] / total)
        return row

    rows = [_row(str(i + 1), s, e)
            for i, (s, e) in enumerate(regime_bounds(n, change_points))]
    rows.append(_row("all", 0, n))
    # Class names go on the first row only, so callers can use them in a
    # header without adding a column to every row.
    if rows:
        rows[0]["class_names"] = "|".join(names)
    return rows


def format_class_distribution(rows, n_classes, emit=print):
    """Print the rows from class_distribution_rows as a readable table."""
    C = int(n_classes)
    header = ("  regime  range                  n   region     "
              + "".join(f"{c:>9d}" for c in range(C)))
    emit(header)
    for r in rows:
        emit(f"  {r['regime']:>6s}  [{r['start']:>6d},{r['end']:>6d}) "
             f"{r['n_samples']:>6d}   {r['region']:<9s} "
             + "".join(f"{r[f'class_{c}_frac']:9.3f}" for c in range(C)))
