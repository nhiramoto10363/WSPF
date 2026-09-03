#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run artifacts.

Each run (method x seed x condition) writes a directory containing:
  - config.json          the run settings (method, benchmark, N, params, seed)
  - selected_params.json the selected hyper-parameters
  - metrics.csv          per-step or summary metric rows
  - diagnostics.npz      diagnostic arrays taken from the filter history
  - environment.txt      Python, numpy, scipy and numba versions
  - git_commit.txt       the commit the run came from
  - data_indices.npz     global train/test indices, for leakage checks

A general table writer (csv/txt/tex) lives here as well.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys

import numpy as np


def sanitize(method):
    """Normalize a method name into a safe file or directory name."""
    return method.replace(" ", "_").replace("-", "_").lower()


def _git_commit():
    """Current git commit hash, or 'unknown' when it cannot be read."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=True, text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _environment_text():
    """Render the environment summary."""
    lines = [
        f"python: {sys.version.splitlines()[0]}",
        f"platform: {platform.platform()}",
        f"executable: {sys.executable}",
    ]
    for pkg in ("numpy", "scipy", "numba"):
        try:
            mod = __import__(pkg)
            lines.append(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
        except Exception:
            lines.append(f"{pkg}: not-installed")
    return "\n".join(lines) + "\n"


def _json_default(o):
    """Make numpy types JSON-serializable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(obj, path):
    """Write a dict as UTF-8 JSON, handling numpy types."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _rows_to_matrix(rows):
    """Turn rows (list[dict] or dict[str, array-like]) into (header, data)."""
    if isinstance(rows, dict):
        keys = list(rows.keys())
        cols = [np.asarray(rows[k]).ravel() for k in keys]
        n = max((c.size for c in cols), default=0)
        data = []
        for i in range(n):
            data.append([cols[j][i] if i < cols[j].size else "" for j in range(len(keys))])
        return keys, data
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    data = [[r.get(k, "") for k in keys] for r in rows]
    return keys, data


def write_table(rows, path, formats=("csv", "txt", "tex")):
    """Write a table in one or more formats.

    Parameters
    ----------
    rows : list[dict] | dict[str, array-like]
    path : str
        Base path without an extension; each format appends its own.
    formats : tuple[str]
        Any combination of "csv", "txt" and "tex".
    """
    keys, data = _rows_to_matrix(rows)
    base = os.path.splitext(path)[0]

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, (np.floating,)):
            return f"{float(v):.6g}"
        return str(v)

    written = []
    if "csv" in formats:
        p = base + ".csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for row in data:
                w.writerow([_fmt(v) for v in row])
        written.append(p)
    if "txt" in formats:
        p = base + ".txt"
        widths = [max(len(str(keys[j])),
                      max((len(_fmt(row[j])) for row in data), default=0))
                  for j in range(len(keys))]
        with open(p, "w", encoding="utf-8") as f:
            f.write("  ".join(str(keys[j]).ljust(widths[j])
                              for j in range(len(keys))) + "\n")
            for row in data:
                f.write("  ".join(_fmt(row[j]).ljust(widths[j])
                                  for j in range(len(keys))) + "\n")
        written.append(p)
    if "tex" in formats:
        p = base + ".tex"
        with open(p, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{" + "l" * len(keys) + "}\n")
            f.write("\\hline\n")
            f.write(" & ".join(str(k).replace("_", "\\_") for k in keys) + " \\\\\n")
            f.write("\\hline\n")
            for row in data:
                f.write(" & ".join(_fmt(v).replace("_", "\\_") for v in row) + " \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
        written.append(p)
    return written


def save_run_dir(out_dir, config, selected_params, metrics_rows,
                 diagnostics, data_indices=None, extra=None):
    """Write the full artifact set for one run into out_dir.

    Parameters
    ----------
    out_dir : str
        Created if absent.
    config : dict
        Run settings (method, benchmark, n_particles, params, seed).
    selected_params : dict
    metrics_rows : list[dict] | dict[str, array-like]
        Rows for metrics.csv.
    diagnostics : dict[str, array-like] | None
        Arrays for diagnostics.npz, typically filter.get_history().
    data_indices : dict[str, array-like] | None
        Global train/test indices, saved as data_indices.npz.
    extra : dict | None
        Anything else worth keeping, saved as extra.json.

    Returns
    -------
    out_dir : str
    """
    os.makedirs(out_dir, exist_ok=True)

    write_json(config, os.path.join(out_dir, "config.json"))
    write_json(selected_params or {}, os.path.join(out_dir, "selected_params.json"))

    if metrics_rows is not None:
        write_table(metrics_rows, os.path.join(out_dir, "metrics"),
                    formats=("csv",))

    if diagnostics is not None:
        arrs = {}
        for k, v in diagnostics.items():
            try:
                arrs[k] = np.asarray(v)
            except Exception:
                continue
        np.savez(os.path.join(out_dir, "diagnostics.npz"), **arrs)

    if data_indices is not None:
        arrs = {k: np.asarray(v) for k, v in data_indices.items()}
        np.savez(os.path.join(out_dir, "data_indices.npz"), **arrs)

    with open(os.path.join(out_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(_environment_text())

    with open(os.path.join(out_dir, "git_commit.txt"), "w", encoding="utf-8") as f:
        f.write(_git_commit() + "\n")

    if extra is not None:
        write_json(extra, os.path.join(out_dir, "extra.json"))

    return out_dir
