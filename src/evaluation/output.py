#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実行成果物の出力(output) — per-run アーティファクト一式

修正方針が要求する再現性のため、1 回の実行(method × seed × 条件)ごとに
以下を run ディレクトリへ保存する:
  - config.json         実行設定(method, benchmark, n_particles, params, seed…)
  - selected_params.json グリッドサーチ等で選ばれたハイパーパラメータ
  - metrics.csv         per-step あるいは要約の指標行
  - diagnostics.npz     診断量(history 由来の配列)
  - environment.txt     Python / numpy / scipy などの環境情報
  - git_commit.txt      再現用の git コミットハッシュ
  - data_indices.npz    使用した train/test のグローバルインデックス(リーク検査)

汎用のテーブル書き出し write_table(csv/txt/tex) も提供する。
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
    """メソッド名をファイル/ディレクトリ名に安全な形へ正規化する。"""
    return method.replace(" ", "_").replace("-", "_").lower()


def _git_commit():
    """現在の git コミットハッシュを取得(失敗時は 'unknown')。"""
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
    """環境情報テキストを生成する。"""
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
    """numpy 型を JSON シリアライズ可能に。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(obj, path):
    """dict を UTF-8 JSON で保存(numpy 型対応)。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _rows_to_matrix(rows):
    """
    行(list[dict] または dict[str, array-like])を (header, data_rows) に整形。
    """
    if isinstance(rows, dict):
        keys = list(rows.keys())
        cols = [np.asarray(rows[k]).ravel() for k in keys]
        n = max((c.size for c in cols), default=0)
        data = []
        for i in range(n):
            data.append([cols[j][i] if i < cols[j].size else "" for j in range(len(keys))])
        return keys, data
    # list[dict]
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    data = [[r.get(k, "") for k in keys] for r in rows]
    return keys, data


def write_table(rows, path, formats=("csv", "txt", "tex")):
    """
    汎用テーブル書き出し。path は拡張子なしのベースパス。

    Parameters
    ----------
    rows : list[dict] | dict[str, array-like]
    path : str
        拡張子なしのベースパス(各 format の拡張子が付与される)。
    formats : tuple[str]
        "csv" / "txt" / "tex" の任意組合せ。
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
    """
    1 回の実行成果物一式を out_dir に保存する。

    Parameters
    ----------
    out_dir : str
        出力ディレクトリ(作成する)。
    config : dict
        実行設定(method, benchmark, n_particles, params, seed など)。
    selected_params : dict
        選択されたハイパーパラメータ。
    metrics_rows : list[dict] | dict[str, array-like]
        metrics.csv に書き出す行。
    diagnostics : dict[str, array-like] | None
        diagnostics.npz に保存する配列群(filter.get_history() など)。
    data_indices : dict[str, array-like] | None
        train/test のグローバルインデックス。data_indices.npz に保存。
    extra : dict | None
        追加で JSON 保存したいもの(extra.json)。

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
