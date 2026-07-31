#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSECTS データセット (Souza et al., 2020) のローダー

USP DS repository の abrupt (balanced) バリアントを想定。
33 実数値特徴 + 6 クラス(蚊 3 種 × 雌雄)。温度制御による既知の
概念スイッチ点を持つ実世界ストリーム。

対応フォーマット:
- 密 ARFF (@attribute 数値 ×33 + class nominal, @data 以降 CSV)
- ヘッダなし CSV (33 数値列 + 最終列ラベル)

リーク除去 (R1-13 と同方針):
- 標準化 (per-feature mean/std) は先頭 [0, scale_fit_end) のみで fit。

診断 (R1-minor と同方針):
- print_regime_class_distribution() で各レジームのクラス分布を出力。
"""

import os
import numpy as np

# --- 既知のスイッチ点 (サンプルインデックス) ---
# Souza et al. (2020) "Challenges in Benchmarking Stream Learning
# Algorithms with Real-world Data" の abrupt (balanced) バリアント。
# ★実験前にデータ添付の README / 論文 Table と必ず照合すること★
# (バリアント取り違えは switch-aligned 解析を全て無効にする)
CHANGE_POINTS_ABRUPT_BALANCED = [14352, 19500, 33240, 38682, 39510]


def _parse_dense_arff(path):
    """密 ARFF をパースし (X_rows, y_labels, class_names) を返す"""
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
                    # class 属性: @attribute class {a,b,...}
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
            # 先頭行がヘッダの可能性: 数値変換に失敗したら読み飛ばす
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
    X : ndarray, shape (n_samples, 33)   標準化済み特徴
    y : ndarray, shape (n_samples,)      整数クラスラベル (0..C-1)
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

        # --- 標準化 (リークフリー: 先頭区間のみで fit) ---
        fit_end = (self.n_samples if scale_fit_end is None
                   else int(scale_fit_end))
        mu = X[:fit_end].mean(axis=0)
        sd = X[:fit_end].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        self.scale_mean_, self.scale_std_ = mu, sd
        self.X = (X - mu) / sd
        self.y = y

        # スイッチ点の妥当性チェック
        for cp in self.change_points:
            if not (0 < cp < self.n_samples):
                raise ValueError(
                    f"change point {cp} is outside the stream "
                    f"(n_samples={self.n_samples}). "
                    "バリアント/CHANGE_POINTS を確認してください。")

    def regime_bounds(self):
        """[(start, end), ...] のレジーム区間リスト"""
        edges = [0] + list(self.change_points) + [self.n_samples]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def print_regime_class_distribution(self, emit=print):
        """各レジームのクラス分布を出力 (R1-minor 対応の多クラス版)"""
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
