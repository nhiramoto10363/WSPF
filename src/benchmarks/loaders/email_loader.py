#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email (elist) データセットのローダー

Katakis et al. (2010) の Email データセットを読み込む。
- 1,500 サンプル, 913 次元 binary bag-of-words (sparse ARFF)
- 二値分類: interesting (1) / junk (0)
- 5 期間 × 300 サンプル, 4 コンセプトドリフト (index 300, 600, 900, 1200)
- Recurring contexts: Concept 0 (Medicine=+) ↔ Concept 1 (Space+Baseball=+)

PCA/SVD による次元削減オプション付き。
"""

import numpy as np


class EmailDataLoader:
    """
    Email データセットのローダー

    sparse ARFF を手動パースし、PCA (SVD) による次元削減を適用して
    バッチストリーミングを提供する。
    """

    def __init__(self, arff_path, n_components=None, seed=42, pca_fit_end=None):
        """
        Parameters
        ----------
        arff_path : str
            sparse ARFF ファイルのパス
        n_components : int or None
            PCA 次元数。None なら元の 913 次元をそのまま使う。
        seed : int
            乱数シード（将来の拡張用）
        pca_fit_end : int or None
            PCA(中心化平均 + 主成分方向)を学習するサンプル区間の終端
            (先頭 [0, pca_fit_end) のみで fit)。None なら全ストリームで
            学習する（リークあり・後方互換用）。リーク除去実験では
            warm-up 区間の終端(例: 600)を渡すこと。
        """
        self.seed = seed
        self.pca_fit_end = pca_fit_end

        n_features, rows = _parse_sparse_arff(arff_path)

        # 密行列に展開
        n_samples = len(rows)
        X = np.zeros((n_samples, n_features), dtype=np.float64)
        y = np.zeros(n_samples, dtype=np.float64)

        label_idx = n_features  # ラベルは最後の属性 (0-indexed)

        for i, sparse_dict in enumerate(rows):
            for idx, val in sparse_dict.items():
                if idx == label_idx:
                    y[i] = val
                else:
                    X[i, idx] = val

        # PCA/SVD による次元削減
        if n_components is not None and n_components < n_features:
            X = self._apply_pca(X, n_components, pca_fit_end)

        self.X = X
        self.y = y
        self._input_dim = self.X.shape[1]
        self._n_samples = len(self.X)

    @staticmethod
    def _apply_pca(X, n_components, fit_end=None):
        """中心化 + truncated SVD で次元削減する。

        fit_end を指定すると先頭 [0, fit_end) のサンプルのみで中心化平均と
        主成分方向を学習し(fit)、その変換を全ストリームに適用する(transform)。
        これにより評価区間のデータが特徴抽出にリークするのを防ぐ。
        fit_end=None のときは全ストリームで学習する(後方互換・リークあり)。
        """
        X_fit = X if fit_end is None else X[:fit_end]
        mean = X_fit.mean(axis=0)
        X_fit_centered = X_fit - mean
        # economy SVD で主成分方向 Vt を学習区間からのみ求める
        U, S, Vt = np.linalg.svd(X_fit_centered, full_matrices=False)
        components = Vt[:n_components]  # (n_components, d)
        # 学習した mean・成分で全ストリームを射影(transform)
        return (X - mean) @ components.T

    @property
    def input_dim(self):
        """特徴量の次元数（PCA 適用後）"""
        return self._input_dim

    @property
    def n_samples(self):
        """総サンプル数"""
        return self._n_samples

    def get_stream(self, batch_size):
        """
        バッチストリーミングのジェネレータ（時系列順）

        Yields
        ------
        X_batch : ndarray, shape (batch_size, input_dim)
        y_batch : ndarray, shape (batch_size,)
        """
        n = self._n_samples
        for start in range(0, n - batch_size + 1, batch_size):
            end = start + batch_size
            yield self.X[start:end], self.y[start:end]


def _parse_sparse_arff(path):
    """
    sparse ARFF ファイルをパースする。

    Returns
    -------
    n_features : int
        特徴量の数（ラベル属性を除く）
    rows : list of dict
        各サンプルの {属性インデックス: 値} 辞書のリスト
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

            # sparse ARFF 行: {idx val, idx val, ...}
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
                # dense ARFF 行 (フォールバック)
                vals = stripped.split(",")
                sparse_dict = {}
                for idx, v in enumerate(vals):
                    v = v.strip()
                    if v == "?" or v == "":
                        continue
                    sparse_dict[idx] = float(v)
                rows.append(sparse_dict)

    # 最後の属性がラベルなので、特徴量数 = 全属性数 - 1
    n_features = n_attributes - 1

    return n_features, rows
