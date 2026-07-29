#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共通ユーティリティ関数
- 活性化関数
- リサンプリング
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="Numba extension module 'numba_dpex.*' failed to load",
)

import numpy as np
from numba import njit, prange


# ================================================================
# 活性化関数
# ================================================================
def sigmoid(z):
    """安定なシグモイド関数"""
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def softplus(x):
    """log(1+exp(x)) の安定な実装"""
    return np.logaddexp(0.0, x)


def relu(x):
    """ReLU活性化関数"""
    return np.maximum(0.0, x)


def relu_derivative(x):
    """ReLU導関数"""
    return (x > 0).astype(np.float64)


def tanh_derivative(x):
    """tanh導関数: 1 - tanh(x)^2"""
    t = np.tanh(x)
    return 1.0 - t ** 2


# ================================================================
# リサンプリング (njit)
# ================================================================
@njit(cache=True)
def _systematic_resample_njit(weights, u):
    """njit版: 1つの一様乱数 u ∈ [0,1) を受け取る"""
    N = weights.shape[0]
    indices = np.empty(N, dtype=np.int64)
    cumsum = np.cumsum(weights)
    j = 0
    for i in range(N):
        pos = (u + i) / N
        while j < N - 1 and cumsum[j] < pos:
            j += 1
        indices[i] = j
    return indices


def systematic_resample(weights, rng):
    """
    システマティックリサンプリング

    Parameters
    ----------
    weights : ndarray, shape (N,)
        正規化された重み
    rng : numpy.random.Generator
        乱数生成器

    Returns
    -------
    indices : ndarray, shape (N,)
        リサンプリングされたインデックス
    """
    u = rng.random()
    return _systematic_resample_njit(weights, u)


@njit(cache=True)
def normalize_logweights(ll):
    """
    対数重みを正規化 (njit)

    Parameters
    ----------
    ll : ndarray, shape (N,)
        対数尤度（対数重み）

    Returns
    -------
    w : ndarray, shape (N,)
        正規化された重み
    """
    N = ll.shape[0]
    m = ll[0]
    for i in range(1, N):
        if ll[i] > m:
            m = ll[i]

    w = np.empty(N)
    s = 0.0
    for i in range(N):
        w[i] = np.exp(ll[i] - m)
        s += w[i]

    if s == 0.0 or not np.isfinite(s):
        inv_n = 1.0 / N
        for i in range(N):
            w[i] = inv_n
    else:
        for i in range(N):
            w[i] /= s

    return w


@njit(cache=True)
def effective_sample_size(w):
    """
    有効サンプルサイズ (ESS) の計算 (njit)

    Parameters
    ----------
    w : ndarray, shape (N,)
        正規化された重み

    Returns
    -------
    ess : float
        有効サンプルサイズ
    """
    ss = 0.0
    for i in range(w.shape[0]):
        ss += w[i] * w[i]
    return 1.0 / ss


# ================================================================
# 診断量 (R1-8, R1-9, R2-2 の共通ログ)
# ================================================================
def weight_diagnostics(w):
    """
    正規化重みから縮退診断量を計算する。

    Parameters
    ----------
    w : ndarray, shape (N,)
        正規化された重み

    Returns
    -------
    ess : float
        有効サンプルサイズ 1/Σ w_i²
    entropy : float
        Shannon エントロピー −Σ w_i log w_i (0 log 0 = 0 とする)
    max_weight : float
        最大正規化重み max_i w_i
    """
    ess = float(effective_sample_size(w))
    nz = w > 0.0
    entropy = float(-(w[nz] * np.log(w[nz])).sum())
    max_weight = float(w.max())
    return ess, entropy, max_weight


def ensemble_spread_trace(particles):
    """
    パラメータ空間での粒子群の広がり = 標本共分散のトレース
    (= Σ_dim Var(particles[:, dim]))。重みなしのアンサンブル spread。

    Parameters
    ----------
    particles : ndarray, shape (N, d)

    Returns
    -------
    spread_trace : float
    """
    return float(particles.var(axis=0).sum())
