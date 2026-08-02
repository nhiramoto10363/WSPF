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


# ================================================================
# φ_t 拡張: 粒子別観測ノイズ (log 分散パラメータ) の共通ヘルパ
# ================================================================
def init_phi(phi_rng, n_particles, phi_init_mean, phi_init_std):
    """φ_0^(i) ~ N(phi_init_mean, phi_init_std²) の初期化。

    phi_init_mean は通常 2 log σ_ref (σ_ref: 観測ノイズの参照スケール)。
    """
    return phi_rng.normal(phi_init_mean, phi_init_std, size=n_particles)


def propagate_phi(phi, tau_phi, phi_rng):
    """φ_t = φ_{t−1} + ν_t, ν_t ~ N(0, τ²) のランダムウォーク遷移。

    観測非依存の事前遷移からのブートストラップ提案なので、
    prior–proposal 補正 (log R̂) には一切寄与しない (設計書 §0.3)。
    τ=0 でも rng を消費する点に注意 (φ 専用 rng なので θ 側とは干渉しない)。
    """
    return phi + phi_rng.normal(0.0, tau_phi, size=phi.shape)


def phi_to_sigma(phi):
    """σ^(i) = exp(φ^(i)/2)。"""
    return np.exp(0.5 * phi)


def weighted_sigma_mean(weights, phi):
    """重み付き平均 σ̂ = Σ_i w_i exp(φ_i/2) (診断: σ̂_t 系列)。"""
    return float(np.sum(weights * phi_to_sigma(phi)))


# ================================================================
# 層化学習率 (Stratified Learning-Rate WSPF, 設計書 §3/§8.1)
# ================================================================
def make_stratified_learning_rates(n_particles, eta_mean, seed=None):
    """粒子スロット別の層化学習率 η_1..η_N を生成する (§8.1)。

    指数分布 (Gamma α=1) の逆CDFを中点配置し、平均を厳密に eta_mean へ
    正規化した後、固定ランダム置換でスロットへ割り当てる。置換は祖先-学習率
    相関 (systematic resample の単調祖先) を除去するため (§3.2)。

    eta_rng は粒子初期化・システムノイズ用の rng から **分離** する。
    これにより固定学習率版と同一シードで θ 側の乱数列が一致し、対照実験が
    成立する (§8.1)。

    Parameters
    ----------
    n_particles : int
    eta_mean : float          学習率分布の平均 η̄ (探索対象はこれのみ)
    seed : int, optional      η 専用 rng のシード (θ 系と分離)

    Returns
    -------
    eta_slots : ndarray (N,)   Σ η_i / N == eta_mean が厳密に成立
    """
    n = int(n_particles)
    u = (np.arange(n) + 0.5) / n
    multipliers = -np.log1p(-u)          # 指数分布の逆CDF (中点)
    multipliers = multipliers / multipliers.mean()   # 平均を厳密に 1
    eta_rng = np.random.default_rng(seed)
    perm = eta_rng.permutation(n)        # 祖先-学習率相関の除去
    return float(eta_mean) * multipliers[perm]


def stratified_eta_diagnostics(weights, eta_slots, eta_mean):
    """重み付き学習率診断 (§9): (wmean, wstd, slow_mass, fast_mass, map_eta)。

    重み更新後・リサンプリング前の重みで評価する。slow/fast は eta_mean の
    0.5 倍未満 / 1.5 倍超で定義する。
    """
    w = np.asarray(weights, dtype=np.float64)
    eta = np.asarray(eta_slots, dtype=np.float64)
    wmean = float(np.sum(w * eta))
    wvar = float(np.sum(w * (eta - wmean) ** 2))
    wstd = float(np.sqrt(max(wvar, 0.0)))
    slow_mass = float(np.sum(w[eta < 0.5 * eta_mean]))
    fast_mass = float(np.sum(w[eta > 1.5 * eta_mean]))
    map_eta = float(eta[int(np.argmax(w))])
    return wmean, wstd, slow_mass, fast_mass, map_eta


def fast_to_slow_rate(ancestor_idx, eta_slots, eta_mean):
    """fast→slow 移行率 (§9): 祖先が高速スロット・新スロットが低速の割合。

    R = (1/N) Σ_i 1[ η(a_i) > 1.5 η̄  かつ  η_i < 0.5 η̄ ]。
    リサンプリングが発動したステップでのみ意味を持つ (それ以外は a_i=i)。
    """
    eta = np.asarray(eta_slots, dtype=np.float64)
    anc = eta[np.asarray(ancestor_idx, dtype=np.int64)]
    fast = anc > 1.5 * eta_mean
    slow = eta < 0.5 * eta_mean
    return float(np.mean(fast & slow))


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
