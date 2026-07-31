#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
評価指標(metrics) — ベンチマーク非依存の純粋関数群

修正方針: 実験ファイルに散らばっていた MSE / R² / カバレッジ / F1 / Brier /
ECE などの指標計算を 1 か所に集約する。ここでの関数は
「すでに計算済みの予測(mean/std/probs)」を受け取るものを基本とし、
model やフィルタの内部実装に依存しない(R2-5 の予測区間規約に忠実)。

回帰の予測区間は観測ノイズ y = f + ε を含める必要がある(R2-5):
    pred_std = sqrt( weighted_particle_var(f) + obs_sigma² )

粒子予測(predict_fn + weights)から予測平均/分散を作る小さなヘルパも提供する。
"""

from __future__ import annotations

import math

import numpy as np


# ======================================================================
# 正規分布ユーティリティ(scipy 非依存)
# ======================================================================
def _norm_cdf(x):
    """標準正規分布の CDF (math.erf ベース、ベクトル化)。"""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x):
    """標準正規分布の PDF。"""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# Acklam の逆正規 CDF 近似(|error| < 1.15e-9)。scipy を使わずに ppf を得る。
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def _norm_ppf_scalar(p):
    """標準正規分布の逆 CDF(Acklam 近似, スカラー)。"""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


def norm_ppf(p):
    """標準正規分布の逆 CDF(スカラー/配列)。"""
    if np.isscalar(p):
        return _norm_ppf_scalar(float(p))
    return np.array([_norm_ppf_scalar(float(pi)) for pi in np.ravel(p)]).reshape(np.shape(p))


def _z_for_level(level):
    """両側被覆確率 level に対応する z 値(= ppf(0.5 + level/2))。"""
    return _norm_ppf_scalar(0.5 + 0.5 * level)


# ======================================================================
# 粒子予測ヘルパ(predict_fn + weights → 予測平均/分散)
# ======================================================================
def _as_pred_matrix(preds):
    """predict_fn の出力を (N, M) に整形する(末尾の singleton 次元を除去)。"""
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds[..., 0]
    if preds.ndim == 1:
        preds = preds[None, :]
    return preds


def weighted_prediction(predict_fn, particles, weights, X):
    """
    粒子群から重み付き予測平均と(粒子由来の)予測分散を計算する。

    Parameters
    ----------
    predict_fn : callable
        (particles[N, d], X) -> 予測 (N, M) あるいは (N, M, 1)
    particles : ndarray, shape (N, d)
    weights : ndarray, shape (N,)
        正規化重み(和 1)。点推定器では [1.0]、particles は (1, d)。
    X : ndarray

    Returns
    -------
    pred_mean : ndarray, shape (M,)
    pred_var : ndarray, shape (M,)
        重み付き粒子分散 Var_w(f)。点推定では 0。
    """
    preds = _as_pred_matrix(predict_fn(particles, X))  # (N, M)
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-300)
    pred_mean = np.sum(w[:, None] * preds, axis=0)  # (M,)
    pred_var = np.sum(w[:, None] * (preds - pred_mean[None, :]) ** 2, axis=0)
    return pred_mean, pred_var


def prediction_std_with_noise(pred_var, obs_sigma):
    """予測 std に観測ノイズを含める(R2-5): sqrt(Var_w(f) + σ_obs²)。"""
    return np.sqrt(np.maximum(pred_var, 0.0) + float(obs_sigma) ** 2)


# ======================================================================
# 回帰指標
# ======================================================================
def test_mse(y, pred_mean):
    """テスト MSE。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    return float(np.mean((pred_mean - y) ** 2))


def test_mae(y, pred_mean):
    """テスト MAE。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    return float(np.mean(np.abs(pred_mean - y)))


def test_r2(y, pred_mean):
    """テスト R²(決定係数)。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    ss_res = float(np.sum((y - pred_mean) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def nll_gaussian(y, mean, std):
    """
    ガウス予測分布の平均負対数尤度(NLL)。
        NLL = ½ log(2π σ²) + (y − μ)² / (2σ²)
    std は観測ノイズを含んだ予測 std(R2-5)を渡すこと。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.asarray(std, dtype=np.float64).ravel()
    var = np.maximum(std ** 2, 1e-30)
    nll = 0.5 * np.log(2.0 * np.pi * var) + (y - mean) ** 2 / (2.0 * var)
    return float(np.mean(nll))


def crps_gaussian(y, mean, std):
    """
    ガウス予測分布の CRPS(連続確率順位スコア)の閉形式:
        CRPS = σ [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],  z = (y − μ)/σ
    小さいほど良い。std は観測ノイズを含む予測 std を渡す。
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.asarray(std, dtype=np.float64).ravel()
    std_safe = np.maximum(std, 1e-30)
    z = (y - mean) / std_safe
    crps = std_safe * (z * (2.0 * _norm_cdf(z) - 1.0)
                       + 2.0 * _norm_pdf(z) - 1.0 / math.sqrt(math.pi))
    return float(np.mean(crps))


def coverage_and_width(y, pred_mean, pred_std, levels=(0.5, 0.8, 0.9, 0.95)):
    """
    ガウス予測区間の被覆率・平均区間幅・被覆誤差(名目 − 実測)を水準別に返す。

    pred_std は観測ノイズを含む予測 std を渡すこと(R2-5)。

    Returns
    -------
    dict[float, dict]
        level -> {"coverage", "width", "cov_error"}
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    pred_std = np.asarray(pred_std, dtype=np.float64).ravel()
    out = {}
    for lvl in levels:
        z = _z_for_level(lvl)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        cov = float(np.mean((y >= lower) & (y <= upper)))
        width = float(np.mean(2.0 * z * pred_std))
        out[float(lvl)] = {
            "coverage": cov,
            "width": width,
            "cov_error": float(lvl) - cov,
        }
    return out


# ======================================================================
# 分類指標
# ======================================================================
def _hard_pred(probs):
    return (np.asarray(probs, dtype=np.float64).ravel() > 0.5).astype(np.float64)


def accuracy(pred, y):
    """正解率(pred はハードラベル)。"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean(pred == y))


def precision(pred, y, pos_label=1.0):
    """適合率。"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tp = np.sum((pred == pos_label) & (y == pos_label))
    fp = np.sum((pred == pos_label) & (y != pos_label))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(pred, y, pos_label=1.0):
    """再現率。"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tp = np.sum((pred == pos_label) & (y == pos_label))
    fn = np.sum((pred != pos_label) & (y == pos_label))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(pred, y, pos_label=1.0):
    """positive クラスの F1 スコア(旧 compute_f1_from_pred と一致)。"""
    p = precision(pred, y, pos_label)
    r = recall(pred, y, pos_label)
    if p + r == 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def balanced_accuracy(pred, y, pos_label=1.0):
    """クラス平均再現率(不均衡データ向け, R2-5)。"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    pos = y == pos_label
    neg = ~pos
    tpr = float(np.mean(pred[pos] == pos_label)) if pos.any() else 0.0
    tnr = float(np.mean(pred[neg] != pos_label)) if neg.any() else 0.0
    return float(0.5 * (tpr + tnr))


def nll_bernoulli(probs, y, eps=1e-10):
    """ベルヌーイ平均負対数尤度(小さいほど良い)。"""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.clip(probs, eps, 1.0 - eps)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return float(-np.mean(ll))


def loglik_bernoulli(probs, y, eps=1e-10):
    """ベルヌーイ平均対数尤度(旧 compute_loglik と一致, 大きいほど良い)。"""
    return -nll_bernoulli(probs, y, eps)


def brier_ece(probs, labels, n_bins=10):
    """
    Brier スコアと ECE(期待較正誤差)、および信頼度図の (x, y) を返す。
    旧 calibration_report._brier_ece と同一実装。

    Returns
    -------
    brier : float
    ece : float
    rel_x : ndarray  各ビンの平均予測確率(confidence)
    rel_y : ndarray  各ビンの実測正例率(accuracy)
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    brier = float(np.mean((probs - labels) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    rel_x, rel_y = [], []
    n = len(probs)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
        rel_x.append(conf)
        rel_y.append(acc)
    return brier, float(ece), np.array(rel_x), np.array(rel_y)
