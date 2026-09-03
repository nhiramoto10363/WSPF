#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation metrics: benchmark-independent pure functions.

The functions take already-computed predictions (mean, std or class
probabilities), so they depend on neither the model nor the filter internals.

Regression prediction intervals must include the observation noise, since
y = f + eps:

    pred_std = sqrt(weighted_particle_var(f) + obs_sigma^2)

Small helpers turn a particle prediction (predict_fn plus weights) into that
mean and variance.
"""

from __future__ import annotations

import math

import numpy as np


# ======================================================================
# Normal-distribution utilities (no scipy dependency)
# ======================================================================
def _norm_cdf(x):
    """Standard normal CDF, vectorized over math.erf."""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# Acklam's inverse normal CDF approximation (|error| < 1.15e-9), so that the
# quantile function is available without scipy.
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def _norm_ppf_scalar(p):
    """Standard normal quantile function (Acklam), scalar input."""
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
    """Standard normal quantile function, scalar or array."""
    if np.isscalar(p):
        return _norm_ppf_scalar(float(p))
    return np.array([_norm_ppf_scalar(float(pi)) for pi in np.ravel(p)]).reshape(np.shape(p))


def _z_for_level(level):
    """z for a two-sided coverage level: ppf(0.5 + level/2)."""
    return _norm_ppf_scalar(0.5 + 0.5 * level)


# ======================================================================
# Particle-prediction helpers
# ======================================================================
def _as_pred_matrix(preds):
    """Reshape a predict_fn output to (N, M), dropping a trailing axis of 1."""
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim == 3 and preds.shape[-1] == 1:
        preds = preds[..., 0]
    if preds.ndim == 1:
        preds = preds[None, :]
    return preds


def particle_predictions(predict_fn, particles, X):
    """Per-particle prediction matrix (N, M), calling predict_fn once.

    Both the mixture NLL and the weighted moments are computed from this same
    matrix.
    """
    return _as_pred_matrix(predict_fn(particles, X))  # (N, M)


def weighted_moments(preds, weights):
    """Weighted mean and variance of a prediction matrix.

    Returns
    -------
    pred_mean : ndarray, shape (M,)
    pred_var : ndarray, shape (M,)
        Weighted particle variance Var_w(f); zero for a point estimator.
    """
    preds = np.asarray(preds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-300)
    pred_mean = np.sum(w[:, None] * preds, axis=0)  # (M,)
    pred_var = np.sum(w[:, None] * (preds - pred_mean[None, :]) ** 2, axis=0)
    return pred_mean, pred_var


def weighted_prediction(predict_fn, particles, weights, X):
    """Weighted predictive mean and particle variance.

    Parameters
    ----------
    predict_fn : callable
        (particles[N, d], X) -> (N, M) or (N, M, 1)
    particles : ndarray, shape (N, d)
    weights : ndarray, shape (N,)
        Normalized weights. A point estimator passes [1.0] with particles of
        shape (1, d).
    X : ndarray

    Returns
    -------
    pred_mean : ndarray, shape (M,)
    pred_var : ndarray, shape (M,)
    """
    return weighted_moments(
        particle_predictions(predict_fn, particles, X), weights)


def prediction_std_with_noise(pred_var, obs_sigma):
    """Add the observation noise: sqrt(Var_w(f) + obs_sigma^2)."""
    return np.sqrt(np.maximum(pred_var, 0.0) + float(obs_sigma) ** 2)


def weighted_prediction_proba(predict_fn, particles, weights, X):
    """Weighted average of the per-particle class probabilities.

    predict_fn(particles[N, d], X) must return class probabilities of shape
    (N, B, C). Averaging the probabilities produced by each particle - rather
    than predicting from an averaged parameter - matches the convention of
    weighted_prediction, and the convex combination is again a probability.

    Returns
    -------
    probs : ndarray, shape (B, C)
    """
    preds = np.asarray(predict_fn(particles, X), dtype=np.float64)
    if preds.ndim == 2:   # binary probabilities (N, B): widen to two classes
        preds = np.stack([1.0 - preds, preds], axis=-1)
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-300)
    return np.einsum("n,nbc->bc", w, preds)   # (B, C)


# ======================================================================
# Regression metrics
# ======================================================================
def test_mse(y, pred_mean):
    """Test MSE."""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    return float(np.mean((pred_mean - y) ** 2))


def test_mae(y, pred_mean):
    """Test MAE."""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    return float(np.mean(np.abs(pred_mean - y)))


def test_r2(y, pred_mean):
    """Test R^2 (coefficient of determination)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    pred_mean = np.asarray(pred_mean, dtype=np.float64).ravel()
    ss_res = float(np.sum((y - pred_mean) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def nll_gaussian(y, mean, std):
    """Mean negative log-likelihood of a Gaussian predictive distribution.

        NLL = 0.5 log(2 pi sigma^2) + (y - mu)^2 / (2 sigma^2)

    Pass a std that already includes the observation noise.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.asarray(std, dtype=np.float64).ravel()
    var = np.maximum(std ** 2, 1e-30)
    nll = 0.5 * np.log(2.0 * np.pi * var) + (y - mean) ** 2 / (2.0 * var)
    return float(np.mean(nll))


def nll_gaussian_mixture(y, preds, weights, obs_sigma):
    """Mean NLL of the *mixture* predictive density.

        NLL = -(1/M) sum_j log[ sum_i w_i N(y_j ; mu_{i,j}, obs_sigma^2) ]

    nll_gaussian instead scores a *single* Gaussian moment-matched to the
    particle distribution, N(y; mubar, Var_w(f) + obs_sigma^2); the two
    disagree when the particles are multimodal or the weights have
    degenerated. The mixture is the correct predictive distribution of a
    particle filter, and stays meaningful even when the weights collapse onto
    one particle.

    Parameters
    ----------
    y : ndarray, shape (M,)
    preds : ndarray, shape (N, M)
        Per-particle predictions, from particle_predictions.
    weights : ndarray, shape (N,)
        Normalized weights.
    obs_sigma : float
        Observation noise, shared by all particles.

    Notes
    -----
    Uses logsumexp for stability. The mixture density degenerates at
    obs_sigma = 0, so the variance is floored at 1e-15; values obtained there
    are only indicative.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim == 1:
        preds = preds[None, :]
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = w / max(w.sum(), 1e-300)

    var = max(float(obs_sigma) ** 2, 1e-15)
    # log N(y_j; mu_{i,j}, sigma^2)                              -> (N, M)
    log_n = -0.5 * (np.log(2.0 * np.pi * var)
                    + (y[None, :] - preds) ** 2 / var)
    a = np.log(np.maximum(w, 1e-300))[:, None] + log_n           # (N, M)
    amax = np.max(a, axis=0)                                     # (M,)
    log_mix = amax + np.log(np.sum(np.exp(a - amax[None, :]), axis=0))
    return float(-np.mean(log_mix))


def crps_gaussian(y, mean, std):
    """Closed-form CRPS of a Gaussian predictive distribution.

        CRPS = sigma [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],
        z = (y - mu) / sigma

    Lower is better. Pass a std that includes the observation noise.
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
    """Coverage, mean interval width and coverage error, by nominal level.

    The coverage error is nominal minus empirical. Pass a pred_std that
    includes the observation noise.

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
# Classification metrics
# ======================================================================
def _hard_pred(probs):
    return (np.asarray(probs, dtype=np.float64).ravel() > 0.5).astype(np.float64)


def accuracy(pred, y):
    """Accuracy; pred holds hard labels."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(np.mean(pred == y))


def precision(pred, y, pos_label=1.0):
    """Precision."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tp = np.sum((pred == pos_label) & (y == pos_label))
    fp = np.sum((pred == pos_label) & (y != pos_label))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(pred, y, pos_label=1.0):
    """Recall."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    tp = np.sum((pred == pos_label) & (y == pos_label))
    fn = np.sum((pred != pos_label) & (y == pos_label))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(pred, y, pos_label=1.0):
    """F1 of the positive class."""
    p = precision(pred, y, pos_label)
    r = recall(pred, y, pos_label)
    if p + r == 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def balanced_accuracy(pred, y, pos_label=1.0):
    """Balanced accuracy: the mean of the per-class recalls."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    pos = y == pos_label
    neg = ~pos
    tpr = float(np.mean(pred[pos] == pos_label)) if pos.any() else 0.0
    tnr = float(np.mean(pred[neg] != pos_label)) if neg.any() else 0.0
    return float(0.5 * (tpr + tnr))


def nll_bernoulli(probs, y, eps=1e-10):
    """Mean Bernoulli negative log-likelihood; lower is better."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.clip(probs, eps, 1.0 - eps)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    return float(-np.mean(ll))


def loglik_bernoulli(probs, y, eps=1e-10):
    """Mean Bernoulli log-likelihood; higher is better."""
    return -nll_bernoulli(probs, y, eps)


# ======================================================================
# Multiclass metrics
# ======================================================================
def macro_f1(pred, y, n_classes):
    """Macro-averaged F1: the unweighted mean of the per-class F1 scores.

    pred and y are integer hard labels. An absent class (tp+fp = 0 and
    tp+fn = 0) contributes an F1 of zero.
    """
    pred = np.asarray(pred).ravel().astype(np.int64)
    y = np.asarray(y).ravel().astype(np.int64)
    f1s = []
    for c in range(int(n_classes)):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(0.0 if prec + rec == 0
                   else 2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def nll_categorical(probs, y, eps=1e-12):
    """Mean categorical negative log-likelihood; probs (B, C), y (B,)."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).ravel().astype(np.int64)
    B = y.shape[0]
    p_true = np.clip(probs[np.arange(B), y], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def brier_multiclass(probs, y, n_classes):
    """Multiclass Brier score, mean_i sum_c (p_ic - onehot_ic)^2."""
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).ravel().astype(np.int64)
    B = y.shape[0]
    onehot = np.zeros((B, int(n_classes)), dtype=np.float64)
    onehot[np.arange(B), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def brier_ece(probs, labels, n_bins=10):
    """Brier score, expected calibration error, and reliability-diagram points.

    Returns
    -------
    brier : float
    ece : float
    rel_x : ndarray  mean predicted probability (confidence) per bin
    rel_y : ndarray  empirical positive rate (accuracy) per bin
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


def brier_ece_multiclass(probs, y, n_classes, n_bins=10):
    """Multiclass Brier, ECE and reliability diagram, matching brier_ece.

    - The Brier score is brier_multiclass, the mean squared error summed over
      classes, with range [0, 2].
    - The ECE uses top-label calibration (confidence = max_c p_ic,
      correctness = 1[argmax_c p_ic == y_i]), the standard multiclass
      definition and the counterpart of the binary ECE, with range [0, 1].
    - The reliability diagram is top-label as well: x is the mean confidence
      in a bin, y its accuracy.

    Parameters
    ----------
    probs : ndarray, shape (B, C)
    y : ndarray, shape (B,)
        Integer class labels.
    n_classes : int
    n_bins : int

    Returns
    -------
    brier : float
    ece : float
    rel_x : ndarray
    rel_y : ndarray
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).ravel().astype(np.int64)
    C = int(n_classes)
    if probs.ndim != 2 or probs.shape[1] != C:
        raise ValueError(
            f"probs must have shape (B, {C}), got {probs.shape}")
    if probs.shape[0] != y.shape[0]:
        raise ValueError(
            f"probs and y have different lengths: "
            f"{probs.shape[0]} vs {y.shape[0]}")

    brier = brier_multiclass(probs, y, C)

    conf_all = probs.max(axis=1)                       # (B,)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf_all, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    rel_x, rel_y = [], []
    n = conf_all.size
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = conf_all[mask].mean()
        acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
        rel_x.append(conf)
        rel_y.append(acc)
    return brier, float(ece), np.array(rel_x), np.array(rel_y)
