#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多クラス分類用ニューラルネットワーク(1隠れ層MLP, softmax出力)

neural_net.py (二値, sigmoid) の多クラス版。INSECTS 実験用。
規約は二値版と同一:
- loglik_batch はバッチ内サンプルの対数尤度の「和」を返す
  (観測モデル p(B_t|θ) = Π_j p(y_j|x_j,θ) に対応)
- grad_nll_batch はバッチ「平均」勾配 (1/B)Σ ∇ℓ_j を返す
- grad_nll_per_sample は per-sample 勾配 (B で割らない) を返し、
  mean(axis=1) が grad_nll_batch と一致する
"""

import numpy as np


def _log_softmax(logits):
    """数値安定な log-softmax (最終軸)。logits: (..., C)"""
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return z - lse


class MulticlassNeuralNetModel:
    """
    入力(input_dim) -> 隠れ層(hidden_dim, tanh) -> 出力(n_classes, softmax)
    """

    def __init__(self, input_dim, hidden_dim, n_classes, activation="tanh"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.output_dim = n_classes
        self.activation = activation

        self.W1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.W2_size = hidden_dim * n_classes
        self.b2_size = n_classes
        self.param_dim = self.W1_size + self.b1_size + self.W2_size + self.b2_size

    def unflatten_params(self, flat_params):
        if flat_params.ndim == 1:
            flat_params = flat_params.reshape(1, -1)
        N = flat_params.shape[0]
        idx = 0
        W1 = flat_params[:, idx: idx + self.W1_size].reshape(
            N, self.input_dim, self.hidden_dim)
        idx += self.W1_size
        b1 = flat_params[:, idx: idx + self.b1_size].reshape(N, 1, self.hidden_dim)
        idx += self.b1_size
        W2 = flat_params[:, idx: idx + self.W2_size].reshape(
            N, self.hidden_dim, self.n_classes)
        idx += self.W2_size
        b2 = flat_params[:, idx: idx + self.b2_size].reshape(N, 1, self.n_classes)
        return W1, b1, W2, b2

    def _forward_core(self, flat_params, X):
        """logits と隠れ層関連を返す共通部"""
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]
        X_exp = X.reshape(1, B, self.input_dim)
        X_broadcast = np.broadcast_to(X_exp, (N, B, self.input_dim))

        hidden = np.einsum("nbi,nih->nbh", X_broadcast, W1) + b1
        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
            hidden_deriv = 1.0 - hidden_act ** 2
        else:  # relu
            hidden_act = np.maximum(0.0, hidden)
            hidden_deriv = (hidden > 0).astype(np.float64)

        logits = np.einsum("nbh,nhc->nbc", hidden_act, W2) + b2
        return logits, hidden_act, hidden_deriv, X_broadcast, W2

    def forward(self, flat_params, X):
        """
        順伝播。二値版と同じ 3 タプル (output, hidden, hidden_act) を返す。
        output はクラス確率 (N, B, C)。
        """
        logits, hidden_act, _, _, _ = self._forward_core(flat_params, X)
        log_p = _log_softmax(logits)
        return np.exp(log_p), None, hidden_act

    def predict(self, flat_params, X):
        """argmax 予測ラベル。flat_params: (param_dim,) or (N, param_dim)"""
        logits, _, _, _, _ = self._forward_core(
            np.atleast_2d(flat_params), X)
        return logits.argmax(axis=-1)  # (N, B)

    def loglik_batch(self, flat_params, X, y):
        """
        バッチ対数尤度(サンプル和)。y: (B,) の整数クラスラベル。
        Returns: (N,)
        """
        logits, _, _, _, _ = self._forward_core(flat_params, X)
        log_p = _log_softmax(logits)                       # (N, B, C)
        y_idx = np.asarray(y, dtype=np.int64)
        B = y_idx.shape[0]
        ll = log_p[:, np.arange(B), y_idx]                 # (N, B)
        return ll.sum(axis=1)

    def _delta_out(self, flat_params, X, y):
        """softmax − onehot と付随テンソルを返す"""
        logits, hidden_act, hidden_deriv, X_broadcast, W2 = \
            self._forward_core(flat_params, X)
        p = np.exp(_log_softmax(logits))                   # (N, B, C)
        y_idx = np.asarray(y, dtype=np.int64)
        B = y_idx.shape[0]
        onehot = np.zeros((1, B, self.n_classes))
        onehot[0, np.arange(B), y_idx] = 1.0
        delta = p - onehot                                 # (N, B, C)
        return delta, hidden_act, hidden_deriv, X_broadcast, W2

    def grad_nll_batch(self, flat_params, X, y):
        """バッチ平均勾配。Returns: (N, param_dim)"""
        delta, hidden_act, hidden_deriv, X_broadcast, W2 = \
            self._delta_out(flat_params, X, y)
        N, B = delta.shape[0], delta.shape[1]
        delta = delta / B

        grad_W2 = np.einsum("nbh,nbc->nhc", hidden_act, delta)
        grad_b2 = delta.sum(axis=1, keepdims=True)
        delta_hidden = np.einsum("nbc,nhc->nbh", delta, W2) * hidden_deriv
        grad_W1 = np.einsum("nbi,nbh->nih", X_broadcast, delta_hidden)
        grad_b1 = delta_hidden.sum(axis=1, keepdims=True)

        return np.concatenate(
            [grad_W1.reshape(N, -1), grad_b1.reshape(N, -1),
             grad_W2.reshape(N, -1), grad_b2.reshape(N, -1)], axis=1)

    def grad_nll_per_sample(self, flat_params, X, y):
        """
        per-sample 勾配 (B で割らない)。Returns: (N, B, param_dim)
        mean(axis=1) は grad_nll_batch と一致する。
        """
        delta, hidden_act, hidden_deriv, X_broadcast, W2 = \
            self._delta_out(flat_params, X, y)
        N, B = delta.shape[0], delta.shape[1]

        grad_W1 = np.einsum(
            "nbi,nbh->nbih", X_broadcast,
            np.einsum("nbc,nhc->nbh", delta, W2) * hidden_deriv)
        grad_b1 = np.einsum("nbc,nhc->nbh", delta, W2) * hidden_deriv
        grad_W2 = np.einsum("nbh,nbc->nbhc", hidden_act, delta)
        grad_b2 = delta

        return np.concatenate(
            [grad_W1.reshape(N, B, -1), grad_b1.reshape(N, B, -1),
             grad_W2.reshape(N, B, -1), grad_b2.reshape(N, B, -1)], axis=2)


def create_mc_grad_fn(model):
    def grad_fn(particles, X, y):
        return model.grad_nll_batch(particles, X, y)
    return grad_fn


def create_mc_loglik_fn(model):
    def loglik_fn(particles, X, y):
        return model.loglik_batch(particles, X, y)
    return loglik_fn


def create_mc_per_sample_grad_fn(model):
    def per_sample_grad_fn(particles, X, y):
        return model.grad_nll_per_sample(particles, X, y)
    return per_sample_grad_fn
