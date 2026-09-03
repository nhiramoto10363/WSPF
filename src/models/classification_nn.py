#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiclass classification MLP with one hidden layer and softmax output.

The multiclass counterpart of binary_nn.py, used for INSECTS. The conventions
match the binary version:

  - loglik_batch returns the *sum* of per-sample log-likelihoods, matching the
    observation model p(B_t | theta) = prod_j p(y_j | x_j, theta);
  - grad_nll_batch returns the batch *mean* gradient (1/B) sum_j grad l_j;
  - grad_nll_per_sample returns per-sample gradients (no division by B), whose
    mean over the sample axis equals grad_nll_batch.
"""

import numpy as np


def _log_softmax(logits):
    """Numerically stable log-softmax over the last axis."""
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return z - lse


class MulticlassNeuralNetModel:
    """input(input_dim) -> hidden(hidden_dim, tanh) -> output(n_classes, softmax)"""

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
        """Shared forward pass returning logits and the hidden-layer tensors."""
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
        """Forward pass returning (class probabilities, None, hidden_act).

        The three-tuple shape matches the binary model.
        """
        logits, hidden_act, _, _, _ = self._forward_core(flat_params, X)
        log_p = _log_softmax(logits)
        return np.exp(log_p), None, hidden_act

    def predict(self, flat_params, X):
        """Argmax class labels, shape (N, B)."""
        logits, _, _, _, _ = self._forward_core(
            np.atleast_2d(flat_params), X)
        return logits.argmax(axis=-1)  # (N, B)

    def loglik_batch(self, flat_params, X, y):
        """Log-likelihood summed over the batch; y holds integer labels."""
        logits, _, _, _, _ = self._forward_core(flat_params, X)
        log_p = _log_softmax(logits)                       # (N, B, C)
        y_idx = np.asarray(y, dtype=np.int64)
        B = y_idx.shape[0]
        ll = log_p[:, np.arange(B), y_idx]                 # (N, B)
        return ll.sum(axis=1)

    def _delta_out(self, flat_params, X, y):
        """Return softmax - onehot along with the tensors the backward pass needs."""
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
        """Batch-mean gradient, shape (N, param_dim)."""
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
        """Per-sample gradients, shape (N, B, param_dim)."""
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
