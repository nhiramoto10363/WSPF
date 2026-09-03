#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Binary classification MLP with one hidden layer.

Provides what the particle filters need: flattening and unflattening of the
parameters, and gradients and log-likelihoods evaluated for all particles at
once.
"""

import numpy as np
from ..filters.base import sigmoid, softplus


class NeuralNetModel:
    """One-hidden-layer network for binary classification.

    input(input_dim) -> hidden(hidden_dim, tanh) -> output(output_dim, sigmoid)
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation="tanh"):
        """
        Parameters
        ----------
        input_dim : int
        hidden_dim : int
        output_dim : int
            1 for binary classification.
        activation : str
            "tanh" or "relu".
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation

        self.W1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.W2_size = hidden_dim * output_dim
        self.b2_size = output_dim
        self.param_dim = self.W1_size + self.b1_size + self.W2_size + self.b2_size

    def unflatten_params(self, flat_params):
        """Split a flat parameter vector into per-layer weights.

        Parameters
        ----------
        flat_params : ndarray, shape (param_dim,) or (N, param_dim)

        Returns
        -------
        W1, b1, W2, b2 : tuple of ndarray
        """
        if flat_params.ndim == 1:
            flat_params = flat_params.reshape(1, -1)

        N = flat_params.shape[0]
        idx = 0

        W1 = flat_params[:, idx : idx + self.W1_size].reshape(
            N, self.input_dim, self.hidden_dim
        )
        idx += self.W1_size

        b1 = flat_params[:, idx : idx + self.b1_size].reshape(N, 1, self.hidden_dim)
        idx += self.b1_size

        W2 = flat_params[:, idx : idx + self.W2_size].reshape(
            N, self.hidden_dim, self.output_dim
        )
        idx += self.W2_size

        b2 = flat_params[:, idx : idx + self.b2_size].reshape(N, 1, self.output_dim)

        return W1, b1, W2, b2

    def forward(self, flat_params, X):
        """Forward pass.

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
        X : ndarray, shape (B, input_dim)

        Returns
        -------
        output : ndarray, shape (N, B, output_dim)
        hidden : ndarray, shape (N, B, hidden_dim)
            Pre-activation, kept for the backward pass.
        hidden_act : ndarray, shape (N, B, hidden_dim)
        """
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        X_exp = X.reshape(1, B, self.input_dim)

        hidden = np.einsum("nbi,nih->nbh", np.broadcast_to(X_exp, (N, B, self.input_dim)), W1) + b1

        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
        else:  # relu
            hidden_act = np.maximum(0.0, hidden)

        logits = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2

        output = sigmoid(logits)

        return output, hidden, hidden_act

    def loglik_batch(self, flat_params, X, y):
        """Log-likelihood of a batch, summed over samples.

        log p(y|x) = y log p + (1-y) log(1-p)
                   = -softplus(-logits) if y = 1, -softplus(logits) if y = 0

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
        X : ndarray, shape (B, input_dim)
        y : ndarray, shape (B,)
            Labels in {0, 1}.

        Returns
        -------
        ll : ndarray, shape (N,)
        """
        output, _, _ = self.forward(flat_params, X)  # (N, B, 1)
        output = output.squeeze(-1)  # (N, B)

        logits = np.clip(np.log(output / (1.0 - output + 1e-10) + 1e-10), -60, 60)

        y_row = y.reshape(1, -1)  # (1, B)
        ll = -softplus(-logits) * y_row - softplus(logits) * (1.0 - y_row)

        return ll.sum(axis=1)  # (N,)

    def grad_nll_batch(self, flat_params, X, y):
        """Batch-mean gradient of the negative log-likelihood.

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
        X : ndarray, shape (B, input_dim)
        y : ndarray, shape (B,)

        Returns
        -------
        grad : ndarray, shape (N, param_dim)
        """
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

        logits = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2
        output = sigmoid(logits)  # (N, B, 1)

        # Output-layer error: dL/d(logits) = output - y
        y_exp = y.reshape(1, B, 1)
        delta_out = (output - y_exp) / B  # (N, B, 1)

        grad_W2 = np.einsum("nbh,nbo->nho", hidden_act, delta_out)
        grad_b2 = delta_out.sum(axis=1, keepdims=True)

        delta_hidden = np.einsum("nbo,nho->nbh", delta_out, W2) * hidden_deriv

        grad_W1 = np.einsum("nbi,nbh->nih", X_broadcast, delta_hidden)
        grad_b1 = delta_hidden.sum(axis=1, keepdims=True)

        grad = np.concatenate(
            [
                grad_W1.reshape(N, -1),
                grad_b1.reshape(N, -1),
                grad_W2.reshape(N, -1),
                grad_b2.reshape(N, -1),
            ],
            axis=1,
        )

        return grad

    def grad_nll_per_sample(self, flat_params, X, y):
        """Per-sample NLL gradients, needed to estimate the noise covariance.

        Averaging over the sample axis recovers grad_nll_batch.

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
        X : ndarray, shape (B, input_dim)
        y : ndarray, shape (B,)

        Returns
        -------
        grad : ndarray, shape (N, B, param_dim)
        """
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

        logits = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2
        output = sigmoid(logits)  # (N, B, 1)

        # Per-sample: no division by B.
        y_exp = y.reshape(1, B, 1)
        delta_out = output - y_exp  # (N, B, 1)

        delta_hidden = (
            np.einsum("nbo,nho->nbh", delta_out, W2) * hidden_deriv
        )  # (N, B, hidden_dim)

        grad_W1 = np.einsum("nbi,nbh->nbih", X_broadcast, delta_hidden)
        grad_b1 = delta_hidden
        grad_W2 = np.einsum("nbh,nbo->nbho", hidden_act, delta_out)
        grad_b2 = delta_out

        grad = np.concatenate(
            [
                grad_W1.reshape(N, B, -1),
                grad_b1.reshape(N, B, -1),
                grad_W2.reshape(N, B, -1),
                grad_b2.reshape(N, B, -1),
            ],
            axis=2,
        )

        return grad


def create_nn_grad_fn(model):
    """Bind a model into a batch-gradient function."""
    def grad_fn(particles, X, y):
        return model.grad_nll_batch(particles, X, y)
    return grad_fn


def create_nn_loglik_fn(model):
    """Bind a model into a log-likelihood function."""
    def loglik_fn(particles, X, y):
        return model.loglik_batch(particles, X, y)
    return loglik_fn


def create_nn_per_sample_grad_fn(model):
    """Bind a model into a per-sample gradient function (WSPF-A/B)."""
    def per_sample_grad_fn(particles, X, y):
        return model.grad_nll_per_sample(particles, X, y)
    return per_sample_grad_fn


def generate_nn_stream_data(
    model,
    T=400,
    batch_size=8,
    theta0_scale=0.5,
    sigma_theta_rw=0.01,
    x_scale=1.0,
    seed=0,
):
    """Generate a stream whose true parameters follow a random walk.

    Parameters
    ----------
    model : NeuralNetModel
    T : int
        Number of time steps.
    batch_size : int
    theta0_scale : float
        Scale of the initial parameters.
    sigma_theta_rw : float
        Random-walk noise on the parameters.
    x_scale : float
    seed : int

    Returns
    -------
    X_list : list of ndarray
    y_list : list of ndarray
    theta_true : ndarray, shape (T, param_dim)
    """
    rng = np.random.default_rng(seed)
    param_dim = model.param_dim

    theta_true = np.empty((T, param_dim))
    theta_true[0] = rng.normal(0.0, theta0_scale, size=param_dim)

    X_list = []
    y_list = []

    for t in range(T):
        if t > 0:
            theta_true[t] = theta_true[t - 1] + rng.normal(
                0.0, sigma_theta_rw, size=param_dim
            )

        X = rng.normal(0.0, x_scale, size=(batch_size, model.input_dim))

        output, _, _ = model.forward(theta_true[t : t + 1], X)
        p = output.squeeze()  # (B,)

        y = rng.binomial(1, np.clip(p, 0.001, 0.999), size=batch_size).astype(np.float64)

        X_list.append(X)
        y_list.append(y)

    return X_list, y_list, theta_true
