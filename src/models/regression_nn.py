#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression MLP with one hidden layer, used to track a drifting function."""

import numpy as np


class NeuralNetRegression:
    """One-hidden-layer regression network.

    input(input_dim) -> hidden(hidden_dim, tanh) -> output(output_dim, linear)
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation="tanh"):
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
        """Split a flat parameter vector into per-layer weights."""
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
            Pre-activation.
        hidden_act : ndarray, shape (N, B, hidden_dim)
            Post-activation.
        """
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        X_exp = X.reshape(1, B, self.input_dim)
        X_broadcast = np.broadcast_to(X_exp, (N, B, self.input_dim))

        hidden = np.einsum("nbi,nih->nbh", X_broadcast, W1) + b1

        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
        else:  # relu
            hidden_act = np.maximum(0.0, hidden)

        output = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2

        return output, hidden, hidden_act

    def predict(self, flat_params, X):
        """Predict for a single parameter vector."""
        output, _, _ = self.forward(flat_params.reshape(1, -1), X)
        return output.squeeze(0).squeeze(-1)  # (B,)

    def loglik_batch(self, flat_params, X, y, noise_std=0.1):
        """Gaussian log-likelihood, summed over the batch.

        log p(y|x,theta) = -0.5 (y - f(x,theta))^2 / sigma^2
                           - 0.5 log(2 pi sigma^2)
        """
        output, _, _ = self.forward(flat_params, X)  # (N, B, 1)
        output = output.squeeze(-1)  # (N, B)

        y_row = y.reshape(1, -1)  # (1, B)
        residual = output - y_row  # (N, B)

        ll = -0.5 * (residual ** 2) / (noise_std ** 2) - 0.5 * np.log(2 * np.pi * noise_std ** 2)
        return ll.sum(axis=1)  # (N,)

    def grad_nll_batch(self, flat_params, X, y, noise_std=0.1):
        """Batch-mean gradient of the negative log-likelihood."""
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        X_exp = X.reshape(1, B, self.input_dim)
        X_broadcast = np.broadcast_to(X_exp, (N, B, self.input_dim))

        hidden = np.einsum("nbi,nih->nbh", X_broadcast, W1) + b1
        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
            hidden_deriv = 1.0 - hidden_act ** 2
        else:
            hidden_act = np.maximum(0.0, hidden)
            hidden_deriv = (hidden > 0).astype(np.float64)

        output = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2  # (N, B, 1)

        y_exp = y.reshape(1, B, 1)
        delta_out = (output - y_exp) / (noise_std ** 2) / B  # (N, B, 1)

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

    def grad_nll_per_sample(self, flat_params, X, y, noise_std=0.1):
        """Per-sample NLL gradients, needed to estimate the noise covariance.

        Averaging over the sample axis recovers grad_nll_batch.

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
        X : ndarray, shape (B, input_dim)
        y : ndarray, shape (B,)
        noise_std : float

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

        output = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2  # (N, B, 1)

        # Per-sample: no division by B.
        y_exp = y.reshape(1, B, 1)
        delta_out = (output - y_exp) / (noise_std ** 2)  # (N, B, 1)

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


def create_regression_grad_fn(model, noise_std=0.1):
    """Bind a model into a batch-gradient function."""
    def grad_fn(particles, X, y):
        return model.grad_nll_batch(particles, X, y, noise_std)
    return grad_fn


def create_regression_loglik_fn(model, noise_std=0.1):
    """Bind a model into a log-likelihood function."""
    def loglik_fn(particles, X, y):
        return model.loglik_batch(particles, X, y, noise_std)
    return loglik_fn


def create_regression_per_sample_grad_fn(model, noise_std=0.1):
    """Bind a model into a per-sample gradient function (WSPF-A/B)."""
    def per_sample_grad_fn(particles, X, y):
        return model.grad_nll_per_sample(particles, X, y, noise_std)
    return per_sample_grad_fn
