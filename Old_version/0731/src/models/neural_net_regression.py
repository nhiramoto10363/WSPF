#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回帰用ニューラルネットワーク（1隠れ層MLP）

時間変動する関数の近似用
"""

import numpy as np


class NeuralNetRegression:
    """
    1隠れ層の回帰用ニューラルネットワーク

    構造:
        入力(input_dim) -> 隠れ層(hidden_dim, tanh) -> 出力(output_dim, 線形)
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation="tanh"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation

        # パラメータのサイズ
        self.W1_size = input_dim * hidden_dim
        self.b1_size = hidden_dim
        self.W2_size = hidden_dim * output_dim
        self.b2_size = output_dim
        self.param_dim = self.W1_size + self.b1_size + self.W2_size + self.b2_size

    def unflatten_params(self, flat_params):
        """フラットなパラメータを各層の重みに展開"""
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
        """
        順伝播

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
            粒子（パラメータ）
        X : ndarray, shape (B, input_dim)
            入力データ

        Returns
        -------
        output : ndarray, shape (N, B, output_dim)
            出力（線形）
        hidden : ndarray, shape (N, B, hidden_dim)
            隠れ層の活性化前
        hidden_act : ndarray, shape (N, B, hidden_dim)
            隠れ層の活性化後
        """
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        X_exp = X.reshape(1, B, self.input_dim)
        X_broadcast = np.broadcast_to(X_exp, (N, B, self.input_dim))

        # 隠れ層
        hidden = np.einsum("nbi,nih->nbh", X_broadcast, W1) + b1

        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
        else:  # relu
            hidden_act = np.maximum(0.0, hidden)

        # 出力層（線形）
        output = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2

        return output, hidden, hidden_act

    def predict(self, flat_params, X):
        """予測（単一パラメータ用の便利関数）"""
        output, _, _ = self.forward(flat_params.reshape(1, -1), X)
        return output.squeeze(0).squeeze(-1)  # (B,)

    def loglik_batch(self, flat_params, X, y, noise_std=0.1):
        """
        ガウスノイズを仮定した対数尤度

        log p(y|x,θ) = -0.5 * (y - f(x,θ))^2 / σ^2 - 0.5*log(2πσ^2)
        """
        output, _, _ = self.forward(flat_params, X)  # (N, B, 1)
        output = output.squeeze(-1)  # (N, B)

        y_row = y.reshape(1, -1)  # (1, B)
        residual = output - y_row  # (N, B)

        # 対数尤度（定数項は省略可能だが含める）
        ll = -0.5 * (residual ** 2) / (noise_std ** 2) - 0.5 * np.log(2 * np.pi * noise_std ** 2)
        return ll.sum(axis=1)  # (N,)

    def grad_nll_batch(self, flat_params, X, y, noise_std=0.1):
        """
        負の対数尤度の勾配（MSE勾配と等価、スケール違い）
        """
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        X_exp = X.reshape(1, B, self.input_dim)
        X_broadcast = np.broadcast_to(X_exp, (N, B, self.input_dim))

        # 順伝播
        hidden = np.einsum("nbi,nih->nbh", X_broadcast, W1) + b1
        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
            hidden_deriv = 1.0 - hidden_act ** 2
        else:
            hidden_act = np.maximum(0.0, hidden)
            hidden_deriv = (hidden > 0).astype(np.float64)

        output = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2  # (N, B, 1)

        # 逆伝播
        y_exp = y.reshape(1, B, 1)
        delta_out = (output - y_exp) / (noise_std ** 2) / B  # (N, B, 1)

        # W2の勾配
        grad_W2 = np.einsum("nbh,nbo->nho", hidden_act, delta_out)
        grad_b2 = delta_out.sum(axis=1, keepdims=True)

        # 隠れ層への誤差伝播
        delta_hidden = np.einsum("nbo,nho->nbh", delta_out, W2) * hidden_deriv

        # W1の勾配
        grad_W1 = np.einsum("nbi,nbh->nih", X_broadcast, delta_hidden)
        grad_b1 = delta_hidden.sum(axis=1, keepdims=True)

        # フラット化
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
        """
        各サンプルの負の対数尤度の勾配を計算（バッチ平均ではなく個別）

        s̄ の推定に必要な per-sample 勾配を返す。
        g_hat = mean(grad_per_sample, axis=1) は grad_nll_batch と一致する。

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
            粒子（パラメータ）
        X : ndarray, shape (B, input_dim)
            入力データ
        y : ndarray, shape (B,)
            目標値
        noise_std : float
            観測ノイズの標準偏差

        Returns
        -------
        grad : ndarray, shape (N, B, param_dim)
            各粒子・各サンプルの勾配
        """
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        # 順伝播
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

        # 逆伝播（per-sample: B で割らない、noise_std でスケール）
        y_exp = y.reshape(1, B, 1)
        delta_out = (output - y_exp) / (noise_std ** 2)  # (N, B, 1)

        # 隠れ層への誤差伝播
        delta_hidden = (
            np.einsum("nbo,nho->nbh", delta_out, W2) * hidden_deriv
        )  # (N, B, hidden_dim)

        # Per-sample 勾配（B 次元を保持）
        grad_W1 = np.einsum("nbi,nbh->nbih", X_broadcast, delta_hidden)
        grad_b1 = delta_hidden
        grad_W2 = np.einsum("nbh,nbo->nbho", hidden_act, delta_out)
        grad_b2 = delta_out

        # フラット化: (N, B, param_dim)
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
    """モデルから勾配関数を作成"""
    def grad_fn(particles, X, y):
        return model.grad_nll_batch(particles, X, y, noise_std)
    return grad_fn


def create_regression_loglik_fn(model, noise_std=0.1):
    """モデルから対数尤度関数を作成"""
    def loglik_fn(particles, X, y):
        return model.loglik_batch(particles, X, y, noise_std)
    return loglik_fn


def create_regression_per_sample_grad_fn(model, noise_std=0.1):
    """モデルから per-sample 勾配関数を作成（WSPF-B用）"""
    def per_sample_grad_fn(particles, X, y):
        return model.grad_nll_per_sample(particles, X, y, noise_std)
    return per_sample_grad_fn
