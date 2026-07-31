#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡易ニューラルネットワーク（1隠れ層MLP）

粒子フィルタ用に以下を提供:
- パラメータのフラット化/復元
- 勾配計算（全粒子に対してバッチ処理）
- 対数尤度計算（全粒子に対してバッチ処理）
"""

import numpy as np
from ..filters.base import sigmoid, softplus


class NeuralNetModel:
    """
    1隠れ層のニューラルネットワーク

    構造:
        入力(input_dim) -> 隠れ層(hidden_dim, tanh) -> 出力(output_dim, sigmoid)

    二値分類用（output_dim=1の場合）
    """

    def __init__(self, input_dim, hidden_dim, output_dim=1, activation="tanh"):
        """
        Parameters
        ----------
        input_dim : int
            入力次元
        hidden_dim : int
            隠れ層のユニット数
        output_dim : int
            出力次元（二値分類なら1）
        activation : str
            隠れ層の活性化関数 ("tanh" or "relu")
        """
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
        """
        フラットなパラメータベクトルを各層の重みに展開

        Parameters
        ----------
        flat_params : ndarray, shape (param_dim,) or (N, param_dim)
            フラットなパラメータ

        Returns
        -------
        W1, b1, W2, b2 : tuple of ndarray
            各層のパラメータ
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
        """
        順伝播

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
            粒子（パラメータ）
        X : ndarray, shape (B, input_dim)
            入力データ（バッチ）

        Returns
        -------
        output : ndarray, shape (N, B, output_dim)
            出力
        hidden : ndarray, shape (N, B, hidden_dim)
            隠れ層の活性化前の値（勾配計算用）
        hidden_act : ndarray, shape (N, B, hidden_dim)
            隠れ層の活性化後の値
        """
        W1, b1, W2, b2 = self.unflatten_params(flat_params)
        N = W1.shape[0]
        B = X.shape[0]

        # X: (B, input_dim) -> (1, B, input_dim)
        X_exp = X.reshape(1, B, self.input_dim)

        # 隠れ層: (N, B, input_dim) @ (N, input_dim, hidden_dim) -> (N, B, hidden_dim)
        hidden = np.einsum("nbi,nih->nbh", np.broadcast_to(X_exp, (N, B, self.input_dim)), W1) + b1

        # 活性化
        if self.activation == "tanh":
            hidden_act = np.tanh(hidden)
        else:  # relu
            hidden_act = np.maximum(0.0, hidden)

        # 出力層: (N, B, hidden_dim) @ (N, hidden_dim, output_dim) -> (N, B, output_dim)
        logits = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2

        # シグモイド出力（二値分類）
        output = sigmoid(logits)

        return output, hidden, hidden_act

    def loglik_batch(self, flat_params, X, y):
        """
        バッチの対数尤度を計算

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
            粒子（パラメータ）
        X : ndarray, shape (B, input_dim)
            入力データ
        y : ndarray, shape (B,)
            ラベル {0, 1}

        Returns
        -------
        ll : ndarray, shape (N,)
            各粒子の対数尤度（バッチ全体の合計）
        """
        output, _, _ = self.forward(flat_params, X)  # (N, B, 1)
        output = output.squeeze(-1)  # (N, B)

        # 二値分類の対数尤度
        # log p(y|x) = y*log(p) + (1-y)*log(1-p)
        # = -softplus(-logits) if y=1, -softplus(logits) if y=0
        logits = np.clip(np.log(output / (1.0 - output + 1e-10) + 1e-10), -60, 60)

        y_row = y.reshape(1, -1)  # (1, B)
        ll = -softplus(-logits) * y_row - softplus(logits) * (1.0 - y_row)

        return ll.sum(axis=1)  # (N,)

    def grad_nll_batch(self, flat_params, X, y):
        """
        負の対数尤度の勾配を計算（バッチ平均）

        Parameters
        ----------
        flat_params : ndarray, shape (N, param_dim)
            粒子（パラメータ）
        X : ndarray, shape (B, input_dim)
            入力データ
        y : ndarray, shape (B,)
            ラベル {0, 1}

        Returns
        -------
        grad : ndarray, shape (N, param_dim)
            各粒子の勾配
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

        logits = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2
        output = sigmoid(logits)  # (N, B, 1)

        # 逆伝播
        # 出力層の誤差: dL/d(logits) = output - y
        y_exp = y.reshape(1, B, 1)
        delta_out = (output - y_exp) / B  # (N, B, 1)

        # W2の勾配: (N, hidden_dim, B) @ (N, B, 1) -> (N, hidden_dim, 1)
        grad_W2 = np.einsum("nbh,nbo->nho", hidden_act, delta_out)  # (N, hidden_dim, output_dim)

        # b2の勾配
        grad_b2 = delta_out.sum(axis=1, keepdims=True)  # (N, 1, 1)

        # 隠れ層への誤差伝播
        # delta_hidden = delta_out @ W2.T * hidden_deriv
        delta_hidden = np.einsum("nbo,nho->nbh", delta_out, W2) * hidden_deriv  # (N, B, hidden_dim)

        # W1の勾配
        grad_W1 = np.einsum("nbi,nbh->nih", X_broadcast, delta_hidden)  # (N, input_dim, hidden_dim)

        # b1の勾配
        grad_b1 = delta_hidden.sum(axis=1, keepdims=True)  # (N, 1, hidden_dim)

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

    def grad_nll_per_sample(self, flat_params, X, y):
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
            ラベル {0, 1}

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

        logits = np.einsum("nbh,nho->nbo", hidden_act, W2) + b2
        output = sigmoid(logits)  # (N, B, 1)

        # 逆伝播（per-sample: B で割らない）
        y_exp = y.reshape(1, B, 1)
        delta_out = output - y_exp  # (N, B, 1)

        # 隠れ層への誤差伝播
        delta_hidden = (
            np.einsum("nbo,nho->nbh", delta_out, W2) * hidden_deriv
        )  # (N, B, hidden_dim)

        # Per-sample 勾配（B 次元を保持）
        # W1: (N, B, input_dim, hidden_dim)
        grad_W1 = np.einsum("nbi,nbh->nbih", X_broadcast, delta_hidden)
        # b1: (N, B, hidden_dim)
        grad_b1 = delta_hidden
        # W2: (N, B, hidden_dim, output_dim)
        grad_W2 = np.einsum("nbh,nbo->nbho", hidden_act, delta_out)
        # b2: (N, B, output_dim)
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


def create_nn_grad_fn(model):
    """モデルから勾配関数を作成"""
    def grad_fn(particles, X, y):
        return model.grad_nll_batch(particles, X, y)
    return grad_fn


def create_nn_loglik_fn(model):
    """モデルから対数尤度関数を作成"""
    def loglik_fn(particles, X, y):
        return model.loglik_batch(particles, X, y)
    return loglik_fn


def create_nn_per_sample_grad_fn(model):
    """モデルから per-sample 勾配関数を作成（WSPF-B用）"""
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
    """
    ニューラルネット用のストリーミングデータを生成

    真のパラメータはランダムウォークで変動（コンセプトドリフト）

    Parameters
    ----------
    model : NeuralNetModel
        モデル
    T : int
        時刻数
    batch_size : int
        各時刻のバッチサイズ
    theta0_scale : float
        初期パラメータのスケール
    sigma_theta_rw : float
        パラメータのランダムウォークのノイズ
    x_scale : float
        入力データのスケール
    seed : int
        乱数シード

    Returns
    -------
    X_list : list of ndarray
        各時刻の入力データ
    y_list : list of ndarray
        各時刻のラベル
    theta_true : ndarray, shape (T, param_dim)
        真のパラメータ
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

        # 真のパラメータで予測
        output, _, _ = model.forward(theta_true[t : t + 1], X)
        p = output.squeeze()  # (B,)

        # ベルヌーイサンプリング
        y = rng.binomial(1, np.clip(p, 0.001, 0.999), size=batch_size).astype(np.float64)

        X_list.append(X)
        y_list.append(y)

    return X_list, y_list, theta_true
