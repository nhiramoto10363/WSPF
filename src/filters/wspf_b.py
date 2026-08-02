#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted SGD Particle Filter — Method B (WSPF-B) 実装 — 修正版

Method B: スカラー近似（∇L 推定不要）による重要度重み補正。

修正内容:
  [Bug1] s̄ のスケーリング: (1/Bd) → (1/B²d)  per-sample分散→バッチ勾配分散
  [Bug2] WSPF_B: 重みの累積（SIS/SIR に忠実に）
  [Bug3] 数値安定性: log_correction のクランプ

理論:
    prior 遷移:   p(θ_t|θ_{t-1}) = N(θ_t | θ_{t-1} − η∇L, η²Σ(θ) + Q_cd)
    proposal:     q(θ_t|θ_{t-1}, B_t) = N(θ_t | θ_{t-1} − ηĝ, Q_cd)
    補正比:       R_t = p(θ_t|θ_{t-1}) / q(θ_t|θ_{t-1}, B_t)

Method B (スカラー近似, ∇L 推定不要):
    E_ξ[log R_t] = -(d/2) log(1/(1−ρ)) + ρ·||ε||²/(2σ²_cd) − (d/2)ρ

    ρ = η²s̄ / (η²s̄ + σ²_cd)    signal-to-drift ratio
    s̄ = tr(Σ̂) / d               バッチ勾配のスカラー分散

    (I)   Volume penalty:     prior が proposal より広い
    (II)  Noise realization:  ||ε|| が大きい粒子を優遇
    (III) Expectation offset:  期待値補正
"""

import time

import numpy as np
from .base import (
    normalize_logweights,
    effective_sample_size,
    systematic_resample,
    weight_diagnostics,
    ensemble_spread_trace,
    init_phi,
    propagate_phi,
    phi_to_sigma,
    weighted_sigma_mean,
)

# ρ を安全域に留めるためのクリップ上限（論文 Alg.2 参照）。
RHO_CLIP = 0.999

# ================================================================
# 補正項の計算
# ================================================================
def compute_gradient_noise_variance(per_sample_grads):
    """
    ミニバッチから **バッチ勾配** のスカラーノイズ分散 s̄ と偏差を推定

    理論:
        ĝ = (1/B) Σ_j g_j
        Var(ĝ) = C(θ) / B,  C(θ) = E[(g_j − ∇L)(g_j − ∇L)^T]
        s̄ = tr(Var(ĝ)) / d

    推定 (論文 eq:shat / Alg. — 不偏推定量):
        ŝ = (1/(d·B(B−1))) Σ_j ||g_j − ĝ||²

    注) 以前は 1/(B²d) を用いていたが、論文の不偏推定量 1/(B(B−1)d) に統一。
        これは Σ̂ = (1/(B(B−1))) Σ_j (g_j−ĝ)(g_j−ĝ)^T のトレース/d と一致し、
        WSPF-A(行列版)と WSPF-B(スカラー版)の推定量が整合する。

    Parameters
    ----------
    per_sample_grads : ndarray, shape (N, B, d)
        各粒子・各サンプルの NLL 勾配

    Returns
    -------
    g_hat : ndarray, shape (N, d)
        バッチ平均勾配
    s_bar : ndarray, shape (N,)
        バッチ勾配のスカラーノイズ分散 ŝ
    deviations : ndarray, shape (N, B, d)
        各サンプル勾配の偏差 g_j − ĝ (WSPF-A の Σ̂ 構成に使用)
    """
    N, B, d = per_sample_grads.shape

    g_hat = per_sample_grads.mean(axis=1)  # (N, d)

    deviations = per_sample_grads - g_hat[:, np.newaxis, :]  # (N, B, d)

    # ŝ = (1/(d·B(B−1))) Σ_j ||g_j − ĝ||²  (論文 eq:shat)
    s_bar = np.sum(deviations ** 2, axis=(1, 2)) / (B * (B - 1) * d)  # (N,)

    return g_hat, s_bar, deviations


def compute_correction_method_b(epsilon, eta, s_bar, sigma_sys_sq, d):
    """
    Method B: スカラー近似による重要度重み補正項

    E_ξ[log R_t^(i)] = -(d/2) log(1 + η²s̄/σ²_cd)
                       + ρ · ||ε||² / (2σ²_cd)
                       - (d/2) ρ

    Parameters
    ----------
    epsilon : ndarray, shape (N, d)
        concept drift ノイズの実現値
    eta : float
        学習率
    s_bar : ndarray, shape (N,)
        各粒子のスカラー勾配ノイズ分散 (バッチ勾配の分散)
    sigma_sys_sq : float
        システムノイズの分散 σ²_cd
    d : int
        パラメータ次元数

    Returns
    -------
    log_correction : ndarray, shape (N,)
        補正項 E_ξ[log R_t^(i)]
    rho : ndarray, shape (N,)
        signal-to-drift ratio ρ^(i)
    """
    eta_sq = eta ** 2
    s_bar_safe = np.maximum(s_bar, 1e-30)
    rho_raw = eta_sq * s_bar_safe / (eta_sq * s_bar_safe + sigma_sys_sq)
    # 論文 Alg.2: ρ ← min(η²ŝ/(η²ŝ+σ²), 0.999)  log(1/(1−ρ)) の発散を防ぐ
    rho = np.minimum(rho_raw, RHO_CLIP)

    eps_norm_sq = np.sum(epsilon ** 2, axis=1)  # (N,)

    # クリップ後の ρ で全項を評価 (論文 eq:logR_rho / Alg.2)
    # (I) Volume penalty:  −(d/2) log(1/(1−ρ)) = (d/2) log(1−ρ)
    term1 = 0.5 * d * np.log1p(-rho)

    # (II) Noise realization bonus
    term2 = rho * eps_norm_sq / (2.0 * sigma_sys_sq)

    # (III) Expectation offset
    term3 = -0.5 * d * rho

    log_correction_raw = term1 + term2 + term3

    # 論文 Alg.2 に存在しない ±d/2 クランプは撤廃(補正を非線形に歪め、
    # 低 σcd 域では恒常的に発動して手法を実質置換していたため)。
    # ρ クリップ(0.999)で volume penalty は有限に抑えられるので、
    # ここでは非有限値のみを中立値 0 でガードする。
    finite = np.isfinite(log_correction_raw)
    log_correction = np.where(finite, log_correction_raw, 0.0)
    nonfinite_count = int(np.sum(~finite))  # 非有限ガード発動数(通常 0)

    return log_correction, rho, nonfinite_count


# ================================================================
# WSPF_B: 標準リサンプリング + 補正
# ================================================================
class WSPF_B:
    """
    Weighted SGD Particle Filter — Method B (WSPF-B)

    システムモデル:
        θ_t = θ_{t-1} − η * ĝ(θ; B_t) + ε_t,  ε_t ~ N(0, σ_sys² I)

    重み更新 (SIS):
        log w_t^(i) = log w_{t-1}^(i) + log p(B_t | θ_t^(i)) + log R_t^(i)

    縮退制御:
        ESS < 閾値 のとき systematic resampling → 重みリセット
    """

    def __init__(
        self,
        n_particles,
        param_dim,
        eta=0.05,
        sigma_sys=0.02,
        prior_mean=0.0,
        prior_std=1.0,
        ess_resample_ratio=0.5,
        grad_clip_norm=None,
        seed=None,
        adaptive_obs=False,
        tau_phi=0.05,
        phi_init_mean=0.0,
        phi_init_std=0.5,
        phi_seed=None,
    ):
        """
        Parameters
        ----------
        n_particles : int
            粒子数 N
        param_dim : int
            パラメータ次元数 d
        eta : float
            学習率
        sigma_sys : float
            システムノイズの標準偏差 σ_cd
        prior_mean : float
            事前分布の平均
        prior_std : float
            事前分布の標準偏差
        ess_resample_ratio : float
            リサンプリング閾値（粒子数に対する比率）
        grad_clip_norm : float or None
            勾配クリッピングのノルム上限
        seed : int, optional
            乱数シード
        adaptive_obs : bool
            True なら観測ノイズ φ_t = log σ_t² を粒子別状態として推定する
            (φ_t 拡張)。False なら従来と完全に同一挙動。
        tau_phi : float
            φ ランダムウォークの標準偏差 τ
        phi_init_mean : float
            φ_0 の初期平均 (通常 2 log σ_ref)
        phi_init_std : float
            φ_0 の初期標準偏差
        phi_seed : int, optional
            φ 専用 rng のシード (θ 系の乱数列と分離)
        """
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.sigma_sys_sq = sigma_sys ** 2
        self.ess_resample_ratio = ess_resample_ratio
        self.grad_clip_norm = grad_clip_norm

        self.rng = np.random.default_rng(seed)

        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        # --- φ_t 拡張: 粒子別観測ノイズ状態 ---
        self.adaptive_obs = bool(adaptive_obs)
        self.tau_phi = float(tau_phi)
        if self.adaptive_obs:
            self.phi_rng = np.random.default_rng(phi_seed)
            self.phi = init_phi(self.phi_rng, self.N,
                                phi_init_mean, phi_init_std)
        else:
            self.phi_rng = None
            self.phi = None

        self.history = {
            "mean": [],
            "std": [],
            "ess": [],
            "ll_mean": [],
            "rho_mean": [],
            "rho_max": [],
            "log_correction_mean": [],
            # --- 縮退診断量 (R1-8, R1-9, R2-2) ---
            "entropy": [],           # 重みエントロピー −Σ w log w
            "max_weight": [],        # 最大正規化重み max_i w_i
            "spread_trace": [],      # パラメータ空間の ensemble spread (tr Cov)
            "unique_particles": [],  # リサンプリング後のユニーク祖先数
            "resampled": [],         # このステップでリサンプリングしたか
            # --- WSPF-B 固有: ρ 監視 (R1-8) ---
            "rho": [],               # 各ステップ・各粒子の ρ^(i) (shape (N,))
            "rho_clip_count": [],    # ρ >= RHO_CLIP(0.999) に到達した粒子数
            "logcorr_nonfinite_count": [],  # log_correction 非有限ガード発動数(通常0)
            # --- 計算コスト計測 (R1-11) ---
            "t_step": [],
            "t_grad": [],            # per-sample 勾配評価の時間 [s]
            "t_loglik": [],
            "t_correction": [],      # 補正項(スカラー)計算の時間 [s]
            "t_weight": [],
            "t_resample": [],
            "sample_grad_evals": [], # このステップの per-sample 勾配評価数(=N*B)
            # --- φ_t 拡張診断 (adaptive_obs 時のみ追記) ---
            "sigma_hat_mean": [],    # 重み付き平均 σ̂_t = Σ w_i exp(φ_i/2)
            "phi_std": [],           # 粒子間 φ の std (φ 縮退の監視)
        }
        # 勾配評価の種別と累積カウント (R1-11)
        self.grad_eval_kind = "per_sample"  # WSPF-B は per-sample 勾配が必要
        self.grad_calls = 0
        self.sample_grad_evals_total = 0

    @property
    def obs_sigma_particles(self):
        """粒子別観測ノイズ std σ^(i) = exp(φ^(i)/2)。非適応時は None。"""
        if not self.adaptive_obs:
            return None
        return phi_to_sigma(self.phi)

    def step(self, X, y, per_sample_grad_fn, loglik_fn, loglik_sigma_fn=None):
        """
        1ステップの更新（補正付き）

        Parameters
        ----------
        X : ndarray, shape (B, input_dim)
            入力データ
        y : ndarray, shape (B,) or (B, output_dim)
            ラベル
        per_sample_grad_fn : callable
            per_sample_grad_fn(particles, X, y) -> ndarray (N, B, param_dim)
            各粒子・各サンプルの NLL 勾配
        loglik_fn : callable
            loglik_fn(particles, X, y) -> ndarray (N,)
            各粒子の対数尤度

        Returns
        -------
        mean : ndarray, shape (param_dim,)
            パラメータの推定値（重み付き平均）
        """
        _t0 = time.perf_counter()

        # 1) Per-sample 勾配 → バッチ平均 & ノイズ分散
        per_grads = per_sample_grad_fn(self.particles, X, y)  # (N, B, d)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(per_grads.shape[0] * per_grads.shape[1])  # N*B
        self.sample_grad_evals_total += n_sample_grads

        g_hat, s_bar, _ = compute_gradient_noise_variance(per_grads)

        # 勾配クリッピング
        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        # 2) SGD 更新 + concept drift ノイズ
        epsilon = self.rng.normal(
            0.0, self.sigma_sys, size=self.particles.shape
        )
        self.particles = self.particles - self.eta * g_hat + epsilon

        # 2b) φ 遷移 (adaptive_obs 時のみ; φ 専用 rng → θ 側の乱数列は不変)
        if self.adaptive_obs:
            self.phi = propagate_phi(self.phi, self.tau_phi, self.phi_rng)

        # 3) 補正項 (Method B) — φ は観測非依存の事前遷移から提案されるため
        #    prior–proposal 比の φ ブロックは恒等的に 1。log R̂ は θ ブロック
        #    のみで従来通り計算する (設計書 §0.3)。
        log_correction, rho, logcorr_nonfinite_count = compute_correction_method_b(
            epsilon, self.eta, s_bar, self.sigma_sys_sq, self.param_dim
        )
        _t_corr = time.perf_counter()

        # 4) 対数尤度 (adaptive_obs 時は粒子別 σ^(i)=exp(φ^(i)/2) を使用)
        if self.adaptive_obs:
            if loglik_sigma_fn is None:
                raise ValueError(
                    "adaptive_obs=True には loglik_sigma_fn が必要です")
            ll = loglik_sigma_fn(self.particles, X, y, phi_to_sigma(self.phi))
        else:
            ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()

        # 5) 重み更新（累積 + 尤度 + 補正）
        #    [Bug2 修正] SIS の逐次累積: log w_t = log w_{t-1} + increments
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        log_weights = log_prev + ll + log_correction
        self.weights = normalize_logweights(log_weights)

        # 6) 推定値の計算（重み付き平均）
        mean = (self.weights[:, None] * self.particles).sum(axis=0)
        var = (self.weights[:, None] * (self.particles - mean) ** 2).sum(
            axis=0
        )
        std = np.sqrt(np.maximum(var, 1e-15))
        ll_mean = float((self.weights * ll).sum())

        ess, entropy, max_weight = weight_diagnostics(self.weights)
        spread_trace = ensemble_spread_trace(self.particles)
        rho_clip_count = int(np.sum(rho >= RHO_CLIP))
        sigma_hat_mean = (weighted_sigma_mean(self.weights, self.phi)
                          if self.adaptive_obs else None)
        _t_wt = time.perf_counter()

        # 7) ESS が低い場合はリサンプリング → 重みリセット
        if ess < self.ess_resample_ratio * self.N:
            idx = systematic_resample(self.weights, self.rng)
            n_unique = int(np.unique(idx).size)
            self.particles = self.particles[idx]
            if self.adaptive_obs:
                self.phi = self.phi[idx]   # φ も同一祖先で引き継ぐ
            self.weights = np.ones(self.N) / self.N
            resampled = True
        else:
            n_unique = self.N
            resampled = False
        _t_rs = time.perf_counter()

        # φ 診断 (adaptive_obs 時のみ)
        if self.adaptive_obs:
            self.history["sigma_hat_mean"].append(sigma_hat_mean)
            self.history["phi_std"].append(float(np.std(self.phi)))

        # 履歴に保存
        self.history["mean"].append(mean.copy())
        self.history["std"].append(std.copy())
        self.history["ess"].append(ess)
        self.history["ll_mean"].append(ll_mean)
        self.history["rho_mean"].append(float(np.mean(rho)))
        self.history["rho_max"].append(float(np.max(rho)))
        self.history["log_correction_mean"].append(
            float(np.mean(log_correction))
        )
        self.history["entropy"].append(entropy)
        self.history["max_weight"].append(max_weight)
        self.history["spread_trace"].append(spread_trace)
        self.history["unique_particles"].append(n_unique)
        self.history["resampled"].append(resampled)
        self.history["rho"].append(rho.copy())
        self.history["rho_clip_count"].append(rho_clip_count)
        self.history["logcorr_nonfinite_count"].append(logcorr_nonfinite_count)
        # 計算コスト (R1-11)
        self.history["t_grad"].append(_t_grad - _t0)
        self.history["t_correction"].append(_t_corr - _t_grad)
        self.history["t_loglik"].append(_t_ll - _t_corr)
        self.history["t_weight"].append(_t_wt - _t_ll)
        self.history["t_resample"].append(_t_rs - _t_wt)
        self.history["t_step"].append(_t_rs - _t0)
        self.history["sample_grad_evals"].append(n_sample_grads)

        return mean

    def run(self, X_list, y_list, per_sample_grad_fn, loglik_fn):
        """全時刻での推定を実行"""
        T = len(X_list)
        means = np.empty((T, self.param_dim))
        for t in range(T):
            means[t] = self.step(
                X_list[t], y_list[t], per_sample_grad_fn, loglik_fn
            )
        return means

    def get_history(self):
        """履歴を ndarray で返す"""
        return {k: np.array(v) for k, v in self.history.items()}

