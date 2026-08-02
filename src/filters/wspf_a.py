#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted SGD Particle Filter — Method A (WSPF-A, 行列版)

論文 Alg.1 / eq:methodA_logR_general の忠実な実装。Method A は勾配ノイズ
共分散を **フル行列** Σ̂ で保持し、方向情報(異方性)を補正に反映する。

  Method B: 勾配ノイズをスカラー ŝ に潰し、ξ を marginalize して期待補正
  Method A: EMA で全データ勾配 ∇L を推定して ξ̂ = ĝ − g̃ を明示的に使い、
            共分散もフル行列 Σ̂ で扱う

理論 (行列版):
    g̃_t^(i) = m_{t-1}^(i) / (1 − β^{t-1})                 # EMA ベースの ∇L 推定
    ξ̂_t^(i) = ĝ_t^(i) − g̃_t^(i),  Δμ̂ = η ξ̂                # ミニバッチノイズ/平均シフト
    Σ̂_t^(i) = (1/(B(B−1))) Σ_j (g_j−ĝ)(g_j−ĝ)^T           # d×d, rank ≤ B−1
    V̂p = η²Σ̂ + Q_cd,  Q_cd = σ²_cd I_d = V_q

    log R̂_A = ½ log(|V_q|/|V̂p|) + ½ ε^T V_q⁻¹ ε
              − ½ (ε − Δμ̂)^T V̂p⁻¹ (ε − Δμ̂)

    m_t^(i) = β m_{t-1}^(i) + (1−β) ĝ_t^(i)                # EMA 更新

数値実装 (R1-10):
    V̂p = c I_d + U U^T  (c = σ²_cd, U = √α · W^T, α = η²/(B(B−1)),
    W = 偏差行列 (B×d)) の低ランク構造を Woodbury 恒等式と行列式補題で活用し、
    d×d の直接反転を回避:
        M = I_B + c⁻¹ (α W W^T)               (B×B, SPD, 固有値 ≥ 1)
        log|V̂p| = d log c + log|M|
        (ε−Δμ̂)^T V̂p⁻¹ (ε−Δμ̂) = c⁻¹(‖v‖² − c⁻¹ pᵀ M⁻¹ p),  p = √α W v
    計算量は粒子ごと O(dB² + B³)(素朴な O(d³) を回避)。M の Cholesky は
    固有値 ≥ 1 より安定で、失敗時のみ jitter を追加、条件数を監視・記録する。
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
    make_stratified_learning_rates,
    stratified_eta_diagnostics,
    fast_to_slow_rate,
)
from .wspf_b import compute_gradient_noise_variance, RHO_CLIP


# ================================================================
# Method A 補正項の計算
# ================================================================
def compute_correction_method_a(epsilon, xi_hat, deviations, eta,
                                sigma_sys_sq, d):
    """
    Method A (行列版): フル共分散 Σ̂ を用いた重要度重み補正項。

    log R̂_A = ½ log(|V_q|/|V̂p|) + ½ ε^T V_q⁻¹ ε
              − ½ (ε − Δμ̂)^T V̂p⁻¹ (ε − Δμ̂),   Δμ̂ = η ξ̂

    V̂p = η²Σ̂ + σ²_cd I_d,  Σ̂ = (1/(B(B−1))) Σ_j (g_j−ĝ)(g_j−ĝ)^T
    を Woodbury/行列式補題で d×d 反転せずに評価する(docstring 参照)。

    Parameters
    ----------
    epsilon : ndarray, shape (N, d)
        concept drift ノイズの実現値
    xi_hat : ndarray, shape (N, d)
        推定ミニバッチノイズ ξ̂ = ĝ − g̃
    deviations : ndarray, shape (N, B, d)
        各サンプル勾配の偏差 g_j − ĝ  (Σ̂ の構成に使用)
    eta : float
        学習率
    sigma_sys_sq : float
        concept drift ノイズの分散 σ²_cd (= V_q の等方スケール)
    d : int
        パラメータ次元数

    Returns
    -------
    log_correction : ndarray, shape (N,)
        補正項 log R̂_A^(i)
    rho : ndarray, shape (N,)
        スカラー相当の signal-to-drift 比 (診断用): η²s̄/(η²s̄+σ²), s̄=tr(Σ̂)/d
    logcorr_nonfinite_count : int
        log_correction が非有限になり中立値 0 でガードした粒子数(通常 0)
    cond_M : ndarray, shape (N,)
        各粒子の M = I_B + c⁻¹αWW^T の条件数(数値安定性の監視用)
    jitter_count : int
        Cholesky が失敗し jitter fallback を要した粒子数(通常 0)
    """
    N, B, dd = deviations.shape
    c = sigma_sys_sq
    # 層化学習率 (§8.4): eta を (N,) に展開。スカラー入力は全粒子同値に
    # broadcast され、従来 (スカラー eta) と数値的に一致する。
    eta = np.broadcast_to(np.asarray(eta, dtype=np.float64), (N,))
    alpha = eta ** 2 / (B * (B - 1))                      # (N,)

    v = epsilon - eta[:, None] * xi_hat  # (N, d)  = ε − Δμ̂ (Δμ̂ = η_i ξ̂)

    # G = α W W^T  (N, B, B),  M = I_B + c⁻¹ G  (SPD, 固有値 ≥ 1)
    G = alpha[:, None, None] * np.einsum("nbd,ncd->nbc", deviations, deviations)
    M = G / c
    diag = np.arange(B)
    M[:, diag, diag] += 1.0

    # p = √α W v  (N, B)
    p = np.sqrt(alpha)[:, None] * np.einsum("nbd,nd->nb", deviations, v)

    # Cholesky（失敗時のみ、失敗した粒子だけに jitter を追加）
    # 高速な batched パスをまず試し、失敗時のみ粒子ごとのループに落として
    # jitter が必要だった粒子数 (jitter_count) を数える。
    jitter_count = 0
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        L = np.empty_like(M)
        for i in range(N):
            try:
                L[i] = np.linalg.cholesky(M[i])
            except np.linalg.LinAlgError:
                # この粒子のみ jitter を追加して再試行
                jit = 1e-8 * np.mean(np.diagonal(M[i]))
                M[i, diag, diag] += jit
                L[i] = np.linalg.cholesky(M[i])
                jitter_count += 1

    # log|M| = 2 Σ log diag(L)。対角は条件数近似にも再利用する。
    Ldiag = np.diagonal(L, axis1=1, axis2=2)  # (N, B)
    logdet_M = 2.0 * np.sum(np.log(Ldiag), axis=1)  # (N,)

    # M⁻¹ p を、log|M| と同一の Cholesky 因子 L を再利用して解く
    # (M = L Lᵀ の前進・後退代入)。solve(M, ...) を再計算しない。
    u = np.linalg.solve(L, p[:, :, None])                      # (N, B, 1)
    z = np.linalg.solve(np.swapaxes(L, -1, -2), u)[:, :, 0]    # (N, B)
    pMp = np.sum(p * z, axis=1)  # pᵀ M⁻¹ p  (N,)

    v_norm_sq = np.sum(v ** 2, axis=1)  # (N,)
    quad_p = (v_norm_sq - pMp / c) / c  # (ε−Δμ̂)ᵀ V̂p⁻¹ (ε−Δμ̂)

    eps_norm_sq = np.sum(epsilon ** 2, axis=1)  # (N,)
    quad_q = eps_norm_sq / c  # εᵀ V_q⁻¹ ε

    # log R̂_A = ½(d log c − log|V̂p|) + ½ quad_q − ½ quad_p
    #         = −½ log|M| + ½ quad_q − ½ quad_p     (d log c は相殺)
    log_correction_raw = -0.5 * logdet_M + 0.5 * quad_q - 0.5 * quad_p

    # スカラー相当 ρ 診断: s̄ = tr(Σ̂)/d。各粒子自身の η_i を使う。
    s_bar = np.sum(deviations ** 2, axis=(1, 2)) / (B * (B - 1) * d)
    rho = eta ** 2 * s_bar / (eta ** 2 * s_bar + c)

    # 条件数の安価な近似(Cholesky 対角比の2乗)。厳密な 2-ノルム条件数の
    # 代理だが、既に計算した L の対角を使うため追加コストがほぼ無く、
    # R1-11 の計時に np.linalg.cond の SVD コストを混入させない。
    cond_M = (Ldiag.max(axis=1) /
              np.maximum(Ldiag.min(axis=1), 1e-300)) ** 2  # (N,)

    # ±d/2 クランプは撤廃(wspf_b と同様)。非有限値のみ中立値 0 でガード。
    finite = np.isfinite(log_correction_raw)
    log_correction = np.where(finite, log_correction_raw, 0.0)
    nonfinite_count = int(np.sum(~finite))

    return log_correction, rho, nonfinite_count, cond_M, jitter_count


# ================================================================
# WSPF_A: Method A (EMA Plug-in) による補正付き粒子フィルタ
# ================================================================
class WSPF_A:
    """
    Weighted SGD Particle Filter — Method A (WSPF-A)

    Method B との違い:
      - 各粒子に EMA 状態 m^(i) を保持し、全データ勾配 ∇L を推定
      - ミニバッチノイズ ξ̂ を明示的に計算して補正項に使用
      - リサンプリング時に EMA 状態も複製

    システムモデル:
        θ_t = θ_{t-1} − η * ĝ(θ; B_t) + ε_t,  ε_t ~ N(0, σ_sys² I)

    重み更新 (SIS):
        log w_t^(i) = log w_{t-1}^(i) + log p(B_t | θ_t^(i)) + log R̂_A^(i)
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
        beta=0.9,
        seed=None,
        adaptive_obs=False,
        tau_phi=0.05,
        phi_init_mean=0.0,
        phi_init_std=0.5,
        phi_seed=None,
        eta_scheme="fixed",
        eta_seed=None,
    ):
        """
        Parameters
        ----------
        n_particles : int
            粒子数 N
        param_dim : int
            パラメータ次元数 d
        eta : float
            学習率。層化版 (eta_scheme="stratified_exp") では分布の平均 η̄ の意味
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
        beta : float
            EMA の忘却係数 β ∈ [0, 1)
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
        eta_scheme : {"fixed", "stratified_exp"}
            "fixed": 全粒子共通のスカラー eta (従来と完全一致)。
            "stratified_exp": 指数分布の層化学習率をスロット別に配置 (§3)。
        eta_seed : int, optional
            η 層化専用 rng のシード (θ 系の乱数列と分離, §8.1)
        """
        self.N = n_particles
        self.param_dim = param_dim
        # --- 層化学習率 (§8.2): eta は平均 η̄。fixed では全スロット η̄ ---
        self.eta_mean = float(eta)
        self.eta_scheme = eta_scheme
        if eta_scheme == "fixed":
            self.eta_slots = np.full(self.N, self.eta_mean)
        elif eta_scheme == "stratified_exp":
            self.eta_slots = make_stratified_learning_rates(
                self.N, self.eta_mean, eta_seed)
        else:
            raise ValueError(f"unknown eta_scheme: {eta_scheme!r}")
        self.eta = self.eta_mean   # 後方互換のスカラー別名
        self.sigma_sys = sigma_sys
        self.sigma_sys_sq = sigma_sys ** 2
        self.ess_resample_ratio = ess_resample_ratio
        self.grad_clip_norm = grad_clip_norm
        self.beta = beta

        self.rng = np.random.default_rng(seed)

        self.particles = self.rng.normal(
            prior_mean, prior_std, size=(self.N, param_dim)
        )
        self.weights = np.ones(self.N) / self.N

        # EMA 状態: 各粒子の勾配移動平均
        self.ema_m = np.zeros((self.N, param_dim))
        self.t_step = 0  # 時刻カウンタ

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
            "entropy": [],
            "max_weight": [],
            "spread_trace": [],
            "unique_particles": [],
            "resampled": [],
            # --- WSPF-A 固有: ρ 監視 (R1-8) ---
            "rho": [],
            "rho_clip_count": [],
            "logcorr_nonfinite_count": [],
            # --- WSPF-A 固有: 行列反転の条件数監視 (R1-10) ---
            "cond_M_mean": [],
            "cond_M_max": [],
            "jitter_count": [],
            # --- 計算コスト計測 (R1-11) ---
            "t_step": [],
            "t_grad": [],            # per-sample 勾配評価の時間 [s]
            "t_loglik": [],
            "t_correction": [],      # 行列補正(Woodbury/Cholesky/logdet)の時間 [s]
            "t_weight": [],
            "t_resample": [],
            "sample_grad_evals": [], # このステップの per-sample 勾配評価数(=N*B)
            # --- φ_t 拡張診断 (adaptive_obs 時のみ追記) ---
            "sigma_hat_mean": [],    # 重み付き平均 σ̂_t = Σ w_i exp(φ_i/2)
            "phi_std": [],           # 粒子間 φ の std (φ 縮退の監視)
            # --- 層化学習率診断 (§9; stratified_exp 時のみ意味を持つ) ---
            "eta_weighted_mean": [],
            "eta_weighted_std": [],
            "eta_slow_mass": [],
            "eta_fast_mass": [],
            "eta_map": [],
            "eta_fast_to_slow_rate": [],
        }
        # 勾配評価の種別と累積カウント (R1-11)
        self.grad_eval_kind = "per_sample"  # WSPF-A は per-sample 勾配が必要
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
        1ステップの更新（Method A 補正付き）

        Parameters
        ----------
        X : ndarray, shape (B, input_dim)
            入力データ
        y : ndarray, shape (B,) or (B, output_dim)
            ラベル
        per_sample_grad_fn : callable
            per_sample_grad_fn(particles, X, y) -> ndarray (N, B, param_dim)
        loglik_fn : callable
            loglik_fn(particles, X, y) -> ndarray (N,)

        Returns
        -------
        mean : ndarray, shape (param_dim,)
            パラメータの推定値（重み付き平均）
        """
        self.t_step += 1

        _t0 = time.perf_counter()

        # 1) Per-sample 勾配 → バッチ平均 & 偏差 (Σ̂ 用)
        per_grads = per_sample_grad_fn(self.particles, X, y)  # (N, B, d)
        _t_grad = time.perf_counter()
        self.grad_calls += 1
        n_sample_grads = int(per_grads.shape[0] * per_grads.shape[1])  # N*B
        self.sample_grad_evals_total += n_sample_grads

        g_hat, _s_bar, deviations = compute_gradient_noise_variance(per_grads)

        # 勾配クリッピング
        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        # 2) EMA ベースの全データ勾配推定（predictive: B_t を見る前の推定値）
        if self.t_step == 1:
            # 初期ステップでは EMA 推定なし → ξ̂ = 0 (Method B と同等)
            g_tilde = g_hat.copy()
        else:
            bias_correction = 1.0 - self.beta ** (self.t_step - 1)
            g_tilde = self.ema_m / bias_correction  # (N, d)

        # 3) ミニバッチノイズ推定
        xi_hat = g_hat - g_tilde  # (N, d)

        # 4) SGD 更新 + concept drift ノイズ
        epsilon = self.rng.normal(
            0.0, self.sigma_sys, size=self.particles.shape
        )
        self.particles = self.particles - self.eta_slots[:, None] * g_hat + epsilon

        # 4b) φ 遷移 (adaptive_obs 時のみ; φ 専用 rng → θ 側の乱数列は不変)
        if self.adaptive_obs:
            self.phi = propagate_phi(self.phi, self.tau_phi, self.phi_rng)

        # 5) 補正項 (Method A, 行列版) — 各粒子自身の η_i を使う (§8.4)。
        #    Woodbury/Cholesky は eta の broadcast で粒子別化 (計算量は実質不変)。
        (log_correction, rho, logcorr_nonfinite_count,
         cond_M, jitter_count) = compute_correction_method_a(
            epsilon, xi_hat, deviations, self.eta_slots,
            self.sigma_sys_sq, self.param_dim,
        )
        _t_corr = time.perf_counter()

        # 6) 対数尤度 (adaptive_obs 時は粒子別 σ^(i)=exp(φ^(i)/2) を使用)
        if self.adaptive_obs:
            if loglik_sigma_fn is None:
                raise ValueError(
                    "adaptive_obs=True には loglik_sigma_fn が必要です")
            ll = loglik_sigma_fn(self.particles, X, y, phi_to_sigma(self.phi))
        else:
            ll = loglik_fn(self.particles, X, y)
        _t_ll = time.perf_counter()

        # 7) 重み更新（累積 + 尤度 + 補正）
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        log_weights = log_prev + ll + log_correction
        self.weights = normalize_logweights(log_weights)

        # 8) 推定値の計算（重み付き平均）
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
        # 層化学習率診断 (§9): リサンプリング前の重みで評価
        eta_wmean, eta_wstd, eta_slow, eta_fast, eta_map = \
            stratified_eta_diagnostics(self.weights, self.eta_slots,
                                       self.eta_mean)
        _t_wt = time.perf_counter()

        # 9) EMA 更新（現在のバッチ勾配で更新）
        self.ema_m = self.beta * self.ema_m + (1.0 - self.beta) * g_hat

        # 10) ESS が低い場合はリサンプリング → 重みリセット + EMA 複製
        #     η_i はスロット固定 (§5): 祖先から複製しない。θ・EMA・φ は複製。
        eta_f2s = 0.0
        if ess < self.ess_resample_ratio * self.N:
            idx = systematic_resample(self.weights, self.rng)
            n_unique = int(np.unique(idx).size)
            eta_f2s = fast_to_slow_rate(idx, self.eta_slots, self.eta_mean)
            self.particles = self.particles[idx]
            self.ema_m = self.ema_m[idx]  # EMA 状態もリサンプリング
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

        # 層化学習率診断 (§9)
        self.history["eta_weighted_mean"].append(eta_wmean)
        self.history["eta_weighted_std"].append(eta_wstd)
        self.history["eta_slow_mass"].append(eta_slow)
        self.history["eta_fast_mass"].append(eta_fast)
        self.history["eta_map"].append(eta_map)
        self.history["eta_fast_to_slow_rate"].append(eta_f2s)

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
        self.history["cond_M_mean"].append(float(np.mean(cond_M)))
        self.history["cond_M_max"].append(float(np.max(cond_M)))
        self.history["jitter_count"].append(jitter_count)
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
