#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オラクル実験 — Regression (R1-5)

WSPF の厳密補正 (eq:exact_logR) は真の全データ勾配 ∇L(θ) と勾配ノイズ
共分散 Σ(θ) を必要とするが、WSPF-A/B はこれらをミニバッチから近似する
(A: EMA + rank-B Σ̂, B: スカラー ŝ)。本実験は真の生成分布が既知である
回帰タスクで、各ステップ・各粒子について ∇L と Σ を **大標本モンテカルロ**
で高精度推定し、それを厳密補正式にそのまま代入する「WSPF-oracle」を実装、
PF / WSPF-A / WSPF-B / WSPF-oracle の 4 者を比較する。

狙い(R1-5): 「補正の枠組みそのものが正しいか(oracle が最良か)」と
「A/B の劣化が推定誤差によるものか(oracle − A の差)」を分離する。

厳密補正(eq:exact_logR):
    log R = ½ log(|V_q|/|V_p|) + ½ εᵀ V_q⁻¹ ε − ½ (ε−Δμ)ᵀ V_p⁻¹ (ε−Δμ)
    Δμ = η(ĝ − ∇L),  V_p = η²Σ(θ) + Q_cd,  V_q = Q_cd = σ²_cd I
オラクルは Σ(θ)=Cov(ĝ_batch)=C(θ)/B, ∇L(θ)=E[∇ℓ] を大標本 MC で与える。
d=25 と小さいため V_p は d×d 直接反転で厳密に評価する。

出力:
  outputs/oracle_regression/
    - oracle_regression.txt / .csv
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats

from src.filters import ParticleFilter
from src.filters.wspf_a import WSPF_A
from src.filters.wspf_b import WSPF_B
import src.experiments.regression_regime_switch as RR
from src.experiments.regression_regime_switch import (
    NeuralNetRegression, generate_regression_regime_data,
    create_regression_per_sample_grad_fn, create_regression_loglik_fn,
    compute_test_mse, load_grid_search_params, compute_regime_split_metrics,
    INPUT_DIM, HIDDEN_DIM, BATCH_SIZE, TEST_SIZE, NOISE_STD,
    T, EVAL_START, N_POST_SWITCH,
    DEFAULT_ETA, DEFAULT_SIGMA_SYS, DEFAULT_PRIOR_STD, DEFAULT_BETA,
)

# オラクル設定
ORACLE_SAMPLES = 10000     # 各ステップの MC 標本数(∇L, Σ 推定)
ORACLE_CHUNK = 2500        # メモリ節約のためのチャンク分割
ORACLE_SEEDS = list(range(5))   # 計算が重いので既定は 5 seed
ORACLE_N_PARTICLES = 100

METHODS = ["PF", "WSPF-A", "WSPF-B", "WSPF-oracle"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "oracle_regression",
)


# ================================================================
# オラクル勾配統計(真の生成分布からの大標本 MC)
# ================================================================
def oracle_grad_stats(model, particles, theta_star, noise_std, M, B,
                      rng, chunk=ORACLE_CHUNK):
    """
    時刻 t の真の分布 (x~N(0,1), y=f(θ*;x)+N(0,σ²)) から大標本を引き、
    各粒子 θ^i について ∇L(θ^i)=E[∇ℓ] と Σ(θ^i)=Cov(ĝ_batch)=C(θ^i)/B を
    高精度推定する。チャンク処理で (N,M,d) の一括確保を避ける。
    """
    N, d = particles.shape
    ps_grad_fn = create_regression_per_sample_grad_fn(model, noise_std)
    sum_g = np.zeros((N, d))
    sum_ggT = np.zeros((N, d, d))
    done = 0
    while done < M:
        m = min(chunk, M - done)
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = ps_grad_fn(particles, X, y)            # (N, m, d)
        sum_g += g.sum(axis=1)
        sum_ggT += np.einsum("nmd,nme->nde", g, g)
        done += m
    grad_L = sum_g / M                              # ∇L  (N,d)
    # C = (Σggᵀ − M·mean meanᵀ)/(M−1)  → 標本(per-sample)勾配共分散
    C = (sum_ggT - M * np.einsum("nd,ne->nde", grad_L, grad_L)) / (M - 1)
    Sigma = C / B                                   # Cov(ĝ_batch) = C/B
    return grad_L, Sigma


def oracle_logR(epsilon, delta_mu, Sigma, eta, sigma_sys_sq, d):
    """厳密補正 (eq:exact_logR) を密 d×d 反転で評価する。"""
    N = epsilon.shape[0]
    c = sigma_sys_sq
    Vp = eta ** 2 * Sigma + c * np.eye(d)[None, :, :]   # (N,d,d)
    v = epsilon - delta_mu                              # (N,d)
    sign, logdet_p = np.linalg.slogdet(Vp)             # (N,)
    # v^T Vp^{-1} v
    sol = np.linalg.solve(Vp, v[:, :, None])[:, :, 0]  # (N,d)
    quad_p = np.sum(v * sol, axis=1)                   # (N,)
    quad_q = np.sum(epsilon ** 2, axis=1) / c          # (N,)
    logdet_q = d * np.log(c)
    return 0.5 * (logdet_q - logdet_p) + 0.5 * quad_q - 0.5 * quad_p


# ================================================================
# WSPF-oracle フィルタ(厳密補正 + 大標本 MC 推定)
# ================================================================
class WSPFOracle:
    def __init__(self, n_particles, param_dim, eta, sigma_sys,
                 prior_std, model, noise_std, B, M,
                 ess_resample_ratio=0.5, grad_clip_norm=None, seed=None):
        self.N = n_particles
        self.param_dim = param_dim
        self.eta = eta
        self.sigma_sys = sigma_sys
        self.sigma_sys_sq = sigma_sys ** 2
        self.ess_resample_ratio = ess_resample_ratio
        self.grad_clip_norm = grad_clip_norm
        self.model = model
        self.noise_std = noise_std
        self.B = B
        self.M = M
        self.rng = np.random.default_rng(seed)
        self.oracle_rng = np.random.default_rng(
            (seed if seed is not None else 0) + 100000)
        self.particles = self.rng.normal(0.0, prior_std,
                                         size=(self.N, param_dim))
        self.weights = np.ones(self.N) / self.N
        self.ps_grad_fn = create_regression_per_sample_grad_fn(model, noise_std)

    def step(self, X, y, theta_star_t, loglik_fn):
        theta_prev = self.particles  # θ_{t-1}

        # 実バッチのミニバッチ勾配 ĝ(θ_{t-1}; B_t)
        per = self.ps_grad_fn(theta_prev, X, y)   # (N,B,d)
        g_hat = per.mean(axis=1)                  # (N,d)
        if self.grad_clip_norm is not None:
            norms = np.linalg.norm(g_hat, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.grad_clip_norm / (norms + 1e-12))
            g_hat = g_hat * scale

        # オラクル: 真の ∇L(θ_{t-1}) と Σ(θ_{t-1})
        grad_L, Sigma = oracle_grad_stats(
            self.model, theta_prev, theta_star_t, self.noise_std,
            self.M, self.B, self.oracle_rng)

        # 伝播
        epsilon = self.rng.normal(0.0, self.sigma_sys,
                                  size=theta_prev.shape)
        self.particles = theta_prev - self.eta * g_hat + epsilon

        # 厳密補正
        delta_mu = self.eta * (g_hat - grad_L)
        log_corr = oracle_logR(epsilon, delta_mu, Sigma,
                               self.eta, self.sigma_sys_sq, self.param_dim)

        # 重み更新 (SIS 累積)
        ll = loglik_fn(self.particles, X, y)
        log_prev = np.log(np.maximum(self.weights, 1e-300))
        w = log_prev + ll + log_corr
        m = w.max()
        w = np.exp(w - m)
        s = w.sum()
        self.weights = w / s if (s > 0 and np.isfinite(s)) else np.ones(self.N) / self.N

        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.ess_resample_ratio * self.N:
            # systematic resample
            u = self.rng.random()
            positions = (u + np.arange(self.N)) / self.N
            cumsum = np.cumsum(self.weights)
            idx = np.searchsorted(cumsum, positions)
            idx = np.clip(idx, 0, self.N - 1)
            self.particles = self.particles[idx]
            self.weights = np.ones(self.N) / self.N

    def mean(self):
        return (self.weights[:, None] * self.particles).sum(axis=0)


# ================================================================
# 1 seed 実行(4 メソッド同一ストリーム)
# ================================================================
def run_single_oracle(seed, n_particles, hp):
    model = NeuralNetRegression(INPUT_DIM, HIDDEN_DIM, output_dim=1,
                                activation="tanh")
    param_dim = model.param_dim

    (X_train, y_train, X_test, y_test,
     theta_true, switch_times) = generate_regression_regime_data(
        model, T=T, batch_size=BATCH_SIZE, test_size=TEST_SIZE,
        noise_std=NOISE_STD, within_regime_drift=0.0005, seed=seed)

    loglik_fn = create_regression_loglik_fn(model, NOISE_STD)
    ps_grad_fn = create_regression_per_sample_grad_fn(model, NOISE_STD)

    def clipped_grad_fn(particles, X, y):
        # PF 用のバッチ勾配(per-sample 平均→クリップ)
        g = ps_grad_fn(particles, X, y).mean(axis=1)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        scale = np.minimum(1.0, RR.MAX_GRAD_NORM / (norms + 1e-12))
        return g * scale

    bp, bb, ba = hp["pf"], hp["wspf_b"], hp["wspf_a"]

    pf = ParticleFilter(n_particles=n_particles, param_dim=param_dim,
                        eta=bp["eta"], sigma_sys=bp["sigma_sys"],
                        prior_std=bp["prior_std"], ess_resample_ratio=0.5,
                        seed=seed + 1)
    wspf_b = WSPF_B(n_particles=n_particles, param_dim=param_dim,
                    eta=bb["eta"], sigma_sys=bb["sigma_sys"],
                    prior_std=bb["prior_std"], ess_resample_ratio=0.5,
                    grad_clip_norm=RR.MAX_GRAD_NORM, seed=seed + 3)
    wspf_a = WSPF_A(n_particles=n_particles, param_dim=param_dim,
                    eta=ba["eta"], sigma_sys=ba["sigma_sys"],
                    prior_std=ba["prior_std"], ess_resample_ratio=0.5,
                    grad_clip_norm=RR.MAX_GRAD_NORM, beta=ba["beta"],
                    seed=seed + 5)
    # oracle は WSPF-A と同じ (η,σcd,σ0) を用いる(estimate vs oracle を分離)
    oracle = WSPFOracle(n_particles=n_particles, param_dim=param_dim,
                        eta=ba["eta"], sigma_sys=ba["sigma_sys"],
                        prior_std=ba["prior_std"], model=model,
                        noise_std=NOISE_STD, B=BATCH_SIZE, M=ORACLE_SAMPLES,
                        ess_resample_ratio=0.5, grad_clip_norm=RR.MAX_GRAD_NORM,
                        seed=seed + 7)

    mse = {m: [] for m in METHODS}
    for t in range(T):
        Xt, yt = X_train[t], y_train[t]
        Xte, yte = X_test[t], y_test[t]

        pf.step(Xt, yt, clipped_grad_fn, loglik_fn)
        wspf_b.step(Xt, yt, ps_grad_fn, loglik_fn)
        wspf_a.step(Xt, yt, ps_grad_fn, loglik_fn)
        oracle.step(Xt, yt, theta_true[t], loglik_fn)

        mse["PF"].append(compute_test_mse(
            model, (pf.weights[:, None] * pf.particles).sum(0), Xte, yte))
        mse["WSPF-B"].append(compute_test_mse(
            model, (wspf_b.weights[:, None] * wspf_b.particles).sum(0), Xte, yte))
        mse["WSPF-A"].append(compute_test_mse(
            model, (wspf_a.weights[:, None] * wspf_a.particles).sum(0), Xte, yte))
        mse["WSPF-oracle"].append(compute_test_mse(model, oracle.mean(), Xte, yte))

    return {"mse": {m: np.array(v) for m, v in mse.items()},
            "switch_times": switch_times}


def _load_hp(n_p):
    """グリッド結果から各メソッドの best HP を取得(なければ default)。"""
    grid = load_grid_search_params()
    if grid is not None and str(n_p) in grid:
        e = grid[str(n_p)]
        bp = e["best_pf"]
        bb = e.get("best_wspf_b", bp)
        ba = e.get("best_wspf_a", {**bp, "beta": DEFAULT_BETA})
    else:
        base = {"eta": DEFAULT_ETA, "sigma_sys": DEFAULT_SIGMA_SYS,
                "prior_std": DEFAULT_PRIOR_STD}
        bp, bb, ba = dict(base), dict(base), {**base, "beta": DEFAULT_BETA}
    return {"pf": bp, "wspf_b": bb, "wspf_a": ba}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    n_p = ORACLE_N_PARTICLES
    hp = _load_hp(n_p)
    emit("=" * 72)
    emit("Oracle experiment — Regression (R1-5)")
    emit(f"  N={n_p}, T={T}, seeds={len(ORACLE_SEEDS)}, "
         f"MC samples/step={ORACLE_SAMPLES}")
    emit(f"  HP: PF={hp['pf']}  WSPF-B={hp['wspf_b']}  WSPF-A={hp['wspf_a']}")
    emit(f"  (oracle は WSPF-A と同じ η,σcd,σ0)")
    emit("=" * 72)

    seed_mse = {m: [] for m in METHODS}
    mse_ts = {m: [] for m in METHODS}
    switch_times = None
    n_workers = min(os.cpu_count() or 1, len(ORACLE_SEEDS))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_single_oracle, s, n_p, hp): s
                for s in ORACLE_SEEDS}
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            for m in METHODS:
                seed_mse[m].append(float(res["mse"][m][EVAL_START:].mean()))
                mse_ts[m].append(res["mse"][m])
            switch_times = res["switch_times"]
            done += 1
            emit(f"  [{done}/{len(ORACLE_SEEDS)}] done")
    emit(f"  elapsed {time.time()-t0:.0f}s")

    # ---- 集計 ----
    emit(f"\n{'='*72}\n  Oracle comparison — eval-region MSE (mean±std)\n{'='*72}")
    emit(f"  {'Method':<13s} {'MSE mean':>10s} {'MSE std':>9s} {'vs PF %':>9s}")
    pf_vals = np.asarray(seed_mse["PF"])
    csv_rows = [("method", "mse_mean", "mse_std", "improve_vs_pf_pct",
                 "paired_t_p_vs_pf")]
    for m in METHODS:
        vals = np.asarray(seed_mse[m])
        imp = 100.0 * (pf_vals.mean() - vals.mean()) / pf_vals.mean()
        pstr = "-"
        if m != "PF" and len(vals) >= 2:
            _, p = stats.ttest_rel(pf_vals, vals)
            pstr = f"{p:.4g}"
        emit(f"  {m:<13s} {vals.mean():>10.4f} {vals.std():>9.4f} "
             f"{imp:>8.2f}%   p(vs PF)={pstr}")
        csv_rows.append((m, f"{vals.mean():.6f}", f"{vals.std():.6f}",
                         f"{imp:.4f}", pstr if pstr != "-" else ""))

    # oracle − WSPF-A = 推定誤差ぶん
    o = np.asarray(seed_mse["WSPF-oracle"]).mean()
    a = np.asarray(seed_mse["WSPF-A"]).mean()
    emit(f"\n  推定誤差の寄与 (WSPF-A − oracle) = {a - o:+.4f} "
         f"({100.0*(a-o)/a:+.2f}% of WSPF-A)")

    split = compute_regime_split_metrics(mse_ts, switch_times, T,
                                         n_post_switch=N_POST_SWITCH)
    emit(f"\n  Regime-split MSE (post-switch {N_POST_SWITCH} / stable):")
    for m in METHODS:
        r = split[m]
        emit(f"  {m:<13s} post={r['post_switch_mean']:.4f}±{r['post_switch_std']:.4f}"
             f"  stable={r['stable_mean']:.4f}±{r['stable_std']:.4f}")

    txt_path = os.path.join(OUTPUT_DIR, "oracle_regression.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(OUTPUT_DIR, "oracle_regression.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
