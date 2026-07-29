#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回帰レジームスイッチ: PF / WSPF-B / WSPF-A 粒子数別グリッドサーチ

eta × sigma_sys × prior_std (× beta for WSPF-A) を粒子数ごとに並列評価し、
各粒子数について PF / WSPF-B / WSPF-A それぞれの最良パラメータを JSON で出力する。

出力:
  outputs/regression_regime_switch/grid_search_result.json

使い方:
  python experiments/grid_search_regression_regime_switch.py
  python experiments/regression_regime_switch.py   # ← 結果を自動読み込み
"""

import sys
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.filters import ParticleFilter
from src.filters.wspf_b import WSPF_B
from src.filters.wspf_a import WSPF_A
from src.models.neural_net_regression import (
    NeuralNetRegression,
    create_regression_grad_fn,
    create_regression_loglik_fn,
    create_regression_per_sample_grad_fn,
)

# ================================================================
# 固定パラメータ（グリッドサーチ対象外）
# ================================================================
INPUT_DIM = 1
HIDDEN_DIM = 8
BATCH_SIZE = 16
TEST_SIZE = 200
MAX_GRAD_NORM = 5.0
NOISE_STD = 0.5
SEED = 1000   # パイロット(選択)用シード。評価シード range(10) と disjoint (R1-minor)

# ================================================================
# グリッドサーチ設定
# ================================================================
GRID_ETA = [0.01, 0.1,0.2,0.3,0.4,0.5]
GRID_SIGMA_SYS = [0.01, .05,  0.1,0.15,0.2]
GRID_PRIOR_STD = [0.0,0.01,0.1, 0.2, 0.3, 0.5]
GRID_BETA = [0.8, 0.9, 0.95, 0.99]  # WSPF-A 専用
N_PARTICLES_LIST = [100]
GRID_SEARCH_T = 200
GRID_EVAL_START = 100

# 出力先
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "regression_regime_switch"
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "grid_search_result.json")


# ================================================================
# データ生成
# ================================================================
def generate_regression_regime_data(model, T, batch_size, test_size, noise_std, seed):
    rng = np.random.default_rng(seed)
    param_dim = model.param_dim

    theta_star_1 = rng.normal(0.0, 0.8, size=param_dim)
    theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)

    while np.linalg.norm(theta_star_1 - theta_star_2) < 1.0:
        theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)

    regime_length = T // 5
    switch_times = [regime_length * (i + 1) for i in range(4)]
    regime_thetas = [
        theta_star_1, theta_star_2, theta_star_1, theta_star_2, theta_star_1
    ]

    within_regime_drift = 0.0005
    theta_true = np.empty((T, param_dim))
    for t in range(T):
        regime_idx = 0
        for st in switch_times:
            if t >= st:
                regime_idx += 1
        if t == 0 or t in switch_times:
            theta_true[t] = regime_thetas[regime_idx].copy()
        else:
            theta_true[t] = theta_true[t - 1] + rng.normal(
                0.0, within_regime_drift, size=param_dim
            )

    X_train, y_train = [], []
    X_test, y_test = [], []

    for t in range(T):
        theta_t = theta_true[t : t + 1]

        X = rng.normal(0.0, 1.0, size=(batch_size, model.input_dim))
        output, _, _ = model.forward(theta_t, X)
        y = output.squeeze() + rng.normal(0.0, noise_std, size=batch_size)
        X_train.append(X)
        y_train.append(y)

        Xte = rng.normal(0.0, 1.0, size=(test_size, model.input_dim))
        output_te, _, _ = model.forward(theta_t, Xte)
        yte = output_te.squeeze() + rng.normal(0.0, noise_std, size=test_size)
        X_test.append(Xte)
        y_test.append(yte)

    return X_train, y_train, X_test, y_test


# ================================================================
# ワーカー関数
# ================================================================
def _evaluate_candidate(args):
    """
    1候補を評価するワーカー（トップレベル関数で pickle 可能）
    PF は batch gradient、WSPF-B / WSPF-A は per-sample gradient を使用。

    args: (method, eta, sigma_sys, prior_std, beta, n_particles)
      beta は WSPF-A のみ使用（PF / WSPF-B では None）
    """
    method, eta, sigma_sys, prior_std, beta, n_particles = args

    model = NeuralNetRegression(
        INPUT_DIM, HIDDEN_DIM, output_dim=1, activation="tanh"
    )
    param_dim = model.param_dim

    X_train, y_train, X_test, y_test = generate_regression_regime_data(
        model, T=GRID_SEARCH_T, batch_size=BATCH_SIZE,
        test_size=TEST_SIZE, noise_std=NOISE_STD, seed=SEED,
    )

    grad_fn = create_regression_grad_fn(model, NOISE_STD)
    loglik_fn = create_regression_loglik_fn(model, NOISE_STD)
    ps_grad_fn = create_regression_per_sample_grad_fn(model, NOISE_STD)

    def clipped_grad_fn(particles, X, y):
        g = grad_fn(particles, X, y)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_GRAD_NORM / (norms + 1e-12))
        return g * scale

    common_kw = dict(
        n_particles=n_particles, param_dim=param_dim, eta=eta,
        sigma_sys=sigma_sys, prior_mean=0.0, prior_std=prior_std,
        ess_resample_ratio=0.5,
    )

    if method == "PF":
        filt = ParticleFilter(**common_kw, seed=SEED + 1)
    elif method == "WSPF-B":
        filt = WSPF_B(
            **common_kw, grad_clip_norm=MAX_GRAD_NORM, seed=SEED + 3,
        )
    else:  # WSPF-A
        filt = WSPF_A(
            **common_kw, grad_clip_norm=MAX_GRAD_NORM,
            beta=beta, seed=SEED + 5,
        )

    mse_list = []
    for t in range(GRID_SEARCH_T):
        Xt, yt = X_train[t], y_train[t]
        Xte, yte = X_test[t], y_test[t]

        if method == "PF":
            filt.step(Xt, yt, clipped_grad_fn, loglik_fn)
        else:
            filt.step(Xt, yt, ps_grad_fn, loglik_fn)

        if t >= GRID_EVAL_START:
            mu = (filt.weights[:, None] * filt.particles).sum(axis=0)
            output, _, _ = model.forward(mu.reshape(1, -1), Xte)
            pred = output.squeeze()
            mse_list.append(float(np.mean((pred - yte) ** 2)))

    mean_mse = float(np.mean(mse_list)) if mse_list else float("inf")

    return (method, n_particles, eta, sigma_sys, prior_std, beta, mean_mse)


# ================================================================
# メイン
# ================================================================
def main():
    print("=" * 70)
    print("Grid Search: Regression Regime-Switch (PF & WSPF-B & WSPF-A, per N)")
    print("=" * 70)

    n_workers = min(os.cpu_count() or 1, 48)

    # 候補生成
    base_candidates = [
        (eta, ss, ps)
        for eta in GRID_ETA
        for ss in GRID_SIGMA_SYS
        for ps in GRID_PRIOR_STD
    ]
    wspf_a_candidates = [
        (eta, ss, ps, beta)
        for eta in GRID_ETA
        for ss in GRID_SIGMA_SYS
        for ps in GRID_PRIOR_STD
        for beta in GRID_BETA
    ]

    n_base = len(base_candidates)
    n_wspf_a = len(wspf_a_candidates)
    n_per_N = n_base * 2 + n_wspf_a
    total = n_per_N * len(N_PARTICLES_LIST)

    print(f"\n  eta:         {GRID_ETA}")
    print(f"  sigma_sys:   {GRID_SIGMA_SYS}")
    print(f"  prior_std:   {GRID_PRIOR_STD}")
    print(f"  beta:        {GRID_BETA}  (WSPF-A only)")
    print(f"  N_particles: {N_PARTICLES_LIST}")
    print(f"  T:           {GRID_SEARCH_T}, eval_start: {GRID_EVAL_START}")
    print(f"  workers:     {n_workers}")
    print(f"  candidates per N: PF={n_base}, WSPF-B={n_base}, WSPF-A={n_wspf_a}, "
          f"total={n_per_N}")
    print(f"  total jobs:  {total}")
    print()

    # タスク生成
    task_args = []
    for n_p in N_PARTICLES_LIST:
        for method in ["PF", "WSPF-B"]:
            for eta, ss, ps in base_candidates:
                task_args.append((method, eta, ss, ps, None, n_p))
        for eta, ss, ps, beta in wspf_a_candidates:
            task_args.append(("WSPF-A", eta, ss, ps, beta, n_p))

    # 結果格納
    results_by_n = {
        n_p: {"PF": [], "WSPF-B": [], "WSPF-A": []}
        for n_p in N_PARTICLES_LIST
    }

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_evaluate_candidate, arg): arg
            for arg in task_args
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            method, n_particles, eta, sigma_sys, prior_std, beta, mean_mse = (
                future.result()
            )
            row = {
                "eta": eta,
                "sigma_sys": sigma_sys,
                "prior_std": prior_std,
                "mse": mean_mse,
            }
            beta_str = ""
            if method == "WSPF-A":
                row["beta"] = beta
                beta_str = f" beta={beta:.2f}"
            results_by_n[n_particles][method].append(row)

            if done_count % 50 == 0 or done_count == total:
                print(
                    f"  [{done_count:4d}/{total}] {method:5s} N={n_particles:5d} "
                    f"eta={eta:.3f} sigma_sys={sigma_sys:.3f} "
                    f"prior={prior_std:.2f}{beta_str} "
                    f"-> MSE={mean_mse:.4f}"
                )

    elapsed = time.time() - t0
    print(f"\nGrid search completed in {elapsed:.1f}s ({n_workers} workers)")

    # ソート（MSE 昇順）& JSON 構築
    output_data = {
        "grid": {
            "eta": GRID_ETA,
            "sigma_sys": GRID_SIGMA_SYS,
            "prior_std": GRID_PRIOR_STD,
            "beta": GRID_BETA,
            "n_particles": N_PARTICLES_LIST,
            "T": GRID_SEARCH_T,
            "eval_start": GRID_EVAL_START,
            "seed": SEED,
        },
        "by_n_particles": {},
    }

    for n_p in N_PARTICLES_LIST:
        for method in ["PF", "WSPF-B", "WSPF-A"]:
            results_by_n[n_p][method].sort(key=lambda r: r["mse"])

        best_pf = {
            k: results_by_n[n_p]["PF"][0][k]
            for k in ("eta", "sigma_sys", "prior_std")
        }
        best_wspf_b = {
            k: results_by_n[n_p]["WSPF-B"][0][k]
            for k in ("eta", "sigma_sys", "prior_std")
        }
        best_wspf_a = {
            k: results_by_n[n_p]["WSPF-A"][0][k]
            for k in ("eta", "sigma_sys", "prior_std", "beta")
        }

        output_data["by_n_particles"][str(n_p)] = {
            "best_pf": best_pf,
            "best_wspf_b": best_wspf_b,
            "best_wspf_a": best_wspf_a,
            "all_pf": results_by_n[n_p]["PF"],
            "all_wspf_b": results_by_n[n_p]["WSPF-B"],
            "all_wspf_a": results_by_n[n_p]["WSPF-A"],
        }

        # 結果表示
        print(f"\n{'=' * 70}")
        print(f"N = {n_p}")
        print(f"{'=' * 70}")

        for label, key in [("PF", "PF"), ("WSPF-B", "WSPF-B")]:
            print(f"\n--- {label} top 5 ---")
            print(f"  {'eta':>5s}  {'sigma_sys':>9s}  {'prior':>6s}  {'MSE':>8s}")
            print("  " + "-" * 34)
            for i, r in enumerate(results_by_n[n_p][key][:5]):
                mark = " <--" if i == 0 else ""
                print(
                    f"  {r['eta']:5.3f}  {r['sigma_sys']:9.3f}  "
                    f"{r['prior_std']:6.2f}  {r['mse']:8.4f}{mark}"
                )

        print(f"\n--- WSPF-A top 5 ---")
        print(
            f"  {'eta':>5s}  {'sigma_sys':>9s}  {'prior':>6s}  "
            f"{'beta':>6s}  {'MSE':>8s}"
        )
        print("  " + "-" * 42)
        for i, r in enumerate(results_by_n[n_p]["WSPF-A"][:5]):
            mark = " <--" if i == 0 else ""
            print(
                f"  {r['eta']:5.3f}  {r['sigma_sys']:9.3f}  "
                f"{r['prior_std']:6.2f}  {r['beta']:6.2f}  "
                f"{r['mse']:8.4f}{mark}"
            )

        print(
            f"\n  Best PF:     {best_pf}  "
            f"(MSE={results_by_n[n_p]['PF'][0]['mse']:.4f})"
        )
        print(
            f"  Best WSPF-B: {best_wspf_b}  "
            f"(MSE={results_by_n[n_p]['WSPF-B'][0]['mse']:.4f})"
        )
        print(
            f"  Best WSPF-A: {best_wspf_a}  "
            f"(MSE={results_by_n[n_p]['WSPF-A'][0]['mse']:.4f})"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fp:
        json.dump(output_data, fp, indent=2)

    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
