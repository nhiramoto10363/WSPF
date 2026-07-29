#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回帰レジームスイッチシミュレーション

粒子数ごとにグリッドサーチで選択された最良ハイパーパラメータを使用して実験を実行する。

事前にグリッドサーチを実行:
  python experiments/grid_search_regression_regime_switch.py

比較:
  - SGD
  - PF
  - WSPF-B
  - WSPF-A

粒子数: 1000, 2000, 3000, 4000, 5000

出力:
  outputs/regression_regime_switch/
    - simulation_summary.csv
    - simulation_summary.txt
    - simulation_summary.tex
    - mse_vs_nparticles.png
"""

import sys
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
# ハイパーパラメータ
# ================================================================
N_PARTICLES_LIST = [100]
T = 500
NOISE_STD = 0.5
SEEDS = list(range(10))
EVAL_START = 50

INPUT_DIM = 1
HIDDEN_DIM = 8
BATCH_SIZE = 16
TEST_SIZE = 200
MAX_GRAD_NORM = 5.0
COVERAGE_ALPHA = 0.1  # 90% 予測区間
COVERAGE_Z = 1.6449   # norm.ppf(1 - COVERAGE_ALPHA / 2)
N_POST_SWITCH = 10    # レジームスイッチ直後の評価ステップ数

# スナップショット（粒子ヒストグラム・関数近似プロット）を取得する時刻
# レジーム境界: [100, 200, 300, 400]
SNAPSHOT_TIMES = [50, 105, 250, 299,300,301,302,303, 304,305, 450]

# デフォルト値（グリッドサーチ結果がない場合に使用）
DEFAULT_ETA = 0.05
DEFAULT_SIGMA_SYS = 0.05
DEFAULT_PRIOR_STD = 0.3
DEFAULT_BETA = 0.9

GRID_SEARCH_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "regression_regime_switch",
    "grid_search_result.json",
)


def load_grid_search_params():
    """粒子数別のグリッドサーチ結果を読み込む。"""
    if not os.path.exists(GRID_SEARCH_JSON):
        return None
    with open(GRID_SEARCH_JSON) as fp:
        data = json.load(fp)
    return data.get("by_n_particles", None)


# ================================================================
# データ生成
# ================================================================
def generate_regression_regime_data(
    model,
    T=500,
    batch_size=16,
    test_size=200,
    noise_std=0.5,
    within_regime_drift=0.0005,
    seed=0,
):
    rng = np.random.default_rng(seed)
    param_dim = model.param_dim

    theta_star_1 = rng.normal(0.0, 0.8, size=param_dim)
    theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)

    while np.linalg.norm(theta_star_1 - theta_star_2) < 1.0:
        theta_star_2 = rng.normal(0.0, 0.8, size=param_dim)

    regime_length = T // 5
    switch_times = [regime_length * (i + 1) for i in range(4)]
    regime_thetas = [theta_star_1, theta_star_2, theta_star_1, theta_star_2, theta_star_1]

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

    return X_train, y_train, X_test, y_test, theta_true, switch_times


# ================================================================
# テストMSE
# ================================================================
def compute_test_mse(model, theta, X, y):
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    pred = output.squeeze()
    return float(np.mean((pred - y) ** 2))


def compute_test_r2(model, theta, X, y):
    """テストR²(決定係数)を計算する。"""
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    pred = output.squeeze()
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_coverage_point(model, theta, X, y, noise_std, z):
    """点推定からの予測区間カバレッジ（SGD用）"""
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    pred = output.squeeze()
    lower = pred - z * noise_std
    upper = pred + z * noise_std
    return float(np.mean((y >= lower) & (y <= upper)))


def compute_coverage_particle(model, particles, weights, X, y, noise_std, z):
    """粒子アンサンブルからの予測区間カバレッジ（正規近似）"""
    output, _, _ = model.forward(particles, X)
    preds = output.squeeze(-1)  # (N, test_size)
    pred_mean = np.sum(weights[:, None] * preds, axis=0)
    pred_var = (
        np.sum(weights[:, None] * (preds - pred_mean[None, :]) ** 2, axis=0)
        + noise_std ** 2
    )
    lower = pred_mean - z * np.sqrt(pred_var)
    upper = pred_mean + z * np.sqrt(pred_var)
    return float(np.mean((y >= lower) & (y <= upper)))


# ================================================================
# 1条件の実験
# ================================================================
def run_single(seed, n_particles, best_pf, best_wspf_b, best_wspf_a,
               snapshot_times=None, batch_size=None):
    model = NeuralNetRegression(INPUT_DIM, HIDDEN_DIM, output_dim=1, activation="tanh")
    param_dim = model.param_dim

    if batch_size is None:
        batch_size = BATCH_SIZE

    (
        X_train,
        y_train,
        X_test,
        y_test,
        theta_true,
        switch_times,
    ) = generate_regression_regime_data(
        model,
        T=T,
        batch_size=batch_size,
        test_size=TEST_SIZE,
        noise_std=NOISE_STD,
        within_regime_drift=0.0005,
        seed=seed,
    )

    grad_fn = create_regression_grad_fn(model, NOISE_STD)
    loglik_fn = create_regression_loglik_fn(model, NOISE_STD)
    ps_grad_fn = create_regression_per_sample_grad_fn(model, NOISE_STD)

    def clipped_grad_fn(particles, X, y):
        g = grad_fn(particles, X, y)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_GRAD_NORM / (norms + 1e-12))
        return g * scale

    # SGD は PF の eta / prior_std を使用
    sgd_eta = best_pf["eta"]
    sgd_prior = best_pf["prior_std"]

    rng_sgd = np.random.default_rng(seed + 10)
    theta_sgd = rng_sgd.normal(0.0, sgd_prior, size=param_dim)

    # PF
    pf = ParticleFilter(
        n_particles=n_particles, param_dim=param_dim, eta=best_pf["eta"],
        sigma_sys=best_pf["sigma_sys"], prior_mean=0.0,
        prior_std=best_pf["prior_std"],
        ess_resample_ratio=0.5, seed=seed + 1,
    )

    # WSPF-B
    wspf_b = WSPF_B(
        n_particles=n_particles, param_dim=param_dim, eta=best_wspf_b["eta"],
        sigma_sys=best_wspf_b["sigma_sys"], prior_mean=0.0,
        prior_std=best_wspf_b["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM, seed=seed + 3,
    )

    # WSPF-A
    wspf_a = WSPF_A(
        n_particles=n_particles, param_dim=param_dim, eta=best_wspf_a["eta"],
        sigma_sys=best_wspf_a["sigma_sys"], prior_mean=0.0,
        prior_std=best_wspf_a["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM,
        beta=best_wspf_a["beta"], seed=seed + 5,
    )

    mse = {"SGD": [], "PF": [], "WSPF-A": [], "WSPF-B": []}
    r2 = {"SGD": [], "PF": [], "WSPF-A": [], "WSPF-B": []}
    coverage = {"SGD": [], "PF": [], "WSPF-A": [], "WSPF-B": []}
    pred_var_methods = ["PF", "WSPF-A", "WSPF-B"]
    pred_var_x0 = {m: [] for m in pred_var_methods}
    x0_pt = np.array([[0.0]])  # x=0 の評価点
    snapshot_set = set(snapshot_times) if snapshot_times else set()
    snapshots = []

    for t in range(T):
        Xt, yt = X_train[t], y_train[t]
        Xte, yte = X_test[t], y_test[t]

        # SGD
        g = clipped_grad_fn(theta_sgd.reshape(1, -1), Xt, yt).squeeze()
        theta_sgd = theta_sgd - sgd_eta * g
        mu_sgd = theta_sgd.copy()

        # PF / WSPF: step() の返り値 = リサンプリング前の重み付き平均を使う
        # (論文の点推定定義に一致。リサンプリング後の一様重み再計算は不可)
        mu_pf = pf.step(Xt, yt, clipped_grad_fn, loglik_fn)
        mu_wspf_b = wspf_b.step(Xt, yt, ps_grad_fn, loglik_fn)
        mu_wspf_a = wspf_a.step(Xt, yt, ps_grad_fn, loglik_fn)

        mse["SGD"].append(compute_test_mse(model, mu_sgd, Xte, yte))
        mse["PF"].append(compute_test_mse(model, mu_pf, Xte, yte))
        mse["WSPF-A"].append(compute_test_mse(model, mu_wspf_a, Xte, yte))
        mse["WSPF-B"].append(compute_test_mse(model, mu_wspf_b, Xte, yte))

        r2["SGD"].append(compute_test_r2(model, mu_sgd, Xte, yte))
        r2["PF"].append(compute_test_r2(model, mu_pf, Xte, yte))
        r2["WSPF-A"].append(compute_test_r2(model, mu_wspf_a, Xte, yte))
        r2["WSPF-B"].append(compute_test_r2(model, mu_wspf_b, Xte, yte))

        # カバレッジ
        coverage["SGD"].append(
            compute_coverage_point(model, mu_sgd, Xte, yte, NOISE_STD, COVERAGE_Z)
        )
        coverage["PF"].append(
            compute_coverage_particle(model, pf.particles, pf.weights,
                                      Xte, yte, NOISE_STD, COVERAGE_Z)
        )
        coverage["WSPF-B"].append(
            compute_coverage_particle(model, wspf_b.particles, wspf_b.weights,
                                      Xte, yte, NOISE_STD, COVERAGE_Z)
        )
        coverage["WSPF-A"].append(
            compute_coverage_particle(model, wspf_a.particles, wspf_a.weights,
                                      Xte, yte, NOISE_STD, COVERAGE_Z)
        )

        # x=0 における粒子予測分散
        for m_name, m_particles, m_weights in [
            ("PF", pf.particles, pf.weights),
            ("WSPF-A", wspf_a.particles, wspf_a.weights),
            ("WSPF-B", wspf_b.particles, wspf_b.weights),
        ]:
            out_x0, _, _ = model.forward(m_particles, x0_pt)
            y_pred_x0 = out_x0.squeeze()  # (N,)
            y_mean_x0 = np.sum(m_weights * y_pred_x0)
            var_x0 = float(np.sum(m_weights * (y_pred_x0 - y_mean_x0) ** 2))
            pred_var_x0[m_name].append(var_x0)

        # スナップショット
        if t in snapshot_set:
            snapshots.append({
                "t": t,
                "theta_true": theta_true[t].copy(),
                "X_train": Xt.copy(),
                "y_train": yt.copy(),
                "sgd_theta": mu_sgd.copy(),
                "pf_particles": pf.particles.copy(),
                "pf_weights": pf.weights.copy(),
                "wspf_b_particles": wspf_b.particles.copy(),
                "wspf_b_weights": wspf_b.weights.copy(),
                "wspf_a_particles": wspf_a.particles.copy(),
                "wspf_a_weights": wspf_a.weights.copy(),
            })

    for k in mse:
        mse[k] = np.array(mse[k])
        r2[k] = np.array(r2[k])
        coverage[k] = np.array(coverage[k])
    for k in pred_var_methods:
        pred_var_x0[k] = np.array(pred_var_x0[k])

    histories = {
        "PF": pf.get_history(),
        "WSPF-A": wspf_a.get_history(),
        "WSPF-B": wspf_b.get_history(),
    }

    result = dict(
        T=T,
        param_dim=param_dim,
        switch_times=switch_times,
        mse=mse,
        r2=r2,
        coverage=coverage,
        pred_var_x0=pred_var_x0,
        histories=histories,
    )
    if snapshots:
        result["snapshots"] = snapshots
    return result


# ================================================================
# テーブル保存
# ================================================================
def save_result_tables(output_dir, all_results, n_particles_list, methods,
                       hp_by_n):
    """結果テーブルを TXT / CSV / LaTeX で保存する。"""
    csv_path = os.path.join(output_dir, "simulation_summary.csv")
    txt_path = os.path.join(output_dir, "simulation_summary.txt")
    tex_path = os.path.join(output_dir, "simulation_summary.tex")

    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n_particles,method,mse_mean,mse_std,rho_bar,ess_mean,coverage\n")
        for n_p in n_particles_list:
            for row in all_results[n_p]:
                f.write(
                    f"{n_p},{row['method']},"
                    f"{row['mean']:.6f},{row['std']:.6f},"
                    f"{row['rho_bar']:.6f},{row['ess_mean']:.2f},"
                    f"{row['cov_mean']:.4f}\n"
                )

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Simulation summary: Regime-switching regression\n")
        f.write("Test MSE (mean +/- std) with best hyperparameters per N\n")
        f.write("=" * 80 + "\n\n")

        for n_p in n_particles_list:
            best_pf, best_wspf_b, best_wspf_a = hp_by_n[n_p]
            f.write(f"N = {n_p}\n")
            f.write(f"  PF:     eta={best_pf['eta']}, "
                    f"sigma_sys={best_pf['sigma_sys']}, "
                    f"prior_std={best_pf['prior_std']}\n")
            f.write(f"  WSPF-B: eta={best_wspf_b['eta']}, "
                    f"sigma_sys={best_wspf_b['sigma_sys']}, "
                    f"prior_std={best_wspf_b['prior_std']}\n")
            f.write(f"  WSPF-A: eta={best_wspf_a['eta']}, "
                    f"sigma_sys={best_wspf_a['sigma_sys']}, "
                    f"prior_std={best_wspf_a['prior_std']}, "
                    f"beta={best_wspf_a['beta']}\n")
            f.write(
                f"  {'Method':<10s} {'MSE (mean +/- std)':>22s} "
                f"{'rho_bar':>8s} {'ESS':>10s} {'Coverage':>10s}\n"
            )
            f.write(f"  {'-' * 68}\n")
            for row in all_results[n_p]:
                ess_str = f"{row['ess_mean']:10.1f}" if row["ess_mean"] > 0 else "         -"
                f.write(
                    f"  {row['method']:<10s} "
                    f"{row['mean']:.4f}+/-{row['std']:.4f} "
                    f"{row['rho_bar']:>8.4f} {ess_str} "
                    f"{row['cov_mean']:10.4f}\n"
                )
            f.write("\n")

    # LaTeX
    n_cols = len(n_particles_list)
    col_spec = "@{}l" + "c" * n_cols + "@{}"

    lookup = {}
    for n_p in n_particles_list:
        for row in all_results[n_p]:
            if row["method"] not in lookup:
                lookup[row["method"]] = {}
            lookup[row["method"]][n_p] = row

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Regime-switching regression: test MSE (mean $\\pm$ std) "
            "and ESS with best hyperparameters selected per particle count.}\n"
        )
        f.write("\\label{tab:regression_regime_switch}\n")
        f.write("\\small\n")
        f.write("\\renewcommand{\\arraystretch}{1.15}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        header = "Method"
        for n_p in n_particles_list:
            header += f" & $N={n_p}$"
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")

        # MSE rows
        f.write("\\multicolumn{" + str(n_cols + 1) + "}{l}"
                "{\\textit{Test MSE (mean $\\pm$ std)}} \\\\\n")
        for m in methods:
            line = m
            for n_p in n_particles_list:
                r = lookup[m][n_p]
                line += f" & ${r['mean']:.4f} \\pm {r['std']:.4f}$"
            f.write(line + " \\\\\n")

        # ESS rows
        f.write("\\midrule\n")
        f.write("\\multicolumn{" + str(n_cols + 1) + "}{l}"
                "{\\textit{ESS (mean)}} \\\\\n")
        ess_methods_tex = ["PF", "WSPF-A", "WSPF-B"]
        for m in ess_methods_tex:
            line = m
            for n_p in n_particles_list:
                r = lookup[m][n_p]
                line += f" & ${r['ess_mean']:.1f}$"
            f.write(line + " \\\\\n")

        # Coverage rows
        f.write("\\midrule\n")
        cov_pct = int((1 - COVERAGE_ALPHA) * 100)
        f.write("\\multicolumn{" + str(n_cols + 1) + "}{l}"
                "{\\textit{" + str(cov_pct) + "\\% PI coverage}} \\\\\n")
        for m in methods:
            line = m
            for n_p in n_particles_list:
                r = lookup[m][n_p]
                line += f" & ${r['cov_mean']:.4f}$"
            f.write(line + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return csv_path, txt_path, tex_path


# ================================================================
# 図保存
# ================================================================
def moving_average(x, w):
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def save_mse_timeseries_plot(output_dir, mse_timeseries, methods,
                             switch_times, eval_start, n_particles):
    """MSE時系列プロットを保存する。

    Parameters
    ----------
    mse_timeseries : dict[str, ndarray]
        手法名 -> shape (T,) のシード平均MSE時系列
    switch_times : list[int]
        レジームスイッチのタイムステップ
    """
    out_path = os.path.join(
        output_dir,
        f"regression_regime_switch_timeseries_N{n_particles}.png",
    )

    step = len(next(iter(mse_timeseries.values())))
    plot_interval = 25

    colors = {
        "SGD": "#888888",
        "PF": "#2196F3",
        "WSPF-A": "#4CAF50",
        "WSPF-B": "#E91E63",
    }
    markers = {
        "SGD": "x",
        "PF": "o",
        "WSPF-A": "D",
        "WSPF-B": "^",
    }

    plt.figure(figsize=(10, 5.5))

    t_idx = np.arange(0, step, plot_interval)
    for m in methods:
        plt.plot(
            t_idx,
            mse_timeseries[m][t_idx],
            color=colors[m],
            marker=markers[m],
            label=m,
            linewidth=1.2,
            markersize=4,
        )
    title = f"Regime-switching regression: test MSE (every {plot_interval} steps, N={n_particles})"

    for i, st in enumerate(switch_times):
        plt.axvline(
            st, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
            label="regime switch" if i == 0 else None,
        )

    plt.axvline(
        eval_start, color="black", linestyle="--", linewidth=0.8,
        alpha=0.6, label="eval start",
    )
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Test MSE", fontsize=18)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc="upper left", borderaxespad=0.2,
               handletextpad=0.4, labelspacing=0.3, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_r2_timeseries_plot(output_dir, r2_timeseries, methods,
                            switch_times, eval_start, n_particles):
    """R²時系列プロットを保存する。

    Parameters
    ----------
    r2_timeseries : dict[str, ndarray]
        手法名 -> shape (T,) のシード平均R²時系列
    switch_times : list[int]
        レジームスイッチのタイムステップ
    """
    out_path = os.path.join(
        output_dir,
        f"regression_regime_switch_r2_timeseries_N{n_particles}.png",
    )

    step = len(next(iter(r2_timeseries.values())))

    colors = {
        "SGD": "#888888",
        "PF": "#2196F3",
        "WSPF-A": "#4CAF50",
        "WSPF-B": "#E91E63",
    }

    plt.figure(figsize=(10, 5.5))

    t_arr = np.arange(step)
    for m in methods:
        plt.plot(
            t_arr, r2_timeseries[m],
            color=colors[m], label=m, linewidth=1.0, alpha=0.85,
        )
    title = f"Regime-switching regression: test $R^2$ (N={n_particles})"

    for i, st in enumerate(switch_times):
        plt.axvline(
            st, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
            label="regime switch" if i == 0 else None,
        )

    plt.axvline(
        eval_start, color="black", linestyle="--", linewidth=0.8,
        alpha=0.6, label="eval start",
    )
    plt.xlabel("Step")
    plt.ylabel("$R^2$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_rho_timeseries_plot(output_dir, rho_timeseries, methods,
                             switch_times, eval_start, n_particles):
    r"""$\bar{\rho}$ 時系列プロットを保存する。

    Parameters
    ----------
    rho_timeseries : dict[str, ndarray]
        手法名 -> shape (T,) のシード平均 rho_mean 時系列
        (WSPF-A, WSPF-B のみ)
    switch_times : list[int]
        レジームスイッチのタイムステップ
    """
    out_path = os.path.join(
        output_dir,
        f"regression_regime_switch_rho_timeseries_N{n_particles}.png",
    )

    step = len(next(iter(rho_timeseries.values())))

    colors = {
        "WSPF-A": "#4CAF50",
        "WSPF-B": "#E91E63",
    }

    plt.figure(figsize=(10, 5.5))

    t_arr = np.arange(step)
    for m in methods:
        if m in rho_timeseries:
            plt.plot(
                t_arr, rho_timeseries[m],
                color=colors[m], label=m, linewidth=1.0, alpha=0.85,
            )
    title = (
        r"Regime-switching regression: $\bar{\rho}$ "
        f"(N={n_particles})"
    )

    for i, st in enumerate(switch_times):
        plt.axvline(
            st, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
            label="regime switch" if i == 0 else None,
        )

    plt.axvline(
        eval_start, color="black", linestyle="--", linewidth=0.8,
        alpha=0.6, label="eval start",
    )
    plt.xlabel("Step")
    plt.ylabel(r"$\bar{\rho}$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_coverage_timeseries_plot(output_dir, cov_timeseries, methods,
                                  switch_times, eval_start, n_particles,
                                  alpha=COVERAGE_ALPHA):
    """カバレッジ時系列プロットを保存する。

    Parameters
    ----------
    cov_timeseries : dict[str, ndarray]
        手法名 -> shape (T,) のシード平均カバレッジ時系列
    switch_times : list[int]
        レジームスイッチのタイムステップ
    alpha : float
        予測区間の有意水準（1-alpha が名目カバレッジ）
    """
    out_path = os.path.join(
        output_dir,
        f"regression_regime_switch_coverage_timeseries_N{n_particles}.png",
    )

    step = len(next(iter(cov_timeseries.values())))
    window = min(50, step // 10) if step > 10 else 1

    colors = {
        "SGD": "#888888",
        "PF": "#2196F3",
        "WSPF-A": "#4CAF50",
        "WSPF-B": "#E91E63",
    }

    plt.figure(figsize=(10, 5.5))

    if window > 1:
        t_ma = np.arange(window - 1, step)
        for m in methods:
            plt.plot(
                t_ma,
                moving_average(cov_timeseries[m], window),
                color=colors[m], label=m, linewidth=1.4,
            )
        title = (f"Regime-switching regression: "
                 f"{int((1 - alpha) * 100)}% PI coverage "
                 f"(MA w={window}, N={n_particles})")
    else:
        t_arr = np.arange(step)
        for m in methods:
            plt.plot(
                t_arr, cov_timeseries[m],
                color=colors[m], label=m, linewidth=1.0, alpha=0.85,
            )
        title = (f"Regime-switching regression: "
                 f"{int((1 - alpha) * 100)}% PI coverage "
                 f"(N={n_particles})")

    # 名目カバレッジの水平線
    plt.axhline(
        1 - alpha, color="black", linestyle="-", linewidth=1.0,
        alpha=0.5, label=f"nominal ({int((1 - alpha) * 100)}%)",
    )

    for i, st in enumerate(switch_times):
        plt.axvline(
            st, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
            label="regime switch" if i == 0 else None,
        )

    plt.axvline(
        eval_start, color="black", linestyle="--", linewidth=0.8,
        alpha=0.6, label="eval start",
    )
    plt.xlabel("Step")
    plt.ylabel("Coverage")
    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_pred_var_timeseries_plot(output_dir, pred_var_timeseries, methods,
                                  switch_times, eval_start, n_particles):
    r"""x=0 における粒子予測分散の時系列プロットを保存する。

    Parameters
    ----------
    pred_var_timeseries : dict[str, ndarray]
        手法名 -> shape (T,) のシード平均予測分散時系列
    switch_times : list[int]
        レジームスイッチのタイムステップ
    """
    out_path = os.path.join(
        output_dir,
        f"regression_regime_switch_pred_var_x0_N{n_particles}.png",
    )

    step = len(next(iter(pred_var_timeseries.values())))

    colors = {
        "PF": "#2196F3",
        "WSPF-A": "#4CAF50",
        "WSPF-B": "#E91E63",
    }

    plt.figure(figsize=(10, 5.5))

    t_arr = np.arange(step)
    for m in methods:
        if m in pred_var_timeseries:
            plt.plot(
                t_arr, pred_var_timeseries[m],
                color=colors[m], label=m, linewidth=1.0, alpha=0.85,
            )
    title = (
        r"Regime-switching regression: prediction variance at $x=0$ "
        f"(N={n_particles})"
    )

    for i, st in enumerate(switch_times):
        plt.axvline(
            st, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
            label="regime switch" if i == 0 else None,
        )

    plt.axvline(
        eval_start, color="black", linestyle="--", linewidth=0.8,
        alpha=0.6, label="eval start",
    )
    plt.xlabel("Step")
    plt.ylabel(r"$\mathrm{Var}[\hat{y}(x\!=\!0)]$")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_ess_timeseries_plot(output_dir, ess_timeseries, methods,
                             switch_times, eval_start, n_particles):
    """ESS時系列プロットを保存する。

    Parameters
    ----------
    ess_timeseries : dict[str, ndarray]
        手法名 -> shape (T,) のシード平均ESS時系列
    switch_times : list[int]
        レジームスイッチのタイムステップ
    """
    out_path = os.path.join(
        output_dir,
        f"regression_regime_switch_ess_timeseries_N{n_particles}.png",
    )

    step = len(next(iter(ess_timeseries.values())))

    colors = {
        "PF": "#2196F3",
        "WSPF-A": "#4CAF50",
        "WSPF-B": "#E91E63",
    }

    plt.figure(figsize=(10, 5.5))

    t_arr = np.arange(step)
    for m in methods:
        if m in ess_timeseries:
            plt.plot(
                t_arr, ess_timeseries[m],
                color=colors[m], label=m, linewidth=1.0, alpha=0.85,
            )
    title = f"Regime-switching regression: ESS (N={n_particles})"

    for i, st in enumerate(switch_times):
        plt.axvline(
            st, color="red", linestyle=":", linewidth=1.0, alpha=0.6,
            label="regime switch" if i == 0 else None,
        )

    plt.axvline(
        eval_start, color="black", linestyle="--", linewidth=0.8,
        alpha=0.6, label="eval start",
    )
    plt.xlabel("Step")
    plt.ylabel("ESS")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_snapshot_plots(output_dir, model, snapshots, switch_times,
                        n_particles, noise_std=NOISE_STD):
    """特定時点での粒子ヒストグラムと関数近似プロットを保存する。

    各時点について、PF / WSPF-B / WSPF-A の 3 手法 × 2 行（関数近似、
    予測ヒストグラム at x=-1, 0, 1）のサブプロットを生成する。

    Parameters
    ----------
    model : NeuralNetRegression
        ネットワークモデル
    snapshots : list[dict]
        run_single() で取得したスナップショットのリスト
    switch_times : list[int]
        レジームスイッチ時刻
    """
    x_grid = np.linspace(-2, 2, 300).reshape(-1, 1)
    x_hist_pts = np.array([[-1.0], [0.0], [1.0]])  # ヒストグラム評価点

    particle_methods = [
        ("PF",     "pf_particles",     "pf_weights",     "#2196F3"),
        ("WSPF-B", "wspf_b_particles", "wspf_b_weights", "#E91E63"),
        ("WSPF-A", "wspf_a_particles", "wspf_a_weights", "#4CAF50"),
    ]

    saved = []
    for snap in snapshots:
        t = snap["t"]
        theta_true = snap["theta_true"]

        # 真の関数
        y_true_grid, _, _ = model.forward(theta_true.reshape(1, -1), x_grid)
        y_true_grid = y_true_grid.squeeze()

        # SGD 予測
        y_sgd_grid, _, _ = model.forward(snap["sgd_theta"].reshape(1, -1), x_grid)
        y_sgd_grid = y_sgd_grid.squeeze()

        # 真の関数の y 範囲を基準に表示範囲を決定
        y_min_true, y_max_true = y_true_grid.min(), y_true_grid.max()
        y_margin = (y_max_true - y_min_true) * 0.5 + noise_std * 3
        y_lim = (y_min_true - y_margin, y_max_true + y_margin)

        # レジーム判定
        regime_idx = sum(1 for st in switch_times if t >= st)

        # --- 事前計算: 全手法の粒子予測・ヒストグラムデータ ---
        hist_colors = ["#42A5F5", "#66BB6A", "#EF5350"]
        x_labels = ["x=-1", "x=0", "x=1"]
        n_bins = 25

        pre = []  # 各手法の事前計算結果
        for col, (mname, pkey, wkey, color) in enumerate(particle_methods):
            particles = snap[pkey]
            weights = snap[wkey]

            out_grid, _, _ = model.forward(particles, x_grid)
            y_particles = out_grid.squeeze(-1)
            y_mean = np.sum(weights[:, None] * y_particles, axis=0)
            y_var = np.sum(
                weights[:, None] * (y_particles - y_mean[None, :]) ** 2, axis=0
            )
            y_std = np.sqrt(y_var)

            out_hp, _, _ = model.forward(particles, x_hist_pts)
            y_hp = out_hp.squeeze(-1)  # (N, 3)

            out_true_hp, _, _ = model.forward(
                theta_true.reshape(1, -1), x_hist_pts
            )
            y_true_hp = out_true_hp.squeeze()  # (3,)

            pre.append(dict(
                mname=mname, color=color,
                particles=particles, weights=weights,
                y_particles=y_particles, y_mean=y_mean, y_std=y_std,
                y_hp=y_hp, y_true_hp=y_true_hp,
            ))

        # ヒストグラム軸を全手法で統一するため、事前に密度・範囲を計算
        hist_ymax = 0.0
        hist_xmin = np.inf
        hist_xmax = -np.inf
        for p in pre:
            for j in range(3):
                counts, edges = np.histogram(
                    p["y_hp"][:, j], bins=n_bins, weights=p["weights"],
                    density=True,
                )
                hist_ymax = max(hist_ymax, counts.max())
                hist_xmin = min(hist_xmin, p["y_hp"][:, j].min(),
                                p["y_true_hp"][j])
                hist_xmax = max(hist_xmax, p["y_hp"][:, j].max(),
                                p["y_true_hp"][j])
        hist_ylim = hist_ymax * 1.12
        hist_xmargin = (hist_xmax - hist_xmin) * 0.08
        hist_xlim = (hist_xmin - hist_xmargin, hist_xmax + hist_xmargin)

        # x=0 のみ版の軸統一
        hist_ymax_x0 = 0.0
        hist_xmin_x0 = np.inf
        hist_xmax_x0 = -np.inf
        for p in pre:
            counts_x0, _ = np.histogram(
                p["y_hp"][:, 1], bins=n_bins, weights=p["weights"],
                density=True,
            )
            hist_ymax_x0 = max(hist_ymax_x0, counts_x0.max())
            hist_xmin_x0 = min(hist_xmin_x0, p["y_hp"][:, 1].min(),
                               p["y_true_hp"][1])
            hist_xmax_x0 = max(hist_xmax_x0, p["y_hp"][:, 1].max(),
                               p["y_true_hp"][1])
        hist_ylim_x0 = hist_ymax_x0 * 1.12
        xm0 = (hist_xmax_x0 - hist_xmin_x0) * 0.08
        hist_xlim_x0 = (hist_xmin_x0 - xm0, hist_xmax_x0 + xm0)

        # ============================================================
        # メインプロット (2行×3列: 関数近似 + ヒストグラム x=-1,0,1)
        # ============================================================
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Snapshot at t={t}  (regime {regime_idx + 1},  N={n_particles})",
            fontsize=26, fontweight="bold",
        )

        for col, p in enumerate(pre):
            mname = p["mname"]
            color = p["color"]
            weights = p["weights"]
            y_particles = p["y_particles"]
            y_mean = p["y_mean"]
            y_std = p["y_std"]
            y_hp = p["y_hp"]
            y_true_hp = p["y_true_hp"]

            # --- Row 1: 関数近似プロット ---
            ax = axes[0, col]

            n_show = min(50, p["particles"].shape[0])
            top_idx = np.argsort(weights)[-n_show:]
            for i in top_idx:
                ax.plot(
                    x_grid.squeeze(), y_particles[i],
                    color=color, alpha=0.07, linewidth=0.5,
                )

            ax.fill_between(
                x_grid.squeeze(), y_mean - 2 * y_std, y_mean + 2 * y_std,
                color=color, alpha=0.12, label=r"$\pm 2\sigma$",
            )
            ax.fill_between(
                x_grid.squeeze(), y_mean - y_std, y_mean + y_std,
                color=color, alpha=0.22, label=r"$\pm 1\sigma$",
            )

            ax.plot(
                x_grid.squeeze(), y_true_grid, "k-",
                linewidth=2.2, label="True", zorder=4,
            )
            ax.plot(
                x_grid.squeeze(), y_mean, color=color,
                linewidth=2, label=f"{mname} mean", zorder=3,
            )
            ax.plot(
                x_grid.squeeze(), y_sgd_grid, color="#888888",
                linewidth=1.5, linestyle="--", label="SGD", zorder=2,
            )

            ax.scatter(
                snap["X_train"].squeeze(), snap["y_train"],
                s=18, c="black", alpha=0.5, zorder=5, label="train data",
            )

            ax.set_title(mname, fontsize=22)
            ax.set_xlabel("x", fontsize=18)
            ax.set_ylabel("y", fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
            ax.set_xlim(-2, 2)
            ax.set_ylim(y_lim)
            ax.legend(fontsize=14, loc="upper left")
            ax.grid(True, alpha=0.3)

            # --- Row 2: 予測ヒストグラム (x=-1, 0, 1) ---
            ax2 = axes[1, col]

            for j in range(3):
                ax2.hist(
                    y_hp[:, j], bins=n_bins, weights=weights, density=True,
                    color=hist_colors[j], alpha=0.45, edgecolor="white",
                    label=x_labels[j],
                )
                ax2.axvline(
                    y_true_hp[j], color=hist_colors[j],
                    linewidth=2, linestyle="-",
                )
                wm = np.sum(weights * y_hp[:, j])
                ax2.axvline(
                    wm, color=hist_colors[j],
                    linewidth=1.5, linestyle="--",
                )

            ax2.set_xlim(hist_xlim)
            ax2.set_ylim(0, hist_ylim)
            ax2.set_title(f"{mname}: prediction histogram", fontsize=20)
            ax2.set_xlabel("y prediction", fontsize=18)
            ax2.set_ylabel("Density", fontsize=18)
            ax2.tick_params(axis="both", labelsize=16)

            ax2.plot([], [], "k-", linewidth=2, label="true f(x)")
            ax2.plot([], [], "k--", linewidth=1.5, label="weighted mean")
            ax2.legend(fontsize=14)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(
            output_dir, f"snapshot_t{t}_N{n_particles}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path)

        # ============================================================
        # 下段のみプロット (1行×3列: ヒストグラム x=0 のみ)
        # ============================================================
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4.5))
        fig2.suptitle(
            f"Prediction histogram at x=0,  t={t}  "
            f"(regime {regime_idx + 1},  N={n_particles})",
            fontsize=24, fontweight="bold",
        )

        for col, p in enumerate(pre):
            mname = p["mname"]
            color = p["color"]
            weights = p["weights"]
            y_hp = p["y_hp"]
            y_true_hp = p["y_true_hp"]

            ax = axes2[col]
            ax.hist(
                y_hp[:, 1], bins=n_bins, weights=weights, density=True,
                color=color, alpha=0.6, edgecolor="white",
            )
            ax.axvline(
                y_true_hp[1], color="black",
                linewidth=2, linestyle="-", label=f"true: {y_true_hp[1]:.2f}",
            )
            wm = float(np.sum(weights * y_hp[:, 1]))
            ax.axvline(
                wm, color=color,
                linewidth=2, linestyle="--", label=f"mean: {wm:.2f}",
            )

            ax.set_xlim(hist_xlim_x0)
            ax.set_ylim(0, hist_ylim_x0)
            ax.set_title(mname, fontsize=22)
            ax.set_xlabel("y prediction (x=0)", fontsize=18)
            ax.set_ylabel("Density", fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
            ax.legend(fontsize=15)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path2 = os.path.join(
            output_dir, f"snapshot_hist_x0_t{t}_N{n_particles}.png"
        )
        plt.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path2)

        # ============================================================
        # 上段のみプロット (1行×3列: 関数近似)
        # ============================================================
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig3.suptitle(
            f"Function approximation,  t={t}  "
            f"(regime {regime_idx + 1},  N={n_particles})",
            fontsize=24, fontweight="bold",
        )

        for col, p in enumerate(pre):
            mname = p["mname"]
            color = p["color"]
            weights = p["weights"]
            y_particles = p["y_particles"]
            y_mean = p["y_mean"]
            y_std = p["y_std"]

            ax = axes3[col]

            n_show = min(50, p["particles"].shape[0])
            top_idx = np.argsort(weights)[-n_show:]
            for i in top_idx:
                ax.plot(
                    x_grid.squeeze(), y_particles[i],
                    color=color, alpha=0.07, linewidth=0.5,
                )

            ax.fill_between(
                x_grid.squeeze(), y_mean - 2 * y_std, y_mean + 2 * y_std,
                color=color, alpha=0.12, label=r"$\pm 2\sigma$",
            )
            ax.fill_between(
                x_grid.squeeze(), y_mean - y_std, y_mean + y_std,
                color=color, alpha=0.22, label=r"$\pm 1\sigma$",
            )

            ax.plot(
                x_grid.squeeze(), y_true_grid, "k-",
                linewidth=2.2, label="True", zorder=4,
            )
            ax.plot(
                x_grid.squeeze(), y_mean, color=color,
                linewidth=2, label=f"{mname} mean", zorder=3,
            )
            ax.plot(
                x_grid.squeeze(), y_sgd_grid, color="#888888",
                linewidth=1.5, linestyle="--", label="SGD", zorder=2,
            )

            ax.scatter(
                snap["X_train"].squeeze(), snap["y_train"],
                s=18, c="black", alpha=0.5, zorder=5, label="train data",
            )

            ax.set_title(mname, fontsize=22)
            ax.set_xlabel("x", fontsize=18)
            ax.set_ylabel("y", fontsize=18)
            ax.tick_params(axis="both", labelsize=16)
            ax.set_xlim(-2, 2)
            ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.3)

        # パネルごとの凡例は冗長で領域を占有するため、図全体で 1 つだけ
        # 下部に共有配置する。バンド・mean は各パネルの色とタイトルで判別
        # できるので、共有凡例では中立色のプロキシハンドルを用いる。
        shared_handles = [
            Line2D([0], [0], color="k", lw=2.2, label="True"),
            Line2D([0], [0], color="0.4", lw=2,
                   label="Ensemble mean (per-panel color)"),
            Patch(facecolor="0.4", alpha=0.30, label=r"$\pm 1\sigma$"),
            Patch(facecolor="0.4", alpha=0.15, label=r"$\pm 2\sigma$"),
            Line2D([0], [0], color="#888888", lw=1.5, ls="--", label="SGD"),
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor="black", markeredgecolor="black",
                   markersize=6, alpha=0.5, label="train data"),
        ]
        fig3.legend(
            handles=shared_handles, loc="lower center", ncol=6,
            fontsize=18, frameon=False, bbox_to_anchor=(0.5, 0.0),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 0.93])
        out_path3 = os.path.join(
            output_dir, f"snapshot_func_t{t}_N{n_particles}.png"
        )
        plt.savefig(out_path3, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path3)

    return saved


def save_weight_boxplots(output_dir, snapshots, switch_times, n_particles):
    """各スナップショット時刻における粒子重み分布の箱ひげ図を保存する。

    時刻ごとに 1×3 (PF / WSPF-B / WSPF-A) の箱ひげ図を個別ファイルとして生成。

    Parameters
    ----------
    snapshots : list[dict]
        run_single() で取得したスナップショット
    switch_times : list[int]
        レジームスイッチ時刻
    n_particles : int
        粒子数
    """
    particle_methods = [
        ("PF",     "pf_weights",     "#2196F3"),
        ("WSPF-B", "wspf_b_weights", "#E91E63"),
        ("WSPF-A", "wspf_a_weights", "#4CAF50"),
    ]
    n_methods = len(particle_methods)
    if len(snapshots) == 0:
        return []

    uniform_w = 1.0 / n_particles

    saved = []
    for snap in snapshots:
        t = snap["t"]
        regime_idx = sum(1 for st in switch_times if t >= st)

        fig, axes = plt.subplots(1, n_methods, figsize=(14, 5), sharey=True)
        fig.suptitle(
            f"Particle weight distribution,  t={t}  "
            f"(regime {regime_idx + 1},  N={n_particles})",
            fontsize=24, fontweight="bold",
        )

        for col, (mname, wkey, color) in enumerate(particle_methods):
            w = snap[wkey]
            ax = axes[col]
            ess = 1.0 / np.sum(w ** 2)

            # 0 の重みを除外（対数スケールで表示不可のため）
            w_pos = w[w > 0]

            # 箱ひげ図
            bp = ax.boxplot(
                [w_pos], patch_artist=True, widths=0.5, showfliers=True,
                flierprops=dict(marker=".", markersize=4, alpha=0.5,
                                markerfacecolor=color),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.5)

            # 一様重みの参照線
            ax.axhline(uniform_w, color="gray", linestyle="--",
                        linewidth=1.0, alpha=0.7, label=f"uniform (1/N={uniform_w:.4f})")

            ax.set_yscale("log")
            ax.set_title(f"{mname}  (ESS={ess:.1f})", fontsize=22)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=16)
            ax.legend(fontsize=14, loc="upper right")
            ax.grid(True, axis="y", alpha=0.3, which="both")

        axes[0].set_ylabel("Weight (log scale)", fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = os.path.join(
            output_dir, f"snapshot_weight_box_t{t}_N{n_particles}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path)

    return saved


def save_mse_vs_nparticles_plot(output_dir, all_results, n_particles_list):
    """粒子数 vs MSE のプロットを保存する。"""
    plot_path = os.path.join(output_dir, "mse_vs_nparticles.png")

    particle_methods = ["PF", "WSPF-A", "WSPF-B"]
    colors = {"PF": "#2196F3", "WSPF-A": "#4CAF50", "WSPF-B": "#E91E63"}
    markers = {"PF": "o", "WSPF-A": "D", "WSPF-B": "^"}

    lookup = {}
    for n_p in n_particles_list:
        for row in all_results[n_p]:
            if row["method"] not in lookup:
                lookup[row["method"]] = {}
            lookup[row["method"]][n_p] = row

    plt.figure(figsize=(8, 5))
    for m in particle_methods:
        means = [lookup[m][n_p]["mean"] for n_p in n_particles_list]
        stds = [lookup[m][n_p]["std"] for n_p in n_particles_list]
        plt.errorbar(
            n_particles_list, means, yerr=stds,
            marker=markers[m], color=colors[m],
            label=m, linewidth=1.5, markersize=6, capsize=3,
        )

    # SGD baseline
    sgd_mean = lookup["SGD"][n_particles_list[0]]["mean"]
    plt.axhline(
        sgd_mean, color="#888888", linestyle="--", linewidth=1.2,
        alpha=0.7, label="SGD",
    )

    plt.xlabel("Number of particles")
    plt.ylabel("Test MSE")
    plt.title(
        r"Regime-switching regression: MSE vs. $N$ "
        r"(best hyperparameters per $N$)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(n_particles_list)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_path


# ================================================================
# レジームスイッチ直後 / 安定期 分離指標
# ================================================================
def compute_regime_split_metrics(mse_ts_dict, switch_times, T,
                                  n_post_switch=N_POST_SWITCH):
    """レジームスイッチ後の初期ステップと安定期のMSEを分離計算する。

    Parameters
    ----------
    mse_ts_dict : dict[str, list[ndarray]]
        手法名 -> シードごとのMSE時系列リスト (各要素 shape (T,))
    switch_times : list[int]
        レジームスイッチのタイムステップ
    T : int
        総ステップ数
    n_post_switch : int
        スイッチ直後の評価ステップ数

    Returns
    -------
    dict[str, dict]
        手法名 -> {post_switch_mean, post_switch_std,
                    stable_mean, stable_std}
    """
    boundaries = switch_times + [T]
    results = {}
    for m, ts_list in mse_ts_dict.items():
        arr = np.array(ts_list)  # (n_seeds, T)
        post_mses = []
        stable_mses = []
        for s in range(arr.shape[0]):
            post_vals = []
            stable_vals = []
            for i, st in enumerate(switch_times):
                end = boundaries[i + 1]
                post_end = min(st + n_post_switch, end)
                post_vals.extend(arr[s, st:post_end].tolist())
                if post_end < end:
                    stable_vals.extend(arr[s, post_end:end].tolist())
            post_mses.append(np.mean(post_vals))
            stable_mses.append(
                np.mean(stable_vals) if stable_vals else float("nan")
            )
        results[m] = {
            "post_switch_mean": float(np.nanmean(post_mses)),
            "post_switch_std": float(np.nanstd(post_mses)),
            "stable_mean": float(np.nanmean(stable_mses)),
            "stable_std": float(np.nanstd(stable_mses)),
        }
    return results


def save_regime_split_tables(output_dir, split_results, n_particles_list,
                              methods, n_post_switch=N_POST_SWITCH):
    """レジームスイッチ直後と安定期の分離MSEテーブルを保存する。"""
    csv_path = os.path.join(output_dir, "regime_split_summary.csv")
    txt_path = os.path.join(output_dir, "regime_split_summary.txt")
    tex_path = os.path.join(output_dir, "regime_split_summary.tex")

    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n_particles,method,"
                "post_switch_mse_mean,post_switch_mse_std,"
                "stable_mse_mean,stable_mse_std\n")
        for n_p in n_particles_list:
            for m in methods:
                r = split_results[n_p][m]
                f.write(
                    f"{n_p},{m},"
                    f"{r['post_switch_mean']:.6f},"
                    f"{r['post_switch_std']:.6f},"
                    f"{r['stable_mean']:.6f},"
                    f"{r['stable_std']:.6f}\n"
                )

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(
            f"Regime-split MSE: post-switch (first {n_post_switch} steps) "
            f"vs stable (step {n_post_switch + 1} ~ next switch)\n"
        )
        f.write("=" * 80 + "\n\n")
        for n_p in n_particles_list:
            f.write(f"N = {n_p}\n")
            f.write(
                f"  {'Method':<10s} "
                f"{'Post-switch (mean+/-std)':>26s}  "
                f"{'Stable (mean+/-std)':>26s}\n"
            )
            f.write(f"  {'-' * 66}\n")
            for m in methods:
                r = split_results[n_p][m]
                f.write(
                    f"  {m:<10s} "
                    f"{r['post_switch_mean']:8.4f}+/-{r['post_switch_std']:.4f}  "
                    f"{r['stable_mean']:8.4f}+/-{r['stable_std']:.4f}\n"
                )
            f.write("\n")

    # LaTeX
    n_cols = len(n_particles_list)
    col_spec = "@{}l" + "c" * n_cols + "@{}"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            f"\\caption{{Regime-split test MSE (mean $\\pm$ std): "
            f"post-switch (first {n_post_switch} steps) vs.\\ "
            f"stable (step {n_post_switch + 1} onward).}}\n"
        )
        f.write("\\label{tab:regime_split}\n")
        f.write("\\small\n")
        f.write("\\renewcommand{\\arraystretch}{1.15}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        header = "Method"
        for n_p in n_particles_list:
            header += f" & $N={n_p}$"
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")

        # Post-switch rows
        f.write(
            "\\multicolumn{" + str(n_cols + 1) + "}{l}"
            "{\\textit{Post-switch MSE (first "
            + str(n_post_switch) + " steps)}} \\\\\n"
        )
        for m in methods:
            line = m
            for n_p in n_particles_list:
                r = split_results[n_p][m]
                line += (
                    f" & ${r['post_switch_mean']:.4f}"
                    f" \\pm {r['post_switch_std']:.4f}$"
                )
            f.write(line + " \\\\\n")

        # Stable rows
        f.write("\\midrule\n")
        f.write(
            "\\multicolumn{" + str(n_cols + 1) + "}{l}"
            "{\\textit{Stable MSE (step "
            + str(n_post_switch + 1) + " $\\sim$ next switch)}} \\\\\n"
        )
        for m in methods:
            line = m
            for n_p in n_particles_list:
                r = split_results[n_p][m]
                line += (
                    f" & ${r['stable_mean']:.4f}"
                    f" \\pm {r['stable_std']:.4f}$"
                )
            f.write(line + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return csv_path, txt_path, tex_path


# ================================================================
# メイン
# ================================================================
def main():
    print("=" * 80)
    print("Regression Regime-Switch: particle count comparison")
    print("=" * 80)
    print(f"  T={T}")
    print(f"  noise_std:    {NOISE_STD}")
    print(f"  N_particles:  {N_PARTICLES_LIST}")
    print(f"  seeds:        {SEEDS}")
    print()

    # -------- ハイパーパラメータ読み込み --------
    gs_data = load_grid_search_params()

    hp_by_n = {}
    # メイン実験の HP ローダーも厳格化(無言フォールバック廃止)。
    # グリッド未実行・N 欠損・旧/欠損キーは明示的に失敗させる。
    if gs_data is None:
        raise RuntimeError(
            f"グリッド結果が見つかりません({GRID_SEARCH_JSON})。先に "
            "grid_search_regression_regime_switch.py を実行してください。")
    print(f"Loaded grid search results from {GRID_SEARCH_JSON}")
    for n_p in N_PARTICLES_LIST:
        key = str(n_p)
        if key not in gs_data:
            raise KeyError(f"グリッド JSON に N={n_p} がありません。")
        entry = gs_data[key]
        for k in ("best_pf", "best_wspf_b", "best_wspf_a"):
            if k not in entry:
                raise KeyError(
                    f"グリッド JSON に '{k}' がありません(旧キー best_cpf 等の"
                    "可能性)。グリッドを再生成してください。")
        best_pf = entry["best_pf"]
        best_wspf_b = entry["best_wspf_b"]
        best_wspf_a = entry["best_wspf_a"]
        if "beta" not in best_wspf_a:
            raise KeyError("best_wspf_a に 'beta' がありません。")
        hp_by_n[n_p] = (best_pf, best_wspf_b, best_wspf_a)
        print(
            f"  N={n_p}:\n"
            f"    PF:     eta={best_pf['eta']}, "
            f"sigma_sys={best_pf['sigma_sys']}, "
            f"prior_std={best_pf['prior_std']}\n"
            f"    WSPF-B: eta={best_wspf_b['eta']}, "
            f"sigma_sys={best_wspf_b['sigma_sys']}, "
            f"prior_std={best_wspf_b['prior_std']}\n"
            f"    WSPF-A: eta={best_wspf_a['eta']}, "
            f"sigma_sys={best_wspf_a['sigma_sys']}, "
            f"prior_std={best_wspf_a['prior_std']}, "
            f"beta={best_wspf_a['beta']}"
        )

    # -------- 保存先 --------
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "regression_regime_switch"
    )
    os.makedirs(output_dir, exist_ok=True)

    # -------- 全 (n_particles, seed) を並列実行 --------
    methods = ["SGD", "PF", "WSPF-A", "WSPF-B"]
    task_list = [
        (n_p, seed) for n_p in N_PARTICLES_LIST for seed in SEEDS
    ]
    total = len(task_list)
    n_workers = min(os.cpu_count() or 1, 48)
    print(f"\n  workers: {n_workers}")
    print(f"  total jobs: {total}")
    print()

    # 結果格納
    raw_mse = {n_p: {m: [] for m in methods} for n_p in N_PARTICLES_LIST}
    raw_rho = {n_p: [] for n_p in N_PARTICLES_LIST}
    # 時系列プロット用: シードごとの時系列を保存
    mse_ts = {n_p: {m: [] for m in methods} for n_p in N_PARTICLES_LIST}
    r2_ts = {n_p: {m: [] for m in methods} for n_p in N_PARTICLES_LIST}
    rho_methods = ["WSPF-A", "WSPF-B"]
    rho_ts = {n_p: {m: [] for m in rho_methods} for n_p in N_PARTICLES_LIST}
    ess_methods = ["PF", "WSPF-A", "WSPF-B"]
    ess_ts = {n_p: {m: [] for m in ess_methods} for n_p in N_PARTICLES_LIST}
    raw_ess = {n_p: {m: [] for m in ess_methods} for n_p in N_PARTICLES_LIST}
    # --- 縮退診断ログ (R1-8, R1-9, R2-2): シードごとの時系列を蓄積 ---
    diag_methods = ["PF", "WSPF-A", "WSPF-B"]
    diag_scalar_keys = ["entropy", "max_weight", "spread_trace",
                        "unique_particles", "resampled",
                        # 計算コスト (R1-11)
                        "t_step", "t_grad", "t_correction", "t_loglik",
                        "t_weight", "t_resample", "sample_grad_evals"]
    diag_ts = {n_p: {m: {k: [] for k in diag_scalar_keys}
                     for m in diag_methods} for n_p in N_PARTICLES_LIST}
    # WSPF-A/B 共通の ρ 系 + WSPF-A 固有の条件数(存在するキーのみ保存)
    rho_diag_keys = ["rho", "rho_clip_count", "logcorr_nonfinite_count",
                     "cond_M_mean", "cond_M_max"]
    rho_diag_ts = {n_p: {m: {k: [] for k in rho_diag_keys}
                         for m in rho_methods} for n_p in N_PARTICLES_LIST}
    cov_ts = {n_p: {m: [] for m in methods} for n_p in N_PARTICLES_LIST}
    raw_cov = {n_p: {m: [] for m in methods} for n_p in N_PARTICLES_LIST}
    pred_var_methods = ["PF", "WSPF-A", "WSPF-B"]
    pred_var_ts = {n_p: {m: [] for m in pred_var_methods} for n_p in N_PARTICLES_LIST}
    switch_times_by_n = {}

    t_total = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for n_p, seed in task_list:
            best_pf, best_wspf_b, best_wspf_a = hp_by_n[n_p]
            future = executor.submit(
                run_single, seed, n_p, best_pf, best_wspf_b, best_wspf_a,
            )
            futures[future] = (n_p, seed)

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            n_p, seed = futures[future]
            result = future.result()

            for m in methods:
                raw_mse[n_p][m].append(
                    result["mse"][m][EVAL_START:].mean()
                )
                mse_ts[n_p][m].append(result["mse"][m])
                r2_ts[n_p][m].append(result["r2"][m])

            for m in rho_methods:
                rho_ts[n_p][m].append(
                    result["histories"][m]["rho_mean"]
                )

            for m in ess_methods:
                ess_arr = result["histories"][m]["ess"]
                ess_ts[n_p][m].append(ess_arr)
                raw_ess[n_p][m].append(ess_arr[EVAL_START:].mean())

            # 縮退診断ログ (R1-8, R1-9, R2-2)
            for m in diag_methods:
                h = result["histories"][m]
                for k in diag_scalar_keys:
                    diag_ts[n_p][m][k].append(h[k])
            for m in rho_methods:
                h = result["histories"][m]
                for k in rho_diag_keys:
                    if k in h:  # cond_M_* は WSPF-A のみ
                        rho_diag_ts[n_p][m][k].append(h[k])

            for m in methods:
                cov_ts[n_p][m].append(result["coverage"][m])
                raw_cov[n_p][m].append(result["coverage"][m][EVAL_START:].mean())

            for m in pred_var_methods:
                pred_var_ts[n_p][m].append(result["pred_var_x0"][m])

            if n_p not in switch_times_by_n:
                switch_times_by_n[n_p] = result["switch_times"]

            rho_wspfb = result["histories"]["WSPF-B"]["rho_mean"]
            raw_rho[n_p].append(rho_wspfb[EVAL_START:].mean())

            if done_count % 10 == 0 or done_count == total:
                print(f"  [{done_count:4d}/{total}] completed")

    elapsed_total = time.time() - t_total
    print(f"\nTotal time: {elapsed_total:.0f}s")

    # -------- サマリー構築 --------
    all_results = {}
    for n_p in N_PARTICLES_LIST:
        rows = []
        rho_bar = float(np.mean(raw_rho[n_p]))
        for m in methods:
            vals = raw_mse[n_p][m]
            ess_mean = float(np.mean(raw_ess[n_p][m])) if m in ess_methods else 0.0
            cov_mean = float(np.mean(raw_cov[n_p][m]))
            rows.append({
                "method": m,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "rho_bar": rho_bar if m == "WSPF-B" else 0.0,
                "ess_mean": ess_mean,
                "cov_mean": cov_mean,
            })
        all_results[n_p] = rows

    # -------- コンソール表示 --------
    for n_p in N_PARTICLES_LIST:
        print(f"\n{'=' * 60}")
        print(f"  TEST MSE (mean +/- std), N={n_p}")
        print(f"{'=' * 60}")
        print(
            f"  {'Method':<10s} {'MSE (mean +/- std)':>22s} "
            f"{'rho_bar':>8s} {'ESS':>10s} {'Coverage':>10s}"
        )
        print(f"  {'-' * 68}")
        for row in all_results[n_p]:
            rho_str = f"{row['rho_bar']:8.4f}" if row["rho_bar"] > 0 else "       -"
            ess_str = f"{row['ess_mean']:10.1f}" if row["ess_mean"] > 0 else "         -"
            print(
                f"  {row['method']:<10s} "
                f"{row['mean']:.4f}+/-{row['std']:.4f} "
                f"{rho_str} {ess_str} "
                f"{row['cov_mean']:10.4f}"
            )

    # -------- サマリー表示 --------
    print(f"\n{'=' * 80}")
    print("Summary: MSE across particle counts")
    print(f"{'=' * 80}")
    print(f"\n{'Method':<10s}", end="")
    for n_p in N_PARTICLES_LIST:
        print(f" {'N=' + str(n_p):>10s}", end="")
    print()
    print("-" * (10 + 11 * len(N_PARTICLES_LIST)))
    for m in methods:
        print(f"{m:<10s}", end="")
        for n_p in N_PARTICLES_LIST:
            for row in all_results[n_p]:
                if row["method"] == m:
                    print(f" {row['mean']:>10.4f}", end="")
                    break
        print()

    # -------- レジーム分離指標 --------
    split_results = {}
    for n_p in N_PARTICLES_LIST:
        split_results[n_p] = compute_regime_split_metrics(
            mse_ts[n_p], switch_times_by_n[n_p], T,
            n_post_switch=N_POST_SWITCH,
        )

    print(f"\n{'=' * 80}")
    print(
        f"Regime-split MSE: post-switch (first {N_POST_SWITCH} steps) "
        f"vs stable (step {N_POST_SWITCH + 1} ~ next switch)"
    )
    print(f"{'=' * 80}")
    for n_p in N_PARTICLES_LIST:
        print(f"\n  N = {n_p}")
        print(
            f"  {'Method':<10s} "
            f"{'Post-switch (mean+/-std)':>26s}  "
            f"{'Stable (mean+/-std)':>26s}"
        )
        print(f"  {'-' * 66}")
        for m in methods:
            r = split_results[n_p][m]
            print(
                f"  {m:<10s} "
                f"{r['post_switch_mean']:8.4f}+/-{r['post_switch_std']:.4f}  "
                f"{r['stable_mean']:8.4f}+/-{r['stable_std']:.4f}"
            )

    # -------- ファイル保存 --------
    saved_files = []

    # 縮退診断ログを npz で保存 (R1-8, R1-9, R2-2)
    for n_p in N_PARTICLES_LIST:
        diag_npz = os.path.join(output_dir, f"diagnostics_N{n_p}.npz")
        sd = {"eval_start": EVAL_START, "T": T,
              "switch_times": np.asarray(switch_times_by_n[n_p])}
        for m in diag_methods:
            pfx = m.replace(" ", "_").replace("-", "_").lower()
            for k in diag_scalar_keys:
                # (n_seeds, T)
                sd[f"{pfx}_{k}"] = np.array(diag_ts[n_p][m][k])
        for m in rho_methods:
            pfx = m.replace(" ", "_").replace("-", "_").lower()
            # rho: (n_seeds, T, N),  *_count / cond_M_*: (n_seeds, T)
            for k in rho_diag_keys:
                vals = rho_diag_ts[n_p][m][k]
                if vals:  # 空(そのメソッドに無いキー)はスキップ
                    sd[f"{pfx}_{k}"] = np.array(vals)
        np.savez_compressed(diag_npz, **sd)
        saved_files.append(diag_npz)
        print(f"Diagnostics (npz) saved to {diag_npz}")

    csv_p, txt_p, tex_p = save_result_tables(
        output_dir, all_results, N_PARTICLES_LIST, methods, hp_by_n,
    )
    saved_files.extend([csv_p, txt_p, tex_p])

    rs_csv, rs_txt, rs_tex = save_regime_split_tables(
        output_dir, split_results, N_PARTICLES_LIST, methods,
    )
    saved_files.extend([rs_csv, rs_txt, rs_tex])

    plot_p = save_mse_vs_nparticles_plot(
        output_dir, all_results, N_PARTICLES_LIST,
    )
    saved_files.append(plot_p)

    # 時系列プロット
    for n_p in N_PARTICLES_LIST:
        avg_mse_ts = {}
        avg_r2_ts = {}
        for m in methods:
            avg_mse_ts[m] = np.mean(mse_ts[n_p][m], axis=0)
            avg_r2_ts[m] = np.mean(r2_ts[n_p][m], axis=0)
        ts_p = save_mse_timeseries_plot(
            output_dir, avg_mse_ts, methods,
            switch_times_by_n[n_p], EVAL_START, n_p,
        )
        saved_files.append(ts_p)
        r2_p = save_r2_timeseries_plot(
            output_dir, avg_r2_ts, methods,
            switch_times_by_n[n_p], EVAL_START, n_p,
        )
        saved_files.append(r2_p)
        avg_rho_ts = {}
        for m in rho_methods:
            avg_rho_ts[m] = np.mean(rho_ts[n_p][m], axis=0)
        rho_p = save_rho_timeseries_plot(
            output_dir, avg_rho_ts, rho_methods,
            switch_times_by_n[n_p], EVAL_START, n_p,
        )
        saved_files.append(rho_p)
        avg_ess_ts = {}
        for m in ess_methods:
            avg_ess_ts[m] = np.mean(ess_ts[n_p][m], axis=0)
        ess_p = save_ess_timeseries_plot(
            output_dir, avg_ess_ts, ess_methods,
            switch_times_by_n[n_p], EVAL_START, n_p,
        )
        saved_files.append(ess_p)
        avg_cov_ts = {}
        for m in methods:
            avg_cov_ts[m] = np.mean(cov_ts[n_p][m], axis=0)
        cov_p = save_coverage_timeseries_plot(
            output_dir, avg_cov_ts, methods,
            switch_times_by_n[n_p], EVAL_START, n_p,
        )
        saved_files.append(cov_p)
        avg_pred_var_ts = {}
        for m in pred_var_methods:
            avg_pred_var_ts[m] = np.mean(pred_var_ts[n_p][m], axis=0)
        pv_p = save_pred_var_timeseries_plot(
            output_dir, avg_pred_var_ts, pred_var_methods,
            switch_times_by_n[n_p], EVAL_START, n_p,
        )
        saved_files.append(pv_p)

    # -------- スナップショットプロット --------
    # seed=0 で 1 回再実行してスナップショットを取得
    n_p_snap = N_PARTICLES_LIST[0]
    best_pf_s, best_wspf_b_s, best_wspf_a_s = hp_by_n[n_p_snap]
    print(f"\nGenerating snapshot plots (seed=0, N={n_p_snap}) ...")
    snap_result = run_single(
        SEEDS[0], n_p_snap, best_pf_s, best_wspf_b_s, best_wspf_a_s,
        snapshot_times=SNAPSHOT_TIMES,
    )
    snap_model = NeuralNetRegression(
        INPUT_DIM, HIDDEN_DIM, output_dim=1, activation="tanh",
    )
    snap_paths = save_snapshot_plots(
        output_dir, snap_model, snap_result["snapshots"],
        snap_result["switch_times"], n_p_snap,
    )
    saved_files.extend(snap_paths)

    wb_paths = save_weight_boxplots(
        output_dir, snap_result["snapshots"],
        snap_result["switch_times"], n_p_snap,
    )
    saved_files.extend(wb_paths)

    print(f"\n{'=' * 80}")
    print("Saved files:")
    print(f"{'=' * 80}")
    for fp in saved_files:
        print(f"  {fp}")


if __name__ == "__main__":
    main()
