#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email 二値分類実験

粒子数ごとにグリッドサーチで選択された最良ハイパーパラメータを使用して実験を実行する。

事前にグリッドサーチを実行:
  python experiments/grid_search_email.py

比較:
  - NoChange
  - SGD
  - PF
  - WSPF-B (Method B)
  - WSPF-A (Method A)

粒子数: N_PARTICLES_LIST で指定

出力:
  outputs/email_binary/
    - email_binary_results.txt
    - email_binary_results.csv
    - email_binary_results.tex
    - email_binary_accuracy_timeseries_N{n}.png
    - email_binary_f1_timeseries_N{n}.png
    - email_binary_loglik_timeseries_N{n}.png
    - email_binary_accuracy_vs_nparticles.png
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import EmailDataLoader
from src.filters import ParticleFilter
from src.filters.wspf_b import WSPF_B
from src.filters.wspf_a import WSPF_A
from src.models.neural_net import (
    NeuralNetModel,
    create_nn_grad_fn,
    create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)

# ================================================================
# ハイパーパラメータ
# ================================================================
#N_PARTICLES_LIST = [5, 10, 50]
N_PARTICLES_LIST = [100]

HIDDEN_DIM = 16
BATCH_SIZE = 16
MAX_GRAD_NORM = 5.0
SEED = 42
TEST_SIZE = 32
PCA_DIM = 50

# リーク除去(R1-13): PCA 学習とハイパラ選択は warm-up/val 区間(期1-2 = 0-599)
# で行い(grid_search_email.py と一致させること)、性能報告は評価区間
# (期3-5 = 600-1499)の test window のみで集計する。フィルタ自体は step0 から
# 因果的にオンライン学習させる(warm-up 区間の学習はリークではない)。
PCA_FIT_END = 600
REPORT_START = 600

# デフォルト値（グリッドサーチ結果がない場合に使用）
DEFAULT_ETA = 0.05
DEFAULT_SIGMA_SYS = 0.01
DEFAULT_PRIOR_STD = 0.1
DEFAULT_BETA = 0.9

# データパス
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Email",
    "email_data.arff",
)

GRID_SEARCH_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "email_binary",
    "grid_search_result.json",
)

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "email_binary",
)

# ドリフトポイント (サンプルインデックス)
DRIFT_POINTS = [300, 600, 900, 1200]

NO_CHANGE_EPS = 1e-3


def load_grid_search_params():
    """粒子数別のグリッドサーチ結果を読み込む。"""
    if not os.path.exists(GRID_SEARCH_JSON):
        return None
    with open(GRID_SEARCH_JSON) as fp:
        data = json.load(fp)
    return data.get("by_n_particles", None)


# ================================================================
# 評価関数
# ================================================================
def compute_accuracy(model, theta, X, y):
    """二値分類の精度"""
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    pred = (output.squeeze() > 0.5).astype(np.float64)
    return float(np.mean(pred == y))


def compute_f1(model, theta, X, y):
    """positive クラス (y=1) の F1 スコア"""
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    pred = (output.squeeze() > 0.5).astype(np.float64)
    return compute_f1_from_pred(pred, y)


def compute_loglik(model, theta, X, y):
    """テストデータでの対数尤度"""
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    p = np.clip(output.squeeze(), 1e-10, 1 - 1e-10)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return float(ll.mean())


def compute_accuracy_from_pred(pred, y):
    return float(np.mean(pred == y))


def compute_f1_from_pred(pred, y, pos_label=1.0):
    tp = np.sum((pred == pos_label) & (y == pos_label))
    fp = np.sum((pred == pos_label) & (y != pos_label))
    fn = np.sum((pred != pos_label) & (y == pos_label))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_loglik_from_hard_pred(pred, y, eps=NO_CHANGE_EPS):
    p = np.where(pred > 0.5, 1.0 - eps, eps).astype(np.float64)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return float(ll.mean())


def clip_gradients(grad, max_norm):
    """勾配クリッピング"""
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    return grad * scale


def moving_average(x, w):
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ================================================================
# 単一粒子数での実験実行
# ================================================================
def run_experiment(n_particles, model, loader, grad_fn, loglik_fn, ps_grad_fn,
                   best_pf, best_wspf_b, best_wspf_a, sgd_eta, sgd_prior, param_dim,
                   seed=None):
    """指定された粒子数でオンライン学習実験を実行する。

    seed は初期粒子・リサンプリング・SGD 初期化にのみ効く(データストリームと
    PCA fit は固定)。複数シード実験(R1-14)では seed を変えて呼ぶ。
    """
    seed_base = SEED if seed is None else seed

    rng_sgd = np.random.default_rng(seed_base + 10)
    theta_sgd = rng_sgd.normal(0.0, sgd_prior, size=param_dim)

    pf = ParticleFilter(
        n_particles=n_particles, param_dim=param_dim, eta=best_pf["eta"],
        sigma_sys=best_pf["sigma_sys"], prior_mean=0.0, prior_std=best_pf["prior_std"],
        ess_resample_ratio=0.5, seed=seed_base + 1,
    )
    wspf_b = WSPF_B(
        n_particles=n_particles, param_dim=param_dim, eta=best_wspf_b["eta"],
        sigma_sys=best_wspf_b["sigma_sys"], prior_mean=0.0, prior_std=best_wspf_b["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM, seed=seed_base + 3,
    )
    wspf_a = WSPF_A(
        n_particles=n_particles, param_dim=param_dim, eta=best_wspf_a["eta"],
        sigma_sys=best_wspf_a["sigma_sys"], prior_mean=0.0, prior_std=best_wspf_a["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM,
        beta=best_wspf_a["beta"], seed=seed_base + 5,
    )

    methods = ["NoChange", "SGD", "PF", "WSPF-A", "WSPF-B"]

    acc = {m: [] for m in methods}
    f1 = {m: [] for m in methods}
    ll = {m: [] for m in methods}
    sample_positions = []

    n = loader.n_samples
    total_steps = (n - TEST_SIZE) // BATCH_SIZE

    last_y_obs = 0.0
    step = 0
    pos = 0
    t0 = time.time()

    # 較正評価用: 報告区間の予測確率とラベルを収集 (R2-5)
    prob_hist = {m: [] for m in ["SGD", "PF", "WSPF-A", "WSPF-B"]}
    label_hist = []

    while pos + BATCH_SIZE + TEST_SIZE <= n:
        X_train = loader.X[pos: pos + BATCH_SIZE]
        y_train = loader.y[pos: pos + BATCH_SIZE]
        X_test = loader.X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        y_test = loader.y[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]

        sample_positions.append(pos)
        pos += BATCH_SIZE

        # --- 評価 ---
        pred_nc = np.full_like(y_test, last_y_obs, dtype=np.float64)
        acc["NoChange"].append(compute_accuracy_from_pred(pred_nc, y_test))
        f1["NoChange"].append(compute_f1_from_pred(pred_nc, y_test, pos_label=1.0))
        ll["NoChange"].append(compute_loglik_from_hard_pred(pred_nc, y_test))

        mu_sgd = theta_sgd.copy()
        mu_pf = (pf.weights[:, None] * pf.particles).sum(axis=0)
        mu_wspf_b = (wspf_b.weights[:, None] * wspf_b.particles).sum(axis=0)
        mu_wspf_a = (wspf_a.weights[:, None] * wspf_a.particles).sum(axis=0)

        mus = {
            "SGD": mu_sgd,
            "PF": mu_pf,
            "WSPF-A": mu_wspf_a,
            "WSPF-B": mu_wspf_b,
        }

        for m in ["SGD", "PF", "WSPF-A", "WSPF-B"]:
            acc[m].append(compute_accuracy(model, mus[m], X_test, y_test))
            f1[m].append(compute_f1(model, mus[m], X_test, y_test))
            ll[m].append(compute_loglik(model, mus[m], X_test, y_test))

        # 較正用: 報告区間(pos は te0 に加算済み)の予測確率とラベル
        if pos >= REPORT_START:
            for m in ["SGD", "PF", "WSPF-A", "WSPF-B"]:
                out_m, _, _ = model.forward(mus[m].reshape(1, -1), X_test)
                p = np.clip(np.asarray(out_m).reshape(-1), 1e-10, 1 - 1e-10)
                prob_hist[m].append(p)
            label_hist.append(np.asarray(y_test, dtype=np.float64).reshape(-1))

        # --- 学習 ---
        g = grad_fn(theta_sgd.reshape(1, -1), X_train, y_train).squeeze()
        theta_sgd = theta_sgd - sgd_eta * g

        pf.step(X_train, y_train, grad_fn, loglik_fn)
        wspf_b.step(X_train, y_train, ps_grad_fn, loglik_fn)
        wspf_a.step(X_train, y_train, ps_grad_fn, loglik_fn)

        last_y_obs = float(y_train[-1])

        step += 1
        if step % 20 == 0:
            elapsed = time.time() - t0
            print(
                f"  Step {step:4d}/{total_steps} ({elapsed:5.1f}s) - "
                f"Acc: NC={acc['NoChange'][-1]:.3f}  SGD={acc['SGD'][-1]:.3f}  "
                f"PF={acc['PF'][-1]:.3f}  CA={acc['WSPF-A'][-1]:.3f}  "
                f"CB={acc['WSPF-B'][-1]:.3f}"
            )

    elapsed = time.time() - t0
    print(f"  Completed {step} steps in {elapsed:.1f}s")

    for m in methods:
        acc[m] = np.array(acc[m], dtype=np.float64)
        f1[m] = np.array(f1[m], dtype=np.float64)
        ll[m] = np.array(ll[m], dtype=np.float64)

    sample_positions = np.array(sample_positions)
    # リーク除去(R1-13): 報告は評価区間(期3-5)の test window のみ。
    # test 区間が REPORT_START 以降に始まる window を集計対象とする
    # (ドリフト@600 をまたぐ境界 window は自然に除外される)。
    test_starts = sample_positions + BATCH_SIZE
    eval_mask = test_starts >= REPORT_START
    report_idx = np.nonzero(eval_mask)[0]
    # eval_start = 報告区間の先頭 window インデックス(表示・プロット用)
    eval_start = int(report_idx[0]) if report_idx.size else len(sample_positions)

    result_rows = []
    for m in methods:
        result_rows.append({
            "method": m,
            "accuracy": float(acc[m][eval_mask].mean()),
            "f1": float(f1[m][eval_mask].mean()),
            "loglik": float(ll[m][eval_mask].mean()),
        })

    # PF 系フィルタの診断ログ (R1-8, R1-9, R2-2)
    diagnostics = {
        "PF": pf.get_history(),
        "WSPF-A": wspf_a.get_history(),
        "WSPF-B": wspf_b.get_history(),
    }
    # 較正評価用の予測確率とラベル (R2-5)
    diagnostics["_calib"] = {
        "probs": {m: (np.concatenate(prob_hist[m]) if prob_hist[m]
                      else np.array([])) for m in prob_hist},
        "labels": (np.concatenate(label_hist) if label_hist else np.array([])),
    }

    return (result_rows, acc, f1, ll, methods, eval_start,
            sample_positions, diagnostics)


# ================================================================
# 保存関数
# ================================================================
def save_result_tables(output_dir, all_results, methods, hp_by_n):
    """粒子数ごとの結果を TXT / CSV / LaTeX で保存する。"""
    txt_path = os.path.join(output_dir, "email_binary_results.txt")
    csv_path = os.path.join(output_dir, "email_binary_results.csv")
    tex_path = os.path.join(output_dir, "email_binary_results.tex")

    n_particles_list = sorted(all_results.keys())

    # TXT
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("Email binary classification results\n")
        fp.write("(best hyperparameters per particle count)\n")
        fp.write("=" * 70 + "\n\n")
        for n_p in n_particles_list:
            best_pf, best_wspf_b, best_wspf_a, _best_sgd = hp_by_n[n_p]
            fp.write(f"N_Particles = {n_p}\n")
            fp.write(
                f"  PF:    eta={best_pf['eta']}, "
                f"sigma_sys={best_pf['sigma_sys']}, "
                f"prior_std={best_pf['prior_std']}\n"
            )
            fp.write(
                f"  WSPF-B: eta={best_wspf_b['eta']}, "
                f"sigma_sys={best_wspf_b['sigma_sys']}, "
                f"prior_std={best_wspf_b['prior_std']}\n"
            )
            fp.write(
                f"  WSPF-A: eta={best_wspf_a['eta']}, "
                f"sigma_sys={best_wspf_a['sigma_sys']}, "
                f"prior_std={best_wspf_a['prior_std']}, "
                f"beta={best_wspf_a['beta']}\n"
            )
            fp.write(f"  {'Method':<10s} {'Accuracy':>10s} {'F1':>10s} {'Log-Lik':>10s}\n")
            fp.write(f"  {'-' * 46}\n")
            for row in all_results[n_p]:
                fp.write(
                    f"  {row['method']:<10s} "
                    f"{row['accuracy']:>10.4f} "
                    f"{row['f1']:>10.4f} "
                    f"{row['loglik']:>10.4f}\n"
                )
            fp.write("\n")

    # CSV
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write("n_particles,method,accuracy,f1,log_likelihood\n")
        for n_p in n_particles_list:
            for row in all_results[n_p]:
                fp.write(
                    f"{n_p},"
                    f"{row['method']},"
                    f"{row['accuracy']:.6f},"
                    f"{row['f1']:.6f},"
                    f"{row['loglik']:.6f}\n"
                )

    # LaTeX
    lookup = {}
    for n_p in n_particles_list:
        for row in all_results[n_p]:
            if row["method"] not in lookup:
                lookup[row["method"]] = {}
            lookup[row["method"]][n_p] = row

    n_cols = len(n_particles_list)
    col_spec = "@{}l" + "c" * n_cols + "@{}"

    with open(tex_path, "w", encoding="utf-8") as fp:
        fp.write("\\begin{table}[t]\n")
        fp.write("\\centering\n")
        fp.write("\\caption{Email binary classification results "
                 "(test-then-train, averaged after warm-up, "
                 "best hyperparameters per particle count).}\n")
        fp.write("\\label{tab:email_binary_results}\n")
        fp.write("\\small\n")
        fp.write("\\renewcommand{\\arraystretch}{1.15}\n")
        fp.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        fp.write("\\toprule\n")

        header = "Method"
        for n_p in n_particles_list:
            header += f" & $N={n_p}$"
        fp.write(header + " \\\\\n")
        fp.write("\\midrule\n")

        # Accuracy
        fp.write(f"\\multicolumn{{{n_cols + 1}}}{{c}}"
                 "{\\textit{Accuracy}} \\\\\n")
        fp.write("\\midrule\n")
        for m in methods:
            line = m
            for n_p in n_particles_list:
                line += f" & ${lookup[m][n_p]['accuracy']:.3f}$"
            fp.write(line + " \\\\\n")

        # F1
        fp.write("\\midrule\n")
        fp.write(f"\\multicolumn{{{n_cols + 1}}}{{c}}"
                 "{\\textit{F1 (Interesting)}} \\\\\n")
        fp.write("\\midrule\n")
        for m in methods:
            line = m
            for n_p in n_particles_list:
                line += f" & ${lookup[m][n_p]['f1']:.3f}$"
            fp.write(line + " \\\\\n")

        # Log-Likelihood
        fp.write("\\midrule\n")
        fp.write(f"\\multicolumn{{{n_cols + 1}}}{{c}}"
                 "{\\textit{Log-Likelihood}} \\\\\n")
        fp.write("\\midrule\n")
        for m in methods:
            line = m
            for n_p in n_particles_list:
                line += f" & ${lookup[m][n_p]['loglik']:.3f}$"
            fp.write(line + " \\\\\n")

        fp.write("\\bottomrule\n")
        fp.write("\\end{tabular}\n")
        fp.write("\\end{table}\n")

    return txt_path, csv_path, tex_path


def _plot_timeseries(output_dir, data, methods, eval_start, n_particles,
                     sample_positions, metric_name, ylabel, filename_base):
    """メトリクスの時系列プロットを保存する。"""
    out_path = os.path.join(output_dir, f"{filename_base}_N{n_particles}.png")

    n_steps = len(next(iter(data.values())))
    window = min(10, n_steps // 5) if n_steps > 5 else 1

    colors = {
        "NoChange": "#444444",
        "SGD":      "#888888",
        "PF":       "#0072B2",
        "WSPF-B":    "#E69F00",
        "WSPF-A":    "#D55E00",
    }

    plt.figure(figsize=(10, 5.5))

    if window > 1:
        for m in methods:
            ma = moving_average(data[m], window)
            t_ma = sample_positions[window - 1:][:len(ma)]
            plt.plot(t_ma, ma, color=colors[m], label=m, linewidth=1.4)
    else:
        for m in methods:
            plt.plot(sample_positions, data[m], color=colors[m], label=m, linewidth=1.2)

    # 報告区間(期3-5)を網掛け表示。期1-2 は PCA fit + HP 選択に使う warm-up/val
    x_max = float(sample_positions[-1]) if len(sample_positions) else REPORT_START
    plt.axvspan(REPORT_START, x_max, color="green", alpha=0.06,
                label="Reported region (periods 3-5)")

    # ドリフトポイントを垂直線で表示
    for i, dp in enumerate(DRIFT_POINTS):
        lbl = "Concept drift" if i == 0 else None
        plt.axvline(dp, color="red", linestyle="--", linewidth=0.9, alpha=0.7, label=lbl)

    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    title = f"Email binary {metric_name} (MA w={window}, N={n_particles})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def save_accuracy_plot(output_dir, acc, methods, eval_start, n_particles, sample_positions):
    return _plot_timeseries(
        output_dir, acc, methods, eval_start, n_particles, sample_positions,
        "test accuracy", "Accuracy", "email_binary_accuracy_timeseries",
    )


def save_f1_plot(output_dir, f1_data, methods, eval_start, n_particles, sample_positions):
    return _plot_timeseries(
        output_dir, f1_data, methods, eval_start, n_particles, sample_positions,
        "test F1", "F1", "email_binary_f1_timeseries",
    )


def save_loglik_plot(output_dir, ll_data, methods, eval_start, n_particles, sample_positions):
    return _plot_timeseries(
        output_dir, ll_data, methods, eval_start, n_particles, sample_positions,
        "test log-likelihood", "Log-Likelihood", "email_binary_loglik_timeseries",
    )


def save_summary_plot(output_dir, all_results, particle_methods):
    """粒子数 vs 精度のサマリープロットを保存する。"""
    out_path = os.path.join(output_dir, "email_binary_accuracy_vs_nparticles.png")

    n_particles_list = sorted(all_results.keys())

    colors = {
        "PF":    "#0072B2",
        "WSPF-A": "#D55E00",
        "WSPF-B": "#E69F00",
    }

    plt.figure(figsize=(8, 5))

    for m in particle_methods:
        accs = []
        for n_p in n_particles_list:
            for row in all_results[n_p]:
                if row["method"] == m:
                    accs.append(row["accuracy"])
                    break
        plt.plot(n_particles_list, accs, "o-",
                 color=colors.get(m, "black"), label=m,
                 linewidth=1.5, markersize=6)

    plt.xlabel("Number of particles")
    plt.ylabel("Accuracy")
    plt.title("Email Binary: Accuracy vs. number of particles\n"
              "(best hyperparameters per $N$)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(n_particles_list)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


# ================================================================
# メイン
# ================================================================
def main():
    print("=" * 70)
    print("Email Binary Classification: particle count comparison")
    print(f"PCA dim: {PCA_DIM}, batch: {BATCH_SIZE}, test: {TEST_SIZE}")
    print("=" * 70)

    # -------- データ読み込み --------
    print("\nLoading data...")
    loader = EmailDataLoader(
        DATA_PATH, n_components=PCA_DIM, seed=SEED, pca_fit_end=PCA_FIT_END,
    )
    print(f"  Samples: {loader.n_samples:,}")
    print(f"  Features: {loader.input_dim} (PCA from 913)")
    print(f"  Mode: binary (junk=0, interesting=1)")
    interesting_ratio = loader.y.sum() / loader.n_samples
    print(f"  Interesting ratio: {interesting_ratio:.3f}")
    print(f"  Drift points: {DRIFT_POINTS}")

    # -------- モデル --------
    model = NeuralNetModel(
        input_dim=loader.input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        activation="tanh",
    )
    param_dim = model.param_dim
    print(f"\nModel: input={loader.input_dim}, hidden={HIDDEN_DIM}, output=1")
    print(f"Total parameters: {param_dim}")

    grad_fn_raw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)

    def grad_fn(particles, X, y):
        g = grad_fn_raw(particles, X, y)
        return clip_gradients(g, MAX_GRAD_NORM)

    # -------- ハイパーパラメータ読み込み --------
    gs_data = load_grid_search_params()

    # 厳格化(無言フォールバック廃止)+ SGD 独自 η(best_sgd)を読み込む。
    if gs_data is None:
        raise RuntimeError(
            f"email グリッド結果が見つかりません({GRID_SEARCH_JSON})。先に "
            "grid_search_email.py を実行してください。")
    hp_by_n = {}
    print(f"\nLoaded grid search results from {GRID_SEARCH_JSON}")
    for n_p in N_PARTICLES_LIST:
        key = str(n_p)
        if key not in gs_data:
            raise KeyError(f"email グリッド JSON に N={n_p} がありません。")
        entry = gs_data[key]
        for k in ("best_pf", "best_wspf_b", "best_wspf_a", "best_sgd"):
            if k not in entry:
                raise KeyError(
                    f"email グリッド JSON に '{k}' がありません(旧キー/SGD未選択の"
                    "可能性)。グリッドを再生成してください。")
        best_pf = entry["best_pf"]
        best_wspf_b = entry["best_wspf_b"]
        best_wspf_a = entry["best_wspf_a"]
        best_sgd = entry["best_sgd"]
        if "beta" not in best_wspf_a:
            raise KeyError("best_wspf_a に 'beta' がありません。")
        hp_by_n[n_p] = (best_pf, best_wspf_b, best_wspf_a, best_sgd)
        print(
            f"  N={n_p}:\n"
            f"    PF:    eta={best_pf['eta']}, "
            f"sigma_sys={best_pf['sigma_sys']}, "
            f"prior_std={best_pf['prior_std']}\n"
            f"    WSPF-B: eta={best_wspf_b['eta']}, "
            f"sigma_sys={best_wspf_b['sigma_sys']}, "
            f"prior_std={best_wspf_b['prior_std']}\n"
            f"    WSPF-A: eta={best_wspf_a['eta']}, "
            f"sigma_sys={best_wspf_a['sigma_sys']}, "
            f"prior_std={best_wspf_a['prior_std']}, "
            f"beta={best_wspf_a['beta']}\n"
            f"    SGD(独立): eta={best_sgd['eta']}, "
            f"prior_std={best_sgd['prior_std']}"
        )

    # -------- 粒子数ごとに実験 --------
    all_results = {}
    all_acc = {}
    all_f1 = {}
    all_ll = {}
    all_sample_pos = {}
    all_diagnostics = {}
    methods = None
    eval_start = None

    for n_particles in N_PARTICLES_LIST:
        best_pf, best_wspf_b, best_wspf_a, best_sgd = hp_by_n[n_particles]
        sgd_eta = best_sgd["eta"]     # SGD 独自 η(PF best 連動をやめ交絡除去)
        sgd_prior = best_sgd["prior_std"]

        print(f"\n{'=' * 70}")
        print(f"Running experiment with N_PARTICLES = {n_particles}")
        print(f"{'=' * 70}")

        (result_rows, acc, f1_dict, ll_dict, methods, eval_start,
         sample_pos, diagnostics) = run_experiment(
            n_particles=n_particles,
            model=model,
            loader=loader,
            grad_fn=grad_fn,
            loglik_fn=loglik_fn,
            ps_grad_fn=ps_grad_fn,
            best_pf=best_pf,
            best_wspf_b=best_wspf_b,
            best_wspf_a=best_wspf_a,
            sgd_eta=sgd_eta,
            sgd_prior=sgd_prior,
            param_dim=param_dim,
        )

        all_results[n_particles] = result_rows
        all_acc[n_particles] = acc
        all_f1[n_particles] = f1_dict
        all_ll[n_particles] = ll_dict
        all_sample_pos[n_particles] = sample_pos
        all_diagnostics[n_particles] = diagnostics

        print(f"\n  === Results (N={n_particles}, reported on periods 3-5, "
              f"samples >= {REPORT_START}, first eval window={eval_start}) ===")
        print(f"  {'Method':<10s} {'Accuracy':>10s} {'F1':>10s} {'Log-Lik':>10s}")
        print(f"  {'-' * 46}")
        for row in result_rows:
            print(
                f"  {row['method']:<10s} "
                f"{row['accuracy']:>10.4f} "
                f"{row['f1']:>10.4f} "
                f"{row['loglik']:>10.4f}"
            )

    # -------- npz 保存 --------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for n_particles in N_PARTICLES_LIST:
        npz_path = os.path.join(OUTPUT_DIR, f"results_N{n_particles}_seed{SEED}.npz")
        sd = {
            "step": len(next(iter(all_acc[n_particles].values()))),
            "eval_start": eval_start,
            "sample_positions": all_sample_pos[n_particles],
        }
        for name in methods:
            prefix = name.replace(" ", "_").replace("-", "_").lower()
            sd[f"{prefix}_acc"] = all_acc[n_particles][name]
            sd[f"{prefix}_f1"] = all_f1[n_particles][name]
            sd[f"{prefix}_ll"] = all_ll[n_particles][name]
        # PF 系フィルタの診断ログ (R1-8, R1-9, R2-2)
        diag = all_diagnostics[n_particles]
        for name, hist in diag.items():
            if name.startswith("_"):
                continue  # "_calib" 等の非履歴エントリ(dict)は npz 化しない
            prefix = name.replace(" ", "_").replace("-", "_").lower()
            for key, arr in hist.items():
                if key in ("mean", "std"):
                    continue  # (T, d) の大きな配列は保存しない
                sd[f"diag_{prefix}_{key}"] = arr
        np.savez(npz_path, **sd)
        print(f"Results (npz) saved to {npz_path}")

    # -------- テーブル保存 --------
    txt_path, csv_path, tex_path = save_result_tables(
        OUTPUT_DIR, all_results, methods, hp_by_n,
    )

    fig_paths = []
    for n_particles in N_PARTICLES_LIST:
        sp = all_sample_pos[n_particles]
        fig_path = save_accuracy_plot(
            OUTPUT_DIR, all_acc[n_particles], methods, eval_start, n_particles, sp
        )
        fig_paths.append(fig_path)
        fig_path = save_f1_plot(
            OUTPUT_DIR, all_f1[n_particles], methods, eval_start, n_particles, sp
        )
        fig_paths.append(fig_path)
        fig_path = save_loglik_plot(
            OUTPUT_DIR, all_ll[n_particles], methods, eval_start, n_particles, sp
        )
        fig_paths.append(fig_path)

    particle_methods = ["PF", "WSPF-A", "WSPF-B"]
    summary_path = save_summary_plot(OUTPUT_DIR, all_results, particle_methods)

    # -------- サマリー表示 --------
    print(f"\n{'=' * 70}")
    print("Summary: Accuracy across particle counts")
    print(f"{'=' * 70}")
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
                    print(f" {row['accuracy']:>10.4f}", end="")
                    break
        print()

    print("\nSaved files:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")
    print(f"  {tex_path}")
    for fp in fig_paths:
        print(f"  {fp}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
