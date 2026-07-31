#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-switch 期における予測確率の分布を可視化する。

目的:
  PF（補正なし）がコンセプトドリフト直後に「弱い（0.5 近傍にヘッジした）
  予測」を出すことを例示する。これが、PF が post-switch 期に
  精度・F1 では劣るのに対数尤度では有利になる理由（自信過剰な誤りへの
  log 損失ペナルティを回避している）を視覚的に裏付ける。

手法:
  email_binary_experiment.py と同一のデータ・モデル・フィルタ・seed・
  グリッドサーチ最良ハイパラでオンライン学習ループを再現し、各ステップで
  各手法の「加重平均パラメータ μ=Σ wᵢθᵢ での予測確率」をテストブロック上で
  記録する。記録した予測確率を期間（stable / post-switch）ごとに集計する。

出力:
  outputs/email_binary/
    - email_binary_post_switch_pred_dist_N{n}.png
    - email_binary_pred_confidence_N{n}.csv
"""

import os
import sys
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _PROJECT_ROOT)

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
# 設定（email_binary_experiment.py と整合）
# ================================================================
HIDDEN_DIM = 16
BATCH_SIZE = 16
MAX_GRAD_NORM = 5.0
SEED = 42
TEST_SIZE = 32
PCA_DIM = 50
PCA_FIT_END = 600               # リーク除去(R1-13): PCA は期1-2 のみで学習
SWITCHES = [300, 600, 900, 1200]
POST_SWITCH_STEPS = 5            # スイッチ後を「遷移期」とみなすステップ数

DATA_PATH = os.path.join(_PROJECT_ROOT, "Email", "email_data.arff")
GRID_SEARCH_JSON = os.path.join(
    _PROJECT_ROOT, "outputs", "email_binary", "grid_search_result.json"
)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "email_binary")

# npz / 内部キー -> 表示名（PF を強調するため並び順を固定）
METHOD_ORDER = ["SGD", "PF", "WSPF-A", "WSPF-B"]
METHOD_COLORS = {
    "SGD": "#9467bd",
    "PF": "#d62728",        # 赤＝主役（ヘッジを示す）
    "WSPF-A": "#1f77b4",
    "WSPF-B": "#2ca02c",
}


def clip_gradients(grad, max_norm):
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    return grad * scale


def load_best_params(n_particles):
    """グリッドサーチ結果から最良ハイパラを取得する。"""
    with open(GRID_SEARCH_JSON) as fp:
        gs = json.load(fp)["by_n_particles"][str(n_particles)]
    best_pf = gs["best_pf"]
    best_wspf_b = gs["best_wspf_b"]
    best_wspf_a = gs["best_wspf_a"]
    return best_pf, best_wspf_b, best_wspf_a


def predict_proba(model, theta, X):
    """加重平均パラメータ θ での陽性クラス予測確率 (shape (B,))。"""
    output, _, _ = model.forward(theta.reshape(1, -1), X)
    return np.clip(output.squeeze(axis=0).squeeze(-1), 1e-10, 1 - 1e-10)


def run_and_collect(n_particles):
    """実験ループを再現し、期間ごとの予測確率を収集する。"""
    loader = EmailDataLoader(DATA_PATH, n_components=PCA_DIM, seed=SEED,
                             pca_fit_end=PCA_FIT_END)
    model = NeuralNetModel(
        input_dim=loader.input_dim, hidden_dim=HIDDEN_DIM,
        output_dim=1, activation="tanh",
    )
    param_dim = model.param_dim

    grad_fn_raw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)

    def grad_fn(particles, X, y):
        return clip_gradients(grad_fn_raw(particles, X, y), MAX_GRAD_NORM)

    best_pf, best_wspf_b, best_wspf_a = load_best_params(n_particles)

    # --- フィルタ・SGD 初期化（run_experiment と同一 seed） ---
    rng_sgd = np.random.default_rng(SEED + 10)
    theta_sgd = rng_sgd.normal(0.0, best_pf["prior_std"], size=param_dim)
    sgd_eta = best_pf["eta"]

    pf = ParticleFilter(
        n_particles=n_particles, param_dim=param_dim, eta=best_pf["eta"],
        sigma_sys=best_pf["sigma_sys"], prior_mean=0.0,
        prior_std=best_pf["prior_std"], ess_resample_ratio=0.5, seed=SEED + 1,
    )
    wspf_b = WSPF_B(
        n_particles=n_particles, param_dim=param_dim, eta=best_wspf_b["eta"],
        sigma_sys=best_wspf_b["sigma_sys"], prior_mean=0.0,
        prior_std=best_wspf_b["prior_std"], ess_resample_ratio=0.5,
        grad_clip_norm=MAX_GRAD_NORM, seed=SEED + 3,
    )
    wspf_a = WSPF_A(
        n_particles=n_particles, param_dim=param_dim, eta=best_wspf_a["eta"],
        sigma_sys=best_wspf_a["sigma_sys"], prior_mean=0.0,
        prior_std=best_wspf_a["prior_std"], ess_resample_ratio=0.5,
        grad_clip_norm=MAX_GRAD_NORM, beta=best_wspf_a["beta"], seed=SEED + 5,
    )

    n = loader.n_samples
    total_steps = (n - TEST_SIZE) // BATCH_SIZE
    eval_start = max(1, total_steps // 10)   # ウォームアップ除外（実験と同一規則）

    # method -> {"stable": [...probs...], "post": [...probs...]}
    probs = {m: {"stable": [], "post": []} for m in METHOD_ORDER}

    pos = 0
    step = 0
    while pos + BATCH_SIZE + TEST_SIZE <= n:
        X_train = loader.X[pos: pos + BATCH_SIZE]
        y_train = loader.y[pos: pos + BATCH_SIZE]
        X_test = loader.X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]

        test_start = pos + BATCH_SIZE
        is_post = any(sw <= test_start < sw + POST_SWITCH_STEPS * BATCH_SIZE
                      for sw in SWITCHES)
        phase = "post" if is_post else "stable"

        # --- 評価（学習前）：各手法の平均パラメータでの予測確率 ---
        if step >= eval_start:
            mus = {
                "SGD": theta_sgd,
                "PF": (pf.weights[:, None] * pf.particles).sum(axis=0),
                "WSPF-A": (wspf_a.weights[:, None] * wspf_a.particles).sum(axis=0),
                "WSPF-B": (wspf_b.weights[:, None] * wspf_b.particles).sum(axis=0),
            }
            for m in METHOD_ORDER:
                probs[m][phase].append(predict_proba(model, mus[m], X_test))

        # --- 学習 ---
        g = grad_fn(theta_sgd.reshape(1, -1), X_train, y_train).squeeze()
        theta_sgd = theta_sgd - sgd_eta * g
        pf.step(X_train, y_train, grad_fn, loglik_fn)
        wspf_b.step(X_train, y_train, ps_grad_fn, loglik_fn)
        wspf_a.step(X_train, y_train, ps_grad_fn, loglik_fn)

        pos += BATCH_SIZE
        step += 1

    # 各リストを 1 次元配列に連結
    for m in METHOD_ORDER:
        for ph in ("stable", "post"):
            probs[m][ph] = (np.concatenate(probs[m][ph])
                            if probs[m][ph] else np.array([]))
    return probs, eval_start


def summarize(probs):
    """期間・手法ごとの予測信頼度サマリ（dict のリスト）。"""
    rows = []
    for m in METHOD_ORDER:
        for ph in ("stable", "post"):
            p = probs[m][ph]
            rows.append({
                "method": m,
                "phase": ph,
                "n": int(p.size),
                "mean_p": float(p.mean()),
                "std_p": float(p.std()),
                "mean_confidence": float(np.abs(p - 0.5).mean()),  # |p-0.5|
                "hedged_frac": float(np.mean((p > 0.4) & (p < 0.6))),  # 0.5近傍率
            })
    return rows


def write_csv(rows, out_path):
    fields = ["method", "phase", "n", "mean_p", "std_p",
              "mean_confidence", "hedged_frac"]
    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                for k in fields
            })


def make_figure(probs, rows, n_particles, out_path):
    """2 パネル図: (a) post-switch 予測確率分布, (b) 信頼度の期間比較。"""
    summ = {(r["method"], r["phase"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- パネル (a): post-switch の予測確率分布（バイオリン） ----
    ax = axes[0]
    data = [probs[m]["post"] for m in METHOD_ORDER]
    positions = np.arange(len(METHOD_ORDER))
    vp = ax.violinplot(data, positions=positions, showmeans=False,
                       showextrema=False, widths=0.8)
    for body, m in zip(vp["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[m])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    # 中央値マーカと 0.5 基準線
    for x, m in zip(positions, METHOD_ORDER):
        med = np.median(probs[m]["post"])
        ax.plot([x - 0.18, x + 0.18], [med, med], color="black", lw=1.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1, zorder=0)
    ax.text(len(METHOD_ORDER) - 0.5, 0.5, "  hedge (0.5)",
            va="center", ha="left", color="gray", fontsize=8)
    # 各手法の「0.5 近傍率」(|p-0.5|<0.1) を注記
    for x, m in zip(positions, METHOD_ORDER):
        frac = summ[(m, "post")]["hedged_frac"]
        ax.text(x, -0.09, f"{100*frac:.0f}%\nnear 0.5",
                ha="center", va="top", fontsize=7.5,
                color=METHOD_COLORS[m])
    ax.set_xticks(positions)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylim(-0.18, 1.02)
    ax.set_ylabel("Predicted probability $p$ (positive class)")
    ax.set_title("(a) Post-switch predicted-probability distribution")

    # ---- パネル (b): 平均信頼度 |p-0.5| の stable vs post 比較 ----
    ax = axes[1]
    width = 0.38
    stable_conf = [summ[(m, "stable")]["mean_confidence"] for m in METHOD_ORDER]
    post_conf = [summ[(m, "post")]["mean_confidence"] for m in METHOD_ORDER]
    ax.bar(positions - width / 2, stable_conf, width,
           color="#bbbbbb", edgecolor="black")
    ax.bar(positions + width / 2, post_conf, width,
           color=[METHOD_COLORS[m] for m in METHOD_ORDER], edgecolor="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylim(0, max(stable_conf) * 1.25)
    ax.set_ylabel("Mean confidence  $\\overline{|p-0.5|}$")
    ax.set_title("(b) Prediction confidence: stable vs post-switch")
    # 凡例: 左バー=stable(灰), 右バー=post-switch(各手法色) を明示
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#bbbbbb", edgecolor="black", label="stable"),
        Patch(facecolor="white", edgecolor="black",
              label="post-switch (method-colored)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8,
              loc="upper right")

    fig.suptitle(
        f"PF hedges its predictions after a concept switch "
        f"(Email stream, $N={n_particles}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    n_particles = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"Re-running email experiment (N={n_particles}) to record predictions...")
    probs, eval_start = run_and_collect(n_particles)
    rows = summarize(probs)

    csv_path = os.path.join(
        OUTPUT_DIR, f"email_binary_pred_confidence_N{n_particles}.csv")
    fig_path = os.path.join(
        OUTPUT_DIR, f"email_binary_post_switch_pred_dist_N{n_particles}.png")
    write_csv(rows, csv_path)
    make_figure(probs, rows, n_particles, fig_path)

    # コンソール要約
    print(f"\nwarm-up end step (eval_start): {eval_start}")
    print(f"{'method':8s} {'phase':7s} {'n':>5s} {'mean_p':>8s} "
          f"{'mean|p-.5|':>11s} {'hedged%':>8s}")
    for r in rows:
        print(f"{r['method']:8s} {r['phase']:7s} {r['n']:5d} "
              f"{r['mean_p']:8.3f} {r['mean_confidence']:11.3f} "
              f"{100*r['hedged_frac']:7.1f}%")
    print(f"\nSaved:\n  {os.path.relpath(csv_path, _PROJECT_ROOT)}"
          f"\n  {os.path.relpath(fig_path, _PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
