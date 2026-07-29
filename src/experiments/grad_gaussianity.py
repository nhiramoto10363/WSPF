#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
勾配ノイズのガウス性検証 — Regression (R1-6)

WSPF はミニバッチ勾配ノイズ ξ = ĝ − ∇L の(近似)ガウス性を仮定する。
本実験は真の生成分布が既知の回帰タスクで、固定した粒子位置における
mini-batch 勾配を多数バッチから採取し、正規性(歪度・尖度・単変量正規性
検定・マハラノビス距離の χ² 適合)と共分散の異方性(固有値スペクトル)を
バッチサイズ B∈{8,16,32,64} × 時点(安定期 / スイッチ直後)で検証する。
併せて各 B での手法性能(MSE)も報告する(R1-6 後段)。

CLT より B が大きいほど ĝ はガウスに近づく(歪度・尖度→0)。仮定の妥当性
と B 依存性を定量化する。

出力:
  outputs/grad_gaussianity/
    - grad_gaussianity.txt / .csv
    - grad_gaussianity_qq.png       (マハラノビス距離² の χ² QQ)
    - grad_gaussianity_spectrum.png (共分散固有値スペクトル)
    - grad_gaussianity_mse_vs_B.png (性能 vs B)
"""

import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.experiments.regression_regime_switch import (
    NeuralNetRegression, generate_regression_regime_data,
    create_regression_per_sample_grad_fn, run_single,
    INPUT_DIM, HIDDEN_DIM, TEST_SIZE, NOISE_STD, T, EVAL_START, SEEDS,
)
from src.experiments.oracle_regression import oracle_grad_stats
from src.experiments.matched_hp_regression import build_matched_hp

BATCH_SIZES = [8, 16, 32, 64]
M_BATCHES = 3000          # ガウス性検証のバッチ数
M_ORACLE = 100000         # ∇L, C の大標本 MC 数
N_PARTICLES = 100
METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "grad_gaussianity",
)


# ================================================================
# Part A: 勾配ノイズのガウス性
# ================================================================
def sample_minibatch_grads(model, theta_eval, theta_star, noise_std,
                           B, M_batches, rng):
    """
    θ_eval で、真の分布 p(θ_star) から B サイズのバッチを M_batches 個引き、
    各バッチのミニバッチ勾配 ĝ_b を返す (M_batches, d)。
    """
    ps_grad = create_regression_per_sample_grad_fn(model, noise_std)
    n_pts = B * M_batches
    X = rng.normal(0.0, 1.0, size=(n_pts, model.input_dim))
    out, _, _ = model.forward(theta_star.reshape(1, -1), X)
    y = out.squeeze() + rng.normal(0.0, noise_std, size=n_pts)
    g = ps_grad(theta_eval.reshape(1, -1), X, y)[0]     # (n_pts, d)
    g = g.reshape(M_batches, B, g.shape[-1])
    return g.mean(axis=1)                                # (M_batches, d)


def gaussianity_stats(xi, grad_L=None):
    """ξ (M,d) の正規性・異方性統計を計算する。"""
    M, d = xi.shape
    # 各次元の歪度・尖度(Fisher: 正規=0)
    skew = stats.skew(xi, axis=0)
    kurt = stats.kurtosis(xi, axis=0)  # excess kurtosis
    # 単変量正規性検定 (D'Agostino) — 有意に非正規な次元の割合
    try:
        _, p_norm = stats.normaltest(xi, axis=0)
        frac_reject = float(np.mean(p_norm < 0.05))
    except Exception:
        frac_reject = float("nan")
    # 共分散の固有分解(ξ の共分散 ≈ C/B)。勾配ノイズは異方性が強く
    # 特異になりうるので、有効ランク部分空間で白色化してマハラノビス²を作る。
    cov = np.cov(xi, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)             # 昇順
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    tol = 1e-8 * max(evals[0], 1e-300)
    keep = evals > tol
    k = int(keep.sum())
    Vk = evecs[:, keep]
    lamk = evals[keep]
    xic = xi - xi.mean(axis=0)
    z = (xic @ Vk) / np.sqrt(lamk)                 # 白色化 (M, k)
    maha2 = np.sum(z ** 2, axis=1)                 # ~ χ²_k
    ks_stat, ks_p = stats.kstest(maha2, "chi2", args=(k,))
    eig = np.maximum(evals, 0.0)
    cond = float(eig[0] / eig[keep][-1]) if k > 0 else float("inf")
    return {
        "skew_mean_abs": float(np.mean(np.abs(skew))),
        "skew_max_abs": float(np.max(np.abs(skew))),
        "kurt_mean_abs": float(np.mean(np.abs(kurt))),
        "kurt_max_abs": float(np.max(np.abs(kurt))),
        "frac_dims_reject_normal": frac_reject,
        "maha_ks_stat": float(ks_stat),
        "maha_ks_p": float(ks_p),
        "maha_df": k,                              # 有効ランク(χ² 自由度)
        "cov_cond": cond,
        "eig": eig,
        "maha2": maha2,
    }


def run_part_a(emit):
    model = NeuralNetRegression(INPUT_DIM, HIDDEN_DIM, output_dim=1,
                                activation="tanh")
    # seed 0 のストリームから代表的な θ*・時点を取得
    (_, _, _, _, theta_true, switch_times) = generate_regression_regime_data(
        model, T=T, batch_size=16, test_size=TEST_SIZE,
        noise_std=NOISE_STD, within_regime_drift=0.0005, seed=0)
    sw = switch_times[0]
    # 安定期: レジーム内で収束した時点(スイッチ直前)。θ_eval=θ*,data=θ*
    t_stable = sw - 5
    # スイッチ直後: 粒子は旧 θ* のまま、データは新 θ*
    scenarios = {
        "stable": {"theta_eval": theta_true[t_stable],
                   "theta_star": theta_true[t_stable]},
        "post-switch": {"theta_eval": theta_true[sw - 1],
                        "theta_star": theta_true[sw]},
    }

    emit(f"\n{'='*72}\n  Part A: 勾配ノイズ ξ=ĝ−∇L のガウス性\n{'='*72}")
    emit(f"  M_batches={M_BATCHES}, M_oracle={M_ORACLE}, "
         f"switch@{sw}, t_stable={t_stable}")

    resultsA = {}          # (scenario,B) -> stats
    spectra = {}           # scenario -> eig (B-非依存, C の固有値)
    for sc, cfg in scenarios.items():
        theta_eval = cfg["theta_eval"]
        theta_star = cfg["theta_star"]
        # ∇L と per-sample 共分散 C(θ) を大標本オラクルで(B=1 → Sigma=C)
        grad_L, C = oracle_grad_stats(
            model, theta_eval.reshape(1, -1), theta_star, NOISE_STD,
            M_ORACLE, B=1, rng=np.random.default_rng(777))
        grad_L = grad_L[0]
        C = C[0]
        eigC = np.maximum(np.linalg.eigvalsh(C)[::-1], 0.0)
        spectra[sc] = eigC
        condC = eigC[0] / eigC[-1] if eigC[-1] > 0 else float("inf")
        emit(f"\n  [{sc}] ||∇L||={np.linalg.norm(grad_L):.4f}, "
             f"tr(C)={np.trace(C):.4f}, cond(C)={condC:.1f} "
             f"(異方性: λmax/λmin)")
        emit(f"  {'B':>4s} {'skew|mean':>10s} {'kurt|mean':>10s} "
             f"{'%dim¬normal':>12s} {'maha χ²(df) KS-p':>18s}")
        for B in BATCH_SIZES:
            rng = np.random.default_rng(1000 + B)
            g_hat = sample_minibatch_grads(model, theta_eval, theta_star,
                                           NOISE_STD, B, M_BATCHES, rng)
            xi = g_hat - grad_L[None, :]
            st = gaussianity_stats(xi)
            resultsA[(sc, B)] = st
            emit(f"  {B:>4d} {st['skew_mean_abs']:>10.4f} "
                 f"{st['kurt_mean_abs']:>10.4f} "
                 f"{100*st['frac_dims_reject_normal']:>11.1f}% "
                 f"{('df='+str(st['maha_df'])+' p='+format(st['maha_ks_p'],'.3g')):>18s}")
    return resultsA, spectra, scenarios, sw


# ================================================================
# Part B: 各 B での手法性能
# ================================================================
def _partb_job(args):
    """Part B の 1 ジョブ(モジュールレベル: ProcessPool でピクル可能)。"""
    B, seed, bp, bb, ba, best_sgd = args
    res = run_single(seed, N_PARTICLES, bp, bb, ba, batch_size=B,
                     best_sgd=best_sgd)
    return B, {m: float(np.asarray(res["mse"][m])[EVAL_START:].mean())
               for m in METHODS}


def run_part_b(emit, matched, beta, best_sgd):
    emit(f"\n{'='*72}\n  Part B: 手法性能 vs バッチサイズ B (eval-region MSE)\n{'='*72}")
    jobs = []
    for B in BATCH_SIZES:
        bp = {**matched, "sigma_sys": matched["sigma_sys"]}
        bb = dict(bp)
        ba = {**bp, "beta": beta}
        for seed in SEEDS:
            jobs.append((B, seed, bp, bb, ba, best_sgd))

    mse_by_B = {B: {m: [] for m in METHODS} for B in BATCH_SIZES}
    n_workers = min(os.cpu_count() or 1, 48)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for B, dd in ex.map(_partb_job, jobs):
            for m in METHODS:
                mse_by_B[B][m].append(dd[m])

    emit(f"  {'B':>4s} " + " ".join(f"{m:>9s}" for m in METHODS))
    curve = {m: [] for m in METHODS}
    for B in BATCH_SIZES:
        row = f"  {B:>4d} "
        for m in METHODS:
            v = float(np.mean(mse_by_B[B][m]))
            curve[m].append(v)
            row += f"{v:>9.4f} "
        emit(row)
    return curve


# ================================================================
# プロット
# ================================================================
def plot_qq(resultsA, scenarios, path):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(11, 4.5),
                             squeeze=False)
    for ax, sc in zip(axes[0], scenarios):
        # χ² 理論分位(自由度 d = param_dim)との QQ
        for B in BATCH_SIZES:
            m2 = np.sort(resultsA[(sc, B)]["maha2"])
            n = len(m2)
            probs = (np.arange(1, n + 1) - 0.5) / n
            # 自由度は共分散の有効ランク
            df = resultsA[(sc, B)]["maha_df"]
            theo = stats.chi2.ppf(probs, df)
            ax.plot(theo, m2, ".", markersize=2, label=f"B={B} (df={df})")
        lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "k--", linewidth=0.8)
        ax.set_xlabel(r"$\chi^2_d$ theoretical quantile")
        ax.set_ylabel("Mahalanobis $d^2$ sample quantile")
        ax.set_title(f"QQ ({sc})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_spectrum(spectra, path):
    plt.figure(figsize=(7, 5))
    for sc, eig in spectra.items():
        e = eig / eig[0]  # 最大固有値で正規化
        plt.plot(np.arange(1, len(e) + 1), e, "o-", label=sc, linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("eigenvalue index (descending)")
    plt.ylabel(r"normalized eigenvalue $\lambda_i/\lambda_1$")
    plt.title("Per-sample gradient-noise covariance spectrum (anisotropy)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mse_vs_B(curve, path):
    colors = {"SGD": "#888888", "PF": "#0072B2",
              "WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
    plt.figure(figsize=(7, 5))
    for m in METHODS:
        plt.plot(BATCH_SIZES, curve[m], "o-", color=colors[m],
                 label=m, linewidth=1.6)
    plt.xscale("log")
    plt.xticks(BATCH_SIZES, [str(b) for b in BATCH_SIZES])
    plt.xlabel("batch size B")
    plt.ylabel("Test MSE (eval region)")
    plt.title("Performance vs batch size (Regression)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched, beta, src, best_sgd = build_matched_hp()[N_PARTICLES]
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Gradient-noise Gaussianity & batch-size study — Regression (R1-6)")
    emit(f"  batch sizes={BATCH_SIZES}, N={N_PARTICLES}, T={T}, seeds={len(SEEDS)}")
    emit(f"  fixed HP (η,σcd,σ0)={matched} [{src}], WSPF-A β={beta}")
    emit("=" * 72)

    t0 = time.time()
    resultsA, spectra, scenarios, sw = run_part_a(emit)
    curve = run_part_b(emit, matched, beta, best_sgd)
    emit(f"\n  total elapsed {time.time()-t0:.0f}s")

    # ---- プロット ----
    qq_png = os.path.join(OUTPUT_DIR, "grad_gaussianity_qq.png")
    spec_png = os.path.join(OUTPUT_DIR, "grad_gaussianity_spectrum.png")
    mse_png = os.path.join(OUTPUT_DIR, "grad_gaussianity_mse_vs_B.png")
    plot_qq(resultsA, scenarios, qq_png)
    plot_spectrum(spectra, spec_png)
    plot_mse_vs_B(curve, mse_png)

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "grad_gaussianity.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scenario,B,skew_mean_abs,kurt_mean_abs,"
                "frac_dims_reject_normal,maha_df,maha_ks_p,cov_cond\n")
        for (sc, B), st in resultsA.items():
            f.write(f"{sc},{B},{st['skew_mean_abs']:.6f},"
                    f"{st['kurt_mean_abs']:.6f},"
                    f"{st['frac_dims_reject_normal']:.6f},"
                    f"{st['maha_df']},{st['maha_ks_p']:.6g},"
                    f"{st['cov_cond']:.6f}\n")
        for m in METHODS:
            for i, B in enumerate(BATCH_SIZES):
                f.write(f"perf,{B},mse,{m},{curve[m][i]:.6f},,\n")

    txt_path = os.path.join(OUTPUT_DIR, "grad_gaussianity.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {qq_png}")
    emit(f"Saved: {spec_png}")
    emit(f"Saved: {mse_png}")


if __name__ == "__main__":
    main()
