#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
計算コスト・ベンチマーク (R1-11)

PF / WSPF-A / WSPF-B について以下を計測・報告する:
  - 1 オンライン更新あたりの実行時間(wall-clock)と内訳
    (勾配 / 補正 / 尤度 / 重み / リサンプリング)
  - per-sample 勾配評価の回数(PF はバッチ勾配 N、WSPF は per-sample N*B)
  - WSPF-A の行列補正における線形代数(Woodbury/Cholesky/logdet)コスト
  - メモリ: WSPF-A が実際に保持する行列(Woodbury: B×B)と、素朴な
    d×d 直接保持(= R1-11 が指摘する ~555MB)の対比

email 相当の設定(input=50, hidden=16, output=1 → param_dim=833, B=16, N=100)。

出力:
  outputs/compute_cost/
    - compute_cost_benchmark.txt
    - compute_cost_benchmark.csv
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.filters import ParticleFilter
from src.filters.wspf_a import WSPF_A, compute_correction_method_a
from src.filters.wspf_b import WSPF_B
from src.models.neural_net import (
    NeuralNetModel,
    create_nn_grad_fn,
    create_nn_loglik_fn,
    create_nn_per_sample_grad_fn,
)

# ---- 設定(email 相当) ----
INPUT_DIM = 50
HIDDEN_DIM = 16
OUTPUT_DIM = 1
BATCH_SIZE = 16
N_PARTICLES = 100
ETA = 0.1
SIGMA_SYS = 0.1
GRAD_CLIP = 5.0
BETA = 0.9
SEED = 42

WARMUP_STEPS = 10   # JIT/キャッシュ・ウォームアップ(集計から除外)
MEASURE_STEPS = 60  # 集計対象ステップ数

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "compute_cost",
)


def _pct(a, q):
    return float(np.percentile(np.asarray(a), q))


def _summarize_timing(history, warmup):
    """history のタイミング配列(warmup 除外)を集計する。"""
    keys = ["t_step", "t_grad", "t_correction", "t_loglik",
            "t_weight", "t_resample"]
    out = {}
    for k in keys:
        arr = np.asarray(history[k][warmup:], dtype=np.float64)
        out[k] = {
            "mean_ms": 1e3 * arr.mean(),
            "median_ms": 1e3 * np.median(arr),
            "p95_ms": 1e3 * _pct(arr, 95),
        }
    return out


def run_filter_bench(name, filt, grad_fn, loglik_fn, model, rng):
    """フィルタを WARMUP+MEASURE ステップ回して計測する。"""
    total = WARMUP_STEPS + MEASURE_STEPS
    for _ in range(total):
        X = rng.standard_normal((BATCH_SIZE, INPUT_DIM))
        # 適当な二値ラベル(尤度評価のため)
        y = (rng.standard_normal(BATCH_SIZE) > 0).astype(np.float64)
        filt.step(X, y, grad_fn, loglik_fn)

    timing = _summarize_timing(filt.history, WARMUP_STEPS)
    return {
        "name": name,
        "grad_eval_kind": filt.grad_eval_kind,
        "sample_grad_evals_per_step": int(
            np.mean(filt.history["sample_grad_evals"][WARMUP_STEPS:])
        ),
        "timing": timing,
    }


def bench_wspf_a_linalg(d, N, B, c=SIGMA_SYS ** 2, reps=30):
    """
    WSPF-A の行列補正における線形代数コストを分解計測し、
    素朴な d×d 直接反転(O(d³))との比較を行う。
    """
    rng = np.random.default_rng(0)
    per = 0.5 * rng.standard_normal((N, B, d))
    g_hat = per.mean(axis=1)
    deviations = per - g_hat[:, None, :]
    epsilon = rng.standard_normal((N, d))
    xi_hat = 0.1 * rng.standard_normal((N, d))

    # Woodbury 版(実装関数)全体
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compute_correction_method_a(epsilon, xi_hat, deviations, ETA, c, d)
        t.append(time.perf_counter() - t0)
    woodbury_ms = 1e3 * np.median(t)

    # 内訳: einsum(G) / cholesky / solve / logdet / cond
    alpha = ETA ** 2 / (B * (B - 1))
    v = epsilon - ETA * xi_hat

    def _time(fn, reps=reps):
        tt = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            tt.append(time.perf_counter() - t0)
        return 1e3 * np.median(tt)

    G_hold = {}

    def _einsum():
        G = alpha * np.einsum("nbd,ncd->nbc", deviations, deviations)
        M = G / c
        M[:, np.arange(B), np.arange(B)] += 1.0
        G_hold["M"] = M
    einsum_ms = _time(_einsum)
    M = G_hold["M"]
    p = np.sqrt(alpha) * np.einsum("nbd,nd->nb", deviations, v)

    chol_ms = _time(lambda: np.linalg.cholesky(M))
    solve_ms = _time(lambda: np.linalg.solve(M, p[:, :, None]))
    L = np.linalg.cholesky(M)
    logdet_ms = _time(
        lambda: 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
    )
    cond_ms = _time(lambda: np.linalg.cond(M))

    # 素朴な密行列 d×d: V̂p を構成→slogdet+solve(粒子ごと O(d³))
    # メモリ・時間が大きいため少数粒子で計測し N にスケール
    N_dense = min(N, 8)
    dense_per_particle = []
    Iy = np.eye(d)
    for n in range(N_dense):
        W = deviations[n]
        t0 = time.perf_counter()
        Sigma = (1.0 / (B * (B - 1))) * (W.T @ W)   # d×d
        Vp = ETA ** 2 * Sigma + c * Iy              # d×d
        np.linalg.slogdet(Vp)
        np.linalg.solve(Vp, v[n])
        dense_per_particle.append(time.perf_counter() - t0)
    dense_ms_per_step = 1e3 * float(np.median(dense_per_particle)) * N

    return {
        "woodbury_ms_per_step": woodbury_ms,
        "einsum_ms": einsum_ms,
        "cholesky_ms": chol_ms,
        "solve_ms": solve_ms,
        "logdet_ms": logdet_ms,
        "cond_ms": cond_ms,
        "dense_ms_per_step_extrap": dense_ms_per_step,
        "N_dense_measured": N_dense,
        "speedup_vs_dense": dense_ms_per_step / max(woodbury_ms, 1e-9),
    }


def memory_report(d, N, B):
    """WSPF-A の行列メモリを Woodbury(B×B) と 素朴(d×d) で対比。"""
    fbytes = 8  # float64
    # Woodbury で確保する主な一時行列
    dev = N * B * d * fbytes           # deviations (N,B,d)
    G_M = 2 * N * B * B * fbytes       # G, M (N,B,B)
    p_vec = N * B * fbytes
    woodbury_bytes = dev + G_M + p_vec
    # 素朴に per-particle d×d を保持した場合
    dense_bytes = N * d * d * fbytes   # Σ̂ or V̂p (N,d,d)
    # 永続状態(参考)
    persistent = 2 * N * d * fbytes    # particles + ema_m
    return {
        "woodbury_transient_MB": woodbury_bytes / 1e6,
        "woodbury_BxB_MB": G_M / 1e6,
        "woodbury_deviations_MB": dev / 1e6,
        "dense_dxd_MB": dense_bytes / 1e6,
        "persistent_state_MB": persistent / 1e6,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print("Compute-cost benchmark (R1-11)")
    print(f"  d(param)={INPUT_DIM*HIDDEN_DIM+HIDDEN_DIM+HIDDEN_DIM*OUTPUT_DIM+OUTPUT_DIM}"
          f", N={N_PARTICLES}, B={BATCH_SIZE}, "
          f"measure_steps={MEASURE_STEPS}")
    print("=" * 72)

    model = NeuralNetModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    d = model.param_dim
    grad_fn = create_nn_grad_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)

    common = dict(n_particles=N_PARTICLES, param_dim=d, eta=ETA,
                  sigma_sys=SIGMA_SYS, prior_mean=0.0, prior_std=0.1,
                  ess_resample_ratio=0.5)

    pf = ParticleFilter(**common, seed=SEED + 1)
    wspf_b = WSPF_B(**common, grad_clip_norm=GRAD_CLIP, seed=SEED + 3)
    wspf_a = WSPF_A(**common, grad_clip_norm=GRAD_CLIP, beta=BETA, seed=SEED + 5)

    rng = np.random.default_rng(SEED)
    results = []
    results.append(run_filter_bench("PF", pf, grad_fn, loglik_fn, model, rng))
    results.append(run_filter_bench("WSPF-B", wspf_b, ps_grad_fn, loglik_fn, model, rng))
    results.append(run_filter_bench("WSPF-A", wspf_a, ps_grad_fn, loglik_fn, model, rng))

    linalg = bench_wspf_a_linalg(d, N_PARTICLES, BATCH_SIZE)
    mem = memory_report(d, N_PARTICLES, BATCH_SIZE)

    # ---- レポート ----
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit(f"{'method':<8s} {'kind':<11s} {'grad_evals/step':>15s} "
         f"{'t_step(ms)':>11s} {'t_grad':>8s} {'t_corr':>8s} "
         f"{'t_llik':>8s} {'t_wt':>7s} {'t_rs':>7s}")
    emit("-" * 92)
    for r in results:
        t = r["timing"]
        emit(f"{r['name']:<8s} {r['grad_eval_kind']:<11s} "
             f"{r['sample_grad_evals_per_step']:>15d} "
             f"{t['t_step']['median_ms']:>11.3f} "
             f"{t['t_grad']['median_ms']:>8.3f} "
             f"{t['t_correction']['median_ms']:>8.3f} "
             f"{t['t_loglik']['median_ms']:>8.3f} "
             f"{t['t_weight']['median_ms']:>7.3f} "
             f"{t['t_resample']['median_ms']:>7.3f}")

    emit()
    emit("WSPF-A 線形代数の内訳(中央値, ms/step) — 行列補正コスト (R1-10/R1-11):")
    emit(f"  einsum(G=αWWᵀ) : {linalg['einsum_ms']:.3f}")
    emit(f"  cholesky(M)    : {linalg['cholesky_ms']:.3f}")
    emit(f"  solve(M⁻¹p)    : {linalg['solve_ms']:.3f}")
    emit(f"  logdet(M)      : {linalg['logdet_ms']:.3f}")
    emit(f"  cond(M) 監視    : {linalg['cond_ms']:.3f}")
    emit(f"  Woodbury 合計   : {linalg['woodbury_ms_per_step']:.3f}")
    emit(f"  素朴 d×d 直接反転(O(d³), N={N_PARTICLES}へ外挿, "
         f"実測{linalg['N_dense_measured']}粒子): "
         f"{linalg['dense_ms_per_step_extrap']:.1f}")
    emit(f"  → Woodbury は約 {linalg['speedup_vs_dense']:.0f}× 高速")

    emit()
    emit("メモリ(行列保持, R1-11):")
    emit(f"  素朴 d×d 保持 (N·d²)         : {mem['dense_dxd_MB']:.1f} MB "
         f"(R1-11 指摘の ~555MB に相当)")
    emit(f"  Woodbury 一時行列 合計       : {mem['woodbury_transient_MB']:.1f} MB")
    emit(f"    ├ B×B 行列 (G,M)          : {mem['woodbury_BxB_MB']:.3f} MB")
    emit(f"    └ 偏差 (N·B·d)            : {mem['woodbury_deviations_MB']:.1f} MB")
    emit(f"  永続状態 (particles+EMA)     : {mem['persistent_state_MB']:.1f} MB")
    emit(f"  → d×d を陽に保持せず {mem['dense_dxd_MB']/mem['woodbury_transient_MB']:.0f}× 削減")

    # ---- 保存 ----
    txt_path = os.path.join(OUTPUT_DIR, "compute_cost_benchmark.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    csv_path = os.path.join(OUTPUT_DIR, "compute_cost_benchmark.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,grad_eval_kind,grad_evals_per_step,"
                "t_step_ms,t_grad_ms,t_correction_ms,t_loglik_ms,"
                "t_weight_ms,t_resample_ms\n")
        for r in results:
            t = r["timing"]
            f.write(f"{r['name']},{r['grad_eval_kind']},"
                    f"{r['sample_grad_evals_per_step']},"
                    f"{t['t_step']['median_ms']:.4f},"
                    f"{t['t_grad']['median_ms']:.4f},"
                    f"{t['t_correction']['median_ms']:.4f},"
                    f"{t['t_loglik']['median_ms']:.4f},"
                    f"{t['t_weight']['median_ms']:.4f},"
                    f"{t['t_resample']['median_ms']:.4f}\n")

    emit()
    emit(f"Saved: {txt_path}")
    emit(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
