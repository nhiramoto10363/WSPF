#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ミニバッチ平均勾配ノイズ解析 (R1-6)

査読対応(R1-6): 解析対象は **ミニバッチ平均勾配** のサンプリングノイズ
    ĝ(θ; B) − ∇L(θ)
である(旧実装は 1 バッチの per-sample 勾配を θ=0 で見ており誤り)。

対象は Regression のみ(真のデータ生成分布から反復サンプリングでき、
母集団勾配 ∇L(θ) を高精度に推定できる)。評価点 θ は各位相の真パラメータ
θ*_phase に固定し、バッチサイズ B と概念切替からの距離(位相)を変えて、
ミニバッチ平均勾配ノイズのガウス近似・異方性・バッチサイズ依存を検証する。

  batch_sizes = [8, 16, 32, 64]
  phases      = {stable: sp-30, post+0: sp, post+5: sp+5, post+20: sp+20}
                (sp = 最初の switch point。範囲外は T でガード)

各条件 (B, phase) で:
  1. 母集団勾配 ∇L(θ*_phase) を ~100_000 の MC 標本で高精度推定
     (真分布 x~N(0,1), y=f(θ*;x)+N(0,σ²) から引き、θ=θ*_phase で評価)。
  2. 同じ真分布から独立に 3000 個のサイズ B ミニバッチを引き、各々の
     バッチ平均勾配 ĝ から ノイズ noise = ĝ − ∇L を集める → noises (3000, d)。
  3. |歪度|平均・尖度平均・正規性検定, Mahalanobis 距離分布(χ²(d) との KS 検定),
     共分散固有値(condition number, effective rank, 上位固有値比率)を報告。

メモリは 100k 母集団推定・3000 ミニバッチともチャンク処理で有界に保つ。
スモークテスト用に環境変数 WSPF_GN_POP / WSPF_GN_BATCHES で標本数を縮小可。

使い方:
    python scripts/analyze_gradient_noise.py --benchmark regression
"""

import argparse
import os

import numpy as np
from scipy import stats

from _common import load_config, build_benchmark
from src.evaluation import write_table
from src.models import create_regression_per_sample_grad_fn

# 母集団勾配推定の MC 標本数 / 独立ミニバッチ数 / チャンク幅
POP_SAMPLES = int(os.environ.get("WSPF_GN_POP", 100_000))
N_BATCHES = int(os.environ.get("WSPF_GN_BATCHES", 3000))
POP_CHUNK = 5000

BATCH_SIZES = [8, 16, 32, 64]


def _phase_steps(switch_points, T):
    """代表的な位相ステップ (stable / post+0 / post+5 / post+20) を返す。"""
    sp = switch_points[0] if switch_points else T // 2
    return {
        "stable": max(sp - 30, 0),
        "post+0": min(sp, T - 1),
        "post+5": min(sp + 5, T - 1),
        "post+20": min(sp + 20, T - 1),
    }


def _population_grad(model, per_sample_grad, theta_star, theta_eval,
                     noise_std, rng, n_samples=POP_SAMPLES, chunk=POP_CHUNK):
    """真分布 (x~N(0,1), y=f(θ*;x)+N(0,σ²)) 上で ∇L(θ_eval)=E[∇ℓ] を MC 推定。

    θ_eval は単一評価粒子 (1, d)。per-sample 勾配 (1, m, d) を squeeze して
    (m, d) とし、全標本平均を取る。チャンク処理で (m, d) の一括確保を避ける。
    """
    d = theta_eval.shape[1]
    sum_g = np.zeros(d)
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = per_sample_grad(theta_eval, X, y)[0]          # (m, d)
        sum_g += g.sum(axis=0)
        done += m
    return sum_g / n_samples                               # ∇L (d,)


def _minibatch_noises(model, per_sample_grad, theta_star, theta_eval,
                      noise_std, B, pop_grad, rng,
                      n_batches=N_BATCHES, chunk=POP_CHUNK):
    """サイズ B の独立ミニバッチを n_batches 個引き、ノイズ ĝ−∇L を集める。

    返り値 noises: (n_batches, d)。チャンク単位でミニバッチをまとめて引くが、
    保持するのはバッチ平均のみなのでメモリは有界。
    """
    d = theta_eval.shape[1]
    noises = np.empty((n_batches, d))
    mb_per_chunk = max(1, chunk // B)     # 1 チャンクで扱うミニバッチ数
    bi = 0
    while bi < n_batches:
        nb = min(mb_per_chunk, n_batches - bi)
        m = nb * B
        X = rng.normal(0.0, 1.0, size=(m, model.input_dim))
        out, _, _ = model.forward(theta_star.reshape(1, -1), X)
        y = out.squeeze() + rng.normal(0.0, noise_std, size=m)
        g = per_sample_grad(theta_eval, X, y)[0]          # (m, d)
        gbar = g.reshape(nb, B, d).mean(axis=1)           # (nb, d) バッチ平均
        noises[bi:bi + nb] = gbar - pop_grad[None, :]
        bi += nb
    return noises


def _noise_stats(noises):
    """ミニバッチ平均勾配ノイズ (n, d) から R1-6 の各種統計を計算する。"""
    n, d = noises.shape

    # 各成分の |歪度| 平均(符号平均は相殺するため絶対値平均: 査読要件) と 尖度平均
    mean_abs_skew = float(np.mean(np.abs(stats.skew(noises, axis=0))))
    mean_kurtosis = float(np.mean(stats.kurtosis(noises, axis=0)))

    # 標本共分散(可逆化のため微小ジッタを加える)
    Sigma = np.cov(noises, rowvar=False)
    if Sigma.ndim == 0:                      # d==1 の保険
        Sigma = Sigma.reshape(1, 1)
    jitter = 1e-8 * (np.trace(Sigma) / d + 1e-30)
    Sigma_j = Sigma + jitter * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma_j)

    # Mahalanobis 距離² と χ²(df=d) との比較
    md2 = np.einsum("ni,ij,nj->n", noises, Sigma_inv, noises)   # (n,)
    mean_maha2 = float(np.mean(md2))
    ks = stats.kstest(md2, "chi2", args=(d,))
    maha_ks_p = float(ks.pvalue)

    # 共分散固有値の要約
    evals = np.linalg.eigvalsh(Sigma_j)
    evals = np.clip(evals, 0.0, None)[::-1]
    total = evals.sum() + 1e-30
    cond = float(evals[0] / max(evals[-1], 1e-30))
    eff_rank = float((evals.sum() ** 2) / (np.sum(evals ** 2) + 1e-30))
    top_ratio = float(evals[0] / total)

    # 正規性検定 (成分別 p の平均、n>=20 のときのみ)
    if n >= 20:
        pvals = [stats.normaltest(noises[:, j]).pvalue for j in range(d)]
        normaltest_p = float(np.nanmean(pvals))
    else:
        normaltest_p = float("nan")

    return {
        "mean_abs_skew": mean_abs_skew,
        "mean_kurtosis": mean_kurtosis,
        "mean_maha2": mean_maha2,
        "expected_maha2": float(d),
        "maha_ks_p": maha_ks_p,
        "cond_number": cond,
        "effective_rank": eff_rank,
        "top_eig_ratio": top_ratio,
        "normaltest_p": normaltest_p,
    }


def analyze(cfg, seed=0):
    """(B, phase) ごとのミニバッチ平均勾配ノイズ統計行のリストを返す。"""
    bench = build_benchmark(cfg)
    model = bench.model
    noise_std = bench.noise_std
    d = bench.param_dim

    # seed の真パラメータ列 theta_true を確定させる
    bench.build_functions(seed=seed)
    theta_true = bench.theta_true
    per_sample_grad = create_regression_per_sample_grad_fn(model, noise_std)

    phase_steps = _phase_steps(bench.switch_points, bench.T)
    rng = np.random.default_rng(12345)

    rows = []
    pop_cache = {}    # step -> ∇L(θ*_step)  (B に依らないので位相ごとに再利用)
    for B in BATCH_SIZES:
        for phase, step in phase_steps.items():
            theta_star = theta_true[step]
            theta_eval = theta_star.reshape(1, d)     # 単一評価粒子 (1, d)
            if step not in pop_cache:
                pop_cache[step] = _population_grad(
                    model, per_sample_grad, theta_star, theta_eval,
                    noise_std, rng)
            pop_grad = pop_cache[step]

            noises = _minibatch_noises(
                model, per_sample_grad, theta_star, theta_eval,
                noise_std, B, pop_grad, rng)
            st = _noise_stats(noises)
            st.update({"batch_size": B, "phase": phase})
            rows.append(st)
            print(f"B={B:3d} {phase:8s} "
                  f"|skew|={st['mean_abs_skew']:.3f} "
                  f"kurt={st['mean_kurtosis']:.3f} "
                  f"maha2={st['mean_maha2']:.2f}/{d} "
                  f"ks_p={st['maha_ks_p']:.3f} "
                  f"cond={st['cond_number']:.1f} "
                  f"eff_rank={st['effective_rank']:.1f} "
                  f"norm_p={st['normaltest_p']:.3f}")
    return rows


# 出力列(査読 R1-6 で要求された指標)
_COLUMNS = ["batch_size", "phase", "mean_abs_skew", "mean_kurtosis",
            "mean_maha2", "expected_maha2", "maha_ks_p", "cond_number",
            "effective_rank", "top_eig_ratio", "normaltest_p"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="regression",
                    help="regression または config パス(Regression のみ対応)")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    if cfg["task_type"] != "regression":
        raise SystemExit("R1-6 勾配ノイズ解析は Regression のみ対応です。")

    rows = analyze(cfg, seed=0)
    rows = [{k: r[k] for k in _COLUMNS} for r in rows]   # 列順を固定

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "grad_noise")
    os.makedirs(out_dir, exist_ok=True)
    write_table(rows, os.path.join(out_dir, "gradient_noise"))
    print(f"保存: {out_dir}")


if __name__ == "__main__":
    main()
