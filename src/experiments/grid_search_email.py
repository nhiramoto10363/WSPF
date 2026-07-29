#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email binary: PF / WSPF-B / WSPF-A grid search

Email (elist) データセットを使用し、
eta, sigma_sys, prior_std (+ WSPF-A の beta) のグリッドサーチを行う。

主な特徴:
- worker 初期化時に X, y を共有メモリとして渡す
- float32 化でメモリ削減
- BLAS スレッド数を 1 に制限（マルチプロセス並列のため）
- task を chunk 化して ProcessPoolExecutor のオーバーヘッドを削減
"""

import os
# numpy より前に設定
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

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
# 固定パラメータ
# ================================================================
#N_PARTICLES_LIST = [5, 10, 50]
N_PARTICLES_LIST = [100]


HIDDEN_DIM = 16
BATCH_SIZE = 16
MAX_GRAD_NORM = 5.0
SEED = 42
TEST_SIZE = 32
PCA_DIM = 50

# 並列化設定
MAX_WORKERS = 20
TASK_CHUNK_SIZE = 8

# データ
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Email",
    "email_data.arff",
)

# ================================================================
# グリッドサーチ設定
# ================================================================
GRID_ETA = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]
GRID_SIGMA_SYS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1,0.125,0.15,0.175,0.2]
GRID_PRIOR_STD = [0.01,0.1, 0.25,0.5,0.75,1.0]
GRID_BETA = [0.1,0.3,0.5, 0.9, 0.95]  # WSPF-A 専用

# リーク除去(R1-13): warm-up/validation 区間(期1-2 = サンプル 0-599,
# ドリフト@300 を1回含む)でのみ PCA 学習とハイパラ選択を行う。
# 評価/報告(期3-5 = 600-1499)は email_binary_experiment.py 側で分離する。
PCA_FIT_END = 600        # PCA(mean+主成分)の学習に使う先頭区間の終端
SELECT_END = 600         # グリッドサーチの F1 評価をこの区間内に限定
# (1500 - 32) / 16 ≈ 91 ステップ(うち選択区間は te1<=600 のみ)
GRID_SEARCH_STEPS = 500  # 実質 SELECT_END で打ち切られる
GRID_EVAL_START = 3      # 選択区間が短いので cold-start をごく短く飛ばす

# 出力先
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "email_binary",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "grid_search_result.json")

# ================================================================
# worker 共有グローバル
# ================================================================
_X = None
_y = None
_INPUT_DIM = None
_N_SAMPLES = None
_BATCH_WINDOWS = None


def _init_worker(X, y, input_dim, n_samples, batch_windows):
    """各 worker 起動時に 1 回だけ呼ばれる"""
    global _X, _y, _INPUT_DIM, _N_SAMPLES, _BATCH_WINDOWS
    _X = X
    _y = y
    _INPUT_DIM = input_dim
    _N_SAMPLES = n_samples
    _BATCH_WINDOWS = batch_windows


# ================================================================
# 補助関数
# ================================================================
def _compute_f1_binary(pred, y):
    """positive クラス (y=1) の F1"""
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def _make_batch_windows(n_samples, batch_size, test_size, max_steps, eval_start,
                        select_end=None):
    """各 step の train/test 範囲を事前計算。

    select_end を指定すると、test 区間が select_end を超えるステップで打ち切る
    (グリッドサーチのフィルタが選択区間=warm-up/validation のみを走るようにし、
    評価区間のデータに一切触れないようにするため)。
    """
    windows = []
    pos = 0
    for step in range(max_steps):
        if pos + batch_size + test_size > n_samples:
            break
        tr0 = pos
        tr1 = pos + batch_size
        te0 = tr1
        te1 = tr1 + test_size
        if select_end is not None and te1 > select_end:
            break
        do_eval = (step >= eval_start)
        windows.append((tr0, tr1, te0, te1, do_eval))
        pos += batch_size
    return windows


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ================================================================
# 候補 1 個の評価
# ================================================================
def _evaluate_candidate(spec):
    """
    spec = (method, eta, sigma_sys, prior_std, beta, n_particles)
    """
    method, eta, sigma_sys, prior_std, beta, n_particles = spec

    X_data = _X
    y_data = _y
    input_dim = _INPUT_DIM

    model = NeuralNetModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=1,
        activation="tanh",
    )
    param_dim = model.param_dim

    grad_fn_raw = create_nn_grad_fn(model)
    loglik_fn = create_nn_loglik_fn(model)
    ps_grad_fn = create_nn_per_sample_grad_fn(model)

    def clipped_grad_fn(particles, X, y, _raw=grad_fn_raw):
        g = _raw(particles, X, y)
        norms = np.linalg.norm(g, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_GRAD_NORM / (norms + 1e-8))
        return g * scale

    common_kw = dict(
        n_particles=n_particles,
        param_dim=param_dim,
        eta=eta,
        sigma_sys=sigma_sys,
        prior_mean=0.0,
        prior_std=prior_std,
        ess_resample_ratio=0.5,
    )

    if method == "WSPF-B":
        filt = WSPF_B(
            **common_kw,
            grad_clip_norm=MAX_GRAD_NORM,
            seed=SEED + 3,
        )
    elif method == "WSPF-A":
        filt = WSPF_A(
            **common_kw,
            grad_clip_norm=MAX_GRAD_NORM,
            beta=beta,
            seed=SEED + 5,
        )
    else:
        filt = ParticleFilter(
            **common_kw,
            seed=SEED + 1,
        )

    acc_sum = 0.0
    f1_sum = 0.0
    n_eval = 0

    for tr0, tr1, te0, te1, do_eval in _BATCH_WINDOWS:
        X_train = X_data[tr0:tr1]
        y_train = y_data[tr0:tr1]
        X_test = X_data[te0:te1]
        y_test = y_data[te0:te1]

        if do_eval:
            mu = (filt.weights[:, None] * filt.particles).sum(axis=0)
            output, _, _ = model.forward(mu.reshape(1, -1), X_test)
            pred = (output.reshape(-1) > 0.5).astype(np.float32)

            acc_sum += float(np.mean(pred == y_test))
            f1_sum += _compute_f1_binary(pred, y_test)
            n_eval += 1

        if method in ("WSPF-B", "WSPF-A"):
            filt.step(X_train, y_train, ps_grad_fn, loglik_fn)
        else:
            filt.step(X_train, y_train, clipped_grad_fn, loglik_fn)

    mean_acc = acc_sum / n_eval if n_eval > 0 else 0.0
    mean_f1 = f1_sum / n_eval if n_eval > 0 else 0.0

    return {
        "method": method,
        "n_particles": n_particles,
        "eta": eta,
        "sigma_sys": sigma_sys,
        "prior_std": prior_std,
        "beta": beta,
        "acc": mean_acc,
        "f1": mean_f1,
    }


def _evaluate_chunk(spec_chunk):
    """候補の小さなまとまりを 1 future として評価"""
    out = []
    for spec in spec_chunk:
        out.append(_evaluate_candidate(spec))
    return out


# ================================================================
# メイン
# ================================================================
def main():
    print("=" * 70)
    print("Grid Search: Email Binary (PF / WSPF-B / WSPF-A)")
    print(f"PCA dim: {PCA_DIM}, batch: {BATCH_SIZE}, test: {TEST_SIZE}")
    print("=" * 70)

    print("\nLoading data...")
    loader = EmailDataLoader(
        DATA_PATH, n_components=PCA_DIM, seed=SEED, pca_fit_end=PCA_FIT_END,
    )

    # float32 化でメモリ削減
    X = np.asarray(loader.X, dtype=np.float32, order="C")
    y = np.asarray(loader.y, dtype=np.float32, order="C")

    input_dim = int(loader.input_dim)
    n_samples = int(loader.n_samples)

    print(f"  Samples: {n_samples:,}, Features: {input_dim}")
    interesting_ratio = y.sum() / n_samples
    print(f"  Interesting ratio: {interesting_ratio:.3f}")

    batch_windows = _make_batch_windows(
        n_samples=n_samples,
        batch_size=BATCH_SIZE,
        test_size=TEST_SIZE,
        max_steps=GRID_SEARCH_STEPS,
        eval_start=GRID_EVAL_START,
        select_end=SELECT_END,
    )
    actual_steps = len(batch_windows)
    actual_eval_steps = sum(1 for _, _, _, _, e in batch_windows if e)
    print(f"  Actual steps: {actual_steps}, eval steps: {actual_eval_steps}")

    # 候補
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

    # task 作成
    task_specs = []
    for n_p in N_PARTICLES_LIST:
        for method in ["PF", "WSPF-B"]:
            for eta, ss, ps in base_candidates:
                task_specs.append((method, eta, ss, ps, None, n_p))
        for eta, ss, ps, beta in wspf_a_candidates:
            task_specs.append(("WSPF-A", eta, ss, ps, beta, n_p))

    # 重いものを先に回して負荷分散
    task_specs.sort(key=lambda x: x[5], reverse=True)

    task_chunks = list(_chunked(task_specs, TASK_CHUNK_SIZE))

    # 結果格納
    results_by_n = {
        n_p: {"PF": [], "WSPF-B": [], "WSPF-A": []}
        for n_p in N_PARTICLES_LIST
    }

    n_workers = min(os.cpu_count() or 1, MAX_WORKERS)

    print(f"\n  eta:         {GRID_ETA}")
    print(f"  sigma_sys:   {GRID_SIGMA_SYS}")
    print(f"  prior_std:   {GRID_PRIOR_STD}")
    print(f"  beta:        {GRID_BETA}  (WSPF-A only)")
    print(f"  N_particles: {N_PARTICLES_LIST}")
    print(f"  steps:       {actual_steps}, eval_start: {GRID_EVAL_START}")
    print(f"  workers:     {n_workers}")
    print(f"  candidates per N: PF={n_base}, WSPF-B={n_base}, "
          f"WSPF-A={n_wspf_a}, total={n_per_N}")
    print(f"  total jobs:  {total}")
    print(f"  chunk size:  {TASK_CHUNK_SIZE}")
    print()

    # Linux なら fork の方が共有しやすい
    if sys.platform.startswith("linux"):
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context("spawn")

    t0 = time.time()
    done_count = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(X, y, input_dim, n_samples, batch_windows),
    ) as executor:
        futures = [executor.submit(_evaluate_chunk, ch) for ch in task_chunks]

        for future in as_completed(futures):
            chunk_results = future.result()
            for r in chunk_results:
                done_count += 1

                method = r["method"]
                n_particles = r["n_particles"]

                row = {
                    "eta": r["eta"],
                    "sigma_sys": r["sigma_sys"],
                    "prior_std": r["prior_std"],
                    "acc": r["acc"],
                    "f1": r["f1"],
                }
                beta_str = ""
                if method == "WSPF-A":
                    row["beta"] = r["beta"]
                    beta_str = f" beta={r['beta']:.2f}"

                results_by_n[n_particles][method].append(row)

                if done_count % 50 == 0 or done_count == total:
                    print(
                        f"  [{done_count:4d}/{total}] "
                        f"{method:5s} N={n_particles:5d} "
                        f"eta={r['eta']:.3f} sigma_sys={r['sigma_sys']:.3f} "
                        f"prior={r['prior_std']:.2f}{beta_str} "
                        f"-> Acc={r['acc']:.4f} F1={r['f1']:.4f}"
                    )

    elapsed = time.time() - t0
    print(f"\nGrid search completed in {elapsed:.1f}s ({n_workers} workers)")

    # JSON 出力
    output_data = {
        "dataset": "Email-elist",
        "pca_dim": PCA_DIM,
        "leak_free": {
            "pca_fit_end": PCA_FIT_END,
            "select_end": SELECT_END,
            "note": "PCA fit と HP 選択は warm-up/val 区間(期1-2, 0-599)のみ。"
                    "報告は期3-5(600-1499)を email_binary_experiment.py で分離。",
        },
        "grid": {
            "eta": GRID_ETA,
            "sigma_sys": GRID_SIGMA_SYS,
            "prior_std": GRID_PRIOR_STD,
            "beta": GRID_BETA,
            "n_particles": N_PARTICLES_LIST,
            "steps": actual_steps,
            "eval_start": GRID_EVAL_START,
        },
        "by_n_particles": {},
    }

    for n_p in N_PARTICLES_LIST:
        for method in ["PF", "WSPF-B", "WSPF-A"]:
            results_by_n[n_p][method].sort(key=lambda r: -r["f1"])

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

        print(f"\n{'=' * 70}")
        print(f"N = {n_p}")
        print(f"{'=' * 70}")

        for label, key in [("PF", "PF"), ("WSPF-B", "WSPF-B")]:
            print(f"\n--- {label} top 5 ---")
            print(f"  {'eta':>5s}  {'sigma_sys':>9s}  {'prior':>6s}  "
                  f"{'Acc':>8s}  {'F1':>8s}")
            print("  " + "-" * 42)
            for i, r in enumerate(results_by_n[n_p][key][:5]):
                mark = " <--" if i == 0 else ""
                print(
                    f"  {r['eta']:5.3f}  {r['sigma_sys']:9.3f}  "
                    f"{r['prior_std']:6.2f}  {r['acc']:8.4f}  "
                    f"{r['f1']:8.4f}{mark}"
                )

        print(f"\n--- WSPF-A top 5 ---")
        print(f"  {'eta':>5s}  {'sigma_sys':>9s}  {'prior':>6s}  "
              f"{'beta':>6s}  {'Acc':>8s}  {'F1':>8s}")
        print("  " + "-" * 52)
        for i, r in enumerate(results_by_n[n_p]["WSPF-A"][:5]):
            mark = " <--" if i == 0 else ""
            print(
                f"  {r['eta']:5.3f}  {r['sigma_sys']:9.3f}  "
                f"{r['prior_std']:6.2f}  {r['beta']:6.2f}  "
                f"{r['acc']:8.4f}  {r['f1']:8.4f}{mark}"
            )

        print(f"\n  Best PF:    {best_pf}  "
              f"(F1={results_by_n[n_p]['PF'][0]['f1']:.4f})")
        print(f"  Best WSPF-B:   {best_wspf_b}  "
              f"(F1={results_by_n[n_p]['WSPF-B'][0]['f1']:.4f})")
        print(f"  Best WSPF-A: {best_wspf_a}  "
              f"(F1={results_by_n[n_p]['WSPF-A'][0]['f1']:.4f})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fp:
        json.dump(output_data, fp, indent=2)

    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
