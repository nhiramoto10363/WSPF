#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSECTS (abrupt balanced) グリッドサーチ

email 版 (grid_search_email.py) の鏡像。設計方針:
- リークフリー: 標準化 fit とハイパラ選択は選択区間 [0, SELECT_END) のみ。
  SELECT_END はレジーム 1-2 の終端 (= 2 番目のスイッチ点 19500) とし、
  選択区間に 1 つのスイッチ (14352) を含める (email の期1-2 と同じ発想)。
- SGD は独自に η を 1 次元探索し best_sgd として保存 (R1-7 交絡の除去)。
- best が η グリッドの端点なら WARNING を出力 (内点確認, R1-minor)。
- JSON スキーマは email/regression と同一
  (by_n_particles -> best_sgd / best_pf / best_wspf_b / best_wspf_a / all_*)。

選択指標: macro-F1 (選択区間の test window 平均)。

実行時間の目安: 選択区間 ~1219 step。PF ~60 構成, WSPF-B ~60,
WSPF-A ~120 (β 2 値)。d≈1286 (h=32) で WSPF-A は ~0.1s/step 程度。
48 並列で全体 10-20 分を想定。
"""

import sys
import os
import json
import time
import itertools
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.data.insects_loader import InsectsDataLoader
from src.filters import ParticleFilter
from src.filters.wspf_b import WSPF_B
from src.filters.wspf_a import WSPF_A
from src.models.neural_net_multiclass import (
    MulticlassNeuralNetModel,
    create_mc_grad_fn,
    create_mc_loglik_fn,
    create_mc_per_sample_grad_fn,
)

# ================================================================
# 設定
# ================================================================
N_PARTICLES_LIST = [100]

HIDDEN_DIM = 32          # d = (33+1)*32 + (32+1)*6 = 1286
BATCH_SIZE = 16
TEST_SIZE = 32
MAX_GRAD_NORM = 5.0
SEED = 42                # 選択用シード (multiseed 評価の 0-9 と disjoint)

SELECT_END = 19500       # 選択区間 = レジーム 1-2 (スイッチ 14352 を含む)
SCALE_FIT_END = SELECT_END

# 粗めグリッド (INSECTS は email の ~35 倍長いストリームのため)
GRID_ETA = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
GRID_SIGMA_SYS = [0.005, 0.01, 0.025, 0.05, 0.1]
GRID_PRIOR_STD = [0.1, 0.5]
GRID_BETA = [0.9, 0.95]

MAX_WORKERS = int(os.environ.get("NCPUS", os.cpu_count() or 1))

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "INSECTS",
    "INSECTS-abrupt_balanced_norm.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "insects",
)
GRID_JSON = os.path.join(OUTPUT_DIR, "grid_search_result.json")

# ================================================================
# worker 共有状態
# ================================================================
_G = {}


def _init_worker(X, y, n_classes, windows):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _G["X"], _G["y"] = X, y
    _G["n_classes"] = n_classes
    _G["windows"] = windows


def _make_windows(n_select):
    """選択区間内の (train_start,) リスト。test も選択区間内に収まる window のみ"""
    wins = []
    pos = 0
    while pos + BATCH_SIZE + TEST_SIZE <= n_select:
        wins.append(pos)
        pos += BATCH_SIZE
    return wins


def _macro_f1(pred, y, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(0.0 if prec + rec == 0
                   else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def _clip_gradients(grad, max_norm):
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    return grad * scale


def _evaluate_sgd(eta, prior_std):
    """SGD 独自評価 (η, σ0 の 2 パラメータ)"""
    X, y = _G["X"], _G["y"]
    model = MulticlassNeuralNetModel(X.shape[1], HIDDEN_DIM, _G["n_classes"])
    grad_fn = create_mc_grad_fn(model)
    rng = np.random.default_rng(SEED + 10)
    theta = rng.normal(0.0, prior_std, size=model.param_dim)
    f1s = []
    for pos in _G["windows"]:
        Xtr = X[pos: pos + BATCH_SIZE]
        ytr = y[pos: pos + BATCH_SIZE]
        Xte = X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        yte = y[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        pred = model.predict(theta, Xte)[0]
        f1s.append(_macro_f1(pred, yte, _G["n_classes"]))
        g = _clip_gradients(
            grad_fn(theta.reshape(1, -1), Xtr, ytr), MAX_GRAD_NORM).squeeze()
        theta = theta - eta * g
    return {"eta": eta, "prior_std": prior_std, "f1": float(np.mean(f1s))}


def _evaluate_candidate(spec):
    """PF / WSPF-B / WSPF-A の 1 構成を選択区間で評価"""
    method, n_p, eta, sigma_sys, prior_std, beta = spec
    X, y = _G["X"], _G["y"]
    model = MulticlassNeuralNetModel(X.shape[1], HIDDEN_DIM, _G["n_classes"])
    grad_fn_raw = create_mc_grad_fn(model)
    loglik_fn = create_mc_loglik_fn(model)
    ps_grad_fn = create_mc_per_sample_grad_fn(model)

    def clipped_grad_fn(particles, Xb, yb, _raw=grad_fn_raw):
        return _clip_gradients(_raw(particles, Xb, yb), MAX_GRAD_NORM)

    common = dict(n_particles=n_p, param_dim=model.param_dim, eta=eta,
                  sigma_sys=sigma_sys, prior_mean=0.0, prior_std=prior_std,
                  ess_resample_ratio=0.5)
    if method == "pf":
        filt = ParticleFilter(**common, seed=SEED + 1)
    elif method == "wspf_b":
        filt = WSPF_B(**common, grad_clip_norm=MAX_GRAD_NORM, seed=SEED + 3)
    else:
        filt = WSPF_A(**common, grad_clip_norm=MAX_GRAD_NORM,
                      beta=beta, seed=SEED + 5)

    f1s = []
    for pos in _G["windows"]:
        Xtr = X[pos: pos + BATCH_SIZE]
        ytr = y[pos: pos + BATCH_SIZE]
        Xte = X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        yte = y[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        mu = (filt.weights[:, None] * filt.particles).sum(axis=0)
        pred = model.predict(mu, Xte)[0]
        f1s.append(_macro_f1(pred, yte, _G["n_classes"]))
        if method == "pf":
            filt.step(Xtr, ytr, clipped_grad_fn, loglik_fn)
        else:
            filt.step(Xtr, ytr, ps_grad_fn, loglik_fn)

    return {"method": method, "n_particles": n_p, "eta": eta,
            "sigma_sys": sigma_sys, "prior_std": prior_std,
            "beta": beta, "f1": float(np.mean(f1s))}


def _warn_if_boundary(name, best, grid_eta):
    if best["eta"] in (min(grid_eta), max(grid_eta)):
        print(f"  *** WARNING: {name} best eta={best['eta']} は "
              f"グリッド端点です。グリッド延長を検討してください ***")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    loader = InsectsDataLoader(
        DATA_PATH, scale_fit_end=SCALE_FIT_END, seed=SEED)
    loader.print_regime_class_distribution()

    if SELECT_END not in ([0] + loader.change_points + [loader.n_samples]):
        print(f"  *** WARNING: SELECT_END={SELECT_END} がスイッチ点と "
              f"一致しません: {loader.change_points} ***")

    X = loader.X[:SELECT_END]
    y = loader.y[:SELECT_END]
    windows = _make_windows(SELECT_END)
    print(f"\n  select region [0,{SELECT_END}), windows={len(windows)}")
    print(f"  eta:       {GRID_ETA}")
    print(f"  sigma_sys: {GRID_SIGMA_SYS}")
    print(f"  prior_std: {GRID_PRIOR_STD}")
    print(f"  beta:      {GRID_BETA}")
    print(f"  workers={MAX_WORKERS}")

    specs = []
    for n_p in N_PARTICLES_LIST:
        for eta, ss, ps in itertools.product(
                GRID_ETA, GRID_SIGMA_SYS, GRID_PRIOR_STD):
            specs.append(("pf", n_p, eta, ss, ps, None))
            specs.append(("wspf_b", n_p, eta, ss, ps, None))
            for beta in GRID_BETA:
                specs.append(("wspf_a", n_p, eta, ss, ps, beta))
    print(f"  candidates: {len(specs)}")

    init_args = (X, y, loader.n_classes, windows)
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                             initializer=_init_worker,
                             initargs=init_args) as ex:
        for i, r in enumerate(ex.map(_evaluate_candidate, specs,
                                     chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(specs)}] done "
                      f"({time.time() - t0:.0f}s)")

    # SGD 独自探索 (η × σ0)
    _init_worker(*init_args)
    sgd_results = sorted(
        (_evaluate_sgd(eta, ps)
         for eta, ps in itertools.product(GRID_ETA, GRID_PRIOR_STD)),
        key=lambda r: -r["f1"])
    best_sgd = {"eta": sgd_results[0]["eta"],
                "prior_std": sgd_results[0]["prior_std"]}
    print(f"\n  Best SGD: {best_sgd}  (F1={sgd_results[0]['f1']:.4f})")
    _warn_if_boundary("SGD", best_sgd, GRID_ETA)

    by_n = {}
    for n_p in N_PARTICLES_LIST:
        sub = [r for r in results if r["n_particles"] == n_p]
        entry = {}
        for meth, key in [("pf", "best_pf"), ("wspf_b", "best_wspf_b"),
                          ("wspf_a", "best_wspf_a")]:
            cands = sorted([r for r in sub if r["method"] == meth],
                           key=lambda r: -r["f1"])
            best = {"eta": cands[0]["eta"],
                    "sigma_sys": cands[0]["sigma_sys"],
                    "prior_std": cands[0]["prior_std"]}
            if meth == "wspf_a":
                best["beta"] = cands[0]["beta"]
            entry[key] = best
            entry[f"all_{meth}"] = cands
            print(f"  Best {key}: {best}  (F1={cands[0]['f1']:.4f})")
            _warn_if_boundary(key, best, GRID_ETA)
        entry["best_sgd"] = best_sgd
        entry["all_sgd"] = sgd_results
        by_n[str(n_p)] = entry

    out = {
        "grid": {"eta": GRID_ETA, "sigma_sys": GRID_SIGMA_SYS,
                 "prior_std": GRID_PRIOR_STD, "beta": GRID_BETA,
                 "n_particles": N_PARTICLES_LIST,
                 "select_end": SELECT_END, "scale_fit_end": SCALE_FIT_END,
                 "hidden_dim": HIDDEN_DIM, "batch_size": BATCH_SIZE,
                 "test_size": TEST_SIZE, "seed": SEED,
                 "select_metric": "macro_f1"},
        "by_n_particles": by_n,
    }
    with open(GRID_JSON, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n  saved: {GRID_JSON}")
    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
