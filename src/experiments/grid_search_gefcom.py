#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEFCom2014 Solar グリッドサーチ

設計 (gefcom_solar_loader.py の定義に従う):
- 選択はゾーン 1 の選択区間 (ts < 2012-10-01) のみで行い、選択された
  ハイパーパラメータを全 3 ゾーンに適用する (HP の転移可能性も同時に
  検証できる)。
- σ_obs (観測ノイズ SD) はハイパラではなく、選択区間で SGD パイロットを
  走らせ後半の残差 RMSE として推定し JSON に保存する (リークフリー)。
- SGD は η×σ0 を独自探索し best_sgd として保存 (R1-7)。
- best の η が端点なら WARNING (R1-minor)。
- JSON スキーマは他実験と同一。

選択指標: 選択区間の test-window MSE (小さいほど良い)。
"""

import sys
import os
import json
import time
import itertools
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.data.gefcom_solar_loader import GefcomSolarLoader
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
# 設定
# ================================================================
N_PARTICLES_LIST = [100]

HIDDEN_DIM = 16          # d = (14+1)*16 + 17 = 257
BATCH_SIZE = 16
TEST_SIZE = 32
MAX_GRAD_NORM = 5.0
SEED = 42                # 選択用 (評価シード 0-9 と disjoint)

SELECT_ZONE = 1
SELECT_END_TS = "2012-10-01"

GRID_ETA = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
GRID_SIGMA_SYS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
GRID_PRIOR_STD = [0.1, 0.5]
GRID_BETA = [0.9, 0.95]

MAX_WORKERS = int(os.environ.get("NCPUS", os.cpu_count() or 1))

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "GEFCom2014", "Solar",
)
PREDICTORS_PATH = os.path.join(DATA_DIR, "predictors15.csv")
# predictors15.csv は POWER 列を内包する結合済み CSV（POWER は train15 と
# 共通行で完全一致し、かつ 1 ヶ月分長い）。冗長な train15 は使わず
# predictors 単独で読む（train_path=None → merged=pred ブランチ）。
TRAIN_PATH = None

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "gefcom",
)
GRID_JSON = os.path.join(OUTPUT_DIR, "grid_search_result.json")

_G = {}


def _init_worker(X, y, noise_std, windows):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _G["X"], _G["y"] = X, y
    _G["noise_std"] = noise_std
    _G["windows"] = windows


def _make_windows(n):
    wins, pos = [], 0
    while pos + BATCH_SIZE + TEST_SIZE <= n:
        wins.append(pos)
        pos += BATCH_SIZE
    return wins


def _clip_gradients(grad, max_norm):
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    return grad * scale


def estimate_noise_std(X, y, eta=0.1, prior_std=0.1, seed=SEED):
    """選択区間 SGD パイロットの後半残差 RMSE で σ_obs を推定"""
    model = NeuralNetRegression(X.shape[1], HIDDEN_DIM)
    grad_fn = create_regression_grad_fn(model, noise_std=0.1)
    rng = np.random.default_rng(seed + 10)
    theta = rng.normal(0.0, prior_std, size=model.param_dim)
    resid = []
    windows = _make_windows(len(y))
    half = len(windows) // 2
    for w_i, pos in enumerate(windows):
        Xtr = X[pos: pos + BATCH_SIZE]
        ytr = y[pos: pos + BATCH_SIZE]
        if w_i >= half:
            pred = model.predict(theta, Xtr).reshape(-1)
            resid.extend((ytr - pred).tolist())
        g = _clip_gradients(
            grad_fn(theta.reshape(1, -1), Xtr, ytr),
            MAX_GRAD_NORM).squeeze()
        theta = theta - eta * g
    s = float(np.sqrt(np.mean(np.square(resid))))
    return max(s, 1e-3)


def _run_stream_mse(filt_or_theta, X, y, windows, model, is_sgd,
                    grad_fn=None, loglik_fn=None, ps_grad_fn=None,
                    eta=None, use_ps=False):
    mses = []
    theta = filt_or_theta if is_sgd else None
    filt = None if is_sgd else filt_or_theta
    for pos in windows:
        Xtr = X[pos: pos + BATCH_SIZE]
        ytr = y[pos: pos + BATCH_SIZE]
        Xte = X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        yte = y[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        mu = theta if is_sgd else \
            (filt.weights[:, None] * filt.particles).sum(axis=0)
        pred = model.predict(mu, Xte).reshape(-1)
        mses.append(float(np.mean((yte - pred) ** 2)))
        if is_sgd:
            g = _clip_gradients(
                grad_fn(theta.reshape(1, -1), Xtr, ytr),
                MAX_GRAD_NORM).squeeze()
            theta = theta - eta * g
        elif use_ps:
            filt.step(Xtr, ytr, ps_grad_fn, loglik_fn)
        else:
            filt.step(Xtr, ytr, grad_fn, loglik_fn)
    return float(np.mean(mses))


def _evaluate_sgd(args):
    eta, prior_std = args
    X, y = _G["X"], _G["y"]
    model = NeuralNetRegression(X.shape[1], HIDDEN_DIM)
    grad_fn = create_regression_grad_fn(model, noise_std=_G["noise_std"])
    rng = np.random.default_rng(SEED + 10)
    theta = rng.normal(0.0, prior_std, size=model.param_dim)
    mse = _run_stream_mse(theta, X, y, _G["windows"], model, True,
                          grad_fn=grad_fn, eta=eta)
    return {"eta": eta, "prior_std": prior_std, "mse": mse}


def _evaluate_candidate(spec):
    method, n_p, eta, sigma_sys, prior_std, beta = spec
    X, y = _G["X"], _G["y"]
    ns = _G["noise_std"]
    model = NeuralNetRegression(X.shape[1], HIDDEN_DIM)
    grad_fn_raw = create_regression_grad_fn(model, noise_std=ns)
    loglik_fn = create_regression_loglik_fn(model, noise_std=ns)
    ps_grad_fn = create_regression_per_sample_grad_fn(model, noise_std=ns)

    def clipped_grad_fn(particles, Xb, yb, _raw=grad_fn_raw):
        return _clip_gradients(_raw(particles, Xb, yb), MAX_GRAD_NORM)

    common = dict(n_particles=n_p, param_dim=model.param_dim, eta=eta,
                  sigma_sys=sigma_sys, prior_mean=0.0, prior_std=prior_std,
                  ess_resample_ratio=0.5)
    if method == "pf":
        filt = ParticleFilter(**common, seed=SEED + 1)
        mse = _run_stream_mse(filt, X, y, _G["windows"], model, False,
                              grad_fn=clipped_grad_fn, loglik_fn=loglik_fn)
    elif method == "wspf_b":
        filt = WSPF_B(**common, grad_clip_norm=MAX_GRAD_NORM, seed=SEED + 3)
        mse = _run_stream_mse(filt, X, y, _G["windows"], model, False,
                              loglik_fn=loglik_fn, ps_grad_fn=ps_grad_fn,
                              use_ps=True)
    else:
        filt = WSPF_A(**common, grad_clip_norm=MAX_GRAD_NORM,
                      beta=beta, seed=SEED + 5)
        mse = _run_stream_mse(filt, X, y, _G["windows"], model, False,
                              loglik_fn=loglik_fn, ps_grad_fn=ps_grad_fn,
                              use_ps=True)

    return {"method": method, "n_particles": n_p, "eta": eta,
            "sigma_sys": sigma_sys, "prior_std": prior_std,
            "beta": beta, "mse": mse}


def _warn_if_boundary(name, best, grid_eta):
    if best["eta"] in (min(grid_eta), max(grid_eta)):
        print(f"  *** WARNING: {name} best eta={best['eta']} は "
              f"グリッド端点です。グリッド延長を検討してください ***")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    loader = GefcomSolarLoader(
        PREDICTORS_PATH, zone=SELECT_ZONE, train_path=TRAIN_PATH,
        select_end_ts=SELECT_END_TS)
    loader.print_summary()

    sel = loader.select_mask
    X, y = loader.X[sel], loader.y[sel]
    noise_std = estimate_noise_std(X, y)
    print(f"\n  σ_obs (SGD pilot RMSE, select region) = {noise_std:.4f}")

    windows = _make_windows(len(y))
    print(f"  select windows={len(windows)}, workers={MAX_WORKERS}")
    print(f"  eta: {GRID_ETA}\n  sigma_sys: {GRID_SIGMA_SYS}\n"
          f"  prior_std: {GRID_PRIOR_STD}\n  beta: {GRID_BETA}")

    specs = []
    for n_p in N_PARTICLES_LIST:
        for eta, ss, ps in itertools.product(
                GRID_ETA, GRID_SIGMA_SYS, GRID_PRIOR_STD):
            specs.append(("pf", n_p, eta, ss, ps, None))
            specs.append(("wspf_b", n_p, eta, ss, ps, None))
            for beta in GRID_BETA:
                specs.append(("wspf_a", n_p, eta, ss, ps, beta))
    print(f"  candidates: {len(specs)}")

    init_args = (X, y, noise_std, windows)
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

    _init_worker(*init_args)
    sgd_results = sorted(
        (_evaluate_sgd((eta, ps))
         for eta, ps in itertools.product(GRID_ETA, GRID_PRIOR_STD)),
        key=lambda r: r["mse"])
    best_sgd = {"eta": sgd_results[0]["eta"],
                "prior_std": sgd_results[0]["prior_std"]}
    print(f"\n  Best SGD: {best_sgd}  (MSE={sgd_results[0]['mse']:.5f})")
    _warn_if_boundary("SGD", best_sgd, GRID_ETA)

    by_n = {}
    for n_p in N_PARTICLES_LIST:
        sub = [r for r in results if r["n_particles"] == n_p]
        entry = {}
        for meth, key in [("pf", "best_pf"), ("wspf_b", "best_wspf_b"),
                          ("wspf_a", "best_wspf_a")]:
            cands = sorted([r for r in sub if r["method"] == meth],
                           key=lambda r: r["mse"])
            best = {"eta": cands[0]["eta"],
                    "sigma_sys": cands[0]["sigma_sys"],
                    "prior_std": cands[0]["prior_std"]}
            if meth == "wspf_a":
                best["beta"] = cands[0]["beta"]
            entry[key] = best
            entry[f"all_{meth}"] = cands
            print(f"  Best {key}: {best}  (MSE={cands[0]['mse']:.5f})")
            _warn_if_boundary(key, best, GRID_ETA)
        entry["best_sgd"] = best_sgd
        entry["all_sgd"] = sgd_results
        by_n[str(n_p)] = entry

    out = {
        "grid": {"eta": GRID_ETA, "sigma_sys": GRID_SIGMA_SYS,
                 "prior_std": GRID_PRIOR_STD, "beta": GRID_BETA,
                 "n_particles": N_PARTICLES_LIST,
                 "select_zone": SELECT_ZONE,
                 "select_end_ts": SELECT_END_TS,
                 "noise_std": noise_std,
                 "hidden_dim": HIDDEN_DIM, "batch_size": BATCH_SIZE,
                 "test_size": TEST_SIZE, "seed": SEED,
                 "select_metric": "mse"},
        "by_n_particles": by_n,
    }
    with open(GRID_JSON, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n  saved: {GRID_JSON}")
    print(f"  total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
