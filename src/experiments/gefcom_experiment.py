#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEFCom2014 Solar 実データ回帰実験

役割:
- R2-1: 実データ回帰ベンチマーク (現状の実データは分類のみ)
- A-vs-B: 自然な異方的勾配構造での差別化 (per-feature 標準化のみ,
  whitening/PCA なし)
- ドリフト性格: 漸進的季節ドリフト (switch-aligned 解析は行わず,
  月別 MSE 時系列で報告)

構成は insects_experiment.py の鏡像 (回帰版):
- ゾーン 1 で選択した HP を全 3 ゾーンに適用
- test-then-train (B=16, test 32), 主指標 MSE, 副 MAE
- 報告区間 = ts >= 2012-10-01 の test window
- 診断 npz (ESS, ρ, 計時, アンサンブル予測) をゾーンごとに保存

事前に grid_search_gefcom.py を実行しておくこと (strict loader)。
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
# 設定 (grid_search_gefcom.py と一致させること)
# ================================================================
N_PARTICLES = 100
HIDDEN_DIM = 16
BATCH_SIZE = 16
TEST_SIZE = 32
MAX_GRAD_NORM = 5.0
SEED = 42

ZONES = [1, 2, 3]
SELECT_END_TS = "2012-10-01"

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

METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]


def load_grid_search_params():
    """strict: グリッド未実行・キー欠損は明示的に失敗させる"""
    if not os.path.exists(GRID_JSON):
        raise RuntimeError(
            f"グリッド結果が見つかりません({GRID_JSON})。先に "
            "grid_search_gefcom.py を実行してください。")
    with open(GRID_JSON) as fp:
        data = json.load(fp)
    by_n = data.get("by_n_particles")
    if by_n is None or str(N_PARTICLES) not in by_n:
        raise KeyError(f"グリッド JSON に N={N_PARTICLES} がありません。")
    entry = by_n[str(N_PARTICLES)]
    for k in ("best_sgd", "best_pf", "best_wspf_b", "best_wspf_a"):
        if k not in entry:
            raise KeyError(f"グリッド JSON に '{k}' がありません。")
    if "beta" not in entry["best_wspf_a"]:
        raise KeyError("best_wspf_a に 'beta' がありません。")
    noise_std = data.get("grid", {}).get("noise_std")
    if noise_std is None:
        raise KeyError("グリッド JSON に 'noise_std' がありません。")
    return (entry["best_pf"], entry["best_wspf_b"],
            entry["best_wspf_a"], entry["best_sgd"], float(noise_std))


def clip_gradients(grad, max_norm):
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    return grad * scale


# ================================================================
# 単一ゾーン・単一シードの実験 (multiseed から再利用)
# ================================================================
def run_experiment(loader, best_pf, best_wspf_b, best_wspf_a, best_sgd,
                   noise_std, seed=None, collect_diagnostics=True,
                   verbose=True):
    seed_base = SEED if seed is None else seed
    model = NeuralNetRegression(loader.n_features, HIDDEN_DIM)

    grad_fn_raw = create_regression_grad_fn(model, noise_std=noise_std)
    loglik_fn = create_regression_loglik_fn(model, noise_std=noise_std)
    ps_grad_fn = create_regression_per_sample_grad_fn(
        model, noise_std=noise_std)

    def grad_fn(particles, Xb, yb, _raw=grad_fn_raw):
        return clip_gradients(_raw(particles, Xb, yb), MAX_GRAD_NORM)

    rng_sgd = np.random.default_rng(seed_base + 10)
    theta_sgd = rng_sgd.normal(
        0.0, best_sgd["prior_std"], size=model.param_dim)
    sgd_eta = best_sgd["eta"]

    pf = ParticleFilter(
        n_particles=N_PARTICLES, param_dim=model.param_dim,
        eta=best_pf["eta"], sigma_sys=best_pf["sigma_sys"],
        prior_mean=0.0, prior_std=best_pf["prior_std"],
        ess_resample_ratio=0.5, seed=seed_base + 1)
    wspf_b = WSPF_B(
        n_particles=N_PARTICLES, param_dim=model.param_dim,
        eta=best_wspf_b["eta"], sigma_sys=best_wspf_b["sigma_sys"],
        prior_mean=0.0, prior_std=best_wspf_b["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM,
        seed=seed_base + 3)
    wspf_a = WSPF_A(
        n_particles=N_PARTICLES, param_dim=model.param_dim,
        eta=best_wspf_a["eta"], sigma_sys=best_wspf_a["sigma_sys"],
        prior_mean=0.0, prior_std=best_wspf_a["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM,
        beta=best_wspf_a["beta"], seed=seed_base + 5)

    mse = {m: [] for m in METHODS}
    mae = {m: [] for m in METHODS}
    # アンサンブル予測の分散 (較正解析用, 報告区間のみ)
    pred_records = {m: [] for m in ["PF", "WSPF-A", "WSPF-B"]}
    y_records = []
    window_ts = []          # test window 先頭時刻
    sample_positions = []

    select_end = None
    n = loader.n_samples
    pos = 0
    step = 0
    t0 = time.time()

    while pos + BATCH_SIZE + TEST_SIZE <= n:
        X_train = loader.X[pos: pos + BATCH_SIZE]
        y_train = loader.y[pos: pos + BATCH_SIZE]
        X_test = loader.X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        y_test = loader.y[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        ts0 = loader.timestamps[pos + BATCH_SIZE]

        sample_positions.append(pos)
        window_ts.append(ts0)
        pos += BATCH_SIZE

        mus = {
            "SGD": theta_sgd,
            "PF": (pf.weights[:, None] * pf.particles).sum(axis=0),
            "WSPF-A": (wspf_a.weights[:, None] * wspf_a.particles).sum(axis=0),
            "WSPF-B": (wspf_b.weights[:, None] * wspf_b.particles).sum(axis=0),
        }
        for m in METHODS:
            pred = model.predict(mus[m], X_test).reshape(-1)
            mse[m].append(float(np.mean((y_test - pred) ** 2)))
            mae[m].append(float(np.mean(np.abs(y_test - pred))))

        in_report = ts0 >= loader.select_end_ts
        if in_report:
            for m, filt in [("PF", pf), ("WSPF-A", wspf_a),
                            ("WSPF-B", wspf_b)]:
                # アンサンブル関数値の平均と分散 (較正解析用)
                fx = model.predict(filt.particles, X_test)  # (N, 32)
                w = filt.weights[:, None]
                fmean = (w * fx).sum(axis=0)
                fvar = (w * (fx - fmean) ** 2).sum(axis=0)
                pred_records[m].append(
                    np.column_stack([fmean, fvar]))
            y_records.append(y_test.copy())

        g = grad_fn(theta_sgd.reshape(1, -1), X_train, y_train).squeeze()
        theta_sgd = theta_sgd - sgd_eta * g
        pf.step(X_train, y_train, grad_fn, loglik_fn)
        wspf_b.step(X_train, y_train, ps_grad_fn, loglik_fn)
        wspf_a.step(X_train, y_train, ps_grad_fn, loglik_fn)

        step += 1
        if verbose and step % 100 == 0:
            print(f"  Step {step:4d} ({time.time() - t0:5.1f}s)  "
                  f"MSE: SGD={mse['SGD'][-1]:.4f} PF={mse['PF'][-1]:.4f} "
                  f"A={mse['WSPF-A'][-1]:.4f} B={mse['WSPF-B'][-1]:.4f}")

    if verbose:
        print(f"  Completed {step} steps in {time.time() - t0:.1f}s")

    for m in METHODS:
        mse[m] = np.asarray(mse[m])
        mae[m] = np.asarray(mae[m])
    window_ts = np.asarray(window_ts)
    eval_mask = np.asarray(
        [t >= loader.select_end_ts for t in window_ts])

    result_rows = []
    for m in METHODS:
        result_rows.append({
            "method": m,
            "mse": float(mse[m][eval_mask].mean()),
            "mae": float(mae[m][eval_mask].mean()),
        })

    diagnostics = None
    if collect_diagnostics:
        diagnostics = {
            "PF": pf.get_history(),
            "WSPF-A": wspf_a.get_history(),
            "WSPF-B": wspf_b.get_history(),
            "_pred": {m: (np.concatenate(pred_records[m], axis=0)
                          if pred_records[m] else np.array([]))
                      for m in pred_records},
            "_y": (np.concatenate(y_records)
                   if y_records else np.array([])),
        }

    return result_rows, mse, mae, window_ts, eval_mask, diagnostics


def monthly_table(mse, window_ts, eval_mask):
    """報告区間の月別 MSE (漸進ドリフトの時系列分解)"""
    months = np.asarray([t.strftime("%Y-%m") for t in window_ts])
    uniq = sorted(set(months[eval_mask]))
    rows = []
    for mo in uniq:
        mask = eval_mask & (months == mo)
        rows.append([mo] + [float(mse[m][mask].mean()) for m in METHODS])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_pf, best_wspf_b, best_wspf_a, best_sgd, noise_std = \
        load_grid_search_params()
    print(f"HP (selected on zone 1): SGD={best_sgd}  PF={best_pf}")
    print(f"  WSPF-B={best_wspf_b}\n  WSPF-A={best_wspf_a}")
    print(f"  noise_std={noise_std:.4f}")

    lines = []
    for zone in ZONES:
        print(f"\n=== Zone {zone} ===")
        loader = GefcomSolarLoader(
            PREDICTORS_PATH, zone=zone, train_path=TRAIN_PATH,
            select_end_ts=SELECT_END_TS)
        loader.print_summary()

        (rows, mse, mae, window_ts, eval_mask,
         diagnostics) = run_experiment(
            loader, best_pf, best_wspf_b, best_wspf_a, best_sgd,
            noise_std, seed=SEED)

        lines.append(f"Zone {zone} (report windows: "
                     f"{int(eval_mask.sum())})")
        lines.append(f"  {'Method':<10s} {'MSE':>10s} {'MAE':>10s}")
        for r in rows:
            lines.append(f"  {r['method']:<10s} {r['mse']:>10.5f} "
                         f"{r['mae']:>10.5f}")

        lines.append(f"\n  Monthly MSE (report region):")
        lines.append("  month    " + "".join(
            f"{m:>10s}" for m in METHODS))
        for row in monthly_table(mse, window_ts, eval_mask):
            lines.append(f"  {row[0]:<9s}" + "".join(
                f"{v:>10.5f}" for v in row[1:]))
        lines.append("")

        # npz 保存
        npz_path = os.path.join(
            OUTPUT_DIR, f"results_zone{zone}_N{N_PARTICLES}_seed{SEED}.npz")
        save = {
            "window_ts": np.asarray(
                [t.isoformat() for t in window_ts]),
            "eval_mask": eval_mask,
            "y_report": diagnostics["_y"],
        }
        for m in METHODS:
            key = m.replace("-", "_")
            save[f"mse_{key}"] = mse[m]
            save[f"mae_{key}"] = mae[m]
        for m in ["PF", "WSPF-A", "WSPF-B"]:
            key = m.replace("-", "_")
            save[f"pred_{key}"] = diagnostics["_pred"][m]
            h = diagnostics[m]
            for hk in ("ess", "resampled", "t_step", "rho",
                       "logcorr_nonfinite_count",
                       "cond_M_mean", "cond_M_max"):
                if hk in h:
                    save[f"diag_{key}_{hk}"] = np.asarray(h[hk])
        np.savez_compressed(npz_path, **save)
        print(f"  saved: {npz_path}")

        # 月別 MSE プロット
        plt.figure(figsize=(11, 4))
        tbl = monthly_table(mse, window_ts, eval_mask)
        xs = np.arange(len(tbl))
        for j, m in enumerate(METHODS):
            plt.plot(xs, [row[1 + j] for row in tbl], marker="o",
                     ms=3, lw=1.2, label=m)
        plt.xticks(xs, [row[0] for row in tbl], rotation=60, fontsize=7)
        plt.ylabel("monthly test MSE")
        plt.title(f"GEFCom2014-S zone {zone} (N={N_PARTICLES})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(
            OUTPUT_DIR, f"gefcom_monthly_mse_zone{zone}.png"), dpi=150)
        plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "gefcom_results.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("GEFCom2014 Solar point-forecast regression results\n")
        fp.write(f"(HP selected on zone {1}, select region "
                 f"ts < {SELECT_END_TS}; report = rest)\n")
        fp.write("=" * 70 + "\n\n")
        fp.write("\n".join(lines) + "\n")
    print(f"\n  saved: {txt_path}")


if __name__ == "__main__":
    main()
