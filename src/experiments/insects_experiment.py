#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSECTS (abrupt balanced) 多クラス分類実験

email_binary_experiment.py の鏡像。R2-1 (より大規模な非定常ベンチマーク)
への対応: n≈53k, d≈1286 (email の 1500 / 833 に対して 35× / 1.5×)。

構成:
- test-then-train (train B=16, test window 32)
- リークフリー: 標準化 fit と HP 選択は [0, REPORT_START)、報告は
  test window が REPORT_START 以降に始まるもののみ
- 比較: NoChange / SGD / PF / WSPF-A / WSPF-B
- 診断 npz (ESS, ρ, 計時, 較正用確率) を保存し multiseed / plots から再利用

事前に grid_search_insects.py を実行しておくこと (strict loader)。
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
# 設定 (grid_search_insects.py と一致させること)
# ================================================================
N_PARTICLES_LIST = [100]

HIDDEN_DIM = 32
BATCH_SIZE = 16
TEST_SIZE = 32
MAX_GRAD_NORM = 5.0
SEED = 42

SELECT_END = 19500
SCALE_FIT_END = SELECT_END
REPORT_START = SELECT_END

# post-switch 解析: 報告区間内のスイッチのみ使用
POST_SWITCH_STEPS = 10

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "INSECTS",
    "INSECTS-abrupt_balanced_norm.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "outputs", "insects",
)
GRID_JSON = os.path.join(OUTPUT_DIR, "grid_search_result.json")

METHODS = ["NoChange", "SGD", "PF", "WSPF-A", "WSPF-B"]
FILTER_METHODS = ["SGD", "PF", "WSPF-A", "WSPF-B"]


# ================================================================
# 共通ユーティリティ
# ================================================================
def load_grid_search_params():
    """strict: グリッド未実行・キー欠損は明示的に失敗させる"""
    if not os.path.exists(GRID_JSON):
        raise RuntimeError(
            f"グリッド結果が見つかりません({GRID_JSON})。先に "
            "grid_search_insects.py を実行してください。")
    with open(GRID_JSON) as fp:
        data = json.load(fp)
    by_n = data.get("by_n_particles")
    if by_n is None:
        raise KeyError("グリッド JSON に 'by_n_particles' がありません。")
    hp_by_n = {}
    for n_p in N_PARTICLES_LIST:
        key = str(n_p)
        if key not in by_n:
            raise KeyError(f"グリッド JSON に N={n_p} がありません。")
        entry = by_n[key]
        for k in ("best_sgd", "best_pf", "best_wspf_b", "best_wspf_a"):
            if k not in entry:
                raise KeyError(f"グリッド JSON に '{k}' がありません。"
                               "グリッドを再生成してください。")
        if "beta" not in entry["best_wspf_a"]:
            raise KeyError("best_wspf_a に 'beta' がありません。")
        hp_by_n[n_p] = (entry["best_pf"], entry["best_wspf_b"],
                        entry["best_wspf_a"], entry["best_sgd"])
    return hp_by_n


def clip_gradients(grad, max_norm):
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    return grad * scale


def macro_f1(pred, y, n_classes):
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


def mean_loglik(model, theta, X, y):
    """テスト window のサンプル平均対数尤度"""
    ll_sum = model.loglik_batch(theta.reshape(1, -1), X, y)[0]
    return float(ll_sum / len(y))


NO_CHANGE_EPS = 1e-3


def nochange_metrics(last_label, y_test, n_classes):
    pred = np.full(len(y_test), last_label, dtype=np.int64)
    acc = float(np.mean(pred == y_test))
    f1 = macro_f1(pred, y_test, n_classes)
    # ハード予測の平滑化尤度 (email の NO_CHANGE_EPS と同方針)
    p_true = np.where(pred == y_test, 1.0 - NO_CHANGE_EPS,
                      NO_CHANGE_EPS / (n_classes - 1))
    ll = float(np.mean(np.log(p_true)))
    return acc, f1, ll


# ================================================================
# 単一シードの実験実行 (multiseed から再利用)
# ================================================================
def run_experiment(n_particles, model, loader,
                   best_pf, best_wspf_b, best_wspf_a, best_sgd,
                   seed=None, collect_diagnostics=True, verbose=True):
    seed_base = SEED if seed is None else seed
    n_classes = loader.n_classes

    grad_fn_raw = create_mc_grad_fn(model)
    loglik_fn = create_mc_loglik_fn(model)
    ps_grad_fn = create_mc_per_sample_grad_fn(model)

    def grad_fn(particles, Xb, yb, _raw=grad_fn_raw):
        return clip_gradients(_raw(particles, Xb, yb), MAX_GRAD_NORM)

    rng_sgd = np.random.default_rng(seed_base + 10)
    theta_sgd = rng_sgd.normal(
        0.0, best_sgd["prior_std"], size=model.param_dim)
    sgd_eta = best_sgd["eta"]

    pf = ParticleFilter(
        n_particles=n_particles, param_dim=model.param_dim,
        eta=best_pf["eta"], sigma_sys=best_pf["sigma_sys"],
        prior_mean=0.0, prior_std=best_pf["prior_std"],
        ess_resample_ratio=0.5, seed=seed_base + 1)
    wspf_b = WSPF_B(
        n_particles=n_particles, param_dim=model.param_dim,
        eta=best_wspf_b["eta"], sigma_sys=best_wspf_b["sigma_sys"],
        prior_mean=0.0, prior_std=best_wspf_b["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM,
        seed=seed_base + 3)
    wspf_a = WSPF_A(
        n_particles=n_particles, param_dim=model.param_dim,
        eta=best_wspf_a["eta"], sigma_sys=best_wspf_a["sigma_sys"],
        prior_mean=0.0, prior_std=best_wspf_a["prior_std"],
        ess_resample_ratio=0.5, grad_clip_norm=MAX_GRAD_NORM,
        beta=best_wspf_a["beta"], seed=seed_base + 5)

    acc = {m: [] for m in METHODS}
    f1 = {m: [] for m in METHODS}
    ll = {m: [] for m in METHODS}
    sample_positions = []
    prob_hist = {m: [] for m in FILTER_METHODS}
    label_hist = []

    n = loader.n_samples
    total_steps = (n - TEST_SIZE) // BATCH_SIZE
    last_label = 0
    pos = 0
    step = 0
    t0 = time.time()

    while pos + BATCH_SIZE + TEST_SIZE <= n:
        X_train = loader.X[pos: pos + BATCH_SIZE]
        y_train = loader.y[pos: pos + BATCH_SIZE]
        X_test = loader.X[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]
        y_test = loader.y[pos + BATCH_SIZE: pos + BATCH_SIZE + TEST_SIZE]

        sample_positions.append(pos)
        pos += BATCH_SIZE

        # --- 評価 (test-then-train) ---
        a, f, l = nochange_metrics(last_label, y_test, n_classes)
        acc["NoChange"].append(a)
        f1["NoChange"].append(f)
        ll["NoChange"].append(l)

        mus = {
            "SGD": theta_sgd,
            "PF": (pf.weights[:, None] * pf.particles).sum(axis=0),
            "WSPF-A": (wspf_a.weights[:, None] * wspf_a.particles).sum(axis=0),
            "WSPF-B": (wspf_b.weights[:, None] * wspf_b.particles).sum(axis=0),
        }
        for m in FILTER_METHODS:
            pred = model.predict(mus[m], X_test)[0]
            acc[m].append(float(np.mean(pred == y_test)))
            f1[m].append(macro_f1(pred, y_test, n_classes))
            ll[m].append(mean_loglik(model, mus[m], X_test, y_test))

        # 較正用: 報告区間の予測確率 (真クラス確率ベクトル) とラベル
        if pos >= REPORT_START:
            for m in FILTER_METHODS:
                out_m, _, _ = model.forward(mus[m].reshape(1, -1), X_test)
                prob_hist[m].append(
                    np.clip(out_m[0], 1e-10, 1.0))  # (32, C)
            label_hist.append(y_test.copy())

        # --- 学習 ---
        g = grad_fn(theta_sgd.reshape(1, -1), X_train, y_train).squeeze()
        theta_sgd = theta_sgd - sgd_eta * g

        pf.step(X_train, y_train, grad_fn, loglik_fn)
        wspf_b.step(X_train, y_train, ps_grad_fn, loglik_fn)
        wspf_a.step(X_train, y_train, ps_grad_fn, loglik_fn)

        last_label = int(y_train[-1])
        step += 1
        if verbose and step % 200 == 0:
            print(f"  Step {step:5d}/{total_steps} "
                  f"({time.time() - t0:6.1f}s)  "
                  f"Acc: SGD={acc['SGD'][-1]:.3f} PF={acc['PF'][-1]:.3f} "
                  f"A={acc['WSPF-A'][-1]:.3f} B={acc['WSPF-B'][-1]:.3f}")

    if verbose:
        print(f"  Completed {step} steps in {time.time() - t0:.1f}s")

    for m in METHODS:
        acc[m] = np.asarray(acc[m])
        f1[m] = np.asarray(f1[m])
        ll[m] = np.asarray(ll[m])
    sample_positions = np.asarray(sample_positions)

    # 報告マスク: test window が REPORT_START 以降に始まるもの
    test_starts = sample_positions + BATCH_SIZE
    eval_mask = test_starts >= REPORT_START

    result_rows = []
    for m in METHODS:
        result_rows.append({
            "method": m,
            "accuracy": float(acc[m][eval_mask].mean()),
            "macro_f1": float(f1[m][eval_mask].mean()),
            "loglik": float(ll[m][eval_mask].mean()),
        })

    diagnostics = None
    if collect_diagnostics:
        diagnostics = {
            "PF": pf.get_history(),
            "WSPF-A": wspf_a.get_history(),
            "WSPF-B": wspf_b.get_history(),
            "_calib": {
                "probs": {m: (np.concatenate(prob_hist[m], axis=0)
                              if prob_hist[m] else np.array([]))
                          for m in FILTER_METHODS},
                "labels": (np.concatenate(label_hist)
                           if label_hist else np.array([])),
            },
        }

    return (result_rows, acc, f1, ll, sample_positions, eval_mask,
            diagnostics)


# ================================================================
# phase split (報告区間内スイッチのみ)
# ================================================================
def phase_split(acc, f1, ll, sample_positions, eval_mask, change_points):
    """報告区間内スイッチの post-switch (最初 POST_SWITCH_STEPS) vs stable"""
    report_switches = [cp for cp in change_points if cp >= REPORT_START]
    step_idx = np.arange(len(sample_positions))
    post_mask = np.zeros(len(sample_positions), dtype=bool)
    for cp in report_switches:
        # スイッチ後最初の train window の step index
        s0 = int(np.searchsorted(sample_positions, cp))
        post_mask[s0: s0 + POST_SWITCH_STEPS] = True
    post_mask &= eval_mask
    stable_mask = eval_mask & ~post_mask

    rows = []
    for m in METHODS:
        rows.append({
            "method": m,
            "post_acc": float(acc[m][post_mask].mean()),
            "post_f1": float(f1[m][post_mask].mean()),
            "post_ll": float(ll[m][post_mask].mean()),
            "stable_acc": float(acc[m][stable_mask].mean()),
            "stable_f1": float(f1[m][stable_mask].mean()),
            "stable_ll": float(ll[m][stable_mask].mean()),
        })
    return rows, report_switches, post_mask, stable_mask


# ================================================================
# main
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hp_by_n = load_grid_search_params()

    loader = InsectsDataLoader(
        DATA_PATH, scale_fit_end=SCALE_FIT_END, seed=SEED)
    loader.print_regime_class_distribution()
    model = MulticlassNeuralNetModel(
        loader.n_features, HIDDEN_DIM, loader.n_classes)
    print(f"\n  model: input={loader.n_features}, hidden={HIDDEN_DIM}, "
          f"classes={loader.n_classes}, d={model.param_dim}")

    summary_lines = []
    for n_p in N_PARTICLES_LIST:
        best_pf, best_wspf_b, best_wspf_a, best_sgd = hp_by_n[n_p]
        print(f"\nN={n_p}")
        print(f"  SGD:    {best_sgd}")
        print(f"  PF:     {best_pf}")
        print(f"  WSPF-B: {best_wspf_b}")
        print(f"  WSPF-A: {best_wspf_a}")

        (rows, acc, f1, ll, sample_positions, eval_mask,
         diagnostics) = run_experiment(
            n_p, model, loader, best_pf, best_wspf_b, best_wspf_a,
            best_sgd, seed=SEED)

        # --- サマリ ---
        summary_lines.append(f"N = {n_p}")
        summary_lines.append(
            f"  SGD: {best_sgd}  PF: {best_pf}")
        summary_lines.append(
            f"  WSPF-B: {best_wspf_b}  WSPF-A: {best_wspf_a}")
        summary_lines.append(
            f"  {'Method':<10s} {'Accuracy':>10s} {'macro-F1':>10s} "
            f"{'LogLik':>10s}")
        for r in rows:
            summary_lines.append(
                f"  {r['method']:<10s} {r['accuracy']:>10.4f} "
                f"{r['macro_f1']:>10.4f} {r['loglik']:>10.4f}")

        # --- phase split ---
        ps_rows, report_switches, post_mask, stable_mask = phase_split(
            acc, f1, ll, sample_positions, eval_mask, loader.change_points)
        summary_lines.append(
            f"\n  Phase split (report-region switches: {report_switches}, "
            f"post={POST_SWITCH_STEPS} steps)")
        summary_lines.append(
            f"  {'Method':<10s} {'post_acc':>9s} {'post_f1':>9s} "
            f"{'post_ll':>9s} {'stab_acc':>9s} {'stab_f1':>9s} "
            f"{'stab_ll':>9s}")
        for r in ps_rows:
            summary_lines.append(
                f"  {r['method']:<10s} {r['post_acc']:>9.4f} "
                f"{r['post_f1']:>9.4f} {r['post_ll']:>9.4f} "
                f"{r['stable_acc']:>9.4f} {r['stable_f1']:>9.4f} "
                f"{r['stable_ll']:>9.4f}")

        # --- npz 保存 (multiseed / plots / calibration 再利用) ---
        npz_path = os.path.join(
            OUTPUT_DIR, f"results_N{n_p}_seed{SEED}.npz")
        save_dict = {
            "sample_positions": sample_positions,
            "eval_mask": eval_mask,
            "post_mask": post_mask,
            "stable_mask": stable_mask,
            "change_points": np.asarray(loader.change_points),
            "report_start": REPORT_START,
            "calib_labels": diagnostics["_calib"]["labels"],
        }
        for m in METHODS:
            key = m.replace("-", "_")
            save_dict[f"acc_{key}"] = acc[m]
            save_dict[f"f1_{key}"] = f1[m]
            save_dict[f"ll_{key}"] = ll[m]
        for m in FILTER_METHODS:
            key = m.replace("-", "_")
            save_dict[f"calib_probs_{key}"] = \
                diagnostics["_calib"]["probs"][m]
        for m in ["PF", "WSPF-A", "WSPF-B"]:
            h = diagnostics[m]
            key = m.replace("-", "_")
            for hk in ("ess", "resampled", "t_step"):
                if hk in h:
                    save_dict[f"diag_{key}_{hk}"] = np.asarray(h[hk])
            for hk in ("rho", "logcorr_nonfinite_count",
                       "cond_M_mean", "cond_M_max"):
                if hk in h:
                    save_dict[f"diag_{key}_{hk}"] = np.asarray(h[hk])
        np.savez_compressed(npz_path, **save_dict)
        print(f"  saved: {npz_path}")

        # --- 時系列プロット ---
        w = 25
        kernel = np.ones(w) / w
        for name, series in [("accuracy", acc), ("macro_f1", f1),
                             ("loglik", ll)]:
            plt.figure(figsize=(11, 4))
            for m in METHODS:
                sm = np.convolve(series[m], kernel, mode="valid")
                xs = sample_positions[w - 1:]
                plt.plot(xs, sm, label=m, lw=1.2)
            for cp in loader.change_points:
                plt.axvline(cp, color="red", ls=":", lw=0.8)
            plt.axvline(REPORT_START, color="gray", ls="--", lw=0.8)
            plt.xlabel("sample position")
            plt.ylabel(name)
            plt.title(f"INSECTS abrupt balanced (N={n_p}, "
                      f"{w}-step moving avg)")
            plt.legend(fontsize=8)
            plt.tight_layout()
            fig_path = os.path.join(
                OUTPUT_DIR, f"insects_{name}_timeseries_N{n_p}.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()

    txt_path = os.path.join(OUTPUT_DIR, "insects_results.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("INSECTS abrupt balanced classification results\n")
        fp.write(f"(leak-free: scale fit + HP select [0,{SELECT_END}), "
                 f"report [{REPORT_START},end))\n")
        fp.write("=" * 70 + "\n\n")
        fp.write("\n".join(summary_lines) + "\n")
    print(f"\n  saved: {txt_path}")


if __name__ == "__main__":
    main()
