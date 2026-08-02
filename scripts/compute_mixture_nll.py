#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合予測密度の NLL (mixture predictive NLL) を計算する。

保存済み出力 (predictions.npz) は重み付き平均 μ̄ と予測 std のみを持ち、
既存の metric_nll は **モーメント整合した単一ガウス** N(y; μ̄, Var_w(f)+σ²) の
NLL である。本スクリプトが計算するのはユーザー定義の **真の混合** NLL:

    NLL = − (1/M) Σ_j log[ Σ_i w_{i} N(y_j ; μ_{i,j}, σ_obs²) ]

    μ_{i,j} = predict_fn(粒子_i, x_j),  σ_obs は観測ノイズ (全粒子共通),
    M = 対象区間の全 test サンプル数。

粒子別予測は保存されていないため、選択 HP でフィルタを **再実行** して
各 report/selection ステップの粒子群 (更新前, test-then-train) から計算する。
数値安定のため logsumexp を使う。degenerate 重み (WSPF-B) でも正しく評価できる。

使い方:
    # Solar (selected_params.json を使用, 全 zone)
    python scripts/compute_mixture_nll.py --benchmark gefcom
    # 一部 seed だけ素早く
    python scripts/compute_mixture_nll.py --benchmark gefcom --seeds 0 1 2
    # Price (selected 無し → 暫定 HP を明示指定)
    python scripts/compute_mixture_nll.py --benchmark gefcom_price \
        --eta 0.025 --sigma-cd 0.025 --prior-std 0.1 --provisional
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from _common import (load_config, resolve_seeds, build_benchmark,  # noqa: E402
                     load_selected, get_params, benchmark_contexts,
                     estimate_obs_noise)
from src.evaluation import mean_std, save_run_dir  # noqa: E402
from src.evaluation.runner import (_build_estimator,  # noqa: E402
                                    _predict_particles_weights)
from src.evaluation import SGD_METHODS  # noqa: E402

_LOG2PI = float(np.log(2.0 * np.pi))


def _mixture_logdensity(preds, logw, y, obs_sigma):
    """1 ブロックの各 test サンプルの log Σ_i w_i N(y; μ_i, σ²) を返す (B,)。

    preds : (N, B) 粒子別予測, logw : (N,) 正規化 log 重み, y : (B,)
    """
    var = float(obs_sigma) ** 2
    var = max(var, 1e-30)
    # logN_{i,j} = -0.5 log(2π σ²) - (y_j - μ_{i,j})² / (2σ²)
    resid2 = (y[None, :] - preds) ** 2                      # (N, B)
    logN = -0.5 * (_LOG2PI + np.log(var)) - resid2 / (2.0 * var)
    a = logw[:, None] + logN                                # (N, B)
    amax = np.max(a, axis=0)                                # (B,)
    return amax + np.log(np.sum(np.exp(a - amax[None, :]), axis=0))


def _run_mixture_nll(method, cfg, ctx, n, params, seed):
    """1 (method, ctx, seed) を再実行し、report/selection の混合 NLL を返す。

    runner と同一の test-then-train 順序・粒子/重み抽出を用いる。
    """
    bench = build_benchmark(cfg, **ctx)
    funcs = bench.build_functions(seed)
    grad_fn = funcs["grad_fn"]
    per_sample_grad_fn = funcs["per_sample_grad_fn"]
    loglik_fn = funcs["loglik_fn"]
    predict_fn = funcs["predict_fn"]
    obs_sigma = funcs.get("obs_sigma", getattr(bench, "noise_std", 0.1))

    est = _build_estimator(method, bench, n, params, seed, funcs)

    acc = {"report": [], "selection": []}
    for stp in bench.stream(seed):
        # -------- test (更新前 θ_{t-1}) --------
        Xte = stp.X_test
        yte = np.asarray(stp.y_test, dtype=np.float64).ravel()
        if Xte is not None and yte.size and not stp.straddles_switch:
            particles, weights = _predict_particles_weights(method, est)
            preds = np.asarray(predict_fn(particles, Xte), dtype=np.float64)
            if preds.ndim == 1:
                preds = preds[None, :]
            w = np.asarray(weights, dtype=np.float64)
            w = w / max(w.sum(), 1e-300)
            logw = np.log(np.maximum(w, 1e-300))
            ld = _mixture_logdensity(preds, logw, yte, obs_sigma)   # (B,)
            if stp.is_report_step:
                acc["report"].append(ld)
            if stp.is_selection_step:
                acc["selection"].append(ld)
        # -------- train (更新) --------
        Xtr, ytr = stp.X_train, stp.y_train
        if method in SGD_METHODS:
            est.train(Xtr, ytr)
        elif method == "PF":
            est.step(Xtr, ytr, grad_fn, loglik_fn)
        else:   # WSPF-A / WSPF-B
            est.step(Xtr, ytr, per_sample_grad_fn, loglik_fn)

    out = {}
    for region, chunks in acc.items():
        if chunks:
            ld = np.concatenate(chunks)
            out[region] = float(-np.mean(ld))   # 混合 NLL (小さいほど良い)
        else:
            out[region] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["PF", "WSPF-A", "WSPF-B"],
                    help="混合が意味を持つのは粒子系。点推定は 1 粒子に退化")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="評価 seed (既定: config seeds.evaluation)")
    ap.add_argument("--n", type=int, default=None)
    # --- 暫定 HP (selected_params が無い場合; 例: Price) ---
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--sigma-cd", type=float, default=None)
    ap.add_argument("--prior-std", type=float, default=None)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--noise-std", type=float, default=None,
                    help="σ_obs を明示指定 (既定: 選択区間から推定)")
    ap.add_argument("--provisional", action="store_true",
                    help="selected_params を使わず暫定 HP で計算する")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    n = args.n or cfg["n_particles"]["main"]
    seeds = args.seeds if args.seeds is not None \
        else resolve_seeds(cfg, "evaluation")

    provisional = args.provisional or (args.eta is not None)
    if provisional:
        # 暫定 HP: 単一コンテキスト。σ_obs は指定 or 選択区間推定。
        if args.eta is None or args.sigma_cd is None or args.prior_std is None:
            ap.error("--provisional 時は --eta --sigma-cd --prior-std が必須")
        sigma_obs = args.noise_std if args.noise_std is not None \
            else estimate_obs_noise(cfg)
        contexts = [{"noise_std": sigma_obs}]
        base = {"eta": args.eta, "sigma_sys": args.sigma_cd,
                "prior_std": args.prior_std, "beta": args.beta}
        get_p = lambda m: dict(base)   # noqa: E731
        print(f"[PROVISIONAL] σ_obs={sigma_obs:.4f}, HP={base}")
    else:
        selected = load_selected(cfg)
        contexts = benchmark_contexts(cfg, selected)
        get_p = lambda m: get_params(selected, m, n)   # noqa: E731

    print(f"=== mixture NLL: benchmark={cfg['benchmark']} N={n} "
          f"seeds={list(seeds)} ===")
    print(f"{'method':<8}{'zone':>5}  {'NLL_report(mean±std)':>24}  "
          f"{'NLL_select(mean±std)':>24}")

    rows = []
    agg = {}   # method -> list of report NLL across ctx×seed
    for ctx in contexts:
        zone = ctx.get("zone")
        for m in args.methods:
            params = get_p(m)
            rep_vals, sel_vals = [], []
            for sd in seeds:
                r = _run_mixture_nll(m, cfg, ctx, n, params, sd)
                rep_vals.append(r["report"])
                sel_vals.append(r["selection"])
            rmu, rsd = mean_std(rep_vals)
            smu, ssd = mean_std(sel_vals)
            rows.append({"method": m, "zone": zone,
                         "mixture_nll_report_mean": rmu,
                         "mixture_nll_report_std": rsd,
                         "mixture_nll_selection_mean": smu,
                         "mixture_nll_selection_std": ssd,
                         "n_seeds": len(seeds)})
            agg.setdefault(m, []).extend(rep_vals)
            ztag = zone if zone is not None else "-"
            print(f"{m:<8}{str(ztag):>5}  {rmu:>10.4f} ± {rsd:<9.4f}  "
                  f"{smu:>10.4f} ± {ssd:<9.4f}")

    if any(r["zone"] is not None for r in rows):
        print("\n--- zone 併合 (report, 全 zone×seed) ---")
        for m in args.methods:
            if agg.get(m):
                mu, sd = mean_std(agg[m])
                print(f"{m:<8} NLL_report = {mu:.4f} ± {sd:.4f} "
                      f"(n={len(agg[m])})")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "mixture_nll")
    sel_dump = {} if provisional else load_selected(cfg)
    save_run_dir(out_dir, config=cfg, selected_params=sel_dump,
                 metrics_rows=rows, diagnostics={})
    with open(os.path.join(out_dir, "mixture_nll.json"), "w",
              encoding="utf-8") as f:
        json.dump({"benchmark": cfg["benchmark"], "n_particles": n,
                   "seeds": list(seeds), "provisional": provisional,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_dir}")


if __name__ == "__main__":
    main()
