#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事前スクリーニング (Go/No-Go, 設計書 §5)

グリッドサーチ **前** に、あるベンチマークが WSPF の適用条件を満たすかを
選択区間のみで機械的に判定する。`run_qcd_sweep.py` と異なり
`selected_params.json` を前提とせず、固定の少数 HP 格子を内蔵する。
全ベンチマーク共通の `--benchmark` を取り、既存データセットの遡及スクリーニング
にも使える。

判定基準 (すべて選択区間, seed 複数の平均):
  S1 (条件1): PF の CRPS (副: MSE) が σ_cd に対して内部最適を持つ
              (最小の σ_cd がグリッド両端点でない)。
  S2 (条件2): S1 の最適 σ_cd における WSPF-B 1 点実行で ρ_q50 が概ね
              [0.1, 0.9]、かつ ESS/N > 0.1。
  S3 (尤度判別力): PF の粒子間 loglik std の中央値 > 1 nat
              (INSECTS 0.37 / Solar 1.14 が参照点)。
判定: S1 ∧ S2 ∧ S3 → 本採用 (Phase 2 へ)。不成立 → 境界事例として記録。

使い方:
    python scripts/screen_qcd.py --benchmark gefcom_price
    python scripts/screen_qcd.py --benchmark gefcom_price \
        --sigma-cd 0.001 0.005 0.01 0.025 0.05 0.1 --eta 0.01 0.05 0.2
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from _common import (load_config, resolve_seeds, build_benchmark,  # noqa: E402
                     region_mask, estimate_obs_noise)
from src.evaluation import (run_method, summarize_history,  # noqa: E402
                            rho_report, save_run_dir)

# S1/S3 は PF 中心。S2 のみ WSPF-B を 1 点実行する。
S2_RHO_LOW, S2_RHO_HIGH = 0.1, 0.9
S2_ESS_MIN = 0.1
S3_LOGLIK_STD_MIN = 1.0


def _coarse_etas(cfg, override):
    """粗い η 格子 (3 点)。override 優先。既定は config grid.eta の log 間隔 3 点。"""
    if override:
        return list(override)
    etas = list(cfg["grid"]["eta"])
    if len(etas) <= 3:
        return etas
    idx = [len(etas) // 4, len(etas) // 2, (3 * len(etas)) // 4]
    return [etas[i] for i in sorted(set(idx))]


def _selection_mean(result, key):
    """選択区間に限定した指標の平均 (straddle 除外は region_mask が担う)。"""
    mask = region_mask(result, "selection")
    v = np.asarray(result["metrics"][key])[mask]
    return float(np.nanmean(v))


def _selection_ess_over_n(result, n):
    """選択区間に限定した ESS/N の平均。"""
    hist = result.get("history")
    if not hist:
        return float("nan")
    from _common import masked_history
    mask = region_mask(result, "selection")
    d = summarize_history(masked_history(hist, mask))
    ess = d["pre_resample"].get("ess", np.nan)
    return float(ess / n) if np.isfinite(ess) else float("nan")


def _pf_loglik_std_median(cfg, ctx, n, params, seed):
    """S3: PF を選択区間で駆動し、各ステップの粒子間 loglik std の中央値を返す。

    PF の予測粒子群 (伝播後・重み付け前) 上で loglik_fn を評価する。
    run_method に計測フックが無いため、ここだけ ParticleFilter を直接駆動する。
    """
    from src.evaluation.runner import _build_estimator
    bench = build_benchmark(cfg, **ctx)
    funcs = bench.build_functions(seed)
    grad_fn, loglik_fn = funcs["grad_fn"], funcs["loglik_fn"]
    pf = _build_estimator("PF", bench, n, params, seed, funcs)
    stds = []
    for stp in bench.stream(seed):
        if not stp.is_selection_step:
            continue
        # 伝播 (drift + system noise) を step 内と同じ順で再現してから ll std を測る
        grad = grad_fn(pf.particles, stp.X_train, stp.y_train)
        prop = (pf.particles + pf.eta * grad
                + pf.rng.normal(0.0, pf.sigma_sys, size=pf.particles.shape))
        ll = np.asarray(loglik_fn(prop, stp.X_test, stp.y_test), float)
        if ll.size and np.isfinite(ll).any():
            stds.append(float(np.nanstd(ll)))
        # 実際の状態遷移は run_method と同一の step で進める (計測と分離)
        pf.step(stp.X_train, stp.y_train, grad_fn, loglik_fn)
    return float(np.median(stds)) if stds else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    help="regression / gefcom / gefcom_price / email / insects "
                         "または config パス")
    ap.add_argument("--sigma-cd", type=float, nargs="+", default=None,
                    help="スクリーニングする σ_cd 候補 (既定: config grid.sigma_sys)")
    ap.add_argument("--eta", type=float, nargs="+", default=None,
                    help="粗い η 格子 (既定: config grid.eta の 3 点)")
    ap.add_argument("--prior-std", type=float, default=None,
                    help="固定 prior_std (既定: config grid.prior_std の中央値)")
    ap.add_argument("--n", type=int, default=None,
                    help="粒子数 (既定: config n_particles.main)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="seed 集合 (既定: config seeds.selection)")
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    n = args.n or cfg["n_particles"]["main"]
    seeds = args.seeds or resolve_seeds(cfg, "selection")
    sigmas = args.sigma_cd or cfg["grid"]["sigma_sys"]
    etas = _coarse_etas(cfg, args.eta)
    ps_grid = cfg["grid"]["prior_std"]
    prior_std = args.prior_std if args.prior_std is not None \
        else ps_grid[len(ps_grid) // 2]

    # スクリーニングは単一コンテキスト。回帰系は本番 (grid_search) と同様に
    # 選択区間から σ_obs を推定して固定する。これにより S3 の loglik std が
    # 参照点 (INSECTS 0.37 / Solar 1.14) と同じスケールで比較可能になる。
    is_reg = cfg["task_type"] == "regression"
    ctx = {}
    if is_reg:
        sigma_obs = estimate_obs_noise(cfg)
        ctx = {"noise_std": sigma_obs}
        print(f"[σ_obs] 選択区間から推定: {sigma_obs:.4f} "
              f"(スクリーニング全実行で固定)")
    primary = "crps" if is_reg else "f1"   # S1 主指標。回帰は小さいほど良い
    secondary = "mse" if is_reg else "nll"
    minimize = is_reg   # 回帰 CRPS/MSE は最小化、分類 F1 は最大化

    print(f"=== screen_qcd: benchmark={cfg['benchmark']} N={n} "
          f"seeds={seeds} ===")
    print(f"σ_cd 候補: {sigmas}")
    print(f"粗い η: {etas}   prior_std(固定): {prior_std}")
    print(f"S1 主指標: {primary} (副: {secondary})\n")

    rows = []
    # --- σ_cd ごとに、粗い η を掃引して PF の最良 (η) を選ぶ ---
    per_sigma = {}
    for sc in sigmas:
        best = None   # (primary_val, eta, sec_val, ess)
        for eta in etas:
            params = {"eta": eta, "sigma_sys": sc, "prior_std": prior_std}
            pv, sv, ev = [], [], []
            for sd in seeds:
                r = run_method("PF", build_benchmark(cfg, **ctx), n, params, sd,
                               collect_diagnostics=True)
                pv.append(_selection_mean(r, primary))
                sv.append(_selection_mean(r, secondary))
                ev.append(_selection_ess_over_n(r, n))
            pmean = float(np.nanmean(pv))
            smean = float(np.nanmean(sv))
            emean = float(np.nanmean(ev))
            score = pmean if minimize else -pmean
            if best is None or score < best[0]:
                best = (score, eta, pmean, smean, emean)
        _, be, bp, bs, bess = best
        per_sigma[sc] = {"eta": be, primary: bp, secondary: bs,
                         "ess_over_N": bess}
        rows.append({"method": "PF", "sigma_cd": sc, "best_eta": be,
                     f"{primary}_mean": bp, f"{secondary}_mean": bs,
                     "ess_over_N": bess})
        print(f"PF  σ_cd={sc:<7} best_η={be:<6} {primary}={bp:.4f} "
              f"{secondary}={bs:.4f} ESS/N={bess:.3f}")

    # --- S1: PF 主指標が σ_cd に対して内部最適を持つか ---
    order = list(sigmas)
    prim_curve = [per_sigma[sc][primary] for sc in order]
    best_idx = int(np.argmin(prim_curve)) if minimize else int(np.argmax(prim_curve))
    s1 = 0 < best_idx < len(order) - 1
    sc_star = order[best_idx]
    eta_star = per_sigma[sc_star]["eta"]
    print(f"\n[S1] {primary} 曲線 (σ_cd順): "
          f"{[round(v,4) for v in prim_curve]}")
    print(f"[S1] 最適 σ_cd={sc_star} (index {best_idx}/{len(order)-1}) "
          f"→ 内部最適={'YES' if s1 else 'NO (端点)'}")

    # --- S2: 最適 σ_cd で WSPF-B を 1 点実行 → ρ_q50, ESS/N ---
    params_b = {"eta": eta_star, "sigma_sys": sc_star, "prior_std": prior_std}
    from _common import masked_history
    rho_q50_vals, ess_b_vals = [], []
    for sd in seeds:
        r = run_method("WSPF-B", build_benchmark(cfg, **ctx), n, params_b, sd,
                       collect_diagnostics=True)
        mask = region_mask(r, "selection")
        hr = masked_history(r["history"], mask)
        rr = rho_report(hr) or {}
        rho_q50_vals.append(rr.get("q50", np.nan))
        ess_b_vals.append(_selection_ess_over_n(r, n))
    rho_q50 = float(np.nanmean(rho_q50_vals))
    ess_b = float(np.nanmean(ess_b_vals))
    s2 = (S2_RHO_LOW <= rho_q50 <= S2_RHO_HIGH) and (ess_b > S2_ESS_MIN)
    print(f"\n[S2] WSPF-B @σ_cd={sc_star}: ρ_q50={rho_q50:.3f} "
          f"(要 [{S2_RHO_LOW},{S2_RHO_HIGH}]), ESS/N={ess_b:.3f} "
          f"(要 >{S2_ESS_MIN}) → {'PASS' if s2 else 'FAIL'}")

    # --- S3: PF の粒子間 loglik std 中央値 ---
    params_pf = {"eta": eta_star, "sigma_sys": sc_star, "prior_std": prior_std}
    ll_std_vals = [_pf_loglik_std_median(cfg, ctx, n, params_pf, sd)
                   for sd in seeds]
    ll_std = float(np.nanmedian(ll_std_vals))
    s3 = ll_std > S3_LOGLIK_STD_MIN
    print(f"[S3] PF 粒子間 loglik std 中央値={ll_std:.3f} "
          f"(要 >{S3_LOGLIK_STD_MIN}; INSECTS 0.37 / Solar 1.14) "
          f"→ {'PASS' if s3 else 'FAIL'}")

    verdict = s1 and s2 and s3
    print("\n" + "=" * 56)
    print(f"判定: S1={'✓' if s1 else '✗'}  S2={'✓' if s2 else '✗'}  "
          f"S3={'✓' if s3 else '✗'}  →  "
          f"{'GO (本採用, Phase 2 へ)' if verdict else 'NO-GO (境界事例に記録)'}")
    print("=" * 56)

    summary = {
        "benchmark": cfg["benchmark"], "n_particles": n, "seeds": list(seeds),
        "sigma_cd": list(sigmas), "coarse_eta": etas, "prior_std": prior_std,
        "primary_metric": primary, "secondary_metric": secondary,
        "per_sigma": {str(k): v for k, v in per_sigma.items()},
        "S1_interior_optimum": bool(s1),
        "S1_best_sigma_cd": float(sc_star), "S1_best_eta": float(eta_star),
        "S2_rho_q50": rho_q50, "S2_ess_over_N": ess_b, "S2_pass": bool(s2),
        "S3_loglik_std_median": ll_std, "S3_pass": bool(s3),
        "verdict_GO": bool(verdict),
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           cfg["output_dir"], "screen_qcd")
    save_run_dir(out_dir, config=cfg, selected_params={},
                 metrics_rows=rows, diagnostics={})
    with open(os.path.join(out_dir, "screen_verdict.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out_dir}")


if __name__ == "__main__":
    main()
