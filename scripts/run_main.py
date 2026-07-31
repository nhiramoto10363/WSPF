#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主結果の実行 (N=main, 評価シード全体)

各手法を、選択済みハイパーパラメータ・評価シード(10個)で実行し、
指標・診断量・実行成果物を outputs/<benchmark>/ に保存する。

  粒子フィルタ (PF/WSPF-A/WSPF-B) : N=main の最良 HP
  点推定 (SGD/PH-SGD/Window-SGD)  : 選択済み HP
  Oracle                          : Regression のみ(run_oracle.py 参照)

P3: GEFCom は config の zones をすべてループする(単一 zone に潰さない)。
    観測ノイズ σ_obs は選択区間から推定して固定する(--estimate-noise)。
P6: method × seed × zone ごとに予測・診断・index・timing を完全保存する
    (metrics.npz / predictions.npz / diagnostics.npz / indices.npz /
     timings.json / meta.json)。予測は報告区間サンプルを flatten して保存し、
     各サンプルがどのステップ由来かを pred_step_index / pred_offsets で対応づける。

使い方:
    python scripts/run_main.py --benchmark gefcom
"""

import argparse
import json
import os

import numpy as np

from _common import (load_config, resolve_seeds, build_benchmark,
                     load_selected, get_params, region_mask, estimate_obs_noise)
from src.evaluation import (run_method, save_run_dir, mean_std, sanitize,
                            write_json, summarize_history, timing_report,
                            paired_compare, wilcoxon_signed)

_REPO = os.path.dirname(os.path.dirname(__file__))

# R1-9 で集約する診断量(report区間平均)のキー
_DIAG_KEYS = ["ess_over_N", "weight_entropy", "max_weight",
              "particle_spread", "unique_ancestor_rate", "resample_rate"]


def _report_diag(history, report_mask, n_particles):
    """フィルタ履歴から report 区間平均の R1-9 診断量を返す(M3)。

    測定時点は固定(重み正規化後・リサンプリング前: ess/entropy/max_weight/
    spread、リサンプリング後: resampled/unique)。report_mask で報告区間のみ集計。
    """
    m = np.asarray(report_mask, bool)

    def rmean(key):
        a = np.asarray(history.get(key, []), dtype=float)
        if a.size != m.size or not m.any():
            return float("nan")
        return float(np.nanmean(a[m]))

    return {
        "ess_over_N": rmean("ess") / n_particles,
        "weight_entropy": rmean("entropy"),
        "max_weight": rmean("max_weight"),
        "particle_spread": rmean("spread_trace"),
        "unique_ancestor_rate": rmean("unique_particles") / n_particles,
        "resample_rate": rmean("resampled"),
    }


def _save_per_seed(base, method, seed, zone, r, cfg):
    """1 実行(method×seed×zone)の完全な成果物を保存(P6)。"""
    tag = f"{sanitize(method)}_seed{seed}"
    if zone is not None:
        tag += f"_zone{zone}"
    d = os.path.join(base, "runs", tag)
    os.makedirs(d, exist_ok=True)

    # metrics(ステップ系列) + マスク
    np.savez_compressed(
        os.path.join(d, "metrics.npz"),
        step_index=r["step_index"],
        report_mask=r["report_mask"], selection_mask=r["selection_mask"],
        straddle_mask=r["straddle_mask"], switch_mask=r["switch_mask"],
        regime_ids=r["regime_ids"],
        **{f"metric_{k}": v for k, v in r["metrics"].items()})
    # 予測・正解・確率(報告区間サンプル単位)
    np.savez_compressed(os.path.join(d, "predictions.npz"), **r["predictions"])
    # train/test インデックス(リーク検査・再現用)
    idx = {}
    for i, (tr, te) in enumerate(zip(r["train_indices"], r["test_indices"])):
        idx[f"train_{i}"] = np.asarray(tr, int)
        idx[f"test_{i}"] = np.asarray(te, int)
    np.savez_compressed(os.path.join(d, "indices.npz"), **idx)
    # 診断履歴(フィルタのみ)
    if r.get("history"):
        np.savez_compressed(os.path.join(d, "diagnostics.npz"), **r["history"])
        write_json(timing_report(r["history"]), os.path.join(d, "timings.json"))
    write_json({"method": method, "seed": int(seed),
                "zone": zone, "n_resets": r["n_resets"]},
               os.path.join(d, "meta.json"))


def _run_one_config(cfg, out_dir, selected, eval_seeds, n_main, methods,
                    zone=None):
    """1 つの (config, zone) について全手法×全シードを実行し集計行を返す。"""
    overrides = {}
    if zone is not None:
        overrides["zone"] = zone
    # GEFCom は grid_search が保存した σ_obs(zone別)を使う。grid と final で
    # 同じ観測ノイズを使うことで尤度・重み更新の条件を一致させる(C2)。
    if cfg["benchmark"] == "gefcom":
        noise_by_zone = selected.get("gefcom_noise", {})
        sigma = noise_by_zone.get(str(zone))
        if sigma is None:                       # 再現性のためデフォルトへ落ちない
            raise KeyError(
                f"GEFCom zone {zone} の noise_std が selected_params にありません。"
                f"grid_search.py --benchmark gefcom を先に実行してください。")
        overrides["noise_std"] = sigma
        print(f"  [zone {zone}] σ_obs = {sigma:.4f} (grid と共通)")

    key = "mse" if cfg["task_type"] == "regression" else "f1"
    ztag = f"[zone {zone}] " if zone is not None else ""
    rows = []
    per_seed = {}       # method -> [report指標(seed別)]  (paired test 用, M2)
    for m in methods:
        params = get_params(selected, m, n_main)
        vals, diags = [], []
        for s in eval_seeds:
            bench = build_benchmark(cfg, **overrides)
            r = run_method(m, bench, n_main, params, s, collect_diagnostics=True)
            mask = region_mask(r, "report")
            vals.append(np.nanmean(np.asarray(r["metrics"][key])[mask]))
            if r.get("history"):
                diags.append(_report_diag(r["history"], r["report_mask"], n_main))
            _save_per_seed(out_dir, m, s, zone, r, cfg)
        per_seed[m] = vals
        mu, sd = mean_std(vals)
        rows.append({"method": m, "N": n_main, "zone": zone, "kind": "performance",
                     "metric": key, "mean": mu, "std": sd,
                     "n_seeds": len(eval_seeds)})
        print(f"  {ztag}{m:12s} {key}={mu:.4f} ± {sd:.4f}")
        # R1-9 診断量の集約(report区間, seed 横断 mean±SD)(M3)
        for dk in _DIAG_KEYS:
            dv = [d[dk] for d in diags if np.isfinite(d.get(dk, np.nan))]
            if dv:
                dm, ds = mean_std(dv)
                rows.append({"method": m, "N": n_main, "zone": zone,
                             "kind": "diagnostic", "metric": dk,
                             "mean": dm, "std": ds, "n_seeds": len(dv)})

    # M2: 主結果の対応検定(WSPF-A/B − PF)。R1-14。
    if "PF" in per_seed:
        pf = np.asarray(per_seed["PF"], float)
        for m in ("WSPF-A", "WSPF-B"):
            if m not in per_seed:
                continue
            a = np.asarray(per_seed[m], float)
            cmp = paired_compare(list(a), list(pf))
            wil = wilcoxon_signed(list(a), list(pf))
            rows.append({
                "method": f"{m}_vs_PF", "N": n_main, "zone": zone,
                "kind": "paired", "metric": f"{key}_paired",
                "mean_difference": cmp.get("mean_diff"),
                "std_difference": float(np.std(a - pf, ddof=1)) if a.size > 1 else 0.0,
                "paired_t_p": cmp.get("p"), "wilcoxon_p": wil.get("p"),
            })
            print(f"  {ztag}{m}−PF: Δ{key}={cmp.get('mean_diff'):.4f} "
                  f"(t-p={cmp.get('p'):.3g})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    args = ap.parse_args()

    cfg = load_config(args.benchmark)
    selected = load_selected(cfg)
    eval_seeds = resolve_seeds(cfg, "evaluation")
    n_main = cfg["n_particles"]["main"]
    methods = [m for m in cfg["methods"] if m != "NoChange"]
    out_dir = os.path.join(_REPO, cfg["output_dir"], "main")

    # GEFCom は zones をすべてループ(P3)。他は zone=None の 1 回。
    zones = cfg.get("data", {}).get("zones") if cfg["benchmark"] == "gefcom" else None

    all_rows = []
    if zones:
        for z in zones:
            all_rows += _run_one_config(
                cfg, out_dir, selected, eval_seeds, n_main, methods, zone=z)
    else:
        all_rows += _run_one_config(
            cfg, out_dir, selected, eval_seeds, n_main, methods)

    save_run_dir(out_dir, config=cfg, selected_params=selected,
                 metrics_rows=all_rows, diagnostics={})
    print(f"保存: {out_dir}  (per-seed 成果物は {out_dir}/runs/)")


if __name__ == "__main__":
    main()
