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
                            write_json, summarize_history, timing_report)

_REPO = os.path.dirname(os.path.dirname(__file__))


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
        if sigma is None:                       # 後方互換: 未保存なら推定
            sigma = estimate_obs_noise(cfg, zone=zone)
        overrides["noise_std"] = sigma
        print(f"  [zone {zone}] σ_obs = {sigma:.4f} (grid と共通)")

    key = "mse" if cfg["task_type"] == "regression" else "f1"
    rows = []
    for m in methods:
        params = get_params(selected, m, n_main)
        vals = []
        for s in eval_seeds:
            bench = build_benchmark(cfg, **overrides)
            r = run_method(m, bench, n_main, params, s, collect_diagnostics=True)
            mask = region_mask(r, "report")
            vals.append(np.nanmean(np.asarray(r["metrics"][key])[mask]))
            _save_per_seed(out_dir, m, s, zone, r, cfg)
        mu, sd = mean_std(vals)
        rows.append({"method": m, "N": n_main, "zone": zone, "metric": key,
                     "mean": mu, "std": sd, "n_seeds": len(eval_seeds)})
        ztag = f"[zone {zone}] " if zone is not None else ""
        print(f"  {ztag}{m:12s} {key}={mu:.4f} ± {sd:.4f}")
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
