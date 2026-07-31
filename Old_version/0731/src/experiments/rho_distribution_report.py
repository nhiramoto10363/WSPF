#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ρ_t(i) の経験分布レポート — 3 ベンチマーク × WSPF-A/B (R1-8)

R1-8 は "the empirical distribution of ρ_t(i)" (per-particle) を要求する。
フィルタは step ごとに ρ の全粒子ベクトル (N,) を history["rho"] に保存して
おり、各ベンチマークの診断 npz に per-particle ρ が (n_steps, N) 形状で
既に永続化されている。よって **フィルタ拡張も再実行も不要な純後処理**
(spec の collect_rho_full フラグ案は、既存 history["rho"] が既に per-particle
を保持しているため不要)。

データソース (いずれも各手法 own-best HP):
  - Regression sim : outputs/regression_regime_switch/diagnostics_N100.npz
                     wspf_{a,b}_rho (n_seeds, T, N)  → seed 0, report 区間
  - GEFCom zone 1  : outputs/gefcom/results_zone1_N100_seed42.npz
                     diag_WSPF_{A,B}_rho (n_steps, N) → report 区間 (eval_mask)
  - INSECTS        : outputs/insects/results_N100_seed42.npz
                     diag_WSPF_{A,B}_rho (n_steps, N) → report 区間 (eval_mask)

集計 (手法・データセット別): 分位点 5/25/50/75/95/99%、mean、max、
P(ρ>0.9)、P(ρ>0.99)、クリップ (ρ≥0.999) 発動割合。
図: 3 パネル (データセット別) × A/B 重ね描きヒストグラム、x∈[0,1] 線形。

設計判断: A の ρ は補正に使われない診断量 (補正に使うのは B のみ) だが、
両方載せて「A/B で ρ 分布はほぼ同一 → B の等方近似が要約する量は共通、
差は方向情報の有無」を示す。

出力: outputs/rho_distribution/rho_distribution.{txt,csv,png}
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO_METHODS = ["WSPF-A", "WSPF-B"]
RHO_CLIP = 0.999
BASE = os.path.join(os.path.dirname(__file__), "..", "..")

OUTPUT_DIR = os.path.join(BASE, "outputs", "rho_distribution")


def _load_regression():
    path = os.path.join(BASE, "outputs", "regression_regime_switch",
                        "diagnostics_N100.npz")
    if not os.path.exists(path):
        return None, path
    d = np.load(path, allow_pickle=True)
    es = int(d["eval_start"]) if "eval_start" in d else 0
    out = {}
    for m, key in [("WSPF-A", "wspf_a_rho"), ("WSPF-B", "wspf_b_rho")]:
        if key not in d:
            return None, path
        rho = np.asarray(d[key])          # (n_seeds, T, N)
        out[m] = rho[0, es:, :].reshape(-1)   # seed 0, report region
    return out, f"{os.path.basename(path)} (seed 0, report region)"


def _load_eval_npz(rel, label):
    path = os.path.join(BASE, *rel)
    if not os.path.exists(path):
        return None, path
    d = np.load(path, allow_pickle=True)
    eval_mask = np.asarray(d["eval_mask"], dtype=bool)
    out = {}
    for m in RHO_METHODS:
        key = "diag_" + m.replace("-", "_") + "_rho"
        if key not in d:
            return None, path
        rho = np.asarray(d[key])          # (n_steps, N)
        sel = rho[eval_mask] if rho.shape[0] == eval_mask.shape[0] else rho
        out[m] = sel.reshape(-1)
    return out, f"{os.path.basename(path)} ({label})"


def _stats(r):
    q = np.percentile(r, [5, 25, 50, 75, 95, 99])
    return {
        "n": int(r.size),
        "mean": float(r.mean()),
        "max": float(r.max()),
        "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
        "p_gt_090": float(np.mean(r > 0.9)),
        "p_gt_099": float(np.mean(r > 0.99)),
        "clip_frac": float(np.mean(r >= RHO_CLIP)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = []
    reg, reg_src = _load_regression()
    datasets.append(("Regression", reg, reg_src))
    gef, gef_src = _load_eval_npz(
        ["outputs", "gefcom", "results_zone1_N100_seed42.npz"],
        "seed 42, report region")
    datasets.append(("GEFCom-z1", gef, gef_src))
    ins, ins_src = _load_eval_npz(
        ["outputs", "insects", "results_N100_seed42.npz"],
        "seed 42, report region")
    datasets.append(("INSECTS", ins, ins_src))

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("Empirical distribution of ρ_t(i) — per-particle (R1-8)")
    emit(f"  clip threshold = {RHO_CLIP}")
    for name, data, src in datasets:
        status = "OK" if data is not None else "MISSING (先に該当実験を実行)"
        emit(f"  {name:<12s}: {src}  [{status}]")
    emit("=" * 78)

    csv_rows = [("dataset", "method", "n", "mean", "max", "p5", "p25", "p50",
                 "p75", "p95", "p99", "P_gt_0.9", "P_gt_0.99", "clip_frac")]
    stats_all = {}
    for name, data, _ in datasets:
        if data is None:
            continue
        emit(f"\n  ===== {name} =====")
        emit(f"  {'method':<8s} {'n':>9s} {'mean':>7s} {'max':>7s} "
             f"{'p5':>6s} {'p25':>6s} {'p50':>6s} {'p75':>6s} {'p95':>6s} "
             f"{'p99':>6s} {'P>.9':>7s} {'P>.99':>7s} {'clip%':>7s}")
        for m in RHO_METHODS:
            st = _stats(data[m])
            stats_all[(name, m)] = st
            emit(f"  {m:<8s} {st['n']:>9d} {st['mean']:>7.4f} "
                 f"{st['max']:>7.4f} {st['p5']:>6.3f} {st['p25']:>6.3f} "
                 f"{st['p50']:>6.3f} {st['p75']:>6.3f} {st['p95']:>6.3f} "
                 f"{st['p99']:>6.3f} {st['p_gt_090']:>7.4f} "
                 f"{st['p_gt_099']:>7.4f} {100 * st['clip_frac']:>6.3f}%")
            csv_rows.append((name, m, st["n"], f"{st['mean']:.6f}",
                             f"{st['max']:.6f}", f"{st['p5']:.6f}",
                             f"{st['p25']:.6f}", f"{st['p50']:.6f}",
                             f"{st['p75']:.6f}", f"{st['p95']:.6f}",
                             f"{st['p99']:.6f}", f"{st['p_gt_090']:.6f}",
                             f"{st['p_gt_099']:.6f}", f"{st['clip_frac']:.6f}"))

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "rho_distribution.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---- 図: 3 パネル (データセット) × A/B 重ね描き ----
    present = [(n, d) for n, d, _ in datasets if d is not None]
    if present:
        colors = {"WSPF-A": "#D55E00", "WSPF-B": "#E69F00"}
        bins = np.linspace(0.0, 1.0, 51)
        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4.2),
                                 squeeze=False)
        for ax, (name, data) in zip(axes[0], present):
            for m in RHO_METHODS:
                ax.hist(data[m], bins=bins, density=True, histtype="step",
                        color=colors[m], linewidth=1.6, label=m)
                ax.axvline(np.median(data[m]), color=colors[m], ls=":", lw=1.0)
            ax.set_title(f"{name}  (ρ empirical dist.)")
            ax.set_xlabel(r"$\rho_t(i)$")
            ax.set_ylabel("density")
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Empirical per-particle ρ distribution "
                     "(own-best HP, report region)")
        plt.tight_layout()
        png = os.path.join(OUTPUT_DIR, "rho_distribution.png")
        plt.savefig(png, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        png = "(no data — 図スキップ)"

    # ---- 解釈補助: A/B の分布類似性と退化端の回避 ----
    emit("\n  解釈:")
    for name, data, _ in datasets:
        if data is None:
            continue
        a = stats_all[(name, "WSPF-A")]
        b = stats_all[(name, "WSPF-B")]
        emit(f"    {name}: median ρ  A={a['p50']:.3f} / B={b['p50']:.3f}, "
             f"clip% A={100 * a['clip_frac']:.2f} / B={100 * b['clip_frac']:.2f} "
             f"(0 なら退化端を回避)")

    txt_path = os.path.join(OUTPUT_DIR, "rho_distribution.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nSaved: {txt_path}")
    emit(f"Saved: {csv_path}")
    emit(f"Saved: {png}")


if __name__ == "__main__":
    main()
