"""gefcom 各手法の 真値 vs 予測値 散布図(report区間, 全seedプール)。

保存済み outputs/gefcom/main/runs/<method>_seed<s>_zone<z>/predictions.npz を読む。
可視化は最大 4000 点にサブサンプル、RMSE/相関は全プールで計算。
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(_HERE), "outputs/gefcom/main/runs")

METHODS = ["SGD", "PH-SGD", "Window-SGD", "PF", "WSPF-A", "WSPF-B"]
ZONES = [1, 2, 3]
SANI = {"SGD": "sgd", "PH-SGD": "ph_sgd", "Window-SGD": "window_sgd",
        "PF": "pf", "WSPF-A": "wspf_a", "WSPF-B": "wspf_b"}
RNG = np.random.default_rng(0)
MAXPTS = 4000


def load_pool(method, zone):
    y_all, p_all = [], []
    for f in sorted(glob.glob(os.path.join(RUNS, f"{SANI[method]}_seed*_zone{zone}",
                                            "predictions.npz"))):
        z = np.load(f)
        y_all.append(np.asarray(z["y"], float))
        p_all.append(np.asarray(z["mean"], float))
    if not y_all:
        return None, None
    return np.concatenate(y_all), np.concatenate(p_all)


def main():
    nr, nc = len(METHODS), len(ZONES)
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.0 * nr),
                             sharex=True, sharey=True)
    for i, m in enumerate(METHODS):
        for j, zn in enumerate(ZONES):
            ax = axes[i, j]
            y, p = load_pool(m, zn)
            if y is None:
                ax.set_visible(False); continue
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            corr = float(np.corrcoef(y, p)[0, 1])
            # サブサンプル(描画のみ)
            if y.size > MAXPTS:
                idx = RNG.choice(y.size, MAXPTS, replace=False)
                ys, ps = y[idx], p[idx]
            else:
                ys, ps = y, p
            ax.scatter(ys, ps, s=3, alpha=0.15, edgecolors="none", color="#2b6cb0")
            lo = min(y.min(), p.min()); hi = max(y.max(), p.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.8)
            ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.35, 1.0)
            ax.text(0.04, 0.96, f"RMSE={rmse:.3f}\nr={corr:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0))
            if i == 0:
                ax.set_title(f"zone {zn}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{m}\npred", fontsize=9)
            if i == nr - 1:
                ax.set_xlabel("true", fontsize=9)
    fig.suptitle("GEFCom2014 Solar: 真値 vs 予測値 (report区間, 全10seedプール)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(os.path.dirname(_HERE), "outputs/gefcom/scatter_pred_vs_true.png")
    fig.savefig(out, dpi=110)
    print("保存:", out)

    # 数値サマリ
    print(f"\n{'method':12s} " + " ".join(f"z{z}:RMSE  r    " for z in ZONES))
    for m in METHODS:
        line = f"{m:12s} "
        for zn in ZONES:
            y, p = load_pool(m, zn)
            rmse = np.sqrt(np.mean((p - y) ** 2)); r = np.corrcoef(y, p)[0, 1]
            line += f"{rmse:.3f} {r:.3f}  "
        print(line)


if __name__ == "__main__":
    main()
