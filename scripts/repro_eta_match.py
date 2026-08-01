"""η を揃えると PF と SGD の順位が逆転するかの再現(回帰, seeds 0-2)。

PF: sigma_sys=0.01, prior_std=0.75 固定で eta のみ掃引。
SGD: 選択HP (eta=0.2, prior_std=0.1)。
report 区間 MSE = nanmean(metrics["mse"][region_mask(r,"report")]) を seed 平均。
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # _common
sys.path.insert(0, os.path.dirname(_HERE))     # src

from _common import load_config, build_benchmark, region_mask
from src.evaluation import run_seeds

SEEDS = [0, 1, 2]
N = 100

def report_mse(method, bench, params):
    results = run_seeds(method, bench, N, params, SEEDS,
                        collect_diagnostics=False)
    vals = []
    for r in results:
        mask = region_mask(r, "report")
        vals.append(np.nanmean(np.asarray(r["metrics"]["mse"])[mask]))
    return float(np.mean(vals)), vals


def main():
    cfg = load_config("regression")
    bench = build_benchmark(cfg)

    etas = [0.01, 0.05, 0.1, 0.2]
    # 各手法: 選択された σ_sys/prior_std/beta を固定し eta のみ掃引。
    fixed = {
        "PF":     {"sigma_sys": 0.01, "prior_std": 0.75},
        "WSPF-A": {"sigma_sys": 0.01, "prior_std": 1.0, "beta": 0.8},
        "WSPF-B": {"sigma_sys": 0.01, "prior_std": 0.5},
    }
    for method, base in fixed.items():
        sel_eta = 0.01  # 3手法とも選択値は eta=0.01
        print(f"=== {method}: {base}, eta 掃引 (seeds 0-2) [選択値 η={sel_eta}] ===")
        for eta in etas:
            params = dict(base, eta=eta)
            mu, vals = report_mse(method, bench, params)
            tag = " ← 選択値" if eta == sel_eta else ""
            print(f"  η={eta:<5} → report MSE = {mu:.3f}   "
                  f"(per-seed {[round(v,3) for v in vals]}){tag}")

    print("=== SGD: 選択HP eta=0.2, prior_std=0.1 (seeds 0-2) ===")
    mu, vals = report_mse("SGD", bench, {"eta": 0.2, "prior_std": 0.1})
    print(f"  SGD η=0.2 → report MSE = {mu:.3f}   (per-seed {[round(v,3) for v in vals]})")


if __name__ == "__main__":
    main()
