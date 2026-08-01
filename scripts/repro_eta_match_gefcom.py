"""gefcom でも η を揃えると順位が逆転するかの確認(zone別, seeds 0-2)。

PF/WSPF-A/WSPF-B: 選択された sigma_sys/prior_std/beta を固定し eta のみ掃引。
SGD/Window-SGD: 選択HP。
zone別 σ_obs は selected_params["gefcom_noise"] を使用(grid と共通, C2)。
report 区間 MSE = nanmean(metrics["mse"][region_mask(r,"report")]) を seed 平均。
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from _common import load_config, build_benchmark, region_mask
from src.evaluation import run_seeds

SEEDS = [0, 1, 2]
N = 100
ETAS = [0.005, 0.01, 0.05, 0.1, 0.2]

# 選択HP(gefcom/selected_params.json より)
FIXED = {
    "PF":     {"sigma_sys": 0.01, "prior_std": 0.1},
    "WSPF-A": {"sigma_sys": 0.01, "prior_std": 0.1, "beta": 0.9},
    "WSPF-B": {"sigma_sys": 0.01, "prior_std": 0.1},
}
SEL_ETA = {"PF": 0.01, "WSPF-A": 0.005, "WSPF-B": 0.01}
BASELINES = {
    "SGD": {"eta": 0.05, "prior_std": 0.01},
    "Window-SGD": {"eta": 0.05, "prior_std": 0.01, "window": 10, "n_passes": 2},
}


def report_mse(method, bench, params):
    results = run_seeds(method, bench, N, params, SEEDS,
                        collect_diagnostics=False)
    vals = [np.nanmean(np.asarray(r["metrics"]["mse"])[region_mask(r, "report")])
            for r in results]
    return float(np.mean(vals)), vals


def main():
    cfg = load_config("gefcom")
    sel = json.load(open(os.path.join(os.path.dirname(_HERE), "outputs/gefcom/selected_params.json")))
    noise_by_zone = sel["gefcom_noise"]
    zones = cfg["data"]["zones"]

    for z in zones:
        sigma = noise_by_zone[str(z)]
        bench = build_benchmark(cfg, zone=z, noise_std=sigma)
        print(f"\n############ zone {z}  (σ_obs={sigma:.4f}) ############")
        for method, base in FIXED.items():
            print(f"--- {method}: {base}, eta 掃引 [選択値 η={SEL_ETA[method]}] ---")
            for eta in ETAS:
                mu, vals = report_mse(method, bench, dict(base, eta=eta))
                tag = " ← 選択値" if eta == SEL_ETA[method] else ""
                print(f"  η={eta:<6} → MSE={mu:.4f}  {[round(v,4) for v in vals]}{tag}")
        for method, params in BASELINES.items():
            mu, vals = report_mse(method, bench, params)
            print(f"--- {method} (η={params['eta']}) → MSE={mu:.4f}  {[round(v,4) for v in vals]}")


if __name__ == "__main__":
    main()
