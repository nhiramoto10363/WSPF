"""選択区間(t<50, 切替なし)と report区間(t>=50, 切替4回)で
PF の最適 eta が逆転することを示す(回帰)。

grid_search が実際に見るのは selection seeds(1000-1002)× selection region。
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from _common import load_config, build_benchmark, region_mask, resolve_seeds
from src.evaluation.runner import run_method

N = 100
ETAS = [0.01, 0.05, 0.1, 0.2]
BASE = {"sigma_sys": 0.01, "prior_std": 0.75}


def region_mse(r, region):
    mask = region_mask(r, region)
    return float(np.nanmean(np.asarray(r["metrics"]["mse"])[mask]))


def main():
    cfg = load_config("regression")
    sel_seeds = resolve_seeds(cfg, "selection")   # [1000,1001,1002]
    print(f"selection seeds = {sel_seeds},  eval_start(=選択/報告境界) = {cfg['eval'].get('eval_start')}")
    print(f"switch_points = {cfg['data']['switch_points']}  (全て報告区間 t>=50 内)\n")

    print(f"{'eta':>6} | {'選択区間MSE (t<50, 切替なし)':>28} | {'報告区間MSE (t>=50, 切替4回)':>28}")
    print("-" * 72)
    sel_scores, rep_scores = {}, {}
    for eta in ETAS:
        params = dict(BASE, eta=eta)
        sels, reps = [], []
        for s in sel_seeds:
            r = run_method("PF", build_benchmark(cfg), N, params, seed=s,
                           collect_diagnostics=False)
            sels.append(region_mse(r, "selection"))
            reps.append(region_mse(r, "report"))
        sel_scores[eta] = float(np.mean(sels))
        rep_scores[eta] = float(np.mean(reps))
        print(f"{eta:>6} | {sel_scores[eta]:>28.4f} | {rep_scores[eta]:>28.4f}")

    best_sel = min(sel_scores, key=sel_scores.get)
    best_rep = min(rep_scores, key=rep_scores.get)
    print(f"\n選択区間で最良の eta = {best_sel}  (grid_search が選ぶ値)")
    print(f"報告区間で最良の eta = {best_rep}  (本来ほしい値)")
    print("→ 逆転している" if best_sel != best_rep else "→ 一致")


if __name__ == "__main__":
    main()
