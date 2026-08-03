import numpy as np
from _common import (load_config, build_benchmark, resolve_seeds,
                     masked_history, region_mask)
from src.evaluation.runner import run_method
from src.evaluation.diagnostics import rho_report
from scripts.screen_qcd import estimate_obs_noise, _selection_mean, _selection_ess_over_n

cfg = load_config("gefcom_price")
n = cfg["n_particles"]["main"]
sd = resolve_seeds(cfg,"selection")[0]
sigma_obs = estimate_obs_noise(cfg)
ctx={"noise_std":sigma_obs}
ps = cfg["grid"]["prior_std"]; prior_std=ps[len(ps)//2]
print(f"σ_obs={sigma_obs:.4f} prior_std={prior_std} seed={sd} N={n}", flush=True)
print(f"{'σ_cd':>7} {'eta':>6} | {'ρ_q50':>7} {'CRPS↓':>8} {'MSE↓':>8} {'ESS/N':>6}  S2?", flush=True)
print("-"*62, flush=True)
for sc in [0.005, 0.025]:
    for eta in [0.01, 0.2, 1.0]:
        p={"eta":eta,"sigma_sys":sc,"prior_std":prior_std}
        r=run_method("WSPF-B", build_benchmark(cfg,**ctx), n, p, sd, collect_diagnostics=True)
        hr=masked_history(r["history"], region_mask(r,"selection"))
        rho=(rho_report(hr) or {}).get("q50",np.nan)
        crps=_selection_mean(r,"crps"); mse=_selection_mean(r,"mse")
        ess=_selection_ess_over_n(r,n)
        ok = "GO" if (0.1<=rho<=0.9 and ess>0.1) else ""
        print(f"{sc:>7} {eta:>6} | {rho:>7.4f} {crps:>8.4f} {mse:>8.4f} {ess:>6.3f}  {ok}", flush=True)
print("DONE", flush=True)
