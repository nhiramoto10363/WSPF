README: Figure and Table Generation for WSPF Paper
====================================================

Script: experiments/generate_paper_figures.py

Usage:
  # Full run (re-run experiments + generate all figures/tables):
  python experiments/generate_paper_figures.py

  # Plot-only mode (reuse intermediate data):
  python experiments/generate_paper_figures.py --plot-only

Data Sources:
  - Regression regime-switch: synthetic data generated per seed
    (T=500, 10 seeds, N=1000 particles)
  - Elec2: Elec2/electricity-normalized.csv
    (seed=42, N=1000 particles, batch_size=64)

Hyperparameters:
  - Loaded from grid search results:
    outputs/regression_regime_switch/grid_search_result.json
    outputs/elec2/grid_search_result.json

Warm-up / Evaluation Exclusion Rules:
  - Regression: first 50 steps excluded from aggregate metrics (EVAL_START=50)
  - Elec2: first 10% of steps excluded (eval_start = step // 10)

Intermediate Data:
  - outputs/intermediate/regression_per_step.npz
    Keys: mse_sgd, mse_pf, mse_wspf_b, mse_wspf_a (shape 10x500),
          rho_wspf_b, rho_wspf_a (shape 10x500),
          switch_times (shape 4,)
  - outputs/intermediate/elec2_per_step.npz
    Keys: acc_*/f1_*/ll_* for each method (shape n_steps,),
          rho_wspf_b, rho_wspf_a (shape n_steps,),
          eval_start (scalar)

Output Files:
  Tables:
    outputs/tables/table1_regime_switching_results.csv
    outputs/tables/table2_elec2_results.csv
  Figures:
    outputs/figures/figure1_regime_switch_recovery.{pdf,png}
    outputs/figures/figure2_elec2_rolling_f1.{pdf,png}
    outputs/figures/figure3_rho_diagnostic.{pdf,png}
  Metadata:
    outputs/captions.txt
    outputs/README_figure_table_generation.txt
