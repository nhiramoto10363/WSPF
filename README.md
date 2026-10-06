# WSPF: Importance Weight Correction for SGD-Driven Particle Filters

Replication package for

> N. Hiramoto, G. Ueno, D. Murakami.
> *Importance Weight Correction for Stochastic Gradient Descent-Driven
> Particle Filters under Concept Drift.*

Every table and figure reported in the paper can be regenerated with the
scripts in this repository.

The paper argues that likelihood-only weighting in an SGD-driven particle
filter is statistically inconsistent, because it omits the prior-proposal
correction for the observation-dependent proposal induced by the realized
mini-batch. This package implements that correction (WSPF-A and WSPF-B), the
oracle experiment showing that the correction removes the inconsistency, and
the diagnostic that quantifies when the correction pays off.

## Contents

```
src/                 methods and benchmarks
  filters/           PF (uncorrected) / WSPF-A / WSPF-B / Oracle
  baselines/         SGD / Window-SGD / PH-SGD
  benchmarks/        regression (synthetic) / email (elist) / insects
    loaders/         ARFF and CSV parsing, leak-free preprocessing
  models/            regression MLP, binary MLP, multiclass MLP
  evaluation/        run loop, metrics, paired tests, diagnostics, output
scripts/             experiment drivers and table/figure generation
configs/             per-benchmark settings (grids, seeds, evaluation windows)
tests/               unit tests for the protocol, statistics and numerics
data/                raw data goes here (not shipped; see data/README.md)
results/             finalized results reported in the paper
run_all.sh           runs everything
```

`results/` lets you inspect the published numbers without running anything.
Re-running writes to `outputs/`, so comparing files of the same name under the
two directories tells you whether the run reproduced the paper.

## Setup

```bash
pip install -r requirements.txt
```

The finalized results were produced with Python 3.9.13, numpy 1.21.4,
scipy 1.7.3 and numba 0.55.1. Python 3.11 with numpy 2.2 also works.

### Data

The regression benchmark is synthetic and needs no download. The two real
streams are not redistributed here; obtain them from their original sources.
Provenance, terms and verification are described in
**[data/README.md](data/README.md)**.

```bash
python scripts/prepare_data.py \
    --email-from-file   /path/to/email_data.arff \
    --insects-from-file /path/to/INSECTS-abrupt_balanced_norm.csv
```

This checks the SHA-256 against the version used for the paper and verifies
the stream structure (sample count, feature dimension, per-period positive
rate for elist, class count and change points for INSECTS). If both pass, you
have the same streams the paper used.

## Running

```bash
./run_all.sh                 # three benchmarks, then tables and figures
./run_all.sh regression      # a single benchmark
./run_all.sh tables          # rebuild tables and figures from existing outputs/
```

The grid search dominates the cost. To run the experiments with the
hyper-parameters selected in the paper instead:

```bash
mkdir -p outputs/regression
cp results/regression/selected_params.json outputs/regression/
SKIP_GRID=1 ./run_all.sh regression
```

Parallelism is controlled by `WSPF_NUM_WORKERS` (defaults to the number of
available cores).

## Mapping to the paper

Re-running writes to the paths below; `results/` holds the finalized versions
under the same structure. Tables are written as `.csv`, `.txt` and `.tex`.

### Main text

| Paper | Content | Artifact | Script |
|---|---|---|---|
| Table 2 | Regression, mechanism axis (PF / WSPF-B / WSPF-A / Oracle) | `outputs/regression/table_mechanism.*` | `summarize_results.py` |
| Table 3 | Regression, context axis (all six methods) | `outputs/regression/table_context.*` | `summarize_results.py` |
| Table 5 | elist classification over the reporting window | `outputs/email/table_context.*` | `summarize_results.py` |
| Table 6 | Applicability diagnostic R and particle budget | `outputs/applicability/table_applicability.*` | `analyze_applicability.py` |
| Table 7 | Absolute ESS against the number of particles | `outputs/*/n_sweep/metrics.*` | `run_n_sweep.py` |
| Table 8 | Regression MSE against mini-batch size | `outputs/regression/batch_size_sweep/metrics.*` | `run_batch_size_sweep.py` |
| Table 9 | Regression, matched-hyper-parameter comparison | `outputs/regression/matched_hp/metrics.*` | `run_matched.py` |
| Table 10 | elist, matched-hyper-parameter comparison | `outputs/email/matched_hp/metrics.*` | `run_matched.py` |
| Figure 2 | Function-approximation snapshot at t=305 | `outputs/figures/snapshot_func_t305_N100.png` | `plot_snapshot_func.py` |
| Figure 3 | Regression MSE over time | `outputs/figures/regression_regime_switch_timeseries_N100.png` | `plot_mse_timeseries.py` |
| Figure 4 | ESS against the number of particles | `outputs/figures/ess_sweep.png` | `plot_ess_sweep.py` |
| Figure 5 | Switch-aligned recovery, regression | `outputs/regression/figures/recovery_post_switch.png` | `plot_recovery.py` |
| Figure 6 | Switch-aligned recovery, elist | `outputs/figures/figure_email_switch_recovery.png` | `plot_recovery_classification.py` |

Table 1 (notation), Table 4 (the elist label-reversal schedule) and Figure 1
(a schematic) are not experimental output and have no generating script.

Figure filenames match the names used in the paper, except Figure 5, which is
written as `recovery_post_switch.png` (use `--outfile` to change it).

### Checking that a run reproduced the paper

Compare `outputs/<bench>/main/metrics.csv` against the file of the same name
under `results/`. For the regression benchmark, running this package with the
published hyper-parameters reproduces **all 1,260 cells** of the finalized
`metrics.csv` under Python 3.11 / numpy 2.2.6 / scipy 1.17.1, even though the
finalized results were produced under Python 3.9 / numpy 1.21 / scipy 1.7.

One caveat concerns the Wilcoxon signed-rank test. scipy has changed the
default rule for switching between the exact null distribution and the normal
approximation, so p-values can differ between scipy versions whenever a paired
difference is exactly zero. `_wilcoxon_method` in
`src/evaluation/statistics.py` fixes the rule (normal approximation if there
are zero differences or ties, exact otherwise), which reproduces the published
p-values on current scipy.

### Supplementary material

`build_supplementary_tables.py` reads only the artifacts under `outputs/` and
writes LaTeX fragments to `outputs/supplementary_tables/`; it never re-runs an
experiment. It ends with a self-check against the F1 scores of Table 5.

| Section | Content | Files |
|---|---|---|
| S1 | Selected hyper-parameters | `s1_selected_hp.tex` |
| S2 | All-pairs paired tests | `s2_paired_tests.tex` |
| S3 | Per-seed SD, calibration, per-level classification | `s3_per_seed_sd.tex`, `s3b_calibration.tex`, `s3c_classification_levels.tex` |
| S4 | Classification detail, per-regime class distribution | `s4_classification.tex`, `s4b_class_distribution.tex` |
| S5 | Reliability diagrams | `{email,insects}_reliability_*.png` |
| S6 | Weight diagnostics and safeguards | `s6_weight_diagnostics.tex`, `s6b_safeguards.tex` |
| S7 | sigma_cd sweep | `s7_qcd_sweep.tex` |
| S8 | Gradient-noise analysis | `s8_gradient_noise.tex` |
| S9 | Compute cost and dimension sweep | `s9_compute_cost.tex`, `s9b_dim_sweep.tex` |

## Experimental protocol

These conventions matter when reproducing the numbers, and the code enforces
all of them.

- **Selection and evaluation are separate.** Hyper-parameters are chosen on
  selection seeds `[1000, 1001, 1002]` over the selection window only; results
  are reported on evaluation seeds `[0..9]` over a disjoint reporting window.
  The selection window contains at least one concept switch, since selecting
  on a stationary stretch alone biases the learning rate downwards.
- **No leakage.** The PCA for elist and the standardization for INSECTS are fit
  on samples preceding the reporting window.
- **Straddling blocks are dropped.** A test block whose window crosses a
  concept switch is excluded from the aggregates.
- **Aggregation order.** Metrics are computed per test block, averaged over
  blocks within a seed, then averaged across seeds. Paired tests pair by seed,
  and multiple comparisons are Holm-corrected.
- **Grid boundaries.** Selecting a point at the edge of a grid raises a
  warning. The paper extended any such axis by one step and re-selected; the
  comments in `configs/*.yaml` record this.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

These cover the protocol (leakage, straddling blocks, grid-boundary
detection), the statistics (paired tests, Holm correction) and filter
numerics (low-rank correction, weight accumulation). Parts of
`tests/test_email_protocol.py` and `tests/test_no_leakage.py` need
`data/email_data.arff` and are skipped when it is absent; with the data in
place all 86 tests run.

## Citation and license

Please cite the paper above. The datasets carry their own citation
requirements, described in [data/README.md](data/README.md). The code is
released under the MIT License (see `LICENSE`).
