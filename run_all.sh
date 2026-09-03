#!/usr/bin/env bash
#
# Run the experiments.
#
#   ./run_all.sh                 all three benchmarks, then tables and figures
#   ./run_all.sh regression      one benchmark
#   ./run_all.sh email insects
#   ./run_all.sh tables          rebuild tables and figures from existing outputs/
#
# Environment:
#   PYTHON            interpreter to use (default: python3)
#   WSPF_NUM_WORKERS  parallelism for grid search and seed runs (default: cores)
#   SKIP_GRID=1       reuse outputs/<bench>/selected_params.json instead of
#                     re-running the grid search
#
# Everything is written to outputs/. Compare against results/ to check whether
# a run reproduced the published numbers.
#
# The grid search dominates the runtime. On the authors' machine (one shared
# compute node, 32 worker processes) the regression benchmark took about half
# an hour end to end, while INSECTS took about fifteen hours (3,300 steps at
# d=1286).

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH=.
PY="${PYTHON:-python3} -u"
SKIP_GRID="${SKIP_GRID:-0}"

step () { echo; echo "================ $* ================"; }

run_benchmark () {
    local BENCH="$1"

    if [ "$SKIP_GRID" = "1" ]; then
        step "[$BENCH] skipping grid_search (SKIP_GRID=1)"
        if [ ! -f "outputs/$BENCH/selected_params.json" ]; then
            echo "error: outputs/$BENCH/selected_params.json is missing."
            echo "       Copy results/$BENCH/selected_params.json, or unset"
            echo "       SKIP_GRID to run the grid search."
            exit 1
        fi
    else
        step "[$BENCH] grid_search  (selects hyper-parameters; everything else depends on it)"
        $PY scripts/grid_search.py --benchmark "$BENCH"
    fi

    if [ "$BENCH" != "regression" ]; then
        step "[$BENCH] report_class_distribution"
        $PY scripts/report_class_distribution.py --benchmark "$BENCH"
    fi

    step "[$BENCH] run_main  (N=100, 10 evaluation seeds)"
    $PY scripts/run_main.py --benchmark "$BENCH"
    step "[$BENCH] run_n_sweep"
    $PY scripts/run_n_sweep.py --benchmark "$BENCH"
    step "[$BENCH] run_matched"
    $PY scripts/run_matched.py --benchmark "$BENCH"
    step "[$BENCH] run_qcd_sweep"
    $PY scripts/run_qcd_sweep.py --benchmark "$BENCH"
    step "[$BENCH] benchmark_compute"
    $PY scripts/benchmark_compute.py --benchmark "$BENCH"
    step "[$BENCH] analyze_applicability"
    $PY scripts/analyze_applicability.py --benchmark "$BENCH"
    step "[$BENCH] calibration_report"
    $PY scripts/calibration_report.py --benchmark "$BENCH"

    if [ "$BENCH" = "regression" ]; then
        step "[regression] run_oracle"
        $PY scripts/run_oracle.py --benchmark regression
        step "[regression] analyze_gradient_noise"
        $PY scripts/analyze_gradient_noise.py --benchmark regression
        step "[regression] run_batch_size_sweep"
        $PY scripts/run_batch_size_sweep.py --benchmark regression
    fi

    # Last, so that it picks up the rows written by the steps above.
    step "[$BENCH] summarize_results"
    $PY scripts/summarize_results.py --benchmark "$BENCH"
}

build_tables_and_figures () {
    step "[all] analyze_applicability  (across benchmarks)"
    $PY scripts/analyze_applicability.py --benchmark regression email insects \
        --out outputs/applicability

    step "[figures]"
    $PY scripts/plot_snapshot_func.py
    $PY scripts/plot_mse_timeseries.py
    $PY scripts/plot_ess_sweep.py
    $PY scripts/plot_recovery.py --benchmark regression
    $PY scripts/plot_recovery_classification.py --benchmark email --metric f1

    step "[tables] supplementary LaTeX fragments"
    $PY scripts/build_supplementary_tables.py
}

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=(regression email insects tables)
fi

for t in "${TARGETS[@]}"; do
    case "$t" in
        regression|email|insects) run_benchmark "$t" ;;
        tables|figures)           build_tables_and_figures ;;
        *) echo "error: unknown target '$t' (regression / email / insects / tables)"
           exit 1 ;;
    esac
done

echo
echo "Done. Artifacts are under outputs/; compare against results/."
