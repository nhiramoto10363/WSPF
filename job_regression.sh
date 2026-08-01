#!/bin/bash
#
# WSPF 実験ジョブ — Regression (本文, 合成; 新 src 構成)
#   本文の主結果。既知の急激な反復ドリフト。Oracle(R1-5)と勾配ノイズ解析
#   (R1-6)は回帰専用のためここに含める。
#   GEFCom は job_solar.sh、Email は job_email.sh 側で実行する。
#
# 各ベンチマークの実行順:
#   1. grid_search        選択区間でHP選択(以降の全実験の前提)
#   2. run_main           主結果(N=main, 評価10シード)
#   3. run_n_sweep        R2-2: N∈{25..400}, N=100のHP固定
#   4. run_matched        R1-7: 共有HPの3×3比較
#   5. run_qcd_sweep      R1-8: σ_cd スイープ + ρ/clip
#   6. benchmark_compute  R1-11/R2-1: 実行時間・スケーリング
#   7. summarize_results  表・選択HP表の集約
#   Regression のみ:
#     run_oracle              R1-5: 厳密補正 vs 近似
#     analyze_gradient_noise  R1-6: 勾配ノイズのガウス性/異方性/B依存

set -euo pipefail

# qsub したディレクトリ(=プロジェクトルート)で実行
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

mkdir -p logs
MAIN_LOG="logs/WSPF_regression.${PBS_JOBID:-manual}.log"
exec >"$MAIN_LOG" 2>&1

# モジュール(環境に合わせて調整)
module purge || true
module load intelpython3/2022.3.1 || true

export PYTHONPATH=.
PY="python3 -u"

run_benchmark () {
    local BENCH="$1"
    echo "================ [$BENCH] grid_search ================"
    $PY scripts/grid_search.py      --benchmark "$BENCH"
    echo "================ [$BENCH] run_main ================"
    $PY scripts/run_main.py         --benchmark "$BENCH"
    echo "================ [$BENCH] run_n_sweep (R2-2) ================"
    $PY scripts/run_n_sweep.py      --benchmark "$BENCH"
    echo "================ [$BENCH] run_matched (R1-7) ================"
    $PY scripts/run_matched.py      --benchmark "$BENCH"
    echo "================ [$BENCH] run_qcd_sweep (R1-8) ================"
    $PY scripts/run_qcd_sweep.py    --benchmark "$BENCH"
    echo "================ [$BENCH] benchmark_compute (R1-11/R2-1) ================"
    $PY scripts/benchmark_compute.py --benchmark "$BENCH"
    echo "================ [$BENCH] summarize_results ================"
    $PY scripts/summarize_results.py --benchmark "$BENCH"
}

# ===== Regression (本文, 合成; Oracle と 勾配ノイズ解析を含む) =====
run_benchmark regression
echo "================ [regression] run_oracle (R1-5) ================"
$PY scripts/run_oracle.py           --benchmark regression
echo "================ [regression] analyze_gradient_noise (R1-6) ================"
$PY scripts/analyze_gradient_noise.py --benchmark regression

echo "Regression 実験が完了しました。"
