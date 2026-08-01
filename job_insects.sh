#!/bin/bash
#
# WSPF 実験ジョブ — INSECTS (abrupt, balanced) 多クラス分類(新 src 構成)
#   実世界ストリーム。33 特徴 + 6 クラス、既知の急激な概念スイッチ。
#   d=1286、約 3,300 ステップと重いため別ジョブに分離。
#
# 実行順(Oracle/勾配ノイズ解析は回帰専用のため無し):
#   1. grid_search        選択区間(先頭1250ステップ, 2スイッチ含む)でHP選択
#   2. run_main           主結果(N=main, 評価10シード)
#   3. run_n_sweep        R2-2: N∈{25..400}, N=100のHP固定
#   4. run_matched        R1-7: 共有HPの3×3比較
#   5. run_qcd_sweep      R1-8: σ_cd スイープ + ρ/clip
#   6. benchmark_compute  R1-11/R2-1: 実行時間・スケーリング
#   7. summarize_results  表・選択HP表の集約

set -euo pipefail

# qsub したディレクトリ(=プロジェクトルート)で実行
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

mkdir -p logs
MAIN_LOG="logs/WSPF_insects.${PBS_JOBID:-manual}.log"
exec >"$MAIN_LOG" 2>&1

# モジュール(環境に合わせて調整)
module purge || true
module load intelpython3/2022.3.1 || true

export PYTHONPATH=.
PY="python3 -u"
BENCH=insects

echo "================ [$BENCH] grid_search ================"
$PY scripts/grid_search.py       --benchmark "$BENCH"
echo "================ [$BENCH] run_main ================"
$PY scripts/run_main.py          --benchmark "$BENCH"
echo "================ [$BENCH] run_n_sweep (R2-2) ================"
$PY scripts/run_n_sweep.py       --benchmark "$BENCH"
echo "================ [$BENCH] run_matched (R1-7) ================"
$PY scripts/run_matched.py       --benchmark "$BENCH"
echo "================ [$BENCH] run_qcd_sweep (R1-8) ================"
$PY scripts/run_qcd_sweep.py     --benchmark "$BENCH"
echo "================ [$BENCH] benchmark_compute (R1-11/R2-1) ================"
$PY scripts/benchmark_compute.py --benchmark "$BENCH"
echo "================ [$BENCH] summarize_results ================"
$PY scripts/summarize_results.py --benchmark "$BENCH"

echo "INSECTS 実験が完了しました。"
