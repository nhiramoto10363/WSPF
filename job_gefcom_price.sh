#!/bin/bash
#
# WSPF 実データ実験ジョブ — GEFCom2014 Price (新 src 構成)
#   本文の大規模実データ結果。Solar と相補ペア:
#     Solar=漸進的季節ドリフト・大 d(1025)、Price=急峻スパイク・確率評価が主・
#     中庸 d(321)。job.sh / job_solar.sh とは別ジョブ。
#
# 実行順 (設計書 §6 のフェーズに対応):
#   Phase 1: screen_qcd    §5 Go/No-Go 事前スクリーニング(選択区間のみ, PF 中心)
#            → verdict_GO=false なら Phase 2 を実行せず終了(境界事例として記録)。
#              FORCE_FULL=1 を渡すと判定に関わらずフルランする。
#   Phase 2 (screen 通過時):
#     1. grid_search        選択区間で σ_obs 推定 + HP 選択(以降の全実験の前提)
#     2. run_main           主結果(N=main, 評価10シード)
#     3. run_n_sweep        R2-2: N∈{25..400}, N=100 の HP 固定
#     4. run_matched        R1-7: 共有 HP の 3×3 比較
#     5. run_qcd_sweep      R1-8: σ_cd スイープ + ρ/clip
#     6. benchmark_compute  R1-11/R2-1: 実行時間・スケーリング
#     7. summarize_results  表・選択 HP 表の集約
#   (Oracle/勾配ノイズ解析は回帰専用のため無し)

set -euo pipefail

# qsub したディレクトリ(=プロジェクトルート)で実行
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

mkdir -p logs
MAIN_LOG="logs/WSPF_gefcom_price.${PBS_JOBID:-manual}.log"
exec >"$MAIN_LOG" 2>&1

# モジュール(環境に合わせて調整)
module purge || true
module load intelpython3/2022.3.1 || true

export PYTHONPATH=.
PY="python3 -u"
BENCH=gefcom_price

# ===== Phase 1: 事前スクリーニング(Go/No-Go) =====
echo "================ [$BENCH] screen_qcd (§5 Go/No-Go) ================"
$PY scripts/screen_qcd.py --benchmark "$BENCH"

VERDICT_JSON="outputs/${BENCH}/screen_qcd/screen_verdict.json"
GO=$($PY -c "import json;print(json.load(open('$VERDICT_JSON'))['verdict_GO'])" \
     2>/dev/null || echo "False")

if [ "$GO" != "True" ] && [ "${FORCE_FULL:-0}" != "1" ]; then
    echo "screen_qcd の判定は NO-GO ($VERDICT_JSON)。"
    echo "Phase 2 (grid_search 以降) をスキップして終了します。"
    echo "判定に関わらずフルランするには FORCE_FULL=1 を指定してください。"
    echo "GEFCom-Price: スクリーニングのみで終了。"
    exit 0
fi
echo "screen_qcd 判定: GO=${GO} (FORCE_FULL=${FORCE_FULL:-0}) → Phase 2 を実行"

# ===== Phase 2: グリッドサーチ → 主結果 → 各スイープ =====
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

echo "GEFCom-Price 実験が完了しました。"
