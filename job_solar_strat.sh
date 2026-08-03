#!/bin/bash
#
# WSPF 実データ実験ジョブ — GEFCom2014 Solar 層化学習率 WSPF (Stratified WSPF)
#   WSPF-A/B の学習率を粒子スロット別の層化配置 (指数分布, 平均 η̄) にした版。
#   探索軸は増やさず eta→eta_wspf_mean (=eta/2.303) と読み替え、HP選択は
#   混合NLL (select_metric: nll)。既存 configs/outputs/gefcom は不変更。
#   対照 (fixed版) は configs/gefcom.yaml で outputs/gefcom に計算済み。
#   本ジョブは層化版 outputs/gefcom_strat を生成し §8.2 のアブレーション比較に用いる。
#
# 実行順 (フルスイート, job_solar.sh と同一; Oracle/勾配ノイズ解析は回帰専用のため無し):
#   1. grid_search        選択区間の混合NLL で HP 選択 (zone 別 σ_obs 推定含む)
#   2. run_main           主結果 (N=100, 評価10シード)。層化診断は diagnostics.npz へ
#   3. run_n_sweep        R2-2: N∈{25..400}, N=100 の HP 固定
#   4. run_matched        R1-7: 共有 HP の 3×3 比較
#   5. run_qcd_sweep      R1-8: σ_cd スイープ + ρ/clip
#   6. benchmark_compute  R1-11/R2-1: 実行時間・スケーリング
#   7. summarize_results  表・選択 HP 表の集約
#
# オプション環境変数:
#   THIN_METHODS=1  WSPF-A を除外した先行確認 config を生成 (候補を大幅削減)。
#   NCPUS / WSPF_NUM_WORKERS  グリッド並列数。

set -euo pipefail

# qsub したディレクトリ(=プロジェクトルート)で実行
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

mkdir -p logs
MAIN_LOG="logs/WSPF_solar_strat.${PBS_JOBID:-manual}.log"
exec >"$MAIN_LOG" 2>&1

module purge || true
module load intelpython3/2022.3.1 || true

export PYTHONPATH=.
if [ -n "${NCPUS:-}" ] && [ -z "${WSPF_NUM_WORKERS:-}" ]; then
    export WSPF_NUM_WORKERS="$NCPUS"
fi
PY="python3 -u"

BENCH=gefcom_strat
# THIN_METHODS=1: WSPF-A を除いた config を一時生成 (層化スキーム・軸は維持)
if [ "${THIN_METHODS:-0}" = "1" ]; then
    echo "[THIN_METHODS] WSPF-A を除外した先行確認 config を生成"
    $PY - <<'PYEOF'
import yaml
cfg = yaml.safe_load(open("configs/gefcom_strat.yaml"))
cfg["methods"] = [m for m in cfg["methods"] if m != "WSPF-A"]
cfg["output_dir"] = "outputs/gefcom_strat_thin"
with open("configs/gefcom_strat_thin.yaml", "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print("生成: configs/gefcom_strat_thin.yaml  methods=", cfg["methods"])
PYEOF
    BENCH=gefcom_strat_thin
fi

echo "================ [$BENCH] grid_search (select_metric=nll, 層化) ================"
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

echo "GEFCom Solar 層化学習率 WSPF 実験が完了しました。"
echo "対照 (fixed) は outputs/gefcom、層化版は outputs/${BENCH} を参照。"
