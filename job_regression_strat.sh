#!/bin/bash
#
# WSPF 実験ジョブ — Regression 層化学習率 WSPF (Stratified Learning-Rate WSPF)
#   WSPF-A/B の学習率を粒子スロット別の層化配置 (指数分布, 平均 η̄) にした版の
#   回帰問題推定。探索軸は増やさず eta→eta_wspf_mean (=eta/2.303) と読み替え、
#   HP選択は混合NLL (select_metric: nll)。既存 configs/outputs は不変更。
#
# 対照 (fixed版 = 従来の共通スカラー eta) は configs/regression.yaml で
# 既に outputs/regression に計算済み。本ジョブは層化版 outputs/regression_strat
# を生成し、両者を比較する (§8.2 のアブレーション)。
#
# 実行内容:
#   1. grid_search        選択区間の混合NLL (select_metric: nll) で HP 選択
#   2. run_main           主結果 (N=100, 評価10シード)。層化診断は per-run の
#                         diagnostics.npz に自動保存 (eta_weighted_mean 等, §9)
#   3. summarize_results  表・選択 HP 表の集約
#   (Oracle/勾配ノイズ解析は本実験では走らせない)
#
# ── 計算規模 ──────────────────────────────────────────────────────
#   grid_search の候補数 = 3,339 (× 選択3シード = 10,017 評価)。
#   内訳: WSPF-A 1,764 が支配的 (beta 軸 × sigma_obs 軸)。
#   先行確認は THIN_METHODS=1 (WSPF-A 除外 → 1,575 候補) を推奨。
# ────────────────────────────────────────────────────────────────
#
# オプション環境変数:
#   THIN_METHODS=1  WSPF-A を除外して先行確認する (候補 3,339→1,575)。
#   NCPUS / WSPF_NUM_WORKERS  グリッド並列数。

set -euo pipefail

# qsub したディレクトリ(=プロジェクトルート)で実行
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

mkdir -p logs
MAIN_LOG="logs/WSPF_regression_strat.${PBS_JOBID:-manual}.log"
exec >"$MAIN_LOG" 2>&1

# モジュール(環境に合わせて調整)
module purge || true
module load intelpython3/2022.3.1 || true

export PYTHONPATH=.
if [ -n "${NCPUS:-}" ] && [ -z "${WSPF_NUM_WORKERS:-}" ]; then
    export WSPF_NUM_WORKERS="$NCPUS"
fi
PY="python3 -u"

BENCH=regression_strat
# THIN_METHODS=1: WSPF-A を除いた config を一時生成 (層化スキーム・軸は維持)
if [ "${THIN_METHODS:-0}" = "1" ]; then
    echo "[THIN_METHODS] WSPF-A を除外した先行確認 config を生成"
    $PY - <<'PYEOF'
import yaml
cfg = yaml.safe_load(open("configs/regression_strat.yaml"))
cfg["methods"] = [m for m in cfg["methods"] if m != "WSPF-A"]
cfg["output_dir"] = "outputs/regression_strat_thin"
with open("configs/regression_strat_thin.yaml", "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print("生成: configs/regression_strat_thin.yaml  methods=", cfg["methods"])
PYEOF
    BENCH=regression_strat_thin
fi

echo "================ [$BENCH] grid_search (select_metric=nll, 層化) ================"
$PY scripts/grid_search.py       --benchmark "$BENCH"
echo "================ [$BENCH] run_main (N=100, eval 10 seeds) ================"
$PY scripts/run_main.py          --benchmark "$BENCH"
echo "================ [$BENCH] summarize_results ================"
$PY scripts/summarize_results.py --benchmark "$BENCH"

echo "Regression 層化学習率 WSPF 実験が完了しました。"
echo "対照 (fixed) は outputs/regression、層化版は outputs/${BENCH#regression_strat_thin} 系を参照。"
