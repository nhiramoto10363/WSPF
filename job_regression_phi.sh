#!/bin/bash
#
# WSPF 実験ジョブ — Regression φ_t 拡張 R1 (regime 主実験)
#   観測ノイズ σ*_t をレジーム同期で切替 (0.3 / 0.7) し、σ_obs を状態空間に
#   組み込む適応手法 (-N) と固定 σ 手法を 2×2 (補正あり/なし × φ適応あり/なし)
#   で比較する (設計書 §5.3)。既存 regression.yaml / outputs/regression には
#   一切影響しない。
#
# 実行内容 (README_phi §5.4 の R1):
#   1. grid_search        選択区間の予測 NLL (select_metric: nll) で HP 選択
#   2. run_main           主結果 (N=100, 評価10シード)
#   3. summarize_results  表・選択 HP 表の集約
#   (φ 実験では Oracle / 勾配ノイズ解析は走らせない: oracle=false)
#
# ── 計算規模の警告 (重要) ──────────────────────────────────────────
#   grid_search の候補数 = 26,775 (× 選択3シード = 80,325 評価)。
#   内訳: WSPF-A / WSPF-A-N が各 8,820 で支配的 (sigma_obs / tau_phi 軸で従来の
#   約5倍)。フルは高コスト。まず軸を間引いた先行確認を強く推奨:
#     - 手法を絞る:   THIN_METHODS=1  (SGD PF WSPF-B PF-N WSPF-B-N のみ; A系除外)
#     - あるいは configs/regression_phi.yaml の grid を手で間引く
#       (例: sigma_obs/tau_phi を各3点、eta を5点に減らす)。
# ────────────────────────────────────────────────────────────────
#
# オプション環境変数:
#   RUN_R0=1        R1 の前に R0 (constant 回帰チェック, regression_phi_const)
#                   を実行する (適応がコストにならないことの sanity check)。
#   THIN_METHODS=1  WSPF-A / WSPF-A-N を除外して先行確認する (候補 26,775→9,135)。
#   NCPUS / WSPF_NUM_WORKERS  グリッド並列数。

set -euo pipefail

# qsub したディレクトリ(=プロジェクトルート)で実行
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

mkdir -p logs
MAIN_LOG="logs/WSPF_regression_phi.${PBS_JOBID:-manual}.log"
exec >"$MAIN_LOG" 2>&1

# モジュール(環境に合わせて調整)
module purge || true
module load intelpython3/2022.3.1 || true

export PYTHONPATH=.
# グリッド並列数: PBS の NCPUS があれば流用 (未設定なら _common の既定)。
if [ -n "${NCPUS:-}" ] && [ -z "${WSPF_NUM_WORKERS:-}" ]; then
    export WSPF_NUM_WORKERS="$NCPUS"
fi
PY="python3 -u"

# THIN_METHODS=1 のとき、A 系を除いた config を一時生成して先行確認する。
BENCH=regression_phi
if [ "${THIN_METHODS:-0}" = "1" ]; then
    echo "[THIN_METHODS] WSPF-A / WSPF-A-N を除外した先行確認 config を生成"
    $PY - <<'PYEOF'
import yaml, os
cfg = yaml.safe_load(open("configs/regression_phi.yaml"))
cfg["methods"] = [m for m in cfg["methods"] if m not in ("WSPF-A", "WSPF-A-N")]
cfg["output_dir"] = "outputs/regression_phi_thin"
os.makedirs("configs", exist_ok=True)
with open("configs/regression_phi_thin.yaml", "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print("生成: configs/regression_phi_thin.yaml  methods=", cfg["methods"])
PYEOF
    BENCH=regression_phi_thin
fi

run_menu () {
    local B="$1"
    echo "================ [$B] grid_search (select_metric=nll) ================"
    $PY scripts/grid_search.py      --benchmark "$B"
    echo "================ [$B] run_main (N=100, eval 10 seeds) ================"
    $PY scripts/run_main.py         --benchmark "$B"
    echo "================ [$B] summarize_results ================"
    $PY scripts/summarize_results.py --benchmark "$B"
}

# ===== R0 (任意): constant 回帰チェック =====
if [ "${RUN_R0:-0}" = "1" ]; then
    echo "################ R0: constant 回帰チェック (regression_phi_const) ################"
    run_menu regression_phi_const
fi

# ===== R1: regime 主実験 =====
echo "################ R1: regime 主実験 ($BENCH) ################"
run_menu "$BENCH"

echo "Regression φ_t R1 実験が完了しました。"
