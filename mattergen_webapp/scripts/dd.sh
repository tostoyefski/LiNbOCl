#!/usr/bin/env bash
# 自定义元素组合的批量生成脚本（独立于原仓库 dd.sh）
# 关键参数（可通过环境变量覆盖）：
#   MODEL_NAME            预训练模型名（传给 mattergen-generate）
#   BASE_RESULTS_DIR      结果输出根目录（可填绝对路径）
#   BATCH_SIZE            mattergen-generate 批大小
#   NUM_BATCHES           mattergen-generate 连续采样批次数
#   E_AH                  energy_above_hull 条件
#   GUIDANCE              扩散引导因子
#   CHEMICAL_SYSTEMS      逗号/空白分隔的化学系统列表（如：Li-Fe-O,Li-Fe-F）
#   CHEMICAL_SYSTEMS_FILE 文件形式的一行一个化学系统列表（# 开头行忽略）
#   ELEMENTS              若未显式给定化学系统，用此元素集合自动组合
#   COMBO_SIZES           自动组合的大小列表（空白分隔，默认 3，例如：3 或 "3 4"）
#   WORKDIR               执行目录（默认当前目录，可设为 mattergen 仓库根）

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-chemical_system_energy_above_hull}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-results/${MODEL_NAME}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
E_AH="${E_AH:-0.05}"
GUIDANCE="${GUIDANCE:-2.0}"
CHEMICAL_SYSTEMS="${CHEMICAL_SYSTEMS:-}"
CHEMICAL_SYSTEMS_FILE="${CHEMICAL_SYSTEMS_FILE:-}"
ELEMENTS="${ELEMENTS:-Li Y Cl Br O}"
COMBO_SIZES="${COMBO_SIZES:-3}"
NUM_BATCHES="${NUM_BATCHES:-${num_batches:-1}}"
WORKDIR="${WORKDIR:-$(pwd)}"

cd "$WORKDIR"

echo "[info] MODEL_NAME=$MODEL_NAME"
echo "[info] BASE_RESULTS_DIR=$BASE_RESULTS_DIR"
echo "[info] BATCH_SIZE=$BATCH_SIZE NUM_BATCHES=$NUM_BATCHES GUIDANCE=$GUIDANCE E_AH=$E_AH"
echo "[info] WORKDIR=$WORKDIR"

mapfile -t SYSTEMS < <(
  # 优先使用显式列表
  if [[ -n "$CHEMICAL_SYSTEMS" ]]; then
    tr ',;' '\n' <<<"$CHEMICAL_SYSTEMS"
  fi

  # 从文件读取
  if [[ -n "$CHEMICAL_SYSTEMS_FILE" && -f "$CHEMICAL_SYSTEMS_FILE" ]]; then
    sed 's/#.*$//' "$CHEMICAL_SYSTEMS_FILE" | sed '/^[[:space:]]*$/d'
  fi
)

if [[ ${#SYSTEMS[@]} -eq 0 ]]; then
  # 自动组合元素
  python - "$ELEMENTS" "$COMBO_SIZES" <<'PY'
import sys, itertools
elements = sys.argv[1].replace(",", " ").split()
sizes = [int(x) for x in sys.argv[2].split()]
seen = set()
for k in sizes:
    for combo in itertools.combinations(elements, k):
        cs = "-".join(combo)
        if cs not in seen:
            seen.add(cs)
            print(cs)
PY
fi

if [[ ${#SYSTEMS[@]} -eq 0 ]]; then
  echo "[error] 没有可用的化学系统（请设置 CHEMICAL_SYSTEMS 或 ELEMENTS/COMBO_SIZES）"
  exit 1
fi

echo "[info] 将运行 mattergen-generate 于 ${#SYSTEMS[@]} 个化学系统"

run_case() {
  local chem_sys="$1"
  local run_dir="${BASE_RESULTS_DIR%/}/${chem_sys}"
  mkdir -p "$run_dir"
  local props
  props=$(printf '{"energy_above_hull": %.2f, "chemical_system": "%s"}' "$E_AH" "$chem_sys")
  echo ">>> Running mattergen-generate for ${chem_sys}"
  export RESULTS_PATH="$run_dir"
  mattergen-generate \
    "$RESULTS_PATH" \
    --pretrained_name="$MODEL_NAME" \
    --batch_size="$BATCH_SIZE" \
    --num_batches="$NUM_BATCHES" \
    --properties_to_condition_on="$props" \
    --diffusion_guidance_factor="$GUIDANCE"
}

for cs in "${SYSTEMS[@]}"; do
  run_case "$cs"
done

echo "[done] 全部完成。输出位于 ${BASE_RESULTS_DIR}"
