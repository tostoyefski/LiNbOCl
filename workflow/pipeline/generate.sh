#!/usr/bin/env bash
# LiNbOCl 批量生成入口；默认生成 Li-Nb-O-Cl 四元素体系。
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
#   COMBO_SIZES           自动组合的大小列表（空白分隔，默认 4，例如：3 或 "3 4"）
#   WORKDIR               执行目录（默认当前目录，可设为 mattergen 仓库根）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-chemical_system_energy_above_hull}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-$PROJECT_ROOT/results}"
BATCH_SIZE="${BATCH_SIZE:-16}"
E_AH="${E_AH:-0.05}"
GUIDANCE="${GUIDANCE:-2.0}"
CHEMICAL_SYSTEMS="${CHEMICAL_SYSTEMS:-}"
CHEMICAL_SYSTEMS_FILE="${CHEMICAL_SYSTEMS_FILE:-}"
ELEMENTS="${ELEMENTS:-Li Nb O Cl}"
COMBO_SIZES="${COMBO_SIZES:-4}"
NUM_BATCHES="${NUM_BATCHES:-${num_batches:-1}}"
WORKDIR="${WORKDIR:-$(pwd)}"

cd "$WORKDIR"

echo "[info] MODEL_NAME=$MODEL_NAME"
echo "[info] BASE_RESULTS_DIR=$BASE_RESULTS_DIR"
echo "[info] BATCH_SIZE=$BATCH_SIZE NUM_BATCHES=$NUM_BATCHES GUIDANCE=$GUIDANCE E_AH=$E_AH"
echo "[info] WORKDIR=$WORKDIR"

# Read in the current shell so generated combinations populate SYSTEMS too.
# A read loop also works with Bash 3.2 (mapfile requires Bash 4).
SYSTEMS=()
while IFS= read -r cs; do
  [[ -n "$cs" ]] && SYSTEMS+=("$cs")
done < <(python - "$CHEMICAL_SYSTEMS" "$CHEMICAL_SYSTEMS_FILE" "$ELEMENTS" "$COMBO_SIZES" <<'PY_SYSTEMS'
import itertools
import pathlib
import re
import sys

explicit, filename, elements_text, sizes_text = sys.argv[1:]
systems = re.split(r"[,;\s]+", explicit.strip()) if explicit.strip() else []
if filename:
    for line in pathlib.Path(filename).read_text().splitlines():
        systems.extend(re.split(r"[,;\s]+", line.split("#", 1)[0].strip()))
systems = [s for s in systems if s]
if not systems:
    elements = list(dict.fromkeys(re.split(r"[,;\s]+", elements_text.strip())))
    sizes = [int(n) for n in sizes_text.split()]
    if not elements or any(not 1 <= n <= len(elements) for n in sizes):
        raise ValueError("COMBO_SIZES must be between 1 and the number of elements")
    systems = ["-".join(combo) for size in sizes for combo in itertools.combinations(elements, size)]
for system in dict.fromkeys(systems):
    print(system)
PY_SYSTEMS
)

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
