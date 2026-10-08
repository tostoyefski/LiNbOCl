#!/usr/bin/env bash

set -euo pipefail
set -x


# ===== 用户可调参数 =====

MODEL_NAME="${MODEL_NAME:-chemical_system_energy_above_hull}"
export MODEL_NAME

BASE_RESULTS_DIR="results/${MODEL_NAME}"
# 单次 16，循环多次补足总数
BATCH_SIZE=16
NUM_BATCHES="${NUM_BATCHES:-${num_batches:-1}}"
N_REPEATS=4   # 16 * 4 = 64
E_AH=0.05
GUIDANCE=2.0
LI_ELEM="Li"   # 确保所有 chemical_system 含 Li

# 固定只跑 Li/Ge/Cl/O，不需要其他元素与混合
HALIDE_MIXES=(Cl)
M3_CATIONS=()
M4_CATIONS=(Ge)


# ===== 跑函数 =====

run_case() {
  local chem_sys="$1"         # 例如：Li-Y-Cl-Br-O
  local run_dir="$2"          # 为每次迭代单独目录，避免覆盖
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


# ===== 主循环 =====

main() {
  local chem_sys

  # 固定单体系，重复多轮以增加样本量
  chem_sys="${LI_ELEM}-${M4_CATIONS[0]}-${HALIDE_MIXES[0]}-O"
  for ((i=1; i<=N_REPEATS; i++)); do
    echo ">>> Repeat ${i}/${N_REPEATS}"
    run_case "$chem_sys" "${BASE_RESULTS_DIR}/${chem_sys}/rep_${i}"
  done
}


main
