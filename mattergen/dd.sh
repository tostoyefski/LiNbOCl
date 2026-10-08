#!/usr/bin/env bash

set -euo pipefail
set -x


# ===== 用户可调参数 =====

MODEL_NAME="${MODEL_NAME:-chemical_system_energy_above_hull}"
export MODEL_NAME

BASE_RESULTS_DIR="results/${MODEL_NAME}"
BATCH_SIZE=16
NUM_BATCHES="${NUM_BATCHES:-${num_batches:-1}}"
E_AH=0.05
GUIDANCE=2.0
LI_ELEM="Li"   # 确保所有 chemical_system 含 Li

# 是否包含 F：若需要把 F 纳入混合卤化，将此处改为 1
INCLUDE_F=0

# 是否生成纯卤化物与/或卤氧化物
GENERATE_HALIDES=1
GENERATE_HALOXIDES=1


# ===== 候选列表 =====

# M3+：优先元素放最前，随后 La–Lu
M3_CATIONS=(Y Sc In Al Ga La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu)

# M4+
M4_CATIONS=(Zr Hf Ti Sn Ge Si)


# 卤素集合（默认 Cl/Br/I）
HALIDES=(Cl Br I)
if [[ "$INCLUDE_F" -eq 1 ]]; then
  HALIDES=(F Cl Br I)
fi


# ===== 生成 1~3 元卤素组合 =====

make_halide_singles() {
  local -n arr=$1
  for el in "${arr[@]}"; do
    echo "$el"
  done
}

make_halide_pairs() {
  local -n arr=$1
  local n=${#arr[@]}
  for ((i=0;i<n;i++)); do
    for ((j=i+1;j<n;j++)); do
      echo "${arr[i]}-${arr[j]}"
    done
  done
}

make_halide_triples() {
  local -n arr=$1
  local n=${#arr[@]}
  for ((i=0;i<n;i++)); do
    for ((j=i+1;j<n;j++)); do
      for ((k=j+1;k<n;k++)); do
        echo "${arr[i]}-${arr[j]}-${arr[k]}"
      done
    done
  done
}

readarray -t HALIDE_MIXES < <(
  make_halide_singles HALIDES
  make_halide_pairs HALIDES
  make_halide_triples HALIDES
)


# ===== 跑函数 =====

run_case() {
  local chem_sys="$1"         # 例如：Li-Y-Cl-Br-O
  local run_dir="${BASE_RESULTS_DIR}/${chem_sys}"   # 避免覆盖
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

  # M3+ 空间
  for m in "${M3_CATIONS[@]}"; do
    for mix in "${HALIDE_MIXES[@]}"; do
      if [[ "$GENERATE_HALIDES" -eq 1 ]]; then
        chem_sys="${LI_ELEM}-${m}-${mix}"
        run_case "$chem_sys"
      fi
      if [[ "$GENERATE_HALOXIDES" -eq 1 ]]; then
        chem_sys="${LI_ELEM}-${m}-${mix}-O"
        run_case "$chem_sys"
      fi
    done
  done

  # M4+ 空间
  for m in "${M4_CATIONS[@]}"; do
    for mix in "${HALIDE_MIXES[@]}"; do
      if [[ "$GENERATE_HALIDES" -eq 1 ]]; then
        chem_sys="${LI_ELEM}-${m}-${mix}"
        run_case "$chem_sys"
      fi
      if [[ "$GENERATE_HALOXIDES" -eq 1 ]]; then
        chem_sys="${LI_ELEM}-${m}-${mix}-O"
        run_case "$chem_sys"
      fi
    done
  done
}


main
