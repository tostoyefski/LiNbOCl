#!/usr/bin/env bash

# 不用 -e，避免单次失败导致全局退出；保留未定义变量 & pipefail 保护

set -u -o pipefail



ROOT="${ROOT:-results/chemical_system_energy_above_hull}"  # 可通过环境变量覆盖，便于跑子目录

LOGDIR="logs_eval"

mkdir -p "$LOGDIR"



# 限制线程，减少内存/CPU 抢占

export OMP_NUM_THREADS=1

export MKL_NUM_THREADS=1

export OPENBLAS_NUM_THREADS=1



shopt -s nullglob



find "$ROOT" -mindepth 1 -maxdepth 1 -type d | while read -r d; do
  if [[ -f "$d/metrics.json" ]]; then
    echo "[rerun] $d 已有 metrics.json，将覆盖"
    rm -f "$d/metrics.json" "$d/relaxed.extxyz"
  fi



  # 选择输入：优先 zip；否则目录内的 cif/extxyz

  STRUCTS=""

  if compgen -G "$d/*.zip" > /dev/null; then

    STRUCTS="$(ls "$d"/*.zip | head -n 1)"

  elif compgen -G "$d/*.extxyz" > /dev/null || compgen -G "$d/*.cif" > /dev/null; then

    STRUCTS="$d"

  else

    echo "[warn] $d 没找到 .zip/.cif/.extxyz，跳过"

    continue

  fi



  echo "[run] $d  <- $STRUCTS"

  LOG="$LOGDIR/$(basename "$d").log"



  # 可选：给每个目录设置超时（例如 90 分钟），避免个别卡死

  # 若系统没有 `timeout` 命令，可以把它去掉

  ( set -x; timeout 90m mattergen-evaluate "$STRUCTS" --relax=True --structure_matcher='disordered' --save_as="$d/metrics.json" --structures_output_path="$d/relaxed.extxyz") &> "$LOG"

  status=$?



  if [[ $status -ne 0 ]]; then

     echo "[fail] $(date '+%F %T') $(basename "$d") exit=$status; see $LOG" | tee -a "$LOGDIR/_failures.txt"

     # 若写出过半成品，清理以便下次干净重试

     rm -f "$d/metrics.json" "$d/relaxed.extxyz"

     continue

  fi

done
