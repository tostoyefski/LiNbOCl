#!/usr/bin/env bash
# 独立拷贝的 eval_all.sh，增加 WORKDIR/ROOT 可配置，避免修改原仓库。
# 用法：ROOT=/path/to/results WORKDIR=/path/to/mattergen bash eval_all.sh
# 一键全流程的分段目录可设置 RECURSIVE=1 递归查找待评估目录。

set -u -o pipefail

WORKDIR="${WORKDIR:-$(pwd)}"
ROOT="${ROOT:-results/chemical_system_energy_above_hull}"
LOGDIR="${LOGDIR:-logs_eval}"
RECURSIVE="${RECURSIVE:-0}"

cd "$WORKDIR"
mkdir -p "$LOGDIR"

# 限制线程，减少内存/CPU 抢占
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

shopt -s nullglob

if [[ "$RECURSIVE" == "1" || "$RECURSIVE" == "true" || "$RECURSIVE" == "yes" ]]; then
  FIND_ARGS=("$ROOT" -mindepth 1 -type d)
else
  FIND_ARGS=("$ROOT" -mindepth 1 -maxdepth 1 -type d)
fi

find "${FIND_ARGS[@]}" | while read -r d; do
  if [[ -f "$d/metrics.json" ]]; then
    echo "[rerun] $d 已有 metrics.json，将覆盖"
    rm -f "$d/metrics.json" "$d/relaxed.extxyz"
  fi

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
  rel="${d#"$ROOT"/}"
  safe="${rel//\//__}"
  safe="${safe// /_}"
  LOG="$LOGDIR/${safe}.log"

  ( set -x; timeout 90m mattergen-evaluate "$STRUCTS" --relax=True --structure_matcher='disordered' --save_as="$d/metrics.json" --structures_output_path="$d/relaxed.extxyz") &> "$LOG"
  status=$?

  if [[ $status -ne 0 ]]; then
     echo "[fail] $(date '+%F %T') $(basename "$d") exit=$status; see $LOG" | tee -a "$LOGDIR/_failures.txt"
     rm -f "$d/metrics.json" "$d/relaxed.extxyz"
     continue
  fi
done
