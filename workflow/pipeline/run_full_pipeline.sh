#!/usr/bin/env bash
# Unified command-line pipeline for local, HPC and container runs.
# It mirrors the webapp "one-click full pipeline":
# segmented generation first, then unified eval/screen/top-K export.

set -euo pipefail

WORKFLOW_ROOT="${WORKFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="$(cd "$WORKFLOW_ROOT/.." && pwd)"
PIPELINE_DIR="$WORKFLOW_ROOT/pipeline"
MATTERGEN_ROOT="${MATTERGEN_ROOT:-$PROJECT_ROOT/mattergen}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$PROJECT_ROOT/_runtime}"

MODEL_NAME="${MODEL_NAME:-chemical_system_energy_above_hull}"
CHEMICAL_SYSTEMS="${CHEMICAL_SYSTEMS:-Li-Nb-O-Cl}"
CHEMICAL_SYSTEMS_FILE="${CHEMICAL_SYSTEMS_FILE:-}"
ELEMENTS="${ELEMENTS:-}"
COMBO_SIZES="${COMBO_SIZES:-4}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES_PER_SEGMENT="${NUM_BATCHES_PER_SEGMENT:-20}"
SEGMENTS="${SEGMENTS:-10}"
E_AH="${E_AH:-0.05}"
GUIDANCE="${GUIDANCE:-2.0}"

SCREEN_TOPK="${SCREEN_TOPK:-300}"
TOPK="${TOPK:-300}"
GPU_WORKERS="${GPU_WORKERS:-1}"
R_CUT="${R_CUT:-3.0}"
SUPERCELL="${SUPERCELL:-2 2 2}"
LIGHT_OXY="${LIGHT_OXY:-0.05 0.35}"
REQUIRE_CHARGE_BALANCE="${REQUIRE_CHARGE_BALANCE:-1}"
USE_SMACT="${USE_SMACT:-1}"
REQUIRED_ELEMENTS="${REQUIRED_ELEMENTS:-Li Nb O Cl}"
ALLOWED_ELEMENTS="${ALLOWED_ELEMENTS-Li Nb O Cl}"
FILTER_LIGHT_OXY="${FILTER_LIGHT_OXY:-1}"
SELECTION_MODE="${SELECTION_MODE:-diverse}"
VOLTAGE_THRESHOLD="${VOLTAGE_THRESHOLD:-0.001}"
TARGET_VOLTAGE="${TARGET_VOLTAGE:-}"
MIN_VOLTAGE_WINDOW="${MIN_VOLTAGE_WINDOW:-0}"
DRY_RUN="${DRY_RUN:-0}"
MATTERSIM_CHECKPOINT="${MATTERSIM_CHECKPOINT:-MatterSim-v1.0.0-1M.pth}"
RELAX_FMAX="${RELAX_FMAX:-0.05}"
RELAX_STEPS="${RELAX_STEPS:-500}"

export TMPDIR="$RUNTIME_ROOT/tmp"
export TMP="$RUNTIME_ROOT/tmp"
export TEMP="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export HF_HOME="$RUNTIME_ROOT/huggingface"
export HF_HUB_CACHE="$RUNTIME_ROOT/huggingface/hub"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/huggingface/transformers"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export CUDA_CACHE_PATH="$RUNTIME_ROOT/cuda"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export UV_CACHE_DIR="$RUNTIME_ROOT/uv"
export PIP_CACHE_DIR="$RUNTIME_ROOT/pip"
export NUMBA_CACHE_DIR="$RUNTIME_ROOT/numba"
export TRITON_CACHE_DIR="$RUNTIME_ROOT/triton"
export PYTHONPYCACHEPREFIX="$RUNTIME_ROOT/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH" \
  "$MPLCONFIGDIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$RESULTS_ROOT"

cd "$MATTERGEN_ROOT"

echo "[info] MATTERGEN_ROOT=$MATTERGEN_ROOT"
echo "[info] WORKFLOW_ROOT=$WORKFLOW_ROOT"
echo "[info] RESULTS_ROOT=$RESULTS_ROOT"
echo "[info] RUNTIME_ROOT=$RUNTIME_ROOT"
echo "[info] SEGMENTS=$SEGMENTS NUM_BATCHES_PER_SEGMENT=$NUM_BATCHES_PER_SEGMENT BATCH_SIZE=$BATCH_SIZE"

SEGMENTS_ROOT="$RESULTS_ROOT/_segments"
MANIFEST="$RESULTS_ROOT/full_pipeline_segments.txt"
: > "$MANIFEST"

for i in $(seq 1 "$SEGMENTS"); do
  seg_name=$(printf "batch%03d" "$i")
  seg_dir="$SEGMENTS_ROOT/$seg_name"
  echo "==== Generate segment $i/$SEGMENTS -> $seg_dir ===="
  printf '%s\n' "$seg_dir" >> "$MANIFEST"
  MODEL_NAME="$MODEL_NAME" \
    BASE_RESULTS_DIR="$seg_dir" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_BATCHES="$NUM_BATCHES_PER_SEGMENT" \
    E_AH="$E_AH" \
    GUIDANCE="$GUIDANCE" \
    CHEMICAL_SYSTEMS="$CHEMICAL_SYSTEMS" \
    CHEMICAL_SYSTEMS_FILE="$CHEMICAL_SYSTEMS_FILE" \
    ELEMENTS="$ELEMENTS" \
    COMBO_SIZES="$COMBO_SIZES" \
    WORKDIR="$MATTERGEN_ROOT" \
    bash "$PIPELINE_DIR/generate.sh"
done

echo "==== Unified evaluate.sh over generated segments ===="
ROOT="$SEGMENTS_ROOT" \
  WORKDIR="$MATTERGEN_ROOT" \
  LOGDIR="$RESULTS_ROOT/logs_eval" \
  RECURSIVE=1 \
  bash "$PIPELINE_DIR/evaluate.sh"

screen_cmd=(
  python "$PIPELINE_DIR/screen_all_extxyz.py"
  --workdir "$MATTERGEN_ROOT"
  --base "$SEGMENTS_ROOT"
  --out "$RESULTS_ROOT/stage2_candidates.csv"
  --screened-out "$RESULTS_ROOT/screened_out.csv"
  --r-cut "$R_CUT"
  --super $SUPERCELL
  --required-elements $REQUIRED_ELEMENTS
  --topk "$SCREEN_TOPK"
  --refs-out "$RESULTS_ROOT/screen_refs.txt"
)
if [[ "$REQUIRE_CHARGE_BALANCE" == "1" || "$REQUIRE_CHARGE_BALANCE" == "true" ]]; then
  screen_cmd+=(--require-charge-balance)
else
  screen_cmd+=(--no-charge-balance)
fi
if [[ "$USE_SMACT" == "1" || "$USE_SMACT" == "true" ]]; then
  screen_cmd+=(--use-smact)
else
  screen_cmd+=(--no-smact)
fi
if [[ -n "$ALLOWED_ELEMENTS" ]]; then
  screen_cmd+=(--allowed-elements $ALLOWED_ELEMENTS)
fi
if [[ "$FILTER_LIGHT_OXY" == "1" || "$FILTER_LIGHT_OXY" == "true" ]]; then
  screen_cmd+=(--light-oxy $LIGHT_OXY)
else
  screen_cmd+=(--no-light-oxy)
fi

echo "==== Unified screen_all_extxyz.py ===="
"${screen_cmd[@]}"

top_cmd=(
  python "$PIPELINE_DIR/run_top300_pipeline.py"
  --workdir "$MATTERGEN_ROOT"
  --output-dir "$RESULTS_ROOT/top300_run"
  --stage2-csv "$RESULTS_ROOT/stage2_candidates.csv"
  --topk "$TOPK"
  --gpu-workers "$GPU_WORKERS"
  --selection-mode "$SELECTION_MODE"
  --refs-out "$RESULTS_ROOT/top300_run/top300_refs.txt"
  --export-dir "$RESULTS_ROOT/top300_run/exported_300cifs"
  --export-prefix cand300
  --export-index-name export_300index.csv
  --ehull-threshold 0.05
  --ehull-out "$RESULTS_ROOT/top300_run/chgnet_hull_top300.csv"
  --filtered-out "$RESULTS_ROOT/top300_run/chgnet_hull_top300_filtered.csv"
  --voltage-out "$RESULTS_ROOT/top300_run/chgnet_voltage_window_top300.csv"
  --voltage-threshold "$VOLTAGE_THRESHOLD"
  --min-voltage-window "$MIN_VOLTAGE_WINDOW"
  --mattersim-checkpoint "$MATTERSIM_CHECKPOINT"
  --relax-fmax "$RELAX_FMAX"
  --relax-steps "$RELAX_STEPS"
)
if [[ -n "$TARGET_VOLTAGE" ]]; then
  top_cmd+=(--target-voltage "$TARGET_VOLTAGE")
fi
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  top_cmd+=(--dry-run)
fi

echo "==== Global run_top300_pipeline.py ===="
"${top_cmd[@]}"

echo "[done] Full pipeline finished: $RESULTS_ROOT"
