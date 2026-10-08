#!/bin/bash
#SBATCH --job-name=mattergen-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Adjust these module commands to your supercomputer.
# module purge
# module load cuda/12.1
# module load anaconda/2023

# Adjust this to the environment you create on the supercomputer.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mattergen

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/LiNbOCl}"
export MATTERGEN_ROOT="$PROJECT_ROOT/mattergen"
export WORKFLOW_ROOT="$PROJECT_ROOT/workflow"

# Put outputs and caches on the high-capacity parallel filesystem, not $HOME.
export RESULTS_ROOT="${RESULTS_ROOT:-$SCRATCH/mattergen_runs/results}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$SCRATCH/mattergen_runs/_runtime}"

# Do not print your real key in logs. Prefer adding it as a scheduler secret if available.
export MP_API_KEY="${MP_API_KEY:?Set MP_API_KEY before sbatch or in a protected env file}"

export CHEMICAL_SYSTEMS="${CHEMICAL_SYSTEMS:-Li-Nb-O-Cl}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_BATCHES_PER_SEGMENT="${NUM_BATCHES_PER_SEGMENT:-20}"
export SEGMENTS="${SEGMENTS:-10}"
export SCREEN_TOPK="${SCREEN_TOPK:-300}"
export TOPK="${TOPK:-300}"

mkdir -p logs "$RESULTS_ROOT" "$RUNTIME_ROOT"

bash "$WORKFLOW_ROOT/pipeline/run_full_pipeline.sh"
