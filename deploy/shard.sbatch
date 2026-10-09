#!/bin/bash
#SBATCH --job-name=LiNbOCl-6000
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-29%2
#SBATCH --output=/home/litao/projects/LiNbOCl/results/run6000_20261008/logs/shard-%A_%a.log
set -euo pipefail
repo=/home/litao/projects/LiNbOCl
export MPLCONFIGDIR="$repo/_runtime/matplotlib"
export TORCH_HOME="$repo/_runtime/torch"
export HF_HOME="$repo/_runtime/huggingface"
mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME"
exec bash "$repo/deploy/in_container.sh" "$repo/_runtime/venv/bin/python" -u "$repo/deploy/run_shard.py"
