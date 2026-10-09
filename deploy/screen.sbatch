#!/bin/bash
#SBATCH --job-name=LiNbOCl-screen
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/litao/projects/LiNbOCl/results/run6000_20261008/logs/screen-%j.log
set -euo pipefail
repo=/home/litao/projects/LiNbOCl
exec bash "$repo/deploy/in_container.sh" /bin/bash "$repo/deploy/screen_inside.sh"
