#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTEXT_DIR="${CONTEXT_DIR:-$SOURCE_ROOT/.mattergen_docker_context}"
IMAGE_NAME="${IMAGE_NAME:-mattergen-webapp:cu118}"

mkdir -p "$CONTEXT_DIR"

rsync -a --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.mypy_cache' \
  --exclude '.pytest_cache' \
  --exclude 'results' \
  --exclude 'results[0-9]*' \
  --exclude 'runs_e' \
  --exclude 'md_traj' \
  --exclude 'md_traj_results*' \
  --exclude 'backend/job_logs/*' \
  "$SOURCE_ROOT/mattergen" \
  "$SOURCE_ROOT/mattergen_webapp" \
  "$CONTEXT_DIR/"

docker build -f "$CONTEXT_DIR/mattergen_webapp/container/Dockerfile" -t "$IMAGE_NAME" "$CONTEXT_DIR"

echo "[done] Built $IMAGE_NAME"
