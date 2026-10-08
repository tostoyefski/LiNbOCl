#!/usr/bin/env bash
# Container-friendly alias for the command-line full pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_full_pipeline_hpc.sh"
