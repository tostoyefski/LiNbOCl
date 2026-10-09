#!/bin/bash
set -euo pipefail
repo=/home/litao/projects/LiNbOCl
export PATH="$repo/_runtime/venv/bin:$PATH"
export MPLCONFIGDIR="$repo/_runtime/matplotlib"
set -a
source /home/litao/projects/LiNbOCl/_runtime/mp.env
set +a
source_root="${SCREEN_INPUT_ROOT:-$repo/results/run6000_20261008}"
root="${SCREEN_OUTPUT_ROOT:-$source_root/screening_no_oxygen}"
mkdir -p "$root"
rm -f "$root/SCREENING_COMPLETE"
python - "$source_root" "$root" <<'PY'
import sys,json
from pathlib import Path
from ase.io import iread
source_root=Path(sys.argv[1])
root=Path(sys.argv[2])
statuses=[json.loads((source_root/'_segments'/f'batch{i:03d}'/'Li-Nb-O-Cl/status.json').read_text()) for i in range(30)]
if any(s['stage']!='completed' for s in statuses): raise RuntimeError('Not all shards completed')
count=sum(sum(1 for _ in iread(source_root/'_segments'/f'batch{i:03d}'/'Li-Nb-O-Cl/relaxed.extxyz',index=':')) for i in range(30))
if count!=6000: raise RuntimeError(f'Expected 6000 relaxed frames, got {count}')
(root/'generation_summary.json').write_text(json.dumps({'generated':6000,'relaxed':count,'conditions':{'chemical_system':'Li-Nb-O-Cl','energy_above_hull':0.05}},indent=2))
print('Verified 6000 generated and relaxed structures',flush=True)
PY
python "$repo/workflow/pipeline/screen_all_extxyz.py" --workdir "$repo/mattergen" --base "$source_root/_segments" --required-elements Li Nb O Cl --allowed-elements Li Nb O Cl --require-charge-balance --use-smact --no-light-oxy --out "$root/stage2_candidates.csv" --screened-out "$root/screened_out.csv" --refs-out "$root/screen_refs.txt"
top_cmd=(
  python "$repo/workflow/pipeline/run_top300_pipeline.py"
  --workdir "$repo/mattergen"
  --stage2-csv "$root/stage2_candidates.csv"
  --output-dir "$root/top300_run"
  --topk 6000
  --selection-mode diverse
  --ehull-threshold 0.05
  --gpu-workers 2
  --mattersim-checkpoint "${MATTERSIM_CHECKPOINT:-$repo/_runtime/MatterSim-v1.0.0-1M.pth}"
  --relax-fmax "${RELAX_FMAX:-0.05}"
  --relax-steps "${RELAX_STEPS:-500}"
  --novelty-training-splits "${NOVELTY_TRAINING_SPLITS:-train}"
)
if [[ -n "${NOVELTY_TRAINING_DATA:-}" ]]; then top_cmd+=(--novelty-training-data "$NOVELTY_TRAINING_DATA"); fi
if [[ -n "${NOVELTY_REFERENCE_DATA:-}" ]]; then top_cmd+=(--novelty-reference-data "$NOVELTY_REFERENCE_DATA"); fi
if [[ "${SKIP_NOVELTY:-0}" == "1" || "${SKIP_NOVELTY:-0}" == "true" ]]; then top_cmd+=(--skip-novelty); fi
"${top_cmd[@]}"
date -u +%FT%TZ > "$root/SCREENING_COMPLETE"
