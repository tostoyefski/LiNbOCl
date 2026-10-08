#!/usr/bin/env python3
"""
Merge many per-run MatterGen `metrics.json` files into ONE `metrics.json`
with the SAME schema that MatterGen README/benchmark expects:
{
  "metric_name": {"value": <float>, "description": "<str>"},
  ...
}

用法示例：
# 每个子目录都是 16 个结构，推荐（RMSD 按成功数加权，其他等价简单平均）
python workflow/analysis/merge_metrics_to_single_json.py --root results/_segments --out results/combined_metrics.json --weighting successful --assume-n 16

# 或者，全等权（简单平均）
python workflow/analysis/merge_metrics_to_single_json.py \
  --root results/_segments \
  --out results/combined_metrics.json \
  --weighting equal

# 只打印合并后的 JSON 预览，不写文件
python workflow/analysis/merge_metrics_to_single_json.py \
  --root results/_segments \
  --out results/combined_metrics.json \
  --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile
import re

MetricName = str


@dataclass
class RunMetrics:
    path: Path                  # path to metrics.json
    metrics: Dict[str, Dict]    # {metric_name: {"value": float, "description": str}}
    n_structs: Optional[int]    # number of structures attempted/considered in that run
    frac_successful: Optional[float]  # frac_successful_jobs if present


def find_metrics_files(roots: List[Path], name: str = "metrics.json") -> List[Path]:
    files: List[Path] = []
    for r in roots:
        if not r.exists():
            print(f"[WARN] Root does not exist: {r}", file=sys.stderr)
            continue
        files.extend(r.rglob(name))
    files = sorted(set(files))
    return files


def safe_read_json(p: Path) -> Optional[Dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {p}: {e}", file=sys.stderr)
        return None


def count_frames_in_extxyz(extxyz_path: Path) -> int:
    """
    Heuristic: count lines containing 'Lattice=' in EXTXYZ header lines.
    Each frame typically has a header/comment line with Lattice=... Properties=...
    """
    cnt = 0
    pattern = re.compile(r"\bLattice=")
    try:
        with extxyz_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if pattern.search(line):
                    cnt += 1
    except Exception:
        return 0
    return cnt


def infer_n_structs(run_dir: Path) -> Optional[int]:
    # 1) Count frames in any *.extxyz
    extxyz_files = list(run_dir.glob("*.extxyz"))
    total = 0
    for p in extxyz_files:
        n = count_frames_in_extxyz(p)
        total += n
    if total > 0:
        return total

    # 2) Count *.cif files
    cif_files = list(run_dir.glob("*.cif"))
    if len(cif_files) > 0:
        return len(cif_files)

    # 3) Count entries in generated_crystals_cif.zip (if present)
    zips = list(run_dir.glob("*.zip"))
    for z in zips:
        try:
            with zipfile.ZipFile(z, "r") as zf:
                n_cif = sum(1 for n in zf.namelist() if n.lower().endswith(".cif"))
                if n_cif > 0:
                    return n_cif
        except Exception:
            continue

    return None


def load_runs(metrics_files: List[Path]) -> List[RunMetrics]:
    runs: List[RunMetrics] = []
    for m in metrics_files:
        payload = safe_read_json(m)
        if payload is None or not isinstance(payload, dict):
            continue
        # normalize: ensure we only keep {"value", "description"} shape
        norm: Dict[str, Dict] = {}
        for k, v in payload.items():
            if isinstance(v, dict) and "value" in v:
                try:
                    val = float(v["value"])
                except Exception:
                    continue
                norm[k] = {"value": val, "description": v.get("description", "")}

        run_dir = m.parent
        n_structs = infer_n_structs(run_dir)

        frac_successful = None
        if "frac_successful_jobs" in norm:
            try:
                frac_successful = float(norm["frac_successful_jobs"]["value"])
            except Exception:
                pass

        runs.append(RunMetrics(path=m, metrics=norm, n_structs=n_structs, frac_successful=frac_successful))
    return runs


def pooled_value_for_metric(
    name: MetricName,
    runs: List[RunMetrics],
    weighting: str = "auto",
    assume_n: Optional[int] = None,
) -> Tuple[Optional[float], str]:
    """
    Compute the pooled metric value across runs.
    weighting:
      - 'equal':       each run weight = 1
      - 'n_structs':   weight = n (use assume_n if unknown)
      - 'successful':  weight = n * frac_successful_jobs for avg_rmsd_from_relaxation, else weight = n
      - 'auto':        alias of 'successful'
    Returns (value, description_used)
    """
    # Pick a description from the first run that has it
    desc = ""
    for r in runs:
        if name in r.metrics and r.metrics[name].get("description"):
            desc = r.metrics[name]["description"]
            break

    vals_weights: List[Tuple[float, float]] = []
    for r in runs:
        if name not in r.metrics:
            continue
        v = r.metrics[name]["value"]

        # Choose base weight
        if weighting == "equal":
            w = 1.0
        else:
            n = r.n_structs if r.n_structs is not None else assume_n
            if n is None:
                # unknown -> NaN weight; will fallback to simple mean
                w = math.nan
            else:
                w = float(n)

        # Special-case for avg_rmsd_from_relaxation in 'successful'/'auto' mode
        if name == "avg_rmsd_from_relaxation" and weighting in ("successful", "auto"):
            # approximate #successful = n * frac_successful_jobs
            if (r.frac_successful is not None) and (not math.isnan(w)):
                w = w * r.frac_successful
            # else keep w as-is (n or NaN)

        vals_weights.append((v, w))

    if not vals_weights:
        return (None, desc)

    def weighted_mean(pairs: List[Tuple[float, float]]) -> float:
        sw = sum(w for _, w in pairs if not math.isnan(w))
        if sw > 0:
            return sum(v * w for v, w in pairs if not math.isnan(w)) / sw
        # fallback simple mean if no valid weights
        return sum(v for v, _ in pairs) / len(pairs)

    if name.startswith("frac_") or name in ("precision", "recall"):
        # Fractions: use chosen weights (equal/n_structs). With equal n across runs, equal == n_structs.
        return (weighted_mean(vals_weights), desc)

    # avg_* and others:
    return (weighted_mean(vals_weights), desc)


def merge_to_single_json(
    roots: List[Path],
    out_path: Path,
    weighting: str,
    assume_n: Optional[int],
    dry_run: bool = False
) -> int:
    metrics_files = find_metrics_files(roots, name="metrics.json")
    if not metrics_files:
        print("[ERROR] No metrics.json files found under:", *roots, file=sys.stderr)
        return 2

    runs = load_runs(metrics_files)
    if not runs:
        print("[ERROR] No readable metrics.json files.", file=sys.stderr)
        return 3

    # Basic stats
    with_n = sum(1 for r in runs if r.n_structs is not None)
    print(f"[INFO] Loaded {len(runs)} metrics files. Structure counts known for {with_n}/{len(runs)} runs.")
    if assume_n is not None:
        print(f"[INFO] Assuming {assume_n} structures per run when unknown.")

    # Union of all metric names
    all_metric_names = set()
    for r in runs:
        all_metric_names.update(r.metrics.keys())

    merged: Dict[str, Dict] = {}
    for name in sorted(all_metric_names):
        val, desc = pooled_value_for_metric(name, runs, weighting=weighting, assume_n=assume_n)
        if val is None:
            continue
        merged[name] = {"value": float(val), "description": desc}

    if dry_run:
        print(json.dumps(merged, indent=2, ensure_ascii=False))
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote merged metrics to: {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Merge many MatterGen metrics.json files into a single metrics.json (pooled)."
    )
    ap.add_argument(
        "--root", nargs="+", required=True,
        help="One or more roots to search recursively for metrics.json files."
    )
    ap.add_argument(
        "--out", required=True,
        help="Path to write the merged metrics.json"
    )
    ap.add_argument(
        "--weighting", choices=["auto", "equal", "n_structs", "successful"], default="auto",
        help="Pooled weighting strategy. Default 'auto' (same as 'successful')."
    )
    ap.add_argument(
        "--assume-n", type=int, default=None,
        help="Assume this many structures per run when unknown (used by 'n_structs'/'successful' modes)."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Only print the merged JSON, don't write."
    )
    args = ap.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.root]
    out_path = Path(args.out).expanduser().resolve()
    return merge_to_single_json(
        roots, out_path, weighting=args.weighting, assume_n=args.assume_n, dry_run=args.dry_run
    )


if __name__ == "__main__":
    raise SystemExit(main())

