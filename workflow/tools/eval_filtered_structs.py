#!/usr/bin/env python3
"""
Batch-evaluate structures listed in final_candidates.csv with mattergen-evaluate.

For each row in the input CSV, this script runs mattergen-evaluate on the referenced CIF
and stores the resulting metrics.json / relaxed.extxyz under an output directory dedicated
to that structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List


DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate each structure from a CHGNet hull CSV using mattergen-evaluate."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RESULTS / "top300_run" / "final_candidates.csv",
        help="Input CSV containing at least 'file' and 'path' columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RESULTS / "evaluation",
        help="Root directory where per-structure outputs will be written.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_RESULTS / "evaluation" / "logs",
        help="Directory for per-structure evaluation logs.",
    )
    parser.add_argument(
        "--mattergen",
        default="mattergen-evaluate",
        help="Command/binary used to invoke mattergen-evaluate.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip structures that already have metrics.json in the output directory.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw arguments appended to the mattergen-evaluate command "
        "(e.g. --extra-arg=--relax=False). Repeatable.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="CSV file to write aggregated metrics from all metrics.json files.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip generating the aggregated metrics summary CSV.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip running mattergen-evaluate and only build the summary CSV "
        "(requires existing metrics.json files).",
    )
    args = parser.parse_args()
    if args.summary_csv is None:
        args.summary_csv = args.out_dir / "metrics_summary.csv"
    return args


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    for col in ("file", "path"):
        if col not in reader.fieldnames:
            raise RuntimeError(f"Required column '{col}' missing from {csv_path}")
    return rows


def structure_name(row: Dict[str, str]) -> str:
    path = Path(row["path"]).expanduser()
    return Path(row.get("file") or path.stem).stem


def flatten(obj, prefix="", result=None):
    if result is None:
        result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            flatten(value, new_key, result)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            flatten(value, new_key, result)
    else:
        if prefix:
            result[prefix] = obj
    return result


def build_summary(rows: List[Dict[str, str]], args) -> None:
    summary_rows: List[Dict[str, object]] = []
    metric_keys = set()

    for row in rows:
        name = structure_name(row)
        metrics_path = args.out_dir / name / "metrics.json"
        if not metrics_path.exists():
            print(f"[WARN] Summary skipping missing metrics: {metrics_path}")
            continue
        try:
            with metrics_path.open() as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[WARN] Failed to read {metrics_path}: {exc}")
            continue

        flat = flatten(data)
        if not flat:
            print(f"[WARN] No metrics extracted from {metrics_path}")
            continue
        metric_keys.update(flat.keys())

        record: Dict[str, object] = {
            "structure": name,
            "metrics_path": str(metrics_path),
            "file": row.get("file"),
            "formula": row.get("formula"),
            "chemsys": row.get("chemsys"),
            "cif_path": row.get("path"),
        }
        record.update(flat)
        summary_rows.append(record)

    if not summary_rows:
        print("[WARN] No metrics were aggregated; summary CSV not written.")
        return

    fieldnames = [
        "structure",
        "metrics_path",
        "file",
        "formula",
        "chemsys",
        "cif_path",
    ] + sorted(metric_keys)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"[INFO] Metrics summary written to {args.summary_csv} ({len(summary_rows)} structures)")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for idx, row in enumerate(rows, start=1):
            source = Path(row["path"]).expanduser()
            if not source.exists():
                print(f"[WARN] ({idx}/{len(rows)}) Missing CIF path, skipping: {source}")
                continue

            name = structure_name(row)
            target_dir = args.out_dir / name
            target_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = target_dir / "metrics.json"
            relaxed_path = target_dir / "relaxed.extxyz"
            log_path = args.log_dir / f"{name}.log"

            if args.skip_existing and metrics_path.exists():
                print(f"[SKIP] ({idx}/{len(rows)}) {name} already has metrics.json")
                continue

            # Clean old outputs to avoid mixing runs
            if metrics_path.exists():
                metrics_path.unlink()
            if relaxed_path.exists():
                relaxed_path.unlink()

            try:
                with tempfile.TemporaryDirectory(prefix=f"eval_{name}_", dir=args.out_dir) as tmp_dir:
                    tmp_dir_path = Path(tmp_dir)
                    tmp_cif = tmp_dir_path / source.name
                    shutil.copy2(source, tmp_cif)

                    cmd = [
                        args.mattergen,
                        str(tmp_dir_path),
                        "--relax=True",
                        "--structure_matcher=disordered",
                        f"--save_as={metrics_path}",
                        f"--structures_output_path={relaxed_path}",
                    ]
                    if args.extra_arg:
                        cmd.extend(args.extra_arg)

                    print(f"[RUN] ({idx}/{len(rows)}) {' '.join(cmd)}")
                    with log_path.open("w") as log_fh:
                        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

                    if proc.returncode != 0:
                        print(f"[FAIL] ({idx}/{len(rows)}) {name} exit={proc.returncode}; see {log_path}")
                        continue

                    print(f"[OK] ({idx}/{len(rows)}) metrics -> {metrics_path}")
            except Exception as exc:
                print(f"[ERROR] ({idx}/{len(rows)}) Failed to evaluate {name}: {exc}")

    if not args.skip_summary:
        build_summary(rows, args)


if __name__ == "__main__":
    main()
