#!/usr/bin/env python3
"""
Collect metrics.json files into a single table.

For every subdirectory under --root containing a metrics.json, this script extracts
the numerical "value" for each metric key (e.g. avg_energy_above_hull_per_atom,
avg_rmsd_from_relaxation, …, recall) and writes a CSV with one row per structure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics.json files into a CSV table.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/mattergen_eval_top300"),
        help="Directory containing per-structure subdirectories with metrics.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/mattergen_eval_top300/metrics_table.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Explicit ordered list of metric keys to include. "
        "If omitted, all keys found in the first metrics.json are used.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Optional[Dict[str, float]]:
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None

    metrics: Dict[str, float] = {}
    for key, payload in data.items():
        if isinstance(payload, dict) and "value" in payload:
            try:
                metrics[key] = float(payload["value"])
            except (TypeError, ValueError):
                continue
    if not metrics:
        print(f"[WARN] No numeric metrics extracted from {path}")
        return None
    return metrics


def discover_entries(root: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        if metrics is None:
            continue
        entries.append({"structure": subdir.name, "metrics": metrics, "path": metrics_path})
    return entries


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    entries = discover_entries(root)
    if not entries:
        raise SystemExit(f"No metrics.json files found under {root}")

    metric_order: List[str]
    if args.metrics:
        metric_order = args.metrics
    else:
        all_keys = set()
        for entry in entries:
            all_keys.update(entry["metrics"].keys())
        metric_order = sorted(all_keys)

    fieldnames = ["structure"] + metric_order
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {"structure": entry["structure"]}
            metrics: Dict[str, float] = entry["metrics"]
            for key in metric_order:
                row[key] = metrics.get(key, "")
            writer.writerow(row)

    print(f"[INFO] Aggregated {len(entries)} structures into {args.out}")


if __name__ == "__main__":
    main()
