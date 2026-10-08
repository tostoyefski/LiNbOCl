#!/usr/bin/env python3
# 独立拷贝版，增加 --workdir 以便与原仓库隔离。

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-K CIF export and voltage window pipeline.")
    parser.add_argument("--workdir", type=Path, default=Path("."), help="执行目录（通常为 mattergen 仓库根）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "top300_run", help="统一的输出目录（refs/hull/voltage等都放这里）")
    parser.add_argument("--stage2-csv", type=Path, default=DEFAULT_RESULTS / "stage2_candidates.csv", help="CSV produced by screen_all_extxyz.py.")
    parser.add_argument("--topk", type=int, default=300, help="Number of highest quick_score entries to export.")
    parser.add_argument("--refs-out", type=Path, default=None, help="Output file storing path::frame references.")
    parser.add_argument("--export-script", type=Path, default=SCRIPTS_DIR / "export_refs_to_structs.py", help="Path to export_refs_to_structs.py.")
    parser.add_argument("--export-dir", type=Path, default=None, help="Directory where CIF files will be written.")
    parser.add_argument("--export-prefix", default="cand300", help="Filename prefix for exported CIFs.")
    parser.add_argument("--export-index-name", default=None, help="Filename (within export-dir) for the CIF index CSV.")
    parser.add_argument("--ehull-script", type=Path, default=SCRIPTS_DIR / "compute_ehull_chgnet.py", help="Path to compute_ehull_chgnet.py.")
    parser.add_argument("--ehull-out", type=Path, default=None, help="Output CSV for hull results.")
    parser.add_argument("--ehull-threshold", type=float, default=0.05, help="Energy above hull cutoff (eV/atom).")
    parser.add_argument("--filtered-out", type=Path, default=None, help="CSV containing rows that pass the hull threshold.")
    parser.add_argument("--voltage-script", type=Path, default=SCRIPTS_DIR / "compute_voltage_window.py", help="Path to compute_voltage_window.py.")
    parser.add_argument("--voltage-out", type=Path, default=None, help="Output CSV for voltage window results.")
    parser.add_argument("--voltage-step", type=float, default=None, help="Optional override for compute_voltage_window.py --voltage-step.")
    parser.add_argument("--voltage-threshold", type=float, default=0.05, help="Override for compute_voltage_window.py --threshold.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs but skip CHGNet / MP computations.")
    return parser.parse_args()


def read_stage2_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage2 CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    if "quick_score" not in reader.fieldnames or "path" not in reader.fieldnames or "frame" not in reader.fieldnames:
        raise RuntimeError("Required columns (quick_score, path, frame) missing from stage2 CSV.")
    return rows


def select_top_refs(rows: Iterable[dict], topk: int) -> List[str]:
    converted: List[tuple[float, str]] = []
    for row in rows:
        try:
            score = float(row["quick_score"])
        except (TypeError, ValueError):
            continue
        path = row.get("path")
        frame = row.get("frame")
        if not path or frame is None:
            continue
        ref = f"{path}::{frame}"
        converted.append((score, ref))
    converted.sort(key=lambda x: x[0], reverse=True)
    if topk < len(converted):
        converted = converted[:topk]
    return [ref for _, ref in converted]


def write_refs(refs: List[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        fh.write("\n".join(refs))
    print(f"[INFO] Wrote {len(refs)} references to {dest}")


def run_command(cmd: List[str], cwd: Path | None = None) -> None:
    display = " ".join(cmd)
    print(f"[CMD] {display}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def run_export(export_script: Path, refs_path: Path, outdir: Path, prefix: str, cwd: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(export_script), "--refs", str(refs_path), "--outdir", str(outdir), "--prefix", prefix]
    run_command(cmd, cwd=cwd)
    index_path = outdir / "export_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Expected index file not found: {index_path}")
    return index_path


def rename_index(index_path: Path, target_name: str) -> Path:
    target_path = index_path.with_name(target_name)
    if target_path.exists():
        target_path.unlink()
    index_path.rename(target_path)
    print(f"[INFO] Renamed index to {target_path}")
    root_target = DEFAULT_RESULTS / target_name
    try:
        if root_target.resolve() != target_path.resolve():
            shutil.copyfile(target_path, root_target)
            print(f"[INFO] Copied index to repository root: {root_target}")
    except OSError as exc:
        print(f"[WARN] Failed to copy index to {root_target}: {exc}")
    return target_path


def run_ehull(ehull_script: Path, cif_dir: Path, out_csv: Path, cwd: Path) -> None:
    cmd = [sys.executable, str(ehull_script), "--cif-dir", str(cif_dir), "--out", str(out_csv)]
    run_command(cmd, cwd=cwd)
    if not out_csv.exists():
        raise FileNotFoundError(f"Hull results missing: {out_csv}")


def filter_hull(in_csv: Path, out_csv: Path, threshold: float) -> int:
    with in_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    if not rows:
        raise RuntimeError(f"No rows in hull output: {in_csv}")
    kept: List[dict] = []
    for row in rows:
        try:
            val = float(row.get("energy_above_hull_eV", "nan"))
        except (TypeError, ValueError):
            continue
        if val <= threshold:
            kept.append(row)
    if not kept:
        print(f"[WARN] No rows satisfy energy_above_hull <= {threshold:.3f} eV/atom.")
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in kept:
            writer.writerow(row)
    print(f"[INFO] Filtered hull results written to {out_csv} ({len(kept)} rows kept)")
    return len(kept)


def run_voltage(
    voltage_script: Path,
    stable_csv: Path,
    out_csv: Path,
    voltage_step: float | None,
    threshold: float | None,
    cwd: Path,
) -> None:
    cmd = [sys.executable, str(voltage_script), "--stable-csv", str(stable_csv), "--out", str(out_csv)]
    if voltage_step is not None:
        cmd.extend(["--voltage-step", str(voltage_step)])
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    run_command(cmd, cwd=cwd)
    if not out_csv.exists():
        raise FileNotFoundError(f"Voltage results missing: {out_csv}")


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fill defaults based on unified output_dir
    if args.refs_out is None:
        args.refs_out = output_dir / "top300_refs.txt"
    if args.export_dir is None:
        args.export_dir = output_dir / "exported_300cifs"
    if args.export_index_name is None:
        args.export_index_name = "export_300index.csv"
    if args.ehull_out is None:
        args.ehull_out = output_dir / "chgnet_hull_top300.csv"
    if args.filtered_out is None:
        args.filtered_out = output_dir / "chgnet_hull_top300_filtered.csv"
    if args.voltage_out is None:
        args.voltage_out = output_dir / "chgnet_voltage_window_top300.csv"

    os.makedirs(args.export_dir, exist_ok=True)
    os.chdir(workdir)

    rows = read_stage2_rows(args.stage2_csv)
    refs = select_top_refs(rows, args.topk)
    if not refs:
        raise RuntimeError("No valid references were found in the stage2 CSV.")
    write_refs(refs, args.refs_out)

    index_path = run_export(args.export_script, args.refs_out, args.export_dir, args.export_prefix, cwd=workdir)
    renamed_index = rename_index(index_path, args.export_index_name)

    if args.dry_run:
        print("[INFO] Dry run requested; skipping hull and voltage computations.")
        return

    run_ehull(args.ehull_script, args.export_dir, args.ehull_out, cwd=workdir)
    kept = filter_hull(args.ehull_out, args.filtered_out, args.ehull_threshold)
    if kept == 0:
        print("[WARN] No structures met the hull criterion; voltage window step will be skipped.")
        return

    run_voltage(args.voltage_script, args.filtered_out, args.voltage_out, args.voltage_step, args.voltage_threshold, cwd=workdir)
    print(f"[INFO] Voltage window results saved to {args.voltage_out}")
    print(f"[INFO] Export index located at {renamed_index}")


if __name__ == "__main__":
    main()
