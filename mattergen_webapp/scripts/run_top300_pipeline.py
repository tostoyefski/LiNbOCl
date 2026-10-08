#!/usr/bin/env python3
"""Deduplicate, select diverse candidates, and apply hull/voltage gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def parse_args(argv=None, *, legacy_defaults=False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run top-K CIF export and voltage window pipeline.")
    parser.add_argument("--workdir", type=Path, default=Path("."), help="执行目录（通常为 mattergen 仓库根）")
    parser.add_argument("--output-dir", type=Path, default=Path(".") if legacy_defaults else DEFAULT_RESULTS / "top300_run", help="统一的输出目录（refs/hull/voltage等都放这里）")
    parser.add_argument("--stage2-csv", type=Path, default=Path("results/stage2_candidates.csv") if legacy_defaults else DEFAULT_RESULTS / "stage2_candidates.csv", help="CSV produced by screen_all_extxyz.py.")
    parser.add_argument("--topk", type=int, default=300, help="Maximum number of unique candidates to export.")
    parser.add_argument("--selection-mode", choices=("diverse", "score"), default="diverse", help="Select by composition round-robin (default), or proxy score; both deduplicate structures first.")
    parser.add_argument("--selection-audit", type=Path, default=None, help="CSV recording selection, duplicates and unreadable inputs.")
    parser.add_argument("--refs-out", type=Path, default=Path("top300_refs.txt") if legacy_defaults else None, help="Output file storing path::frame references.")
    parser.add_argument("--export-script", type=Path, default=SCRIPTS_DIR / "export_refs_to_structs.py", help="Path to export_refs_to_structs.py.")
    parser.add_argument("--export-dir", type=Path, default=Path("results/exported_300cifs") if legacy_defaults else None, help="Directory where CIF files will be written.")
    parser.add_argument("--export-prefix", default="cand300", help="Filename prefix for exported CIFs.")
    parser.add_argument("--export-index-name", default=None, help="Filename (within export-dir) for the CIF index CSV.")
    parser.add_argument("--ehull-script", type=Path, default=SCRIPTS_DIR / "compute_ehull_chgnet.py", help="Path to compute_ehull_chgnet.py.")
    parser.add_argument("--ehull-out", type=Path, default=None, help="Output CSV for hull results.")
    parser.add_argument("--ehull-threshold", type=float, default=0.05, help="Energy above hull cutoff (eV/atom).")
    parser.add_argument("--filtered-out", type=Path, default=None, help="CSV containing rows that pass the hull threshold.")
    parser.add_argument("--voltage-script", type=Path, default=SCRIPTS_DIR / "compute_voltage_window.py", help="Path to compute_voltage_window.py.")
    parser.add_argument("--voltage-out", type=Path, default=None, help="Output CSV for voltage window results.")
    parser.add_argument("--voltage-step", type=float, default=None, help="Optional override for compute_voltage_window.py --voltage-step.")
    parser.add_argument("--voltage-threshold", type=float, default=1e-3, help="Grand-hull tolerance in eV/non-Li atom; a numerical tolerance, distinct from the bulk hull cutoff.")
    parser.add_argument("--target-voltage", type=float, default=None, help="Optional working voltage vs Li/Li+; candidates must be stable at this exact voltage.")
    parser.add_argument("--min-voltage-window", type=float, default=0.0, help="Minimum verified width in V; zero still requires a nonzero stable interval.")
    parser.add_argument("--final-out", type=Path, default=None, help="CSV of candidates passing both hull and voltage gates.")
    parser.add_argument("--voltage-filter-audit", type=Path, default=None, help="CSV recording every voltage acceptance/rejection.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs but skip CHGNet / MP computations.")
    args = parser.parse_args(argv)
    if args.topk < 1:
        parser.error("--topk must be positive")
    for name in ("ehull_threshold", "voltage_threshold", "min_voltage_window"):
        val = getattr(args, name)
        if not math.isfinite(val) or val < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.voltage_step is not None and (not math.isfinite(args.voltage_step) or args.voltage_step <= 0):
        parser.error("--voltage-step must be finite and positive")
    if args.target_voltage is not None and not math.isfinite(args.target_voltage):
        parser.error("--target-voltage must be finite")
    return args


def read_stage2_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage2 CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    fields = reader.fieldnames or []
    if not {"quick_score", "path", "frame"}.issubset(fields):
        raise RuntimeError("Required columns (quick_score, path, frame) missing from stage2 CSV.")
    if "score_kind" not in fields or any(row.get("score_kind") != "li_periodic_geometry_proxy_v1" for row in rows):
        raise RuntimeError("Stage2 CSV uses an old or unsupported score. Re-run screen_all_extxyz.py before selecting candidates.")
    return rows


def select_top_refs(rows: Iterable[dict], topk: int, selection_mode="diverse") -> List[str]:
    from candidate_selection import select_candidates
    return select_candidates(rows, topk, selection_mode=selection_mode).refs


def write_refs(refs: List[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        fh.write("\n".join(refs))
    print(f"[INFO] Wrote {len(refs)} references to {dest}")


def write_empty_csv(path: Path, fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=list(fields)).writeheader()


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
    expected = set(refs_path.read_text().splitlines())
    with index_path.open(newline="") as fh:
        exported = list(csv.DictReader(fh))
    if len(exported) != len(expected) or {row.get("ref") for row in exported} != expected:
        raise RuntimeError("Export index does not contain exactly the selected references")
    return index_path


def rename_index(index_path: Path, target_name: str) -> Path:
    target_path = index_path.with_name(target_name)
    if index_path != target_path:
        index_path.replace(target_path)
    print(f"[INFO] Renamed index to {target_path}")
    root_target = DEFAULT_RESULTS / target_name
    try:
        if root_target.resolve() != target_path.resolve():
            shutil.copyfile(target_path, root_target)
            print(f"[INFO] Copied index to repository root: {root_target}")
    except OSError as exc:
        print(f"[WARN] Failed to copy index to {root_target}: {exc}")
    return target_path


def run_ehull(ehull_script: Path, cif_dir: Path, out_csv: Path, cwd: Path, index_csv: Path | None = None) -> None:
    cmd = [sys.executable, str(ehull_script), "--cif-dir", str(cif_dir), "--out", str(out_csv)]
    if index_csv is not None:
        cmd.extend(["--cif-index", str(index_csv)])
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
        if math.isfinite(val) and val <= threshold:
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
    target_voltage: float | None = None,
) -> None:
    cmd = [sys.executable, str(voltage_script), "--stable-csv", str(stable_csv), "--out", str(out_csv)]
    if voltage_step is not None:
        cmd.extend(["--voltage-step", str(voltage_step)])
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])
    if target_voltage is not None:
        cmd.extend(["--target-voltage", str(target_voltage)])
    run_command(cmd, cwd=cwd)
    if not out_csv.exists():
        raise FileNotFoundError(f"Voltage results missing: {out_csv}")


def filter_voltage(stable_csv: Path, voltage_csv: Path, final_csv: Path,
                   audit_csv: Path, target_voltage: float | None = None,
                   min_window: float = 0.0) -> int:
    """Fail closed on missing/failed results and record every gate decision."""
    if not math.isfinite(min_window) or min_window < 0:
        raise ValueError("min_window must be finite and nonnegative")
    with stable_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        stable_rows, stable_fields = list(reader), reader.fieldnames or []
    with voltage_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        voltage_rows, voltage_fields = list(reader), reader.fieldnames or []
    by_file = {}
    for row in voltage_rows:
        key = row.get("file")
        if key in by_file:
            raise RuntimeError(f"Duplicate voltage result for {key}")
        by_file[key] = row
    fields = list(dict.fromkeys(stable_fields + voltage_fields + ["passes_voltage_filter", "voltage_filter_reason"]))
    audit, kept = [], []
    for row in stable_rows:
        voltage = by_file.get(row.get("file"))
        combined = {**row, **(voltage or {})}
        reason = "passed"
        if voltage is None:
            reason = "missing_voltage_result"
        elif voltage.get("window_status") not in ("stable_window", "scan_censored"):
            reason = voltage.get("window_status") or "missing_window_status"
        else:
            try:
                width = float(voltage.get("window", "nan"))
            except (TypeError, ValueError):
                width = math.nan
            if not math.isfinite(width) or width <= 0:
                reason = "no_verified_nonzero_window"
            elif width < min_window:
                reason = "window_below_minimum"
            elif target_voltage is not None:
                try:
                    evaluated_target = float(voltage.get("target_voltage", "nan"))
                except (TypeError, ValueError):
                    evaluated_target = math.nan
                if not math.isfinite(evaluated_target) or not math.isclose(evaluated_target, target_voltage, abs_tol=1e-9, rel_tol=0):
                    reason = "missing_target_evaluation"
                elif str(voltage.get("stable_at_target", "")).lower() not in ("true", "1"):
                    reason = "unstable_at_target"
        combined["passes_voltage_filter"] = reason == "passed"
        combined["voltage_filter_reason"] = reason
        audit.append(combined)
        if reason == "passed":
            kept.append(combined)
    for path, rows in ((final_csv, kept), (audit_csv, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"[INFO] Voltage gate: {len(kept)}/{len(stable_rows)} candidates retained -> {final_csv}")
    return len(kept)


def main(argv=None, *, legacy_defaults=False) -> None:
    args = parse_args(argv, legacy_defaults=legacy_defaults)
    workdir = args.workdir.resolve()
    os.chdir(workdir)
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
    if args.selection_audit is None:
        args.selection_audit = output_dir / "selection_audit.csv"
    if args.final_out is None:
        args.final_out = output_dir / "final_candidates.csv"
    if args.voltage_filter_audit is None:
        args.voltage_filter_audit = output_dir / "voltage_filter_audit.csv"

    os.makedirs(args.export_dir, exist_ok=True)
    os.chdir(workdir)

    rows = read_stage2_rows(args.stage2_csv)
    from candidate_selection import select_candidates, write_selection_audit
    selection = select_candidates(rows, args.topk, selection_mode=args.selection_mode, base_dir=workdir)
    write_selection_audit(selection, args.selection_audit)
    (output_dir / "selection_summary.json").write_text(
        json.dumps({"counts": selection.counts, "metadata": selection.metadata}, indent=2) + "\n"
    )
    print(f"[INFO] Candidate selection ({args.selection_mode}): {selection.counts}")
    refs = selection.refs
    write_refs(refs, args.refs_out)
    if not refs:
        if rows:
            raise RuntimeError("No valid references were found in the stage2 CSV; see selection_audit.csv.")
        write_empty_csv(args.export_dir / args.export_index_name,
                        ("ref", "cif", "poscar", "formula", "spg", "n_atoms", "density"))
        if not args.dry_run:
            write_empty_csv(args.ehull_out, ("file", "path", "energy_above_hull_eV"))
            write_empty_csv(args.filtered_out, ("file", "path", "energy_above_hull_eV"))
            write_empty_csv(args.voltage_out, ("file", "window_status", "window"))
            for path in (args.final_out, args.voltage_filter_audit):
                write_empty_csv(path, ("file", "window_status", "window", "passes_voltage_filter", "voltage_filter_reason"))
        print("[INFO] Chemical screening retained no candidates; subsequent stages skipped.")
        return

    index_path = run_export(args.export_script, args.refs_out, args.export_dir, args.export_prefix, cwd=workdir)
    renamed_index = rename_index(index_path, args.export_index_name)

    if args.dry_run:
        print("[INFO] Dry run requested; skipping hull and voltage computations.")
        return

    run_ehull(args.ehull_script, args.export_dir, args.ehull_out, cwd=workdir, index_csv=renamed_index)
    kept = filter_hull(args.ehull_out, args.filtered_out, args.ehull_threshold)
    if kept == 0:
        print("[WARN] No structures met the hull criterion; voltage window step will be skipped.")
        with args.filtered_out.open(newline="") as fh:
            fieldnames = csv.DictReader(fh).fieldnames or []
        write_empty_csv(args.voltage_out, ("file", "window_status", "window"))
        for path in (args.final_out, args.voltage_filter_audit):
            write_empty_csv(path, list(dict.fromkeys(fieldnames + ["window_status", "window", "passes_voltage_filter", "voltage_filter_reason"])))
        return

    run_voltage(args.voltage_script, args.filtered_out, args.voltage_out, args.voltage_step, args.voltage_threshold, cwd=workdir, target_voltage=args.target_voltage)
    filter_voltage(args.filtered_out, args.voltage_out, args.final_out, args.voltage_filter_audit,
                   target_voltage=args.target_voltage, min_window=args.min_voltage_window)
    print(f"[INFO] Voltage window results saved to {args.voltage_out}")
    print(f"[INFO] Export index located at {renamed_index}")


if __name__ == "__main__":
    main()
