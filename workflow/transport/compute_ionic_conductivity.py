#!/usr/bin/env python3
"""Run CHGNet MD only for candidates accepted by the hull/voltage pipeline."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from pymatgen.core import Structure

E_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23  # J/K
REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSPORT_RESULTS = REPO_ROOT / "results" / "transport"


def load_md_candidates(csv_path: Path | str) -> List[Dict[str, str]]:
    """Validate final-selection metadata before loading an MD model."""
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "passes_voltage_filter", "window_status", "window"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "MD input must be the audited final_candidates.csv; "
                f"missing columns: {sorted(missing)}"
            )
        rows = list(reader)
    accepted = []
    for row in rows:
        reason = None
        if (row.get("passes_voltage_filter") or "").strip().lower() not in {"true", "1", "yes"}:
            reason = "passes_voltage_filter is not true"
        elif row.get("window_status") not in {"stable_window", "scan_censored"}:
            reason = f"ineligible window_status={row.get('window_status')}"
        elif (row.get("error") or "").strip():
            reason = row["error"]
        elif (row.get("stable_at_target") or "").strip().lower() in {"false", "0", "no"}:
            reason = "candidate is unstable at target_voltage"
        elif not (row.get("file") or "").strip():
            reason = "missing CIF filename"
        else:
            try:
                width = float(row["window"])
                if not math.isfinite(width) or width <= 0:
                    reason = "window must be finite and positive"
                target = (row.get("target_voltage") or "").strip()
                if target:
                    if not math.isfinite(float(target)):
                        reason = "target_voltage must be finite"
                    elif (row.get("stable_at_target") or "").strip().lower() not in {"true", "1", "yes"}:
                        reason = "candidate is not confirmed stable at target_voltage"
            except (TypeError, ValueError):
                reason = "invalid numeric voltage metadata"
        if reason:
            print(f"[REJECT] {row.get('file') or '(unnamed candidate)'}: {reason}")
        else:
            accepted.append(row)
    return accepted

def candidate_cif_path(row: Dict[str, str], csv_path: Path | str, cif_dir: Path | str) -> Path:
    """Use the audited geometry; never replace a missing optimized CIF silently."""
    recorded = (row.get("path") or "").strip()
    if not recorded:
        return Path(cif_dir) / row["file"]
    path = Path(recorded).expanduser()
    if not path.is_absolute():
        path = Path(csv_path).resolve().parent / path
    if path.name != row["file"] or path.suffix.lower() != ".cif":
        raise ValueError(f"Audited CIF path does not match candidate {row['file']}: {path}")
    return path


def run_md(struct: Structure, temp: float, timestep_fs: float,
           total_ps: float, replicate: int, calculator: CHGNetCalculator, model: CHGNet,
           log_interval: int, traj_path: Path, log_path: Path,
           rng: np.random.Generator | None = None) -> Tuple[Path, 'ase.Atoms']:
    import numpy as np
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from chgnet.model.dynamics import MolecularDynamics
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(struct)
    if replicate > 1:
        atoms = atoms.repeat((replicate, replicate, replicate))
    atoms.calc = calculator
    n_steps = int(total_ps * 1000 / timestep_fs)
    if rng is None:
        rng = np.random.default_rng()

    # Initialize independent velocities at target temperature.
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    md = MolecularDynamics(
        atoms=atoms,
        model=model,
        ensemble="nvt",
        temperature=temp,
        timestep=timestep_fs,
        trajectory=str(traj_path),
        logfile=str(log_path),
        loginterval=log_interval,
    )
    md.run(n_steps)
    return traj_path, atoms


def resolve_disorder(
    struct: Structure,
    strategy: str,
    sqs_scale: int,
    sqs_steps: int,
    max_order_factor: int,
    mcsqs_rcut: float,
    mcsqs_timeout: int,
) -> Structure:
    """Convert disordered structures to an ordered representation before MD.

    strategy: 'sqs', 'order', or 'skip'
    sqs_scale: isotropic supercell multiplier when attempting SQS
    sqs_steps: MC steps for SQS (if available)
    """
    from pymatgen.transformations.advanced_transformations import SQSTransformation
    from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

    if struct.is_ordered or strategy == "none":
        return struct

    if strategy == "skip":
        raise ValueError("Structure is disordered and strategy=skip")

    if strategy == "sqs":
        try:
            # Try pymatgen's SQSTransformation (uses mcsqs if present)
            scaling = [sqs_scale, sqs_scale, sqs_scale]
            sqs_trans = SQSTransformation(
                scaling=scaling,
                search_time=mcsqs_timeout,  # seconds
                sqs_method="mcsqs",
            )
            result = sqs_trans.apply_transformation(struct)
            if result is not None:
                return result
            raise RuntimeError("SQSTransformation returned None")
        except Exception as exc:
            # On SQS failure, abort as requested
            raise SystemExit(f"SQSTransformation failed: {exc}") from exc

    # Fallback: pick the best ordered structure generated by ODST, optionally on a larger supercell
    trans = OrderDisorderedStructureTransformation()
    max_factor = max(max_order_factor, sqs_scale)
    last_exc: Exception | None = None
    for factor in range(1, max_factor + 1):
        try_struct = struct.copy()
        if factor > 1:
            try_struct.make_supercell([factor, factor, factor])
        try:
            ordered_list = trans.apply_transformation(try_struct, return_ranked_list=1)
            ordered_struct = ordered_list[0]["structure"] if ordered_list else None
            if ordered_struct is not None and ordered_struct.is_ordered:
                return ordered_struct
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            continue

    msg = f"Failed to order disordered structure via fallback transformation (tried factors 1..{max_factor})."
    if last_exc:
        msg += f" Last error: {last_exc}"
    raise ValueError(msg)

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Estimate ionic conductivity via CHGNet MD.")
    ap.add_argument("--csv", default=str(REPO_ROOT / "results/top300_run/final_candidates.csv"), help="Audited CSV of candidates passing both hull and voltage gates.")
    ap.add_argument("--cif-dir", default=str(REPO_ROOT / "results/top300_run/exported_300cifs"))
    ap.add_argument("--mobile-element", default="Li")
    ap.add_argument("--temperature", type=float, default=700.0)
    ap.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated list of temperatures (K). Overrides --temperature to enable Arrhenius batch.",
    )
    ap.add_argument("--timestep-fs", type=float, default=2.0)
    ap.add_argument("--total-ps", type=float, default=10.0)
    ap.add_argument(
        "--equil-ps",
        type=float,
        default=0.0,
        help="Discard this many ps from the start of each trajectory before diffusion fit.",
    )
    ap.add_argument("--log-interval", type=int, default=10, help="write traj/log every N MD steps")
    ap.add_argument("--replicate", type=int, default=1, help="make supercell to reduce finite-size effects")
    ap.add_argument("--device", default="auto", help="cuda, cuda:0, cpu, or auto (default)")
    ap.add_argument(
        "--disorder-strategy",
        choices=["order", "sqs", "skip", "none"],
        default="order",
        help="How to handle disordered structures: order=ODST fallback, sqs=try SQS then order, skip=ignore, none=assume ordered.",
    )
    ap.add_argument("--sqs-scale", type=int, default=2, help="Isotropic scaling for SQS / fallback supercell")
    ap.add_argument("--sqs-steps", type=int, default=5000, help="MC steps when attempting SQS")
    ap.add_argument(
        "--max-ordering-factor",
        type=int,
        default=9,
        help="Max isotropic factor to try when ordering disordered structures (ODST fallback).",
    )
    ap.add_argument("--mcsqs-rcut", type=float, default=5.0, help="Pair cutoff (Ang) for mcsqs if used.")
    ap.add_argument("--mcsqs-timeout", type=int, default=300, help="Timeout (s) for mcsqs run.")
    ap.add_argument("--haven-ratio", type=float, default=1.0, help="Haven ratio for Nernst-Einstein")
    ap.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent MD runs per temperature; results are averaged.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed for velocity initialization; run index is added to differentiate runs.",
    )
    ap.add_argument("--traj-dir", default=str(TRANSPORT_RESULTS / "md_traj"), help="directory to store MD trajectories/logs")
    ap.add_argument("--out", default=str(TRANSPORT_RESULTS / "chgnet_ionic_conductivity.csv"))
    ap.add_argument(
        "--arrhenius-summary",
        default=None,
        help="Optional path to write ln(sigma*T) vs 1/T fit per structure when multiple temperatures are provided; defaults to <out> with _arrhenius suffix.",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        rows = load_md_candidates(args.csv)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not rows:
        print("[INFO] No eligible final candidates; no MD runs requested.")
        return

    import numpy as np
    import torch
    from ase import units
    from ase.io.trajectory import Trajectory
    from ase.md.analysis import DiffusionCoefficient
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from pymatgen.core import Structure

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WARN] Requested {device} but CUDA is unavailable; falling back to CPU.")
            device = "cpu"

    model = CHGNet.load(use_device=device)
    calculator = CHGNetCalculator(model=model, use_device=device)
    results = []
    total_rows = len(rows)
    traj_dir = Path(args.traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)
    if args.temperatures:
        temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    else:
        temps = [args.temperature]

    frames_to_skip = int(
        args.equil_ps * 1000.0 / (args.timestep_fs * args.log_interval)
    ) if args.equil_ps > 0 else 0
    for idx, row in enumerate(rows):
        try:
            cif_path = candidate_cif_path(row, args.csv, args.cif_dir)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue
        if not cif_path.exists():
            print(f"[WARN] missing CIF {cif_path}")
            continue
        try:
            struct = Structure.from_file(cif_path)
        except Exception as exc:
            print(f"[WARN] failed to load structure {cif_path}: {exc}")
            continue

        try:
            struct = resolve_disorder(
                struct,
                strategy=args.disorder_strategy,
                sqs_scale=args.sqs_scale,
                sqs_steps=args.sqs_steps,
                max_order_factor=args.max_ordering_factor,
                mcsqs_rcut=args.mcsqs_rcut,
                mcsqs_timeout=args.mcsqs_timeout,
            )
        except ValueError as exc:
            print(f"[WARN] Disordered structure handling failed for {row['file']}: {exc}")
            continue
        except Exception as exc:
            print(f"[WARN] Unexpected disorder handling error for {row['file']}: {exc}")
            continue

        for temp in temps:
            print(
                f"[INFO] Running {row['file']} ({idx + 1}/{total_rows}) at {temp} K with {args.total_ps} ps, "
                f"dt={args.timestep_fs} fs, replicate={args.replicate}, log_interval={args.log_interval}, "
                f"disorder_strategy={args.disorder_strategy}, runs={args.runs}, equil_ps={args.equil_ps}"
            )
            run_Ds = []
            run_sigmas = []
            traj_paths: List[str] = []
            for run_idx in range(args.runs):
                base_name = Path(row["file"]).stem
                if args.runs == 1 and len(temps) == 1:
                    traj_path = traj_dir / f"{base_name}.traj"
                    log_path = traj_dir / f"{base_name}.log"
                else:
                    temp_tag = f"T{int(temp)}K"
                    run_tag = f"run{run_idx + 1}"
                    traj_path = traj_dir / f"{base_name}_{temp_tag}_{run_tag}.traj"
                    log_path = traj_dir / f"{base_name}_{temp_tag}_{run_tag}.log"
                seed = None if args.seed is None else args.seed + run_idx + int(temp)
                rng = np.random.default_rng(seed)
                try:
                    traj_path, atoms = run_md(
                        struct, temp, args.timestep_fs, args.total_ps, args.replicate,
                        calculator, model, args.log_interval, traj_path, log_path, rng=rng
                    )
                except Exception as exc:
                    print(f"[WARN] MD failed for {row['file']} (run {run_idx + 1} at {temp} K): {exc}")
                    continue

                # Analyze diffusion from trajectory using ASE's DiffusionCoefficient
                try:
                    traj = Trajectory(traj_path)
                except Exception as exc:
                    print(f"[WARN] failed to read trajectory {traj_path}: {exc}")
                    continue
                if len(traj) < 2:
                    print(f"[WARN] trajectory too short for {row['file']} (run {run_idx + 1} at {temp} K)")
                    continue
                symbols = traj[0].get_chemical_symbols()
                mobile_indices = [i for i, s in enumerate(symbols) if s == args.mobile_element]
                if not mobile_indices:
                    print(f"[WARN] no {args.mobile_element} found in {row['file']}, skipping")
                    continue
                frame_dt_ase = args.timestep_fs * args.log_interval * units.fs

                if frames_to_skip > 0:
                    if frames_to_skip >= len(traj):
                        print(
                            f"[WARN] equilibration skip ({frames_to_skip} frames) >= trajectory length ({len(traj)}); "
                            f"skipping run {run_idx + 1} for {row['file']} at {temp} K"
                        )
                        continue
                    traj_slice = traj[frames_to_skip:]
                else:
                    traj_slice = traj

                try:
                    diff = DiffusionCoefficient(
                        traj=traj_slice, timestep=frame_dt_ase, atom_indices=mobile_indices, molecule=False
                    )
                    slopes, std = diff.get_diffusion_coefficients()
                    D_A2_per_ase = float(slopes[0])
                except Exception as exc:
                    print(f"[WARN] diffusion analysis failed for {row['file']} (run {run_idx + 1} at {temp} K): {exc}")
                    continue

                D_A2_per_fs = D_A2_per_ase * units.fs
                D_m2_s = D_A2_per_fs * 1.0e-5

                volume_A3 = traj_slice[-1].get_volume()
                volume_m3 = volume_A3 * 1.0e-30
                number_density = len(mobile_indices) / volume_m3 if volume_m3 > 0 else None
                if number_density is None:
                    print(f"[WARN] invalid volume for {row['file']} (run {run_idx + 1} at {temp} K), skipping")
                    continue
                sigma_S_m = args.haven_ratio * number_density * (E_CHARGE**2) * D_m2_s / (K_B * temp)

                traj_paths.append(str(traj_path))
                run_Ds.append(D_m2_s)
                run_sigmas.append(sigma_S_m)

            if not run_sigmas:
                print(f"[WARN] no valid runs for {row['file']} at {temp} K")
                continue

            D_mean = float(np.mean(run_Ds))
            D_std = float(np.std(run_Ds)) if len(run_Ds) > 1 else 0.0
            sigma_mean = float(np.mean(run_sigmas))
            sigma_std = float(np.std(run_sigmas)) if len(run_sigmas) > 1 else 0.0

            results.append(
                {
                    "file": row["file"],
                    "formula": row.get("formula"),
                    "chemsys": row.get("chemsys"),
                    "temperature_K": temp,
                    "runs": len(run_sigmas),
                    "D_m2_s_mean": D_mean,
                    "D_m2_s_std": D_std,
                    "sigma_S_m_mean": sigma_mean,
                    "sigma_S_m_std": sigma_std,
                    "sigma_mS_cm_mean": sigma_mean * 10 if sigma_mean is not None else None,  # 1 S/m = 10 mS/cm
                    "sigma_mS_cm_std": sigma_std * 10 if sigma_std is not None else None,
                    "traj_paths": ";".join(traj_paths),
                }
            )

    if not results:
        print("[WARN] no results produced; check warnings above.")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[INFO] wrote {len(results)} rows to {args.out}")

    # Optional Arrhenius analysis if multiple temperatures were evaluated.
    if len(set(temps)) > 1:
        summary_path = args.arrhenius_summary
        if summary_path is None:
            out_path = Path(args.out)
            summary_path = out_path.with_name(f"{out_path.stem}_arrhenius.csv")

        grouped: Dict[str, List[Dict[str, float]]] = {}
        for r in results:
            if "temperature_K" not in r:
                continue
            file_key = r["file"]
            grouped.setdefault(file_key, []).append(r)

        summary_rows = []
        for file_key, vals in grouped.items():
            # Need at least two temperature points to fit.
            temps_K = []
            sigmas = []
            for v in vals:
                temp_val = v.get("temperature_K")
                sigma_val = v.get("sigma_S_m_mean")
                if temp_val is None or sigma_val is None:
                    continue
                if sigma_val <= 0:
                    continue
                temps_K.append(float(temp_val))
                sigmas.append(float(sigma_val))
            if len(temps_K) < 2:
                continue

            temps_K = np.array(temps_K, dtype=float)
            sigmas = np.array(sigmas, dtype=float)
            x = 1.0 / temps_K  # 1/K
            y = np.log(sigmas * temps_K)  # ln(sigma*T)
            if not np.all(np.isfinite(y)):
                continue
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            Ea_J = -slope * K_B
            Ea_eV = Ea_J / E_CHARGE
            prefactor_sigmaT = float(np.exp(intercept))

            summary_rows.append(
                {
                    "file": file_key,
                    "n_points": len(temps_K),
                    "slope_ln_sigmaT_vs_invT": slope,
                    "intercept": intercept,
                    "Ea_eV": Ea_eV,
                    "prefactor_sigmaT_S_m_K": prefactor_sigmaT,
                    "r2": r2,
                    "temps_K": ";".join(f"{t:.1f}" for t in temps_K),
                    "sigma_S_m_mean": ";".join(f"{s:.4e}" for s in sigmas),
                }
            )

        if summary_rows:
            Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                for r in summary_rows:
                    writer.writerow(r)
            print(f"[INFO] wrote Arrhenius summary for {len(summary_rows)} structures to {summary_path}")
        else:
            print("[WARN] Arrhenius summary skipped (insufficient temperature points or invalid data).")

if __name__ == "__main__":
    main()
