#!/usr/bin/env python3
import argparse
import csv
import sys
from glob import glob
from pathlib import Path

TRANSPORT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "transport"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.geometry import find_mic
from ase.io.trajectory import Trajectory


def collect_traj_paths(traj_args: list[str], glob_args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in traj_args:
        paths.append(Path(item))
    for pattern in glob_args:
        for match in glob(pattern, recursive=True):
            paths.append(Path(match))
    unique = sorted({path.resolve() for path in paths})
    return unique


def parse_species(raw: str) -> list[str]:
    tokens = [tok.strip() for tok in raw.replace(",", " ").split() if tok.strip()]
    if not tokens:
        tokens = ["Li"]
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def msd_from_traj(
    traj_path: Path,
    species_list: list[str],
    frame_dt_fs: float,
    discard: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    traj = Trajectory(traj_path)
    if len(traj) < 2:
        return None

    symbols = traj[0].get_chemical_symbols()
    species_indices = {
        spec: [i for i, s in enumerate(symbols) if s == spec]
        for spec in species_list
    }
    species_indices = {spec: idxs for spec, idxs in species_indices.items() if idxs}
    if not species_indices:
        return None

    times_fs: list[float] = []
    state = {
        spec: {
            "indices": idxs,
            "origin": None,
            "prev": None,
            "unwrapped": None,
            "msd": [],
        }
        for spec, idxs in species_indices.items()
    }

    for idx, atoms in enumerate(traj):
        if idx < discard:
            continue
        if state[next(iter(state))]["origin"] is None:
            for spec, spec_state in state.items():
                pos = atoms.get_positions()[spec_state["indices"]]
                spec_state["origin"] = pos.copy()
                spec_state["prev"] = pos.copy()
                spec_state["unwrapped"] = pos.copy()
            if (idx - discard) % stride == 0:
                times_fs.append(0.0)
                for spec_state in state.values():
                    spec_state["msd"].append(0.0)
            continue

        for spec_state in state.values():
            pos = atoms.get_positions()[spec_state["indices"]]
            delta = pos - spec_state["prev"]
            delta_mic, _ = find_mic(delta, atoms.cell, pbc=atoms.pbc)
            spec_state["unwrapped"] = spec_state["unwrapped"] + delta_mic
            spec_state["prev"] = pos

        if (idx - discard) % stride != 0:
            continue

        times_fs.append((idx - discard) * frame_dt_fs)
        for spec_state in state.values():
            disp = spec_state["unwrapped"] - spec_state["origin"]
            msd = np.mean(np.sum(disp ** 2, axis=1))
            spec_state["msd"].append(float(msd))

    if len(times_fs) < 2:
        return None

    result = {spec: np.array(spec_state["msd"]) for spec, spec_state in state.items()}
    return np.array(times_fs), result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MSD vs time from ASE trajectories."
    )
    parser.add_argument(
        "--traj",
        action="append",
        default=[],
        help="Trajectory file path (repeatable).",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern for trajectory files (repeatable, supports **).",
    )
    parser.add_argument(
        "--species",
        default="Li",
        help="Species to include (comma or space separated).",
    )
    parser.add_argument(
        "--frame-dt-fs",
        type=float,
        default=None,
        help="Time between frames in fs.",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=None,
        help="MD timestep in fs (used with --log-interval).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="MD log interval (used with --timestep-fs).",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=0,
        help="Discard first N frames of each trajectory.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sample every k frames after discard.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=TRANSPORT_RESULTS / "msd.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=TRANSPORT_RESULTS / "msd.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature label for the plot (K).",
    )
    args = parser.parse_args()

    if args.frame_dt_fs is None:
        if args.timestep_fs is None or args.log_interval is None:
            print("Provide --frame-dt-fs or both --timestep-fs and --log-interval.", file=sys.stderr)
            sys.exit(1)
        frame_dt_fs = args.timestep_fs * args.log_interval
    else:
        frame_dt_fs = args.frame_dt_fs

    traj_paths = collect_traj_paths(args.traj, args.glob)
    if not traj_paths:
        print("No trajectory files provided.", file=sys.stderr)
        sys.exit(1)

    species_list = parse_species(args.species)
    times_by_spec: dict[str, list[np.ndarray]] = {spec: [] for spec in species_list}
    msd_by_spec: dict[str, list[np.ndarray]] = {spec: [] for spec in species_list}

    for traj_path in traj_paths:
        result = msd_from_traj(
            traj_path,
            species_list,
            frame_dt_fs,
            args.discard,
            args.stride,
        )
        if result is None:
            print(f"Skipping {traj_path} (no data).", file=sys.stderr)
            continue
        times_fs, msd_map = result
        for spec, msd_vals in msd_map.items():
            times_by_spec.setdefault(spec, []).append(times_fs)
            msd_by_spec.setdefault(spec, []).append(msd_vals)

    if not any(msd_by_spec.values()):
        print("No MSD data computed.", file=sys.stderr)
        sys.exit(1)

    results = {}
    for spec, runs in msd_by_spec.items():
        if not runs:
            continue
        min_len = min(len(msd) for msd in runs)
        msd_stack = np.stack([msd[:min_len] for msd in runs], axis=0)
        times_fs = times_by_spec[spec][0][:min_len]
        msd_mean = msd_stack.mean(axis=0)
        msd_std = msd_stack.std(axis=0) if msd_stack.shape[0] > 1 else np.zeros_like(msd_mean)
        results[spec] = {
            "times_ps": times_fs / 1000.0,
            "mean": msd_mean,
            "std": msd_std,
            "n_runs": msd_stack.shape[0],
        }

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "time_ps", "msd_A2_mean", "msd_A2_std", "n_runs"])
        for spec in species_list:
            if spec not in results:
                continue
            data = results[spec]
            for t, mean, std in zip(data["times_ps"], data["mean"], data["std"]):
                writer.writerow([spec, f"{t:.6f}", f"{mean:.6f}", f"{std:.6f}", data["n_runs"]])

    fig, ax = plt.subplots(figsize=(6, 4))
    color_cycle = plt.cm.tab10.colors
    for idx, spec in enumerate(species_list):
        if spec not in results:
            continue
        data = results[spec]
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(data["times_ps"], data["mean"], color=color, linewidth=1.6, label=spec)
        if np.any(data["std"] > 0):
            ax.fill_between(
                data["times_ps"],
                data["mean"] - data["std"],
                data["mean"] + data["std"],
                color=color,
                alpha=0.2,
            )
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (A^2)")
    title = f"MSD ({', '.join(species_list)})"
    if args.temperature is not None:
        title += f" at {args.temperature:.0f} K"
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    plt.close(fig)

    run_counts = [results[spec]["n_runs"] for spec in results]
    runs_note = f"{min(run_counts)}-{max(run_counts)}" if run_counts else "0"
    print(f"Wrote {args.plot} and {args.csv} using {runs_note} run(s).")


if __name__ == "__main__":
    main()
