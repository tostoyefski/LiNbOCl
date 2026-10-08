#!/usr/bin/env python3
import argparse
import math
import sys
from glob import glob
from pathlib import Path

import numpy as np
from ase.io import write
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
    return tokens if tokens else ["Li"]


def gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return data
    nx, ny, nz = data.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kz = np.fft.fftfreq(nz)
    k2 = (
        kx[:, None, None] ** 2
        + ky[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    kernel = np.exp(-2.0 * (math.pi ** 2) * (sigma ** 2) * k2)
    return np.fft.ifftn(np.fft.fftn(data) * kernel).real


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Li probability density grid from ASE trajectories."
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
        "--output",
        type=Path,
        default=Path("li_density.cube"),
        help="Output volumetric file (.cube).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=100,
        help="Grid size along each axis (e.g., 100).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma in grid points (0 to disable).",
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
        "--species",
        default="Li",
        help="Species to include (comma or space separated).",
    )
    args = parser.parse_args()

    traj_paths = collect_traj_paths(args.traj, args.glob)
    if not traj_paths:
        print("No trajectory files provided.", file=sys.stderr)
        sys.exit(1)

    species = set(parse_species(args.species))
    grid_n = args.grid
    if grid_n <= 0:
        print("Grid size must be positive.", file=sys.stderr)
        sys.exit(1)

    density = np.zeros((grid_n, grid_n, grid_n), dtype=np.float64)
    total_samples = 0
    atoms_ref = None
    ref_cell = None

    for traj_path in traj_paths:
        traj = Trajectory(traj_path)
        for idx, atoms in enumerate(traj):
            if idx < args.discard:
                continue
            if (idx - args.discard) % args.stride != 0:
                continue
            if atoms_ref is None:
                atoms_ref = atoms.copy()
                atoms_ref.pbc = True
                ref_cell = atoms_ref.cell.array.copy()
            elif ref_cell is not None:
                diff = np.abs(atoms.cell.array - ref_cell).max()
                if diff > 1e-5:
                    print(f"Warning: cell differs in {traj_path}", file=sys.stderr)
                    ref_cell = None

            symbols = np.array(atoms.get_chemical_symbols())
            mask = np.isin(symbols, list(species))
            if not np.any(mask):
                continue
            scaled = atoms.get_scaled_positions(wrap=True)
            positions = scaled[mask]
            indices = np.floor(positions * grid_n).astype(int) % grid_n
            np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), 1.0)
            total_samples += len(indices)

    if atoms_ref is None or total_samples == 0:
        print("No valid samples found in trajectories.", file=sys.stderr)
        sys.exit(1)

    if args.sigma > 0:
        density = gaussian_smooth(density, args.sigma)

    voxel_volume = atoms_ref.get_volume() / float(grid_n ** 3)
    density = density / float(total_samples) / voxel_volume

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms_ref, data=density)
    print(f"Wrote density grid to {output_path} using {total_samples} samples.")


if __name__ == "__main__":
    main()
