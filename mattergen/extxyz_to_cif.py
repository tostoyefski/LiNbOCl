#!/usr/bin/env python3
"""
Convert one or more frames from an .extxyz file into CIF files.

Usage:
    python extxyz_to_cif.py --input path/to/structure.extxyz --outdir output_dir
    python extxyz_to_cif.py --input structure.extxyz --frames 0 5 7 --prefix sample

By default, all frames are exported. Use --frames to specify specific frame indices (0-based).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frames from an extxyz file to CIF.")
    parser.add_argument("--input", required=True, help="Path to the .extxyz file.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("exported_cifs_from_extxyz"),
        help="Directory where CIF files will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Prefix used for exported CIF filenames.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        type=int,
        default=None,
        help="Specific frame indices to export (0-based). Default: all frames.",
    )
    parser.add_argument(
        "--use-ase-writer",
        action="store_true",
        help="Use ASE writer instead of pymatgen's CifWriter (fallback).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.frames is None:
            atoms_list = ase_read(str(input_path), index=":")
        else:
            atoms_list = [ase_read(str(input_path), index=i) for i in args.frames]
    except Exception as exc:
        raise SystemExit(f"Failed to read {input_path}: {exc}")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    for idx, atoms in enumerate(atoms_list):
        frame_idx = args.frames[idx] if args.frames is not None else idx
        filename = f"{args.prefix}_{frame_idx:04d}.cif"
        cif_path = args.outdir / filename

        try:
            if args.use_ase_writer:
                ase_write(str(cif_path), atoms, format="cif")
            else:
                structure = AseAtomsAdaptor().get_structure(atoms)
                CifWriter(structure, symprec=None).write_file(str(cif_path))
        except Exception as exc:
            print(f"[WARN] Failed to write CIF for frame {frame_idx}: {exc}")
            continue

        print(f"[OK] Exported frame {frame_idx} -> {cif_path}")


if __name__ == "__main__":
    main()
