#!/usr/bin/env python3
"""Compute ΔE_hull for CIF structures using CHGNet energies.

This script scans a directory of CIF files, predicts their total energies with CHGNet,
fetches competing structures from the Materials Project, evaluates the convex hull,
and reports the energy above hull (ΔE_hull) together with a stability flag.

python compute_ehull_chgnet.py --cif-dir results/exported_cifs --out chgnet_hull_results.csv
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from time import sleep
from typing import Dict, Iterable, List, Optional, Tuple

from chgnet.model import CHGNet
from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core import Structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ΔE_hull for CIF files with CHGNet.")
    parser.add_argument(
        "--cif-dir",
        default="results/exported_cifs",
        help="Directory containing CIF files to evaluate.",
    )
    parser.add_argument(
        "--out",
        default="chgnet_hull_results.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--mp-api-key",
        default=os.environ.get("MP_API_KEY"),
        help="Materials Project API key. Defaults to the MP_API_KEY environment variable.",
    )
    parser.add_argument(
        "--max-mp-competitors",
        type=int,
        default=None,
        help="Optional upper bound on the number of MP competitor structures per chemical system.",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=1e-3,
        help="ΔE_hull threshold (eV/atom) for marking a structure as stable.",
    )
    return parser.parse_args()


def get_energy_per_atom(pred: Dict) -> float:
    """Extract eV/atom from CHGNet predictions produced by predict_structure."""
    for key in ("e", "energy"):
        if key in pred:
            return float(pred[key])
    if hasattr(pred, "e"):
        return float(pred.e)
    if hasattr(pred, "energy"):
        return float(pred.energy)
    raise KeyError("CHGNet prediction does not contain an energy per atom.")


def load_candidate_structures(cif_dir: Path) -> List[Tuple[Path, Structure]]:
    structures: List[Tuple[Path, Structure]] = []
    for path in sorted(cif_dir.glob("*.cif")):
        try:
            structures.append((path, Structure.from_file(path)))
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
    return structures


def chem_system(struct: Structure) -> str:
    return "-".join(sorted({el.symbol for el in struct.composition.elements}))


def search_with_retry(mpr: MPRester, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            return list(mpr.materials.summary.search(**kwargs))
        except MPRestError as exc:
            sleep(1.5 * (attempt + 1))
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("search_with_retry failed without raising MPRestError")


def fetch_mp_competitor_structures(
    chemsys: str,
    mpr: MPRester,
    limit: Optional[int] = None,
) -> List[Structure]:
    elems = chemsys.split("-")
    collected = []
    seen = set()

    def add_docs(docs: Iterable) -> None:
        nonlocal collected
        for doc in docs:
            mid = getattr(doc, "material_id", None)
            if mid in seen:
                continue
            struct = getattr(doc, "structure", None)
            if struct is None:
                continue
            collected.append(struct)
            seen.add(mid)
            if limit is not None and len(collected) >= limit:
                return

    # Ensure unary references first
    for el in elems:
        docs = search_with_retry(
            mpr,
            chemsys=el,
            fields=["material_id", "structure", "is_stable"],
        )
        pick = next((d for d in docs if getattr(d, "is_stable", False)), docs[0] if docs else None)
        if pick is not None:
            add_docs([pick])
        if limit is not None and len(collected) >= limit:
            return collected

    # Add higher-order subsystems, stable first
    for k in range(2, len(elems) + 1):
        for combo in combinations(elems, k):
            sub_csys = "-".join(sorted(combo))
            docs = search_with_retry(
                mpr,
                chemsys=[sub_csys],
                fields=["material_id", "structure", "is_stable"],
            )
            stable = [d for d in docs if getattr(d, "is_stable", False)]
            others = [d for d in docs if not getattr(d, "is_stable", False)]
            add_docs(stable + others)
            if limit is not None and len(collected) >= limit:
                return collected

    return collected


def structures_to_entries(structs: Iterable[Structure], model: CHGNet) -> List[PDEntry]:
    entries: List[PDEntry] = []
    for struct in structs:
        try:
            pred = model.predict_structure(struct)
            e_pa = get_energy_per_atom(pred)
            if not (isinstance(e_pa, (int, float)) and math.isfinite(e_pa)):
                print("[WARN] Non-finite CHGNet energy for competitor; skipping structure.")
                continue
            e_tot = e_pa * len(struct)
            entries.append(PDEntry(struct.composition, e_tot))
        except Exception as exc:
            print(f"[WARN] CHGNet failed for competitor structure: {exc}")
    return entries


def main() -> None:
    args = parse_args()

    if not args.mp_api_key:
        raise SystemExit("MP API key not provided. Use --mp-api-key or set MP_API_KEY.")

    cif_dir = Path(args.cif_dir).expanduser().resolve()
    if not cif_dir.is_dir():
        raise SystemExit(f"CIF directory not found: {cif_dir}")

    candidates = load_candidate_structures(cif_dir)
    if not candidates:
        raise SystemExit(f"No CIF files found in {cif_dir}")

    print(f"[INFO] Loaded {len(candidates)} CIF files from {cif_dir}")

    model = CHGNet.load()
    print("[INFO] CHGNet model loaded.")

    grouped_entries: Dict[str, List[PDEntry]] = defaultdict(list)
    grouped_meta: Dict[str, List[Dict]] = defaultdict(list)

    for path, struct in candidates:
        try:
            pred = model.predict_structure(struct)
            e_pa = get_energy_per_atom(pred)
            if not (isinstance(e_pa, (int, float)) and math.isfinite(e_pa)):
                print(f"[WARN] Non-finite energy for {path}, skipping.")
                continue
            e_tot = e_pa * len(struct)
            entry = PDEntry(struct.composition, e_tot)
            csys = chem_system(struct)
            grouped_entries[csys].append(entry)
            grouped_meta[csys].append(
                {
                    "file": path.name,
                    "path": str(path),
                    "formula": struct.composition.reduced_formula,
                    "chemsys": csys,
                    "natoms": len(struct),
                    "energy_per_atom": e_pa,
                    "energy_total": e_tot,
                    "entry": entry,
                }
            )
        except Exception as exc:
            print(f"[WARN] Failed CHGNet prediction for {path}: {exc}")

    if not grouped_entries:
        raise SystemExit("No valid candidate entries were generated.")

    results: List[Dict] = []

    with MPRester(api_key=args.mp_api_key) as mpr:
        for csys, my_entries in grouped_entries.items():
            print(f"[INFO] Processing chemical system {csys} ({len(my_entries)} candidates)")
            mp_structs = fetch_mp_competitor_structures(csys, mpr, args.max_mp_competitors)
            if not mp_structs:
                print(f"[WARN] No competitor structures found for {csys}; skipping system.")
                continue
            mp_entries = structures_to_entries(mp_structs, model)
            if not mp_entries:
                print(f"[WARN] Failed to obtain CHGNet energies for competitors in {csys}; skipping.")
                continue

            all_elems = set(csys.split("-"))
            elem_in_comp = set()
            for entry in mp_entries:
                elem_in_comp.update({el.symbol for el in entry.composition.elements})
            missing = all_elems - elem_in_comp
            if missing:
                print(f"[WARN] Missing elemental references {sorted(missing)} for {csys}; skipping system.")
                continue

            try:
                pd = PhaseDiagram(mp_entries + my_entries)
            except Exception as exc:
                print(f"[WARN] Phase diagram failed for {csys}: {exc}")
                continue

            for meta in grouped_meta[csys]:
                entry = meta["entry"]
                try:
                    ehull = float(pd.get_e_above_hull(entry))
                    fe_pa = float(pd.get_form_energy_per_atom(entry))
                except Exception as exc:
                    print(f"[WARN] Failed to compute hull metrics for {meta['file']}: {exc}")
                    continue
                results.append(
                    {
                        "file": meta["file"],
                        "path": meta["path"],
                        "formula": meta["formula"],
                        "chemsys": meta["chemsys"],
                        "natoms_cell": meta["natoms"],
                        "energy_per_atom_eV": meta["energy_per_atom"],
                        "energy_total_eV": meta["energy_total"],
                        "formation_energy_per_atom_eV": fe_pa,
                        "energy_above_hull_eV": ehull,
                        "is_stable": int(ehull <= args.stable_threshold),
                    }
                )

    if not results:
        raise SystemExit("No hull results were produced; see warnings above.")

    fieldnames = [
        "file",
        "path",
        "formula",
        "chemsys",
        "natoms_cell",
        "energy_per_atom_eV",
        "energy_total_eV",
        "formation_energy_per_atom_eV",
        "energy_above_hull_eV",
        "is_stable",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[INFO] Wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
