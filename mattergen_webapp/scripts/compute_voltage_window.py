#!/usr/bin/env python3
"""Evaluate high-voltage stability windows for CHGNet-screened candidates.

The workflow consumes a CSV (e.g. ``chgnet_hull_results_stable.csv``) and, for each
entry, assembles the relevant competing phases in the same chemical system. All
energies are (re-)evaluated on a consistent CHGNet baseline so that formation
energies and grand-potential hulls can be compared fairly. A voltage grid is then
scanned to determine the first oxidation (V_ox) and reduction (V_red) limits where
the candidate becomes unstable, and the resulting electrochemical window width is
reported.

Assumptions:
    * A Materials Project API key is available via ``MP_API_KEY`` or
      ``--mp-api-key``.
    * CHGNet is installed and accessible in the current Python environment.
    * The working redox element (default Li) is present in the candidate and the
      competing phase set includes a unary reference for that element.

Usage example::

    python compute_voltage_window.py \
        --stable-csv chgnet_hull_results_stable.csv \
        --working-element Li \
        --voltage-min 0.0 --voltage-max 6.0 --voltage-step 0.05 \
        --out chgnet_voltage_window.csv

"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chgnet.model import CHGNet
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PDEntry

try:  # compat across pymatgen versions
    from pymatgen.analysis.phase_diagram import GrandPotentialPDEntry as _GrandPotentialPDEntry
except ImportError:
    from pymatgen.analysis.phase_diagram import GrandPotPDEntry as _GrandPotentialPDEntry

from pymatgen.core import Element, Structure

from compute_ehull_chgnet import fetch_mp_competitor_structures, structures_to_entries


DEFAULT_THRESHOLD = 1e-3  # eV/atom stability tolerance when evaluating grand hulls


def make_grand_entry(entry: PDEntry, chempots: Dict[Element, float]):
    """Compat helper to build grand-potential entries across pymatgen versions."""
    if hasattr(_GrandPotentialPDEntry, "from_pd_entry"):
        return _GrandPotentialPDEntry.from_pd_entry(entry, chempots)
    return _GrandPotentialPDEntry(entry, chempots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute oxidation/reduction voltage limits for CHGNet candidates."
    )
    parser.add_argument(
        "--stable-csv",
        default="chgnet_hull_results_stable.csv",
        help="CSV containing the filtered stable candidates (must include path & energy columns).",
    )
    parser.add_argument(
        "--mp-api-key",
        default=os.environ.get("MP_API_KEY"),
        help="Materials Project API key.",
    )
    parser.add_argument(
        "--working-element",
        default="Li",
        help="Element symbol used as the electrochemical working ion (default: Li).",
    )
    parser.add_argument(
        "--voltage-min",
        type=float,
        default=0.0,
        help="Lower bound of the voltage grid (V).",
    )
    parser.add_argument(
        "--voltage-max",
        type=float,
        default=6.0,
        help="Upper bound of the voltage grid (V).",
    )
    parser.add_argument(
        "--voltage-step",
        type=float,
        default=0.05,
        help="Voltage increment for scanning the electrochemical window (V).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Energy-above-hull tolerance (eV/atom) for declaring instability on the grand potential hull.",
    )
    parser.add_argument(
        "--max-mp-competitors",
        type=int,
        default=None,
        help="Optional cap on MP competitor structures per chemical system (helps control runtime).",
    )
    parser.add_argument(
        "--out",
        default="chgnet_voltage_window.csv",
        help="Output CSV file with voltage window metrics.",
    )
    return parser.parse_args()


def load_stable_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}.")
    required = {"path", "chemsys", "energy_total_eV"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")
    return rows


def generate_voltage_grid(vmin: float, vmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("voltage-step must be positive")
    if vmax <= vmin:
        raise ValueError("voltage-max must exceed voltage-min")
    count = int(math.floor((vmax - vmin) / step))
    grid = [vmin + i * step for i in range(count + 1)]
    if grid[-1] < vmax - 1e-9:
        grid.append(vmax)
    return grid


@dataclass
class Candidate:
    row: Dict[str, str]
    structure: Structure
    entry: PDEntry


def build_candidates(rows: Sequence[Dict[str, str]]) -> List[Candidate]:
    candidates: List[Candidate] = []
    for row in rows:
        cif_path = Path(row["path"]).expanduser()
        if not cif_path.exists():
            print(f"[WARN] CIF path missing for {row.get('file', cif_path.name)}: {cif_path}")
            continue
        try:
            structure = Structure.from_file(cif_path)
        except Exception as exc:
            print(f"[WARN] Failed to load structure {cif_path}: {exc}")
            continue
        try:
            energy_total = float(row["energy_total_eV"])
        except (TypeError, ValueError):
            print(f"[WARN] Invalid energy_total_eV for {cif_path}, skipping candidate.")
            continue
        entry = PDEntry(structure.composition, energy_total)
        candidates.append(Candidate(row=row, structure=structure, entry=entry))
    return candidates


def locate_reference_mu(entries: Iterable[PDEntry], element: Element) -> Optional[float]:
    mu: Optional[float] = None
    for entry in entries:
        elems = {el.symbol for el in entry.composition.elements}
        if elems == {element.symbol}:
            e_pa = entry.energy_per_atom
            if mu is None or e_pa < mu:
                mu = e_pa
    return mu


def evaluate_voltage_window(
    candidate: Candidate,
    base_entries: Sequence[PDEntry],
    candidate_index: int,
    work_element: Element,
    mu_ref: float,
    voltages: Sequence[float],
    energy_tol: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    stability: List[bool] = []

    for V in voltages:
        mu = mu_ref - V  # μ(V) = μ_ref - eV (with e = 1 in eV units)
        chempots = {work_element: mu}
        gp_entries = [make_grand_entry(entry, chempots) for entry in base_entries]
        try:
            gppd = GrandPotentialPhaseDiagram(gp_entries, chempots)
        except Exception as exc:
            print(
                f"[WARN] Failed to build grand potential diagram for {candidate.row.get('file')} "
                f"at {V:.3f} V: {exc}"
            )
            stability.append(False)
            continue

        gp_entry = gp_entries[candidate_index]
        try:
            e_above = float(gppd.get_e_above_hull(gp_entry))
        except Exception as exc:
            print(
                f"[WARN] Could not evaluate ΔE_hull for {candidate.row.get('file')} at {V:.3f} V: {exc}"
            )
            stability.append(False)
            continue
        stability.append(e_above <= energy_tol)

    V_red: Optional[float] = None
    V_ox: Optional[float] = None

    # Oxidation limit: sweep to higher potentials
    for i in range(1, len(voltages)):
        if stability[i - 1] and not stability[i]:
            V_ox = voltages[i]
            break

    # Reduction limit: sweep downwards from high to low
    for i in range(len(voltages) - 1, 0, -1):
        if stability[i] and not stability[i - 1]:
            V_red = voltages[i]
            break

    window = None
    if V_red is not None and V_ox is not None:
        window = max(0.0, V_ox - V_red)

    return V_red, V_ox, window


def main() -> None:
    args = parse_args()

    if not args.mp_api_key:
        raise SystemExit("MP API key not provided. Use --mp-api-key or set MP_API_KEY.")

    stable_csv = Path(args.stable_csv).expanduser()
    rows = load_stable_rows(stable_csv)

    grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["chemsys"]].append(row)

    voltages = generate_voltage_grid(args.voltage_min, args.voltage_max, args.voltage_step)
    work_element = Element(args.working_element)

    model = CHGNet.load()
    print("[INFO] CHGNet model loaded for voltage window analysis.")

    results: List[Dict[str, Optional[float]]] = []

    # Keep MP responses as dicts to avoid older local document-model validation
    # rejecting newer alphanumeric Materials Project ids.
    with MPRester(args.mp_api_key, use_document_model=False) as mpr:
        for chemsys, chemsys_rows in grouped_rows.items():
            print(f"[INFO] Processing chemical system {chemsys} ({len(chemsys_rows)} candidates)")
            candidates = build_candidates(chemsys_rows)
            if not candidates:
                print(f"[WARN] No valid candidates found for {chemsys}, skipping.")
                continue

            comp_structs = fetch_mp_competitor_structures(
                chemsys, mpr, args.max_mp_competitors
            )
            if not comp_structs:
                print(f"[WARN] Failed to obtain MP competitor structures for {chemsys}, skipping.")
                continue

            comp_entries = structures_to_entries(comp_structs, model)
            if not comp_entries:
                print(f"[WARN] Competitor entries empty for {chemsys}, skipping.")
                continue

            mu_ref = locate_reference_mu(comp_entries, work_element)
            if mu_ref is None:
                print(
                    f"[WARN] No unary reference for {work_element} in {chemsys}; cannot determine μ_ref."
                )
                continue

            base_entries = comp_entries + [cand.entry for cand in candidates]

            for local_idx, cand in enumerate(candidates):
                if work_element not in {el for el in cand.entry.composition.elements}:
                    print(
                        f"[WARN] Working element {work_element} absent in {cand.row.get('file')}, skipping candidate."
                    )
                    continue

                V_red, V_ox, window = evaluate_voltage_window(
                    candidate=cand,
                    base_entries=base_entries,
                    candidate_index=len(comp_entries) + local_idx,
                    work_element=work_element,
                    mu_ref=mu_ref,
                    voltages=voltages,
                    energy_tol=args.threshold,
                )

                record: Dict[str, Optional[float]] = {
                    "file": cand.row.get("file"),
                    "formula": cand.row.get("formula"),
                    "chemsys": chemsys,
                    "V_red": V_red,
                    "V_ox": V_ox,
                    "window": window,
                }
                results.append(record)

    if not results:
        raise SystemExit("Voltage window analysis produced no results. Inspect warnings above.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file", "formula", "chemsys", "V_red", "V_ox", "window"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[INFO] Saved voltage window metrics to {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
