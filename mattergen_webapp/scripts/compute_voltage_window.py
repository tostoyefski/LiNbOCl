#!/usr/bin/env python3
"""Evaluate high-voltage stability windows for CHGNet-screened candidates.

The workflow consumes a CSV (e.g. ``chgnet_hull_results_stable.csv``) and, for each
entry, assembles the relevant competing phases in the same chemical system. All
energies are (re-)evaluated on a consistent CHGNet baseline so that formation
energies and grand-potential hulls can be compared fairly. A voltage grid is then
scanned to report contiguous intervals of stable sampled voltages. V_red and V_ox
are the first and last stable grid points in the widest interval, with brackets
and censor flags describing what the scan can establish about phase boundaries.
An optional target voltage is evaluated directly, independently of grid spacing.

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
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PDEntry

try:  # compat across pymatgen versions
    from pymatgen.analysis.phase_diagram import GrandPotentialPDEntry as _GrandPotentialPDEntry
except ImportError:
    from pymatgen.analysis.phase_diagram import GrandPotPDEntry as _GrandPotentialPDEntry

from pymatgen.core import Element, Structure

DEFAULT_THRESHOLD = 1e-3  # eV/non-working-element atom on the grand hull
INTERVAL_POLICY = "widest_then_lowest_voltage"


def make_grand_entry(entry: PDEntry, chempots: Dict[Element, float]):
    """Compat helper to build grand-potential entries across pymatgen versions."""
    if hasattr(_GrandPotentialPDEntry, "from_pd_entry"):
        return _GrandPotentialPDEntry.from_pd_entry(entry, chempots)
    return _GrandPotentialPDEntry(entry, chempots)


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def nonnegative_finite_float(value: str) -> float:
    number = finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_finite_float(value: str) -> float:
    number = finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
        type=finite_float,
        default=0.0,
        help="Lower bound of the voltage grid (V).",
    )
    parser.add_argument(
        "--voltage-max",
        type=finite_float,
        default=6.0,
        help="Upper bound of the voltage grid (V).",
    )
    parser.add_argument(
        "--voltage-step",
        type=positive_finite_float,
        default=0.05,
        help="Voltage increment for scanning the electrochemical window (V).",
    )
    parser.add_argument(
        "--threshold",
        type=nonnegative_finite_float,
        default=DEFAULT_THRESHOLD,
        help="Stability tolerance in eV/non-working-element atom (default: 0.001).",
    )
    parser.add_argument(
        "--target-voltage",
        type=finite_float,
        default=None,
        help="Optional voltage (V) to evaluate directly against the grand hull.",
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
    args = parser.parse_args(argv)
    if args.voltage_max <= args.voltage_min:
        parser.error("--voltage-max must exceed --voltage-min")
    return args


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
    if not all(math.isfinite(value) for value in (vmin, vmax, step)):
        raise ValueError("voltage bounds and step must be finite")
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


def build_candidates(
    rows: Sequence[Dict[str, str]],
    failures: Optional[List[Tuple[Dict[str, str], str]]] = None,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for row in rows:
        cif_path = Path(row["path"]).expanduser()
        if not cif_path.exists():
            error = f"CIF path missing: {cif_path}"
            print(f"[WARN] {error}")
            if failures is not None:
                failures.append((row, error))
            continue
        try:
            structure = Structure.from_file(cif_path)
        except Exception as exc:
            error = f"Failed to load structure {cif_path}: {exc}"
            print(f"[WARN] {error}")
            if failures is not None:
                failures.append((row, error))
            continue
        try:
            energy_total = float(row["energy_total_eV"])
            if not math.isfinite(energy_total):
                raise ValueError("energy_total_eV must be finite")
        except (TypeError, ValueError):
            error = f"Invalid or non-finite energy_total_eV for {cif_path}"
            print(f"[WARN] {error}")
            if failures is not None:
                failures.append((row, error))
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


@dataclass
class VoltageWindowResult:
    """Grid bounds are conservative samples, not exact thermodynamic boundaries.

    Iteration preserves the former three-value unpacking API. Additional fields
    distinguish scan censoring, absent stable intervals and calculation errors.
    """

    V_red: Optional[float] = None
    V_ox: Optional[float] = None
    window: Optional[float] = None
    window_status: str = "no_stable_window"
    stable_intervals: List[Dict[str, object]] = field(default_factory=list)
    lower_boundary_bracket: Optional[Tuple[float, float]] = None
    upper_boundary_bracket: Optional[Tuple[float, float]] = None
    lower_bound_censored: Optional[bool] = None
    upper_bound_censored: Optional[bool] = None
    target_voltage: Optional[float] = None
    stable_at_target: Optional[bool] = None
    target_e_above_hull_eV: Optional[float] = None
    energy_tolerance_eV: float = DEFAULT_THRESHOLD
    e_above_hull_unit: str = "eV/non-Li atom"
    interval_policy: str = INTERVAL_POLICY
    error: Optional[str] = None

    def __iter__(self) -> Iterator[Optional[float]]:
        return iter((self.V_red, self.V_ox, self.window))

    def as_record(self) -> Dict[str, object]:
        record = dict(vars(self))
        record["stable_intervals_json"] = json.dumps(record.pop("stable_intervals"))
        for name in ("lower_boundary_bracket", "upper_boundary_bracket"):
            if record[name] is not None:
                record[name] = json.dumps(record[name])
        return record


def stable_intervals_from_grid(
    voltages: Sequence[float], stability: Sequence[bool]
) -> List[Dict[str, object]]:
    """Keep disjoint stable runs separate; each boundary uses adjacent samples."""
    intervals: List[Dict[str, object]] = []
    index = 0
    while index < len(voltages):
        if not stability[index]:
            index += 1
            continue
        first = index
        while index + 1 < len(voltages) and stability[index + 1]:
            index += 1
        last = index
        intervals.append(
            {
                "V_red": voltages[first],
                "V_ox": voltages[last],
                "window": voltages[last] - voltages[first],
                "lower_boundary_bracket": (
                    None if first == 0 else (voltages[first - 1], voltages[first])
                ),
                "upper_boundary_bracket": (
                    None if last == len(voltages) - 1 else (voltages[last], voltages[last + 1])
                ),
                "lower_bound_censored": first == 0,
                "upper_bound_censored": last == len(voltages) - 1,
            }
        )
        index += 1
    return intervals


def evaluate_voltage_point(
    base_entries: Sequence[PDEntry],
    candidate_index: int,
    work_element: Element,
    mu_ref: float,
    voltage: float,
) -> float:
    """Evaluate ΔE_hull in eV per atom remaining after removing the open ion.

    Original total-energy entries go into GPPD. Both it and the candidate's grand
    entry apply μ(V) = μ_ref - V exactly once; no formation-energy rebasing occurs.
    """
    mu = mu_ref - voltage
    if not math.isfinite(mu):
        raise ValueError("non-finite working-element chemical potential")
    for index, entry in enumerate(base_entries):
        if not math.isfinite(entry.energy):
            # Pymatgen may silently discard non-finite competing phases when
            # selecting its hull facets, which would fabricate stability.
            raise ValueError(f"non-finite total energy in phase entry {index}")
    chempots = {work_element: mu}
    diagram = GrandPotentialPhaseDiagram(base_entries, chempots)
    gp_entry = make_grand_entry(base_entries[candidate_index], chempots)
    e_above = float(diagram.get_e_above_hull(gp_entry))
    if not math.isfinite(e_above):
        raise ValueError("non-finite energy above grand-potential hull")
    return e_above


def evaluate_voltage_window(
    candidate: Candidate,
    base_entries: Sequence[PDEntry],
    candidate_index: int,
    work_element: Element,
    mu_ref: float,
    voltages: Sequence[float],
    energy_tol: float = DEFAULT_THRESHOLD,
    target_voltage: Optional[float] = None,
) -> VoltageWindowResult:
    if not math.isfinite(energy_tol) or energy_tol < 0:
        raise ValueError("energy tolerance must be finite and non-negative")
    if not math.isfinite(mu_ref):
        raise ValueError("reference chemical potential must be finite")
    voltages = list(voltages)
    if not voltages or not all(math.isfinite(value) for value in voltages):
        raise ValueError("voltage grid must be nonempty and finite")
    if any(right <= left for left, right in zip(voltages, voltages[1:])):
        raise ValueError("voltage grid must be strictly increasing")
    if target_voltage is not None and not math.isfinite(target_voltage):
        raise ValueError("target voltage must be finite")
    if not 0 <= candidate_index < len(base_entries):
        raise ValueError("candidate index is outside the entry list")

    result = VoltageWindowResult(
        target_voltage=target_voltage,
        energy_tolerance_eV=energy_tol,
        e_above_hull_unit=f"eV/non-{work_element.symbol} atom",
    )
    stability: List[bool] = []
    for voltage in voltages:
        try:
            e_above = evaluate_voltage_point(
                base_entries, candidate_index, work_element, mu_ref, voltage
            )
        except Exception as exc:
            result.window_status = "calculation_failed"
            result.error = f"Grand-potential calculation at {voltage:g} V failed: {exc}"
            return result
        stability.append(e_above <= energy_tol)

    if target_voltage is not None:
        try:
            result.target_e_above_hull_eV = evaluate_voltage_point(
                base_entries, candidate_index, work_element, mu_ref, target_voltage
            )
        except Exception as exc:
            result.window_status = "calculation_failed"
            result.error = f"Target-voltage calculation at {target_voltage:g} V failed: {exc}"
            return result
        result.stable_at_target = result.target_e_above_hull_eV <= energy_tol

    result.stable_intervals = stable_intervals_from_grid(voltages, stability)
    if result.stable_intervals:
        primary = min(
            result.stable_intervals,
            key=lambda interval: (-interval["window"], interval["V_red"]),
        )
        for name, value in primary.items():
            setattr(result, name, value)
        result.window_status = (
            "scan_censored"
            if result.lower_bound_censored or result.upper_bound_censored
            else "stable_window"
        )
    return result


def main() -> None:
    args = parse_args()

    if not args.mp_api_key:
        raise SystemExit("MP API key not provided. Use --mp-api-key or set MP_API_KEY.")

    # Keep pure phase-diagram helpers importable without torch, CHGNet or an API
    # client. These are needed only for the model/MP command-line workflow.
    from chgnet.model import CHGNet
    from mp_api.client import MPRester

    from compute_ehull_chgnet import fetch_mp_competitor_structures, structures_to_entries

    stable_csv = Path(args.stable_csv).expanduser()
    rows = load_stable_rows(stable_csv)

    grouped_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["chemsys"]].append(row)

    voltages = generate_voltage_grid(args.voltage_min, args.voltage_max, args.voltage_step)
    work_element = Element(args.working_element)

    model = CHGNet.load()
    print("[INFO] CHGNet model loaded for voltage window analysis.")

    results: List[Dict[str, object]] = []

    def add_record(row: Dict[str, str], result: VoltageWindowResult) -> None:
        record = {key: row.get(key) for key in ("file", "path", "formula", "chemsys")}
        record.update(result.as_record())
        results.append(record)

    def record_failure(row: Dict[str, str], error: str) -> None:
        print(f"[WARN] {row.get('file', row.get('path'))}: {error}")
        add_record(
            row,
            VoltageWindowResult(
                window_status="calculation_failed",
                target_voltage=args.target_voltage,
                energy_tolerance_eV=args.threshold,
                e_above_hull_unit=f"eV/non-{work_element.symbol} atom",
                error=error,
            ),
        )

    # Keep MP responses as dicts to avoid older local document-model validation
    # rejecting newer alphanumeric Materials Project ids.
    with MPRester(args.mp_api_key, use_document_model=False) as mpr:
        for chemsys, chemsys_rows in grouped_rows.items():
            print(f"[INFO] Processing chemical system {chemsys} ({len(chemsys_rows)} candidates)")
            build_failures: List[Tuple[Dict[str, str], str]] = []
            candidates = build_candidates(chemsys_rows, build_failures)
            for row, error in build_failures:
                record_failure(row, error)
            if not candidates:
                print(f"[WARN] No valid candidates found for {chemsys}, skipping.")
                continue

            try:
                comp_structs = fetch_mp_competitor_structures(
                    chemsys, mpr, args.max_mp_competitors
                )
                if not comp_structs:
                    raise ValueError(f"No MP competitor structures obtained for {chemsys}")
                comp_entries = structures_to_entries(comp_structs, model)
                if not comp_entries:
                    raise ValueError(f"Competitor entries empty for {chemsys}")
                if len(comp_entries) != len(comp_structs):
                    raise ValueError(
                        "Incomplete competing-phase energies: "
                        f"evaluated {len(comp_entries)} of {len(comp_structs)} structures"
                    )
                mu_ref = locate_reference_mu(comp_entries, work_element)
                if mu_ref is None or not math.isfinite(mu_ref):
                    raise ValueError(f"No finite unary reference for {work_element} in {chemsys}")
            except Exception as exc:
                for cand in candidates:
                    record_failure(cand.row, f"Competing-phase preparation failed: {exc}")
                continue

            base_entries = comp_entries + [cand.entry for cand in candidates]

            for local_idx, cand in enumerate(candidates):
                if work_element not in {el for el in cand.entry.composition.elements}:
                    record_failure(
                        cand.row, f"Working element {work_element} absent in candidate"
                    )
                    continue

                result = evaluate_voltage_window(
                    candidate=cand,
                    base_entries=base_entries,
                    candidate_index=len(comp_entries) + local_idx,
                    work_element=work_element,
                    mu_ref=mu_ref,
                    voltages=voltages,
                    energy_tol=args.threshold,
                    target_voltage=args.target_voltage,
                )
                if result.error:
                    print(f"[WARN] {cand.row.get('file')}: {result.error}")
                add_record(cand.row, result)

    if not results:
        raise SystemExit("Voltage window analysis produced no results. Inspect warnings above.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file", "path", "formula", "chemsys", *VoltageWindowResult().as_record()]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[INFO] Saved voltage window metrics to {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
