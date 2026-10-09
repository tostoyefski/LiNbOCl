#!/usr/bin/env python3
"""Evaluate high-voltage stability windows for CHGNet-screened candidates.

The workflow consumes a CSV (e.g. ``chgnet_hull_results_stable.csv``) and, for each
entry, reuses the complete optimized phase set from the hull reference snapshot.
Its candidate and MP geometries were optimized with identical MatterSim settings
and evaluated with CHGNet-0.3.0. No reference refetch or energy prediction occurs.
A voltage grid is then
scanned to report contiguous intervals of stable sampled voltages. V_red and V_ox
are the first and last stable grid points in the widest interval, with brackets
and censor flags describing what the scan can establish about phase boundaries.
An optional target voltage is evaluated directly, independently of grid spacing.

Assumptions:
    * The hull CSV and its reference snapshot were produced by the uniform
      MatterSim → CHGNet pipeline with the same relaxation settings.
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

DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "top300_run"
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
    from mattersim_relaxation import add_relaxation_arguments
    parser = argparse.ArgumentParser(
        description="Compute oxidation/reduction voltage limits for CHGNet candidates."
    )
    parser.add_argument(
        "--reference-snapshot", type=Path, default=None,
        help="Uniform optimized phase snapshot; default: <stable-csv parent>/relaxation/reference_entries.json.",
    )
    parser.add_argument(
        "--stable-csv",
        default=str(DEFAULT_RESULTS / "chgnet_hull_top300_filtered.csv"),
        help="CSV containing the filtered stable candidates (must include path & energy columns).",
    )
    parser.add_argument(
        "--mp-api-key",
        default=os.environ.get("MP_API_KEY"),
        help="Legacy argument retained for compatibility; unused by snapshot-only voltage analysis.",
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
        help="Legacy argument retained for compatibility; cached phase sets are never capped or refetched here.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS / "chgnet_voltage_window_top300.csv"),
        help="Output CSV file with voltage window metrics.",
    )
    add_relaxation_arguments(parser)
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


def load_reference_snapshot(path: Path, settings) -> Dict:
    from compute_ehull_chgnet import ENERGY_MODEL

    if not path.is_file():
        raise ValueError(f"Uniform reference snapshot missing: {path}; rerun hull calculation")
    with path.open() as handle:
        snapshot = json.load(handle)
    if snapshot.get("schema_version") != 1 or snapshot.get("energy_model") != ENERGY_MODEL:
        raise ValueError("Old or incompatible energy snapshot; rerun uniform hull calculation")
    if snapshot.get("relaxation_settings") != settings.as_dict():
        raise ValueError("MatterSim snapshot settings mismatch; rerun hull calculation")
    return snapshot


def snapshot_phase_entries(snapshot: Dict, chemsys: str, settings):
    """Check saved geometry/energy provenance for the complete cached diagram."""
    from compute_ehull_chgnet import validate_relaxed_record

    state = snapshot.get("systems", {}).get(chemsys, {})
    if state.get("status") != "complete":
        raise ValueError(f"Reference snapshot incomplete for {chemsys}: {state.get('error')}")
    entries = [PDEntry.from_dict(data) for data in snapshot["entries_by_chemsys"][chemsys]]
    if not entries or any(not math.isfinite(entry.energy) for entry in entries):
        raise ValueError("Empty or non-finite cached phase entries")
    entries_by_id = {entry.name: entry for entry in entries}
    if len(entries_by_id) != len(entries):
        raise ValueError("Duplicate phase identifiers in reference snapshot")
    elements = set(chemsys.split("-"))
    references = [record for record in snapshot.get("references", []) if record["chemsys"] == chemsys]
    if "fetched_count" in state and (
        state["fetched_count"] != len(references) or state.get("prepared_count") != len(references)
    ):
        raise ValueError("Not every fetched MP competitor is present in the optimized snapshot")
    candidates = [record for record in snapshot.get("candidates", [])
                  if {el.symbol for el in Structure.from_dict(record["structure"]).composition.elements} <= elements]
    records = references + candidates
    if {record["id"] for record in records} != set(entries_by_id) or len(records) != len(entries):
        raise ValueError("Reference snapshot does not contain the complete generated/reference phase set")
    for record in records:
        structure = validate_relaxed_record(record, settings)
        entry = entries_by_id[record["id"]]
        if entry.composition != structure.composition or not math.isclose(
            entry.energy, float(record["energy_total_eV"]), rel_tol=1e-10, abs_tol=1e-8
        ):
            raise ValueError("Cached phase entry and optimized structure/energy disagree")
    unary = {record_entry.composition.elements[0].symbol for record_entry in entries
             if len(record_entry.composition.elements) == 1 and record_entry.name in {record["id"] for record in references}}
    if elements - unary:
        raise ValueError("Optimized unary references are missing")
    return entries, candidates


def validated_snapshot_candidate(row: Dict[str, str], records, entries, settings, snapshot_path: Path):
    from compute_ehull_chgnet import ENERGY_MODEL, validate_relaxed_record

    if row.get("relaxation_status") != "converged" or row.get("energy_model") != ENERGY_MODEL:
        raise ValueError("Missing uniform MatterSim/CHGNet candidate metadata; rerun hull calculation")
    if json.loads(row.get("relaxation_settings_json") or "null") != settings.as_dict():
        raise ValueError("Candidate relaxation settings mismatch; rerun hull calculation")
    audit = json.loads(row.get("relaxation_audit_json") or "null")
    if not isinstance(audit, dict) or audit.get("status") != "converged" or audit.get("converged") is not True or audit.get("settings") != settings.as_dict():
        raise ValueError("Missing or mismatched candidate convergence audit; rerun hull calculation")
    if row.get("hull_status") not in {"complete", "success"} or row.get("error"):
        raise ValueError("Candidate hull calculation failed")
    declared = row.get("reference_snapshot_path")
    if not declared or Path(declared).expanduser().resolve() != snapshot_path.resolve():
        raise ValueError("Candidate belongs to a different or missing reference snapshot")
    path = Path(row["path"]).expanduser().resolve()
    matches = [record for record in records if Path(record["path"]).resolve() == path and record.get("file") == row.get("file")]
    if len(matches) != 1:
        raise ValueError("Candidate is missing or ambiguous in the complete phase snapshot")
    record = matches[0]
    if audit != record["relaxation"]:
        raise ValueError("Candidate CSV convergence audit differs from its energy snapshot")
    if row.get("structure_sha256") != record["structure_sha256"]:
        raise ValueError("Candidate CSV and snapshot geometry hashes differ")
    energy = float(row["energy_total_eV"])
    if not math.isfinite(energy) or not math.isclose(energy, float(record["energy_total_eV"]), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Candidate CSV and snapshot CHGNet energies differ")
    structure = validate_relaxed_record(record, settings)
    index = next(index for index, entry in enumerate(entries) if entry.name == record["id"])
    return Candidate(row=row, structure=structure, entry=entries[index]), index


def main(argv=None) -> None:
    args = parse_args(argv)
    from mattersim_relaxation import settings_from_args
    settings = settings_from_args(args)
    stable_csv = Path(args.stable_csv).expanduser().resolve()
    rows = load_stable_rows(stable_csv)
    snapshot_path = (args.reference_snapshot or stable_csv.parent / "relaxation" / "reference_entries.json").expanduser().resolve()
    voltages = generate_voltage_grid(args.voltage_min, args.voltage_max, args.voltage_step)
    work_element = Element(args.working_element)
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row["chemsys"]].append(row)
    results = []

    def add_record(row, result):
        results.append({**row, **result.as_record()})

    def record_failure(row, error):
        print(f"[WARN] {row.get('file', row.get('path'))}: {error}")
        add_record(row, VoltageWindowResult(
            window_status="calculation_failed", target_voltage=args.target_voltage,
            energy_tolerance_eV=args.threshold,
            e_above_hull_unit=f"eV/non-{work_element.symbol} atom", error=error,
        ))

    try:
        snapshot = load_reference_snapshot(snapshot_path, settings)
    except Exception as exc:
        for row in rows:
            record_failure(row, str(exc))
    else:
        for chemsys, chemsys_rows in grouped_rows.items():
            try:
                entries, candidate_records = snapshot_phase_entries(snapshot, chemsys, settings)
                mu_ref = locate_reference_mu(entries, work_element)
                if mu_ref is None or not math.isfinite(mu_ref):
                    raise ValueError(f"No finite optimized unary reference for {work_element}")
            except Exception as exc:
                for row in chemsys_rows:
                    record_failure(row, f"Complete phase snapshot validation failed: {exc}")
                continue
            for row in chemsys_rows:
                try:
                    candidate, index = validated_snapshot_candidate(
                        row, candidate_records, entries, settings, snapshot_path)
                    if work_element not in candidate.structure.composition.elements:
                        raise ValueError(f"Working element {work_element} absent in candidate")
                    result = evaluate_voltage_window(
                        candidate, entries, index, work_element, mu_ref, voltages,
                        args.threshold, args.target_voltage)
                    add_record(row, result)
                except Exception as exc:
                    record_failure(row, str(exc))

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in results for key in row))
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Reused uniform optimized phase snapshot; wrote {len(results)} voltage rows to {out_path}")


if __name__ == "__main__":
    main()
