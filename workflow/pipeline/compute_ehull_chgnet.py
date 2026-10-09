#!/usr/bin/env python3
"""Compute ΔE_hull for CIF structures using CHGNet energies.

Candidates and every fetched Materials Project competitor are optimized with the
same MatterSim configuration before CHGNet single-point energies are evaluated.
The optimized structures, audits and complete phase-diagram entries are saved for
the voltage calculation to reuse without refetching or changing energy baselines.

Run with workflow/pipeline/run_top300_pipeline.py to apply the export manifest and filters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from time import sleep
from typing import Dict, Iterable, List, Optional, Tuple

from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core import Structure


DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "results" / "top300_run"
ENERGY_MODEL = "CHGNet-0.3.0"
SNAPSHOT_NAME = "reference_entries.json"


def parse_args(argv=None) -> argparse.Namespace:
    from mattersim_relaxation import add_relaxation_arguments
    parser = argparse.ArgumentParser(description="Evaluate ΔE_hull for CIF files with CHGNet.")
    parser.add_argument(
        "--cif-dir",
        default=str(DEFAULT_RESULTS / "exported_300cifs"),
        help="Directory containing CIF files to evaluate.",
    )
    parser.add_argument("--cif-index", type=Path, default=None,
                        help="Optional export index CSV; evaluate only its CIF files, excluding stale exports.")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS / "chgnet_hull_top300.csv"),
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
    add_relaxation_arguments(parser)
    args = parser.parse_args(argv)
    if not math.isfinite(args.stable_threshold) or args.stable_threshold < 0:
        parser.error("--stable-threshold must be finite and non-negative")
    return args


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


def load_candidate_structures(cif_dir: Path, index_csv: Path | None = None,
                              failures: Optional[List[Dict]] = None) -> List[Tuple[Path, Structure]]:
    structures: List[Tuple[Path, Structure]] = []
    if index_csv is None:
        paths = sorted(cif_dir.glob("*.cif"))
    else:
        with index_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if "cif" not in (reader.fieldnames or []):
                raise ValueError("Export index must contain a cif column")
            paths = [Path(row["cif"]).expanduser().resolve() for row in reader]
        if len(paths) != len(set(paths)):
            raise ValueError("Export index contains duplicate CIF paths")
        for path in paths:
            if path.parent != cif_dir.resolve() or path.suffix.lower() != ".cif" or not path.is_file():
                raise ValueError(f"Invalid CIF path in export index: {path}")
    for path in paths:
        try:
            structures.append((path, Structure.from_file(path)))
        except Exception as exc:
            if failures is not None:
                failures.append({"file": path.name, "source_path": str(path.resolve()),
                                 "hull_status": "calculation_failed", "relaxation_status": "failed",
                                 "is_stable": 0, "error": f"Candidate structure could not be read: {exc}"})
                continue
            if index_csv is not None:
                raise ValueError(f"Selected CIF could not be read: {path}") from exc
            print(f"[WARN] Failed to load {path}: {exc}")
    return structures


def chem_system(struct: Structure) -> str:
    return "-".join(sorted({el.symbol for el in struct.composition.elements}))


def search_with_retry(mpr: MPRester, **kwargs):
    from mp_api.client.core.client import MPRestError
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


def doc_get(doc, key: str, default=None):
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


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
            mid = doc_get(doc, "material_id")
            if mid is None or not str(mid).strip():
                raise ValueError("MP competitor is missing material_id; cannot audit a complete reference set")
            if mid in seen:
                continue
            struct = doc_get(doc, "structure")
            if struct is None:
                raise ValueError(f"MP competitor {mid} has no structure; incomplete reference sets cannot be used")
            if isinstance(struct, dict):
                struct = Structure.from_dict(struct)
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
        pick = next((d for d in docs if doc_get(d, "is_stable", False)), docs[0] if docs else None)
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
            stable = [d for d in docs if doc_get(d, "is_stable", False)]
            others = [d for d in docs if not doc_get(d, "is_stable", False)]
            add_docs(stable + others)
            if limit is not None and len(collected) >= limit:
                return collected

    return collected


def structures_to_entries(structs: Iterable[Structure], model: CHGNet) -> List[PDEntry]:
    """Strict single-point helper for structures already optimized uniformly."""
    entries: List[PDEntry] = []
    for struct in structs:
        pred = model.predict_structure(struct)
        e_pa = get_energy_per_atom(pred)
        if not math.isfinite(e_pa):
            raise ValueError("Non-finite CHGNet energy; incomplete phase sets cannot be used")
        entries.append(PDEntry(struct.composition, e_pa * len(struct)))
    return entries


def structure_file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_relaxation_audit(audit: Dict, settings) -> None:
    if audit.get("status") != "converged" or audit.get("converged") is not True:
        raise ValueError("Missing converged MatterSim audit; rerun hull calculation")
    if audit.get("settings") != settings.as_dict():
        raise ValueError("MatterSim relaxation settings mismatch; rerun hull calculation")
    for key, expected in {"optimizer": "FIRE", "cell_filter": "ExpCellFilter", "relax_cell": True,
                          "scalar_pressure_eV_A3": 0.0, "constrain_symmetry": False}.items():
        if key not in audit or audit[key] != expected:
            raise ValueError(f"Uniform relaxation method mismatch: {key}")
    force = float(audit.get("fmax_final", float("nan")))
    if not math.isfinite(force) or force < 0 or force > settings.fmax * (1 + 1e-7):
        raise ValueError("Unverified MatterSim full-cell force convergence")
    steps = audit.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= settings.max_steps:
        raise ValueError("Unverified MatterSim optimizer step count")


def validate_relaxed_record(record: Dict, settings) -> Structure:
    """Verify that a cached energy belongs to this configuration and saved CIF."""
    audit = record.get("relaxation", {})
    validate_relaxation_audit(audit, settings)
    cif_path = Path(record["path"]).expanduser().resolve()
    if not record.get("structure_sha256") or structure_file_sha256(cif_path) != record["structure_sha256"]:
        raise ValueError(f"Optimized CIF missing or changed: {cif_path}; rerun hull calculation")
    structure = Structure.from_file(cif_path)
    total = float(record["energy_total_eV"])
    per_atom = float(record["energy_per_atom_eV"])
    if not math.isfinite(total) or not math.isfinite(per_atom):
        raise ValueError("Non-finite cached CHGNet energy")
    if not math.isclose(total, per_atom * len(structure), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("Cached total and per-atom CHGNet energies disagree")
    if "structure" in record:
        original = Structure.from_dict(record["structure"])
        if original.composition != structure.composition:
            raise ValueError("Cached CHGNet structure composition differs from optimized CIF")
        # CIF canonicalizes the cell orientation and may reorder sites. Compare
        # lengths/angles and periodically matched coordinates at CIF precision.
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        if not np.allclose(original.lattice.abc, structure.lattice.abc, rtol=1e-7, atol=1e-6) or not np.allclose(
            original.lattice.angles, structure.lattice.angles, rtol=1e-7, atol=1e-5
        ):
            raise ValueError("Cached CHGNet structure cell differs from optimized CIF")
        if len(original) != len(structure):
            raise ValueError("Cached CHGNet site count differs from optimized CIF")
        for species in {str(site.species) for site in original}:
            left = [site.frac_coords for site in original if str(site.species) == species]
            right = [site.frac_coords for site in structure if str(site.species) == species]
            if len(left) != len(right):
                raise ValueError("Cached CHGNet site species differ from optimized CIF")
            distances = original.lattice.get_all_distances(left, right)
            rows, columns = linear_sum_assignment(distances)
            if np.max(distances[rows, columns]) > 1e-5:
                raise ValueError("Cached CHGNet coordinates differ from optimized CIF")
    return structure


def relax_and_evaluate(structure: Structure, model, relaxer, path: Path,
                       entry_id: str, chemsys: str, source_path: Optional[Path] = None):
    from mattersim_relaxation import save_relaxed_structure

    relaxed, audit = relaxer.relax(structure)
    validate_relaxation_audit(audit, relaxer.settings)
    if relaxed.composition != structure.composition:
        raise ValueError("MatterSim changed composition")
    entry = structures_to_entries([relaxed], model)[0]
    save_relaxed_structure(relaxed, path)
    record = {
        "id": entry_id, "chemsys": chemsys, "path": str(path.resolve()),
        "structure": relaxed.as_dict(), "structure_sha256": structure_file_sha256(path),
        "energy_total_eV": entry.energy, "energy_per_atom_eV": entry.energy_per_atom,
        "relaxation": audit,
    }
    if source_path is not None:
        record.update(file=source_path.name, source_path=str(source_path.resolve()))
    entry.name = entry_id
    return record, entry


def write_reference_snapshot(snapshot: Dict, path: Path) -> None:
    """Publish cache metadata only after all groups have explicit final states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(snapshot, handle, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_uniform_hull(candidates, model, relaxer, fetch_competitors, out_path: Path,
                     stable_threshold: float = 1e-3, initial_failures=None):
    """Serial MatterSim → CHGNet pipeline, with no partially successful references."""
    settings = relaxer.settings
    relaxation_dir = out_path.parent / "relaxation"
    snapshot_path = relaxation_dir / SNAPSHOT_NAME
    settings_json = json.dumps(settings.as_dict(), sort_keys=True)
    snapshot = {"schema_version": 1, "energy_model": ENERGY_MODEL,
                "relaxation_settings": settings.as_dict(), "entries_by_chemsys": {},
                "references": [], "candidates": [], "systems": {}}
    results = list(initial_failures or [])
    audit_records = []

    def add_audit(record, role, error=None, exception=None):
        audit = record.get("relaxation") or getattr(exception, "audit", None) or {}
        audit_records.append({
            "id": record.get("id"), "file": record.get("file"), "role": role,
            "chemsys": record.get("chemsys"), "source_path": record.get("source_path"),
            "path": record.get("path"), "relaxation_status": audit.get("status", "failed"),
            "energy_model": ENERGY_MODEL, "energy_per_atom_eV": record.get("energy_per_atom_eV"),
            "energy_total_eV": record.get("energy_total_eV"), "error": error,
            "relaxation_audit_json": json.dumps(audit, sort_keys=True, allow_nan=False),
            "relaxation_settings_json": settings_json,
        })

    for failure in results:
        add_audit(failure, "candidate", failure.get("error"))
    grouped = defaultdict(list)
    candidate_entries = []
    for index, (source, structure) in enumerate(candidates):
        csys = chem_system(structure)
        metadata = {"file": source.name, "source_path": str(source.resolve()),
                    "formula": structure.composition.reduced_formula, "chemsys": csys}
        try:
            record, entry = relax_and_evaluate(
                structure, model, relaxer, relaxation_dir / "candidates" / source.name,
                f"candidate:{index}:{source.name}", csys, source)
            snapshot["candidates"].append(record)
            add_audit(record, "candidate")
            candidate_entries.append(entry)
            metadata.update(path=record["path"], natoms_cell=len(structure),
                            energy_per_atom_eV=record["energy_per_atom_eV"],
                            energy_total_eV=record["energy_total_eV"],
                            relaxation_status="converged", structure_sha256=record["structure_sha256"],
                            relaxation_audit_json=json.dumps(record["relaxation"], sort_keys=True))
            grouped[csys].append((metadata, entry))
        except Exception as exc:
            metadata.update(hull_status="calculation_failed", relaxation_status="failed",
                            is_stable=0, error=f"Candidate relaxation/CHGNet failed: {exc}")
            results.append(metadata)
            add_audit({**metadata, "id": f"candidate:{index}:{source.name}"}, "candidate", str(exc), exc)
    for csys, group in grouped.items():
        references = []
        structures = []
        try:
            structures = list(fetch_competitors(csys))
            if not structures:
                raise ValueError("No MP competing structures returned")
            reference_errors = []
            for index, structure in enumerate(structures):
                entry_id = f"reference:{csys}:{index}"
                path = relaxation_dir / "references" / csys / f"reference_{index:05d}.cif"
                try:
                    record, entry = relax_and_evaluate(structure, model, relaxer, path, entry_id, csys)
                except Exception as exc:
                    reference_errors.append(f"{entry_id}: {exc}")
                    add_audit({"id": entry_id, "chemsys": csys, "path": str(path.resolve())},
                              "reference", str(exc), exc)
                    continue
                snapshot["references"].append(record)
                add_audit(record, "reference")
                references.append(entry)
            if reference_errors:
                raise ValueError(f"{len(reference_errors)} of {len(structures)} MP competitors failed: " + "; ".join(reference_errors))
            unary = {entry.composition.elements[0].symbol for entry in references
                     if len(entry.composition.elements) == 1}
            if set(csys.split("-")) - unary:
                raise ValueError("Missing optimized unary reference phases")
            # Include every successfully evaluated generated phase in this
            # chemical space, including candidates later excluded by the gate.
            elements = set(csys.split("-"))
            pool = references + [entry for entry in candidate_entries
                                 if {element.symbol for element in entry.composition.elements} <= elements]
            diagram = PhaseDiagram(pool)
            snapshot["entries_by_chemsys"][csys] = [entry.as_dict() for entry in pool]
            snapshot["systems"][csys] = {"status": "complete", "error": None,
                                       "fetched_count": len(structures), "prepared_count": len(references)}
            for metadata, entry in group:
                ehull = float(diagram.get_e_above_hull(entry))
                formation = float(diagram.get_form_energy_per_atom(entry))
                if not math.isfinite(ehull) or not math.isfinite(formation):
                    raise ValueError("Non-finite hull metrics")
                metadata.update(formation_energy_per_atom_eV=formation, energy_above_hull_eV=ehull,
                                is_stable=int(ehull <= stable_threshold), hull_status="complete", error=None)
        except Exception as exc:
            snapshot["systems"][csys] = {"status": "calculation_failed", "error": str(exc),
                                       "fetched_count": len(structures), "prepared_count": len(references)}
            snapshot["entries_by_chemsys"].pop(csys, None)
            for metadata, _ in group:
                metadata.update(hull_status="calculation_failed", is_stable=0, error=f"Complete phase diagram unavailable: {exc}")
                metadata.pop("energy_above_hull_eV", None)
                metadata.pop("formation_energy_per_atom_eV", None)
        results.extend(metadata for metadata, _ in group)
    for record in results:
        record.update(energy_model=ENERGY_MODEL, relaxation_settings_json=settings_json,
                      reference_snapshot_path=str(snapshot_path.resolve()))
    write_reference_snapshot(snapshot, snapshot_path)
    write_reference_snapshot({"relaxation_settings": settings.as_dict(), "records": audit_records},
                             relaxation_dir / "audit.json")
    audit_fields = ["id", "file", "role", "chemsys", "source_path", "path", "relaxation_status",
                    "energy_model", "energy_per_atom_eV", "energy_total_eV", "error",
                    "relaxation_audit_json", "relaxation_settings_json"]
    with (relaxation_dir / "audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_records)
    return results, snapshot


def main(argv=None) -> None:
    args = parse_args(argv)
    from mattersim_relaxation import MatterSimRelaxer, settings_from_args
    from chgnet.model import CHGNet
    from mp_api.client import MPRester

    if not args.mp_api_key:
        raise SystemExit("MP API key not provided. Use --mp-api-key or set MP_API_KEY.")
    cif_dir = Path(args.cif_dir).expanduser().resolve()
    if not cif_dir.is_dir():
        raise SystemExit(f"CIF directory not found: {cif_dir}")
    failures = []
    candidates = load_candidate_structures(cif_dir, args.cif_index, failures)
    if not candidates and not failures:
        raise SystemExit(f"No CIF files found in {cif_dir}")
    settings = settings_from_args(args)
    relaxer = MatterSimRelaxer(settings=settings)
    model = CHGNet.load(model_name="0.3.0")
    out_path = Path(args.out).expanduser().resolve()
    with MPRester(api_key=args.mp_api_key, use_document_model=False) as mpr:
        results, _ = run_uniform_hull(
            candidates, model, relaxer,
            lambda chemsys: fetch_mp_competitor_structures(chemsys, mpr, args.max_mp_competitors),
            out_path, args.stable_threshold, failures,
        )
    fieldnames = [
        "file", "path", "source_path", "formula", "chemsys", "natoms_cell",
        "energy_per_atom_eV", "energy_total_eV", "formation_energy_per_atom_eV",
        "energy_above_hull_eV", "is_stable", "hull_status", "energy_model",
        "relaxation_status", "relaxation_settings_json", "relaxation_audit_json",
        "structure_sha256", "reference_snapshot_path", "error",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Wrote {len(results)} hull rows and uniform-relaxation snapshot to {out_path}")


if __name__ == "__main__":
    main()
