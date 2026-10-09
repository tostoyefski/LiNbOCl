#!/usr/bin/env python3
"""Fail-closed, post-voltage structural novelty filtering against local releases.

An ``unmatched`` result means only that the candidate did not match any structure
in the explicitly supplied, completely readable sources at the recorded
tolerances. It is not a claim of experimental novelty or exhaustive literature
coverage. No structures are relaxed, no potential is loaded, and no data is
downloaded by this stage. Comparisons use MatterGen's existing
``DisorderedStructureMatcher`` and ``get_matches`` implementations directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from pymatgen.core import Structure

from novelty_data import composition_key, import_mattergen_module, load_novelty_source


NOVELTY_FIELDS = (
    "novelty_status", "passes_novelty_filter", "novelty_actual_formula",
    "candidate_structure_sha256", "matched_source_roles", "matched_reference_ids",
    "matched_splits", "novelty_matches", "novelty_error",
)


def load_matching_api() -> tuple[Any, Any]:
    """Load MatterGen's matching implementation only when candidates exist.

    There is deliberately no fallback matcher. An unavailable MatterGen
    installation is an incomplete comparison, rather than evidence of novelty.
    """
    module = import_mattergen_module("mattergen.evaluation.utils.dataset_matcher")
    return module.DisorderedStructureMatcher, module.get_matches


def _chemical_system(structure: Structure) -> str:
    return "-".join(sorted(element.symbol for element in structure.composition.elements))


def _matching_metadata(matcher: Any, get_matches: Any) -> dict[str, Any]:
    def source_path(value: Any) -> str | None:
        try:
            filename = inspect.getsourcefile(value)
        except (TypeError, OSError):
            filename = None
        return str(Path(filename).resolve()) if filename else None

    matcher_class = type(matcher)
    metadata: dict[str, Any] = {
        "matcher_class": f"{matcher_class.__module__}.{matcher_class.__qualname__}",
        "matcher_module_path": source_path(matcher_class),
        "matching_function": f"{get_matches.__module__}.{get_matches.__qualname__}",
        "matching_function_path": source_path(get_matches),
        "mattergen_version": getattr(sys.modules.get("mattergen"), "__version__", None),
    }
    if hasattr(matcher, "as_dict"):
        parameters = matcher.as_dict()
        for name in (
            "relative_radius_difference_threshold", "electronegativity_difference_threshold",
            "reduced_formula_atol", "reduced_formula_rtol",
        ):
            if hasattr(matcher, name):
                parameters[name] = getattr(matcher, name)
        ordered_matcher = getattr(matcher, "ordered_structurematcher", None)
        if ordered_matcher is not None and hasattr(ordered_matcher, "as_dict"):
            parameters["ordered_structurematcher"] = ordered_matcher.as_dict()
        metadata["effective_parameters"] = parameters
    return metadata


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _atomic_write(path: Path, callback: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            callback(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    def write(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write(path, write)


def _structure_without_oxidation_states(structure: Structure) -> Structure:
    cleaned = structure.copy()
    cleaned.remove_oxidation_states()
    if not len(cleaned):
        raise ValueError("Structure has no sites")
    if not np.isfinite(cleaned.lattice.matrix).all() or not np.isfinite(cleaned.cart_coords).all():
        raise ValueError("Non-finite coordinates or lattice")
    if abs(float(np.linalg.det(cleaned.lattice.matrix))) <= 1e-8:
        raise ValueError("Singular structure lattice")
    composition_key(cleaned)
    return cleaned


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def run_novelty_gate(
    input_csv: str | Path,
    final_csv: str | Path,
    audit_csv: str | Path,
    summary_json: str | Path,
    training_paths: Iterable[str | Path],
    reference_paths: Iterable[str | Path],
    *,
    training_splits: Iterable[str] = ("train",),
    scratch_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> dict[str, Any]:
    """Keep only candidates with complete, unsuccessful structural comparisons.

    ``input_csv`` is the separate CSV retained by the voltage gate. Its ``path``
    column points at each candidate CIF/POSCAR. Relative candidate paths resolve
    against ``base_dir`` or, by default, the input CSV's parent directory. All
    existing columns survive in both outputs; formula metadata is never trusted
    for comparison. The original candidates and CSV are not modified.

    Sources are grouped by chemical system, as in MatterGen's disordered dataset
    matcher. Similar compositions and partial occupancies are left to the
    official matcher; there is no additional exact-composition rejection.

    At least one training source and one reference source are required for a
    nonempty input. Incomplete source coverage makes every otherwise unmatched
    candidate ``unverified``. Candidate read errors or comparison failures are
    audited and cannot pass. Ordinary data failures return ``status=incomplete``
    so callers must fail their pipeline; invalid configuration raises after
    removing stale outputs. Empty inputs finish without reading any sources.
    """
    input_path = Path(input_csv).expanduser().resolve()
    destinations = [Path(path).expanduser().resolve() for path in (final_csv, audit_csv, summary_json)]
    sources = [(role, Path(path).expanduser().resolve())
               for role, paths in (("training", training_paths), ("reference", reference_paths))
               for path in paths]
    protected = {input_path, *(path for _, path in sources)}
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else input_path.parent
    # Reject accidental output aliases before stale-output cleanup can remove a
    # candidate geometry. Parsing failures are audited by the main read below.
    try:
        with input_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_path = row.get("path")
                if isinstance(raw_path, str) and raw_path.strip():
                    candidate_path = Path(raw_path).expanduser()
                    protected.add((root / candidate_path).resolve() if not candidate_path.is_absolute()
                                  else candidate_path.resolve())
    except Exception:
        pass
    if len(set(destinations)) != len(destinations) or protected.intersection(destinations):
        raise ValueError("Novelty input, source, final, audit and summary paths must be distinct")
    # In particular, an invalid threshold must not leave an old successful final.
    for path in destinations:
        path.unlink(missing_ok=True)
    final_path, audit_path, summary_path = destinations

    tolerances = {}
    for name, value in (("ltol", ltol), ("stol", stol), ("angle_tol", angle_tol)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive")
        tolerances[name] = number
    splits = tuple(dict.fromkeys(str(split).strip() for split in training_splits))
    if not splits or any(not split for split in splits):
        raise ValueError("training_splits must contain at least one nonempty split")
    settings = {
        **tolerances, "primitive_cell": True, "scale": True,
        "attempt_supercell": True, "allow_subset": True,
        "oxidation_states_removed": True, "composition_shortlist": "chemical_system_only",
        "backend": "mattergen.evaluation.utils.dataset_matcher",
        "api_loaded": False,
    }
    summary: dict[str, Any] = {
        "status": "incomplete", "coverage_complete": False,
        "source_coverage_complete": False, "input_csv": str(input_path),
        "final_csv": str(final_path), "audit_csv": str(audit_path),
        "training_splits": list(splits), "matcher": settings, "sources": [], "errors": [],
        "interpretation": "unmatched_only_against_supplied_complete_sources_at_recorded_tolerances",
        "exact_checkpoint_training_membership_verified": False,
    }
    rows: list[dict[str, Any]] = []
    fields: list[str] = []
    try:
        with input_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            if len(set(fields)) != len(fields) or any(not field for field in fields):
                raise ValueError("Input CSV has duplicate or empty column names")
            for row in reader:
                if None in row:
                    raise ValueError("Input CSV has more values than header columns")
                rows.append(dict(row))
    except Exception as exc:
        summary["errors"].append(f"Input CSV: {type(exc).__name__}: {exc}")
    fields = list(dict.fromkeys(fields + list(NOVELTY_FIELDS)))
    audits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        audit = {
            **row, "novelty_status": "unverified", "passes_novelty_filter": False,
            "novelty_actual_formula": "", "candidate_structure_sha256": "",
            "matched_source_roles": "[]", "matched_reference_ids": "[]",
            "matched_splits": "[]", "novelty_matches": "[]", "novelty_error": "",
        }
        audits.append(audit)
        try:
            raw_path = row.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("Missing candidate structure path")
            path = Path(raw_path).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            structure = _structure_without_oxidation_states(Structure.from_file(path))
            audit["novelty_actual_formula"] = structure.composition.reduced_formula
            audit["candidate_structure_sha256"] = hashlib.sha256(
                _json(structure.as_dict()).encode("utf-8")
            ).hexdigest()
            candidates.append({"index": index, "audit": audit, "structure": structure,
                               "chemical_system": _chemical_system(structure)})
        except Exception as exc:
            audit["novelty_error"] = f"Candidate structure: {type(exc).__name__}: {exc}"
            summary["errors"].append(f"Candidate row {index}: {audit['novelty_error']}")

    prepared_sources: list[dict[str, Any]] = []
    if rows:
        matcher, get_matches = None, None
        matching_api_error = ""
        if candidates:
            try:
                matcher_class, get_matches = load_matching_api()
                matcher = matcher_class(**tolerances)
                summary["matcher"].update(_matching_metadata(matcher, get_matches))
                for name in ("primitive_cell", "scale", "attempt_supercell", "allow_subset"):
                    parameters = summary["matcher"].get("effective_parameters", {})
                    if name in parameters:
                        summary["matcher"][name] = parameters[name]
                summary["matcher"]["api_loaded"] = True
            except Exception as exc:
                matching_api_error = f"MatterGen matching API unavailable: {type(exc).__name__}: {exc}"
                summary["errors"].append(matching_api_error)
        for role in ("training", "reference"):
            if not any(source_role == role for source_role, _ in sources):
                summary["errors"].append(f"No required {role} source supplied")
        for role, path in sources:
            manifest: dict[str, Any] = {
                "source_role": role, "source_path": str(path), "status": "unavailable",
                "coverage_complete": False, "errors": [],
            }
            groups: dict[str, list[tuple[Any, Structure]]] = {}
            try:
                if matching_api_error:
                    raise RuntimeError(matching_api_error)
                if not candidates:
                    raise RuntimeError("No readable candidates; reference loading was skipped")
                result = load_novelty_source(
                    path, role, [candidate["structure"] for candidate in candidates],
                    training_splits=splits, scratch_dir=scratch_dir,
                )
                raw_manifest = _attribute(result, "manifest")
                if not isinstance(raw_manifest, Mapping):
                    raise ValueError("Novelty loader returned no source manifest")
                manifest.update(dict(raw_manifest))
                manifest.update(source_role=role, source_path=str(path))
                for reference in _attribute(result, "references", []):
                    structure = _structure_without_oxidation_states(_attribute(reference, "structure"))
                    groups.setdefault(_chemical_system(structure), []).append((reference, structure))
                if manifest.get("errors") or manifest.get("status") not in ("complete", "completed"):
                    manifest["coverage_complete"] = False
                if manifest.get("coverage_complete") is not True:
                    manifest["status"] = "incomplete" if manifest.get("status") != "unavailable" else "unavailable"
            except Exception as exc:
                manifest["coverage_complete"] = False
                manifest["status"] = "incomplete"
                errors = list(manifest.get("errors") or [])
                errors.append({"reason": "source_loading_failed", "error": f"{type(exc).__name__}: {exc}"})
                manifest["errors"] = errors
            summary["sources"].append(manifest)
            prepared_sources.append({"role": role, "path": str(path), "manifest": manifest, "groups": groups})
            if manifest.get("coverage_complete") is not True:
                reasons = []
                for error in list(manifest.get("errors") or [])[:3]:
                    if isinstance(error, Mapping):
                        reasons.append(f"{error.get('reason', 'source_read_failed')}: {error.get('error', '')}".rstrip(": "))
                    else:
                        reasons.append(str(error))
                detail = "; ".join(reasons) or "structure coverage is incomplete"
                summary["errors"].append(f"Incomplete {role} source: {path} ({detail})")

        required_roles = {role for role, _ in sources} >= {"training", "reference"}
        source_complete = required_roles and all(
            source["manifest"].get("coverage_complete") is True for source in prepared_sources
        )
        summary["source_coverage_complete"] = source_complete
        for candidate in candidates:
            audit = candidate["audit"]
            matches, errors = [], []
            for source in prepared_sources:
                references = source["groups"].get(candidate["chemical_system"], [])
                if matching_api_error or matcher is None or get_matches is None:
                    continue
                try:
                    matched_indices = get_matches(
                        matcher, [candidate["structure"]], [structure for _, structure in references]
                    )
                    if not isinstance(matched_indices, Mapping) or any(key != 0 for key in matched_indices):
                        raise ValueError("MatterGen get_matches returned invalid candidate indices")
                    indices = matched_indices.get(0, [])
                    if not isinstance(indices, (list, tuple)) or any(
                        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(references)
                        for index in indices
                    ):
                        raise ValueError("MatterGen get_matches returned invalid reference indices")
                    for index in dict.fromkeys(indices):
                        reference, _ = references[index]
                        matches.append({
                            "source_role": source["role"], "source_path": source["path"],
                            "reference_id": str(_attribute(reference, "reference_id", "")),
                            "split": _attribute(reference, "split"),
                            "row_index": _attribute(reference, "row_index"),
                            "member": _attribute(reference, "member"),
                            "metadata": _attribute(reference, "metadata", {}),
                        })
                except Exception as exc:
                    errors.append(f"{source['role']} {source['path']}: {type(exc).__name__}: {exc}")
            audit["novelty_matches"] = _json(matches)
            audit["matched_source_roles"] = _json(sorted({match["source_role"] for match in matches}))
            audit["matched_reference_ids"] = _json(sorted({match["reference_id"] for match in matches}))
            audit["matched_splits"] = _json(sorted({str(match["split"]) for match in matches if match["split"] is not None}))
            if matches:
                audit["novelty_status"] = "matched"
            elif errors:
                audit["novelty_status"] = "comparison_failed"
            elif source_complete and not any(error.startswith("Input CSV:") for error in summary["errors"]):
                audit["novelty_status"] = "unmatched"
                audit["passes_novelty_filter"] = True
            else:
                audit["novelty_status"] = "unverified"
            if errors:
                audit["novelty_error"] = " | ".join(errors)
                summary["errors"].extend(f"Candidate row {candidate['index']}: {error}" for error in errors)
            if matching_api_error:
                audit["novelty_error"] = " | ".join(filter(None, (audit["novelty_error"], matching_api_error)))
            if not source_complete:
                coverage_error = "Required training/reference source coverage is incomplete"
                audit["novelty_error"] = " | ".join(filter(None, (audit["novelty_error"], coverage_error)))
    elif not summary["errors"]:
        summary["source_coverage_complete"] = True
        summary["sources_skipped_reason"] = "no_candidates_after_voltage_filter"

    kept = [row for row in audits if row["passes_novelty_filter"]]
    summary["counts"] = {
        "input_candidates": len(audits),
        **{status: sum(row["novelty_status"] == status for row in audits)
           for status in ("matched", "unmatched", "unverified", "comparison_failed")},
        "passed": len(kept),
    }
    summary["coverage_complete"] = bool(summary["source_coverage_complete"] and not summary["errors"])
    summary["status"] = "completed" if summary["coverage_complete"] else "incomplete"
    _write_csv(audit_path, audits, fields)
    _write_csv(final_path, kept, fields)
    _atomic_write(summary_path, lambda handle: handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n"))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Verified unmatched candidates only")
    parser.add_argument("--audit-out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--training-data", action="append", required=True, type=Path)
    parser.add_argument("--reference-data", action="append", required=True, type=Path)
    parser.add_argument("--training-splits", nargs="+", default=["train"])
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--ltol", type=float, default=0.2)
    parser.add_argument("--stol", type=float, default=0.3)
    parser.add_argument("--angle-tol", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_novelty_gate(
            args.input_csv, args.out, args.audit_out, args.summary_out,
            args.training_data, args.reference_data,
            training_splits=args.training_splits, scratch_dir=args.scratch_dir,
            base_dir=args.base_dir, ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol,
        )
    except Exception as exc:
        print(f"[ERROR] Novelty gate: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[INFO] Novelty gate {summary['status']}: {summary['counts']}")
    if summary["status"] != "completed":
        for error in summary["errors"]:
            print(f"[ERROR] {error}", file=sys.stderr)
        print(f"[ERROR] See audit {args.audit_out} and summary {args.summary_out}", file=sys.stderr)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
