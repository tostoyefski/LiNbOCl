"""Select candidates from their source frames, with structural duplicate auditing.

``select_candidates(rows, topk, base_dir=workdir)`` returns a ``SelectionResult``.
Its ``selected_rows`` keep the input CSV metadata; ``refs`` are absolute source
``path::frame`` references suitable for the export script. ``audit_rows`` retain
one record per input row, in input order. No exported CIF directory is consulted.

Scores are a geometric proxy supplied by the caller, not a calibrated measure
of conductivity or stability. Structural matching is tolerance dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import math

import numpy as np
from ase.io import read as ase_read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor


@dataclass
class SelectionResult:
    selected_rows: list[dict[str, Any]]
    refs: list[str]
    audit_rows: list[dict[str, Any]]
    counts: dict[str, int]
    metadata: dict[str, Any]


AUDIT_FIELDS = (
    "input_index", "ref", "reduced_composition", "parsed_score", "score_status",
    "structure_status", "selection_status", "selection_reason", "duplicate_of",
    "duplicate_of_input_index", "selection_rank", "error",
)


def write_selection_audit(result: SelectionResult, destination: str | Path) -> None:
    """Save per-input-row decisions, errors, and original metadata as UTF-8 CSV.

    ``counts`` and ``metadata`` remain available separately for the pipeline's
    JSON run report. ``invalid_scores`` counts every non-finite/non-numeric input
    score, including rows also marked as duplicates or source-read failures.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AUDIT_FIELDS)
    seen = set(fieldnames)
    for row in result.audit_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.audit_rows)


def select_candidates(
    rows: Iterable[Mapping[str, Any]],
    topk: int,
    selection_mode: str = "diverse",
    *,
    base_dir: str | Path | None = None,
    score_key: str = "quick_score",
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> SelectionResult:
    """Read source structures, deduplicate them, then choose at most ``topk``.

    Input rows must provide ``path``, a nonnegative zero-based ``frame``, and a
    finite ``score_key`` value to be eligible. Relative paths resolve against
    ``base_dir`` (the current directory by default). Input ``formula`` fields
    are retained as metadata but never used for grouping or matching.

    A structural class keeps its highest-scoring row, with source path and
    numeric frame as deterministic tie breakers. Matching uses primitive-cell
    reduction, uniform volume scaling, and supercell matching. Reading or
    matching failures are excluded and recorded, never assumed to pass.

    ``diverse`` takes the best candidate from each real reduced-composition
    group before taking its second polymorph, and continues round robin. Groups
    are ordered by their best candidate's score and source reference. ``score``
    preserves descending proxy-score ranking after structural deduplication.
    """
    if isinstance(topk, bool) or not isinstance(topk, int) or topk < 0:
        raise ValueError("topk must be a nonnegative integer")
    if selection_mode not in {"diverse", "score"}:
        raise ValueError("selection_mode must be 'diverse' or 'score'")
    tolerances = {}
    for name, value in (("ltol", ltol), ("stol", stol), ("angle_tol", angle_tol)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive")
        tolerances[name] = number
    root = Path(base_dir or Path.cwd()).resolve()
    matcher = StructureMatcher(
        **tolerances,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
    )
    audits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for input_index, input_row in enumerate(rows):
        row = dict(input_row)
        audit = {
            **row,
            "input_index": input_index,
            "ref": "",
            "reduced_composition": "",
            "parsed_score": "",
            "score_status": "invalid",
            "structure_status": "not_read",
            "selection_status": "read_error",
            "selection_reason": "",
            "duplicate_of": "",
            "duplicate_of_input_index": "",
            "selection_rank": "",
            "error": "",
        }
        audits.append(audit)
        try:
            score = float(row.get(score_key))
        except (TypeError, ValueError, OverflowError):
            score = math.nan
        if math.isfinite(score):
            audit["parsed_score"] = score
            audit["score_status"] = "finite"
        try:
            raw_path = row.get("path")
            if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
                raise ValueError("missing source path")
            frame_text = str(row.get("frame", "")).strip()
            if not frame_text.isdigit():
                raise ValueError("frame must be a nonnegative integer")
            frame = int(frame_text)
            path = Path(raw_path).expanduser()
            path = (root / path).resolve() if not path.is_absolute() else path.resolve()
            ref = f"{path}::{frame}"
            audit["ref"] = ref
            atoms = ase_read(str(path), index=frame)
            if not len(atoms) or not all(atoms.pbc):
                raise ValueError("expected a nonempty structure periodic in all three axes")
            if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
                raise ValueError("non-finite structure coordinates or lattice")
            if abs(float(np.linalg.det(atoms.cell.array))) <= 1e-8:
                raise ValueError("singular structure lattice")
            structure = AseAtomsAdaptor.get_structure(atoms)
            composition = structure.composition.element_composition.reduced_composition
            composition_key = tuple(sorted(composition.get_el_amt_dict().items()))
            audit["reduced_composition"] = composition.reduced_formula
            audit["structure_status"] = "read"
        except Exception as exc:
            audit["selection_reason"] = "source_structure_read_failed"
            audit["error"] = f"{type(exc).__name__}: {exc}"
            continue
        candidates.append({
            "row": row, "audit": audit, "structure": structure,
            "composition_key": composition_key,
            "score": score, "path": str(path), "frame": frame,
        })

    def ranking(candidate: dict[str, Any]) -> tuple:
        score = candidate["score"]
        return (
            0 if math.isfinite(score) else 1,
            -score if math.isfinite(score) else 0,
            candidate["path"], candidate["frame"],
            candidate["audit"]["input_index"],
        )

    unique_by_composition: dict[tuple, list[dict[str, Any]]] = {}
    for candidate in sorted(candidates, key=ranking):
        audit = candidate["audit"]
        representatives = unique_by_composition.setdefault(candidate["composition_key"], [])
        try:
            duplicate = next(
                (representative for representative in representatives
                 if matcher.fit(candidate["structure"], representative["structure"])),
                None,
            )
        except Exception as exc:
            audit["structure_status"] = "match_error"
            audit["selection_status"] = "match_error"
            audit["selection_reason"] = "structure_comparison_failed"
            audit["error"] = f"{type(exc).__name__}: {exc}"
            continue
        if duplicate is not None:
            audit["selection_status"] = "duplicate"
            audit["selection_reason"] = "structure_match"
            audit["duplicate_of"] = duplicate["audit"]["ref"]
            audit["duplicate_of_input_index"] = duplicate["audit"]["input_index"]
            continue
        representatives.append(candidate)
        audit["selection_status"] = "not_selected" if math.isfinite(candidate["score"]) else "invalid_score"
        audit["selection_reason"] = "topk_limit" if math.isfinite(candidate["score"]) else "no_finite_score"

    eligible = sorted(
        (candidate for group in unique_by_composition.values() for candidate in group
         if math.isfinite(candidate["score"])),
        key=ranking,
    )
    if selection_mode == "score":
        selected = eligible[:topk]
    else:
        groups = [
            deque(candidate for candidate in group if math.isfinite(candidate["score"]))
            for group in unique_by_composition.values()
        ]
        group_queue = deque(sorted((group for group in groups if group), key=lambda group: ranking(group[0])))
        selected = []
        while group_queue and len(selected) < topk:
            group = group_queue.popleft()
            selected.append(group.popleft())
            if group:
                group_queue.append(group)
    selected_rows = []
    for rank, candidate in enumerate(selected, 1):
        audit = candidate["audit"]
        audit["selection_status"] = "selected"
        audit["selection_reason"] = "composition_round_robin" if selection_mode == "diverse" else "highest_proxy_score"
        audit["selection_rank"] = rank
        selected_rows.append({
            **candidate["row"],
            "selection_ref": audit["ref"],
            "selection_composition": audit["reduced_composition"],
        })
    counts = {
        "input_rows": len(audits),
        "structures_read": len(candidates),
        "read_errors": sum(row["selection_status"] == "read_error" for row in audits),
        "match_errors": sum(row["selection_status"] == "match_error" for row in audits),
        "invalid_scores": sum(row["score_status"] == "invalid" for row in audits),
        "duplicates": sum(row["selection_status"] == "duplicate" for row in audits),
        "unique_structures": sum(len(group) for group in unique_by_composition.values()),
        "eligible_unique": len(eligible),
        "selected": len(selected),
        "not_selected": len(eligible) - len(selected),
        "eligible_compositions": len({candidate["composition_key"] for candidate in eligible}),
        "selected_compositions": len({candidate["composition_key"] for candidate in selected}),
    }
    return SelectionResult(
        selected_rows=selected_rows,
        refs=[candidate["audit"]["ref"] for candidate in selected],
        audit_rows=audits,
        counts=counts,
        metadata={
            "selection_mode": selection_mode, "topk": topk, "score_key": score_key,
            "score_interpretation": "uncalibrated_geometry_proxy",
            "composition_group_order": "best_proxy_score_then_source_reference",
            "matcher": {**tolerances, "primitive_cell": True, "scale": True, "attempt_supercell": True},
        },
    )
