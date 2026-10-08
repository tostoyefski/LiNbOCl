"""Candidate selection regression tests using real source extxyz frames."""

import csv

import pytest
from ase import Atoms
from ase.io import write

from workflow.pipeline.candidate_selection import select_candidates


def crystal(symbols="LiCl", positions=None):
    return Atoms(
        symbols,
        scaled_positions=positions or [[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[4, 4, 4],
        pbc=True,
    )


def write_rows(tmp_path, frames, scores):
    path = tmp_path / "relaxed.extxyz"
    write(path, frames, format="extxyz")
    return [
        {
            "path": str(path),
            "frame": str(frame),
            "quick_score": str(score),
            "formula": "untrusted CSV formula",
            "source_note": f"frame-{frame}",
        }
        for frame, score in enumerate(scores)
    ]


def test_structural_duplicates_are_removed_before_topk(tmp_path):
    base = crystal()
    shifted = base.copy()
    shifted.set_scaled_positions(base.get_scaled_positions() + [0.13, 0.27, 0.31])
    shifted.wrap()
    rows = write_rows(
        tmp_path,
        [base, shifted, base.repeat((2, 1, 1)), crystal("NaCl")],
        [9, 10, 8, 7],
    )

    result = select_candidates(rows, 2)

    assert {row["frame"] for row in result.selected_rows} == {"1", "3"}
    assert result.counts["duplicates"] == 2
    assert result.counts["selected"] == 2
    for frame in (0, 2):
        assert result.audit_rows[frame]["selection_status"] == "duplicate"
        assert result.audit_rows[frame]["duplicate_of"] == result.audit_rows[1]["ref"]
    assert result.selected_rows[0]["source_note"] == "frame-1"
    assert result.selected_rows[0]["selection_composition"] == "LiCl"


def test_diverse_mode_covers_real_reduced_compositions_before_second_polymorph(tmp_path):
    rows = write_rows(
        tmp_path,
        [
            crystal(),
            crystal(positions=[[0, 0, 0], [0.5, 0, 0]]),
            crystal("Li2Cl", [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]),
            crystal("NaCl"),
        ],
        [100, 99, 10, 9],
    )

    result = select_candidates(rows, 3)

    assert [row["frame"] for row in result.selected_rows] == ["0", "2", "3"]
    assert {row["selection_composition"] for row in result.selected_rows} == {"LiCl", "Li2Cl", "NaCl"}
    assert result.audit_rows[1]["selection_status"] == "not_selected"
    assert result.counts["unique_structures"] == 4
    assert result.counts["selected_compositions"] == 3


def test_audit_csv_preserves_every_input_row_and_failure(tmp_path):
    from workflow.pipeline.candidate_selection import write_selection_audit

    rows = write_rows(tmp_path, [crystal(), crystal()], [3, 2])
    rows.append({"path": str(tmp_path / "missing.extxyz"), "frame": "0", "quick_score": "4"})
    result = select_candidates(rows, 10)
    destination = tmp_path / "audit" / "selection.csv"

    write_selection_audit(result, destination)

    with destination.open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == len(rows)
    assert [record["selection_status"] for record in records] == ["selected", "duplicate", "read_error"]
    assert records[1]["duplicate_of"] == records[0]["ref"]
    assert records[0]["formula"] == "untrusted CSV formula"
    assert records[0]["reduced_composition"] == "LiCl"
    assert records[2]["selection_reason"] == "source_structure_read_failed"
    assert "FileNotFoundError" in records[2]["error"]


def test_score_mode_retains_ranking_after_deduplication(tmp_path):
    rows = write_rows(
        tmp_path,
        [crystal(), crystal(), crystal(positions=[[0, 0, 0], [0.5, 0, 0]]), crystal("NaCl")],
        [9, 10, 8, 7],
    )

    result = select_candidates(rows, 2, selection_mode="score")

    assert [row["frame"] for row in result.selected_rows] == ["1", "2"]
    assert result.audit_rows[0]["duplicate_of"] == result.refs[0]
    assert result.audit_rows[1]["selection_reason"] == "highest_proxy_score"
    assert result.counts["selected_compositions"] == 1
    assert result.metadata["score_interpretation"] == "uncalibrated_geometry_proxy"


def test_ties_and_round_robin_are_deterministic_and_topk_can_exceed_available(tmp_path):
    rows = write_rows(
        tmp_path,
        [
            crystal(), crystal(),
            crystal(positions=[[0, 0, 0], [0.5, 0, 0]]),
            crystal("Li2Cl", [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]),
            crystal("NaCl"),
        ],
        [5] * 5,
    )
    for row in rows:
        row["path"] = "relaxed.extxyz"
    original_rows = [dict(row) for row in rows]

    result = select_candidates(rows, 20, base_dir=tmp_path)
    reversed_result = select_candidates(reversed(rows), 20, base_dir=tmp_path)

    assert [row["frame"] for row in result.selected_rows] == ["0", "3", "4", "2"]
    assert result.refs == reversed_result.refs
    assert result.counts["selected"] == 4
    assert result.counts["unique_structures"] == 4
    assert result.audit_rows[1]["duplicate_of"] == result.audit_rows[0]["ref"]
    assert rows == original_rows
    assert all(ref.startswith(str(tmp_path.resolve())) for ref in result.refs)


def test_nonfinite_and_non_numeric_scores_are_audited_and_never_selected(tmp_path):
    frames = [crystal(), crystal()] + [crystal(symbols) for symbols in ("LiF", "NaF", "KF", "RbF", "CsF")]
    rows = write_rows(tmp_path, frames, [-1, "inf", "nan", "-inf", "bad", "", "None"])
    rows[-1]["quick_score"] = None

    result = select_candidates(rows, 10)

    assert [row["frame"] for row in result.selected_rows] == ["0"]
    assert result.counts["invalid_scores"] == 6
    assert result.counts["duplicates"] == 1
    assert result.audit_rows[1]["score_status"] == "invalid"
    assert result.audit_rows[1]["selection_status"] == "duplicate"
    assert all(row["selection_status"] == "invalid_score" for row in result.audit_rows[2:])
    assert all(row["selection_reason"] == "no_finite_score" for row in result.audit_rows[2:])
    assert all(row["structure_status"] == "read" for row in result.audit_rows)

    only_invalid = select_candidates(rows[2:], 10)
    assert only_invalid.selected_rows == []
    assert only_invalid.refs == []
    assert only_invalid.counts["eligible_unique"] == 0


def test_missing_malformed_and_nonperiodic_source_structures_fail_closed(tmp_path):
    nonperiodic = crystal("NaCl")
    nonperiodic.pbc = False
    singular = crystal("LiF")
    singular.set_cell([0, 0, 0])
    nonfinite = crystal("NaF")
    nonfinite.positions[0, 0] = float("nan")
    rows = write_rows(tmp_path, [crystal(), nonperiodic, singular, nonfinite], [1, 99, 99, 99])
    malformed = tmp_path / "broken.extxyz"
    malformed.write_text("not an extxyz structure\n", encoding="utf-8")
    rows.extend([
        {"path": str(malformed), "frame": "0", "quick_score": "99"},
        {"path": rows[0]["path"], "frame": "99", "quick_score": "99"},
        {"path": rows[0]["path"], "frame": "-1", "quick_score": "99"},
        {"path": rows[0]["path"], "frame": "0.0", "quick_score": "99"},
        {"frame": "0", "quick_score": "99"},
    ])

    result = select_candidates(rows, 10)

    assert [row["frame"] for row in result.selected_rows] == ["0"]
    assert result.counts["read_errors"] == 8
    assert all(row["selection_status"] == "read_error" for row in result.audit_rows[1:])
    assert all(row["error"] for row in result.audit_rows[1:])
    assert all(row["selection_reason"] == "source_structure_read_failed" for row in result.audit_rows[1:])


def test_empty_input_and_zero_topk_still_write_an_audit(tmp_path):
    from workflow.pipeline.candidate_selection import write_selection_audit

    empty = select_candidates([], 300)
    write_selection_audit(empty, tmp_path / "empty.csv")
    with (tmp_path / "empty.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert "selection_status" in reader.fieldnames
        assert list(reader) == []
    rows = write_rows(tmp_path, [crystal()], [1])
    zero = select_candidates(rows, 0)
    assert zero.selected_rows == []
    assert zero.audit_rows[0]["selection_status"] == "not_selected"


@pytest.mark.parametrize("options", [
    {"ltol": float("nan")}, {"stol": -0.1}, {"angle_tol": float("inf")},
    {"ltol": 0}, {"stol": "invalid"}, {"topk": -1}, {"selection_mode": "unknown"},
])
def test_invalid_selection_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        select_candidates([], **{"topk": 10, **options})
