"""Novelty must be established by structural matching and complete sources."""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "pipeline"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mattergen"))

import novelty_screening as screening
from mattergen.evaluation.utils.dataset_matcher import get_matches
from mattergen.evaluation.utils.structure_matcher import DisorderedStructureMatcher


def rocksalt():
    return Structure(
        Lattice.cubic(5.6), ["Li"] * 4 + ["Cl"] * 4,
        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
         [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]],
    )


def cscl_type():
    # Same LiCl composition as rocksalt, but a different coordination topology.
    return Structure(Lattice.cubic(3.4), ["Li", "Cl"],
                     [[0, 0, 0], [.5, .5, .5]])


def write_input(tmp_path, structures, metadata=None):
    rows = []
    for i, structure in enumerate(structures):
        path = tmp_path / f"candidate_{i}.cif"
        structure.to(filename=str(path))
        rows.append({"path": str(path), "formula": "LiCl", "rank": str(i + 1),
                     **(metadata or {})})
    input_csv = tmp_path / "voltage_pass.csv"
    with input_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["path", "formula", "rank"])
        writer.writeheader()
        writer.writerows(rows)
    return input_csv


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def reference(structure, reference_id="mp-known", split="train"):
    return SimpleNamespace(reference_id=reference_id, structure=structure,
                           split=split, row_index=0, member=f"{split}.csv",
                           metadata={"declared_formula": structure.composition.reduced_formula})


def install_sources(monkeypatch, *, training=None, reference_set=None,
                    incomplete=None):
    calls = []
    source_references = {
        "training": list(training or []), "reference": list(reference_set or []),
    }
    def load(path, source_kind, candidate_compositions, **kwargs):
        calls.append((path, source_kind, candidate_compositions, kwargs))
        complete = source_kind != incomplete
        manifest = {
            "source_kind": source_kind, "source_role": source_kind,
            "path": str(path), "sha256": "a" * 64,
            "coverage_complete": complete,
            "status": "completed" if complete else "incomplete",
            "errors": [] if complete else ["missing structural data"],
        }
        return SimpleNamespace(references=source_references[source_kind], manifest=manifest)
    monkeypatch.setattr(screening, "load_novelty_source", load)
    return calls


def run_gate(tmp_path, input_csv, **kwargs):
    return screening.run_novelty_gate(
        input_csv, tmp_path / "final.csv", tmp_path / "novelty_audit.csv",
        tmp_path / "novelty_summary.json", [tmp_path / "training.zip"],
        [tmp_path / "reference.zip"], **kwargs,
    )


@pytest.mark.parametrize("transformation", ["identity", "origin", "site_order", "supercell"])
def test_known_structure_is_removed_despite_cell_or_site_representation(tmp_path, monkeypatch, transformation):
    known = rocksalt()
    candidate = known.copy()
    if transformation == "origin":
        candidate.translate_sites(range(len(candidate)), [.127, .219, .381],
                                  frac_coords=True, to_unit_cell=True)
    elif transformation == "site_order":
        candidate = Structure.from_sites(list(reversed(candidate.sites)))
    elif transformation == "supercell":
        candidate.make_supercell([2, 1, 1])
    input_csv = write_input(tmp_path, [candidate])
    original_input = input_csv.read_bytes()
    install_sources(monkeypatch, training=[reference(known)])

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "completed"
    assert summary["counts"]["matched"] == 1
    assert summary["counts"]["passed"] == 0
    assert read_rows(tmp_path / "final.csv") == []
    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert row["novelty_status"] == "matched"
    assert row["passes_novelty_filter"] == "False"
    assert json.loads(row["matched_reference_ids"]) == ["mp-known"]
    assert json.loads(row["matched_splits"]) == ["train"]
    assert row["candidate_structure_sha256"]
    assert input_csv.read_bytes() == original_input


def test_same_composition_different_structure_remains_with_full_source_coverage(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt(), cscl_type()])
    install_sources(monkeypatch, training=[reference(rocksalt())])

    summary = run_gate(tmp_path, input_csv)

    assert summary["coverage_complete"] is True
    assert summary["counts"]["matched"] == 1
    assert summary["counts"]["unmatched"] == 1
    assert summary["counts"]["passed"] == 1
    final = read_rows(tmp_path / "final.csv")
    assert [row["rank"] for row in final] == ["2"]
    assert final[0]["novelty_status"] == "unmatched"
    assert json.loads(final[0]["matched_reference_ids"]) == []


def test_structural_composition_overrides_wrong_candidate_formula_metadata(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()], {"formula": "LiNbOCl4"})
    calls = install_sources(monkeypatch, training=[reference(rocksalt())])

    summary = run_gate(tmp_path, input_csv)

    assert summary["counts"]["matched"] == 1
    assert calls
    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert row["formula"] == "LiNbOCl4"  # Preserve the preceding-stage record.
    assert row["novelty_actual_formula"] == "LiCl"


def test_known_reference_match_records_source_role_and_provenance(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    install_sources(monkeypatch, reference_set=[reference(rocksalt(), "icsd-42", "reference")])

    run_gate(tmp_path, input_csv)

    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert json.loads(row["matched_source_roles"]) == ["reference"]
    match = json.loads(row["novelty_matches"])[0]
    assert match["reference_id"] == "icsd-42"
    assert match["member"] == "reference.csv"
    assert match["row_index"] == 0
    assert match["metadata"]["declared_formula"] == "LiCl"


def test_incomplete_source_blocks_unmatched_candidates_but_retains_known_match_audit(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt(), cscl_type()])
    install_sources(monkeypatch, training=[reference(rocksalt())], incomplete="reference")
    # A previous run's success must not survive an incomplete verification.
    (tmp_path / "final.csv").write_text("path\n/stale/accepted.cif\n")

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "incomplete"
    assert summary["coverage_complete"] is False
    assert summary["counts"]["matched"] == 1
    assert summary["counts"]["unverified"] == 1
    assert summary["counts"]["passed"] == 0
    assert read_rows(tmp_path / "final.csv") == []
    assert [row["novelty_status"] for row in read_rows(tmp_path / "novelty_audit.csv")] == ["matched", "unverified"]


def test_invalid_candidate_never_passes_and_failure_is_audited(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    (tmp_path / "candidate_0.cif").write_text("not a crystal structure")
    install_sources(monkeypatch, training=[reference(rocksalt())])

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "incomplete"
    assert summary["counts"]["passed"] == 0
    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert row["novelty_status"] == "unverified"
    assert row["novelty_error"]
    assert read_rows(tmp_path / "final.csv") == []


def test_structure_comparison_exception_never_becomes_an_unmatched_pass(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    install_sources(monkeypatch, training=[reference(rocksalt())])
    class BrokenMatcher:
        def __init__(self, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            raise RuntimeError("synthetic symmetry failure")
    monkeypatch.setattr(screening, "load_matching_api", lambda: (BrokenMatcher, get_matches))

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "incomplete"
    assert summary["counts"]["comparison_failed"] == 1
    assert summary["counts"]["passed"] == 0
    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert row["novelty_status"] == "comparison_failed"
    assert "synthetic symmetry failure" in row["novelty_error"]
    assert read_rows(tmp_path / "final.csv") == []


def test_relative_candidate_paths_resolve_against_explicit_base_directory(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    text = input_csv.read_text().replace(str(tmp_path / "candidate_0.cif"), "candidate_0.cif")
    input_csv.write_text(text)
    install_sources(monkeypatch, training=[reference(rocksalt())])

    summary = run_gate(tmp_path, input_csv, base_dir=tmp_path)

    assert summary["counts"]["matched"] == 1


def test_training_split_configuration_is_forwarded_and_recorded(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [cscl_type()])
    calls = install_sources(monkeypatch)

    summary = run_gate(tmp_path, input_csv, training_splits=("train", "val"))

    training_calls = [call for call in calls if call[1] == "training"]
    assert len(training_calls) == 1
    assert tuple(training_calls[0][3]["training_splits"]) == ("train", "val")
    assert summary["counts"]["passed"] == 1
    assert len(summary["sources"]) == 2
    assert all(source["sha256"] == "a" * 64 for source in summary["sources"])


def test_empty_preceding_stage_does_not_load_large_sources(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [])
    def unexpected_load(*args, **kwargs):
        raise AssertionError("there are no candidates to compare")
    monkeypatch.setattr(screening, "load_novelty_source", unexpected_load)

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "completed"
    assert summary["counts"]["input_candidates"] == 0
    assert summary["counts"]["passed"] == 0
    assert read_rows(tmp_path / "final.csv") == []
    assert read_rows(tmp_path / "novelty_audit.csv") == []
    assert json.loads((tmp_path / "novelty_summary.json").read_text()) == summary


@pytest.mark.parametrize("missing_role", ["training", "reference"])
def test_each_required_source_role_must_be_configured(tmp_path, monkeypatch, missing_role):
    input_csv = write_input(tmp_path, [cscl_type()])
    install_sources(monkeypatch)
    paths = {"training": [tmp_path / "training.zip"], "reference": [tmp_path / "reference.zip"]}
    paths[missing_role] = []

    summary = screening.run_novelty_gate(
        input_csv, tmp_path / "final.csv", tmp_path / "novelty_audit.csv",
        tmp_path / "novelty_summary.json", paths["training"], paths["reference"],
    )

    assert summary["status"] == "incomplete"
    assert summary["counts"]["unverified"] == 1
    assert summary["counts"]["passed"] == 0
    assert read_rows(tmp_path / "final.csv") == []


def test_loader_exception_is_audited_and_cannot_pass_as_an_empty_source(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    def broken_load(*args, **kwargs):
        raise RuntimeError("synthetic unreadable source")
    monkeypatch.setattr(screening, "load_novelty_source", broken_load)

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "incomplete"
    assert summary["counts"]["passed"] == 0
    assert summary["counts"]["unverified"] == 1
    assert "synthetic unreadable source" in json.dumps(summary["sources"])


@pytest.mark.parametrize("status,errors", [
    ("incomplete", []), ("complete", [{"reason": "unreadable relevant entry"}]),
])
def test_contradictory_source_manifest_cannot_claim_complete_coverage(tmp_path, monkeypatch, status, errors):
    input_csv = write_input(tmp_path, [cscl_type()])
    install_sources(monkeypatch)
    original_load = screening.load_novelty_source
    def contradictory_load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        result.manifest.update(coverage_complete=True, status=status, errors=errors)
        return result
    monkeypatch.setattr(screening, "load_novelty_source", contradictory_load)

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "incomplete"
    assert summary["source_coverage_complete"] is False
    assert summary["counts"]["unverified"] == 1
    assert summary["counts"]["passed"] == 0


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), True])
def test_invalid_matcher_tolerance_removes_stale_success_output(tmp_path, tolerance):
    input_csv = write_input(tmp_path, [rocksalt()])
    final = tmp_path / "final.csv"
    final.write_text("path\n/stale/accepted.cif\n")

    with pytest.raises(ValueError, match="finite and positive"):
        run_gate(tmp_path, input_csv, stol=tolerance)

    assert not final.exists()
    assert input_csv.is_file()


def test_output_must_not_alias_and_delete_the_preceding_stage_input(tmp_path):
    input_csv = write_input(tmp_path, [rocksalt()])
    before = input_csv.read_bytes()

    with pytest.raises(ValueError, match="distinct"):
        screening.run_novelty_gate(
            input_csv, input_csv, tmp_path / "audit.csv", tmp_path / "summary.json",
            [tmp_path / "training.zip"], [tmp_path / "reference.zip"],
        )

    assert input_csv.read_bytes() == before


def test_oxidation_states_in_a_reference_do_not_hide_a_known_structure(tmp_path, monkeypatch):
    known = rocksalt()
    known.add_oxidation_state_by_element({"Li": 1, "Cl": -1})
    input_csv = write_input(tmp_path, [rocksalt()])
    install_sources(monkeypatch, training=[reference(known)])

    summary = run_gate(tmp_path, input_csv)

    assert summary["counts"]["matched"] == 1
    assert summary["counts"]["passed"] == 0


def test_cli_exits_nonzero_when_reference_sources_are_unavailable(tmp_path, capsys):
    input_csv = write_input(tmp_path, [rocksalt()])

    status = screening.main([
        "--input-csv", str(input_csv), "--out", str(tmp_path / "final.csv"),
        "--audit-out", str(tmp_path / "audit.csv"),
        "--summary-out", str(tmp_path / "summary.json"),
        "--training-data", str(tmp_path / "missing-training.zip"),
        "--reference-data", str(tmp_path / "missing-reference.lmdb"),
    ])

    assert status == 1
    assert read_rows(tmp_path / "final.csv") == []
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "incomplete"
    stderr = capsys.readouterr().err
    assert str(tmp_path / "missing-training.zip") in stderr
    assert str(tmp_path / "missing-reference.lmdb") in stderr
    assert "source_file_unavailable" in stderr


def test_actual_mattergen_matching_api_is_used(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    install_sources(monkeypatch, training=[reference(rocksalt())])
    matcher_class, matching_function = screening.load_matching_api()
    assert matcher_class is DisorderedStructureMatcher
    assert matching_function is get_matches
    calls = []
    def observed_matches(matcher, candidates, references):
        calls.append((matcher, candidates, references))
        return matching_function(matcher, candidates, references)
    monkeypatch.setattr(screening, "load_matching_api", lambda: (matcher_class, observed_matches))

    summary = run_gate(tmp_path, input_csv)

    assert summary["counts"]["matched"] == 1
    assert calls
    assert all(type(call[0]) is DisorderedStructureMatcher for call in calls)
    assert all(type(call[0]).__module__ == "mattergen.evaluation.utils.structure_matcher" for call in calls)
    assert any(call[2] for call in calls)


def test_disordered_reference_with_close_composition_is_not_excluded_before_matching(tmp_path, monkeypatch):
    ordered = rocksalt()
    partially_occupied = Structure(
        ordered.lattice, ["Li"] * 4 + [{"Cl": .98}] * 4, ordered.frac_coords,
    )
    assert ordered.composition.fractional_composition != partially_occupied.composition.fractional_composition
    assert DisorderedStructureMatcher().fit(ordered, partially_occupied)
    input_csv = write_input(tmp_path, [ordered])
    install_sources(monkeypatch, reference_set=[reference(partially_occupied, "disordered-known", "reference")])

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "completed"
    assert summary["counts"]["matched"] == 1
    assert summary["counts"]["passed"] == 0
    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert json.loads(row["matched_reference_ids"]) == ["disordered-known"]


def test_real_source_loading_and_mattergen_matching_remove_partial_occupancy_reference(tmp_path):
    ordered = rocksalt()
    partially_occupied = Structure(
        ordered.lattice, ["Li"] * 4 + [{"Cl": .98}] * 4, ordered.frac_coords,
    )
    input_csv = write_input(tmp_path, [ordered])
    source_paths = []
    for filename, identifier, structure in (
        ("train.csv", "different-training-polymorph", cscl_type()),
        ("reference.csv", "partial-occupancy-known", partially_occupied),
    ):
        path = tmp_path / filename
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["entry_id", "formula", "structure"])
            writer.writeheader()
            writer.writerow({"entry_id": identifier,
                             "formula": structure.composition.reduced_formula,
                             "structure": json.dumps(structure.as_dict())})
        source_paths.append(path)

    summary = screening.run_novelty_gate(
        input_csv, tmp_path / "final.csv", tmp_path / "novelty_audit.csv",
        tmp_path / "novelty_summary.json", source_paths[:1], source_paths[1:],
    )

    assert summary["status"] == "completed"
    assert summary["counts"]["matched"] == 1
    assert summary["counts"]["passed"] == 0
    row = read_rows(tmp_path / "novelty_audit.csv")[0]
    assert json.loads(row["matched_reference_ids"]) == ["partial-occupancy-known"]
    assert all(source["sha256"] for source in summary["sources"])
    assert all(source["selected_rows"] == 1 for source in summary["sources"])


def test_missing_mattergen_matching_dependency_is_an_explicit_incomplete_result(tmp_path, monkeypatch):
    input_csv = write_input(tmp_path, [rocksalt()])
    install_sources(monkeypatch, training=[reference(rocksalt())])
    def import_without_mattergen(name):
        raise ModuleNotFoundError("No module named 'mattergen'")
    monkeypatch.setattr(screening, "import_mattergen_module", import_without_mattergen)

    summary = run_gate(tmp_path, input_csv)

    assert summary["status"] == "incomplete"
    assert summary["counts"]["passed"] == 0
    assert summary["counts"]["unverified"] == 1
    assert "mattergen" in json.dumps(summary["errors"]).lower()
    assert read_rows(tmp_path / "final.csv") == []


def test_output_must_not_alias_and_delete_a_candidate_structure(tmp_path):
    input_csv = write_input(tmp_path, [rocksalt()])
    candidate_path = tmp_path / "candidate_0.cif"
    input_before = input_csv.read_bytes()
    candidate_before = candidate_path.read_bytes()

    with pytest.raises(ValueError, match="distinct"):
        screening.run_novelty_gate(
            input_csv, candidate_path, tmp_path / "audit.csv", tmp_path / "summary.json",
            [tmp_path / "training.zip"], [tmp_path / "reference.zip"],
        )

    assert input_csv.read_bytes() == input_before
    assert candidate_path.read_bytes() == candidate_before
