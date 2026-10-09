"""Source coverage and split selection for post-screening novelty checks."""

import csv
import gzip
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import pytest
from pymatgen.core import Composition, Lattice, Structure


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "pipeline"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mattergen"))

import novelty_data as data


def crystal(species=("Li", "Cl")):
    return Structure(Lattice.cubic(4), list(species),
                     [[i / len(species)] * 3 for i in range(len(species))])


def csv_text(rows, fields=("material_id", "reduced_formula", "cif")):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def cif_row(identifier, structure=None, **overrides):
    structure = crystal() if structure is None else structure
    return {"material_id": identifier,
            "reduced_formula": structure.composition.reduced_formula,
            "cif": structure.to(fmt="cif"), **overrides}


def write_release(tmp_path, members):
    path = tmp_path / "training-release.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)
    return path


def reasons(result):
    return {error["reason"] for error in result.manifest["errors"]}


def test_default_training_scope_uses_train_only_and_keeps_provenance(tmp_path):
    path = write_release(tmp_path, {
        "data/train.csv": csv_text([cif_row("mp-train")]),
        "data/val.csv": csv_text([cif_row("mp-val")]),
        "data/test.csv": csv_text([cif_row("mp-test")]),
    })

    result = data.load_novelty_source(path, "training", [Composition("LiCl")])

    assert result.manifest["coverage_complete"] is True
    assert result.manifest["status"] == "complete"
    assert result.manifest["requested_splits"] == ["train"]
    assert result.manifest["available_requested_splits"] == ["train"]
    assert result.manifest["scanned_rows"] == 1
    assert result.manifest["comparable_rows"] == 1
    assert result.manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.manifest["size_bytes"] == path.stat().st_size
    assert result.manifest["exact_checkpoint_training_membership_verified"] is False
    reference = result.references[0]
    assert reference.reference_id == "mp-train"
    assert reference.split == "train"
    assert reference.row_index == 0
    assert reference.member == "data/train.csv"
    assert reference.composition_key == data.composition_key("LiCl")


def test_validation_split_is_compared_only_when_explicitly_selected(tmp_path):
    path = write_release(tmp_path, {
        "train.csv": csv_text([cif_row("mp-train")]),
        "val.csv": csv_text([cif_row("mp-val")]),
    })

    result = data.load_novelty_source(path, "training", ["LiCl"],
                                      training_splits=("train", "val"))

    assert result.manifest["coverage_complete"] is True
    assert [reference.reference_id for reference in result.references] == ["mp-train", "mp-val"]
    assert [reference.split for reference in result.references] == ["train", "val"]
    assert result.manifest["scanned_rows"] == 2
    assert set(result.manifest["splits"]) == {"train", "val"}


def test_chemical_system_prefilter_keeps_other_ratios_for_mattergen_disorder_matching(tmp_path):
    supercell = crystal()
    supercell.make_supercell([2, 1, 1])
    other_ratio = crystal(("Li", "Li", "Cl"))
    path = write_release(tmp_path, {"train.csv": csv_text([
        cif_row("supercell", supercell, reduced_formula="Li2Cl2"),
        cif_row("different-ratio", other_ratio),
        cif_row("other-elements", crystal(("Li", "F"))),
    ])})

    result = data.load_novelty_source(path, "training", ["ClLi"])

    assert result.manifest["coverage_complete"] is True
    assert result.manifest["scanned_rows"] == 3
    assert result.manifest["selected_rows"] == 2
    assert [reference.reference_id for reference in result.references] == ["supercell", "different-ratio"]
    assert data.composition_key(result.references[0].structure) == data.composition_key("LiCl")


def test_reference_metadata_csv_cannot_establish_structural_coverage(tmp_path):
    path = tmp_path / "ref.csv"
    path.write_text(csv_text([{"material_id": "mp-1", "reduced_formula": "LiCl"}],
                             fields=("material_id", "reduced_formula")))

    result = data.load_novelty_source(path, "reference", ["LiCl"])

    assert result.references == []
    assert result.manifest["coverage_complete"] is False
    assert result.manifest["status"] == "incomplete"
    assert result.manifest["metadata_only_rows"] == 1
    assert "csv_has_no_structure_column" in reasons(result)


def test_training_ref_metadata_is_reported_without_being_mistaken_for_train_structures(tmp_path):
    path = write_release(tmp_path, {
        "train.csv": csv_text([cif_row("train-1")]),
        "ref.csv": csv_text([{"material_id": "metadata-only", "reduced_formula": "LiCl"}],
                             fields=("material_id", "reduced_formula")),
    })

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.manifest["coverage_complete"] is True
    assert [reference.reference_id for reference in result.references] == ["train-1"]
    assert result.manifest["metadata_only_rows"] == 1
    assert result.manifest["comparable_rows"] == 1


@pytest.mark.parametrize("bad_geometry", ["", "not a CIF"])
def test_relevant_missing_or_malformed_geometry_makes_coverage_incomplete(tmp_path, bad_geometry):
    path = write_release(tmp_path, {"train.csv": csv_text([
        cif_row("good"), cif_row("unreadable", cif=bad_geometry),
    ])})

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert [reference.reference_id for reference in result.references] == ["good"]
    assert result.manifest["coverage_complete"] is False
    assert result.manifest["selected_rows"] == 2
    assert "selected_structure_unavailable" in reasons(result)
    error = result.manifest["errors"][0]
    assert error["row_index"] == 1
    assert error["member"] == "train.csv"
    assert error["split"] == "train"


def test_geometry_composition_must_agree_with_selected_formula_metadata(tmp_path):
    path = write_release(tmp_path, {"train.csv": csv_text([
        cif_row("wrong-formula", crystal(("Li", "F")), reduced_formula="LiCl"),
    ])})

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.references == []
    assert result.manifest["coverage_complete"] is False
    assert "selected_structure_unavailable" in reasons(result)
    assert "composition" in result.manifest["errors"][0]["error"].lower()


def test_missing_composition_metadata_does_not_silently_skip_uncertain_structure(tmp_path):
    path = write_release(tmp_path, {"train.csv": csv_text([
        cif_row("unknown-composition", reduced_formula=""),
    ])})

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.manifest["coverage_complete"] is False
    assert "uncertain_composition" in reasons(result)


def test_json_structure_column_is_loaded_and_manifest_detects_source_changes(tmp_path):
    path = tmp_path / "reference.csv"
    row = {"entry_id": "json-reference", "formula": "LiCl",
           "structure": json.dumps(crystal().as_dict())}
    path.write_text(csv_text([row], fields=("entry_id", "formula", "structure")))

    first = data.load_novelty_source(path, "reference", ["LiCl"])
    path.write_text(csv_text([{**row, "entry_id": "changed-reference"}],
                             fields=("entry_id", "formula", "structure")))
    second = data.load_novelty_source(path, "reference", ["LiCl"])

    assert first.manifest["coverage_complete"] is True
    assert second.manifest["coverage_complete"] is True
    assert first.references[0].reference_id == "json-reference"
    assert second.references[0].reference_id == "changed-reference"
    assert first.manifest["sha256"] != second.manifest["sha256"]


def test_missing_requested_split_prevents_a_complete_training_claim(tmp_path):
    path = write_release(tmp_path, {"val.csv": csv_text([cif_row("val-only")])})

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.references == []
    assert result.manifest["coverage_complete"] is False
    assert "requested_split_unavailable" in reasons(result)


def test_header_only_training_split_cannot_establish_training_coverage(tmp_path):
    path = write_release(tmp_path, {"train.csv": csv_text([])})

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.references == []
    assert result.manifest["coverage_complete"] is False
    assert "requested_split_empty" in reasons(result)


def test_header_only_reference_csv_cannot_establish_reference_coverage(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text(csv_text([]))

    result = data.load_novelty_source(path, "reference", ["LiCl"])

    assert result.references == []
    assert result.manifest["coverage_complete"] is False
    assert "reference_csv_empty" in reasons(result)


@pytest.mark.parametrize("source_kind", ["training", "reference"])
def test_unavailable_source_is_not_a_successful_empty_reference_set(tmp_path, source_kind):
    result = data.load_novelty_source(tmp_path / "absent.zip", source_kind, ["LiCl"])

    assert result.references == []
    assert result.manifest["status"] == "unavailable"
    assert result.manifest["coverage_complete"] is False
    assert result.manifest["sha256"] is None
    assert "source_file_unavailable" in reasons(result)


def test_git_lfs_pointer_cannot_be_treated_as_downloaded_training_structures(tmp_path):
    path = tmp_path / "release.zip"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 12345\n")

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.manifest["status"] == "unavailable"
    assert result.manifest["coverage_complete"] is False
    assert result.manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "git_lfs_pointer_has_no_structures" in reasons(result)


def test_corrupted_archive_cannot_produce_complete_coverage(tmp_path):
    path = tmp_path / "release.zip"
    path.write_bytes(b"PK\x03\x04corrupt archive")

    result = data.load_novelty_source(path, "training", ["LiCl"])

    assert result.manifest["coverage_complete"] is False
    assert result.manifest["status"] == "incomplete"
    assert result.manifest["errors"]


def make_lmdb(tmp_path, *, extra_key=False, missing_entry=False):
    lmdb = pytest.importorskip("lmdb")
    path = tmp_path / "reference.lmdb"
    records = {
        "name": "fixture-structures", "chemical_systems": ["Cl-Li"],
        "Cl-Li.reduced_formulas": ["LiCl"], "Cl-Li.LiCl.length": 1,
        "Cl-Li.LiCl.0": {"structure": crystal().as_dict(), "entry_id": "known-lmdb"},
    }
    if missing_entry:
        del records["Cl-Li.LiCl.0"]
    if extra_key:
        records["Cl-Li.LiCl.1"] = records["Cl-Li.LiCl.0"]
    with lmdb.open(str(path), subdir=False, map_size=1024 * 1024) as env:
        with env.begin(write=True) as transaction:
            for key, value in records.items():
                transaction.put(key.encode("ascii"), pickle.dumps(value))
    return path


def test_gzip_lmdb_loads_structures_and_cleans_decompression_scratch(tmp_path):
    path = make_lmdb(tmp_path)
    archive = tmp_path / "reference.lmdb.gz"
    with gzip.open(archive, "wb") as handle:
        handle.write(path.read_bytes())
    scratch = tmp_path / "scratch"

    result = data.load_novelty_source(archive, "reference", ["LiCl"], scratch_dir=scratch)

    assert result.manifest["coverage_complete"] is True
    assert result.manifest["dataset_name"] == "fixture-structures"
    assert result.manifest["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result.manifest["decompressed_size_bytes"] == path.stat().st_size
    assert result.references[0].reference_id == "known-lmdb"
    assert result.references[0].member == "Cl-Li.LiCl.0"
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", ["missing_entry", "extra_key"])
def test_lmdb_index_mismatch_rejects_an_incomplete_structural_reference(tmp_path, failure):
    path = make_lmdb(tmp_path, **{failure: True})

    result = data.load_novelty_source(path, "reference", ["LiCl"])

    assert result.manifest["coverage_complete"] is False
    assert "lmdb_scoped_index_key_mismatch" in reasons(result)
    if failure == "missing_entry":
        assert result.references == []
        assert "selected_structure_unavailable" in reasons(result)


def test_empty_lmdb_chemical_system_index_cannot_establish_reference_coverage(tmp_path):
    lmdb = pytest.importorskip("lmdb")
    path = tmp_path / "empty.lmdb"
    with lmdb.open(str(path), subdir=False, map_size=1024 * 1024) as env:
        with env.begin(write=True) as transaction:
            transaction.put(b"name", pickle.dumps("empty-fixture"))
            transaction.put(b"chemical_systems", pickle.dumps([]))

    result = data.load_novelty_source(path, "reference", ["LiCl"])

    assert result.references == []
    assert result.manifest["coverage_complete"] is False
    assert "source_read_failed" in reasons(result)


def test_composition_key_is_independent_of_formula_order_and_cell_size():
    assert data.composition_key("Li2Cl2") == data.composition_key("ClLi")
    assert data.composition_key(crystal()) == data.composition_key("LiCl")
    assert data.composition_key("Li2Cl") != data.composition_key("LiCl")


@pytest.mark.parametrize("source_kind", ["unknown", "validation"])
def test_invalid_source_role_is_explicitly_rejected(tmp_path, source_kind):
    with pytest.raises(ValueError, match="source_kind"):
        data.load_novelty_source(tmp_path / "source.zip", source_kind, ["LiCl"])
