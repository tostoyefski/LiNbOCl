"""Ensure downstream hull evaluation reads only the current export manifest."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["mattergen_webapp/scripts", "mattergen"])
def hull_module(request):
    path = REPO / request.param / "compute_ehull_chgnet.py"
    name = "hull_manifest_" + request.param.replace("/", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_cif(path, species="Li"):
    Structure(Lattice.cubic(8), [species], [[0, 0, 0]]).to(filename=path)
    return path.resolve()


def write_manifest(path, cifs):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ref", "cif"])
        writer.writeheader()
        writer.writerows({"ref": f"source.extxyz::{i}", "cif": str(cif)} for i, cif in enumerate(cifs))
    return path


def test_manifest_excludes_stale_cifs_and_preserves_selected_order(hull_module, tmp_path):
    export = tmp_path / "exported cifs"
    export.mkdir()
    selected = [make_cif(export / "second.cif", "Na"), make_cif(export / "first.cif")]
    make_cif(export / "stale_previous_run.cif", "K")
    # A malformed stale file must not affect an otherwise valid current run.
    (export / "broken_previous_run.cif").write_text("not a CIF")
    manifest = write_manifest(export / "export_index.csv", selected)
    candidates = hull_module.load_candidate_structures(export, manifest)
    assert [path for path, _ in candidates] == selected
    assert [str(structure.composition.reduced_formula) for _, structure in candidates] == ["Na", "Li"]


@pytest.mark.parametrize("invalid_kind", ["duplicate", "outside", "missing_cif", "unreadable_cif"])
def test_manifest_rejects_invalid_selected_paths(hull_module, tmp_path, invalid_kind):
    export = tmp_path / "exported cifs"
    export.mkdir()
    valid = make_cif(export / "selected.cif")
    if invalid_kind == "duplicate":
        cifs, error = [valid, valid], "duplicate CIF"
    elif invalid_kind == "outside":
        cifs, error = [make_cif(tmp_path / "outside.cif")], "Invalid CIF path"
    elif invalid_kind == "missing_cif":
        cifs, error = [export / "missing.cif"], "Invalid CIF path"
    else:
        bad = export / "unreadable.cif"
        bad.write_text("not a CIF")
        cifs, error = [bad], "Selected CIF could not be read"
    manifest = write_manifest(export / "export_index.csv", cifs)
    with pytest.raises(ValueError, match=error):
        hull_module.load_candidate_structures(export, manifest)


def test_missing_manifest_never_falls_back_to_scanning(hull_module, tmp_path):
    make_cif(tmp_path / "previous_run.cif")
    with pytest.raises(FileNotFoundError):
        hull_module.load_candidate_structures(tmp_path, tmp_path / "missing_index.csv")


def test_manifest_requires_cif_column(hull_module, tmp_path):
    manifest = tmp_path / "export_index.csv"
    manifest.write_text("ref,filename\nsource.extxyz::0,selected.cif\n")
    with pytest.raises(ValueError, match="cif column"):
        hull_module.load_candidate_structures(tmp_path, manifest)
