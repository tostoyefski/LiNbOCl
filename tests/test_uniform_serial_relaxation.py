"""Offline integration checks for uniform geometry and cached phase energies."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "workflow/pipeline"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(PIPELINE))
    for name in ("torch", "chgnet", "mp_api"):
        monkeypatch.setitem(sys.modules, name, None)
    loaded = []
    for name in ("compute_ehull_chgnet", "compute_voltage_window"):
        spec = importlib.util.spec_from_file_location(name, PIPELINE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    from mattersim_relaxation import RelaxationSettings
    return *loaded, RelaxationSettings()


def make_structure(species, offset=0):
    coords = [[offset + index * 0.2] * 3 for index in range(len(species))]
    return Structure(Lattice.cubic(5), species, coords)


class MovingRelaxer:
    def __init__(self, settings, fail=None):
        self.settings = settings
        self.fail = fail or (lambda structure: False)
        self.seen = []

    def relax(self, structure):
        self.seen.append(structure.copy())
        if self.fail(structure):
            raise RuntimeError("synthetic relaxation did not converge")
        relaxed = Structure(Lattice(structure.lattice.matrix * 1.1), structure.species,
                            structure.frac_coords + 0.1)
        audit = {"status": "converged", "converged": True, "settings": self.settings.as_dict(),
                 "optimizer": "FIRE", "cell_filter": "ExpCellFilter", "relax_cell": True,
                 "scalar_pressure_eV_A3": 0.0, "constrain_symmetry": False,
                 "steps": 3, "fmax_final": 0.01, "atomic_fmax_final": 0.01,
                 "stress_max_abs_eV_A3": 0, "device": "offline", "versions": {}}
        return relaxed, audit


class GeometryEnergyModel:
    def __init__(self, fail=None):
        self.seen = []
        self.fail = fail or (lambda structure: False)

    def predict_structure(self, structure):
        # Energies are deliberately defined only at the moved cell/coordinates.
        assert structure.lattice.a == pytest.approx(5.5)
        assert structure.frac_coords[0, 0] >= 0.099999
        self.seen.append(structure.copy())
        if self.fail(structure):
            raise RuntimeError("synthetic CHGNet single-point failure")
        amounts = structure.composition.get_el_amt_dict()
        if amounts == {"Li": 1}:
            total = -1
        elif amounts == {"Cl": 1}:
            total = 0
        elif amounts == {"Li": 2, "Cl": 1}:
            total = -3.5
        elif amounts == {"Li": 1, "Cl": 1}:
            total = -1.5 if structure.frac_coords[0, 0] > 0.2 else -2
        else:
            total = -len(structure)
        return {"e": total / len(structure)}


def run_example(modules, tmp_path, fail_relax=None, fail_energy=None, second=False):
    hull, voltage, settings = modules
    source = tmp_path / "exported" / "candidate.cif"
    source.parent.mkdir(parents=True)
    structure = make_structure(["Li", "Cl"])
    structure.to(filename=str(source), fmt="cif")
    candidates = [(source, structure)]
    if second:
        second_path = source.with_name("excluded.cif")
        second_structure = make_structure(["Li", "Cl"], offset=0.2)
        second_structure.to(filename=str(second_path), fmt="cif")
        candidates.append((second_path, second_structure))
    refs = [make_structure(["Li"]), make_structure(["Cl"]), make_structure(["Li", "Li", "Cl"])]
    model = GeometryEnergyModel(fail_energy)
    relaxer = MovingRelaxer(settings, fail_relax)
    rows, snapshot = hull.run_uniform_hull(candidates, model, relaxer, lambda csys: refs,
                                           tmp_path / "run" / "hull.csv")
    return rows, snapshot, model, relaxer


def write_rows(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_voltage(voltage, rows, tmp_path, extra=None):
    path = write_rows(tmp_path / "run" / "stable.csv", rows)
    output = path.with_name("voltage.csv")
    voltage.main(["--stable-csv", str(path), "--out", str(output),
                  "--voltage-max", "1.5", "--voltage-step", "0.5", "--target-voltage", "0.75", *(extra or [])])
    with output.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_both_sides_relax_identically_before_chgnet_and_voltage_reuses_entire_snapshot(modules, tmp_path):
    hull, voltage, settings = modules
    rows, snapshot, model, relaxer = run_example(modules, tmp_path, second=True)
    assert len(model.seen) == len(relaxer.seen) == 5
    assert all(struct.lattice.a == pytest.approx(5.5) for struct in model.seen)
    assert all(struct.lattice.a == pytest.approx(5) for struct in relaxer.seen)
    assert rows[0]["path"] != rows[0]["source_path"]
    assert Path(rows[0]["path"]).parent.name == "candidates"
    assert hull.structure_file_sha256(rows[0]["path"]) == rows[0]["structure_sha256"]
    assert snapshot["energy_model"] == "CHGNet-0.3.0"
    assert snapshot["relaxation_settings"] == settings.as_dict()
    assert len(snapshot["entries_by_chemsys"]["Cl-Li"]) == 5
    assert rows[1]["energy_above_hull_eV"] > 0.05
    # Only the accepted candidate is in the filtered CSV; the excluded generated
    # phase remains in the reused diagram, and no model/API packages are needed.
    actual = run_voltage(voltage, [rows[0]], tmp_path)[0]
    assert actual["window_status"] == "stable_window"
    assert float(actual["V_red"]) == 0.5
    assert float(actual["V_ox"]) == 1
    assert actual["stable_at_target"] == "True"
    assert len(model.seen) == len(relaxer.seen) == 5
    assert actual["path"] == rows[0]["path"]
    assert actual["reference_snapshot_path"] == rows[0]["reference_snapshot_path"]


@pytest.mark.parametrize("failure_stage", ["relaxation", "single_point"])
def test_one_failed_mp_reference_invalidates_system_and_is_audited(modules, tmp_path, failure_stage):
    is_reference = lambda structure: len(structure) == 3
    rows, snapshot, model, relaxer = run_example(
        modules, tmp_path,
        fail_relax=is_reference if failure_stage == "relaxation" else None,
        fail_energy=is_reference if failure_stage == "single_point" else None,
    )
    assert rows[0]["hull_status"] == "calculation_failed"
    assert rows[0]["is_stable"] == 0
    assert "energy_above_hull_eV" not in rows[0]
    assert snapshot["systems"]["Cl-Li"]["status"] == "calculation_failed"
    assert "Cl-Li" not in snapshot["entries_by_chemsys"]
    audit = json.loads((tmp_path / "run/relaxation/audit.json").read_text())["records"]
    assert len(audit) == 4
    assert any(row["role"] == "reference" and row["error"] for row in audit)
    actual = run_voltage(modules[1], rows, tmp_path)[0]
    assert actual["window_status"] == "calculation_failed"
    assert actual["window"] == actual["stable_at_target"] == ""


def test_failed_candidate_is_explicit_and_never_an_unrelaxed_entry(modules, tmp_path):
    rows, snapshot, model, relaxer = run_example(
        modules, tmp_path, second=True,
        fail_relax=lambda structure: structure.frac_coords[0, 0] > 0.15,
    )
    failed = next(row for row in rows if row["file"] == "excluded.cif")
    assert failed["hull_status"] == "calculation_failed"
    assert failed["relaxation_status"] == "failed"
    assert "path" not in failed
    assert "energy_total_eV" not in failed
    assert all(struct.frac_coords[0, 0] < 0.2 for struct in model.seen)
    assert len(snapshot["candidates"]) == 1
    assert next(row for row in rows if row["file"] == "candidate.cif")["hull_status"] == "complete"


@pytest.mark.parametrize("tamper", ["candidate_energy", "settings", "geometry", "cached_geometry", "missing_file", "missing_reference", "method", "fmax", "row_audit", "entry_energy", "missing_phase", "legacy_csv", "missing_snapshot"])
def test_voltage_rejects_stale_mixed_or_incomplete_input(modules, tmp_path, tamper):
    hull, voltage, settings = modules
    rows, snapshot, _, _ = run_example(modules, tmp_path)
    extra = []
    snapshot_path = Path(rows[0]["reference_snapshot_path"])
    if tamper == "candidate_energy":
        rows[0]["energy_total_eV"] += 0.25
    elif tamper == "settings":
        extra = ["--relax-fmax", "0.1"]
    elif tamper == "geometry":
        with Path(snapshot["references"][0]["path"]).open("a") as handle:
            handle.write("\n# geometry file changed\n")
    elif tamper == "entry_energy":
        snapshot["entries_by_chemsys"]["Cl-Li"][0]["energy"] += 0.2
        hull.write_reference_snapshot(snapshot, snapshot_path)
    elif tamper == "missing_file":
        Path(snapshot["references"][0]["path"]).unlink()
    elif tamper == "cached_geometry":
        original = Structure.from_dict(snapshot["candidates"][0]["structure"])
        original.translate_sites([0], [0.025, 0, 0], frac_coords=True)
        snapshot["candidates"][0]["structure"] = original.as_dict()
        hull.write_reference_snapshot(snapshot, snapshot_path)
    elif tamper == "missing_reference":
        snapshot["references"].pop()
        hull.write_reference_snapshot(snapshot, snapshot_path)
    elif tamper == "method":
        snapshot["references"][0]["relaxation"]["relax_cell"] = False
        hull.write_reference_snapshot(snapshot, snapshot_path)
    elif tamper == "fmax":
        snapshot["references"][0]["relaxation"]["fmax_final"] = 0.1
        hull.write_reference_snapshot(snapshot, snapshot_path)
    elif tamper == "row_audit":
        audit = json.loads(rows[0]["relaxation_audit_json"])
        audit["steps"] += 1
        rows[0]["relaxation_audit_json"] = json.dumps(audit)
    elif tamper == "missing_phase":
        snapshot["entries_by_chemsys"]["Cl-Li"].pop()
        hull.write_reference_snapshot(snapshot, snapshot_path)
    elif tamper == "legacy_csv":
        rows = [{key: rows[0][key] for key in ("file", "path", "chemsys", "energy_total_eV")}]
    elif tamper == "missing_snapshot":
        snapshot_path.unlink()
    actual = run_voltage(voltage, rows, tmp_path, extra)[0]
    assert actual["window_status"] == "calculation_failed"
    assert actual["window"] == actual["stable_at_target"] == ""
    assert actual["error"]


def test_saved_optimized_geometry_hash_and_energy_are_validated(modules, tmp_path):
    hull, _, settings = modules
    _, snapshot, _, _ = run_example(modules, tmp_path)
    for record in snapshot["references"] + snapshot["candidates"]:
        geometry = hull.validate_relaxed_record(record, settings)
        assert geometry.lattice.a == pytest.approx(5.5)
        assert np.allclose(geometry.frac_coords[0], [0.1] * 3)
        assert record["energy_total_eV"] == pytest.approx(record["energy_per_atom_eV"] * len(geometry))


def test_global_pool_keeps_generated_subsystem_phases(modules, tmp_path):
    hull, voltage, settings = modules
    smaller = make_structure(["Li", "Cl"])
    larger = make_structure(["Li", "Nb", "Cl"])
    refs = [make_structure(["Li"]), make_structure(["Cl"]), make_structure(["Nb"])]
    rows, snapshot = hull.run_uniform_hull(
        [(tmp_path / "smaller.cif", smaller), (tmp_path / "larger.cif", larger)],
        GeometryEnergyModel(), MovingRelaxer(settings),
        lambda csys: [ref for ref in refs if set(ref.composition.get_el_amt_dict()) <= set(csys.split("-"))],
        tmp_path / "run" / "hull.csv",
    )
    entries, records = voltage.snapshot_phase_entries(snapshot, "Cl-Li-Nb", settings)
    assert len(entries) == 5
    assert {record["file"] for record in records} == {"smaller.cif", "larger.cif"}


@pytest.mark.parametrize("missing", ["material_id", "structure"])
def test_fetched_mp_document_missing_identity_or_geometry_is_never_silently_dropped(modules, monkeypatch, missing):
    hull = modules[0]
    document = {"material_id": "mp-offline", "structure": make_structure(["Cl"]), "is_stable": True}
    document.pop(missing)
    monkeypatch.setattr(hull, "search_with_retry", lambda mpr, **kwargs: [document])
    with pytest.raises(ValueError, match="missing material_id|has no structure"):
        hull.fetch_mp_competitor_structures("Cl-Li", object())


def test_nonorthogonal_rotated_cell_and_reordered_periodic_sites_survive_real_cif_roundtrip(modules, tmp_path):
    hull, _, settings = modules
    from mattersim_relaxation import save_relaxed_structure

    angle = np.deg2rad(37)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    lattice = Lattice(Lattice.from_parameters(5, 6, 7, 80, 95, 105).matrix @ rotation)
    original = Structure(
        lattice, ["Li", "Cl", "Li", "Cl"],
        [[0.9, 0.1, 0.2], [-0.2, 0.3, 0.5], [1.15, 0.85, 0.75], [0.45, 1.35, -0.1]],
    )
    relaxed, audit = MovingRelaxer(settings).relax(original)
    path = tmp_path / "nonorthogonal.cif"
    save_relaxed_structure(relaxed, path)
    saved = Structure.from_file(path)
    assert not np.allclose(saved.lattice.matrix, relaxed.lattice.matrix)
    assert [site.species_string for site in saved] != [site.species_string for site in relaxed]
    assert np.any(relaxed.frac_coords < 0) and np.any(relaxed.frac_coords > 1)
    record = {
        "path": str(path), "structure": relaxed.as_dict(),
        "structure_sha256": hull.structure_file_sha256(path), "relaxation": audit,
        "energy_total_eV": -4, "energy_per_atom_eV": -1,
    }
    verified = hull.validate_relaxed_record(record, settings)
    assert verified.composition == relaxed.composition
