"""Offline regressions for uniform geometry in the retained all-frame hull CLI."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uniform_hull_tool", REPO / "workflow/tools/compute_hull_from_relaxed.py")
tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tool
spec.loader.exec_module(tool)


def crystal(species=("Li", "Cl")):
    coordinates = [[.1, .1, .1], [.6, .6, .6]][:len(species)]
    return Structure(Lattice.cubic(5), species, coordinates)


class MockRelaxer:
    def __init__(self, settings=None, device=None):
        from mattersim_relaxation import RelaxationSettings
        self.settings = settings or RelaxationSettings()
        self.calls = []
    def relax(self, original):
        relaxed = original.copy()
        relaxed.scale_lattice(original.volume * 1.05)
        relaxed.translate_sites(range(len(relaxed)), [.05, .02, .01], frac_coords=True)
        audit = {"status": "converged", "converged": True,
                 "settings": self.settings.as_dict(), "steps": 7, "fmax_final": .01}
        self.calls.append((original, relaxed))
        return relaxed, audit


class RecordingCalculator(Calculator):
    implemented_properties = ["energy"]
    def __init__(self, output=-2):
        super().__init__()
        self.output = output
        self.seen = []
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.seen.append(atoms.copy())
        self.results = {"energy": self.output}


def test_candidate_and_mp_reference_pass_changed_geometry_to_chgnet(tmp_path, monkeypatch):
    relaxer = MockRelaxer()
    candidate, reference = crystal(), crystal(("Li",))
    reference.properties["source_id"] = "mp-1"
    candidate_before, reference_before = candidate.as_dict(), reference.as_dict()
    calculator = RecordingCalculator()
    monkeypatch.setattr(tool, "_chgnet_calculator", lambda device=None: calculator)
    groups = tool.build_entries_chgnet([("/generated/relaxed.extxyz", 3, candidate)], relaxer=relaxer, output_dir=tmp_path)
    reference_entries = tool.entries_from_structures_with_chgnet([reference], relaxer=relaxer, output_dir=tmp_path)
    candidate_entry = groups[tool.get_chemsys(candidate)][0]
    assert len(relaxer.calls) == 2
    assert len(calculator.seen) == 2
    for i, entry in enumerate([candidate_entry, reference_entries[0]]):
        original, relaxed = relaxer.calls[i]
        assert not np.allclose(original.cart_coords, relaxed.cart_coords)
        assert not np.allclose(original.lattice.matrix, relaxed.lattice.matrix)
        assert np.allclose(calculator.seen[i].positions, relaxed.cart_coords)
        assert np.allclose(calculator.seen[i].cell.array, relaxed.lattice.matrix)
        assert np.allclose(entry.structure.cart_coords, relaxed.cart_coords)
        assert entry.data["settings"] == relaxer.settings.as_dict()
        assert entry.data["status"] == "complete"
        assert entry.data["relaxation_audit"]["status"] == "converged"
        saved = Structure.from_file(entry.data["path"])
        assert np.allclose(saved.lattice.matrix, relaxed.lattice.matrix, atol=1e-6)
        assert entry.data["source_id"]
    assert reference_entries[0].entry_id == "mp-1"
    assert candidate.as_dict() == candidate_before
    assert reference.as_dict() == reference_before
    assert len(list(tmp_path.rglob("*.audit.json"))) == 2


def test_candidate_relaxation_failure_is_excluded_and_audited(tmp_path, monkeypatch):
    from mattersim_relaxation import RelaxationError
    class Broken(MockRelaxer):
        def relax(self, original):
            raise RelaxationError("mock nonconvergence", {"status": "failed", "converged": False, "steps": 500, "settings": self.settings.as_dict()})
    def never_load(*args):
        raise AssertionError("CHGNet must not load for a failed relaxation")
    monkeypatch.setattr(tool, "_chgnet_calculator", never_load)
    frames = [("/generated/relaxed.extxyz", 0, crystal())]
    assert dict(tool.build_entries_chgnet(frames, relaxer=Broken(), output_dir=tmp_path)) == {}
    audits = list(tmp_path.rglob("*.audit.json"))
    assert len(audits) == 1
    trace = json.loads(audits[0].read_text())
    assert trace["status"] == "calculation_failed"
    assert trace["relaxation_audit"]["status"] == "failed"
    assert trace["path"] is None
    rows = tool._candidate_failure_rows(frames, tmp_path)
    assert rows[0]["calculation_status"] == "calculation_failed"
    assert rows[0]["source_id"].endswith("#0")


@pytest.mark.parametrize("stage", ["relaxation", "single_point", "nonfinite"])
def test_reference_failure_aborts_instead_of_dropping_phase(tmp_path, monkeypatch, stage):
    class Broken(MockRelaxer):
        def relax(self, original):
            if stage == "relaxation":
                raise RuntimeError("mock reference relaxation failed")
            return super().relax(original)
    calculator = RecordingCalculator(output=float("nan") if stage == "nonfinite" else -2)
    monkeypatch.setattr(tool, "_chgnet_calculator", lambda device=None: calculator)
    if stage == "single_point":
        monkeypatch.setattr(tool, "chgnet_energy_for_structure", lambda *args: (_ for _ in ()).throw(RuntimeError("mock SP failed")))
    with pytest.raises(RuntimeError, match="phase diagram aborted"):
        tool.entries_from_structures_with_chgnet([crystal(("Li",))], relaxer=Broken(), output_dir=tmp_path)
    trace = json.loads(next(tmp_path.rglob("*.audit.json")).read_text())
    assert trace["status"] == "calculation_failed"
    assert "error" in trace


def test_candidate_nonfinite_single_point_never_becomes_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "_chgnet_calculator", lambda device=None: RecordingCalculator(output=float("inf")))
    entries = tool.build_entries_chgnet([("/generated/relaxed.extxyz", 0, crystal())], relaxer=MockRelaxer(), output_dir=tmp_path)
    assert dict(entries) == {}
    trace = json.loads(next(tmp_path.rglob("*.audit.json")).read_text())
    assert trace["status"] == "calculation_failed"
    assert trace["relaxation_audit"]["status"] == "converged"
    assert Path(trace["path"]).exists()


def test_unverified_relaxation_never_reaches_chgnet(tmp_path, monkeypatch):
    class FalseSuccess(MockRelaxer):
        def relax(self, original):
            relaxed, audit = super().relax(original)
            audit["converged"] = False
            return relaxed, audit
    monkeypatch.setattr(tool, "_chgnet_calculator", lambda *a: (_ for _ in ()).throw(AssertionError("must not load CHGNet")))
    assert dict(tool.build_entries_chgnet([("/generated/relaxed.extxyz", 0, crystal())], relaxer=FalseSuccess(), output_dir=tmp_path)) == {}


def test_cli_flags_validate_settings():
    args = tool.parse_args(["--root", ".", "--out", "result.csv", "--mode", "chgnet"])
    assert tool.settings_from_args(args).as_dict()["checkpoint"] == "MatterSim-v1.0.0-1M.pth"
    with pytest.raises(SystemExit):
        tool.parse_args(["--root", ".", "--out", "result.csv", "--mode", "chgnet", "--relax-fmax", "nan"])


def mock_mpr(monkeypatch):
    class MPRester:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    module = types.ModuleType("mp_api.client")
    module.MPRester = MPRester
    monkeypatch.setitem(sys.modules, "mp_api", types.ModuleType("mp_api"))
    monkeypatch.setitem(sys.modules, "mp_api.client", module)


def test_dft_mode_never_loads_or_applies_ml_relaxation(tmp_path, monkeypatch):
    mock_mpr(monkeypatch)
    original = crystal()
    frames = [(str(tmp_path / "relaxed.extxyz"), 0, original)]
    monkeypatch.setattr(tool, "load_relaxed_frames", lambda root: frames)
    uid = frames[0][0] + "#0"
    energies = tmp_path / "energies.csv"
    energies.write_text("id,energy_eV\n" + uid + ",-3\n")
    monkeypatch.setattr(tool, "build_entries_mp", lambda f, e: {"Cl-Li": [ComputedStructureEntry(original, e[uid])]})
    monkeypatch.setattr(tool, "fetch_mp_competitors_entries", lambda *a: [ComputedStructureEntry(crystal(("Li",)), -1), ComputedStructureEntry(crystal(("Cl",)), -1)])
    def forbidden(*args, **kwargs):
        raise AssertionError("DFT branch must not load ML")
    monkeypatch.setattr(tool, "MatterSimRelaxer", forbidden)
    monkeypatch.setattr(tool, "_chgnet_calculator", forbidden)
    output = tmp_path / "mp.csv"
    tool.main(["--root", str(tmp_path), "--out", str(output), "--mode", "mp", "--energies-csv", str(energies)])
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert float(rows[0]["energy_eV_cell"]) == -3
    assert rows[0]["relaxation_status"] == "not_applicable"
    assert not (tmp_path / "relaxation").exists()


def test_main_uses_one_relaxer_for_both_sides_and_outputs_provenance(tmp_path, monkeypatch):
    mock_mpr(monkeypatch)
    instances = []
    class Shared(MockRelaxer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)
    monkeypatch.setattr(tool, "MatterSimRelaxer", Shared)
    monkeypatch.setattr(tool, "load_relaxed_frames", lambda root: [(str(tmp_path / "relaxed.extxyz"), 0, crystal())])
    monkeypatch.setattr(tool, "fetch_mp_competitors_structures", lambda *a: [crystal(("Li",)), crystal(("Cl",))])
    monkeypatch.setattr(tool, "_chgnet_calculator", lambda device=None: RecordingCalculator(output=-2))
    output = tmp_path / "chgnet.csv"
    tool.main(["--root", str(tmp_path), "--out", str(output), "--mode", "chgnet"])
    assert len(instances) == 1
    assert len(instances[0].calls) == 3
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    assert rows[0]["source_id"].endswith("#0")
    assert rows[0]["relaxation_status"] == "converged"
    assert rows[0]["calculation_status"] == "complete"
    assert json.loads(rows[0]["relaxation_settings"]) == instances[0].settings.as_dict()
    assert Path(rows[0]["path"]).is_relative_to(tmp_path / "relaxation" / "all_frames")
    assert len(list((tmp_path / "relaxation" / "all_frames").rglob("*.cif"))) == 3


def test_aux_calculator_pins_same_chgnet_checkpoint_as_pipeline(monkeypatch):
    calls = []
    model_instance = object()
    class Model:
        @staticmethod
        def load(**kwargs):
            calls.append(("model", kwargs))
            return model_instance
    class Calc:
        def __init__(self, **kwargs):
            calls.append(("calculator", kwargs))
    model_module = types.ModuleType("chgnet.model")
    model_module.CHGNet = Model
    dynamics_module = types.ModuleType("chgnet.model.dynamics")
    dynamics_module.CHGNetCalculator = Calc
    monkeypatch.setitem(sys.modules, "chgnet", types.ModuleType("chgnet"))
    monkeypatch.setitem(sys.modules, "chgnet.model", model_module)
    monkeypatch.setitem(sys.modules, "chgnet.model.dynamics", dynamics_module)
    tool._chgnet_calculator("cpu")
    assert calls[0] == ("model", {"model_name": "0.3.0", "use_device": "cpu"})
    assert calls[1] == ("calculator", {"model": model_instance, "use_device": "cpu"})


def test_reference_failure_clears_old_ml_output_and_preserves_sources(tmp_path, monkeypatch):
    mock_mpr(monkeypatch)
    original = crystal()
    before = original.as_dict()
    class ReferenceFailure(MockRelaxer):
        def relax(self, value):
            if len(value) == 1:
                raise RuntimeError("mock reference nonconvergence")
            return super().relax(value)
    monkeypatch.setattr(tool, "MatterSimRelaxer", ReferenceFailure)
    monkeypatch.setattr(tool, "load_relaxed_frames", lambda root: [(str(tmp_path / "relaxed.extxyz"), 0, original)])
    monkeypatch.setattr(tool, "fetch_mp_competitors_structures", lambda *a: [crystal(("Li",))])
    monkeypatch.setattr(tool, "_chgnet_calculator", lambda device=None: RecordingCalculator())
    output = tmp_path / "old_ml.csv"
    output.write_text("old successful hull result\n")
    with pytest.raises(RuntimeError, match="phase diagram aborted"):
        tool.main(["--root", str(tmp_path), "--out", str(output), "--mode", "chgnet"])
    assert not output.exists()
    assert original.as_dict() == before
    audit = json.loads(next((tmp_path / "relaxation/all_frames/references").glob("*.audit.json")).read_text())
    assert audit["status"] == "calculation_failed"


def test_failed_dft_run_preserves_previous_output_semantics(tmp_path, monkeypatch):
    mock_mpr(monkeypatch)
    monkeypatch.setattr(tool, "load_relaxed_frames", lambda root: [(str(tmp_path / "relaxed.extxyz"), 0, crystal())])
    output = tmp_path / "old_dft.csv"
    output.write_text("old DFT hull result\n")
    with pytest.raises(SystemExit, match="energies-csv"):
        tool.main(["--root", str(tmp_path), "--out", str(output), "--mode", "mp"])
    assert output.read_text() == "old DFT hull result\n"
