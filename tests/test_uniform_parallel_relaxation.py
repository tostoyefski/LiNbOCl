"""Uniform relaxation precedes every isolated worker's single-point energy."""
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import pytest
from pymatgen.core import Lattice, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow/pipeline"))
import parallel_screening as screening


def task(identity, species):
    count = len(species)
    value = Structure(Lattice.cubic(6), species,
                      [[i / count] * 3 for i in range(count)])
    return {"id": identity, "path": "/source/example.cif", "chemsys": "Cl-Li",
            "structure": value.as_dict()}


@pytest.fixture
def fake_models(monkeypatch):
    initialized, relaxed, predicted = [], [], []
    @dataclass(frozen=True)
    class Settings:
        checkpoint: str = "MatterSim-v1.0.0-1M.pth"
        fmax: float = .05
        max_steps: int = 500
        def as_dict(self):
            return dict(checkpoint=self.checkpoint, fmax=self.fmax, max_steps=self.max_steps)
    class Relaxer:
        def __init__(self, settings=None, device=None):
            self.settings = settings or Settings()
            initialized.append((self.settings.as_dict(), device))
        def relax(self, value):
            relaxed.append(value.copy())
            if len(value) == 3:
                raise RuntimeError("synthetic relaxation did not converge")
            value = value.copy()
            value.scale_lattice(value.volume * 1.331)
            return value, {"status": "converged", "converged": True,
                           "settings": self.settings.as_dict(), "device": "cuda:0",
                           "optimizer": "FIRE", "cell_filter": "ExpCellFilter", "relax_cell": True,
                           "scalar_pressure_eV_A3": 0.0, "constrain_symmetry": False,
                           "steps": 1, "fmax_final": .01}
    relaxation = types.ModuleType("mattersim_relaxation")
    relaxation.RelaxationSettings = Settings
    relaxation.MatterSimRelaxer = Relaxer
    monkeypatch.setitem(sys.modules, "mattersim_relaxation", relaxation)
    class Model:
        def parameters(self):
            return iter([types.SimpleNamespace(device="cuda:0")])
        def predict_structure(self, value):
            predicted.append(value.copy())
            return {"e": -value.volume / 1000}
    model = types.ModuleType("chgnet.model")
    model.CHGNet = type("CHGNet", (), {"load": staticmethod(lambda **kwargs: Model())})
    monkeypatch.setitem(sys.modules, "chgnet", types.ModuleType("chgnet"))
    monkeypatch.setitem(sys.modules, "chgnet.model", model)
    return Settings, initialized, relaxed, predicted


def test_changed_geometry_and_identical_settings_reach_chgnet_for_candidates_and_references(fake_models):
    Settings, initialized, relaxed, predicted = fake_models
    settings = Settings(checkpoint="custom.pth", fmax=.02, max_steps=20).as_dict()
    tasks = [task("candidate:0", ["Li", "Cl"]), task("reference:Cl-Li:0", ["Li"])]
    result = screening.predict_worker(tasks, {"relaxation_settings": settings})
    assert initialized == [(settings, "cuda:0")]
    assert len(relaxed) == len(predicted) == 2
    assert [record["id"] for record in result] == [value["id"] for value in tasks]
    for original, optimized, record in zip(relaxed, predicted, result):
        assert optimized.lattice.a == pytest.approx(original.lattice.a * 1.1)
        assert Structure.from_dict(record["structure"]).lattice.a == pytest.approx(optimized.lattice.a)
        assert record["relaxation"]["settings"] == settings
        assert record["error"] is None


def test_one_and_multiple_worker_preparation_is_identical(fake_models):
    Settings, initialized, _, _ = fake_models
    context = {"relaxation_settings": Settings().as_dict()}
    tasks = [task(f"candidate:{i}", ["Li", "Cl"]) for i in range(7)]
    single = screening.predict_worker(tasks, context)
    split = [record for shard in [tasks[i::4] for i in range(4)]
             for record in screening.predict_worker(shard, context)]
    by_id = {record["id"]: record for record in split}
    assert single == [by_id[value["id"]] for value in tasks]
    assert len(initialized) == 5
    assert all(value == (context["relaxation_settings"], "cuda:0") for value in initialized)


def test_failed_relaxation_blocks_candidate_and_aborts_reference_hull(fake_models):
    Settings, _, _, predicted = fake_models
    candidate = task("candidate:0", ["Li", "Li", "Cl"])
    reference = task("reference:Cl-Li:0", ["Li", "Li", "Li"])
    result = screening.predict_worker([candidate, reference],
                                      {"relaxation_settings": Settings().as_dict()})
    assert predicted == []
    assert all(record["energy_per_atom"] is None and record["error"] for record in result)
    by_id = {record["id"]: record for record in result}
    rows, entries = screening.calculate_hull([candidate], by_id, {})
    assert rows[0]["calculation_status"] == "calculation_failed"
    assert rows[0]["relaxation_status"] == "failed"
    assert not entries
    with pytest.raises(RuntimeError, match="Incomplete competing-phase"):
        screening.reference_entries([reference], by_id)
