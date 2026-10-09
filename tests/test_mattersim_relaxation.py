"""Offline tests: real ASE optimizers/calculators, without MatterSim or Torch."""
import argparse
import importlib.util
from pathlib import Path
import sys
import types

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
import numpy as np
from pymatgen.core import Lattice, Structure
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "pipeline"
sys.path.insert(0, str(SCRIPTS))
import mattersim_relaxation as relaxation


@pytest.fixture(autouse=True)
def clear_model_cache():
    relaxation._MODEL_CACHE.clear()
    yield
    relaxation._MODEL_CACHE.clear()


def structure():
    return Structure(Lattice.cubic(4), ["Li", "Cl"], [[0, 0, 0], [.5, .5, .5]])


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3)), "stress": np.zeros(6)}


def install_calculator(monkeypatch, calculator):
    calls = []
    def load(checkpoint, device):
        calls.append((checkpoint, device))
        return relaxation._ModelBundle(object(), calculator, device or "cpu", {"ase": "test", "mattersim": "mock"})
    monkeypatch.setattr(relaxation, "_load_matter_sim", load)
    return calls


def test_settings_cli_round_trip_and_exact_default_1m():
    parser = argparse.ArgumentParser()
    relaxation.add_relaxation_arguments(parser)
    defaults = relaxation.settings_from_args(parser.parse_args([]))
    assert defaults.as_dict() == {"checkpoint": "MatterSim-v1.0.0-1M.pth", "fmax": .05, "max_steps": 500}
    assert relaxation.RelaxationSettings(**defaults.as_dict()) == defaults
    args = parser.parse_args(["--mattersim-checkpoint", "/models/custom.pth", "--relax-fmax", ".03", "--relax-steps", "20"])
    assert relaxation.settings_from_args(args) == relaxation.RelaxationSettings("/models/custom.pth", .03, 20)


@pytest.mark.parametrize("kwargs", [{"checkpoint": ""}, {"fmax": float("nan")}, {"fmax": 0}, {"fmax": -1}, {"max_steps": 0}, {"max_steps": 1.5}, {"max_steps": True}])
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        relaxation.RelaxationSettings(**kwargs)


def test_import_and_constructor_do_not_import_gpu_libraries(monkeypatch):
    attempted = []
    import builtins
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "mattersim"):
            attempted.append(name)
            raise AssertionError("GPU libraries must remain lazy")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("isolated_relaxation", SCRIPTS / "mattersim_relaxation.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.MatterSimRelaxer()
    assert attempted == []


def test_shared_model_and_same_settings_for_input_copies(monkeypatch):
    calls = install_calculator(monkeypatch, ZeroCalculator())
    settings = relaxation.RelaxationSettings()
    original = structure()
    before = original.as_dict()
    a, audit_a = relaxation.MatterSimRelaxer(settings, "cpu").relax(original)
    b, audit_b = relaxation.MatterSimRelaxer(settings, "cpu").relax(original.copy())
    assert calls == [(settings.checkpoint, "cpu")]
    assert original.as_dict() == before
    assert a.as_dict() == b.as_dict()
    assert audit_a == audit_b
    assert audit_a["status"] == "converged"
    assert audit_a["steps"] == 0
    assert audit_a["settings"] == settings.as_dict()
    assert audit_a["optimizer"] == "FIRE"
    assert audit_a["cell_filter"] == "ExpCellFilter"
    assert audit_a["constrain_symmetry"] is False
    assert audit_a["scalar_pressure_eV_A3"] == 0


def test_real_ase_optimizer_returns_changed_cell_and_coordinates(monkeypatch):
    install_calculator(monkeypatch, LennardJones(sigma=2.5, epsilon=.25, rc=6.0))
    original = Structure(Lattice.cubic(4.5), ["Cu"] * 4,
                         [[.015, .005, .002], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    before = original.as_dict()
    settings = relaxation.RelaxationSettings(fmax=.03, max_steps=500)
    relaxed, audit = relaxation.MatterSimRelaxer(settings, "cpu").relax(original)
    assert audit["converged"] is True
    assert audit["steps"] > 0
    assert audit["fmax_final"] <= settings.fmax
    assert not np.allclose(relaxed.lattice.matrix, original.lattice.matrix)
    assert not np.allclose(relaxed.frac_coords, original.frac_coords)
    assert relaxed.composition == original.composition
    assert original.as_dict() == before


class StubOptimizer:
    run_result = True
    nsteps = 500
    mutation = None
    def __init__(self, cell_filter, logfile=None):
        self.atoms = cell_filter.atoms
        self.filter = cell_filter
    def attach(self, callback, interval=1):
        self.callback = callback
    def run(self, fmax, steps):
        if self.mutation:
            self.mutation(self.atoms)
        return self.run_result
    def get_number_of_steps(self):
        return self.nsteps


def test_success_on_last_permitted_step_uses_ase_convergence_bool(monkeypatch):
    install_calculator(monkeypatch, ZeroCalculator())
    monkeypatch.setattr(relaxation, "FIRE", StubOptimizer)
    _, audit = relaxation.MatterSimRelaxer().relax(structure())
    assert audit["converged"] is True
    assert audit["steps"] == 500


def test_nonconvergence_never_falls_back_to_input(monkeypatch):
    install_calculator(monkeypatch, ZeroCalculator())
    class NonConverged(StubOptimizer):
        run_result = False
        nsteps = 1
    monkeypatch.setattr(relaxation, "FIRE", NonConverged)
    with pytest.raises(relaxation.RelaxationError, match="did not converge") as exc:
        relaxation.MatterSimRelaxer().relax(structure())
    assert exc.value.audit["converged"] is False
    assert exc.value.audit["status"] == "failed"
    assert exc.value.audit["steps"] == 1


@pytest.mark.parametrize("bad_property", ["forces", "stress", "energy"])
def test_nonfinite_model_output_raises_and_never_falls_back(monkeypatch, bad_property):
    class BadCalculator(ZeroCalculator):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results[bad_property] = self.results[bad_property] * np.nan
    install_calculator(monkeypatch, BadCalculator())
    with pytest.raises(relaxation.RelaxationError, match="non-finite") as exc:
        relaxation.MatterSimRelaxer().relax(structure())
    assert exc.value.audit["status"] == "failed"
    assert exc.value.audit["converged"] is False


def test_false_success_with_large_final_forces_is_rejected(monkeypatch):
    class ForceCalculator(ZeroCalculator):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["forces"][:] = 1
    install_calculator(monkeypatch, ForceCalculator())
    monkeypatch.setattr(relaxation, "FIRE", StubOptimizer)
    with pytest.raises(relaxation.RelaxationError, match="exceed fmax"):
        relaxation.MatterSimRelaxer().relax(structure())


@pytest.mark.parametrize("mutation,message", [
    (lambda atoms: atoms.set_atomic_numbers([3, 8]), "composition"),
    (lambda atoms: atoms.set_positions(np.full((2, 3), np.nan)), "invalid structure"),
    (lambda atoms: atoms.set_pbc(False), "boundary"),
])
def test_invalid_final_geometry_and_changed_composition_are_rejected(monkeypatch, mutation, message):
    install_calculator(monkeypatch, ZeroCalculator())
    class MutatingOptimizer(StubOptimizer):
        def run(self, fmax, steps):
            mutation(self.atoms)
            return True
    monkeypatch.setattr(relaxation, "FIRE", MutatingOptimizer)
    with pytest.raises(relaxation.RelaxationError, match=message):
        relaxation.MatterSimRelaxer().relax(structure())


def test_invalid_input_is_rejected_before_loading_model(monkeypatch):
    calls = install_calculator(monkeypatch, ZeroCalculator())
    coincident = Structure(Lattice.cubic(4), ["Li", "Cl"], [[0, 0, 0], [0, 0, 0]])
    with pytest.raises(relaxation.RelaxationError, match="coincident"):
        relaxation.MatterSimRelaxer().relax(coincident)
    assert calls == []


def test_model_initialization_error_never_falls_back(monkeypatch):
    def broken(*args):
        raise ImportError("mock MatterSim unavailable")
    monkeypatch.setattr(relaxation, "_load_matter_sim", broken)
    with pytest.raises(relaxation.RelaxationError, match="MatterSim unavailable") as exc:
        relaxation.MatterSimRelaxer().relax(structure())
    assert exc.value.audit["status"] == "failed"


def test_native_api_loads_explicit_1m_once_with_stress(monkeypatch):
    calls = []
    class Potential:
        device = "cpu"
        @classmethod
        def from_checkpoint(cls, **kwargs):
            calls.append(("potential", kwargs))
            return cls()
    class MatterSimCalculator(ZeroCalculator):
        def __init__(self, **kwargs):
            calls.append(("calculator", kwargs))
            super().__init__()
    module = types.ModuleType("mattersim.forcefield.potential")
    module.Potential, module.MatterSimCalculator = Potential, MatterSimCalculator
    monkeypatch.setitem(sys.modules, "mattersim", types.ModuleType("mattersim"))
    monkeypatch.setitem(sys.modules, "mattersim.forcefield", types.ModuleType("mattersim.forcefield"))
    monkeypatch.setitem(sys.modules, "mattersim.forcefield.potential", module)
    helper = relaxation.MatterSimRelaxer(device="cpu")
    helper.relax(structure())
    helper.relax(structure())
    assert len(calls) == 2
    assert calls[0][1] == {"load_path": "MatterSim-v1.0.0-1M.pth", "load_training_state": False, "device": "cpu"}
    assert calls[1][1]["compute_stress"] is True
    assert calls[1][1]["potential"].device == "cpu"


def test_atomic_cif_save_and_write_failure_preserve_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "relaxed" / "candidate.cif"
    assert relaxation.save_relaxed_structure(structure(), path) == path
    saved = Structure.from_file(path)
    assert saved.composition == structure().composition
    previous = path.read_bytes()
    def broken(self, filename, fmt):
        Path(filename).write_text("partial")
        raise RuntimeError("mock write failure")
    monkeypatch.setattr(Structure, "to", broken)
    with pytest.raises(RuntimeError, match="write failure"):
        relaxation.save_relaxed_structure(structure(), path)
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]
