"""Uniform MatterSim cell/position relaxation before CHGNet energy evaluation.

Both generated candidates and competing MP phases must use this same helper.
MatterSim and Torch are imported only when a validated structure is relaxed.
No failure path returns the original geometry as a successful relaxation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from importlib import metadata
import math
import numbers
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
from ase.filters import ExpCellFilter
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


@dataclass(frozen=True)
class RelaxationSettings:
    checkpoint: str = "MatterSim-v1.0.0-1M.pth"
    fmax: float = 0.05
    max_steps: int = 500

    def __post_init__(self):
        try:
            checkpoint = os.fspath(self.checkpoint)
        except TypeError as exc:
            raise ValueError("MatterSim checkpoint must be a nonempty model name or path") from exc
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            raise ValueError("MatterSim checkpoint must be a nonempty model name or path")
        if isinstance(self.fmax, bool) or not isinstance(self.fmax, numbers.Real):
            raise ValueError("relaxation fmax must be finite and positive")
        if not math.isfinite(float(self.fmax)) or self.fmax <= 0:
            raise ValueError("relaxation fmax must be finite and positive")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, numbers.Integral) or self.max_steps <= 0:
            raise ValueError("relaxation max_steps must be a positive integer")
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "fmax", float(self.fmax))
        object.__setattr__(self, "max_steps", int(self.max_steps))

    def as_dict(self):
        """Only constructor fields, suitable for serializing to worker context."""
        return asdict(self)


def add_relaxation_arguments(parser):
    defaults = RelaxationSettings()
    parser.add_argument("--mattersim-checkpoint", default=defaults.checkpoint,
                        help="MatterSim checkpoint name/path used for all candidates and MP phases")
    parser.add_argument("--relax-fmax", type=float, default=defaults.fmax,
                        help="ASE FIRE convergence threshold for the full-cell filter (default: 0.05)")
    parser.add_argument("--relax-steps", type=int, default=defaults.max_steps,
                        help="Maximum FIRE steps for every candidate and competing phase (default: 500)")
    return parser


def settings_from_args(args):
    defaults = RelaxationSettings()
    return RelaxationSettings(checkpoint=getattr(args, "mattersim_checkpoint", defaults.checkpoint),
                              fmax=getattr(args, "relax_fmax", defaults.fmax),
                              max_steps=getattr(args, "relax_steps", defaults.max_steps))


class RelaxationError(RuntimeError):
    """Failed relaxation with a serializable audit for downstream rejection."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


@dataclass
class _ModelBundle:
    potential: object
    calculator: object
    device: str
    versions: dict


# One potential/calculator per checkpoint and device in each process, reused
# across candidate and reference calls. The PID prevents reuse after a fork.
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _library_versions():
    versions = {}
    for name in ("mattersim", "torch", "ase", "pymatgen", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _load_matter_sim(checkpoint, device):
    # Native API source:
    # https://github.com/microsoft/mattersim/blob/main/src/mattersim/forcefield/potential.py
    # Passing the explicit 1M alias matches Potential.from_checkpoint(None).
    from mattersim.forcefield.potential import MatterSimCalculator, Potential

    kwargs = {"load_training_state": False}
    if device is not None:
        kwargs["device"] = device
    potential = Potential.from_checkpoint(load_path=checkpoint, **kwargs)
    actual_device = str(getattr(potential, "device", device or "unknown"))
    calculator = MatterSimCalculator(potential=potential, device=actual_device,
                                    compute_stress=True, args_dict={})
    return _ModelBundle(potential, calculator, actual_device, _library_versions())


def _validate_structure(structure):
    if not isinstance(structure, Structure):
        raise ValueError("relaxation requires a pymatgen Structure")
    if not structure.num_sites or not structure.is_ordered:
        raise ValueError("relaxation requires a nonempty ordered structure")
    if not all(structure.lattice.pbc):
        raise ValueError("relaxation requires periodicity in all three directions")
    cell = np.asarray(structure.lattice.matrix, dtype=float)
    coords = np.asarray(structure.cart_coords, dtype=float)
    if not np.isfinite(cell).all() or not np.isfinite(coords).all():
        raise ValueError("structure lattice and positions must be finite")
    if not math.isfinite(float(structure.volume)) or structure.volume <= 1e-12:
        raise ValueError("structure cell must have positive nonzero volume")
    if structure.num_sites > 1:
        distances = structure.distance_matrix
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) <= 1e-8:
            raise ValueError("structure contains coincident sites")


def _validate_runtime(atoms, cell_filter, original_composition):
    if not len(atoms) or Counter(atoms.get_atomic_numbers()) != original_composition:
        raise ValueError("relaxation changed the elemental composition")
    if not np.all(atoms.pbc):
        raise ValueError("relaxation changed periodic boundary conditions")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    volume = float(atoms.get_volume())
    if not np.isfinite(cell).all() or not np.isfinite(positions).all() or not math.isfinite(volume) or volume <= 1e-12:
        raise ValueError("relaxation produced a non-finite or invalid structure")
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("MatterSim produced non-finite or invalid atomic forces")
    stress = np.asarray(atoms.get_stress(), dtype=float)
    if stress.shape != (6,) or not np.isfinite(stress).all():
        raise ValueError("MatterSim produced non-finite or invalid cell stress")
    energy = float(atoms.get_potential_energy())
    if not math.isfinite(energy):
        raise ValueError("MatterSim produced a non-finite energy")
    generalized = np.asarray(cell_filter.get_forces(), dtype=float)
    if generalized.shape != (len(atoms) + 3, 3) or not np.isfinite(generalized).all():
        raise ValueError("full-cell filter produced non-finite or invalid forces")
    return {"fmax_final": float(np.max(np.linalg.norm(generalized, axis=1))),
            "atomic_fmax_final": float(np.max(np.linalg.norm(forces, axis=1))),
            "stress_max_abs_eV_A3": float(np.max(np.abs(stress)))}


class MatterSimRelaxer:
    """FIRE + ExpCellFilter, all cell/position degrees, zero applied pressure."""
    def __init__(self, settings=None, device=None):
        self.settings = settings if settings is not None else RelaxationSettings()
        if not isinstance(self.settings, RelaxationSettings):
            raise TypeError("settings must be RelaxationSettings")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device must be a nonempty string or None")
        self.device = device

    def _get_model(self):
        key = (os.getpid(), self.settings.checkpoint, self.device)
        with _MODEL_CACHE_LOCK:
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = _load_matter_sim(self.settings.checkpoint, self.device)
            return _MODEL_CACHE[key]

    def relax(self, structure):
        """Return relaxed geometry and audit; raise on every unverified result.

        ``fmax_final`` is the maximum row norm of the ExpCellFilter generalized
        forces (atomic plus cell degrees), matching the ASE optimizer criterion.
        CHGNet must evaluate the returned structure, never the input geometry.
        """
        audit = {"status": "failed", "converged": False,
                 "settings": self.settings.as_dict(), "optimizer": "FIRE",
                 "cell_filter": "ExpCellFilter", "relax_cell": True,
                 "scalar_pressure_eV_A3": 0.0, "constrain_symmetry": False,
                 "steps": 0, "fmax_final": None, "atomic_fmax_final": None,
                 "stress_max_abs_eV_A3": None, "device": self.device,
                 "versions": _library_versions()}
        optimizer = None
        try:
            _validate_structure(structure)
            # Conversion uses a copy so neither positions/cell nor site
            # properties on the caller's structure are mutated by the optimizer.
            atoms = AseAtomsAdaptor.get_atoms(structure.copy())
            atoms.set_constraint([])
            original_composition = Counter(atoms.get_atomic_numbers())
            bundle = self._get_model()
            atoms.calc = bundle.calculator
            audit.update(device=bundle.device, versions=bundle.versions)
            cell_filter = ExpCellFilter(atoms, mask=[True] * 6,
                                        hydrostatic_strain=False,
                                        constant_volume=False,
                                        scalar_pressure=0.0)
            audit.update(_validate_runtime(atoms, cell_filter, original_composition))
            optimizer = FIRE(cell_filter, logfile=None)
            # Check initial and intermediate model outputs, not only the final
            # state, so NaNs cannot silently propagate through an optimizer.
            def validate_step():
                audit.update(_validate_runtime(atoms, cell_filter, original_composition))
            optimizer.attach(validate_step, interval=1)
            converged = bool(optimizer.run(fmax=self.settings.fmax,
                                          steps=self.settings.max_steps))
            audit["steps"] = int(optimizer.get_number_of_steps())
            audit.update(_validate_runtime(atoms, cell_filter, original_composition))
            if not converged:
                raise RuntimeError(f"ASE FIRE did not converge within {self.settings.max_steps} steps")
            if audit["fmax_final"] > self.settings.fmax * (1 + 1e-7):
                raise RuntimeError("ASE reported convergence but final full-cell forces exceed fmax")
            relaxed = AseAtomsAdaptor.get_structure(atoms)
            _validate_structure(relaxed)
            if relaxed.composition.get_el_amt_dict() != structure.composition.get_el_amt_dict():
                raise ValueError("relaxation changed the elemental composition")
            audit.update(status="converged", converged=True)
            return relaxed, audit
        except Exception as exc:
            if optimizer is not None:
                audit["steps"] = int(optimizer.get_number_of_steps())
            audit["error"] = f"{type(exc).__name__}: {exc}"
            raise RelaxationError("MatterSim relaxation failed: " + audit["error"], audit) from exc


def save_relaxed_structure(structure, path):
    """Validate and atomically save a CIF, preserving old files on write failure."""
    _validate_structure(structure)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".cif", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        structure.to(filename=str(temporary), fmt="cif")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
