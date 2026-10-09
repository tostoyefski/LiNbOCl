"""Read local MatterGen structure releases for bounded novelty comparisons.

The loaders never download data or run a potential.  Training membership refers
to the explicitly selected release splits, not an unverifiable checkpoint
membership list.  CSV composition metadata shortlists structures; the returned
composition is always checked against the actual geometry.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from pymatgen.core import Composition, Element, Structure


@dataclass
class NoveltyReference:
    reference_id: str
    structure: Structure
    split: str | None
    row_index: int | None
    member: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def composition_key(self) -> str:
        return composition_key(self.structure)


@dataclass
class NoveltySourceResult:
    references: list[NoveltyReference]
    manifest: dict[str, Any]


def composition_key(value: Composition | Structure | str | dict[str, float]) -> str:
    """Element fractions, independent of formula spelling and supercell size."""
    composition = value.composition if isinstance(value, Structure) else value
    composition = Composition(composition).element_composition
    counts = composition.get_el_amt_dict()
    total = sum(counts.values())
    if total <= 0 or not math.isfinite(total) or any(
        not math.isfinite(count) or count <= 0 for count in counts.values()
    ):
        raise ValueError("Composition must have positive finite element counts")
    return json.dumps(
        {symbol: round(count / total, 12) for symbol, count in sorted(counts.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def _target_compositions(values: Iterable[Any]) -> tuple[set[str], set[str]]:
    keys, systems = set(), set()
    for value in values:
        key = composition_key(value)
        keys.add(key)
        systems.add("-".join(sorted(json.loads(key))))
    return keys, systems


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path, role: str, format_name: str) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "source_role": role,
        "format": format_name,
        "status": "incomplete",
        "coverage_complete": False,
        "sha256": None,
        "size_bytes": None,
        "scanned_rows": 0,
        "selected_rows": 0,
        "comparable_rows": 0,
        "metadata_only_rows": 0,
        "splits": {},
        "errors": [],
        "training_membership": "selected_public_release_splits_only" if role == "training" else None,
        "exact_checkpoint_training_membership_verified": False,
        "shortlisting_policy": "chemical_system_metadata_index; actual_geometry_composition_verified_for_shortlisted_rows",
    }


def _error(manifest: dict[str, Any], reason: str, **details: Any) -> None:
    manifest["errors"].append({"reason": reason, **details})


def _prepare(path: Path, manifest: dict[str, Any]) -> bool:
    if not path.is_file():
        manifest["status"] = "unavailable"
        _error(manifest, "source_file_unavailable",
               error=f"Required structure dataset file is missing or not a regular file: {path.resolve()}")
        return False
    manifest["size_bytes"] = path.stat().st_size
    manifest["sha256"] = _digest(path)
    with path.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            manifest["status"] = "unavailable"
            _error(manifest, "git_lfs_pointer_has_no_structures",
                   error="The file is a Git LFS pointer; the actual structure dataset is unavailable")
            return False
    return True


def _finish(result: NoveltySourceResult) -> NoveltySourceResult:
    manifest = result.manifest
    manifest["comparable_rows"] = len(result.references)
    if manifest["status"] != "unavailable":
        manifest["coverage_complete"] = not manifest["errors"]
        manifest["status"] = "complete" if manifest["coverage_complete"] else "incomplete"
    return result


def _present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "nan", "none", "null"}


def _row_formula(row: dict[str, Any]) -> Any:
    return next(
        (row[name] for name in ("reduced_formula", "pretty_formula", "formula") if _present(row.get(name))),
        None,
    )


def _row_chemsys(row: dict[str, Any]) -> str | None:
    value = row.get("chemical_system") or row.get("chemsys")
    if not _present(value):
        return None
    parts = str(value).strip().split("-")
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("Invalid chemical-system metadata")
    for part in parts:
        Element(part)
    return "-".join(sorted(parts))


def _geometry(row: dict[str, Any]) -> Structure:
    cif = row.get("cif")
    if _present(cif):
        return Structure.from_str(cif, fmt="cif")
    structure = row.get("structure")
    if not _present(structure):
        raise ValueError("No CIF or structure geometry")
    if isinstance(structure, str):
        structure = json.loads(structure)
    return Structure.from_dict(structure)


def _validate_geometry(structure: Structure) -> None:
    if len(structure) == 0:
        raise ValueError("Empty structure")
    if not all(math.isfinite(float(value)) for vector in structure.lattice.matrix for value in vector):
        raise ValueError("Nonfinite lattice")
    if not math.isfinite(structure.volume) or structure.volume <= 0:
        raise ValueError("Nonfinite or singular lattice")
    if not all(math.isfinite(float(value)) for vector in structure.frac_coords for value in vector):
        raise ValueError("Nonfinite coordinates")


def _scan_csv(
    handle: Any,
    result: NoveltySourceResult,
    member: str,
    split: str | None,
    target_keys: set[str],
    target_systems: set[str],
    *,
    metadata_only: bool = False,
) -> None:
    manifest = result.manifest
    reader = csv.DictReader(handle)
    fields = reader.fieldnames or []
    counts = manifest["splits"].setdefault(
        split or member, {"members": [], "scanned_rows": 0, "selected_rows": 0, "comparable_rows": 0}
    )
    counts["members"].append(member)
    has_geometry = "cif" in fields or "structure" in fields
    if not has_geometry and not metadata_only:
        _error(manifest, "csv_has_no_structure_column", member=member, split=split)
    for row_index, row in enumerate(reader):
        counts["scanned_rows"] += 1
        manifest["scanned_rows"] += 1
        if metadata_only or not has_geometry:
            manifest["metadata_only_rows"] += 1
            continue
        details = {"member": member, "split": split, "row_index": row_index}
        try:
            chemsys = _row_chemsys(row)
            if chemsys is not None and chemsys not in target_systems:
                continue
            formula = _row_formula(row)
            if formula is None:
                raise ValueError("No composition metadata to determine relevance")
            row_key = composition_key(formula)
            actual_system = "-".join(sorted(json.loads(row_key)))
            if chemsys is not None and actual_system != chemsys:
                raise ValueError("Formula and chemical-system metadata disagree")
            if actual_system not in target_systems:
                continue
        except Exception as exc:
            _error(manifest, "uncertain_composition", error=f"{type(exc).__name__}: {exc}", **details)
            continue
        counts["selected_rows"] += 1
        manifest["selected_rows"] += 1
        try:
            structure = _geometry(row)
            if composition_key(structure) != row_key:
                raise ValueError("Geometry composition differs from CSV metadata")
            _validate_geometry(structure)
            identifier = next(
                (str(row[name]) for name in ("material_id", "structure_id", "entry_id", "id") if _present(row.get(name))),
                f"{member}:{row_index}",
            )
            result.references.append(
                NoveltyReference(identifier, structure, split, row_index, member, {
                    "row_id": row.get(""), "formula": formula, "chemical_system": actual_system
                })
            )
            counts["comparable_rows"] += 1
        except Exception as exc:
            _error(manifest, "selected_structure_unavailable", error=f"{type(exc).__name__}: {exc}", **details)


def load_training_archive(
    path: str | Path,
    candidate_compositions: Iterable[Any],
    *,
    training_splits: Iterable[str] = ("train",),
) -> NoveltySourceResult:
    """Scan selected train/val/test CSV splits of a local official release ZIP.

    A plain CSV is accepted when its stem identifies a requested split.  The
    official ZIP's ref.csv has no geometries and is counted only as metadata.
    """
    path = Path(path)
    result = NoveltySourceResult([], _manifest(path, "training", "csv_zip"))
    manifest = result.manifest
    requested = tuple(dict.fromkeys(training_splits))
    manifest["requested_splits"] = list(requested)
    try:
        target_keys, target_systems = _target_compositions(candidate_compositions)
        manifest["candidate_composition_keys"] = sorted(target_keys)
        if not requested or any(split not in {"train", "val", "test"} for split in requested):
            raise ValueError("training_splits must explicitly name train, val, or test")
        if not _prepare(path, manifest):
            return result
        found: set[str] = set()
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                if len(members) != len(set(members)):
                    _error(manifest, "duplicate_archive_member_names")
                for member in sorted(members):
                    if not member.lower().endswith(".csv"):
                        continue
                    split = Path(member).stem
                    if split not in requested:
                        if split == "ref":
                            with archive.open(member) as raw:
                                _scan_csv(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), result,
                                          member, None, target_keys, target_systems, metadata_only=True)
                        continue
                    if split in found:
                        _error(manifest, "multiple_members_for_requested_split", split=split)
                    found.add(split)
                    with archive.open(member) as raw:
                        _scan_csv(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), result,
                                  member, split, target_keys, target_systems)
        else:
            manifest["format"] = "csv"
            split = path.stem
            if split in requested:
                found.add(split)
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    _scan_csv(handle, result, path.name, split, target_keys, target_systems)
        manifest["available_requested_splits"] = sorted(found)
        for split in requested:
            if split not in found:
                _error(manifest, "requested_split_unavailable", split=split)
            elif manifest["splits"].get(split, {}).get("scanned_rows", 0) == 0:
                _error(manifest, "requested_split_empty", split=split)
    except Exception as exc:
        _error(manifest, "source_read_failed", error=f"{type(exc).__name__}: {exc}")
    return _finish(result)


def import_mattergen_module(module_name: str) -> ModuleType:
    """Reuse installed MatterGen, or this checkout when no real package exists.

    Missing third-party dependencies propagate as explicit errors.  A real
    installed package is never silently replaced with a different version.
    """
    if module_name != "mattergen" and not module_name.startswith("mattergen."):
        raise ValueError("Only MatterGen module names are accepted")
    # A checkout's outer mattergen/ directory can otherwise be imported as an
    # empty namespace package.  Prefer a real installed package, or select the
    # actual bundled source before importing the official utility functions.
    mattergen_spec = importlib.util.find_spec("mattergen")
    if mattergen_spec is None or mattergen_spec.origin is None:
        mattergen_source = Path(__file__).resolve().parents[2] / "mattergen"
        if not (mattergen_source / "mattergen" / "__init__.py").is_file():
            raise ModuleNotFoundError("No installed or bundled MatterGen package available", name="mattergen")
        if str(mattergen_source) not in sys.path:
            sys.path.insert(0, str(mattergen_source))
        namespace = sys.modules.get("mattergen")
        if namespace is not None and getattr(namespace, "__file__", None) is None:
            if not any(name.startswith("mattergen.") for name in sys.modules):
                del sys.modules["mattergen"]
    return importlib.import_module(module_name)


def _read_lmdb(
    path: Path, result: NoveltySourceResult, target_keys: set[str], target_systems: set[str]
) -> None:
    lmdb_api = import_mattergen_module("mattergen.evaluation.utils.lmdb_utils")
    lmdb_get, lmdb_open = lmdb_api.lmdb_get, lmdb_api.lmdb_open

    manifest = result.manifest
    manifest["lmdb_reader"] = "mattergen.evaluation.utils.lmdb_utils.lmdb_open/lmdb_get"
    manifest["lmdb_reader_source"] = lmdb_api.__file__
    with lmdb_open(path, readonly=True) as env:
        with env.begin() as transaction:
            def read(key: str) -> Any:
                return lmdb_get(transaction, key)

            systems = read("chemical_systems")
            if not isinstance(systems, (list, tuple)) or len(set(systems)) != len(systems):
                raise ValueError("Invalid LMDB chemical_systems index")
            if not systems:
                raise ValueError("LMDB chemical_systems index is empty")
            manifest["dataset_name"] = read("name")
            manifest["chemical_system_count"] = len(systems)
            for chemsys in sorted(target_systems):
                prefix = f"{chemsys}."
                encoded_prefix = prefix.encode("ascii")
                actual_keys: list[str] = []
                with transaction.cursor() as cursor:
                    if cursor.set_range(encoded_prefix):
                        for encoded_key in cursor.iternext(values=False):
                            if not encoded_key.startswith(encoded_prefix):
                                break
                            actual_keys.append(encoded_key.decode("ascii"))
                if chemsys not in systems:
                    if actual_keys:
                        _error(manifest, "unindexed_chemical_system", chemical_system=chemsys)
                    continue
                formulas = read(f"{chemsys}.reduced_formulas")
                if not isinstance(formulas, (list, tuple)) or len(set(formulas)) != len(formulas):
                    raise ValueError(f"Invalid reduced_formulas index for {chemsys}")
                expected_keys = {f"{chemsys}.reduced_formulas"}
                for formula in formulas:
                    length = read(f"{chemsys}.{formula}.length")
                    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
                        raise ValueError(f"Invalid LMDB length for {chemsys}.{formula}")
                    expected_keys.add(f"{chemsys}.{formula}.length")
                    expected_keys.update(f"{chemsys}.{formula}.{index}" for index in range(length))
                    manifest["scanned_rows"] += length
                    row_key = composition_key(formula)
                    if "-".join(sorted(json.loads(row_key))) != chemsys:
                        raise ValueError(f"Formula and chemical-system index disagree: {formula}")
                    for index in range(length):
                        key = f"{chemsys}.{formula}.{index}"
                        manifest["selected_rows"] += 1
                        try:
                            entry = read(key)
                            structure = Structure.from_dict(entry["structure"])
                            if composition_key(structure) != row_key:
                                raise ValueError("Geometry composition differs from LMDB index")
                            _validate_geometry(structure)
                            identifier = entry.get("data", {}).get("material_id") or entry.get("entry_id") or key
                            result.references.append(NoveltyReference(
                                str(identifier), structure, None, index, key,
                                {"lmdb_key": key, "formula": formula, "chemical_system": chemsys},
                            ))
                        except Exception as exc:
                            _error(manifest, "selected_structure_unavailable", lmdb_key=key,
                                   error=f"{type(exc).__name__}: {exc}")
                if set(actual_keys) != expected_keys:
                    _error(manifest, "lmdb_scoped_index_key_mismatch", chemical_system=chemsys,
                           missing_key_count=len(expected_keys - set(actual_keys)),
                           extra_key_count=len(set(actual_keys) - expected_keys))
            manifest["scanned_rows_scope"] = "indexed_candidate_chemical_systems"


def load_reference_lmdb(
    path: str | Path,
    candidate_compositions: Iterable[Any],
    *,
    scratch_dir: str | Path | None = None,
) -> NoveltySourceResult:
    """Read a trusted local MatterGen LMDB, optionally gzip compressed.

    MatterGen's official LMDB uses pickled dictionaries.  Only use files from
    trusted sources.  Decompression is streamed to a temporary file and the
    temporary directory is removed when this call returns.
    """
    path = Path(path)
    result = NoveltySourceResult([], _manifest(path, "reference", "lmdb_gzip" if path.suffix == ".gz" else "lmdb"))
    manifest = result.manifest
    try:
        target_keys, target_systems = _target_compositions(candidate_compositions)
        manifest["candidate_composition_keys"] = sorted(target_keys)
        if not _prepare(path, manifest):
            return result
        if path.suffix == ".gz":
            if scratch_dir is not None:
                Path(scratch_dir).mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="novelty-reference-", dir=scratch_dir) as temporary:
                database_path = Path(temporary) / "reference.lmdb"
                with gzip.open(path, "rb") as source, database_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                manifest["decompressed_size_bytes"] = database_path.stat().st_size
                _read_lmdb(database_path, result, target_keys, target_systems)
        else:
            _read_lmdb(path, result, target_keys, target_systems)
    except Exception as exc:
        _error(manifest, "source_read_failed", error=f"{type(exc).__name__}: {exc}")
    return _finish(result)


def load_novelty_source(
    path: str | Path,
    source_kind: str,
    candidate_compositions: Iterable[Any],
    *,
    training_splits: Iterable[str] = ("train",),
    scratch_dir: str | Path | None = None,
) -> NoveltySourceResult:
    """Load a local training release or structural reference for this scope."""
    if source_kind == "training":
        return load_training_archive(path, candidate_compositions, training_splits=training_splits)
    if source_kind != "reference":
        raise ValueError("source_kind must be training or reference")
    path = Path(path)
    if path.suffix.lower() == ".csv":
        result = NoveltySourceResult([], _manifest(path, "reference", "csv"))
        try:
            keys, systems = _target_compositions(candidate_compositions)
            if not _prepare(path, result.manifest):
                return result
            with path.open(encoding="utf-8-sig", newline="") as handle:
                _scan_csv(handle, result, path.name, None, keys, systems)
            if result.manifest["scanned_rows"] == 0:
                _error(result.manifest, "reference_csv_empty")
        except Exception as exc:
            _error(result.manifest, "source_read_failed", error=f"{type(exc).__name__}: {exc}")
        return _finish(result)
    return load_reference_lmdb(path, candidate_compositions, scratch_dir=scratch_dir)
