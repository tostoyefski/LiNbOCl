#!/usr/bin/env python3
"""Parallel model inference with complete shared bulk and grand-potential hulls."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Element, Structure

from compute_ehull_chgnet import (
    chem_system, get_energy_per_atom, load_candidate_structures, structure_file_sha256,
    validate_relaxation_audit, validate_relaxed_record, write_reference_snapshot,
)
from compute_voltage_window import (
    Candidate, VoltageWindowResult, evaluate_voltage_window,
    generate_voltage_grid, locate_reference_mu,
)

ENERGY_MODEL = "CHGNet-0.3.0"
HULL_FIELDS = ["file", "path", "source_path", "formula", "chemsys", "natoms_cell",
               "energy_per_atom_eV", "energy_total_eV",
               "formation_energy_per_atom_eV", "energy_above_hull_eV",
               "is_stable", "calculation_status", "hull_status", "energy_model",
               "reference_snapshot_path", "relaxation_status", "relaxation_settings_json",
               "relaxation_audit_json", "structure_sha256", "error"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def predict_worker(tasks, context=None):
    # CUDA visibility is set by the parent before importing the model.
    if not tasks:
        return []
    from mattersim_relaxation import RelaxationSettings, MatterSimRelaxer
    settings = RelaxationSettings(**(context or {}).get("relaxation_settings", {}))
    # Within this child's one-device CUDA visibility, the assigned GPU is
    # always logical cuda:0. CHGNet's NVML free-memory selector uses physical
    # indices and must not override the scheduler's device isolation.
    try:
        from chgnet.model import CHGNet
        relaxer = MatterSimRelaxer(settings=settings, device="cuda:0")
        model = CHGNet.load(model_name="0.3.0", use_device="cuda:0", check_cuda_mem=False)
        print(f"[INFO] Energy worker CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} model_device={next(model.parameters()).device}", flush=True)
    except Exception as exc:
        error = f"Worker initialization failed: {type(exc).__name__}: {exc}"
        return [{"id": task["id"], "energy_per_atom": None, "structure": None,
                 "relaxation": {"status": "failed", "converged": False,
                                "settings": settings.as_dict(), "error": error}, "error": error}
                for task in tasks]
    results = []
    for task in tasks:
        preparation_succeeded = False
        optimized, audit = None, {"status": "failed", "converged": False,
                                  "settings": settings.as_dict()}
        try:
            structure = Structure.from_dict(task["structure"])
            optimized, audit = relaxer.relax(structure)
            _validate_geometry(structure, optimized)
            validate_relaxation_audit(audit, settings)
            preparation_succeeded = True
            energy = get_energy_per_atom(model.predict_structure(optimized))
            if not math.isfinite(energy):
                raise ValueError("Non-finite model energy")
            results.append({"id": task["id"], "energy_per_atom": energy,
                            "structure": optimized.as_dict(), "relaxation": audit, "error": None})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "audit"):
                audit = exc.audit
            if not preparation_succeeded:
                optimized = None
                audit = {**audit, "status": "failed", "converged": False, "error": error}
            results.append({"id": task["id"], "energy_per_atom": None,
                            "structure": optimized.as_dict() if optimized is not None else None,
                            "relaxation": audit, "error": error})
    return results


def _validate_geometry(original, optimized):
    if original.composition != optimized.composition or len(original) != len(optimized):
        raise ValueError("Relaxation changed candidate composition or cell atom count")
    if not optimized.is_ordered or optimized.volume <= 0 or not math.isfinite(optimized.volume):
        raise ValueError("Relaxed structure must be ordered with finite positive volume")
    values = list(optimized.lattice.matrix.flat) + list(optimized.frac_coords.flat)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Non-finite relaxed lattice or positions")


def validated_prediction(task, pred, settings=None):
    if pred.get("id") != task["id"]:
        raise ValueError("Prediction identity differs from the input manifest")
    energy = pred.get("energy_per_atom")
    if pred.get("error") or energy is None or not math.isfinite(energy):
        raise ValueError(pred.get("error") or "Non-finite or missing model energy")
    audit = pred.get("relaxation") or {}
    if audit.get("status") != "converged" or audit.get("converged") is not True:
        raise ValueError("Missing converged MatterSim relaxation audit")
    if settings is not None and audit.get("settings") != settings:
        raise ValueError("Relaxation settings differ from the common preparation settings")
    if settings is not None:
        from mattersim_relaxation import RelaxationSettings
        validate_relaxation_audit(audit, RelaxationSettings(**settings))
    structure = Structure.from_dict(pred["structure"])
    _validate_geometry(Structure.from_dict(task["structure"]), structure)
    return structure, float(energy), audit


def reference_entries(reference_tasks, predictions, settings=None):
    grouped = defaultdict(list)
    for task in reference_tasks:
        try:
            structure, energy, _ = validated_prediction(task, predictions[task["id"]], settings)
        except Exception as exc:
            raise RuntimeError(f"Incomplete competing-phase energies for {task['chemsys']}: {task['id']}: {exc}") from exc
        grouped[task["chemsys"]].append(PDEntry(structure.composition, energy * len(structure), name=task["id"]))
    return dict(grouped)


def calculate_hull(candidate_tasks, predictions, references, settings=None):
    """Use every successful candidate of a system in its common phase diagram."""
    rows, entries, grouped = [], {}, defaultdict(list)
    for task in candidate_tasks:
        structure = Structure.from_dict(task["structure"])
        pred = predictions.get(task["id"], {"id": task["id"], "error": "Missing prediction"})
        energy = pred.get("energy_per_atom")
        audit = pred.get("relaxation") or {}
        row = dict(file=Path(task["path"]).name, path=task["path"],
                   source_path=task.get("source_path", task["path"]),
                   formula=structure.composition.reduced_formula, chemsys=task["chemsys"],
                   natoms_cell=len(structure), energy_per_atom_eV=energy,
                   energy_total_eV=None, formation_energy_per_atom_eV=None,
                   energy_above_hull_eV=None, is_stable=0,
                   calculation_status="calculation_failed", hull_status="calculation_failed",
                   energy_model=ENERGY_MODEL,
                   reference_snapshot_path=task.get("reference_snapshot_path", ""),
                   structure_sha256=task.get("structure_sha256", ""),
                   relaxation_status="converged" if audit.get("status") == "converged" else "failed",
                   relaxation_settings_json=json.dumps(audit.get("settings", settings or {}), sort_keys=True),
                   relaxation_audit_json=json.dumps(audit, sort_keys=True), error=pred.get("error"))
        rows.append(row)
        try:
            structure, energy, audit = validated_prediction(task, pred, settings)
        except Exception as exc:
            row["error"] = row["error"] or f"{type(exc).__name__}: {exc}"
            continue
        entry = PDEntry(structure.composition, energy * len(structure), name=task["id"])
        row["energy_total_eV"] = entry.energy
        entries[task["path"]] = entry
        grouped[task["chemsys"]].append((row, entry))
    for csys, group in grouped.items():
        refs = references.get(csys, [])
        unary = {next(iter(e.composition.elements)).symbol for e in refs if e.composition.is_element}
        if not set(csys.split("-")).issubset(unary):
            raise RuntimeError(f"Missing unary competing-phase references for {csys}")
        # Generated phases from narrower systems also compete in a superset's hull.
        elements = set(csys.split("-"))
        generated = [entry for entry in entries.values()
                     if {element.symbol for element in entry.composition.elements}.issubset(elements)]
        diagram = PhaseDiagram(refs + generated)
        for row, entry in group:
            ehull = float(diagram.get_e_above_hull(entry))
            formation = float(diagram.get_form_energy_per_atom(entry))
            if not all(math.isfinite(v) for v in (ehull, formation)):
                raise RuntimeError(f"Non-finite hull result for {row['file']}")
            row.update(energy_above_hull_eV=ehull, formation_energy_per_atom_eV=formation,
                       is_stable=int(ehull <= 1e-3), calculation_status="success",
                       hull_status="complete", error=None)
    return rows, entries


def persist_preparation(candidate_tasks, reference_tasks, predictions, relaxation_dir, settings):
    """Save only verified optimized geometries; retain every preparation failure."""
    from mattersim_relaxation import save_relaxed_structure

    relaxation_dir = Path(relaxation_dir).resolve()
    snapshot_path = relaxation_dir / "reference_entries.json"
    configuration = settings.as_dict()
    snapshot = {"schema_version": 1, "energy_model": ENERGY_MODEL,
                "relaxation_settings": configuration, "entries_by_chemsys": {},
                "references": [], "candidates": [],
                "systems": {csys: {"status": "calculation_failed", "error": "Hull preparation incomplete"}
                            for csys in sorted({task["chemsys"] for task in candidate_tasks})}}
    prepared, audits = [], []
    for role, tasks in (("candidate", candidate_tasks), ("reference", reference_tasks)):
        for task in tasks:
            pred = predictions.get(task["id"], {"id": task["id"], "error": "Missing prediction"})
            audit = pred.get("relaxation") or {"status": "failed", "converged": False,
                                                "settings": configuration}
            item = {**task, "reference_snapshot_path": str(snapshot_path)}
            source = str(Path(task["path"]).resolve()) if role == "candidate" else ""
            row = {"id": task["id"], "file": Path(source).name if source else "",
                   "role": role, "chemsys": task["chemsys"], "source_path": source,
                   "path": source, "relaxation_status": audit.get("status", "failed"),
                   "energy_model": ENERGY_MODEL, "energy_per_atom_eV": None,
                   "energy_total_eV": None, "error": pred.get("error"),
                   "relaxation_audit_json": json.dumps(audit, sort_keys=True),
                   "relaxation_settings_json": json.dumps(audit.get("settings", configuration), sort_keys=True)}
            try:
                structure, energy, audit = validated_prediction(task, pred, configuration)
                destination = (relaxation_dir / "candidates" / Path(source).name if role == "candidate"
                               else relaxation_dir / "references" / task["chemsys"]
                               / f"reference_{int(task['id'].rsplit(':', 1)[-1]):05d}.cif")
                save_relaxed_structure(structure, destination)
                record = {"id": task["id"], "chemsys": task["chemsys"], "path": str(destination),
                          "structure": structure.as_dict(), "structure_sha256": structure_file_sha256(destination),
                          "energy_total_eV": energy * len(structure), "energy_per_atom_eV": energy,
                          "relaxation": audit}
                if role == "candidate":
                    record.update(file=Path(source).name, source_path=source)
                validate_relaxed_record(record, settings)
                snapshot["candidates" if role == "candidate" else "references"].append(record)
                item.update(path=record["path"], source_path=source,
                            structure_sha256=record["structure_sha256"])
                row.update(path=record["path"], relaxation_status="converged",
                           energy_per_atom_eV=energy, energy_total_eV=record["energy_total_eV"], error=None)
            except Exception as exc:
                error = pred.get("error") or f"{type(exc).__name__}: {exc}"
                predictions[task["id"]] = {**pred, "energy_per_atom": None, "error": error}
                row["error"] = error
                if role == "reference":
                    snapshot["systems"][task["chemsys"]] = {"status": "calculation_failed", "error": error}
            audits.append(row)
            if role == "candidate":
                prepared.append(item)
    fields = ["id", "file", "role", "chemsys", "source_path", "path", "relaxation_status",
              "energy_model", "energy_per_atom_eV", "energy_total_eV", "error",
              "relaxation_audit_json", "relaxation_settings_json"]
    write_json(relaxation_dir / "audit.json", {"relaxation_settings": configuration, "records": audits})
    write_csv(relaxation_dir / "audit.csv", audits, fields)
    # Publish even a failed preparation so a previous successful cache is not reused.
    write_reference_snapshot(snapshot, snapshot_path)
    return prepared, snapshot, snapshot_path


def voltage_worker(tasks, context):
    # Voltage phase diagrams are CPU calculations. Each worker gets the FULL
    # competing set, including candidates assigned to the other workers.
    groups = {key: [PDEntry.from_dict(e) for e in values]
              for key, values in context["base_entries"].items()}
    element = Element("Li")
    voltages = generate_voltage_grid(0.0, 6.0, context["voltage_step"])
    results = []
    for task in tasks:
        row = task["row"]
        base = groups[row["chemsys"]]
        index = task["candidate_index"]
        candidate = Candidate(row=row, structure=None, entry=base[index])
        mu = context["mu_ref"][row["chemsys"]]
        result = evaluate_voltage_window(candidate, base, index, element, mu, voltages,
                                         context["threshold"], context["target_voltage"])
        record = dict(row)
        record.update(result.as_record())
        results.append({"id": task["id"], "record": record})
    return results


def run_parallel_screening(args, index_path, workdir):
    directory = args.ehull_out.parent / "parallel_screening"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A failed rerun must not leave an older final table looking current.
        for output in (args.ehull_out, args.filtered_out, args.voltage_out,
                       args.final_out, args.voltage_filter_audit):
            output.unlink(missing_ok=True)
        write_json(directory / "summary.json", {"status": "running"})
        try:
            _run_parallel_screening(args, index_path, workdir)
        except BaseException as exc:
            write_json(directory / "summary.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise


def _run_parallel_screening(args, index_path, workdir):
    from mp_api.client import MPRester
    from compute_ehull_chgnet import fetch_mp_competitor_structures
    from parallel_utils import gpu_devices, run_shards
    from run_top300_pipeline import filter_hull, filter_voltage
    from mattersim_relaxation import settings_from_args

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY is required for parallel screening")
    devices = gpu_devices(args.gpu_workers)
    settings = settings_from_args(args)
    preparation_context = {"relaxation_settings": settings.as_dict()}
    directory = args.ehull_out.parent / "parallel_screening"
    directory.mkdir(parents=True, exist_ok=True)
    # A fresh snapshot is made on every invocation; old cache results are never reused.
    candidates = load_candidate_structures(args.export_dir.resolve(), index_path)
    if len({p.name for p, _ in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate file identities")
    tasks = [{"id": f"candidate:{i}", "path": str(path), "chemsys": chem_system(s),
              "structure": s.as_dict()} for i, (path, s) in enumerate(candidates)]
    reference_tasks = []
    with MPRester(key, use_document_model=False) as mpr:
        for csys in sorted({task["chemsys"] for task in tasks}):
            structures = fetch_mp_competitor_structures(csys, mpr)
            if not structures:
                raise RuntimeError(f"No competing phases fetched for {csys}")
            reference_tasks.extend({"id": f"reference:{csys}:{i}", "chemsys": csys,
                                    "structure": s.as_dict()} for i, s in enumerate(structures))
    print(f"[INFO] Parallel screening: {len(tasks)} candidates, {len(reference_tasks)} references, GPUs {devices}", flush=True)
    write_json(directory / "input_snapshot.json", {"candidates": tasks, "references": reference_tasks,
                                                   "model": ENERGY_MODEL, "devices": devices,
                                                   "relaxation_settings": settings.as_dict()})
    predictions = run_shards(tasks + reference_tasks, args.gpu_workers, directory / "energies",
                              "predict", Path(__file__), workdir, devices=devices,
                              context=preparation_context)
    by_id = {item["id"]: item for item in predictions}
    tasks, snapshot, snapshot_path = persist_preparation(
        tasks, reference_tasks, by_id, args.ehull_out.parent / "relaxation", settings)
    try:
        references = reference_entries(reference_tasks, by_id, settings.as_dict())
        hull_rows, entries = calculate_hull(tasks, by_id, references, settings.as_dict())
        for csys, refs in references.items():
            elements = set(csys.split("-"))
            generated = [entry for entry in entries.values()
                         if {element.symbol for element in entry.composition.elements}.issubset(elements)]
            snapshot["entries_by_chemsys"][csys] = [entry.as_dict() for entry in refs + generated]
            snapshot["systems"][csys] = {
                "status": "complete", "error": None,
                "fetched_count": sum(task["chemsys"] == csys for task in reference_tasks),
                "prepared_count": len(refs),
            }
        write_reference_snapshot(snapshot, snapshot_path)
    except Exception as exc:
        # A common hull cannot be certified with any incomplete reference set.
        # Keep the optimized candidates and their preparation audit, but publish
        # no bulk or voltage success from this invocation.
        error = f"Complete phase diagram unavailable: {type(exc).__name__}: {exc}"
        failed = {identity: {**record, "error": record.get("error") or error}
                  for identity, record in by_id.items()}
        hull_rows, _ = calculate_hull(tasks, failed, {}, settings.as_dict())
        write_csv(args.ehull_out, hull_rows, HULL_FIELDS)
        filter_hull(args.ehull_out, args.filtered_out, args.ehull_threshold)
        snapshot["entries_by_chemsys"] = {}
        snapshot["systems"] = {csys: {"status": "calculation_failed", "error": error,
                                      "fetched_count": sum(task["chemsys"] == csys for task in reference_tasks),
                                      "prepared_count": sum(record["chemsys"] == csys for record in snapshot["references"])}
                               for csys in snapshot["systems"]}
        write_reference_snapshot(snapshot, snapshot_path)
        raise
    write_csv(args.ehull_out, hull_rows, HULL_FIELDS)
    kept = filter_hull(args.ehull_out, args.filtered_out, args.ehull_threshold)
    stable = [row for row in hull_rows if row["calculation_status"] == "success"
              and row["energy_above_hull_eV"] <= args.ehull_threshold]
    base_entries, mu_ref, voltage_tasks = {}, {}, []
    for csys in sorted({row["chemsys"] for row in stable}):
        group = [row for row in stable if row["chemsys"] == csys]
        complete_pool = [PDEntry.from_dict(entry) for entry in snapshot["entries_by_chemsys"][csys]]
        mu = locate_reference_mu(complete_pool, Element("Li"))
        if mu is None or not math.isfinite(mu):
            raise RuntimeError(f"No finite Li reference for {csys}")
        mu_ref[csys] = mu
        # Reuse the exact complete pool published by the common bulk hull, also
        # including generated phases outside the bulk gate and from other shards.
        base_entries[csys] = snapshot["entries_by_chemsys"][csys]
        entry_indexes = {entry["name"]: index for index, entry in enumerate(base_entries[csys])}
        path_to_identity = {task["path"]: task["id"] for task in tasks}
        for row in group:
            voltage_tasks.append({"id": row["path"], "row": row,
                                  "candidate_index": entry_indexes[path_to_identity[row["path"]]]})
    context = {"base_entries": base_entries, "mu_ref": mu_ref,
               "voltage_step": args.voltage_step or 0.05,
               "threshold": args.voltage_threshold, "target_voltage": args.target_voltage}
    write_json(directory / "voltage_snapshot.json", context)
    results = run_shards(voltage_tasks, args.gpu_workers, directory / "voltage",
                         "voltage", Path(__file__), workdir, context=context)
    by_path = {result["id"]: result["record"] for result in results}
    voltage_rows = [by_path[row["path"]] for row in stable]
    voltage_fields = list(dict.fromkeys(HULL_FIELDS + list(VoltageWindowResult().as_record())))
    write_csv(args.voltage_out, voltage_rows, voltage_fields)
    filter_voltage(args.filtered_out, args.voltage_out, args.final_out, args.voltage_filter_audit,
                   target_voltage=args.target_voltage, min_window=args.min_voltage_window)
    write_json(directory / "summary.json", {"status": "completed", "devices": devices, "selected": len(tasks),
               "energy_failures": sum(r["calculation_status"] != "success" for r in hull_rows),
               "hull_passed": kept, "voltage_evaluated": len(voltage_rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("predict", "voltage"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    results = (predict_worker(payload["tasks"], payload["context"]) if args.worker == "predict"
               else voltage_worker(payload["tasks"], payload["context"]))
    write_json(args.output, results)


if __name__ == "__main__":
    main()
