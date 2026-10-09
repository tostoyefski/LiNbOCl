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

from compute_ehull_chgnet import chem_system, get_energy_per_atom, load_candidate_structures
from compute_voltage_window import (
    Candidate, VoltageWindowResult, evaluate_voltage_window,
    generate_voltage_grid, locate_reference_mu,
)

HULL_FIELDS = ["file", "path", "formula", "chemsys", "natoms_cell",
               "energy_per_atom_eV", "energy_total_eV",
               "formation_energy_per_atom_eV", "energy_above_hull_eV",
               "is_stable", "calculation_status", "error"]


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


def predict_worker(tasks):
    # CUDA visibility is set by the parent before importing the model.
    from chgnet.model import CHGNet
    # Within this child's one-device CUDA visibility, the assigned GPU is
    # always logical cuda:0. CHGNet's NVML free-memory selector uses physical
    # indices and must not override the scheduler's device isolation.
    model = CHGNet.load(model_name="0.3.0", use_device="cuda:0", check_cuda_mem=False)
    print(f"[INFO] Energy worker CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} model_device={next(model.parameters()).device}", flush=True)
    results = []
    for task in tasks:
        try:
            structure = Structure.from_dict(task["structure"])
            energy = get_energy_per_atom(model.predict_structure(structure))
            if not math.isfinite(energy):
                raise ValueError("Non-finite model energy")
            results.append({"id": task["id"], "energy_per_atom": energy, "error": None})
        except Exception as exc:
            results.append({"id": task["id"], "energy_per_atom": None,
                            "error": f"{type(exc).__name__}: {exc}"})
    return results


def reference_entries(reference_tasks, predictions):
    grouped = defaultdict(list)
    for task in reference_tasks:
        pred = predictions[task["id"]]
        energy = pred.get("energy_per_atom")
        if pred.get("error") or energy is None or not math.isfinite(energy):
            raise RuntimeError(f"Incomplete competing-phase energies for {task['chemsys']}: {task['id']}")
        structure = Structure.from_dict(task["structure"])
        grouped[task["chemsys"]].append(PDEntry(structure.composition, energy * len(structure)))
    return dict(grouped)


def calculate_hull(candidate_tasks, predictions, references):
    """Use every successful candidate of a system in its common phase diagram."""
    rows, entries, grouped = [], {}, defaultdict(list)
    for task in candidate_tasks:
        structure = Structure.from_dict(task["structure"])
        pred = predictions[task["id"]]
        energy = pred.get("energy_per_atom")
        row = dict(file=Path(task["path"]).name, path=task["path"],
                   formula=structure.composition.reduced_formula, chemsys=task["chemsys"],
                   natoms_cell=len(structure), energy_per_atom_eV=energy,
                   energy_total_eV=None, formation_energy_per_atom_eV=None,
                   energy_above_hull_eV=None, is_stable=0,
                   calculation_status="calculation_failed", error=pred.get("error"))
        rows.append(row)
        if pred.get("error") or energy is None or not math.isfinite(energy):
            row["error"] = row["error"] or "Non-finite or missing model energy"
            continue
        entry = PDEntry(structure.composition, energy * len(structure))
        row["energy_total_eV"] = entry.energy
        entries[task["path"]] = entry
        grouped[task["chemsys"]].append((row, entry))
    for csys, group in grouped.items():
        refs = references.get(csys, [])
        unary = {next(iter(e.composition.elements)).symbol for e in refs if e.composition.is_element}
        if not set(csys.split("-")).issubset(unary):
            raise RuntimeError(f"Missing unary competing-phase references for {csys}")
        diagram = PhaseDiagram(refs + [entry for _, entry in group])
        for row, entry in group:
            ehull = float(diagram.get_e_above_hull(entry))
            formation = float(diagram.get_form_energy_per_atom(entry))
            if not all(math.isfinite(v) for v in (ehull, formation)):
                raise RuntimeError(f"Non-finite hull result for {row['file']}")
            row.update(energy_above_hull_eV=ehull, formation_energy_per_atom_eV=formation,
                       is_stable=int(ehull <= 1e-3), calculation_status="success", error=None)
    return rows, entries


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
        record = {key: row[key] for key in ("file", "path", "formula", "chemsys")}
        record.update(result.as_record())
        results.append({"id": task["id"], "record": record})
    return results


def run_parallel_screening(args, index_path, workdir):
    directory = args.ehull_out.parent / "parallel_screening"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A failed rerun must not leave an older final table looking current.
        for output in (args.final_out, args.voltage_filter_audit):
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

    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY is required for parallel screening")
    devices = gpu_devices(args.gpu_workers)
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
                                                   "model": "CHGNet-0.3.0", "devices": devices})
    predictions = run_shards(tasks + reference_tasks, args.gpu_workers, directory / "energies",
                              "predict", Path(__file__), workdir, devices=devices)
    by_id = {item["id"]: item for item in predictions}
    references = reference_entries(reference_tasks, by_id)
    hull_rows, entries = calculate_hull(tasks, by_id, references)
    write_csv(args.ehull_out, hull_rows, HULL_FIELDS)
    kept = filter_hull(args.ehull_out, args.filtered_out, args.ehull_threshold)
    stable = [row for row in hull_rows if row["calculation_status"] == "success"
              and row["energy_above_hull_eV"] <= args.ehull_threshold]
    base_entries, mu_ref, voltage_tasks = {}, {}, []
    for csys in sorted({row["chemsys"] for row in stable}):
        group = [row for row in stable if row["chemsys"] == csys]
        refs = references[csys]
        mu = locate_reference_mu(refs, Element("Li"))
        if mu is None or not math.isfinite(mu):
            raise RuntimeError(f"No finite Li reference for {csys}")
        mu_ref[csys] = mu
        base_entries[csys] = [e.as_dict() for e in refs + [entries[row["path"]] for row in group]]
        for i, row in enumerate(group):
            voltage_tasks.append({"id": row["path"], "row": row, "candidate_index": len(refs) + i})
    context = {"base_entries": base_entries, "mu_ref": mu_ref,
               "voltage_step": args.voltage_step or 0.05,
               "threshold": args.voltage_threshold, "target_voltage": args.target_voltage}
    write_json(directory / "voltage_snapshot.json", context)
    results = run_shards(voltage_tasks, args.gpu_workers, directory / "voltage",
                         "voltage", Path(__file__), workdir, context=context)
    by_path = {result["id"]: result["record"] for result in results}
    voltage_rows = [by_path[row["path"]] for row in stable]
    voltage_fields = ["file", "path", "formula", "chemsys"] + list(VoltageWindowResult().as_record())
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
    results = (predict_worker(payload["tasks"]) if args.worker == "predict"
               else voltage_worker(payload["tasks"], payload["context"]))
    write_json(args.output, results)


if __name__ == "__main__":
    main()
