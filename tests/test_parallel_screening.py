"""Regression tests for shared hulls across parallel screening workers."""

import csv
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Lattice, Structure


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "pipeline"
sys.path.insert(0, str(SCRIPTS))

import compute_ehull_chgnet as hull_helpers
import compute_voltage_window as voltage_helpers
import parallel_screening as screening
import parallel_utils
import run_top300_pipeline as pipeline
import mattersim_relaxation as relaxation_helpers


def structure(species):
    count = len(species)
    return Structure(
        Lattice.cubic(6), species,
        [[i / count, i / count, i / count] for i in range(count)],
    )


def candidate_task(index, species, directory=Path("/selected")):
    return {
        "id": f"candidate:{index}", "path": str(directory / f"cand_{index}.cif"),
        "chemsys": "Cl-Li", "structure": structure(species).as_dict(),
    }


def prediction(task, energy_per_atom, error=None):
    optimized = Structure.from_dict(task["structure"])
    optimized.scale_lattice(optimized.volume * 1.331)
    return {"id": task["id"], "energy_per_atom": energy_per_atom,
            "structure": optimized.as_dict(),
            "relaxation": {"status": "converged", "converged": True,
                           "settings": relaxation_helpers.RelaxationSettings().as_dict(),
                           "optimizer": "FIRE", "cell_filter": "ExpCellFilter", "relax_cell": True,
                           "scalar_pressure_eV_A3": 0.0, "constrain_symmetry": False,
                           "steps": 1, "fmax_final": .01}, "error": error}


def unary_references():
    return {"Cl-Li": [PDEntry("Li", -1), PDEntry("Cl", 0)]}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_hull_preserves_competing_candidate_from_another_gpu_shard():
    # LiCl=-2 total is .25 eV/atom above the common hull with Li2Cl=-5.
    # An independent LiCl-only shard would incorrectly report zero.
    tasks = [candidate_task(0, ["Li", "Cl"]), candidate_task(1, ["Li", "Li", "Cl"])]
    predictions = {
        tasks[0]["id"]: prediction(tasks[0], -1),
        tasks[1]["id"]: prediction(tasks[1], -5 / 3),
    }
    rows, entries = screening.calculate_hull(tasks, predictions, unary_references())
    assert [row["path"] for row in rows] == [task["path"] for task in tasks]
    assert rows[0]["energy_total_eV"] == pytest.approx(-2)
    assert rows[1]["energy_total_eV"] == pytest.approx(-5)
    assert rows[0]["energy_above_hull_eV"] == pytest.approx(0.25)
    assert rows[1]["energy_above_hull_eV"] == pytest.approx(0)
    assert all(row["calculation_status"] == "success" for row in rows)
    assert set(entries) == {task["path"] for task in tasks}
    isolated, _ = screening.calculate_hull(tasks[:1], predictions, unary_references())
    assert isolated[0]["energy_above_hull_eV"] == pytest.approx(0)


@pytest.mark.parametrize("bad_energy,error", [
    (None, "model failed"), (float("nan"), None),
    (float("inf"), None), (-float("inf"), None), (None, None),
    (-3.0, "failed despite partial prediction"),
])
def test_failed_candidate_is_audited_and_never_added_to_the_hull(bad_energy, error):
    tasks = [candidate_task(0, ["Li", "Cl"]), candidate_task(1, ["Li", "Li", "Cl"])]
    predictions = {
        tasks[0]["id"]: prediction(tasks[0], -1),
        tasks[1]["id"]: prediction(tasks[1], bad_energy, error),
    }
    rows, entries = screening.calculate_hull(tasks, predictions, unary_references())
    assert len(rows) == len(tasks)
    assert rows[0]["calculation_status"] == "success"
    failed = rows[1]
    assert failed["path"] == tasks[1]["path"]
    assert failed["calculation_status"] == "calculation_failed"
    assert failed["error"]
    assert failed["energy_above_hull_eV"] is None
    assert failed["energy_total_eV"] is None
    assert failed["is_stable"] == 0
    assert set(entries) == {tasks[0]["path"]}


@pytest.mark.parametrize("failure", ["missing", "none", "nan", "inf", "error"])
def test_any_missing_or_failed_reference_rejects_incomplete_reference_set(failure):
    tasks = [
        {"id": f"reference:Cl-Li:{i}", "chemsys": "Cl-Li", "structure": structure([element]).as_dict()}
        for i, element in enumerate(("Li", "Cl"))
    ]
    predictions = {task["id"]: prediction(task, -1 if i == 0 else 0)
                   for i, task in enumerate(tasks)}
    if failure == "missing":
        del predictions[tasks[1]["id"]]
    else:
        predictions[tasks[1]["id"]] = prediction(
            tasks[1], {"none": None, "nan": float("nan"), "inf": float("inf"), "error": 0}[failure],
            "model failure" if failure == "error" else None,
        )
    with pytest.raises((RuntimeError, KeyError)):
        screening.reference_entries(tasks, predictions)


def test_reference_energy_per_atom_is_converted_to_original_total_energy():
    task = {"id": "reference:Cl-Li:0", "chemsys": "Cl-Li",
            "structure": structure(["Li", "Cl"]).as_dict()}
    references = screening.reference_entries([task], {task["id"]: prediction(task, -2.5)})
    assert references["Cl-Li"][0].energy == pytest.approx(-5)
    assert references["Cl-Li"][0].composition.num_atoms == 2


def test_missing_unary_reference_rejects_common_hull():
    task = candidate_task(0, ["Li", "Cl"])
    with pytest.raises(RuntimeError, match="Missing unary"):
        screening.calculate_hull([task], {task["id"]: prediction(task, -1)},
                                {"Cl-Li": [PDEntry("LiCl", -2)]})


def test_generated_phase_from_a_subset_system_competes_in_superset_hull():
    binary = candidate_task(0, ["Li", "Li", "Cl"])
    ternary = {**candidate_task(1, ["Li", "Cl", "O"]), "chemsys": "Cl-Li-O"}
    tasks = [binary, ternary]
    values = {binary["id"]: prediction(binary, -5 / 3), ternary["id"]: prediction(ternary, -2 / 3)}
    references = {**unary_references(), "Cl-Li-O": unary_references()["Cl-Li"] + [PDEntry("O", 0)]}
    rows, _ = screening.calculate_hull(tasks, values, references)
    assert rows[1]["energy_above_hull_eV"] == pytest.approx(1 / 6)
    isolated, _ = screening.calculate_hull([ternary], values, references)
    assert isolated[0]["energy_above_hull_eV"] == 0


@pytest.mark.parametrize("invalid", ["legacy_energy_only", "wrong_settings", "changed_composition", "wrong_id"])
def test_untrusted_worker_preparation_cannot_enter_common_hull(invalid):
    task = candidate_task(0, ["Li", "Cl"])
    value = prediction(task, -1)
    if invalid == "legacy_energy_only":
        value = {"id": task["id"], "energy_per_atom": -1, "error": None}
    elif invalid == "wrong_settings":
        value["relaxation"]["settings"]["fmax"] = .2
    elif invalid == "changed_composition":
        value["structure"] = structure(["Li", "Li"]).as_dict()
    else:
        value["id"] = "another-candidate"
    rows, entries = screening.calculate_hull([task], {task["id"]: value}, unary_references(),
                                             relaxation_helpers.RelaxationSettings().as_dict())
    assert not entries
    assert rows[0]["calculation_status"] == "calculation_failed"
    assert rows[0]["error"]


def voltage_context(entries):
    return {
        "base_entries": {"Cl-Li": [entry.as_dict() for entry in entries]},
        "mu_ref": {"Cl-Li": -1}, "voltage_step": 0.25,
        "threshold": 1e-3, "target_voltage": 0,
    }


def voltage_task(index, entry):
    path = f"/selected/candidate_{index}.cif"
    return {"id": path, "candidate_index": index, "row": {
        "file": Path(path).name, "path": path,
        "formula": entry.composition.reduced_formula, "chemsys": "Cl-Li",
    }}


def test_voltage_worker_keeps_candidates_assigned_to_other_shards_in_base_hull():
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("Li2Cl", -3.5), PDEntry("LiCl", -2)]
    task = voltage_task(3, entries[3])
    result = screening.voltage_worker([task], voltage_context(entries))[0]
    assert result["id"] == task["id"]
    record = result["record"]
    assert (record["V_red"], record["V_ox"], record["window"]) == (0.5, 1, 0.5)
    assert record["stable_at_target"] is False
    assert record["e_above_hull_unit"] == "eV/non-Li atom"
    split_entries = entries[:2] + entries[3:]
    isolated_task = {**task, "candidate_index": 2}
    isolated = screening.voltage_worker([isolated_task], voltage_context(split_entries))[0]["record"]
    assert (isolated["V_red"], isolated["V_ox"], isolated["window"]) == (0, 1, 1)
    assert isolated["stable_at_target"] is True


def test_one_and_four_cpu_shards_have_identical_results_and_input_order(tmp_path):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0)] + [
        PDEntry("LiCl", -2), PDEntry("Li2Cl", -3.5), PDEntry("Li3Cl", -4),
        PDEntry("Li4Cl", -4.5), PDEntry("Li5Cl", -5),
    ]
    tasks = [voltage_task(i, entry) for i, entry in enumerate(entries) if i >= 2]
    context = voltage_context(entries)
    serial = parallel_utils.run_shards(tasks, 1, tmp_path / "one", "voltage",
                                      SCRIPTS / "parallel_screening.py", tmp_path, context=context)
    parallel = parallel_utils.run_shards(tasks, 4, tmp_path / "four", "voltage",
                                        SCRIPTS / "parallel_screening.py", tmp_path, context=context)
    assert serial == parallel
    assert [result["id"] for result in parallel] == [task["id"] for task in tasks]
    assert [result["record"]["path"] for result in parallel] == [task["id"] for task in tasks]
    assert len(list((tmp_path / "four").glob("*.output.json"))) == 4
    for shard_input in (tmp_path / "four").glob("*.input.json"):
        payload = json.loads(shard_input.read_text())
        assert len(payload["context"]["base_entries"]["Cl-Li"]) == len(entries)


def test_integrated_four_gpu_screening_preserves_global_hull_and_failure_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "offline-key")
    export = tmp_path / "export"
    export.mkdir()
    candidate_structures = [structure(["Li", "Cl"]), structure(["Li", "Li", "Cl"]),
                            structure(["Li", "Li", "Li", "Cl"])]
    candidates = [(export / f"cand_{i}.cif", value) for i, value in enumerate(candidate_structures)]
    monkeypatch.setattr(screening, "load_candidate_structures", lambda *args: candidates)

    client = types.ModuleType("mp_api.client")
    class FakeMPRester:
        def __init__(self, key, use_document_model):
            assert key == "offline-key"
            assert use_document_model is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
    client.MPRester = FakeMPRester
    monkeypatch.setitem(sys.modules, "mp_api", types.ModuleType("mp_api"))
    monkeypatch.setitem(sys.modules, "mp_api.client", client)
    fetch_calls = []
    def fetch(chemsys, mpr):
        fetch_calls.append(chemsys)
        return [structure(["Li"]), structure(["Cl"])]
    monkeypatch.setattr(hull_helpers, "fetch_mp_competitor_structures", fetch)
    monkeypatch.setattr(parallel_utils, "gpu_devices", lambda workers: ["3", "2", "1", "0"])
    shard_calls = []
    def run_shards(tasks, workers, directory, mode, script, workdir, devices=None, context=None):
        shard_calls.append((workers, mode, devices))
        assert workers == 4
        if mode == "predict":
            assert context == {"relaxation_settings": relaxation_helpers.RelaxationSettings().as_dict()}
            energies = {"candidate:0": -1, "candidate:1": -5 / 3,
                        "candidate:2": None, "reference:Cl-Li:0": -1, "reference:Cl-Li:1": 0}
            # Completion order differs from manifest order, as it does on real GPUs.
            return [prediction(task, energies[task["id"]],
                               "synthetic candidate failure" if task["id"] == "candidate:2" else None)
                    for task in reversed(tasks)]
        return screening.voltage_worker(tasks, context)
    monkeypatch.setattr(parallel_utils, "run_shards", run_shards)
    args = SimpleNamespace(
        gpu_workers=4, export_dir=export, ehull_out=tmp_path / "hull.csv",
        filtered_out=tmp_path / "filtered.csv", voltage_out=tmp_path / "voltage.csv",
        final_out=tmp_path / "final.csv", voltage_filter_audit=tmp_path / "voltage_audit.csv",
        ehull_threshold=0.05, voltage_step=0.5, voltage_threshold=1e-3,
        target_voltage=None, min_voltage_window=0,
    )
    screening.run_parallel_screening(args, export / "index.csv", tmp_path)
    hull = read_rows(args.ehull_out)
    assert [row["source_path"] for row in hull] == [str(path) for path, _ in candidates]
    assert [row["path"] for row in hull[:2]] == [
        str(tmp_path / "relaxation/candidates" / path.name) for path, _ in candidates[:2]]
    assert Structure.from_file(hull[0]["path"]).lattice.a == pytest.approx(6.6)
    assert float(hull[0]["energy_above_hull_eV"]) == pytest.approx(0.25)
    assert hull[2]["calculation_status"] == "calculation_failed"
    assert hull[2]["error"] == "synthetic candidate failure"
    assert hull[2]["energy_above_hull_eV"] == ""
    assert [row["file"] for row in read_rows(args.filtered_out)] == ["cand_1.cif"]
    assert [row["file"] for row in read_rows(args.voltage_out)] == ["cand_1.cif"]
    assert [row["file"] for row in read_rows(args.final_out)] == ["cand_1.cif"]
    assert read_rows(args.voltage_filter_audit)[0]["passes_voltage_filter"] == "True"
    assert fetch_calls == ["Cl-Li"]
    assert shard_calls == [(4, "predict", ["3", "2", "1", "0"]), (4, "voltage", None)]
    summary = json.loads((tmp_path / "parallel_screening" / "summary.json").read_text())
    assert summary["selected"] == 3
    assert summary["energy_failures"] == 1
    assert summary["hull_passed"] == summary["voltage_evaluated"] == 1
    snapshot = json.loads((tmp_path / "relaxation/reference_entries.json").read_text())
    assert snapshot["energy_model"] == "CHGNet-0.3.0"
    assert len(snapshot["candidates"]) == len(snapshot["references"]) == 2
    assert len(snapshot["entries_by_chemsys"]["Cl-Li"]) == 4
    assert {entry["name"] for entry in snapshot["entries_by_chemsys"]["Cl-Li"]} == {
        "candidate:0", "candidate:1", "reference:Cl-Li:0", "reference:Cl-Li:1"}
    assert len(read_rows(tmp_path / "relaxation/audit.csv")) == 5
    for value in snapshot["candidates"] + snapshot["references"]:
        assert value["structure_sha256"] == hull_helpers.structure_file_sha256(value["path"])
    # The serial voltage entry point accepts a parallel preparation without
    # refetching MP phases, re-relaxing structures, or loading an energy model.
    settings = relaxation_helpers.RelaxationSettings()
    snapshot_path = tmp_path / "relaxation/reference_entries.json"
    cached = voltage_helpers.load_reference_snapshot(snapshot_path, settings)
    cached_entries, cached_candidates = voltage_helpers.snapshot_phase_entries(cached, "Cl-Li", settings)
    candidate, index = voltage_helpers.validated_snapshot_candidate(
        read_rows(args.filtered_out)[0], cached_candidates, cached_entries, settings, snapshot_path)
    assert candidate.structure.lattice.a == pytest.approx(6.6)
    assert cached_entries[index].name == "candidate:1"
    assert len(cached_entries) == 4
    audit = json.loads((tmp_path / "relaxation/audit.json").read_text())
    assert audit["relaxation_settings"] == settings.as_dict()
    assert len(audit["records"]) == 5


@pytest.fixture
def integrated_screening(tmp_path, monkeypatch):
    """An offline MP snapshot and current candidate manifest for failure cases."""
    monkeypatch.setenv("MP_API_KEY", "offline-key")
    export = tmp_path / "export"
    export.mkdir()
    candidates = [
        (export / f"cand_{i}.cif", structure(["Li"] * (i + 1) + ["Cl"]))
        for i in range(3)
    ]
    monkeypatch.setattr(screening, "load_candidate_structures", lambda *args: candidates)
    client = types.ModuleType("mp_api.client")
    class FakeMPRester:
        def __init__(self, key, use_document_model):
            assert key == "offline-key"
            assert use_document_model is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
    client.MPRester = FakeMPRester
    monkeypatch.setitem(sys.modules, "mp_api", types.ModuleType("mp_api"))
    monkeypatch.setitem(sys.modules, "mp_api.client", client)
    monkeypatch.setattr(hull_helpers, "fetch_mp_competitor_structures",
                        lambda *args: [structure(["Li"]), structure(["Cl"])])
    monkeypatch.setattr(parallel_utils, "gpu_devices", lambda workers: ["0", "1", "2", "3"])
    args = SimpleNamespace(
        gpu_workers=4, export_dir=export, ehull_out=tmp_path / "hull.csv",
        filtered_out=tmp_path / "filtered.csv", voltage_out=tmp_path / "voltage.csv",
        final_out=tmp_path / "final.csv", voltage_filter_audit=tmp_path / "voltage_audit.csv",
        ehull_threshold=0.05, voltage_step=0.5, voltage_threshold=1e-3,
        target_voltage=None, min_voltage_window=0,
    )
    return args


@pytest.mark.parametrize("crash_stage", ["predict", "voltage"])
def test_worker_crash_removes_stale_final_outputs_and_records_failed_run(
    tmp_path, monkeypatch, integrated_screening, crash_stage,
):
    args = integrated_screening
    args.final_out.write_text("file\nstale_final.cif\n")
    args.voltage_filter_audit.write_text("file\nstale_audit.cif\n")
    calls = []
    def run_shards(tasks, workers, directory, mode, script, workdir, devices=None, context=None):
        calls.append(mode)
        assert workers == 4
        if mode == crash_stage:
            raise subprocess.CalledProcessError(1, ["synthetic_worker", mode])
        assert mode == "predict"
        energies = {"candidate:0": -1, "candidate:1": -5 / 3, "candidate:2": 1,
                    "reference:Cl-Li:0": -1, "reference:Cl-Li:1": 0}
        return [prediction(task, energies[task["id"]]) for task in tasks]
    monkeypatch.setattr(parallel_utils, "run_shards", run_shards)
    monkeypatch.setattr(pipeline, "filter_voltage",
                        lambda *args, **kwargs: pytest.fail("A failed worker must block final merge"))
    with pytest.raises(subprocess.CalledProcessError):
        screening.run_parallel_screening(args, args.export_dir / "index.csv", tmp_path)
    assert not args.final_out.exists()
    assert not args.voltage_filter_audit.exists()
    summary = json.loads((tmp_path / "parallel_screening" / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert "CalledProcessError" in summary["error"]
    assert "synthetic_worker" in summary["error"]
    assert calls == (["predict"] if crash_stage == "predict" else ["predict", "voltage"])


def test_zero_hull_pass_writes_empty_tables_without_starting_voltage_workers(
    tmp_path, monkeypatch, integrated_screening,
):
    args = integrated_screening
    for path in (args.filtered_out, args.voltage_out, args.final_out, args.voltage_filter_audit):
        path.write_text("file\nstale.cif\n")
    started_stages = []
    def run_shards(tasks, workers, directory, mode, script, workdir, devices=None, context=None):
        assert workers == 4
        if not tasks:
            assert mode == "voltage"
            return []  # Real run_shards starts no subprocess for an empty task set.
        started_stages.append(mode)
        assert mode == "predict", "No voltage worker may run when the hull gate is empty"
        return [prediction(task, -1 if task["id"] == "reference:Cl-Li:0" else (
            0 if task["id"] == "reference:Cl-Li:1" else 1
        )) for task in tasks]
    monkeypatch.setattr(parallel_utils, "run_shards", run_shards)
    screening.run_parallel_screening(args, args.export_dir / "index.csv", tmp_path)
    assert started_stages == ["predict"]
    assert len(read_rows(args.ehull_out)) == 3
    assert all(float(row["energy_above_hull_eV"]) > args.ehull_threshold for row in read_rows(args.ehull_out))
    for path in (args.filtered_out, args.voltage_out, args.final_out, args.voltage_filter_audit):
        assert path.exists()
        assert read_rows(path) == []
        assert "file" in path.read_text().splitlines()[0].split(",")
        assert len(path.read_text().splitlines()) == 1
    summary = json.loads((tmp_path / "parallel_screening" / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["selected"] == 3
    assert summary["energy_failures"] == summary["hull_passed"] == summary["voltage_evaluated"] == 0


def test_failed_reference_relaxation_is_audited_and_aborts_all_common_hulls(
    tmp_path, monkeypatch, integrated_screening,
):
    args = integrated_screening
    stages = []
    def run_shards(tasks, workers, directory, mode, script, workdir, devices=None, context=None):
        stages.append(mode)
        assert mode == "predict", "Reference failures must prevent voltage work"
        records = [prediction(task, -1 if task["id"] == "reference:Cl-Li:0" else 0) for task in tasks]
        bad = next(record for record in records if record["id"] == "reference:Cl-Li:1")
        bad.update(energy_per_atom=None, structure=None, error="MatterSim reference did not converge")
        bad["relaxation"].update(status="failed", converged=False)
        return records
    monkeypatch.setattr(parallel_utils, "run_shards", run_shards)
    args.final_out.write_text("file\nprevious.cif\n")
    with pytest.raises(RuntimeError, match="Incomplete competing-phase"):
        screening.run_parallel_screening(args, args.export_dir / "index.csv", tmp_path)
    assert stages == ["predict"]
    assert not args.final_out.exists()
    assert not args.voltage_out.exists()
    assert all(row["calculation_status"] == "calculation_failed" for row in read_rows(args.ehull_out))
    assert read_rows(args.filtered_out) == []
    snapshot = json.loads((tmp_path / "relaxation/reference_entries.json").read_text())
    assert snapshot["entries_by_chemsys"] == {}
    assert snapshot["systems"]["Cl-Li"]["status"] == "calculation_failed"
    audit = read_rows(tmp_path / "relaxation/audit.csv")
    failed = next(row for row in audit if row["id"] == "reference:Cl-Li:1")
    assert failed["relaxation_status"] == "failed"
    assert failed["energy_per_atom_eV"] == ""
    assert "did not converge" in failed["error"]
    assert len(audit) == 5


def test_gpu_worker_cli_defaults_to_serial_and_accepts_four_workers():
    assert pipeline.parse_args([]).gpu_workers == 1
    assert pipeline.parse_args(["--gpu-workers", "4"]).gpu_workers == 4


@pytest.mark.parametrize("workers", ["0", "-1", "bad", "1.5"])
def test_gpu_worker_cli_rejects_nonpositive_or_noninteger_counts(workers):
    with pytest.raises(SystemExit) as exc:
        pipeline.parse_args(["--gpu-workers", workers])
    assert exc.value.code == 2


@pytest.mark.parametrize("calculator", ["--ehull-script", "--voltage-script"])
def test_parallel_cli_rejects_custom_calculators_but_serial_keeps_them(calculator):
    with pytest.raises(SystemExit) as exc:
        pipeline.parse_args(["--gpu-workers", "4", calculator, "/custom/calculator.py"])
    assert exc.value.code == 2
    args = pipeline.parse_args([calculator, "/custom/calculator.py"])
    assert args.gpu_workers == 1
    assert getattr(args, calculator[2:].replace("-", "_")) == Path("/custom/calculator.py")


def test_prediction_worker_uses_isolated_logical_gpu_and_records_each_task_failure(monkeypatch, capsys):
    # Physical GPU 2 becomes logical cuda:0 inside its isolated subprocess.
    # CHGNet's memory selector must never reinterpret that assignment.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    outcomes = [-1, float("nan"), RuntimeError("synthetic prediction failure"), float("inf"), -2]
    loaded = []
    predicted_sizes = []
    relaxed_sizes, relaxer_initializations = [], []
    class FakeRelaxer:
        def __init__(self, settings, device):
            self.settings = settings
            relaxer_initializations.append((settings.as_dict(), device))
        def relax(self, value):
            relaxed_sizes.append(len(value))
            optimized = value.copy()
            optimized.scale_lattice(value.volume * 1.331)
            return optimized, {"status": "converged", "converged": True,
                               "settings": self.settings.as_dict(),
                               "optimizer": "FIRE", "cell_filter": "ExpCellFilter", "relax_cell": True,
                               "scalar_pressure_eV_A3": 0.0, "constrain_symmetry": False,
                               "steps": 1, "fmax_final": .01}
    monkeypatch.setattr(relaxation_helpers, "MatterSimRelaxer", FakeRelaxer)
    class FakeModel:
        def parameters(self):
            return iter([SimpleNamespace(device="cuda:0")])
        def predict_structure(self, value):
            assert value.lattice.a == pytest.approx(6.6)
            predicted_sizes.append(len(value))
            outcome = outcomes[len(predicted_sizes) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return {"e": outcome}
    def load(**kwargs):
        loaded.append(kwargs)
        return FakeModel()
    model_module = types.ModuleType("chgnet.model")
    model_module.CHGNet = type("FakeCHGNet", (), {"load": staticmethod(load)})
    monkeypatch.setitem(sys.modules, "chgnet", types.ModuleType("chgnet"))
    monkeypatch.setitem(sys.modules, "chgnet.model", model_module)
    tasks = [candidate_task(i, ["Li"] * (i + 1) + ["Cl"]) for i in range(len(outcomes))]
    records = screening.predict_worker(tasks)
    assert loaded == [{"model_name": "0.3.0", "use_device": "cuda:0", "check_cuda_mem": False}]
    assert predicted_sizes == [2, 3, 4, 5, 6]
    assert relaxed_sizes == predicted_sizes
    assert relaxer_initializations == [(relaxation_helpers.RelaxationSettings().as_dict(), "cuda:0")]
    assert [record["id"] for record in records] == [task["id"] for task in tasks]
    assert records[0]["energy_per_atom"] == -1
    assert records[-1]["energy_per_atom"] == -2
    assert records[0]["error"] is records[-1]["error"] is None
    for index in (1, 2, 3):
        assert records[index]["energy_per_atom"] is None
        assert records[index]["error"]
    assert "Non-finite model energy" in records[1]["error"]
    assert "synthetic prediction failure" in records[2]["error"]
    assert "Non-finite model energy" in records[3]["error"]
    assert "CUDA_VISIBLE_DEVICES=2 model_device=cuda:0" in capsys.readouterr().out
