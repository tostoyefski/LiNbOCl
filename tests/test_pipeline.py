import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "workflow" / "pipeline"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("top_pipeline", SCRIPTS / "run_top300_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def write_csv(path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pipeline_defaults_are_diverse_and_strict(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pipeline"])
    args = pipeline.parse_args()
    assert args.voltage_threshold == 1e-3
    assert args.selection_mode == "diverse"
    assert args.target_voltage is None
    results = Path(__file__).resolve().parents[1] / "results"
    assert args.output_dir == results / "top300_run"
    assert args.stage2_csv == results / "stage2_candidates.csv"
    assert args.export_script == SCRIPTS / "export_refs_to_structs.py"
    assert args.ehull_script == SCRIPTS / "compute_ehull_chgnet.py"
    assert args.voltage_script == SCRIPTS / "compute_voltage_window.py"
    assert not args.skip_novelty
    assert args.novelty_training_splits == ("train",)


def test_old_score_csv_requires_rescreening(tmp_path):
    csv_path = tmp_path / "old.csv"
    write_csv(csv_path, [{"quick_score": 1, "path": "/old.extxyz", "frame": 0}])
    with pytest.raises(RuntimeError, match="Re-run"):
        pipeline.read_stage2_rows(csv_path)


def test_empty_current_screen_finishes_without_reusing_old_outputs(tmp_path, monkeypatch):
    stage2 = tmp_path / "screen.csv"
    stage2.write_text("quick_score,path,frame,score_kind\n")
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "final_candidates.csv").write_text("file\nstale.cif\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "run_command", lambda *a, **k: pytest.fail("No model or export calls expected"))
    pipeline.main(["--workdir", str(tmp_path), "--stage2-csv", str(stage2), "--output-dir", str(output)])
    assert (output / "top300_refs.txt").read_text() == ""
    with (output / "final_candidates.csv").open() as fh:
        assert list(csv.DictReader(fh)) == []
    with (output / "exported_300cifs" / "export_300index.csv").open() as fh:
        assert list(csv.DictReader(fh)) == []
    summary = json.loads((output / "selection_summary.json").read_text())
    assert summary["counts"]["input_rows"] == 0


def test_voltage_gate_rejects_missing_failed_and_unstable_rows(tmp_path):
    hull = tmp_path / "hull.csv"
    voltage = tmp_path / "voltage.csv"
    final = tmp_path / "final.csv"
    audit = tmp_path / "audit.csv"
    write_csv(hull, [{"file": f"{name}.cif", "path": f"/{name}.cif"} for name in ("good", "failed", "unstable", "missing")])
    write_csv(voltage, [
        {"file": "good.cif", "window_status": "scan_censored", "window": "1.0", "stable_at_target": "", "target_voltage": "", "stable_intervals_json": json.dumps([{"V_red": 0, "V_ox": 1, "window": 1}])},
        {"file": "failed.cif", "window_status": "calculation_failed", "window": "", "stable_at_target": "", "target_voltage": "", "stable_intervals_json": "[]"},
        {"file": "unstable.cif", "window_status": "no_stable_window", "window": "", "stable_at_target": "", "target_voltage": "", "stable_intervals_json": "[]"},
    ])
    assert pipeline.filter_voltage(hull, voltage, final, audit) == 1
    with final.open() as fh:
        assert [row["file"] for row in csv.DictReader(fh)] == ["good.cif"]
    with audit.open() as fh:
        rows = {row["file"]: row for row in csv.DictReader(fh)}
    assert rows["missing.cif"]["voltage_filter_reason"] == "missing_voltage_result"
    assert rows["failed.cif"]["voltage_filter_reason"] == "calculation_failed"


def test_target_voltage_requires_exact_successful_evaluation(tmp_path):
    hull, voltage, final, audit = [tmp_path / f"{n}.csv" for n in ("hull", "voltage", "final", "audit")]
    write_csv(hull, [{"file": f"{n}.cif"} for n in ("good", "off_target", "wrong_target")])
    write_csv(voltage, [
        {"file": "good.cif", "window_status": "stable_window", "window": "1", "stable_at_target": "True", "target_voltage": "4", "stable_intervals_json": "[]"},
        {"file": "off_target.cif", "window_status": "stable_window", "window": "1", "stable_at_target": "False", "target_voltage": "4", "stable_intervals_json": "[]"},
        {"file": "wrong_target.cif", "window_status": "stable_window", "window": "1", "stable_at_target": "True", "target_voltage": "3", "stable_intervals_json": "[]"},
    ])
    assert pipeline.filter_voltage(hull, voltage, final, audit, target_voltage=4) == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_voltage_width_cannot_pass(tmp_path, value):
    hull, voltage, final, audit = [tmp_path / f"{n}.csv" for n in ("hull", "voltage", "final", "audit")]
    write_csv(hull, [{"file": "bad.cif"}])
    write_csv(voltage, [{"file": "bad.cif", "window_status": "stable_window", "window": value}])
    assert pipeline.filter_voltage(hull, voltage, final, audit) == 0


def test_voltage_command_passes_optional_target(monkeypatch, tmp_path):
    seen = []
    output = tmp_path / "voltage.csv"
    def run(cmd, cwd=None):
        seen.extend(cmd)
        output.touch()
    monkeypatch.setattr(pipeline, "run_command", run)
    pipeline.run_voltage(Path("voltage.py"), Path("hull.csv"), output, 0.05, 0.001, tmp_path, target_voltage=4.25)
    assert seen[seen.index("--target-voltage") + 1] == "4.25"


def test_parallel_pipeline_runs_novelty_after_voltage_and_preserves_intermediate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import candidate_selection
    import parallel_screening

    stage2 = tmp_path / "stage2.csv"
    write_csv(stage2, [{"quick_score": 1, "path": "/source.extxyz", "frame": 0,
                       "score_kind": "li_periodic_geometry_proxy_v1"}])
    output = tmp_path / "output"
    events = []
    monkeypatch.setattr(candidate_selection, "select_candidates", lambda *a, **k:
                        SimpleNamespace(refs=["/source.extxyz::0"], counts={"selected": 1}, metadata={}))
    monkeypatch.setattr(candidate_selection, "write_selection_audit", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "run_export", lambda *a, **k: tmp_path / "index.csv")
    monkeypatch.setattr(pipeline, "rename_index", lambda path, name: path)

    def thermal(args, index, workdir):
        events.append("voltage")
        assert args.final_out == output / "pre_novelty_candidates.csv"
        write_csv(args.final_out, [{"file": "candidate.cif", "path": "/candidate.cif",
                                   "passes_voltage_filter": True}])

    def novelty(args, workdir):
        events.append("novelty")
        assert args.final_out == output / "final_candidates.csv"
        assert args.pre_novelty_out.exists()
        args.final_out.write_text("file,passes_novelty_filter\ncandidate.cif,True\n")
        return {"status": "completed"}

    monkeypatch.setattr(parallel_screening, "run_parallel_screening", thermal)
    monkeypatch.setattr(pipeline, "finalize_novelty", novelty)
    monkeypatch.chdir(tmp_path)
    pipeline.main(["--workdir", str(tmp_path), "--stage2-csv", str(stage2),
                   "--output-dir", str(output), "--gpu-workers", "2"])
    assert events == ["voltage", "novelty"]
    assert (output / "pre_novelty_candidates.csv").exists()


def test_novelty_failure_aborts_pipeline_finalize(tmp_path):
    args = pipeline.parse_args([])
    args.pre_novelty_out = tmp_path / "voltage_pass.csv"
    args.final_out = tmp_path / "final.csv"
    args.novelty_audit = tmp_path / "audit.csv"
    args.novelty_summary = tmp_path / "summary.json"
    write_csv(args.pre_novelty_out, [{"file": "missing.cif", "path": str(tmp_path / "missing.cif")}])
    args.final_out.write_text("file\nstale.cif\n")
    with pytest.raises(RuntimeError, match="Novelty screening incomplete"):
        pipeline.finalize_novelty(args, tmp_path)
    with args.final_out.open() as fh:
        assert list(csv.DictReader(fh)) == []
    assert json.loads(args.novelty_summary.read_text())["status"] == "incomplete"


def test_explicit_skip_does_not_claim_novelty(tmp_path):
    args = pipeline.parse_args(["--skip-novelty"])
    args.pre_novelty_out = tmp_path / "voltage_pass.csv"
    args.final_out = tmp_path / "final.csv"
    args.novelty_summary = tmp_path / "summary.json"
    write_csv(args.pre_novelty_out, [{"file": "candidate.cif", "path": "/candidate.cif"}])
    summary = pipeline.finalize_novelty(args, tmp_path)
    assert summary["status"] == "skipped"
    with args.final_out.open() as fh:
        row = next(csv.DictReader(fh))
    assert row["novelty_status"] == "not_checked"
    assert row["passes_novelty_filter"] == "False"


def test_novelty_output_cannot_overwrite_stage2_input(tmp_path, monkeypatch):
    stage2 = tmp_path / "input.csv"
    stage2.write_text("protected input\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        pipeline.main(["--workdir", str(tmp_path), "--stage2-csv", str(stage2),
                       "--output-dir", str(tmp_path / "out"), "--final-out", str(stage2)])
    assert stage2.read_text() == "protected input\n"


@pytest.mark.parametrize("output_option", ["--final-out", "--pre-novelty-out", "--novelty-audit", "--novelty-summary"])
def test_novelty_output_cannot_delete_generated_source_structures(tmp_path, monkeypatch, output_option):
    source = tmp_path / "relaxed.extxyz"
    source.write_text("protected generated structures\n")
    stage2 = tmp_path / "stage2.csv"
    write_csv(stage2, [{"quick_score": 1, "path": str(source), "frame": 0,
                       "score_kind": "li_periodic_geometry_proxy_v1"}])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="must not overwrite source structures"):
        pipeline.main(["--workdir", str(tmp_path), "--stage2-csv", str(stage2),
                       "--output-dir", str(tmp_path / "out"), output_option, str(source)])
    assert source.read_text() == "protected generated structures\n"


def test_missing_novelty_dataset_reason_is_reported_by_pipeline(tmp_path):
    from pymatgen.core import Lattice, Structure

    structure = tmp_path / "candidate.cif"
    Structure(Lattice.cubic(3.4), ["Li", "Cl"], [[0, 0, 0], [.5, .5, .5]]).to(filename=str(structure))
    args = pipeline.parse_args([])
    args.pre_novelty_out = tmp_path / "voltage_pass.csv"
    args.final_out = tmp_path / "final.csv"
    args.novelty_audit = tmp_path / "audit.csv"
    args.novelty_summary = tmp_path / "summary.json"
    args.novelty_training_data = [tmp_path / "missing_train.zip"]
    args.novelty_reference_data = [tmp_path / "missing_reference.gz"]
    write_csv(args.pre_novelty_out, [{"file": structure.name, "path": str(structure)}])
    with pytest.raises(RuntimeError, match="source_file_unavailable.*Required structure dataset file is missing") as exc:
        pipeline.finalize_novelty(args, tmp_path)
    assert "missing_train.zip" in str(exc.value)
    assert "missing_reference.gz" in str(exc.value)
    with args.final_out.open() as fh:
        assert list(csv.DictReader(fh)) == []


@pytest.mark.parametrize("arguments", [["--novelty-training-splits", ""],
    ["--novelty-training-splits", "train,train"],
    ["--skip-novelty", "--novelty-training-data", "/train.zip"]])
def test_invalid_novelty_cli_configuration_is_rejected(arguments):
    with pytest.raises(SystemExit):
        pipeline.parse_args(arguments)
