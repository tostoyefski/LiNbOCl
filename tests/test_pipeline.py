import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "mattergen_webapp" / "scripts"
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
