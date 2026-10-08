"""Offline checks for audited MD inputs and voltage plot interpretation."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_module(relative_path):
    spec = importlib.util.spec_from_file_location("analysis_tool_" + Path(relative_path).stem, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def md_module(monkeypatch):
    for name in ("torch", "chgnet"):
        monkeypatch.setitem(sys.modules, name, None)
    return load_module("workflow/transport/compute_ionic_conductivity.py")


@pytest.fixture
def voltage_plot():
    return load_module("workflow/analysis/plot_voltage_window.py")


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def final_candidate(**updates):
    row = {
        "file": "accepted.cif", "passes_voltage_filter": "True",
        "window_status": "stable_window", "window": "1",
        "target_voltage": "", "stable_at_target": "", "error": "",
    }
    return {**row, **updates}


def test_md_defaults_are_repo_results_independent_of_cwd(md_module, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = md_module.parse_args([])
    assert Path(args.csv) == REPO / "results/top300_run/final_candidates.csv"
    assert Path(args.cif_dir) == REPO / "results/top300_run/exported_300cifs"
    assert Path(args.out) == REPO / "results/transport/chgnet_ionic_conductivity.csv"
    assert Path(args.traj_dir) == REPO / "results/transport/md_traj"
    assert args.temperature == 700
    assert args.temperatures is None
    assert args.timestep_fs == 2
    assert args.total_ps == 10


@pytest.mark.parametrize("changes", [
    {"passes_voltage_filter": "False"},
    {"passes_voltage_filter": ""},
    {"window_status": "calculation_failed"},
    {"window_status": "no_stable_window"},
    {"window_status": ""},
    {"window": "nan"},
    {"window": "inf"},
    {"window": "-1"},
    {"window": "0"},
    {"window": ""},
    {"error": "partial competing-phase failure"},
    {"stable_at_target": "False"},
    {"target_voltage": "4", "stable_at_target": ""},
    {"target_voltage": "nan", "stable_at_target": "True"},
])
def test_md_rejects_unqualified_rows_before_model_load(md_module, tmp_path, capsys, changes):
    csv_path = write_csv(tmp_path / "input.csv", [final_candidate(**changes)])
    assert md_module.load_md_candidates(csv_path) == []
    md_module.main(["--csv", str(csv_path)])
    output = capsys.readouterr().out
    assert "[REJECT] accepted.cif:" in output
    assert "No eligible final candidates" in output


def test_md_accepts_only_audited_rows_and_keeps_valid_negative_target(md_module, tmp_path):
    path = write_csv(tmp_path / "input.csv", [
        final_candidate(file="rejected.cif", passes_voltage_filter="False"),
        final_candidate(file="bounded.cif"),
        final_candidate(file="censored.cif", window_status="scan_censored", target_voltage="-0.5", stable_at_target="True"),
    ])
    assert [row["file"] for row in md_module.load_md_candidates(path)] == ["bounded.cif", "censored.cif"]


def test_md_requires_final_selection_metadata(md_module, tmp_path):
    path = write_csv(tmp_path / "legacy.csv", [{"file": "legacy.cif"}])
    with pytest.raises(ValueError, match="audited final_candidates.csv"):
        md_module.load_md_candidates(path)


def test_plot_reads_all_intervals_and_marks_scan_limits(voltage_plot, tmp_path):
    intervals = [
        {"V_red": -0.5, "V_ox": 0.5, "window": 1, "lower_bound_censored": True},
        {"V_red": 2, "V_ox": 2.5, "window": 0.5, "upper_bound_censored": True},
    ]
    path = write_csv(tmp_path / "windows.csv", [{
        "formula": "Li2Cl2", "window_status": "scan_censored",
        "V_red": -0.5, "V_ox": 0.5, "window": 1,
        "stable_intervals_json": json.dumps(intervals),
    }])
    rows = voltage_plot.read_rows(path)
    assert len(rows) == 1
    assert len(rows[0]["stable_intervals"]) == 2
    png_path = tmp_path / "plots" / "window.png"
    figure = voltage_plot.plot_voltage_window(rows, png_path, False)
    axes = figure.axes[0]
    assert [(bar.get_x(), bar.get_width()) for bar in axes.patches] == [(-0.5, 1), (2, 0.5)]
    assert len(axes.collections) == 2
    assert axes.get_xlim()[0] < -0.5
    assert "boundary unobserved" in axes.get_legend().get_texts()[0].get_text()
    assert png_path.read_bytes().startswith(b"\x89PNG")


def test_plot_legacy_csv_remains_usable(voltage_plot, tmp_path):
    path = write_csv(tmp_path / "legacy.csv", [{"formula": "LiCl", "V_red": "0", "V_ox": "1", "window": "1"}])
    rows = voltage_plot.read_rows(path)
    assert rows[0]["window"] == 1
    assert rows[0]["stable_intervals"][0]["lower_bound_censored"] is False


@pytest.mark.parametrize("changes,reason", [
    ({"window_status": "calculation_failed", "error": "phase diagram failed"}, "phase diagram failed"),
    ({"window_status": "no_stable_window"}, "no_stable_window"),
    ({"window_status": ""}, "window_status="),
    ({"V_red": ""}, "could not convert"),
    ({"V_ox": "nan"}, "must be finite"),
    ({"window": "inf"}, "must be finite"),
    ({"window": "0"}, "disagrees"),
    ({"V_red": "2"}, "invalid stable interval bounds"),
    ({"stable_intervals_json": "[]"}, "no stable intervals"),
    ({"stable_intervals_json": "{not-json"}, "Expecting"),
    ({"passes_voltage_filter": "False"}, "passes_voltage_filter"),
])
def test_plot_skips_failed_or_invalid_rows_with_reason(voltage_plot, tmp_path, capsys, changes, reason):
    row = {"formula": "LiCl", "window_status": "stable_window", "V_red": "0", "V_ox": "1", "window": "1"}
    path = write_csv(tmp_path / "bad.csv", [{**row, **changes}])
    assert voltage_plot.read_rows(path) == []
    assert reason in capsys.readouterr().out


def test_plot_empty_or_rejected_csv_outputs_no_window_information(voltage_plot, tmp_path, capsys):
    path = write_csv(tmp_path / "empty.csv", [], fields=["formula", "V_red", "V_ox", "window"])
    out = tmp_path / "empty.png"
    voltage_plot.main(["--csv", str(path), "--out", str(out)])
    assert out.exists()
    assert "No valid stability windows" in capsys.readouterr().out


def test_plot_defaults_and_explicit_legacy_paths(voltage_plot, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = voltage_plot.parse_args([])
    assert args.csv == REPO / "results/top300_run/chgnet_voltage_window_top300.csv"
    assert args.out == REPO / "results/analysis/chgnet_voltage_window_top300.png"
    assert voltage_plot.parse_args(["old.csv", "old.png"]).csv == "old.csv"
    assert voltage_plot.parse_args(["old.csv", "--csv", "named.csv"]).csv == "named.csv"


@pytest.mark.parametrize("script", [
    "workflow/transport/compute_ionic_conductivity.py",
    "workflow/transport/plot_arrhenius.py",
    "workflow/transport/plot_msd_from_traj.py",
    "workflow/transport/li_density_from_traj.py",
    "workflow/analysis/plot_voltage_window.py",
])
def test_moved_cli_help_runs_from_other_directory_without_models(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / script), "--help"], cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
