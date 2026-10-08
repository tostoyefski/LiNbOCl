import importlib.util
import sys
import shlex
from pathlib import Path

import pytest
from pydantic import ValidationError


BACKEND = Path(__file__).resolve().parents[1] / "mattergen_webapp" / "backend" / "main.py"
spec = importlib.util.spec_from_file_location("screening_backend", BACKEND)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


def test_default_rules_and_selection():
    screen = backend.ScreenRequest()
    assert screen.require_charge_balance and screen.use_smact
    assert screen.required_elements == ["Li", "Nb", "O", "Cl"]
    assert backend.Top300Request().selection_mode == "diverse"
    assert backend.Top300Request().voltage_threshold == 0.001


def test_standalone_commands_pass_new_parameters(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    screen = backend.ScreenRequest(require_charge_balance=False, use_smact=False, filter_light_oxy=False)
    backend.run_screen(screen)
    cmd = calls[-1]
    assert Path(cmd[1]).parent == BACKEND.parents[2] / "workflow" / "pipeline"
    assert "--no-charge-balance" in cmd and "--no-smact" in cmd and "--no-light-oxy" in cmd
    assert "--light-oxy" not in cmd
    assert Path(cmd[cmd.index("--screened-out") + 1]) == Path(screen.out).with_name("screened_out.csv")
    assert cmd[cmd.index("--required-elements") + 1:cmd.index("--required-elements") + 5] == ["Li", "Nb", "O", "Cl"]
    backend.run_top300(backend.Top300Request(target_voltage=4.25, min_voltage_window=0.2, selection_mode="score"))
    cmd = calls[-1]
    assert Path(cmd[1]).parent == BACKEND.parents[2] / "workflow" / "pipeline"
    assert cmd[cmd.index("--selection-mode") + 1] == "score"
    assert cmd[cmd.index("--target-voltage") + 1] == "4.25"
    assert cmd[cmd.index("--min-voltage-window") + 1] == "0.2"


def test_full_pipeline_propagates_screen_and_voltage_rules(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    payload = backend.FullPipelineRequest(
        dd=backend.GenerateRequest(base_results_dir=str(tmp_path / "run space")),
        screen=backend.ScreenRequest(require_charge_balance=False, use_smact=False, filter_light_oxy=False),
        top300=backend.Top300Request(target_voltage=4.5, selection_mode="score"),
    )
    backend.run_full(payload)
    script = calls[-1][-1]
    assert "--no-charge-balance" in script and "--no-smact" in script
    commands = [shlex.split(line) for line in script.splitlines() if line.startswith("'python'")]
    screen, top = commands
    assert "--no-light-oxy" in screen and "--light-oxy" not in screen
    assert screen[screen.index("--required-elements") + 1:screen.index("--required-elements") + 5] == ["Li", "Nb", "O", "Cl"]
    assert top[top.index("--selection-mode") + 1] == "score"
    assert top[top.index("--target-voltage") + 1] == "4.5"
    assert top[top.index("--voltage-threshold") + 1] == "0.001"


def test_default_allowed_elements_cannot_exclude_required_elements():
    with pytest.raises(ValidationError, match="every required element"):
        backend.ScreenRequest(required_elements=["Li", "Fe"])
    assert backend.ScreenRequest(required_elements=["Li", "Fe"], allowed_elements=None).allowed_elements is None


def test_full_pipeline_anchors_default_screen_outputs_to_custom_run(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    output = tmp_path / "custom run"
    backend.run_full(backend.FullPipelineRequest(dd=backend.GenerateRequest(base_results_dir=str(output))))
    commands = [shlex.split(line) for line in calls[-1][-1].splitlines() if line.startswith("'python'")]
    screen, top = commands
    for flag, expected in (
        ("--out", output / "stage2_candidates.csv"),
        ("--screened-out", output / "screened_out.csv"),
        ("--refs-out", output / "screen_refs.txt"),
    ):
        assert Path(screen[screen.index(flag) + 1]) == expected
    assert Path(top[top.index("--stage2-csv") + 1]) == output / "stage2_candidates.csv"
    for flag, name in (
        ("--output-dir", ""),
        ("--refs-out", "top300_refs.txt"),
        ("--export-dir", "exported_300cifs"),
        ("--ehull-out", "chgnet_hull_top300.csv"),
        ("--filtered-out", "chgnet_hull_top300_filtered.csv"),
        ("--voltage-out", "chgnet_voltage_window_top300.csv"),
    ):
        assert Path(top[top.index(flag) + 1]) == output / "top300_run" / name


@pytest.mark.parametrize("kind", ["inside", "prefix_collision", "outside"])
def test_result_path_remapping_respects_directory_boundaries(tmp_path, kind):
    source, destination = tmp_path / "results", tmp_path / "custom run"
    if kind == "inside":
        original, expected = source / "nested" / "output.csv", destination / "nested" / "output.csv"
    elif kind == "prefix_collision":
        original = expected = tmp_path / "results_backup" / "output.csv"
    else:
        original = expected = tmp_path / "external" / "results" / "output.csv"
    assert Path(backend._remap_results_path(str(original), source, destination)) == expected


def test_full_pipeline_remaps_web_defaults_to_custom_run(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    defaults = backend.defaults()
    output = tmp_path / "custom run"
    backend.run_full(backend.FullPipelineRequest(
        dd=backend.GenerateRequest(base_results_dir=str(output)),
        screen=backend.ScreenRequest(out=defaults["screen_out"]),
        top300=backend.Top300Request(output_dir=defaults["top300_output_dir"],
                                    export_dir=defaults["top300_export_dir"]),
    ))
    screen, top = [shlex.split(line) for line in calls[-1][-1].splitlines() if line.startswith("'python'")]
    assert Path(screen[screen.index("--out") + 1]) == output / "stage2_candidates.csv"
    assert Path(top[top.index("--output-dir") + 1]) == output / "top300_run"
    assert Path(top[top.index("--export-dir") + 1]) == output / "top300_run" / "exported_300cifs"


def test_full_pipeline_preserves_explicit_external_absolute_outputs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    external = tmp_path / "external outputs"
    backend.run_full(backend.FullPipelineRequest(
        dd=backend.GenerateRequest(base_results_dir=str(tmp_path / "custom run")),
        screen=backend.ScreenRequest(out=str(external / "screen.csv"), refs_out=str(external / "screen refs.txt")),
        top300=backend.Top300Request(output_dir=str(external / "selected"),
                                    refs_out=str(external / "top refs.txt"),
                                    export_dir=str(external / "cifs")),
    ))
    screen, top = [shlex.split(line) for line in calls[-1][-1].splitlines() if line.startswith("'python'")]
    assert Path(screen[screen.index("--out") + 1]) == external / "screen.csv"
    assert Path(screen[screen.index("--refs-out") + 1]) == external / "screen refs.txt"
    assert Path(top[top.index("--output-dir") + 1]) == external / "selected"
    assert Path(top[top.index("--refs-out") + 1]) == external / "top refs.txt"
    assert Path(top[top.index("--export-dir") + 1]) == external / "cifs"


def test_full_pipeline_resolves_relative_run_and_output_paths_once(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    backend.run_full(backend.FullPipelineRequest(
        dd=backend.GenerateRequest(base_results_dir="results/custom run"),
        top300=backend.Top300Request(output_dir="selected", export_dir="cifs", refs_out="refs.txt"),
    ))
    screen, top = [shlex.split(line) for line in calls[-1][-1].splitlines() if line.startswith("'python'")]
    output = (backend.MATTERGEN_ROOT / "results" / "custom run").resolve()
    assert Path(screen[screen.index("--out") + 1]) == output / "stage2_candidates.csv"
    assert Path(top[top.index("--output-dir") + 1]) == output / "selected"
    assert Path(top[top.index("--export-dir") + 1]) == output / "selected" / "cifs"
    assert Path(top[top.index("--refs-out") + 1]) == output / "selected" / "refs.txt"


def test_full_pipeline_nested_run_root_does_not_remap_twice(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kw: calls.append(command))
    output = backend.DEFAULT_RESULTS_ROOT / "named run"
    backend.run_full(backend.FullPipelineRequest(dd=backend.GenerateRequest(base_results_dir=str(output))))
    screen, top = [shlex.split(line) for line in calls[-1][-1].splitlines() if line.startswith("'python'")]
    assert Path(screen[screen.index("--out") + 1]) == output / "stage2_candidates.csv"
    for flag, name in (
        ("--output-dir", ""), ("--refs-out", "top300_refs.txt"),
        ("--export-dir", "exported_300cifs"), ("--ehull-out", "chgnet_hull_top300.csv"),
    ):
        assert Path(top[top.index(flag) + 1]) == output / "top300_run" / name


def test_result_path_remapping_keeps_already_mapped_nested_outputs(tmp_path):
    source = tmp_path / "results"
    destination = source / "named run"
    path = destination / "top300_run" / "output.csv"
    assert Path(backend._remap_results_path(str(path), source, destination)) == path


@pytest.mark.parametrize("kwargs", [{"target_voltage": float("nan")}, {"voltage_threshold": float("inf")}, {"min_voltage_window": -1}, {"selection_mode": "invalid"}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValidationError):
        backend.Top300Request(**kwargs)
