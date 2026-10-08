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
    backend.run_screen(backend.ScreenRequest(require_charge_balance=False, use_smact=False, filter_light_oxy=False))
    cmd = calls[-1]
    assert "--no-charge-balance" in cmd and "--no-smact" in cmd and "--no-light-oxy" in cmd
    assert "--light-oxy" not in cmd
    assert cmd[cmd.index("--screened-out") + 1] == "results/screened_out.csv"
    assert cmd[cmd.index("--required-elements") + 1:cmd.index("--required-elements") + 5] == ["Li", "Nb", "O", "Cl"]
    backend.run_top300(backend.Top300Request(target_voltage=4.25, min_voltage_window=0.2, selection_mode="score"))
    cmd = calls[-1]
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


@pytest.mark.parametrize("kwargs", [{"target_voltage": float("nan")}, {"voltage_threshold": float("inf")}, {"min_voltage_window": -1}, {"selection_mode": "invalid"}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValidationError):
        backend.Top300Request(**kwargs)
