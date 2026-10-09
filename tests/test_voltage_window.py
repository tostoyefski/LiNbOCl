"""Deterministic electrochemical-window tests; no model, API or network needed."""

import importlib.util
import csv
import json
import sys
from pathlib import Path

import pytest
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Element


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def voltage_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "workflow" / "pipeline"))
    # Any eager model or API dependency now fails the import, ensuring these
    # regressions remain usable in the small, pure-pymatgen environment.
    for name in ("chgnet", "mp_api", "compute_ehull_chgnet"):
        monkeypatch.setitem(sys.modules, name, None)
    module_name = "voltage_under_test_workflow_pipeline"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "workflow" / "pipeline" / "compute_voltage_window.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def evaluate(module, entries, voltages, energy_tol=1e-3, mu_ref=-1.0, **kwargs):
    candidate = module.Candidate(row={"file": "artificial.cif"}, structure=None, entry=entries[-1])
    return module.evaluate_voltage_window(
        candidate, entries, len(entries) - 1, Element("Li"), mu_ref, voltages, energy_tol, **kwargs
    )


def test_stability_starting_at_zero_has_window(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    assert tuple(evaluate(voltage_module, entries, [0, 0.5, 1, 1.5])) == (0, 1, 1)


def test_full_scan_stability_is_censored(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1])
    assert tuple(result) == (0, 1, 1)
    assert result.window_status == "scan_censored"
    assert result.lower_bound_censored and result.upper_bound_censored


def test_full_scan_instability_is_distinct(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", 1)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1])
    assert tuple(result) == (None, None, None)
    assert result.window_status == "no_stable_window"
    assert result.stable_intervals == []


def test_upper_bound_is_last_stable_grid_point(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("Li2Cl", -3.5), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.25, 0.5, 0.75, 1, 1.25])
    assert tuple(result) == (0.5, 1, 0.5)
    assert result.lower_boundary_bracket == (0.25, 0.5)
    assert result.upper_boundary_bracket == (1, 1.25)


def test_diagram_failure_is_unknown_not_instability(voltage_module, monkeypatch):
    real_diagram = voltage_module.GrandPotentialPhaseDiagram

    def fail_in_middle(entries, chempots):
        if chempots[Element("Li")] == -1.5:
            raise RuntimeError("synthetic phase-diagram failure")
        return real_diagram(entries, chempots)

    monkeypatch.setattr(voltage_module, "GrandPotentialPhaseDiagram", fail_in_middle)
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1, 1.5])
    assert tuple(result) == (None, None, None)
    assert result.window_status == "calculation_failed"
    assert result.stable_intervals == []
    assert "synthetic phase-diagram failure" in result.error


def test_target_voltage_is_evaluated_directly(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("Li2Cl", -3.5), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1, 1.5], target_voltage=0.49)
    assert tuple(result) == (0.5, 1, 0.5)
    assert result.stable_at_target is False
    assert result.target_e_above_hull_eV == pytest.approx(0.01)
    # A directly stable target need not equal a stable grid point.
    result = evaluate(voltage_module, entries, [0, 0.5, 1, 1.5], target_voltage=1.0005)
    assert result.stable_at_target is True
    assert result.target_e_above_hull_eV == pytest.approx(0.0005)


def test_grand_hull_uses_non_li_atom_units(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("Li2Cl2", -4)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1], energy_tol=0.15, target_voltage=1.2)
    assert result.target_e_above_hull_eV == pytest.approx(0.2)
    assert result.stable_at_target is False
    assert result.e_above_hull_unit == "eV/non-Li atom"


def test_original_total_energy_and_reference_baseline_preserved(voltage_module):
    entries = [PDEntry("Li", -10), PDEntry("Cl", -20), PDEntry("LiCl", -31)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1, 1.5], mu_ref=-10)
    assert tuple(result) == (0, 1, 1)


def test_target_unspecified_remains_unknown_in_record(voltage_module):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    record = evaluate(voltage_module, entries, [0, 0.5, 1]).as_record()
    assert record["target_voltage"] is None
    assert record["stable_at_target"] is None
    assert record["target_e_above_hull_eV"] is None
    assert json.loads(record["stable_intervals_json"])[0]["window"] == 1


def test_target_failure_cannot_leave_a_window(voltage_module, monkeypatch):
    real_evaluate = voltage_module.evaluate_voltage_point

    def fail_at_target(entries, candidate_index, work_element, mu_ref, voltage):
        if voltage == 0.7:
            raise RuntimeError("target failure")
        return real_evaluate(entries, candidate_index, work_element, mu_ref, voltage)

    monkeypatch.setattr(voltage_module, "evaluate_voltage_point", fail_at_target)
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1], target_voltage=0.7)
    assert result.window_status == "calculation_failed"
    assert tuple(result) == (None, None, None)
    assert result.stable_intervals == []
    assert result.stable_at_target is None
    assert "target failure" in result.error


def test_multiple_intervals_stay_separate_and_policy_is_deterministic(voltage_module, monkeypatch):
    # An intentionally nonphysical point evaluator verifies that sampled
    # disconnected runs cannot accidentally be joined into a fictitious window.
    monkeypatch.setattr(
        voltage_module,
        "evaluate_voltage_point",
        lambda entries, candidate_index, work_element, mu_ref, voltage: (
            0 if voltage in (0, 1, 3, 4) else 1
        ),
    )
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 1, 2, 3, 4, 5])
    assert tuple(result) == (0, 1, 1)
    assert len(result.stable_intervals) == 2
    assert result.interval_policy == "widest_then_lowest_voltage"
    assert result.stable_intervals[1]["V_red"] == 3
    assert result.stable_intervals[1]["V_ox"] == 4
    assert result.stable_intervals[1]["lower_boundary_bracket"] == (2, 3)


@pytest.mark.parametrize("threshold", ["-0.001", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_threshold(voltage_module, threshold):
    with pytest.raises(SystemExit) as exc:
        voltage_module.parse_args(["--threshold", threshold])
    assert exc.value.code == 2


def test_cli_keeps_strict_tolerance_and_no_implicit_target(voltage_module):
    args = voltage_module.parse_args([])
    assert args.threshold == 1e-3
    assert args.target_voltage is None
    output = ROOT / "results" / "top300_run"
    assert Path(args.stable_csv) == output / "chgnet_hull_top300_filtered.csv"
    assert Path(args.out) == output / "chgnet_voltage_window_top300.csv"


@pytest.mark.parametrize(
    "argv",
    [
        ["--voltage-min", "nan"],
        ["--voltage-max", "inf"],
        ["--voltage-step", "0"],
        ["--voltage-step", "nan"],
        ["--target-voltage", "inf"],
        ["--voltage-min", "2", "--voltage-max", "1"],
    ],
)
def test_cli_rejects_invalid_voltage_configuration(voltage_module, argv):
    with pytest.raises(SystemExit) as exc:
        voltage_module.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_hull_is_calculation_failure(voltage_module, monkeypatch, nonfinite):
    class NonfiniteDiagram:
        def __init__(self, entries, chempots):
            pass

        def get_e_above_hull(self, entry):
            return nonfinite

    monkeypatch.setattr(voltage_module, "GrandPotentialPhaseDiagram", NonfiniteDiagram)
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1])
    assert result.window_status == "calculation_failed"
    assert tuple(result) == (None, None, None)
    assert "non-finite" in result.error


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("entry_index", [0, 2, 3])
def test_nonfinite_original_phase_energy_is_calculation_failure(
    voltage_module, nonfinite, entry_index
):
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("Li2Cl", -3.5), PDEntry("LiCl", -2)]
    entries[entry_index] = PDEntry(entries[entry_index].composition, nonfinite)
    result = evaluate(voltage_module, entries, [0, 0.5, 1, 1.5], target_voltage=0.25)
    assert result.window_status == "calculation_failed"
    assert tuple(result) == (None, None, None)
    assert result.stable_intervals == []
    assert result.stable_at_target is None
    assert "non-finite total energy" in result.error


def test_negative_finite_target_voltage_is_allowed(voltage_module):
    args = voltage_module.parse_args(["--target-voltage", "-0.5"])
    assert args.target_voltage == -0.5
    entries = [PDEntry("Li", -1), PDEntry("Cl", 0), PDEntry("LiCl", -2)]
    result = evaluate(voltage_module, entries, [0, 0.5, 1], target_voltage=-0.5)
    assert result.stable_at_target is True
    assert result.target_e_above_hull_eV == 0


def test_cli_rejects_legacy_energies_without_uniform_snapshot(voltage_module, tmp_path):
    input_csv = tmp_path / "legacy.csv"
    output_csv = tmp_path / "voltage.csv"
    rows = [{"file": "legacy.cif", "path": str(tmp_path / "legacy.cif"),
             "formula": "LiCl", "chemsys": "Cl-Li", "energy_total_eV": -2}]
    with input_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # The fixture blocks both model and API packages. A stale table produces a
    # failed output row instead of refetching competitors or mixing baselines.
    voltage_module.main(["--stable-csv", str(input_csv), "--out", str(output_csv)])
    with output_csv.open(newline="") as handle:
        actual = list(csv.DictReader(handle))
    assert actual[0]["window_status"] == "calculation_failed"
    assert actual[0]["window"] == actual[0]["stable_at_target"] == ""
    assert actual[0]["error"]
