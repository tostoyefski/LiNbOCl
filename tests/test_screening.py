"""Synthetic crystal regression tests for chemical and periodic screening."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.io import write
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "mattergen_webapp" / "scripts" / "screen_all_extxyz.py"
spec = importlib.util.spec_from_file_location("screening_under_test", SCRIPT)
screen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = screen
spec.loader.exec_module(screen)


def chain(dim):
    return Structure(Lattice.orthorhombic(*([2.5] * dim + [12.0] * (3 - dim))), ["Li"], [[0, 0, 0]])


def triangle():
    return Structure(Lattice.cubic(15), ["Li"] * 3, [[0.3, 0.3, 0.3], [0.43, 0.3, 0.3], [0.365, 0.4126, 0.3]])


def test_absent_li_is_missing_and_never_scores():
    row = screen.comp_features(Structure(Lattice.cubic(8), ["O"], [[0, 0, 0]]))
    row.update(screen.li_connectivity(Structure(Lattice.cubic(8), ["O"], [[0, 0, 0]])))
    assert np.isnan(row["min_li_li"])
    assert screen.quick_score_row(row) == 0


def test_isolated_triangle_is_not_a_periodic_channel():
    result = screen.li_connectivity(triangle())
    assert result["li_conn"] == 0
    assert result["li_percolation_dim"] == 0
    assert screen.quick_score_row(result) == 0


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_periodic_channels_have_correct_winding_dimension(dim):
    result = screen.li_connectivity(chain(dim))
    assert result["li_percolation_dim"] == dim
    assert result["li_percolation_fraction"] == 1
    assert result["li_channel_score"] == pytest.approx(dim / 3)
    assert result["min_li_li"] == pytest.approx(2.5)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_periodic_rank_and_score_do_not_depend_on_cell_representation(dim):
    original = screen.li_connectivity(chain(dim))
    enlarged = screen.li_connectivity(chain(dim) * (2, 3, 2))
    for key in ("li_percolation_dim", "li_percolation_fraction", "li_channel_score", "min_li_li"):
        assert enlarged[key] == pytest.approx(original[key])


def test_real_smact_rejects_the_fixed_non_neutral_composition():
    assert screen.smact_ok(Composition("Li2O")) is True
    assert screen.smact_ok(Composition("LiO")) is False


def test_charge_balance_exception_does_not_pass(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic chemistry error")
    monkeypatch.setattr(Composition, "oxi_state_guesses", broken)
    assert screen.charge_balance_ok(Composition("Li2O")) is False


def test_smact_exception_does_not_pass(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic SMACT error")
    monkeypatch.setattr(screen, "smact_pauling_test", broken)
    assert screen.smact_ok(Composition("Li2O")) is False


def test_cli_actually_filters_light_oxy_and_required_species(tmp_path, monkeypatch):
    folder = tmp_path / "input"
    folder.mkdir()
    structures = [
        Structure(Lattice.cubic(10), ["Li", "Nb", "O", "Cl"], [[0, 0, 0], [.2, .2, .2], [.4, .4, .4], [.6, .6, .6]]),
        Structure(Lattice.cubic(10), ["Li", "Nb", "Cl"], [[0, 0, 0], [.3, .3, .3], [.6, .6, .6]]),
        Structure(Lattice.cubic(10), ["Nb", "O", "Cl"], [[0, 0, 0], [.3, .3, .3], [.6, .6, .6]]),
    ]
    write(folder / "relaxed.extxyz", [AseAtomsAdaptor.get_atoms(s) for s in structures])
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--workdir", str(tmp_path), "--base", "input", "--out", "accepted.csv", "--light-oxy", "0.05", "0.35", "--required-elements", "Li", "Nb", "O", "Cl", "--no-charge-balance", "--no-smact"])
    screen.main()
    accepted = pd.read_csv(tmp_path / "accepted.csv")
    rejected = pd.read_csv(tmp_path / "screened_out.csv")
    assert accepted.empty
    assert len(rejected) == 3
    assert "light_oxy_out_of_range" in rejected.loc[rejected.frame == 0, "filtered_reasons"].iloc[0]
    assert "missing_required_elements:O" in rejected.loc[rejected.frame == 1, "filtered_reasons"].iloc[0]
    assert "missing_li" in rejected.loc[rejected.frame == 2, "filtered_reasons"].iloc[0]


def test_mixed_valence_is_preserved_with_fixed_stoichiometry():
    composition = Composition("LiNb2O3Cl2")
    charge = screen.charge_balance_result(composition)
    assert charge.status == "pass"
    assert charge.guesses[0]["Nb"] == pytest.approx(3.5)
    assert screen.smact_ok(composition) is True
    assert screen.charge_balance_ok(Composition("Fe3O4")) is True
    assert screen.smact_ok(Composition("Fe3O4")) is True


def test_smact_receives_integer_valences_enegs_and_exact_formula(monkeypatch):
    calls = []
    original = screen.smact_pauling_test
    def record(states, enegs, **kwargs):
        calls.append((states, enegs, kwargs["symbols"]))
        return original(states, enegs, **kwargs)
    monkeypatch.setattr(screen, "smact_pauling_test", record)
    assert screen.smact_ok(Composition("LiNb2O3Cl2")) is True
    states, enegs, symbols = calls[0]
    assert len(states) == len(enegs) == len(symbols) == 8
    assert all(isinstance(v, int) for v in states)
    assert sum(states) == 0
    assert symbols.count("Nb") == 2
    assert sorted(v for v, s in zip(states, symbols) if s == "Nb") == [3, 4]
    assert all(e == screen.SmactElement(s).pauling_eneg for e, s in zip(enegs, symbols))


def test_chemistry_unknown_and_error_have_distinct_status(monkeypatch):
    assert screen.charge_balance_result(Composition("LiNe")).status == "unknown"
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic chemistry error")
    monkeypatch.setattr(Composition, "oxi_state_guesses", broken)
    result = screen.charge_balance_result(Composition("Li2O"))
    assert result.status == "error"
    assert result.ok is False
    assert "synthetic chemistry error" in result.reason


def test_unknown_error_are_audited_and_rejected(monkeypatch):
    structure = Structure(Lattice.cubic(8), ["Li", "Cl"], [[0, 0, 0], [.5, .5, .5]])
    monkeypatch.setattr(screen, "charge_balance_result", lambda *a, **kw: screen.ChemicalCheck("unknown", "fixture_unknown"))
    row = screen.screen_structure(structure, light_oxy=None)
    assert row["charge_balance_ok"] is False
    assert row["smact_ok"] is False
    assert "charge_balance_unknown" in row["filtered_reasons"]
    assert "smact_unknown" in row["filtered_reasons"]


def test_missing_smact_is_fail_closed_and_default_cli_terminates(monkeypatch):
    monkeypatch.setattr(screen, "smact_pauling_test", None)
    assert screen.smact_ok(Composition("LiCl")) is False
    assert screen.smact_result(Composition("LiCl")).status == "unknown"
    with pytest.raises(SystemExit) as exc:
        screen.main([])
    assert exc.value.code == 2


def test_default_cli_enables_both_chemical_filters_and_requires_li():
    args = screen.build_parser().parse_args([])
    assert args.require_charge_balance is True
    assert args.use_smact is True
    assert args.required_elements == ["Li"]


def test_density_entropy_and_minimum_distance_do_not_add_score():
    row = {"li_percolation_dim": 0, "li_channel_score": 0, "li_conn": 1, "min_li_li": 999, "density": 2.6, "f_o": .2, "hal_entropy": .7}
    assert screen.quick_score_row(row) == 0
    row.update(li_percolation_dim=1, li_channel_score=1 / 3)
    before = screen.quick_score_row(row)
    row.update(min_li_li=2, density=0.1, hal_entropy=0, f_o=.9)
    assert screen.quick_score_row(row) == before


def test_periodic_fraction_accounts_for_an_isolated_component():
    # The first Li forms a 1D chain. The second is farther than r_cut from it
    # and from its own images by translating it in a doubled x supercell.
    structure = Structure(Lattice.orthorhombic(5, 20, 20), ["Li"] * 3, [[0, 0, 0], [.5, 0, 0], [.25, .5, .5]])
    result = screen.li_connectivity(structure)
    assert result["li_percolation_dim"] == 1
    assert result["li_percolation_fraction"] == pytest.approx(2 / 3)
    assert result["li_component_count"] == 2
    assert result["li_channel_score"] == pytest.approx(2 / 9)


def test_coincident_sites_are_rejected_before_scoring():
    structure = Structure(Lattice.cubic(10), ["Li", "Cl"], [[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="coincident"):
        screen.screen_structure(structure, light_oxy=None)


def test_library_value_error_is_an_error_not_unknown(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("synthetic library failure")
    monkeypatch.setattr(Composition, "oxi_state_guesses", broken)
    assert screen.charge_balance_result(Composition("LiCl")).status == "error"


def test_target_element_whitelist_rejects_extra_elements():
    structure = Structure(Lattice.cubic(12), ["Li", "Nb", "O", "Cl", "Na"], [[0, 0, 0], [.2, .2, .2], [.4, .4, .4], [.6, .6, .6], [.8, .8, .8]])
    row = screen.screen_structure(structure, required_elements=["Li", "Nb", "O", "Cl"], allowed_elements=["Li", "Nb", "O", "Cl"], light_oxy=None, require_charge_balance=False, use_smact=False)
    assert "unexpected_elements:Na" in row["filtered_reasons"]


def test_finite_triangle_crossing_cell_boundary_still_has_zero_rank():
    structure = triangle()
    structure.translate_sites(list(range(3)), [.65, 0, 0], frac_coords=True, to_unit_cell=True)
    result = screen.li_connectivity(structure)
    assert result["li_percolation_dim"] == 0
    assert result["li_conn"] == 0


def test_cli_default_chemistry_accepts_neutral_target_and_audits_non_neutral(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    species = ["Li", "Nb", "O", "Cl", "Cl", "Cl", "Cl"]
    coordinates = [[0, 0, 0], [.2, .15, .1], [.3, .3, .3], [.4, .4, .4], [.5, .55, .5], [.7, .7, .7], [.8, .9, .8]]
    valid = Structure(Lattice.orthorhombic(2.5, 12, 12), species, coordinates)
    invalid = Structure(Lattice.orthorhombic(2.5, 12, 12), ["Li"] * 3 + species, [[.8, .1, .8], [.6, .1, .8], [.4, .1, .8], *coordinates])
    write(folder / "relaxed.extxyz", [AseAtomsAdaptor.get_atoms(s) for s in (valid, invalid)])
    assert screen.main(["--workdir", str(tmp_path), "--base", "input", "--out", "accepted.csv", "--required-elements", "Li", "Nb", "O", "Cl", "--allowed-elements", "Li", "Nb", "O", "Cl"]) == 0
    accepted = pd.read_csv(tmp_path / "accepted.csv")
    rejected = pd.read_csv(tmp_path / "screened_out.csv")
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted.iloc[0].charge_balance_status == "pass"
    assert accepted.iloc[0].smact_status == "pass"
    assert accepted.iloc[0].li_percolation_dim == 1
    assert accepted.iloc[0].quick_score == pytest.approx(1 / 3)
    assert rejected.iloc[0].charge_balance_status == "fail"
    assert "charge_balance_fail" in rejected.iloc[0].filtered_reasons
    assert (tmp_path / "top150_refs.txt").read_text().endswith("relaxed.extxyz::0")
