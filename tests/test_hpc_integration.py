"""Exercise the HPC entry with real screening/export and stubbed GPU commands."""
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pymatgen.core import Structure


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "mattergen_webapp" / "scripts"

STUB = '''import json
import os
import sys
from pathlib import Path

mode, *args = sys.argv[1:]
if mode == "python":
    with Path(os.environ["COMMAND_LOG"]).open("a") as fh:
        fh.write(json.dumps(args) + "\\n")
    python = os.environ["REAL_PYTHON"]
    os.execv(python, [python, *args])
elif mode == "generate":
    destination = Path(args[0])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "generated.zip").write_text("GPU generation stub")
elif mode == "evaluate":
    from ase.io import write
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    flags = dict(arg[2:].split("=", 1) for arg in args if arg.startswith("--"))
    species = ["Li", "Nb", "O", "Cl", "Cl", "Cl", "Cl"]
    coordinates = [[0, 0, 0], [.2, .15, .1], [.3, .3, .3], [.4, .4, .4],
                   [.5, .55, .5], [.7, .7, .7], [.8, .9, .8]]
    lattice = Lattice.orthorhombic(2.5, 12, 12)
    valid = Structure(lattice, species, coordinates)
    duplicate = valid.copy()
    duplicate.translate_sites(list(range(len(duplicate))), [.13, .27, .31],
                              frac_coords=True, to_unit_cell=True)
    invalid_chemistry = Structure(lattice, ["Li"] * 3 + species,
        [[.8, .1, .8], [.6, .1, .8], [.4, .1, .8], *coordinates])
    without_oxygen = Structure(lattice, [s for s in species if s != "O"],
                              [c for s, c in zip(species, coordinates) if s != "O"])
    frames = [valid, duplicate, invalid_chemistry, without_oxygen]
    write(flags["structures_output_path"], [AseAtomsAdaptor.get_atoms(s) for s in frames])
    Path(flags["save_as"]).write_text("{}")
else:
    raise SystemExit("unexpected fake-command mode")
'''


def read_rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def invoke_hpc(tmp_path, overrides):
    # An isolated copy exercises script paths containing spaces and keeps the
    # legacy export-index copy away from the user's checkout.
    webapp = tmp_path / "web app"
    scripts = webapp / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_full_pipeline_hpc.sh", "dd.sh", "eval_all.sh", "screen_all_extxyz.py",
                 "run_top300_pipeline.py", "candidate_selection.py", "export_refs_to_structs.py"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    (webapp / "results").mkdir()
    mattergen = tmp_path / "mattergen work dir"
    mattergen.mkdir()
    results = tmp_path / "pipeline results"
    bin_dir = tmp_path / "fake commands"
    bin_dir.mkdir()
    stub = tmp_path / "fake gpu commands.py"
    stub.write_text(STUB)
    for command, mode in (("python", "python"), ("mattergen-generate", "generate"),
                          ("mattergen-evaluate", "evaluate")):
        entry = bin_dir / command
        entry.write_text('#!/bin/bash\nexec "$REAL_PYTHON" "$GPU_STUB" ' + mode + ' "$@"\n')
        entry.chmod(0o755)
    timeout = bin_dir / "timeout"
    timeout.write_text('#!/bin/bash\nshift\nexec "$@"\n')
    timeout.chmod(0o755)
    command_log = tmp_path / "python commands.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "GPU_STUB": str(stub),
        "COMMAND_LOG": str(command_log),
        "WEBAPP_ROOT": str(webapp),
        "MATTERGEN_ROOT": str(mattergen),
        "RESULTS_ROOT": str(results),
        "RUNTIME_ROOT": str(tmp_path / "runtime cache"),
        "CHEMICAL_SYSTEMS": "Li-Nb-O-Cl",
        "CHEMICAL_SYSTEMS_FILE": "",
        "ELEMENTS": "Li Nb O Cl",
        "COMBO_SIZES": "4",
        "SEGMENTS": "1",
        "NUM_BATCHES_PER_SEGMENT": "1",
        "BATCH_SIZE": "1",
        "SCREEN_TOPK": "5",
        "TOPK": "5",
        "REQUIRED_ELEMENTS": "Li Nb O Cl",
        "ALLOWED_ELEMENTS": "Li Nb O Cl",
        "REQUIRE_CHARGE_BALANCE": "1",
        "USE_SMACT": "1",
        "FILTER_LIGHT_OXY": "1",
        "SELECTION_MODE": "diverse",
        "VOLTAGE_THRESHOLD": "0.001",
        "TARGET_VOLTAGE": "",
        "MIN_VOLTAGE_WINDOW": "0",
        "DRY_RUN": "1",
        **overrides,
    }
    # Remove the policy variables for the default case so this checks shell
    # defaults, including environments in which these names were already set.
    if not overrides:
        for key in ("REQUIRED_ELEMENTS", "ALLOWED_ELEMENTS", "REQUIRE_CHARGE_BALANCE",
                    "USE_SMACT", "FILTER_LIGHT_OXY", "SELECTION_MODE", "VOLTAGE_THRESHOLD",
                    "TARGET_VOLTAGE", "MIN_VOLTAGE_WINDOW"):
            env.pop(key)
    process = subprocess.run(["bash", str(scripts / "run_full_pipeline_hpc.sh")],
                             env=env, text=True, capture_output=True, timeout=90)
    assert process.returncode == 0, process.stdout + process.stderr
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    screen_command = next(cmd for cmd in commands if cmd[0].endswith("/screen_all_extxyz.py"))
    top_command = next(cmd for cmd in commands if cmd[0].endswith("/run_top300_pipeline.py"))
    return results, screen_command, top_command, process.stdout


def value(command, flag):
    return command[command.index(flag) + 1]


def rejection_audit(command):
    if "--screened-out" in command:
        return Path(value(command, "--screened-out"))
    return Path(value(command, "--workdir")) / "screened_out.csv"


def assert_export_matches_selection(results, expected_unique):
    output = results / "top300_run"
    refs = (output / "top300_refs.txt").read_text().splitlines()
    index = read_rows(output / "exported_300cifs" / "export_300index.csv")
    audit = read_rows(output / "selection_audit.csv")
    assert len(refs) == len(index) == expected_unique
    assert {row["ref"] for row in index} == set(refs)
    assert sum(row["selection_status"] == "duplicate" for row in audit) == 1
    assert sum(row["selection_status"] == "selected" for row in audit) == expected_unique
    for row in index:
        assert Path(row["cif"]).is_file()
        assert len(Structure.from_file(row["cif"])) == int(row["n_atoms"])
    assert not (output / "chgnet_hull_top300.csv").exists()


def test_hpc_defaults_filter_chemistry_and_deduplicate_before_export(tmp_path):
    results, screen_command, top_command, stdout = invoke_hpc(tmp_path, {})
    assert "--require-charge-balance" in screen_command
    assert "--use-smact" in screen_command
    assert "--no-light-oxy" not in screen_command
    assert value(top_command, "--selection-mode") == "diverse"
    assert value(top_command, "--voltage-threshold") == "0.001"
    assert "--target-voltage" not in top_command
    accepted = read_rows(results / "stage2_candidates.csv")
    rejected = read_rows(rejection_audit(screen_command))
    assert len(accepted) == 2
    assert len(rejected) == 2
    assert all(row["charge_balance_status"] == row["smact_status"] == "pass" for row in accepted)
    assert all(float(row["quick_score"]) == pytest.approx(1 / 3) for row in accepted)
    assert any("charge_balance_fail" in row["filtered_reasons"] for row in rejected)
    assert any("missing_required_elements:O" in row["filtered_reasons"] for row in rejected)
    assert "Candidate selection (diverse)" in stdout
    assert_export_matches_selection(results, expected_unique=1)


def test_hpc_explicit_policy_flags_and_target_reach_real_cli(tmp_path):
    results, screen_command, top_command, stdout = invoke_hpc(tmp_path, {
        "REQUIRE_CHARGE_BALANCE": "0", "USE_SMACT": "false", "FILTER_LIGHT_OXY": "0",
        "REQUIRED_ELEMENTS": "Li Nb Cl", "ALLOWED_ELEMENTS": "",
        "SELECTION_MODE": "score", "VOLTAGE_THRESHOLD": "0.002",
        "TARGET_VOLTAGE": "4.5", "MIN_VOLTAGE_WINDOW": "0.8",
    })
    assert {"--no-charge-balance", "--no-smact", "--no-light-oxy"}.issubset(screen_command)
    assert "--allowed-elements" not in screen_command
    assert screen_command[screen_command.index("--required-elements") + 1:screen_command.index("--topk")] == ["Li", "Nb", "Cl"]
    assert value(top_command, "--selection-mode") == "score"
    assert value(top_command, "--voltage-threshold") == "0.002"
    assert value(top_command, "--target-voltage") == "4.5"
    assert value(top_command, "--min-voltage-window") == "0.8"
    assert len(read_rows(results / "stage2_candidates.csv")) == 4
    assert read_rows(rejection_audit(screen_command)) == []
    assert "Candidate selection (score)" in stdout
    assert_export_matches_selection(results, expected_unique=3)
