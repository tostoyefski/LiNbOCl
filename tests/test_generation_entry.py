import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "mattergen_webapp" / "scripts" / "dd.sh"


def invoke(tmp_path, systems):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "arguments.txt"
    stub = bin_dir / "mattergen-generate"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$CAPTURE"\n')
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
           "WORKDIR": str(tmp_path), "BASE_RESULTS_DIR": str(tmp_path / "outputs"),
           "CHEMICAL_SYSTEMS": systems, "CHEMICAL_SYSTEMS_FILE": "",
           "ELEMENTS": "Li Nb O Cl", "COMBO_SIZES": "4", "CAPTURE": str(capture)}
    result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return capture.read_text()


def test_default_auto_combination_is_captured(tmp_path):
    output = invoke(tmp_path, "")
    assert '"chemical_system": "Li-Nb-O-Cl"' in output


def test_explicit_whitespace_list_and_duplicates(tmp_path):
    output = invoke(tmp_path, "Li-Nb-O-Cl Li-Ta-O-Cl,Li-Nb-O-Cl")
    assert output.count('"chemical_system": "Li-Nb-O-Cl"') == 1
    assert output.count('"chemical_system": "Li-Ta-O-Cl"') == 1
