#!/usr/bin/env python3
"""Legacy CLI entry point for the shared, audited screening pipeline."""
from pathlib import Path
import importlib.util
import sys

_scripts = Path(__file__).resolve().parents[1] / "mattergen_webapp" / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("_linbocl_top300_pipeline", _scripts / "run_top300_pipeline.py")
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)
for _name in ("read_stage2_rows", "select_top_refs", "write_refs", "run_command", "run_export", "rename_index", "run_ehull", "filter_hull", "run_voltage", "filter_voltage"):
    globals()[_name] = getattr(_core, _name)


def parse_args(argv=None):
    return _core.parse_args(argv, legacy_defaults=True)


def main(argv=None):
    return _core.main(argv, legacy_defaults=True)


if __name__ == "__main__":
    main()
