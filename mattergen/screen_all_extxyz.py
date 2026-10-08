#!/usr/bin/env python3
"""Compatibility entry point for the canonical periodic/chemical screening CLI."""
from pathlib import Path
import runpy

_CANONICAL = Path(__file__).resolve().parents[1] / "mattergen_webapp" / "scripts" / "screen_all_extxyz.py"

if __name__ == "__main__":
    runpy.run_path(str(_CANONICAL), run_name="__main__")
else:
    globals().update(runpy.run_path(str(_CANONICAL)))
