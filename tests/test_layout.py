"""Verify relocated entry points without loading a model or building an image."""

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "workflow" / "pipeline"
CONTAINER = REPO / "mattergen_webapp" / "container"


@pytest.mark.parametrize("command", [
    "mattergen-generate", "mattergen-train", "mattergen-finetune",
    "mattergen-evaluate", "csv-to-dataset",
])
def test_official_mattergen_cli_targets_still_exist(command):
    configuration = (REPO / "mattergen" / "pyproject.toml").read_text(encoding="utf-8")
    scripts = configuration.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    target = re.search(rf'^{re.escape(command)}\s*=\s*"([^\"]+)"', scripts, re.MULTILINE)
    assert target is not None
    module_name, function_name = target.group(1).split(":")
    source = REPO / "mattergen" / Path(*module_name.split(".")).with_suffix(".py")
    assert source.is_file()
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert function_name in functions


def test_backend_launches_relocated_pipeline_entries(monkeypatch):
    monkeypatch.delenv("WORKFLOW_ROOT", raising=False)
    monkeypatch.delenv("MATTERGEN_DEFAULT_RESULTS_DIR", raising=False)
    monkeypatch.delenv("RESULTS_ROOT", raising=False)
    spec = importlib.util.spec_from_file_location("layout_backend", REPO / "mattergen_webapp" / "backend" / "main.py")
    backend = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, backend)
    spec.loader.exec_module(backend)
    assert backend.WORKFLOW_ROOT == REPO / "workflow"
    assert backend.SCRIPTS_DIR == PIPELINE
    assert backend.DEFAULT_RESULTS_ROOT == REPO / "results"
    assert Path(backend.GenerateRequest().base_results_dir) == REPO / "results"
    assert Path(backend.EvalRequest().root) == REPO / "results"
    assert Path(backend.ScreenRequest().out) == REPO / "results" / "stage2_candidates.csv"
    calls = []
    monkeypatch.setattr(backend, "launch_job", lambda kind, command, **kwargs: calls.append(command))
    for run, payload, name in (
        (backend.run_dd, backend.GenerateRequest(), "generate.sh"),
        (backend.run_eval, backend.EvalRequest(), "evaluate.sh"),
        (backend.run_screen, backend.ScreenRequest(), "screen_all_extxyz.py"),
        (backend.run_top300, backend.Top300Request(), "run_top300_pipeline.py"),
    ):
        run(payload)
        assert Path(calls[-1][1]) == PIPELINE / name
        assert Path(calls[-1][1]).is_file()
    top_command = calls[-1]
    output_dir = REPO / "results" / "top300_run"
    for flag, expected in (
        ("--output-dir", output_dir),
        ("--refs-out", output_dir / "top300_refs.txt"),
        ("--export-dir", output_dir / "exported_300cifs"),
    ):
        assert Path(top_command[top_command.index(flag) + 1]) == expected


def test_exporter_cli_defaults_use_current_top300_run(monkeypatch):
    spec = importlib.util.spec_from_file_location("layout_exporter", PIPELINE / "export_refs_to_structs.py")
    exporter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, exporter)
    spec.loader.exec_module(exporter)
    parse_args = argparse.ArgumentParser.parse_args

    class ParsedDefaults(Exception):
        pass

    def capture_defaults(parser, *args, **kwargs):
        raise ParsedDefaults(parse_args(parser, []))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_defaults)
    with pytest.raises(ParsedDefaults) as exc:
        exporter.main()
    args = exc.value.args[0]
    output = REPO / "results" / "top300_run"
    assert Path(args.refs) == output / "top300_refs.txt"
    assert Path(args.outdir) == output / "exported_300cifs"


@pytest.mark.parametrize("script", [
    PIPELINE / "generate.sh", PIPELINE / "evaluate.sh",
    PIPELINE / "run_full_pipeline.sh", CONTAINER / "build_image.sh",
])
def test_relocated_shell_entries_parse(script):
    result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_container_paths_include_workflow():
    dockerfile = (CONTAINER / "Dockerfile").read_text(encoding="utf-8")
    compose = (CONTAINER / "docker-compose.yml").read_text(encoding="utf-8")
    assert "WORKFLOW_ROOT=/workspace/workflow" in dockerfile
    assert "COPY workflow /workspace/workflow" in dockerfile
    assert "WORKFLOW_ROOT: /workspace/workflow" in compose
    assert "/workspace/workflow/pipeline/run_full_pipeline.sh" in compose
    assert "dockerfile: mattergen_webapp/container/Dockerfile" in compose
    context = re.search(r"^\s+context:\s*(\S+)\s*$", compose, re.MULTILINE)
    assert context is not None
    assert (CONTAINER / context.group(1)).resolve() == REPO / ".mattergen_docker_context"


def test_build_image_stages_workflow_in_docker_context(tmp_path):
    project = tmp_path / "project with spaces"
    container = project / "mattergen_webapp" / "container"
    container.mkdir(parents=True)
    shutil.copyfile(CONTAINER / "build_image.sh", container / "build_image.sh")
    shutil.copyfile(CONTAINER / "Dockerfile", container / "Dockerfile")
    workflow_entry = project / "workflow" / "pipeline" / "run_full_pipeline.sh"
    workflow_entry.parent.mkdir(parents=True)
    workflow_entry.write_text("#!/bin/bash\n", encoding="utf-8")
    mattergen_entry = project / "mattergen" / "mattergen" / "scripts" / "generate.py"
    mattergen_entry.parent.mkdir(parents=True)
    mattergen_entry.write_text("# official CLI placeholder\n", encoding="utf-8")

    bin_dir = tmp_path / "fake commands"
    bin_dir.mkdir()
    stub = tmp_path / "fake build.py"
    stub.write_text('''import json
import os
import shutil
import sys
from pathlib import Path

mode, *args = sys.argv[1:]
with Path(os.environ["BUILD_LOG"]).open("a") as handle:
    handle.write(json.dumps([mode, *args]) + "\\n")
if mode == "rsync":
    positional = []
    iterator = iter(args)
    for arg in iterator:
        if arg == "--exclude":
            next(iterator)
        elif not arg.startswith("-"):
            positional.append(Path(arg))
    destination = positional[-1]
    for source in positional[:-1]:
        shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
elif mode == "docker":
    assert args[0] == "build"
    context = Path(args[-1])
    assert (context / "workflow" / "pipeline" / "run_full_pipeline.sh").is_file()
    assert (context / "mattergen" / "mattergen" / "scripts" / "generate.py").is_file()
    assert Path(args[args.index("-f") + 1]).is_file()
else:
    raise SystemExit("Unexpected build command")
''', encoding="utf-8")
    for command in ("rsync", "docker"):
        entry = bin_dir / command
        entry.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$BUILD_STUB" ' + command + ' "$@"\n', encoding="utf-8")
        entry.chmod(0o755)
    context = tmp_path / "build context"
    log = tmp_path / "build commands.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_PYTHON": sys.executable,
        "BUILD_STUB": str(stub),
        "BUILD_LOG": str(log),
        "CONTEXT_DIR": str(context),
        "IMAGE_NAME": "offline-layout-test",
    }
    result = subprocess.run(["bash", str(container / "build_image.sh")], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[0] for call in calls] == ["rsync", "docker"]
    assert str(project / "workflow") in calls[0]
    assert calls[1][-1] == str(context)
    assert (context / "workflow" / "pipeline" / "run_full_pipeline.sh").read_text() == workflow_entry.read_text()
