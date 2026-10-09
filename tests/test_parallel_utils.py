"""Dependency-free regression tests for isolated screening subprocesses."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "workflow" / "pipeline" / "parallel_utils.py"
spec = importlib.util.spec_from_file_location("parallel_utils_under_test", SCRIPT)
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_worker_count_must_be_positive_integer(workers):
    with pytest.raises(ValueError):
        parallel.gpu_devices(workers, {})


def test_gpu_device_allocation_preserves_scheduler_tokens():
    assert parallel.gpu_devices(3, {}) == ["0", "1", "2"]
    assert parallel.gpu_devices(2, {"CUDA_VISIBLE_DEVICES": "3, 7,9"}) == ["3", "7"]
    assert parallel.gpu_devices(2, {"CUDA_VISIBLE_DEVICES": "GPU-aa,GPU-bb"}) == ["GPU-aa", "GPU-bb"]


@pytest.mark.parametrize("value", ["", "-1", "0,", "0,0", "0", "0,-1", "0,,1"])
def test_invalid_or_insufficient_gpu_allocation_fails(value):
    with pytest.raises(ValueError):
        parallel.gpu_devices(2, {"CUDA_VISIBLE_DEVICES": value})


@pytest.fixture
def worker_script(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text('''import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--worker")
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()
payload = json.loads(Path(args.input).read_text())
tasks = payload["tasks"]
context = payload["context"]
if context.get("barrier"):
    markers = Path(context["barrier"])
    (markers / tasks[0]["id"]).touch()
    deadline = time.monotonic() + 5
    while len(list(markers.iterdir())) != context["worker_count"]:
        if time.monotonic() > deadline:
            print("workers did not start concurrently", flush=True)
            sys.exit(7)
        time.sleep(0.01)
behavior = context.get("behavior", "normal")
if behavior == "fail":
    print("synthetic failure", flush=True)
    sys.exit(9)
if behavior == "no_output":
    sys.exit(0)
records = [{"id": task["id"], "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": os.environ.get("OMP_NUM_THREADS"),
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "cwd": os.getcwd(), "mode": args.worker} for task in reversed(tasks)]
if behavior == "missing":
    records = records[:-1]
elif behavior == "duplicate":
    records.append(records[0])
elif behavior == "extra":
    records.append({"id": "unrequested"})
elif behavior == "invalid_id":
    records[0]["id"] = 123
elif behavior == "nonlist":
    records = {"records": records}
Path(args.output).write_text(json.dumps(records))
print("worker complete", flush=True)
''', encoding="utf-8")
    return script


def invoke(tmp_path, worker_script, tasks=None, workers=2, **kwargs):
    return parallel.run_shards(
        tasks=[{"id": str(index)} for index in range(6)] if tasks is None else tasks,
        workers=workers, directory=tmp_path / "shards", mode="hull",
        worker_script=worker_script, cwd=tmp_path, **kwargs,
    )


def test_gpu_workers_launch_concurrently_and_keep_input_order(tmp_path, worker_script, monkeypatch):
    barrier = tmp_path / "markers"
    barrier.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-device")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    records = invoke(tmp_path, worker_script, workers=4,
                     devices=["GPU-aa", "GPU-bb", "GPU-cc", "GPU-dd"],
                     context={"barrier": str(barrier), "worker_count": 4})
    assert [record["id"] for record in records] == [str(index) for index in range(6)]
    assert [record["device"] for record in records] == ["GPU-aa", "GPU-bb", "GPU-cc", "GPU-dd", "GPU-aa", "GPU-bb"]
    assert all(record["threads"] == record["blas_threads"] == "1" for record in records)
    assert all(record["mode"] == "hull" and record["cwd"] == str(tmp_path) for record in records)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-device"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "16"
    assert len(list((tmp_path / "shards").glob("*.log"))) == 4
    assert all(path.read_text() == "worker complete\n" for path in (tmp_path / "shards").glob("*.log"))


def test_cpu_workers_have_no_visible_gpu(tmp_path, worker_script, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    records = invoke(tmp_path, worker_script)
    assert all(record["device"] == "" for record in records)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_only_nonempty_shards_are_launched(tmp_path, worker_script):
    records = invoke(tmp_path, worker_script, tasks=[{"id": "only"}], workers=4)
    assert [record["id"] for record in records] == ["only"]
    assert len(list((tmp_path / "shards").glob("*.input.json"))) == 1


def test_empty_tasks_do_not_launch(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("empty tasks launched a subprocess")
    monkeypatch.setattr(parallel.subprocess, "run", unexpected)
    assert invoke(tmp_path, tmp_path / "unused.py", tasks=[]) == []


@pytest.mark.parametrize("tasks", [[{"id": "same"}, {"id": "same"}], [{"id": 1}], [{}], [None]])
def test_invalid_tasks_are_rejected_before_launch(tmp_path, worker_script, tasks):
    with pytest.raises(ValueError):
        invoke(tmp_path, worker_script, tasks=tasks)
    assert not (tmp_path / "shards").exists()


@pytest.mark.parametrize("behavior", ["missing", "duplicate", "extra", "invalid_id", "nonlist"])
def test_incomplete_or_invalid_result_coverage_fails(tmp_path, worker_script, behavior):
    with pytest.raises(ValueError):
        invoke(tmp_path, worker_script, context={"behavior": behavior})


def test_subprocess_failure_propagates_and_retains_log(tmp_path, worker_script):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        invoke(tmp_path, worker_script, context={"behavior": "fail"})
    assert failure.value.returncode == 9
    assert "synthetic failure" in (tmp_path / "shards" / "worker_000.log").read_text()


def test_stale_output_cannot_hide_child_missing_output(tmp_path, worker_script):
    directory = tmp_path / "shards"
    directory.mkdir()
    output = directory / "worker_000.output.json"
    output.write_text(json.dumps([{"id": "old"}]))
    with pytest.raises(FileNotFoundError):
        invoke(tmp_path, worker_script, tasks=[{"id": "old"}], workers=1,
               context={"behavior": "no_output"})
    assert not output.exists()


def test_successful_rerun_replaces_stale_output(tmp_path, worker_script):
    invoke(tmp_path, worker_script)
    records = invoke(tmp_path, worker_script, tasks=[{"id": "new"}], workers=1)
    assert [record["id"] for record in records] == ["new"]
    fresh = json.loads((tmp_path / "shards" / "worker_000.output.json").read_text())
    assert [record["id"] for record in fresh] == ["new"]
