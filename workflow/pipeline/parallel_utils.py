"""Isolated subprocess shards for CPU and one-GPU screening workers."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _validate_workers(workers: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")


def _validate_devices(devices: list[str], workers: int) -> list[str]:
    tokens = [str(device).strip() for device in devices]
    if any(not token or token == "-1" or "," in token for token in tokens):
        raise ValueError("GPU devices must contain nonempty visible device tokens")
    if len(set(tokens)) != len(tokens):
        raise ValueError("GPU devices must be unique")
    if len(tokens) < workers:
        raise ValueError(f"Requested {workers} GPU workers but only {len(tokens)} devices are visible")
    return tokens[:workers]


def gpu_devices(workers: int, environ=None) -> list[str]:
    """Return CUDA tokens without changing their numbering or UUID identities."""
    _validate_workers(workers)
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(index) for index in range(workers)]
    return _validate_devices(visible.split(","), workers)


def _write_json_atomic(path: Path, value) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _task_ids(records: list[dict], label: str) -> list[str]:
    ids = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"{label} must contain objects with string ids")
        ids.append(record["id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate ids")
    return ids


def run_shards(
    tasks: list[dict],
    workers: int,
    directory: Path,
    mode: str,
    worker_script: Path,
    cwd: Path,
    devices: list[str] | None = None,
    context: dict | None = None,
) -> list[dict]:
    """Run round-robin shards and return exactly one record per input task.

    Each child sees either one allocated CUDA token or no CUDA devices. All
    inputs, outputs, and combined stdout/stderr logs are retained in directory.
    The caller must supply a distinct directory for concurrently running stages.
    """
    _validate_workers(workers)
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")
    ordered_ids = _task_ids(tasks, "tasks")
    selected_devices = None if devices is None else _validate_devices(devices, workers)
    if not tasks:
        return []

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    worker_script = Path(worker_script).resolve()
    cwd = Path(cwd).resolve()
    shards = [tasks[index::workers] for index in range(min(workers, len(tasks)))]
    jobs = []
    for index, shard in enumerate(shards):
        stem = directory / f"worker_{index:03d}"
        input_path = stem.with_suffix(".input.json")
        output_path = stem.with_suffix(".output.json")
        log_path = stem.with_suffix(".log")
        # A successful child must create a fresh result on this invocation.
        output_path.unlink(missing_ok=True)
        _write_json_atomic(input_path, {"tasks": shard, "context": context or {}})
        jobs.append((index, shard, input_path, output_path, log_path))

    def run_worker(job):
        index, shard, input_path, output_path, log_path = job
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "" if selected_devices is None else selected_devices[index]
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
        ):
            environment[name] = "1"
        command = [
            sys.executable, str(worker_script), "--worker", mode,
            "--input", str(input_path), "--output", str(output_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        with output_path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Worker {index} output must be a list")
        result_ids = _task_ids(records, f"Worker {index} output")
        expected_ids = {task["id"] for task in shard}
        if set(result_ids) != expected_ids:
            missing = sorted(expected_ids - set(result_ids))
            extra = sorted(set(result_ids) - expected_ids)
            raise ValueError(f"Worker {index} coverage mismatch: missing={missing}, extra={extra}")
        return records

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        batches = list(executor.map(run_worker, jobs))
    records = [record for batch in batches for record in batch]
    result_ids = _task_ids(records, "Combined worker output")
    if set(result_ids) != set(ordered_ids):
        raise ValueError("Combined worker output does not exactly cover input tasks")
    by_id = {record["id"]: record for record in records}
    return [by_id[task_id] for task_id in ordered_ids]
