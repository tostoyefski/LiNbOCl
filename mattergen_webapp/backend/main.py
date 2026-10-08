#!/usr/bin/env python3
"""FastAPI wrapper around MatterGen batch scripts.

Endpoints:
- /api/run/dd           -> run dd.sh with parameter overrides
- /api/run/eval         -> run eval_all.sh
- /api/run/screen       -> run screen_all_extxyz.py
- /api/run/top300       -> run run_top300_pipeline.py
- /api/jobs             -> list jobs
- /api/jobs/{job_id}    -> fetch a job
- /api/jobs/{job_id}/log-> tail log output
"""

from __future__ import annotations

import os
import math
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal
import base64
import io

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator


# ---- basic config ----

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
DEFAULT_MATTERGEN_ROOT = REPO_ROOT / "mattergen"
MATTERGEN_ROOT = Path(os.environ.get("MATTERGEN_ROOT", DEFAULT_MATTERGEN_ROOT)).resolve()
SCRIPTS_DIR = REPO_ROOT / "mattergen_webapp" / "scripts"
LOG_DIR = ROOT / "job_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("MATTERGEN_RUNTIME_ROOT", "/mnt/e/mattergen_runs/_runtime")).expanduser()
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("MATTERGEN_DEFAULT_RESULTS_DIR", os.environ.get("RESULTS_ROOT", "/mnt/e/mattergen_runs/results200"))
).expanduser()


class JobStatus(str):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    kind: str
    command: List[str]
    workdir: str
    status: str = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    log_path: str
    params: Dict = Field(default_factory=dict)
    pid: Optional[int] = None
    pgid: Optional[int] = None
    elapsed_seconds: Optional[float] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.copy() if job else None

    def all(self) -> List[Job]:
        with self._lock:
            return [job.copy() for job in self._jobs.values()]


jobs = JobStore()
job_procs: Dict[str, subprocess.Popen] = {}


def ensure_script(name: str) -> Path:
    path = SCRIPTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"script not found: {path}")
    return path


def apply_job_runtime_env(full_env: Dict[str, str]) -> None:
    """Route large temporary/cache writes for spawned jobs to the runtime disk."""
    runtime_root = Path(full_env.get("MATTERGEN_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT))).expanduser()
    paths = {
        "TMPDIR": runtime_root / "tmp",
        "TMP": runtime_root / "tmp",
        "TEMP": runtime_root / "tmp",
        "XDG_CACHE_HOME": runtime_root / "cache",
        "HF_HOME": runtime_root / "huggingface",
        "HF_HUB_CACHE": runtime_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": runtime_root / "huggingface" / "transformers",
        "TORCH_HOME": runtime_root / "torch",
        "PYTORCH_KERNEL_CACHE_PATH": runtime_root / "torch" / "kernels",
        "CUDA_CACHE_PATH": runtime_root / "cuda",
        "MPLCONFIGDIR": runtime_root / "matplotlib",
        "UV_CACHE_DIR": runtime_root / "uv",
        "PIP_CACHE_DIR": runtime_root / "pip",
        "NUMBA_CACHE_DIR": runtime_root / "numba",
        "TRITON_CACHE_DIR": runtime_root / "triton",
        "PYTHONPYCACHEPREFIX": runtime_root / "pycache",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        full_env[key] = str(path)
    full_env["HF_HUB_OFFLINE"] = "1"
    full_env["HF_DATASETS_OFFLINE"] = "1"
    full_env["MATTERGEN_RUNTIME_ROOT"] = str(runtime_root)


def launch_job(kind: str, command: List[str], env: Optional[dict] = None, params: Optional[Dict] = None) -> Job:
    job_id = uuid.uuid4().hex[:10]
    log_path = LOG_DIR / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{kind}_{job_id}.log"
    log_path.touch()
    job = Job(
        id=job_id,
        kind=kind,
        command=command,
        workdir=str(MATTERGEN_ROOT),
        log_path=str(log_path),
        params=params or {},
    )
    jobs.add(job)

    def _runner() -> None:
        jobs.update(job_id, status=JobStatus.RUNNING, started_at=datetime.utcnow())
        full_env = os.environ.copy()
        venv_bin = MATTERGEN_ROOT / ".venv" / "bin"
        if venv_bin.is_dir():
            full_env["PATH"] = f"{venv_bin}:{full_env.get('PATH', '')}"
            full_env.setdefault("VIRTUAL_ENV", str(MATTERGEN_ROOT / ".venv"))
        if env:
            full_env.update({k: str(v) for k, v in env.items() if v is not None})
        apply_job_runtime_env(full_env)
        try:
            with open(log_path, "w") as log_f:
                proc = subprocess.Popen(
                    command,
                    cwd=str(MATTERGEN_ROOT),
                    env=full_env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # new process group for clean cancel
                )
                job_procs[job_id] = proc
                try:
                    pgid = os.getpgid(proc.pid)
                except Exception:
                    pgid = None
                jobs.update(job_id, pid=proc.pid, pgid=pgid)
                ret = proc.wait()
            status = JobStatus.SUCCEEDED if ret == 0 else JobStatus.FAILED
            jobs.update(
                job_id,
                status=status,
                finished_at=datetime.utcnow(),
                exit_code=ret,
            )
            if ret != 0:
                jobs.update(job_id, error=f"exit code {ret}")
        except Exception as exc:  # noqa: BLE001
            jobs.update(
                job_id,
                status=JobStatus.FAILED,
                finished_at=datetime.utcnow(),
                exit_code=-1,
                error=str(exc),
            )
        finally:
            job_procs.pop(job_id, None)

    threading.Thread(target=_runner, daemon=True).start()
    return job


def tail_log(log_path: Path, lines: int = 2000) -> str:
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="log not found")
    dq = deque(maxlen=lines)
    with open(log_path, "r") as fh:
        for line in fh:
            dq.append(line.rstrip("\n"))
    return "\n".join(dq)


# ---- request schemas ----


class GenerateRequest(BaseModel):
    model_name: str = Field("chemical_system_energy_above_hull", description="MODEL_NAME env for dd.sh")
    base_results_dir: str = Field("results/chemical_system_energy_above_hull", description="BASE_RESULTS_DIR env")
    batch_size: int = 16
    num_batches: int = Field(1, ge=1, description="mattergen-generate --num_batches")
    e_ah: float = 0.05
    guidance: float = 2.0
    elements: List[str] = Field(default_factory=lambda: ["Li", "Nb", "O", "Cl"], description="元素集合，用于自动组合")
    combo_sizes: List[int] = Field(default_factory=lambda: [4], description="组合大小（可多选）")
    chemical_systems: Optional[List[str]] = Field(default=None, description="显式化学系统列表，逗号或数组")
    chemical_systems_file: Optional[str] = Field(default=None, description="化学系统文件路径，一行一个")


class EvalRequest(BaseModel):
    root: str = Field("results/chemical_system_energy_above_hull", description="ROOT env passed to eval_all.sh")


class ScreenRequest(BaseModel):
    base: str = "results/chemical_system_energy_above_hull"
    out: str = "results/stage2_candidates.csv"
    r_cut: float = Field(3.0, gt=0)
    supercell: List[int] = Field(default_factory=lambda: [2, 2, 2], min_items=3, max_items=3)
    light_oxy: List[float] = Field(default_factory=lambda: [0.05, 0.35], min_items=2, max_items=2)
    filter_light_oxy: bool = True
    required_elements: List[str] = Field(default_factory=lambda: ["Li", "Nb", "O", "Cl"])
    allowed_elements: Optional[List[str]] = Field(default_factory=lambda: ["Li", "Nb", "O", "Cl"])
    require_charge_balance: bool = True
    use_smact: bool = True
    topk: int = Field(150, ge=1)
    refs_out: str = "top150_refs.txt"

    @validator("supercell")
    def _super_len(cls, v: List[int]) -> List[int]:
        if len(v) != 3 or any(x < 1 for x in v):
            raise ValueError("supercell must have 3 positive integers")
        return v

    @validator("light_oxy")
    def _light_len(cls, v: List[float]) -> List[float]:
        if len(v) != 2 or not all(math.isfinite(x) for x in v) or not 0 <= v[0] <= v[1] <= 1:
            raise ValueError("light_oxy expects 0 <= low <= high <= 1")
        return v

    @validator("r_cut")
    def _finite_cutoff(cls, v):
        if not math.isfinite(v):
            raise ValueError("r_cut must be finite")
        return v

    @validator("required_elements")
    def _required_elements(cls, v):
        if not v or "Li" not in v:
            raise ValueError("required_elements must include Li")
        if any(not re.fullmatch(r"[A-Z][a-z]?", el) for el in v):
            raise ValueError("required_elements must contain element symbols")
        return list(dict.fromkeys(v))

    @validator("allowed_elements", always=True)
    def _allowed_elements(cls, v, values):
        if v is None:
            return v
        if not v or any(not re.fullmatch(r"[A-Z][a-z]?", el) for el in v):
            raise ValueError("allowed_elements must contain element symbols or be null")
        if not set(values.get("required_elements", [])).issubset(v):
            raise ValueError("allowed_elements must contain every required element")
        return list(dict.fromkeys(v))


class Top300Request(BaseModel):
    stage2_csv: str = "results/stage2_candidates.csv"
    topk: int = Field(300, ge=1)
    selection_mode: Literal["diverse", "score"] = "diverse"
    refs_out: str = "top300_refs.txt"
    output_dir: Optional[str] = None
    export_dir: str = "results/exported_300cifs"
    export_prefix: str = "cand300"
    export_index_name: str = "export_300index.csv"
    ehull_threshold: float = Field(0.05, ge=0)
    ehull_out: Optional[str] = None
    filtered_out: Optional[str] = None
    voltage_out: Optional[str] = None
    voltage_step: Optional[float] = Field(None, gt=0)
    voltage_threshold: float = Field(1e-3, ge=0)
    target_voltage: Optional[float] = None
    min_voltage_window: float = Field(0.0, ge=0)
    dry_run: bool = False

    @validator("ehull_threshold", "voltage_step", "voltage_threshold", "target_voltage", "min_voltage_window")
    def _finite_values(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("screening thresholds and voltages must be finite")
        return v


class FullPipelineRequest(BaseModel):
    dd: GenerateRequest = GenerateRequest()
    eval: EvalRequest = EvalRequest()
    screen: ScreenRequest = ScreenRequest()
    top300: Top300Request = Top300Request()
    num_batches: int = Field(1, ge=1, description="连续运行全流程的批次数（>=1）")


# ---- app setup ----

app = FastAPI(title="MatterGen Web Runner", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mattergen_root": str(MATTERGEN_ROOT),
        "mattergen_root_exists": MATTERGEN_ROOT.exists(),
        "scripts_dir": str(SCRIPTS_DIR),
    }


@app.get("/api/defaults")
def defaults():
    base = str(DEFAULT_RESULTS_ROOT)
    top_dir = str(DEFAULT_RESULTS_ROOT / "top300_run")
    return {
        "base_results_dir": base,
        "eval_root": base,
        "screen_base": base,
        "screen_out": str(DEFAULT_RESULTS_ROOT / "stage2_candidates.csv"),
        "top300_stage2_csv": str(DEFAULT_RESULTS_ROOT / "stage2_candidates.csv"),
        "top300_output_dir": top_dir,
        "top300_export_dir": str(DEFAULT_RESULTS_ROOT / "top300_run" / "exported_300cifs"),
        "voltage_path": str(DEFAULT_RESULTS_ROOT / "top300_run" / "chgnet_voltage_window_top300.csv"),
        "viewer_dir": base,
    }


@app.post("/api/run/dd")
def run_dd(payload: GenerateRequest):
    script = ensure_script("dd.sh")
    env = {
        "MODEL_NAME": payload.model_name,
        "BASE_RESULTS_DIR": payload.base_results_dir,
        "BATCH_SIZE": payload.batch_size,
        "NUM_BATCHES": payload.num_batches,
        "E_AH": payload.e_ah,
        "GUIDANCE": payload.guidance,
        "CHEMICAL_SYSTEMS": " ".join(payload.chemical_systems) if payload.chemical_systems else "",
        "CHEMICAL_SYSTEMS_FILE": payload.chemical_systems_file or "",
        "ELEMENTS": " ".join(payload.elements) if payload.elements else "",
        "COMBO_SIZES": " ".join(str(x) for x in payload.combo_sizes) if payload.combo_sizes else "",
        "WORKDIR": MATTERGEN_ROOT,
    }
    job = launch_job("dd", ["bash", str(script)], env=env, params=payload.dict())
    return job


@app.post("/api/run/eval")
def run_eval(payload: EvalRequest):
    script = ensure_script("eval_all.sh")
    env = {"ROOT": payload.root, "WORKDIR": MATTERGEN_ROOT}
    job = launch_job("eval", ["bash", str(script)], env=env, params=payload.dict())
    return job


@app.post("/api/run/screen")
def run_screen(payload: ScreenRequest):
    script = ensure_script("screen_all_extxyz.py")
    cmd = [
        "python",
        str(script),
        "--workdir",
        str(MATTERGEN_ROOT),
        "--base",
        payload.base,
        "--out",
        payload.out,
        "--r-cut",
        str(payload.r_cut),
        "--super",
        *(str(x) for x in payload.supercell),
        "--topk",
        str(payload.topk),
        "--refs-out",
        payload.refs_out,
    ]
    if payload.require_charge_balance:
        cmd.append("--require-charge-balance")
    else:
        cmd.append("--no-charge-balance")
    if payload.use_smact:
        cmd.append("--use-smact")
    else:
        cmd.append("--no-smact")
    if payload.filter_light_oxy:
        cmd.extend(["--light-oxy", *(str(x) for x in payload.light_oxy)])
    else:
        cmd.append("--no-light-oxy")
    cmd.extend(["--screened-out", str(Path(payload.out).with_name("screened_out.csv"))])
    cmd.extend(["--required-elements", *payload.required_elements])
    if payload.allowed_elements:
        cmd.extend(["--allowed-elements", *payload.allowed_elements])
    job = launch_job("screen", cmd, params=payload.dict())
    return job


@app.post("/api/run/top300")
def run_top300(payload: Top300Request):
    script = ensure_script("run_top300_pipeline.py")
    cmd = [
        "python",
        str(script),
        "--workdir",
        str(MATTERGEN_ROOT),
        "--output-dir",
        payload.output_dir or str(SCRIPTS_DIR.parent / "results"),
        "--stage2-csv",
        payload.stage2_csv,
        "--topk",
        str(payload.topk),
        "--selection-mode",
        payload.selection_mode,
        "--refs-out",
        payload.refs_out,
        "--export-dir",
        payload.export_dir,
        "--export-prefix",
        payload.export_prefix,
        "--export-index-name",
        payload.export_index_name,
        "--ehull-threshold",
        str(payload.ehull_threshold),
    ]
    if payload.ehull_out:
        cmd.extend(["--ehull-out", payload.ehull_out])
    if payload.filtered_out:
        cmd.extend(["--filtered-out", payload.filtered_out])
    if payload.voltage_out:
        cmd.extend(["--voltage-out", payload.voltage_out])
    if payload.voltage_step is not None:
        cmd.extend(["--voltage-step", str(payload.voltage_step)])
    cmd += [
        "--voltage-threshold",
        str(payload.voltage_threshold),
        "--min-voltage-window",
        str(payload.min_voltage_window),
    ]
    if payload.target_voltage is not None:
        cmd.extend(["--target-voltage", str(payload.target_voltage)])
    if payload.dry_run:
        cmd.append("--dry-run")
    job = launch_job("top300", cmd, params=payload.dict())
    return job


def _quote(val: str) -> str:
    return "'" + val.replace("'", "'\"'\"'") + "'"


def _bump_results_dir(base: Path, offset: int) -> Path:
    """Increment trailing digits of base directory name by offset."""
    name = base.name
    m = re.match(r"^(.*?)(\d+)$", name)
    if m:
        prefix, num = m.groups()
        return base.with_name(f"{prefix}{int(num) + offset}")
    if offset == 0:
        return base
    return base.with_name(f"{name}{offset + 1}")


def _full_segments_root(base: Path) -> Path:
    return base / "_segments"


def _full_segment_dir(base: Path, offset: int) -> Path:
    return _full_segments_root(base) / f"batch{offset + 1:03d}"


def _remap_results_path(path: Optional[str], base_from: Path, base_to: Path) -> Optional[str]:
    """Replace base_from with base_to inside a path string, or anchor relative paths to base_to."""
    if path is None:
        return None
    base_from_str = str(base_from)
    candidate = str(Path(path).expanduser())
    if base_from_str and base_from_str in candidate:
        return candidate.replace(base_from_str, str(base_to))
    p = Path(candidate)
    if not p.is_absolute():
        return str(base_to / p)
    return candidate


@app.post("/api/run/full")
def run_full(payload: FullPipelineRequest):
    base_results = Path(payload.dd.base_results_dir).expanduser()
    batches = max(1, payload.num_batches)
    segments_root = _full_segments_root(base_results)
    run_bases = [_full_segment_dir(base_results, i) for i in range(batches)]

    script_lines = ["set -euo pipefail", f"cd {_quote(str(MATTERGEN_ROOT))}"]
    for i, run_base in enumerate(run_bases):
        batch_no = i + 1
        dd_env = {
            "MODEL_NAME": payload.dd.model_name,
            "BASE_RESULTS_DIR": str(run_base),
            "BATCH_SIZE": payload.dd.batch_size,
            "NUM_BATCHES": payload.dd.num_batches,
            "E_AH": payload.dd.e_ah,
            "GUIDANCE": payload.dd.guidance,
            "CHEMICAL_SYSTEMS": " ".join(payload.dd.chemical_systems) if payload.dd.chemical_systems else "",
            "CHEMICAL_SYSTEMS_FILE": payload.dd.chemical_systems_file or "",
            "ELEMENTS": " ".join(payload.dd.elements) if payload.dd.elements else "",
            "COMBO_SIZES": " ".join(str(x) for x in payload.dd.combo_sizes) if payload.dd.combo_sizes else "",
            "WORKDIR": MATTERGEN_ROOT,
        }
        env_prefix = " ".join([f"{k}={_quote(str(v))}" for k, v in dd_env.items()])

        script_lines += [
            f"echo {_quote(f'Generate segment {batch_no}/{batches}: dd.sh -> {run_base}')}",
            f"{env_prefix} bash {_quote(str(ensure_script('dd.sh')))}",
        ]

    manifest_path = base_results / "full_pipeline_segments.txt"
    script_lines += [
        f"echo {_quote(f'Merge generated segments -> {segments_root}')}",
        f"mkdir -p {_quote(str(base_results))}",
        f"printf '%s\\n' {' '.join(_quote(str(p)) for p in run_bases)} > {_quote(str(manifest_path))}",
        f"echo {_quote(f'[info] Segment manifest: {manifest_path}')}",
    ]

    eval_root = str(segments_root)
    eval_log_dir = str(base_results / "logs_eval")

    screen = payload.screen
    screen_base = eval_root
    screen_out = _remap_results_path(screen.out, base_results, base_results) or screen.out
    screen_refs = _remap_results_path(screen.refs_out, base_results, base_results) or screen.refs_out

    screen_cmd = [
        "python",
        str(ensure_script("screen_all_extxyz.py")),
        "--workdir",
        str(MATTERGEN_ROOT),
        "--base",
        screen_base,
        "--out",
        screen_out,
        "--r-cut",
        str(screen.r_cut),
        "--super",
        *(str(x) for x in screen.supercell),
        "--topk",
        str(screen.topk),
        "--refs-out",
        screen_refs,
    ]
    if screen.require_charge_balance:
        screen_cmd.append("--require-charge-balance")
    else:
        screen_cmd.append("--no-charge-balance")
    if screen.use_smact:
        screen_cmd.append("--use-smact")
    else:
        screen_cmd.append("--no-smact")
    if screen.filter_light_oxy:
        screen_cmd.extend(["--light-oxy", *(str(x) for x in screen.light_oxy)])
    else:
        screen_cmd.append("--no-light-oxy")
    screen_cmd.extend(["--screened-out", str(Path(screen_out).with_name("screened_out.csv"))])
    screen_cmd.extend(["--required-elements", *screen.required_elements])
    if screen.allowed_elements:
        screen_cmd.extend(["--allowed-elements", *screen.allowed_elements])

    top = payload.top300
    top_stage2 = screen_out

    default_top_out = str(base_results / "top300_run")
    top_output_dir_raw = top.output_dir or default_top_out
    top_output_dir = _remap_results_path(top_output_dir_raw, base_results, base_results) or top_output_dir_raw

    def _anchor_to_output_dir(path_str: str) -> str:
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        return str(Path(top_output_dir_raw) / p)

    default_export_dir = str(Path(top_output_dir_raw) / "exported_300cifs")
    top_export_dir_raw = _anchor_to_output_dir(top.export_dir or default_export_dir)
    top_export_dir = _remap_results_path(top_export_dir_raw, base_results, base_results) or top_export_dir_raw

    top_refs_out_raw = _anchor_to_output_dir(top.refs_out or "top300_refs.txt")
    top_refs_out = _remap_results_path(top_refs_out_raw, base_results, base_results) or top_refs_out_raw

    top_index_name = top.export_index_name or "export_300index.csv"

    top_ehull_out_raw = _anchor_to_output_dir(top.ehull_out or "chgnet_hull_top300.csv")
    top_ehull_out = _remap_results_path(top_ehull_out_raw, base_results, base_results) or top_ehull_out_raw

    top_filtered_out_raw = _anchor_to_output_dir(top.filtered_out or "chgnet_hull_top300_filtered.csv")
    top_filtered_out = _remap_results_path(top_filtered_out_raw, base_results, base_results) or top_filtered_out_raw

    top_voltage_out_raw = _anchor_to_output_dir(top.voltage_out or "chgnet_voltage_window_top300.csv")
    top_voltage_out = _remap_results_path(top_voltage_out_raw, base_results, base_results) or top_voltage_out_raw

    top_cmd = [
        "python",
        str(ensure_script("run_top300_pipeline.py")),
        "--workdir",
        str(MATTERGEN_ROOT),
        "--output-dir",
        top_output_dir,
        "--stage2-csv",
        top_stage2,
        "--topk",
        str(top.topk),
        "--selection-mode",
        top.selection_mode,
        "--refs-out",
        top_refs_out,
        "--export-dir",
        top_export_dir,
        "--export-prefix",
        top.export_prefix,
        "--export-index-name",
        top_index_name,
        "--ehull-threshold",
        str(top.ehull_threshold),
        "--ehull-out",
        top_ehull_out,
        "--filtered-out",
        top_filtered_out,
        "--voltage-out",
        top_voltage_out,
        "--voltage-threshold",
        str(top.voltage_threshold),
        "--min-voltage-window",
        str(top.min_voltage_window),
    ]
    if top.target_voltage is not None:
        top_cmd.extend(["--target-voltage", str(top.target_voltage)])
    if top.voltage_step is not None:
        top_cmd.extend(["--voltage-step", str(top.voltage_step)])
    if top.dry_run:
        top_cmd.append("--dry-run")

    script_lines += [
        "echo '==== Unified eval_all.sh over generated segments ===='",
        f"ROOT={_quote(eval_root)} WORKDIR={_quote(str(MATTERGEN_ROOT))} LOGDIR={_quote(eval_log_dir)} RECURSIVE=1 bash {_quote(str(ensure_script('eval_all.sh')))}",
        "echo '==== Unified screen_all_extxyz.py over all relaxed structures ===='",
        " ".join(_quote(x) for x in screen_cmd),
        "echo '==== Global run_top300_pipeline.py from unified stage2 CSV ===='",
        " ".join(_quote(x) for x in top_cmd),
    ]

    full_cmd = ["bash", "-c", "\n".join(script_lines)]
    job = launch_job("full_pipeline", full_cmd, params=payload.dict())
    return job


@app.get("/api/jobs")
def list_jobs():
    out = []
    for job in jobs.all():
        job.elapsed_seconds = _compute_elapsed(job)
        out.append(job)
    return out


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    job.elapsed_seconds = _compute_elapsed(job)
    return job


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_job_log(job_id: str, lines: int = 400):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return tail_log(Path(job.log_path), lines=lines)


def _compute_elapsed(job: Job) -> Optional[float]:
    if job.started_at:
        end_time = job.finished_at or datetime.utcnow()
        return (end_time - job.started_at).total_seconds()
    return None


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status not in {JobStatus.RUNNING, JobStatus.QUEUED}:
        return {"status": "not_running"}
    proc = job_procs.get(job_id)
    if not proc:
        return {"status": "not_running"}
    try:
        # kill process group first
        if job.pgid:
            try:
                os.killpg(job.pgid, signal.SIGTERM)
            except Exception:
                pass
        proc.send_signal(signal.SIGTERM)
        time.sleep(2)
        if proc.poll() is None:
            if job.pgid:
                try:
                    os.killpg(job.pgid, signal.SIGKILL)
                except Exception:
                    pass
            proc.kill()
        jobs.update(job_id, status=JobStatus.FAILED, finished_at=datetime.utcnow(), error="cancelled by user")
        return {"status": "cancelled"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/structures")
def list_structures(directory: str):
    base = Path(directory).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=404, detail="directory not found")
    files = []
    for ext in ("*.cif", "*.extxyz"):
        files.extend(sorted(str(p) for p in base.glob(ext)))
    return {"directory": str(base), "files": files}


@app.get("/api/structures/content", response_class=PlainTextResponse)
def get_structure_content(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        with open(p, "r") as fh:
            return fh.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _structure_from_file(path: Path):
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"pymatgen not available: {exc}")
    if path.suffix.lower() in {".cif", ".vasp", ".poscar"}:
        return Structure.from_file(path)
    if path.suffix.lower() in {".xyz", ".extxyz"}:
        try:
            import ase.io
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"ase not available: {exc}")
        atoms = ase.io.read(path)
        return AseAtomsAdaptor().get_structure(atoms)
    raise HTTPException(status_code=400, detail="Unsupported structure format")


@app.get("/api/structures/preview")
def get_structure_preview(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        from pymatgen.vis.structure_plt import StructureMPLPlotter
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"matplotlib/pymatgen vis not available: {exc}")
    s = _structure_from_file(p)
    plotter = StructureMPLPlotter(s)
    fig = plotter.get_plot()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    fig.clf()
    return {"path": str(p), "image": f"data:image/png;base64,{img_b64}"}


@app.get("/api/voltage")
def get_voltage(path: Optional[str] = None):
    target = Path(path or (SCRIPTS_DIR.parent / "results" / "chgnet_voltage_window_top300.csv")).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail="voltage CSV not found")
    import csv
    rows = []
    with target.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return {"path": str(target), "rows": rows}


# ---- static files (frontend) ----

FRONTEND_DIR = REPO_ROOT / "mattergen_webapp" / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
