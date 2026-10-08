FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/mattergen-venv/bin:$PATH \
    MATTERGEN_ROOT=/workspace/mattergen \
    WORKFLOW_ROOT=/workspace/workflow \
    MATTERGEN_RUNTIME_ROOT=/runs/_runtime \
    RESULTS_ROOT=/runs/results \
    RUNTIME_ROOT=/runs/_runtime \
    TMPDIR=/runs/_runtime/tmp \
    HF_HOME=/runs/_runtime/huggingface \
    HF_HUB_CACHE=/runs/_runtime/huggingface/hub \
    TRANSFORMERS_CACHE=/runs/_runtime/huggingface/transformers \
    TORCH_HOME=/runs/_runtime/torch \
    CUDA_CACHE_PATH=/runs/_runtime/cuda \
    MPLCONFIGDIR=/runs/_runtime/matplotlib \
    UV_CACHE_DIR=/runs/_runtime/uv \
    PIP_CACHE_DIR=/runs/_runtime/pip \
    NUMBA_CACHE_DIR=/runs/_runtime/numba \
    TRITON_CACHE_DIR=/runs/_runtime/triton \
    PYTHONPYCACHEPREFIX=/runs/_runtime/pycache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    build-essential \
    cmake \
    ninja-build \
    gfortran \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && python -m pip install --upgrade pip uv \
    && uv venv /opt/mattergen-venv --python 3.10

WORKDIR /workspace

COPY mattergen /workspace/mattergen
COPY mattergen_webapp /workspace/mattergen_webapp
COPY workflow /workspace/workflow

RUN . /opt/mattergen-venv/bin/activate \
    && cd /workspace/mattergen \
    && uv pip install -e . \
    && uv pip install -r /workspace/mattergen_webapp/backend/requirements.txt \
    && uv pip install chgnet mp-api \
    && python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

RUN mkdir -p /runs /runs/_runtime /runs/results

WORKDIR /workspace/mattergen_webapp

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
