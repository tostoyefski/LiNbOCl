# 在 RTX 4090 容器里运行 MatterGen

这套方案适用于一台 Linux 主机 + RTX 4090 + Docker。容器里可以打开网页，也可以直接跑命令行全流程。

## 1. 4090 主机前提

宿主机需要安装：

```bash
nvidia-smi
docker --version
docker compose version
```

并安装 NVIDIA Container Toolkit。安装后这个命令必须能看到 4090：

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

如果这一步失败，先修宿主机 Docker GPU 透传，不要先调 MatterGen。

## 2. 上传项目

建议放成：

```text
/data/project/
  mattergen/
  mattergen_webapp/

/data/mattergen_runs/
  _runtime/
  results200/
```

示例：

```bash
rsync -avh --exclude '.venv' --exclude '__pycache__' \
  /home/tao/mattergen user@4090host:/data/project/

rsync -avh --exclude '.venv' --exclude '__pycache__' \
  /home/tao/mattergen_webapp user@4090host:/data/project/
```

如果容器不能联网，提前上传 HuggingFace 缓存：

```bash
rsync -avh /mnt/e/mattergen_runs/_runtime/huggingface/ \
  user@4090host:/data/mattergen_runs/_runtime/huggingface/
```

## 3. 构建镜像

在 4090 主机上：

```bash
cd /data/project/mattergen_webapp
IMAGE_NAME=mattergen-webapp:cu118 bash container/build_image.sh
```

这个脚本会创建精简 build context，避免把历史结果、`.venv`、`md_traj` 等大目录打进镜像。

## 4. 启动网页

```bash
cd /data/project/mattergen_webapp/container

export MP_API_KEY='你的 Materials Project API key'
export MATTERGEN_RUNS_DIR=/data/mattergen_runs

docker compose up mattergen-web
```

浏览器打开：

```text
http://4090主机IP:8000
```

网页默认路径需要改成容器内路径：

```text
BASE_RESULTS_DIR = /runs/results200
ROOT             = /runs/results200
screen base      = /runs/results200
screen out       = /runs/results200/stage2_candidates.csv
top300 stage2_csv = /runs/results200/stage2_candidates.csv
top300 output_dir = /runs/results200/top300_run
top300 export_dir = /runs/results200/top300_run/exported_300cifs
```

当前后端会从 `RESULTS_ROOT` 自动注入这些默认路径；如果你使用本文的 compose 配置，页面加载后应自动显示 `/runs/results200`。如果浏览器缓存导致仍显示旧的 `/mnt/e/...`，强制刷新页面。

宿主机上实际文件会在：

```text
/data/mattergen_runs/results200
```

## 5. 不开网页，直接跑全流程

```bash
cd /data/project/mattergen_webapp/container

export MP_API_KEY='你的 Materials Project API key'
export MATTERGEN_RUNS_DIR=/data/mattergen_runs
export CHEMICAL_SYSTEMS='Li-Nb-O-Cl'
export BATCH_SIZE=16
export NUM_BATCHES_PER_SEGMENT=20
export SEGMENTS=10
export TOPK=300

docker compose --profile cli run --rm mattergen-cli
```

总生成量约为：

```text
BATCH_SIZE * NUM_BATCHES_PER_SEGMENT * SEGMENTS
```

例如 `16 * 20 * 10 = 3200` 个结构。

## 6. 输出目录

容器内：

```text
/runs/results200/_segments/batch001/...
/runs/results200/stage2_candidates.csv
/runs/results200/top300_run/
```

宿主机：

```text
/data/mattergen_runs/results200/_segments/batch001/...
/data/mattergen_runs/results200/stage2_candidates.csv
/data/mattergen_runs/results200/top300_run/
```

缓存和临时文件都在：

```text
/data/mattergen_runs/_runtime
```

这样不会写到容器系统盘，也不会吃 Windows C 盘。

## 7. 进入容器检查

```bash
docker compose run --rm mattergen-web bash

python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY

mattergen-generate --help
mattergen-evaluate --help
```

## 8. 常见调整

- 4090 显存 24GB，`BATCH_SIZE=16` 通常比 1650 稳很多；如果仍 OOM，改成 `BATCH_SIZE=8`。
- 如果不想联网下载模型，确认 `/data/mattergen_runs/_runtime/huggingface` 已经有本机迁移过去的缓存。
- 如果 CHGNet / MP 阶段失败，先用 `DRY_RUN=1` 跑导出链路，再检查 `MP_API_KEY` 和容器网络。
- 如果网页里仍显示 `/mnt/e/...`，这是本机 WSL 路径；容器内要统一改成 `/runs/...`。
