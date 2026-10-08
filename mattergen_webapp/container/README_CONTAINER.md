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

容器 CLI 使用与 HPC 相同的筛选脚本：默认要求 `Li Nb O Cl` 全部存在，并限制为这四种元素；开启化学检查与氧比例过滤，Top-K 前去重后按组成轮流选择。具体规则见 [筛选说明](../README.md#筛选规则与审计)，环境变量见 [HPC 参数表](../hpc/README_HPC.md#7-调整运行规模)。

Compose 当前只自动透传配置中列出的宿主机变量。附加筛选参数请通过 `run -e` 传入，例如要求候选在 4.5 V 稳定且已采样窗口至少 1 V：

```bash
docker compose --profile cli run --rm \
  -e TARGET_VOLTAGE=4.5 -e MIN_VOLTAGE_WINDOW=1.0 mattergen-cli
```

省略 `TARGET_VOLTAGE` 时不附加工作电压条件。先检查去重和 CIF 导出可运行 `docker compose --profile cli run --rm -e DRY_RUN=1 mattergen-cli`；此模式跳过体相和电压计算。

## 6. 输出目录

容器内：

```text
/runs/results200/_segments/batch001/...
/runs/results200/stage2_candidates.csv
/runs/results200/screened_out.csv
/runs/results200/top300_run/selection_audit.csv
/runs/results200/top300_run/voltage_filter_audit.csv
/runs/results200/top300_run/final_candidates.csv
```

宿主机：

```text
/data/mattergen_runs/results200/_segments/batch001/...
/data/mattergen_runs/results200/stage2_candidates.csv
/data/mattergen_runs/results200/top300_run/
```

`final_candidates.csv` 包含通过体相和电压筛选的候选。电压容差默认 0.001 eV/non-Li atom；失败、无窗口或未满足额外电压条件的候选保留在审计中。历史 stage2 CSV 需从已有 `relaxed.extxyz` 重新筛选，生成新的 `score_kind=li_periodic_geometry_proxy_v1` 后再运行 Top-K。

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
- 如果 CHGNet / MP 阶段失败，先用 `docker compose --profile cli run --rm -e DRY_RUN=1 mattergen-cli` 跑导出链路，再检查 `MP_API_KEY` 和容器网络。
- 如果网页里仍显示 `/mnt/e/...`，这是本机 WSL 路径；容器内要统一改成 `/runs/...`。
