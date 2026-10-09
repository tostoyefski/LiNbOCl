# GPU 容器部署

容器包含整个 LiNbOCl 项目：核心框架、Web 控制台和统一 `workflow/`。Web 与 CLI 都调用同一套工作流。宿主机准备 Docker、Compose、rsync 和 NVIDIA Container Toolkit，并确认 GPU 可见：

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 1. 准备仓库并构建

克隆或上传完整仓库，保留 `workflow/`、`mattergen/`、`mattergen_webapp/` 的相对布局。以下命令从仓库根目录执行：

```bash
git clone https://github.com/tostoyefski/LiNbOCl.git /data/LiNbOCl
cd /data/LiNbOCl
IMAGE_NAME=mattergen-webapp:cu118 bash mattergen_webapp/container/build_image.sh
```

构建脚本将代码复制到精简 context，排除 `.git`、虚拟环境、结果、轨迹和日志。模型权重在容器运行时另行下载；计算节点不能联网时，提前准备 Hugging Face 缓存到宿主机运行目录的 `_runtime/huggingface/`。

## 2. 启动 Web

```bash
export MATTERGEN_RUNS_DIR=/data/LiNbOCl_runs
# 需要 Materials Project 查询时，在启动前设置自己的 MP_API_KEY。
docker compose -f mattergen_webapp/container/docker-compose.yml up mattergen-web
```

浏览器访问 `http://主机IP:8000`。Compose 将宿主机运行目录挂载到 `/runs`，默认变量如下：

| 变量 | 容器路径 |
| --- | --- |
| `MATTERGEN_ROOT` | `/workspace/mattergen` |
| `WORKFLOW_ROOT` | `/workspace/workflow` |
| `RESULTS_ROOT` | `/runs/results` |
| `RUNTIME_ROOT` / `MATTERGEN_RUNTIME_ROOT` | `/runs/_runtime` |

页面会从后端读取默认结果路径。使用其他运行目录时修改 Compose 的对应环境变量，表单中的生成、评估、筛选和 Top-K 路径需指向容器内路径。

候选和 MP 竞争结构统一用 MatterSim 优化，再用 CHGNet 0.3.0 计算能量。Compose 的 `MATTERSIM_CHECKPOINT` 可使用默认模型名，或 `/runs/_runtime/` 内自备权重的容器绝对路径；CLI 还传入 `RELAX_FMAX`、`RELAX_STEPS`，默认 0.05 eV/Å、500 步。Web API 的优化参数见 [工作流指南](../../workflow/README.md#2-生成并运行全流程)。

## 3. 直接跑 CLI 全流程

```bash
export CHEMICAL_SYSTEMS='Li-Nb-O-Cl'
export BATCH_SIZE=16
export NUM_BATCHES_PER_SEGMENT=20
export SEGMENTS=10
export TOPK=300
docker compose -f mattergen_webapp/container/docker-compose.yml \
  --profile cli run --rm mattergen-cli
```

总生成量约为 `BATCH_SIZE × NUM_BATCHES_PER_SEGMENT × SEGMENTS`。CLI 直接调用 `/workspace/workflow/pipeline/run_full_pipeline.sh`，不需要 Web 服务。

Compose 只自动传入配置中列出的宿主机变量；附加参数通过 `run -e` 传入，例如要求 4.5 V 稳定且已采样稳定区间至少 1 V：

```bash
docker compose -f mattergen_webapp/container/docker-compose.yml \
  --profile cli run --rm \
  -e TARGET_VOLTAGE=4.5 -e MIN_VOLTAGE_WINDOW=1.0 mattergen-cli
```

省略工作电压时不附加该条件。`run --rm -e DRY_RUN=1 mattergen-cli` 仍生成和评估，随后只做去重、选择和 CIF 导出。只检查已有结果时进入容器，按 [工作流重跑命令](../../workflow/README.md#3-从已有-relaxedextxyz-重跑) 操作。

## 4. 结果与检查

```text
容器内 /runs/results/
宿主机 /data/LiNbOCl_runs/results/

  stage2_candidates.csv
  screened_out.csv
  top300_run/selection_audit.csv
  top300_run/exported_300cifs/
  top300_run/voltage_filter_audit.csv
  top300_run/final_candidates.csv
```

缓存与临时文件位于宿主机 `/data/LiNbOCl_runs/_runtime/`。全部输出与筛选字段见 [操作指南](../../workflow/README.md#4-查找结果与筛选依据)。历史 stage2 CSV 需从已有 `relaxed.extxyz` 重跑 screen，生成当前 `score_kind` 后再运行 Top-K。

进入容器检查环境：

```bash
docker compose -f mattergen_webapp/container/docker-compose.yml \
  run --rm mattergen-web bash
```

容器内可运行 `mattergen-generate --help`、`mattergen-evaluate --help`，或使用 `python /workspace/workflow/pipeline/run_top300_pipeline.py --help`。实际生成与 CHGNet 计算需要 GPU 和模型环境；显存不足时降低批大小。
