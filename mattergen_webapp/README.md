# MatterGen 逆向设计 Web 控制台

基于 `mattergen` 仓库的四个批处理脚本（`dd.sh`, `eval_all.sh`, `screen_all_extxyz.py`, `run_top300_pipeline.py`）封装的轻量 Web UI，支持参数化运行、作业排队与尾部日志查看。默认假设 `mattergen` 仓库在本项目上级目录（`../mattergen`），可通过环境变量 `MATTERGEN_ROOT` 覆盖。

## 快速启动

### 在另一台机器上准备环境

本仓库只保存代码和部署说明，不包含历史运行结果、分析图表、轨迹、日志、虚拟环境、模型权重或 API 密钥。
需要单独准备 `mattergen` 依赖，并让两个项目位于同一父目录：

```text
project/
├── mattergen/
└── mattergen_webapp/
```

按 `mattergen` 自身的安装说明安装生成和评估环境，再安装下面的 Web 后端依赖。
如果使用其他目录布局，请设置 `MATTERGEN_ROOT`。
模型权重需在新机器上另行下载；Materials Project 密钥通过 `MP_API_KEY` 环境变量提供。
GPU 容器部署见 [容器说明](container/README_CONTAINER.md)，集群部署见 [HPC 说明](hpc/README_HPC.md)。

直接启动时，请显式设置新机器上的结果目录和运行缓存目录：

```bash
export MATTERGEN_ROOT=/path/to/mattergen
export RESULTS_ROOT=/path/to/mattergen_runs/results
export MATTERGEN_RUNTIME_ROOT=/path/to/mattergen_runs/_runtime
```

### 启动 Web 控制台

```bash
cd mattergen_webapp
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 运行 API + 前端静态页
uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

打开浏览器访问 `http://localhost:8000`。所有 API 路径为 `/api/...`，前端静态文件直接由 FastAPI 提供。

> **不改动原脚本**：`scripts/` 目录里是拷贝后的独立脚本（dd.sh、eval_all.sh、screen_all_extxyz.py、run_top300_pipeline.py），后端调用这些脚本，并通过 `WORKDIR`/`--workdir` 指向真实的 `mattergen` 仓库运行。

## 功能说明

- **dd.sh**：批量生成候选材料，可显式传入 `CHEMICAL_SYSTEMS`/`CHEMICAL_SYSTEMS_FILE`，若未指定则按 `ELEMENTS` 与 `COMBO_SIZES` 自动组合。输出根目录 `BASE_RESULTS_DIR` 可自定义（支持绝对路径）。
- **eval_all.sh**：对指定 ROOT 下的子目录执行 `mattergen-evaluate`，写入 `metrics.json` / `relaxed.extxyz`。
- **screen_all_extxyz.py**：递归读取 `relaxed.extxyz`，计算组成/连通性特征，输出 `stage2_candidates.csv` 与 `top150_refs.txt`。
- **run_top300_pipeline.py**：从 stage2 CSV 选 Top-K，导出 CIF，并可调用 CHGNet/MP 计算稳定性与电压窗口（支持 `--dry-run`）。
- **作业管理**：所有任务后台线程运行，状态记录在内存，日志保存在 `backend/job_logs`，前端可点击行查看尾部日志。

## 参数 & 环境

- `MATTERGEN_ROOT`：若 `mattergen` 仓库不在默认位置，可在启动服务前设置，例如  
  `MATTERGEN_ROOT=/path/to/mattergen uvicorn backend.main:app --app-dir backend`
- 四个表单的默认值与脚本原始默认参数一致，可按需修改后提交。

## 已知限制

- 未实现作业取消/并发队列控制，提交后会直接启动子进程。
- 前端未做身份/权限校验；在共享环境中部署时请加上反向代理或认证。
- 运行 eval/screen/top300 依赖对应脚本所需的第三方包（CHGNet、pymatgen、SMACT 等）已正确安装，否则会在日志中失败。
