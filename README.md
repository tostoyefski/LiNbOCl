# LiNbOCl 材料筛选项目

从这里开始。所有项目脚本集中在 [`workflow/`](workflow/README.md)，Web、HPC 和容器共用同一套流程。

```text
LiNbOCl/
├── workflow/              操作指南与项目脚本
│   ├── pipeline/          生成 → 评估 → 筛选 → 去重 → 体相/电压筛选
│   ├── analysis/          电压绘图、评估指标汇总
│   ├── transport/         可选 MD、电导估计及轨迹后处理
│   └── tools/             可选全量凸包、再评估、格式转换
├── mattergen/             MatterGen 核心框架，原说明和许可证保留
├── mattergen_webapp/      Web 界面、HPC 模板与容器部署
└── results/               本地运行结果（不纳入 Git）
```

## 安装

克隆整个仓库，无需分别下载或上传两个项目。下面沿用 MatterGen 的 Linux/CUDA 安装方式；模型权重在运行时单独下载或预先准备。

```bash
git clone https://github.com/tostoyefski/LiNbOCl.git
cd LiNbOCl/mattergen
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install -r ../mattergen_webapp/backend/requirements.txt chgnet mp-api
cd ..
```

框架环境、权重和其他平台安装见 [MatterGen 原说明](mattergen/README.md)。本仓库不包含历史结果、轨迹、虚拟环境、模型权重或 API 密钥。

## 选择运行方式

- [命令行操作指南](workflow/README.md)：全流程、从已有结果重跑、结果位置、绘图和可选 MD。
- [Web 控制台](mattergen_webapp/README.md)：通过表单运行同一流程。
- [HPC / Slurm](mattergen_webapp/hpc/README_HPC.md)：将整个仓库部署到集群。
- [本次服务器部署配置](deploy/README.md)：复用现有 MatterGen，以最多 2 个 GPU 生成 6000 个候选并自动筛选。
- [GPU 容器](mattergen_webapp/container/README_CONTAINER.md)：Docker Web 或命令行运行。

已安装环境时，可从仓库根启动 Web：

```bash
source mattergen/.venv/bin/activate
export MATTERGEN_ROOT="$PWD/mattergen"
export WORKFLOW_ROOT="$PWD/workflow"
export RESULTS_ROOT="$PWD/results"
export MATTERGEN_RUNTIME_ROOT="$PWD/_runtime"
uvicorn main:app --app-dir mattergen_webapp/backend --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000`。需要 Materials Project 查询时，在运行前设置自己的 `MP_API_KEY`。

## 从旧版本迁移

旧 `mattergen/` 根目录和 Web 的脚本副本已整理到 `workflow/`，请更新自定义命令的路径。历史 stage2 CSV 必须从已有 `relaxed.extxyz` 重跑筛选，生成 `score_kind=li_periodic_geometry_proxy_v1` 后再运行 Top-K。

已移除旧生成脚本、混合能量基线的 `run_chgnet_phase.py`、重复指标汇总和针对历史 results200/results10 的分析脚本。删除详情及保留工具见 [操作指南](workflow/README.md#整理说明)；旧文件可从 Git 历史查找。
