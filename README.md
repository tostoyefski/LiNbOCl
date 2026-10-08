# LiNbOCl / MatterGen 材料生成项目

本仓库包含 `mattergen/` 核心项目和 `mattergen_webapp/` Web 控制台。
核心项目的原始许可证、NOTICE 和说明保存在 `mattergen/` 中。
## 下载并运行

```bash
git clone https://github.com/tostoyefski/LiNbOCl.git
cd LiNbOCl/mattergen
```

按照 [MatterGen 安装说明](mattergen/README.md) 安装核心环境。Linux 依赖使用 CUDA 11.8 / PyTorch 2.2.1，请优先沿用该说明的 `uv` 安装方式。
在已激活的 MatterGen 环境中安装 Web 和筛选依赖：

```bash
uv pip install -r ../mattergen_webapp/backend/requirements.txt
uv pip install chgnet mp-api
cd ../mattergen_webapp
export MATTERGEN_ROOT="$(cd ../mattergen && pwd)"
export RESULTS_ROOT="$PWD/results"
export MATTERGEN_RUNTIME_ROOT="$PWD/_runtime"
# 需要 Materials Project 查询时，设置自己的 MP_API_KEY 环境变量。
uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000`。模型权重在新机器上单独准备，参见核心项目说明。

- [Web 控制台说明](mattergen_webapp/README.md)
- [GPU 容器部署](mattergen_webapp/container/README_CONTAINER.md)
- [HPC / Slurm 部署](mattergen_webapp/hpc/README_HPC.md)


