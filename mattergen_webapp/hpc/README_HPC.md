# HPC / Slurm 部署

HPC 使用统一入口 `workflow/pipeline/run_full_pipeline.sh`，无需启动 Web。操作参数和筛选规则见 [工作流指南](../../workflow/README.md)。

## 1. 准备完整仓库与环境

在集群克隆整个仓库，或一次上传完整 LiNbOCl 目录，保留三个子目录的相对布局：

```bash
git clone https://github.com/tostoyefski/LiNbOCl.git "$HOME/LiNbOCl"
cd "$HOME/LiNbOCl"
```

```text
LiNbOCl/
├── workflow/
├── mattergen/
└── mattergen_webapp/
```

按照 [根 README](../../README.md#安装) 和 [MatterGen 环境说明](../../mattergen/README.md#installation) 安装环境。若集群使用 module/conda，按集群要求配置 CUDA 和 Python，再安装 `mattergen/`、Web 后端依赖、CHGNet 与 mp-api。模型缓存需要单独准备；计算节点不能联网时，应预先放到 `$RUNTIME_ROOT/huggingface`。

## 2. 修改调度模板

编辑 `mattergen_webapp/hpc/slurm_full_pipeline.sbatch`：调整 partition、GPU 数量、CPU、内存、walltime，以及环境激活命令。模板的环境激活必须与实际安装方式一致，例如使用 conda，或激活仓库内的 MatterGen 虚拟环境。

完整体相/电压计算前设置自己的 `MP_API_KEY`，不要将真实密钥提交到 Git。结果与缓存建议放在集群的大容量文件系统；下面假定集群已定义 `$SCRATCH`：

```bash
export PROJECT_ROOT="$PWD"
export MATTERGEN_ROOT="$PROJECT_ROOT/mattergen"
export WORKFLOW_ROOT="$PROJECT_ROOT/workflow"
export RESULTS_ROOT="$SCRATCH/LiNbOCl/results"
export RUNTIME_ROOT="$SCRATCH/LiNbOCl/_runtime"
```

`PROJECT_ROOT` 指向整个仓库，`WORKFLOW_ROOT` 指向其中的 `workflow/`。不会分别上传或安装两套项目脚本。

候选和 MP 竞争结构都在筛选阶段使用相同 MatterSim 设置优化，再计算 CHGNet 0.3.0 能量；电压复用同一优化快照。计算节点无网络时提前准备 MatterSim 权重，设置 `MATTERSIM_CHECKPOINT` 为其绝对路径；可用 `RELAX_FMAX`、`RELAX_STEPS` 调整双方共同的收敛设置，默认 0.05 eV/Å、500 步。

## 3. 提交任务

从仓库根目录提交：

```bash
export CHEMICAL_SYSTEMS='Li-Nb-O-Cl'
export BATCH_SIZE=16
export NUM_BATCHES_PER_SEGMENT=20
export SEGMENTS=10
export TOPK=300
mkdir -p logs
sbatch mattergen_webapp/hpc/slurm_full_pipeline.sbatch
```

总生成量约为 `16 × 20 × 10 = 3200` 个结构。查看调度状态和日志，下面将 12345 替换为实际任务号：

```bash
squeue -u "$USER"
tail -f logs/mattergen-full-12345.out
```

不经调度模板、在已分配的交互计算节点运行时：

```bash
bash workflow/pipeline/run_full_pipeline.sh
```

## 4. 调整筛选

常用参数及默认值统一见 [工作流参数表](../../workflow/README.md#2-生成并运行全流程)。默认要求 `Li Nb O Cl` 全部存在且只允许这四种元素，启用化学与氧比例过滤；先去重、按组成轮流选择；电压容差为 0.001 eV/non-Li atom。

例如附加工作电压条件后再提交：

```bash
export SELECTION_MODE=diverse
export TARGET_VOLTAGE=4.5
export MIN_VOLTAGE_WINDOW=1.0
sbatch mattergen_webapp/hpc/slurm_full_pipeline.sbatch
```

`TARGET_VOLTAGE` 为空时不附加工作电压条件。`DRY_RUN=1` 仍执行生成和评估，随后只做选择和 CIF 导出。已有结构重跑筛选时，用 [已有结果命令](../../workflow/README.md#3-从已有-relaxedextxyz-重跑)，无需重新提交生成。

## 5. 输出与迁移

```text
$RESULTS_ROOT/_segments/                  生成及评估结果
$RESULTS_ROOT/logs_eval/                  评估日志
$RESULTS_ROOT/stage2_candidates.csv       化学通过候选
$RESULTS_ROOT/screened_out.csv            拒绝/错误审计
$RESULTS_ROOT/top300_run/                 选择、导出、体相和电压结果
$RESULTS_ROOT/top300_run/final_candidates.csv
```

选择审计、最终电压审计和各表含义见 [结果说明](../../workflow/README.md#4-查找结果与筛选依据)。历史 stage2 分数不兼容当前 `score_kind=li_periodic_geometry_proxy_v1`，从已有 `relaxed.extxyz` 重跑 screen 和 Top-K 即可。

显存不足时降低 `BATCH_SIZE`；walltime 不足时减小 `SEGMENTS`。CHGNet/MP 阶段失败可对已有 stage2 单独运行 Top-K dry-run 检查导出，再检查模型依赖、密钥和计算节点网络。
