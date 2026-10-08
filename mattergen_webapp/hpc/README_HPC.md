# MatterGen 超算迁移说明

这套迁移方案不依赖网页服务，直接用调度系统跑命令行版全流程：

1. 分段运行 `dd.sh` 生成结构。
2. 所有分段生成完成后统一 `eval_all.sh`。
3. 对全部 `relaxed.extxyz` 统一 `screen_all_extxyz.py`。
4. 从全局 `stage2_candidates.csv` 先去重并选取最多 Top-K 个候选，导出 CIF。
5. 计算体相稳定性与电压窗口，写出通过两道筛选的 `final_candidates.csv` 和接受/拒绝审计。

筛选规则、字段含义和手动 CLI 示例见 [Web/CLI 说明](../README.md#筛选规则与审计)。HPC 与 Web 使用同一套筛选脚本。

## 1. 需要迁移的内容

建议在超算上放成：

```text
$HOME/mattergen_project/
  mattergen/
  mattergen_webapp/
```

本仓库只保存代码和部署说明；生成环境、模型权重和缓存需要单独准备。结果目录可选迁移；若只重新跑，不需要搬旧结果。

## 2. 上传代码和模型缓存

示例：

```bash
rsync -avh --exclude '.venv' --exclude '__pycache__' \
  /home/tao/mattergen user@cluster:$HOME/mattergen_project/

rsync -avh --exclude '.venv' --exclude '__pycache__' \
  /home/tao/mattergen_webapp user@cluster:$HOME/mattergen_project/

rsync -avh /mnt/e/mattergen_runs/_runtime/huggingface/ \
  user@cluster:$SCRATCH/mattergen_runs/_runtime/huggingface/
```

如果超算计算节点不能联网，模型缓存必须提前传上去。

## 3. 创建 Python 环境

在超算登录节点或交互节点上：

```bash
module load cuda/12.1        # 按超算实际模块修改
module load anaconda/2023    # 按超算实际模块修改

conda create -n mattergen python=3.10 -y
conda activate mattergen

cd $HOME/mattergen_project/mattergen
pip install -e .

cd $HOME/mattergen_project/mattergen_webapp
pip install -r backend/requirements.txt
pip install chgnet mp-api ase pymatgen pandas tqdm smact
```

如果集群推荐 Apptainer/Singularity，优先用容器，避免节点环境差异。

## 4. 设置 MP API Key

不要把真实 key 写进提交脚本。可在提交前执行：

```bash
export MP_API_KEY='你的 Materials Project API key'
```

或写入权限受限的文件：

```bash
echo "export MP_API_KEY='你的key'" > ~/.mattergen_secrets
chmod 600 ~/.mattergen_secrets
source ~/.mattergen_secrets
```

## 5. 修改 Slurm 模板

编辑：

```text
mattergen_webapp/hpc/slurm_full_pipeline.sbatch
```

重点改这些：

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00

module load cuda/12.1
module load anaconda/2023
conda activate mattergen
```

以及结果和缓存目录：

```bash
export RESULTS_ROOT="$SCRATCH/mattergen_runs/results200"
export RUNTIME_ROOT="$SCRATCH/mattergen_runs/_runtime"
```

## 6. 提交任务

```bash
cd $HOME/mattergen_project/mattergen_webapp
mkdir -p logs
sbatch hpc/slurm_full_pipeline.sbatch
```

查看状态：

```bash
squeue -u $USER
tail -f logs/mattergen-full-<jobid>.out
```

## 7. 调整运行规模

等价于本机网页中的“一键全流程”：

```bash
export CHEMICAL_SYSTEMS='Li-Nb-O-Cl'
export BATCH_SIZE=16
export NUM_BATCHES_PER_SEGMENT=20
export SEGMENTS=10
export TOPK=300
sbatch hpc/slurm_full_pipeline.sbatch
```

总生成量约为：

```text
BATCH_SIZE * NUM_BATCHES_PER_SEGMENT * SEGMENTS
```

比如 `16 * 20 * 10 = 3200` 个结构。

筛选相关环境变量如下；列表使用空格分隔：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `REQUIRED_ELEMENTS` | `Li Nb O Cl` | 每个候选必须包含的元素 |
| `ALLOWED_ELEMENTS` | `Li Nb O Cl` | 元素白名单；设置为空可取消白名单 |
| `REQUIRE_CHARGE_BALANCE` | `1` | 固定计量式电中性过滤 |
| `USE_SMACT` | `1` | SMACT Pauling 规则及电中性过滤 |
| `FILTER_LIGHT_OXY` | `1` | 是否实际过滤氧/卤素比例 |
| `LIGHT_OXY` | `0.05 0.35` | `O/(O+F+Cl+Br+I)` 的闭区间 |
| `R_CUT` | `3.0` | 周期 Li 图的近邻截断距离，Å |
| `SELECTION_MODE` | `diverse` | 约化组成轮流选取；`score` 为分数排序；两者都先去重 |
| `VOLTAGE_THRESHOLD` | `0.001` | 电压稳定性数值容差，eV/non-Li atom |
| `TARGET_VOLTAGE` | 空 | 可选工作电压，相对 Li/Li⁺；指定后要求该精确电压稳定 |
| `MIN_VOLTAGE_WINDOW` | `0.0` | 最小已采样稳定区间宽度，V；默认仍要求非零宽度 |
| `DRY_RUN` | `0` | `1` 时止于去重、选择和 CIF 导出 |

例如附加工作电压条件：

```bash
export REQUIRED_ELEMENTS='Li Nb O Cl'
export ALLOWED_ELEMENTS='Li Nb O Cl'
export SELECTION_MODE=diverse
export VOLTAGE_THRESHOLD=0.001
export TARGET_VOLTAGE=4.5
export MIN_VOLTAGE_WINDOW=1.0
sbatch hpc/slurm_full_pipeline.sbatch
```

不指定工作电压时保持 `TARGET_VOLTAGE` 为空。切换其他体系时应同步调整生成体系与 required/allowed 元素；氧比例过滤和化学检查只有通过显式设置对应开关为 `0` 才会关闭。关闭独立电中性过滤不会关闭 SMACT 内部的电中性检查。

## 8. 输出位置

```text
$RESULTS_ROOT/_segments/batch001/...       分段生成结果
$RESULTS_ROOT/full_pipeline_segments.txt   分段清单
$RESULTS_ROOT/logs_eval/                   评估日志
$RESULTS_ROOT/stage2_candidates.csv        全部通过化学与组成过滤的候选
$RESULTS_ROOT/screened_out.csv             拒绝/读取错误及原因
$RESULTS_ROOT/top300_refs.txt              几何分数参考列表，未做最终去重
$RESULTS_ROOT/top300_run/top300_refs.txt    去重和多样性选择后的引用
$RESULTS_ROOT/top300_run/selection_audit.csv
$RESULTS_ROOT/top300_run/selection_summary.json
$RESULTS_ROOT/top300_run/exported_300cifs/  本次精选 CIF 及导出索引
$RESULTS_ROOT/top300_run/chgnet_hull_top300.csv
$RESULTS_ROOT/top300_run/chgnet_hull_top300_filtered.csv
$RESULTS_ROOT/top300_run/chgnet_voltage_window_top300.csv
$RESULTS_ROOT/top300_run/voltage_filter_audit.csv
$RESULTS_ROOT/top300_run/final_candidates.csv
```

`quick_score` 只表示周期 Li 图的几何贯通程度，不能解释为电导率。电压结果保留所有稳定区间、边界夹区和扫描截尾标记；计算失败独立记录，不会充当分解边界。最终列表要求体相通过且有非零稳定区间，附加的工作电压和最小宽度条件也必须满足；它仍需后续 DFT/MD 验证。

旧的 stage2 分数不兼容 `score_kind=li_periodic_geometry_proxy_v1`。迁移历史结果后，应从已有 `relaxed.extxyz` 重跑 screen 和 Top-K。`DRY_RUN=1` 不产生本次体相/电压计算或最终候选结果。

## 9. 常见问题

- 不要把 `RESULTS_ROOT` 或 `RUNTIME_ROOT` 放在 `$HOME`，应放在 `$SCRATCH`、`$WORK` 或超算提供的大容量文件系统。
- 如果计算节点不能联网，需要提前上传 HuggingFace 模型缓存，并确认 `HF_HOME` 指向 `$RUNTIME_ROOT/huggingface`。
- 如果 CHGNet / MP 阶段失败，先用 `DRY_RUN=1` 测试导出逻辑，再单独检查 `MP_API_KEY` 和网络策略。
- 如果显存不足，把 `BATCH_SIZE` 从 16 降到 8。
- 如果单个作业时间限制短，把 `SEGMENTS` 拆小，分多次作业运行，或者让管理员提供更长 walltime 队列。
