# MatterGen 超算迁移说明

这套迁移方案不依赖网页服务，直接用调度系统跑命令行版全流程：

1. 分段运行 `dd.sh` 生成结构。
2. 所有分段生成完成后统一 `eval_all.sh`。
3. 对全部 `relaxed.extxyz` 统一 `screen_all_extxyz.py`。
4. 从全局 `stage2_candidates.csv` 里运行 `run_top300_pipeline.py`，得到全局 Top-K。

## 1. 需要迁移的内容

建议在超算上放成：

```text
$HOME/mattergen_project/
  mattergen/
  mattergen_webapp/
```

本机当前大致大小：

```text
mattergen                         约 11G
mattergen_webapp                  约 1.6G
MatterGen/HuggingFace 模型缓存     约 3.5G
```

结果目录可选迁移；如果只是重新跑，不需要搬旧结果。

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

## 8. 输出位置

```text
$RESULTS_ROOT/_segments/batch001/...       分段生成结果
$RESULTS_ROOT/full_pipeline_segments.txt   分段清单
$RESULTS_ROOT/logs_eval/                   评估日志
$RESULTS_ROOT/stage2_candidates.csv        全局快筛结果
$RESULTS_ROOT/top300_refs.txt              全局快筛 Top-K 引用
$RESULTS_ROOT/top300_run/                  全局精选导出、CHGNet、MP 电压结果
```

## 9. 常见问题

- 不要把 `RESULTS_ROOT` 或 `RUNTIME_ROOT` 放在 `$HOME`，应放在 `$SCRATCH`、`$WORK` 或超算提供的大容量文件系统。
- 如果计算节点不能联网，需要提前上传 HuggingFace 模型缓存，并确认 `HF_HOME` 指向 `$RUNTIME_ROOT/huggingface`。
- 如果 CHGNet / MP 阶段失败，先用 `DRY_RUN=1` 测试导出逻辑，再单独检查 `MP_API_KEY` 和网络策略。
- 如果显存不足，把 `BATCH_SIZE` 从 16 降到 8。
- 如果单个作业时间限制短，把 `SEGMENTS` 拆小，分多次作业运行，或者让管理员提供更长 walltime 队列。
