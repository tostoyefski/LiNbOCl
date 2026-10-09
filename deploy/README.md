# 本次服务器部署配置

这里保存 6000 个 Li-Nb-O-Cl 候选的实际 Slurm 部署脚本，GPU 总预算为最多 2 个。脚本使用本次服务器的绝对路径；迁移时需修改项目目录、容器目录、Python 环境及权重路径。

- `in_container.sh`：通过现有 proot 容器执行命令，复用服务器已有的 MatterGen 安装。
- `run_shard.py` / `shard.sbatch`：30 组，每组生成 200 个候选并松弛；生成条件为 `chemical_system=Li-Nb-O-Cl`、`energy_above_hull=0.05`，使用已有 `chemical_system_energy_above_hull` 权重。
- `screen_inside.sh` / `screen.sbatch`：校验 6000 个松弛结构，不限制 O/(O+Cl)，统一进行四元素、电荷平衡、SMACT 筛选及去重，再使用 2 个 GPU 对候选和 MP 竞争结构统一 MatterSim 优化、计算 CHGNet 能量，2 个 CPU 进程扫描电压窗口；凸包筛选阈值为 0.05 eV/atom。电压筛选之后，在 CPU 上复用 MatterGen 的结构匹配功能比较训练集和结构参考集，移除匹配到已有结构的候选并记录来源审计。

运行前需要存在 `_runtime/venv/bin/python`（复用已有 MatterGen 依赖的独立环境）、MatterSim 权重 `_runtime/MatterSim-v1.0.0-1M.pth`、CHGNet 0.3.0 模型及 `_runtime/mp.env`。密钥文件权限应设为 `600`；环境、权重、密钥和结果目录均不纳入 Git。

筛选默认共用上述 MatterSim 权重，`RELAX_FMAX=0.05`、`RELAX_STEPS=500`；可用 `MATTERSIM_CHECKPOINT` 指定其他权重的绝对路径。优化后结构及审计保存到本次 Top-K 输出的 `relaxation/`，电压复用同一快照。旧筛选结果需重新运行筛选阶段才能获得统一基线；重新生成候选并非必要。

从仓库根目录提交，先验证首组，再运行剩余组和筛选：

```bash
mkdir -p results/run6000_20261008/logs
first=$(sbatch --parsable --array=0 deploy/shard.sbatch)
rest=$(sbatch --parsable --dependency="afterok:$first" --array=1-29%2 deploy/shard.sbatch)
screen=$(sbatch --parsable --dependency="afterok:$first:$rest" deploy/screen.sbatch)
```

生成和松弛结果位于 `results/run6000_20261008/`，取消氧比例限制后的筛选默认写入其 `screening_no_oxygen/` 子目录。每组完整生成 200 个结构后保存生成检查点；生成中途被终止时需重新生成该组。松弛每 20 个结构保存一次检查点，重新提交会复用已完成的生成和松弛检查点。

仅重新筛选已有结构时，直接提交 `deploy/screen.sbatch`。可通过 `SCREEN_INPUT_ROOT` 指定包含 `_segments/` 的输入目录，通过 `SCREEN_OUTPUT_ROOT` 指定独立输出目录；每次重筛使用新目录以保留先前结果。此操作不重新生成候选，体相筛选仍会对候选和 MP 竞争结构统一重新优化。GPU 总预算保持为 2。

新颖性步骤默认要求 MatterGen `data-release/alex-mp/` 下真实的训练 ZIP 和结构参考 LMDB.gz；仅 Git LFS 指针或缺失数据会导致筛选报错，不会判为新结构。可用 `NOVELTY_TRAINING_DATA`、`NOVELTY_REFERENCE_DATA` 指定服务器已有数据路径，`NOVELTY_TRAINING_SPLITS` 默认 `train`。通过电压筛选的中间名单保存在 `top300_run/pre_novelty_candidates.csv`，最终名单及新颖性审计另存；显式 `SKIP_NOVELTY=1` 会标记 `not_checked`。详见 [流程说明](../workflow/README.md)。
