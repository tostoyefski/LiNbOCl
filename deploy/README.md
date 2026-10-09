# 本次服务器部署配置

这里保存 6000 个 Li-Nb-O-Cl 候选的实际 Slurm 部署脚本，GPU 总预算为最多 2 个。脚本使用本次服务器的绝对路径；迁移时需修改项目目录、容器目录、Python 环境及权重路径。

- `in_container.sh`：通过现有 proot 容器执行命令，复用服务器已有的 MatterGen 安装。
- `run_shard.py` / `shard.sbatch`：30 组，每组生成 200 个候选并松弛；生成条件为 `chemical_system=Li-Nb-O-Cl`、`energy_above_hull=0.05`，使用已有 `chemical_system_energy_above_hull` 权重。
- `screen_inside.sh` / `screen.sbatch`：校验 6000 个松弛结构，统一化学筛选及去重，再使用 2 个 GPU 预测能量、2 个 CPU 进程扫描电压窗口；凸包筛选阈值为 0.05 eV/atom。

运行前需要存在 `_runtime/venv/bin/python`（复用已有 MatterGen 依赖的独立环境）、MatterSim 权重 `_runtime/MatterSim-v1.0.0-1M.pth`、CHGNet 0.3.0 模型及 `_runtime/mp.env`。密钥文件权限应设为 `600`；环境、权重、密钥和结果目录均不纳入 Git。

从仓库根目录提交，先验证首组，再运行剩余组和筛选：

```bash
mkdir -p results/run6000_20261008/logs
first=$(sbatch --parsable --array=0 deploy/shard.sbatch)
rest=$(sbatch --parsable --dependency="afterok:$first" --array=1-29%2 deploy/shard.sbatch)
screen=$(sbatch --parsable --dependency="afterok:$first:$rest" deploy/screen.sbatch)
```

结果位于 `results/run6000_20261008/`。每组完整生成 200 个结构后保存生成检查点；生成中途被终止时需重新生成该组。松弛每 20 个结构保存一次检查点，重新提交会复用已完成的生成和松弛检查点。
