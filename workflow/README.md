# LiNbOCl 操作指南

所有命令从 **LiNbOCl 仓库根目录** 执行，并使用已安装的 MatterGen 环境。首次安装见 [根 README](../README.md#安装)。

## 1. 设置本次运行目录

```bash
source mattergen/.venv/bin/activate
export MATTERGEN_ROOT="$PWD/mattergen"
export WORKFLOW_ROOT="$PWD/workflow"
export RESULTS_ROOT="$PWD/results"
export RUNTIME_ROOT="$PWD/_runtime"
# 完整体相/电压计算需要先设置自己的 MP_API_KEY。
```

`RESULTS_ROOT` 可以指向其他绝对路径，历史结果无需移动。生成模型权重、MatterSim、CHGNet 和 Materials Project 查询依赖需已准备好。

## 2. 生成并运行全流程

先用小批量检查部署，再调整规模：

```bash
CHEMICAL_SYSTEMS='Li-Nb-O-Cl' \
  BATCH_SIZE=8 NUM_BATCHES_PER_SEGMENT=1 SEGMENTS=1 TOPK=300 \
  bash workflow/pipeline/run_full_pipeline.sh
```

顺序为分段生成、统一评估、化学筛选、结构去重与多样性选择、CIF 导出、候选与 MP 竞争结构统一 MatterSim 优化、CHGNet 能量及体相凸包筛选、电压筛选、训练集及参考集结构新颖性筛选。生成数量约为 `BATCH_SIZE × NUM_BATCHES_PER_SEGMENT × SEGMENTS`。`DRY_RUN=1` 仍会生成和评估，随后止于去重与 CIF 导出；仅检查已有结构时使用下一节。

常用环境参数如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `CHEMICAL_SYSTEMS` | `Li-Nb-O-Cl` | 生成体系；多个体系用逗号或空格分隔 |
| `REQUIRED_ELEMENTS` / `ALLOWED_ELEMENTS` | `Li Nb O Cl` | 必须包含的元素 / 允许的元素；白名单可显式设为空 |
| `REQUIRE_CHARGE_BALANCE` / `USE_SMACT` | `1` / `1` | 电中性与 SMACT 化学过滤 |
| `FILTER_LIGHT_OXY` / `LIGHT_OXY` | `1` / `0.05 0.35` | 氧/卤素比例实际过滤及闭区间 |
| `R_CUT` | `3.0` | 周期 Li 图近邻距离，Å |
| `TOPK` / `SELECTION_MODE` | `300` / `diverse` | 最多导出数量 / 组成轮流选择；`score` 也先去重 |
| `MATTERSIM_CHECKPOINT` | `MatterSim-v1.0.0-1M.pth` | 双方共用的 MatterSim 权重；自备文件请使用绝对路径 |
| `RELAX_FMAX` / `RELAX_STEPS` | `0.05` / `500` | 优化收敛阈值，eV/Å / 最大优化步数 |
| `GPU_WORKERS` | `1` | MatterSim 优化与 CHGNet 能量预测进程数；每个进程使用一个可见 GPU，电压扫描使用相同数量的 CPU 进程 |
| `NOVELTY_TRAINING_DATA` / `NOVELTY_REFERENCE_DATA` | MatterGen `data-release/alex-mp/` 中的官方训练 ZIP / MP2020 参考 LMDB.gz | 本地结构数据路径；数据须实际下载，Git LFS 指针不算有效数据 |
| `NOVELTY_TRAINING_SPLITS` | `train` | 默认只检查训练划分；显式设置 `train,val` 可同时检查验证划分 |
| `SKIP_NOVELTY` | `0` | 默认执行新颖性筛选；设为 `1` 显式跳过时，输出标记为 `not_checked` |
| `VOLTAGE_THRESHOLD` | `0.001` | 电压数值容差，eV/non-Li atom |
| `TARGET_VOLTAGE` | 空 | 可选精确工作电压，相对 Li/Li⁺ |
| `MIN_VOLTAGE_WINDOW` | `0` | 最小已采样稳定区间宽度，V；仍要求非零宽度 |

需要在 4.5 V 稳定且已采样区间至少 1 V 时，可在全流程前执行 `export TARGET_VOLTAGE=4.5 MIN_VOLTAGE_WINDOW=1.0`。未指定工作电压时保持为空。

有 4 个可用 GPU 时，设置 `GPU_WORKERS=4`，或给 `run_top300_pipeline.py` 传入 `--gpu-workers 4`。Slurm 作业须同时申请 `--gres=gpu:4`。程序遵守 `CUDA_VISIBLE_DEVICES`；可见 GPU 少于请求数时会报错。化学筛选、去重与导出统一执行，候选和 MP 参考结构的优化及能量计算并行执行，然后用完整候选集合建立共同凸包。电压扫描的每个进程也使用完整竞争相集合，避免分片改变筛选结果。MP 数据每次运行统一获取，竞争相只优化和计算一次并复用；候选计算失败会保留失败记录，任一竞争相失败则阻止使用不完整的参考集筛选。分片输入、输出、日志及状态保存在输出目录的 `parallel_screening/`。

双方从各自输入结构开始，使用相同权重、FIRE 优化器及 ExpCellFilter，同时优化原子位置和晶胞，外压为零；达到力阈值后才使用固定的 CHGNet 0.3.0 计算单点能量。MatterSim 的能量不进入凸包或电压相图。候选即使已经经过生成阶段的优化，也会在这里按本次设置重新优化。未收敛、非有限能量或参考集缺失不能通过筛选。

Top-K、独立凸包脚本及可选全量 CHGNet 工具都支持 `--mattersim-checkpoint`、`--relax-fmax`、`--relax-steps`。电压脚本复用凸包阶段保存的结构与能量快照，并核对优化设置和 CIF 文件；修改优化设置后需重跑凸包。Web 的单独 Top-K 和全流程使用相同默认值，API 的 `Top300Request` 可设置 `mattersim_checkpoint`、`relax_fmax`、`relax_steps`。

## 3. 从已有 relaxed.extxyz 重跑

不会重新生成结构。screen 与 Top-K dry-run 读取历史 `relaxed.extxyz`；实际体相筛选时会对候选和 MP 竞争结构统一重新优化。将 `--base` 指向包含历史 `relaxed.extxyz` 的目录；输出写入本次 `RESULTS_ROOT`：

```bash
python workflow/pipeline/screen_all_extxyz.py \
  --workdir "$MATTERGEN_ROOT" --base "$RESULTS_ROOT" \
  --required-elements Li Nb O Cl --allowed-elements Li Nb O Cl \
  --light-oxy 0.05 0.35 \
  --out "$RESULTS_ROOT/stage2_candidates.csv" \
  --screened-out "$RESULTS_ROOT/screened_out.csv" \
  --refs-out "$RESULTS_ROOT/screen_refs.txt"

python workflow/pipeline/run_top300_pipeline.py \
  --workdir "$MATTERGEN_ROOT" \
  --stage2-csv "$RESULTS_ROOT/stage2_candidates.csv" \
  --output-dir "$RESULTS_ROOT/top300_run" \
  --topk 300 --selection-mode diverse --dry-run
```

检查 `selection_audit.csv` 和导出 CIF 后，重跑第二条命令并去掉 `--dry-run`，计算体相与电压结果。附加工作电压条件使用 `--target-voltage 4.5 --min-voltage-window 1.0`。

旧分数不兼容当前 `score_kind=li_periodic_geometry_proxy_v1`，必须重跑 screen 后再运行 Top-K。screen 的 `--topk` 只限制便于查看的参考列表；stage2 CSV 保留全部通过候选。通用 screen CLI 默认仅要求 Li、没有元素白名单，本项目的示例显式锁定 Li/Nb/O/Cl。

需要单独生成或重新评估时：

```bash
WORKDIR="$MATTERGEN_ROOT" BASE_RESULTS_DIR="$RESULTS_ROOT/generated" \
  CHEMICAL_SYSTEMS='Li-Nb-O-Cl' BATCH_SIZE=8 NUM_BATCHES=1 \
  bash workflow/pipeline/generate.sh

WORKDIR="$MATTERGEN_ROOT" ROOT="$RESULTS_ROOT/generated" \
  LOGDIR="$RESULTS_ROOT/logs_eval" RECURSIVE=1 \
  bash workflow/pipeline/evaluate.sh
```

## 4. 查找结果与筛选依据

| 路径（相对 `RESULTS_ROOT`） | 内容 |
| --- | --- |
| `_segments/`、`logs_eval/` | 全流程生成、评估结构与日志 |
| `stage2_candidates.csv`、`screened_out.csv` | 化学通过候选、拒绝及错误理由 |
| `top300_run/selection_audit.csv`、`top300_run/selection_summary.json` | 去重/选择逐条审计、计数与匹配参数 |
| `top300_run/exported_300cifs/` | 本次入选 CIF 与 `export_300index.csv` |
| `top300_run/relaxation/` | 双方优化后的 CIF、优化审计及共用的 `reference_entries.json` 相图快照 |
| `top300_run/chgnet_hull_top300.csv` | 体相凸包结果，筛后结果另存 `_filtered.csv` |
| `top300_run/chgnet_voltage_window_top300.csv` | 电压结果、全部稳定区间、失败状态 |
| `top300_run/voltage_filter_audit.csv` | 体相通过候选的电压接受/拒绝理由 |
| `top300_run/pre_novelty_candidates.csv` | 通过体相与电压筛选的新颖性比对输入 |
| `top300_run/novelty_filter_audit.csv` | 每个候选的训练/参考集匹配编号、数据划分、状态与错误 |
| `top300_run/novelty_summary.json` | 数据文件 SHA256、划分覆盖、比对参数及通过/拒绝数量 |
| `top300_run/final_candidates.csv` | 同时通过体相、电压和训练/参考集新颖性筛选的候选；显式跳过新颖性时标记 `not_checked` |

化学检查默认开启，异常或未知状态不会通过。`f_o=O/(O+F+Cl+Br+I)` 必须在设置区间内。默认 Li 为 +1、Nb 可取 +3/+4/+5、O 为 −2、卤素为 −1；支持固定计量下的混合价，CLI 可用 `--oxidation-states` 覆盖。切换体系时同步修改生成与 required/allowed 元素；用 `--no-light-oxy`、`--no-charge-balance`、`--no-smact` 显式关闭对应规则。关闭独立电中性检查时，启用的 SMACT 仍检查电中性。

周期 Li 图区分有限团簇/孤立环（秩 0）和沿 1/2/3 个独立晶格方向贯通的分量。`quick_score` 是按 Li 数量加权并归一化的几何贯通代理，不代表电导率或迁移势垒。Top-K 前用真实结构去重，默认按约化组成轮流选取；`score` 模式也会去重。旧 `--super` 参数保留兼容，计算不依赖有限超胞复制。

体相筛选默认为 0.05 eV/atom。电压扫描默认为 0–6 V、步长 0.05 V，容差为 0.001 eV/non-Li atom。`stable_intervals_json` 保留分离区间；主结果取最宽区间，同宽取低电压区间。`*_boundary_bracket` 给出采样边界夹区，`*_bound_censored` 表示扫描端点截尾，真实边界尚未确定。

`window_status` 区分 `stable_window`、`scan_censored`、`no_stable_window`、`calculation_failed`。电压筛选要求成功、有限且非零的稳定区间、满足最小宽度；截尾区间可通过并保留标记。指定 target 时额外在精确电压计算并要求稳定。失败结果保留错误，不充当分解边界。没有候选通过时写出空的本次结果。

新颖性筛选在电压筛选之后执行，复用 MatterGen 的 `DisorderedStructureMatcher` 和 `get_matches`，遵循其有序/无序结构、原胞和超胞匹配判定。默认容差为 `ltol=0.2`、`stol=0.3`（归一化位移容差）、`angle_tol=5°`。数据读取按候选的真实元素体系筛选，结构是否等价由 MatterGen 判断；摘要记录实际匹配器模块、版本、代码路径及参数。匹配到任一数据源的结构被淘汰；只有全部所需数据源覆盖完整且未找到匹配的候选进入最终表。数据缺失、仅有组成而没有结构、损坏结构或比较失败均不能通过，流程保留审计并报错。ZIP 中的 `ref.csv` 若只有元数据，不会充当结构参考集。

`run_top300_pipeline.py` 可重复传入 `--novelty-training-data` 和 `--novelty-reference-data` 以检查多个数据源。默认路径为 `--workdir` 下的 `data-release/alex-mp/alex_mp_20.zip` 和 `reference_MP2020correction.gz`，也可传其他本地训练 ZIP/CSV、参考 CSV 或 LMDB/.gz。参考 LMDB 解压采用流式临时文件，需预留解压磁盘空间；不会安装 MatterGen、调用 GPU 或重新松弛候选。`--skip-novelty` 只用于显式跳过，跳过输出的 `passes_novelty_filter=False`。

此步骤验证的是“相对于指定数据文件及匹配参数未找到已有结构”，不等同于完成最新数据库或文献检索。训练覆盖是指定公开数据划分，未证明某个检查点训练时每条记录的实际使用情况。

表中的 `path` 指向实际计算能量的优化后 CIF，`source_path` 保留输入来源；`relaxation_settings_json` 和 `relaxation_status` 记录优化设置与状态。电压直接复用上述快照，不能把旧凸包 CSV 和新流程混用；旧结果需从 Top-K 重新计算。正式重跑会先清除原最终通过表，避免失败后误读上一次结果。

统一 MatterSim 优化解决候选与竞争结构处理不一致的问题。统一 DFT 校准与充分的输运验证尚未接入；最终候选用于下一步计算验证。

## 5. 绘图与指标汇总

电压图建议直接使用最终候选表：

```bash
python workflow/analysis/plot_voltage_window.py \
  --csv "$RESULTS_ROOT/top300_run/final_candidates.csv" \
  --out "$RESULTS_ROOT/top300_run/voltage_window.png"

python workflow/analysis/merge_metrics_to_single_json.py \
  --root "$RESULTS_ROOT/_segments" \
  --out "$RESULTS_ROOT/combined_metrics.json" --weighting successful
```

## 6. 可选 MD 与轨迹后处理

这些工具需单独运行，不属于默认筛选 gate。MD 只接受带通过审计的最终候选，默认读取本仓库 `results/top300_run/final_candidates.csv`，输出到 `results/transport/`。优先使用表中 `path` 指向的优化后 CIF；`--cif-dir` 只用于没有 `path` 的旧表，已记录的优化文件丢失时会跳过并报原因。默认 700 K、10 ps 仅用于探索；自定义结果目录时显式传输入和输出。下面只是参数用法示例，采样长度与重复数需按材料的扩散行为制定。

```bash
mkdir -p "$RESULTS_ROOT/transport"
python workflow/transport/compute_ionic_conductivity.py \
  --csv "$RESULTS_ROOT/top300_run/final_candidates.csv" \
  --cif-dir "$RESULTS_ROOT/top300_run/exported_300cifs" \
  --temperatures 700,800,900 --total-ps 100 --equil-ps 20 --runs 3 \
  --traj-dir "$RESULTS_ROOT/transport/md_traj" \
  --out "$RESULTS_ROOT/transport/chgnet_ionic_conductivity.csv" \
  --arrhenius-summary "$RESULTS_ROOT/transport/chgnet_ionic_conductivity_arrhenius.csv"

python workflow/transport/plot_arrhenius.py \
  --input "$RESULTS_ROOT/transport/chgnet_ionic_conductivity_arrhenius.csv" \
  --output "$RESULTS_ROOT/transport/arrhenius_plots"

python workflow/transport/plot_msd_from_traj.py \
  --traj /absolute/path/to/one_trajectory.traj --timestep-fs 2 --log-interval 10 \
  --plot "$RESULTS_ROOT/transport/msd.png" --csv "$RESULTS_ROOT/transport/msd.csv"

python workflow/transport/li_density_from_traj.py \
  --traj /absolute/path/to/one_trajectory.traj \
  --output "$RESULTS_ROOT/transport/li_density.cube"
```

MSD 中的帧间隔必须与实际 MD 输出一致；多轨迹输入会按物种平均，应只合并同一材料、同一温度的独立重复。Li 密度统计应使用相容晶胞的同一材料轨迹，不混合不同候选。

## 7. 可选工具

- `tools/compute_hull_from_relaxed.py`：直接对全部 relaxed 帧算凸包。CHGNet 模式用 `--root "$RESULTS_ROOT" --mode chgnet --out "$RESULTS_ROOT/all_hull.csv"`，同样统一优化双方，优化结果保存到输出旁的 `relaxation/all_frames/`。MP 模式用 `--mode mp --energies-csv dft_energies.csv`，CSV 只读取 `id` 和 `energy_eV`；使用前需确认 DFT 能量及兼容性元数据是否可比，该入口不接收完整校正元数据。
- `tools/eval_filtered_structs.py`：对指定 CSV 的 CIF 再评估并汇总指标，例如 `--csv "$RESULTS_ROOT/top300_run/final_candidates.csv" --out-dir "$RESULTS_ROOT/re_evaluated" --summary-csv "$RESULTS_ROOT/re_evaluated/metrics_summary.csv"`。
- `tools/extxyz_to_cif.py`：任意 extxyz 帧转 CIF，例如 `--input /absolute/path/relaxed.extxyz --frames 0 2 --outdir "$RESULTS_ROOT/manual_cifs"`。

各工具可用 `python workflow/<目录>/<脚本>.py --help` 查看参数。

## 整理说明

项目实现统一放在 `workflow/`，没有另设 legacy 副本。已删除 `mattergen/` 根目录的重复导出、凸包、电压及筛选入口，删除旧 `dd.sh`、`dd_single.sh`、`eval_all.sh`、`run_chgnet_phase.py`、`aggregate_metrics_table.py`，并移除 Web 的历史 `analyze_results200_distribution.py` 和容器流程别名。仍有独立用途的分析、MD、转换和再评估工具按上面的目录保留。

旧文件可用 `git log --all -- 原路径` 查询历史，或用 `git show 提交号:原路径` 查看。旧命令请改为本文路径。

## 离线检查

在单独的轻量测试环境中安装 `requirements-screening-test.txt`，运行 `python -m pytest -q tests/`。回归测试使用合成结构和纯相图，生成/评估调用使用桩命令，不需要模型权重、GPU 或 MP 密钥；它们不验证实际模型预测质量。
