# MatterGen 逆向设计 Web 控制台

基于 MatterGen 的生成、评估、筛选和精选导出脚本封装的轻量 Web UI，支持参数化运行、后台作业与尾部日志查看。默认目标体系为 `Li-Nb-O-Cl`。默认假设 `mattergen` 仓库在本项目上级目录（`../mattergen`），可通过环境变量 `MATTERGEN_ROOT` 覆盖。

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

后端调用 `scripts/` 下的脚本，并通过 `WORKDIR`/`--workdir` 指向 MatterGen 执行目录。`mattergen/screen_all_extxyz.py` 和 `mattergen/run_top300_pipeline.py` 是兼容入口，转发到这套筛选实现。

## 功能说明

- **dd.sh**：批量生成候选材料，可显式传入 `CHEMICAL_SYSTEMS`/`CHEMICAL_SYSTEMS_FILE`，若未指定则按 `ELEMENTS` 与 `COMBO_SIZES` 自动组合；自动组合默认使用 `Li Nb O Cl` 的四元体系。输出根目录 `BASE_RESULTS_DIR` 可自定义（支持绝对路径）。
- **eval_all.sh**：对指定 ROOT 下的子目录执行 `mattergen-evaluate`，写入 `metrics.json` / `relaxed.extxyz`。
- **screen_all_extxyz.py**：递归读取 `relaxed.extxyz`，执行组成和化学过滤、计算周期 Li 图特征，输出全部通过的 `stage2_candidates.csv`、拒绝审计和几何分数排序的参考列表。此处的 `--topk` 只限制参考列表，不截断 stage2 CSV。
- **run_top300_pipeline.py**：先去除等价结构，再选最多 Top-K 个候选、导出 CIF，计算 CHGNet/MP 体相稳定性和电压窗口，最后写出同时通过两道筛选的 `final_candidates.csv`。`--dry-run` 只执行去重、选择和 CIF 导出。
- **作业管理**：所有任务后台线程运行，状态记录在内存，日志保存在 `backend/job_logs`，前端可点击行查看尾部日志。

## 参数 & 环境

- `MATTERGEN_ROOT`：若 `mattergen` 仓库不在默认位置，可在启动服务前设置，例如  
  `MATTERGEN_ROOT=/path/to/mattergen uvicorn backend.main:app --app-dir backend`
- Web 和 HPC 默认要求包含 `Li Nb O Cl`，并将允许元素限制在这四种。通用 `screen_all_extxyz.py` CLI 默认只要求 `Li`，无元素白名单；用于本项目时应显式传入 `--required-elements Li Nb O Cl --allowed-elements Li Nb O Cl`。

## 筛选规则与审计

化学筛选默认开启固定计量式电中性检查和 SMACT Pauling 规则。检查异常、未知状态或结构读取失败不会被当成通过；拒绝理由写入 `screened_out.csv`。启用 SMACT 而未安装依赖时，脚本直接报错。默认离子态使用 Li `+1`、Nb `+3/+4/+5`、O `-2`、卤素 `-1`，支持在固定计量下的混合价；可用 `--oxidation-states Nb=3,4,5` 等参数覆盖。

`--light-oxy 0.05 0.35` 实际过滤 `f_o = O / (O + F + Cl + Br + I)`，上下界均包含。用于其他体系时，可显式设置 `--no-light-oxy`、`--no-charge-balance` 或 `--no-smact`；关闭独立电中性过滤后，启用的 SMACT 仍会检查电中性。

Li 图使用 `--r-cut` 内的周期近邻及晶格平移信息。`li_percolation_dim` 是贯通分量的最大平移秩：0 表示有限团簇或孤立环，1/2/3 分别表示沿一/二/三个独立晶格方向贯通。`li_conn` 和 `li_percolation_fraction` 是处于贯通分量中的 Li 比例。`quick_score` 为各分量的 Li 数量与秩加权后除以 `3 × Li 总数`，范围 0–1，仅是几何代理（geometry proxy），不代表离子电导率或迁移势垒。旧的 `--super` 参数保留兼容，周期图计算不再依赖有限超胞复制。

Top-K 选择用真实源结构的 `StructureMatcher` 去重，包含平移和超胞等价结构。默认 `--selection-mode diverse` 按约化组成轮流取候选，每组内部按有限几何分数降序排列；`--selection-mode score` 按分数选取，也会先去重。`selection_audit.csv` 记录入选、重复、超过配额、无效分数和结构读取错误及其原因；`selection_summary.json` 保存计数及匹配参数。CIF 索引只包含本次入选结构，后续体相计算按此索引读取，避免混入目录中的旧 CIF。化学筛选没有通过候选时，后续阶段跳过并写出空的本次结果。

新的 stage2 CSV 必须包含 `score_kind=li_periodic_geometry_proxy_v1`。历史 CSV 的旧分数不再兼容，应从已有 `relaxed.extxyz` 重新运行 screen，再重跑 Top-K；无需仅为此重新生成结构。

## 电压窗口与最终结果

体相阈值默认仍为 `--ehull-threshold 0.05` eV/atom。电压稳定性容差默认改为 `--voltage-threshold 0.001` eV/non-Li atom，二者的含义和归一化单位不同。

电压扫描默认为 0–6 V、步长 0.05 V（相对 Li/Li⁺）。`stable_intervals_json` 保留所有分离的稳定区间；主结果 `V_red`/`V_ox` 取最宽区间的首/末稳定采样点，宽度相同则取低电压区间。`lower_boundary_bracket`/`upper_boundary_bracket` 给出相邻稳定/不稳定点形成的边界夹区；到达扫描端点仍稳定时标为 `*_bound_censored`，表示真实边界尚未确定。

`window_status` 区分 `stable_window`、`scan_censored`、`no_stable_window` 和 `calculation_failed`。计算失败保留错误信息，不充当分解边界。最终筛选要求计算成功、有有限且非零的已采样稳定区间，并满足 `--min-voltage-window`（默认 0 V）。扫描截尾结果可通过这一条件，但仍保留截尾标记。

`--target-voltage` 默认不设置。指定后会在该精确电压额外计算，不受扫描步长限制，并要求 `stable_at_target=True`。`final_candidates.csv` 保存同时通过体相和电压筛选的候选；`voltage_filter_audit.csv` 记录每个体相通过候选的电压接受或拒绝理由。未通过或失败结果可在原始电压 CSV 和审计中追查。

此次修复覆盖化学过滤、周期 Li 图、Top-K 选择和电压筛选。统一 DFT 校准与充分的 MD/输运验证尚未接入，最终列表仍用于进一步计算验证。

## 命令行示例

从已有评估结构重新筛选，并先检查去重、选择和导出：

```bash
export MATTERGEN_ROOT=/path/to/mattergen
export RESULTS_ROOT=/path/to/mattergen_runs/results
python scripts/screen_all_extxyz.py \
  --workdir "$MATTERGEN_ROOT" --base "$RESULTS_ROOT" \
  --required-elements Li Nb O Cl --allowed-elements Li Nb O Cl \
  --light-oxy 0.05 0.35 \
  --out "$RESULTS_ROOT/stage2_candidates.csv" \
  --screened-out "$RESULTS_ROOT/screened_out.csv" \
  --refs-out "$RESULTS_ROOT/screen_refs.txt"

python scripts/run_top300_pipeline.py \
  --workdir "$MATTERGEN_ROOT" \
  --stage2-csv "$RESULTS_ROOT/stage2_candidates.csv" \
  --output-dir "$RESULTS_ROOT/top300_run" \
  --topk 300 --selection-mode diverse --dry-run
```

完整计算时去掉 `--dry-run`，并设置 `MP_API_KEY`。例如需要在 4.5 V 稳定且已采样窗口至少 1 V 时，附加 `--target-voltage 4.5 --min-voltage-window 1.0`；未指定工作电压时省略 `--target-voltage`。

## 离线回归测试

在仓库根目录安装轻量测试依赖并运行；无需模型权重、GPU 或 MP 密钥：

```bash
python -m pip install -r requirements-screening-test.txt
python -m pytest -q tests/
```

测试使用合成晶体和纯 pymatgen 相图，覆盖化学、周期图、候选选择、电压边界和 CLI/HPC 参数。HPC 测试中的生成与松弛使用桩命令，筛选、去重和 CIF 导出执行真实脚本；这些测试不验证 CHGNet 预测准确性或实际生成质量。

## 已知限制

- 未实现并发队列控制，提交后会直接启动子进程。
- 前端未做身份/权限校验；在共享环境中部署时请加上反向代理或认证。
- 运行 eval/screen/top300 依赖对应脚本所需的第三方包（CHGNet、pymatgen、SMACT 等）已正确安装，否则会在日志中失败。
