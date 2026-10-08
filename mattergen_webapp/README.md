# Web 控制台

Web 只负责参数表单、后台作业和日志查看，实际调用仓库 [`workflow/pipeline/`](../workflow/README.md) 中的统一脚本。默认目标为 `Li-Nb-O-Cl`。

## 启动

按 [根 README](../README.md#安装) 安装完整环境后，从 **仓库根目录** 执行：

```bash
source mattergen/.venv/bin/activate
export MATTERGEN_ROOT="$PWD/mattergen"
export WORKFLOW_ROOT="$PWD/workflow"
export RESULTS_ROOT="$PWD/results"
export MATTERGEN_RUNTIME_ROOT="$PWD/_runtime"
uvicorn main:app --app-dir mattergen_webapp/backend --host 0.0.0.0 --port 8000
```

浏览器访问 `http://localhost:8000`。需要体相/电压计算时，在启动前设置 `MP_API_KEY`。模型权重和运行结果不在 Git 中，需要单独准备。

| 页面操作 | 实际流程 |
| --- | --- |
| 生成 | `workflow/pipeline/generate.sh` |
| 评估 | `workflow/pipeline/evaluate.sh` |
| 筛选 | `workflow/pipeline/screen_all_extxyz.py` |
| Top-K | `workflow/pipeline/run_top300_pipeline.py` |
| 一键全流程 | 分段生成，随后统一评估、筛选与 Top-K |

`MATTERGEN_ROOT` 指向核心框架，`WORKFLOW_ROOT` 指向项目工作流文件夹，`RESULTS_ROOT` 是运行结果目录。使用自定义布局或迁移已有结果时，将这些变量设置为对应的绝对路径。

## 使用

默认开启电中性、SMACT 和氧比例过滤，并要求 Li/Nb/O/Cl 全部存在、没有其他元素。Top-K 默认先去重再按组成轮流选取。可选工作电压留空时不附加该电压条件；指定后要求精确点评估稳定。

已有 `relaxed.extxyz` 时直接使用“筛选”和“Top-K”，无需重新生成。历史 stage2 CSV 必须重跑 screen，生成 `score_kind=li_periodic_geometry_proxy_v1`。仅检查选择和 CIF 导出可启用 Top-K 的 dry-run；一键全流程的 dry-run 仍会生成和评估。

详细规则、CSV 字段、结果位置、绘图和 MD 用法统一见 [操作指南](../workflow/README.md)。最终候选为 `RESULTS_ROOT/top300_run/final_candidates.csv`，选择和电压审计放在同目录。

作业状态保存在服务内存中，日志位于 `mattergen_webapp/backend/job_logs/`，页面可查看尾部日志。提交会直接启动子进程；当前未实现并发队列控制。共享部署请在外部配置认证。

部署到其他机器时克隆或上传整个 LiNbOCl 仓库，保留 `workflow/`、`mattergen/` 和 `mattergen_webapp/` 的相对布局。集群见 [HPC 说明](hpc/README_HPC.md)，GPU 主机见 [容器说明](container/README_CONTAINER.md)。
