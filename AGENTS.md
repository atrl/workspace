# AGENTS.md

本仓库是 SOXL 全平台研究与验证工作区。

目标是在当前 repo 内维护从数据、因子、模型、objective、回测、analysis、
dashboard export、schedule 到 paper API 的完整链路。可复用能力必须沉到
`src/`；单次研究和生产方案必须落在 notebook 中，便于中文 review 和复现。

不再考虑强化学习。`src/objective` 负责分类、MSE、IC、ranking、收益风险
utility 和晋升评分；不要新增 RL 训练面。

## Startup

开始实质工作前：

1. 读取 `AGENTS.md`。
2. 读取 `SPEC.md`。
3. 选择或创建一个 `research/<experiment>/<experiment>.ipynb` 或
   `production/<strategy>/<strategy>.ipynb` 作为本次工作的主要证据面。

不要把聊天记录当作持久状态。

## 核心 Surface

- `src/core`: 稳定契约、schema version、manifest、hash、配置、校验。
- `src/data`: 数据源、数据更新、qlib/calendar 对齐、canonical snapshot。
- `src/factor`: 单因子、多因子、IC、相关性、分层回测、因子组合诊断。
- `src/model`: QuantBench 模型注册、训练、预测、保存、加载、模型能力说明。
- `src/objective`: label、loss、IC/ranking objective、收益风险 utility、晋升评分。
- `src/backtest`: 信号转仓位、成本、滑点、walk-forward、压力窗口、基准对比。
- `src/analysis`: qlib analyzer 扩展、regime 分层、区间对比、模型差异解释。
- `src/dashboard`: 只读取统一 manifest/export，不直接读散落实验文件。
- `src/schedule`: 数据更新、模型刷新、dashboard export、通知、锁、重试。
- `src/api`: broker/data adapter contract；v1 只实现 paper/sim。
- `research/<experiment>/`: 研究中实验，每个实验一个主 `.ipynb`。
- `production/<strategy>/`: 晋升后的可运行方案，每个方案一个主 `.ipynb`。

## Notebook 规则

- notebook 是研究和生产方案的主要可读证据。
- 每个 notebook 最后必须有中文结论 cell，记录：做了什么、核心指标、
  regime 差异、是否晋升、风险、下一步。
- CSV/JSON 只能作为 sidecar，不能替代 notebook 结论。
- 外部知识抓取先进入 `research/knowledge_ingest/`，再经过结构化假设和
  可测试接口进入 factor/model/objective backlog。

## Skill 边界

skill 只用于抽象研究方法、启发假设、提示如何拆因子、设计 regime、解释
模型差异。skill 不实现业务逻辑、不保存项目状态、不作为功能入口。

如果 skill 中出现具体路径、模型结果、固定参数或项目状态，应迁回 notebook、
`src/core` manifest 或稳定框架模块。

## 验证

使用本仓库 Python 环境：

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests -q
```

每个文件变更任务应完成聚焦验证、commit 和 push；除非权限或网络失败。
