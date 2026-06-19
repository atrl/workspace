# analysis 框架契约设计

日期：2026-06-19

## 背景

当前 SOXL 工作区已经把可复用能力迁入 `src/`，研究和生产方案通过
`research/<experiment>/<experiment>.ipynb` 与
`production/<strategy>/<strategy>.ipynb` 留下中文证据。强化学习不再进入
v1 架构，原 reward 概念由 `src/objective` 中的分类、MSE、IC、ranking、
收益风险 utility 和晋升评分承接。

`src/analysis` 已经具备 qlib recorder 归一化、market regime 生成、
window/regime 交叉对比、模型差异矩阵、模型诊断、中文 review briefing、
failure modes 和 manifest export。当前需要补齐的是稳定高层入口和中文
review 规范，让 notebook、dashboard、schedule 和 future production 调用同一
个分析契约，而不是继续依赖散落函数或旧 skill 里的项目状态。

## 目标

1. `src/analysis` 提供稳定高层 API，用于生成模型在不同时间区间和 regime 下的
   效果差异报告。
2. qlib recorder、本地 daily backtest、market panel 和手工 window 定义进入同
   一套 `AnalysisReport`/`analysis_manifest.json` 输出。
3. dashboard 和 schedule 只读取 manifest/export，不重新计算分析，不直接读取散
   落实验文件。
4. skill 只保留方法论、发散假设、regime 候选和解释提示，不再保存路径、模型结
   果、固定参数或项目状态。
5. 每次新增稳定 analysis 能力必须有聚焦测试和一个中文 notebook 证据面。

## 非目标

- 不新增强化学习、reward trainer 或 RL 训练入口。
- 不把单次实验参数硬编码进 `src/analysis`。
- 不让 dashboard/schedule 读取 notebook 内部变量或私有 sidecar。
- 不在本轮重写已有 `reports.py`、`qlib_ext.py`、`regimes.py` 的底层计算逻辑。
- 不迁移历史 runtime/card 结构；旧 skill 中的路径只能作为历史参考。

## 推荐架构

新增一个稳定门面模块：

```text
src/analysis/pipeline.py
```

导出的核心对象：

```text
AnalysisSpec
build_model_regime_comparison_report
```

`AnalysisSpec` 负责保存可复用分析契约，不保存单次研究结论：

- `artifact_id`
- `baseline_model`
- `windows`
- `freq`
- `market_regime_config`
- `summary_note`
- `source_type`
- `schema_version`

`build_model_regime_comparison_report(...)` 负责把已有稳定能力组合成一站式报告：

- daily backtest 输入：调用 `analyze_model_effects_by_regime_and_window` 和
  `build_regime_window_analysis_report`。
- market panel 输入：调用 `build_market_regime_frame` 与 `attach_market_regimes`
  后再生成 report。
- qlib recorder 输入：调用 `load_qlib_recorder_analysis` 和
  `build_qlib_recorder_regime_window_analysis_report`。
- 输出统一 `AnalysisReport`，再由 `write_analysis_report` 写出公开 sidecar 和
  `analysis_manifest.json`。

这个门面不替代底层函数。底层函数继续承担具体计算；门面承担契约编排、默认
字段、summary 归一化和 notebook/dashboard/schedule 之间的稳定入口。

## 数据流

```text
qlib recorder 或 daily_backtest
        |
        v
src/analysis 归一化输入表
        |
        +-- market panel -> market_regimes -> attach_market_regimes
        |
        v
regime/window/all/window_regime 分段指标
        |
        +-- qlib_return_frame
        +-- qlib_risk_metrics
        +-- segment_effect_matrix
        +-- model_effect_deltas
        +-- model_regime_window_diagnostics
        +-- model_review_briefing
        +-- failure_modes
        |
        v
AnalysisReport -> write_analysis_report -> analysis_manifest.json
        |
        +-- dashboard 只读 manifest/export
        +-- schedule 只读 overview/briefing/failure mode 摘要
        +-- notebook 写中文结论和晋升判断
```

## qlib 扩展边界

qlib 原始 artifact 不直接暴露给 dashboard 或 production。所有 qlib 结果必须先
通过 `src/analysis` 归一化成稳定表：

- portfolio report 归一化为 `date/model/strategy_return/benchmark_return/
  excess_return/nav/benchmark_nav`。
- qlib risk analysis 归一化为 `model/metric/value/return_kind/source_artifact`。
- positions、orders、indicator analysis 作为可选公开表，路径必须进入
  `analysis_manifest.json`。

如果 qlib 版本或 recorder 结构变化，只允许在 `src/analysis` 适配层处理，不应让
dashboard、schedule 或 notebook 复制路径猜测逻辑。

## Regime 和区间分析

稳定框架内置的 regime 维度包括：

- VIX 高低。
- 趋势上行、下行、震荡。
- 回撤深度和压力段。
- 实现波动率。
- 成交量冲击。
- SOXL 相对 SOXX 或 underlying 的强弱。

稳定 window 维度包括：

- YTD。
- trailing 1/3/6/12 个月。
- 2020、2022、2023-2024 等压力或结构性窗口。
- notebook 传入的自定义窗口。

框架输出必须同时保留：

- 全区间模型比较。
- regime 内比较。
- window 内比较。
- window 与 regime 交叉比较。
- 相对 baseline 的收益、回撤、稳定性和换手差异。

## 模型差异解释

模型差异不能只看收益。稳定 report 至少包含：

- `model_comparison.csv`
- `regime_metrics.csv`
- `window_metrics.csv`
- `window_regime_metrics.csv`
- `segment_effect_matrix.csv`
- `model_effect_deltas.csv`
- `model_regime_window_diagnostics.csv`
- `model_review_briefing.csv`
- `failure_modes.csv`
- `analysis_summary.json`
- `analysis_manifest.json`

中文 review 规则：

- `promote_candidate` 只是分段候选，不等于全局晋升。
- 同一模型如存在 `block_or_rework`，需要优先处理压力区间弱点。
- 高收益但回撤或换手明显扩大时，必须标记风险或成本复核。
- regime 优势过度集中时，只能作为分段模型或组合候选。

## Skill 边界

skill 可以做：

- 提醒研究者如何提出假设。
- 发散候选 regime 或分段。
- 建议如何解释模型差异。
- 提醒需要检查的风险、成本、样本量和未来函数。

skill 不可以做：

- 保存项目路径、模型结果、固定参数或当前状态。
- 作为 dashboard、schedule、production 的功能入口。
- 直接生成业务 sidecar 或 manifest。
- 替代 `research/` 或 `production/` notebook 的中文结论。

如果 skill 中出现具体路径、模型结果、固定参数或项目状态，应迁回：

- notebook 的参数和结论 cell。
- `src/core` manifest 或 schema。
- `src/analysis` 稳定框架模块。

## Notebook 证据面

新增或修改稳定 analysis 能力时，必须选择或创建一个 notebook 作为主要证据面。
本设计对应的实施证据面为：

```text
research/analysis_framework_contract_v1/analysis_framework_contract_v1.ipynb
```

notebook 至少验证：

- 使用 `AnalysisSpec` 构造一次完整分析。
- daily backtest + market panel 可以输出完整 `AnalysisReport`。
- `analysis_manifest.json` 包含公开 sidecar，不暴露私有实验文件。
- `analysis_overview`、`model_review_briefing`、`window_regime_metrics` 可被
  dashboard/schedule 消费。
- 最后中文结论 cell 记录：做了什么、核心指标、regime 差异、是否晋升、风险、
  下一步。

## 测试计划

聚焦测试新增：

- `tests/test_analysis_framework_contract.py`
  - `AnalysisSpec` 序列化稳定。
  - daily backtest 路径生成标准 report。
  - market panel 路径会重算 market regime 并覆盖 legacy regime。
  - qlib recorder 路径继续输出 recorder 原始公开表和统一 model diff 表。
  - report summary 包含中文 review note、baseline、window definitions 和
    analysis overview。

回归测试运行：

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests/test_analysis_qlib_extensions.py tests/test_analysis_report_export.py tests/test_analysis_market_regime_report.py tests/test_analysis_qlib_recorder_adapter.py tests/test_dashboard_schedule_api_pipeline.py tests/test_schedule_model_dashboard_pipeline.py -q
```

完成后再运行仓库全量测试：

```bash
PYTHONPATH=. /Users/rziiiii/dev/workspace/qlib_venv/bin/python -m pytest tests -q
```

## 迁移顺序

1. 新增 `AnalysisSpec` 和高层 report builder，先复用现有底层函数。
2. 为新门面补聚焦测试，先看 RED，再写最小实现。
3. 创建并执行 `research/analysis_framework_contract_v1/analysis_framework_contract_v1.ipynb`。
4. 更新 `src/analysis/README.md`，把高层入口和 skill 边界写清楚。
5. 如旧 skill 仍包含 runtime/card 或项目状态，另开一轮只保留方法论内容。

## 风险

- 门面 API 如果过早抽象太多输入类型，会变成第二套框架。v1 只支持当前已有的
  daily backtest、market panel 和 qlib recorder 三类输入。
- regime 阈值必须通过 config 传入，不能在 production notebook 之外偷偷改变。
- dashboard/schedule 只能消费 manifest/export；任何绕过 manifest 的快捷读取都
  会重新制造散落状态。
- 当前工作区存在未完成的模型适配测试改动；analysis 提交必须避免混入无关文件。

## 验收标准

- 设计文档提交到仓库，并使用中文描述 analysis 框架边界。
- 用户 review 通过后，实施阶段必须提供：
  - 新增或更新的稳定 `src/analysis` API。
  - 聚焦测试和全量测试结果。
  - 可执行 notebook 证据。
  - 更新后的 `analysis_manifest.json`/sidecar 示例。
  - 不混入未完成模型适配或历史 untracked 产物的 commit。
