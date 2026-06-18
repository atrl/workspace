# src/analysis 设计边界

`src/analysis` 是模型回测之后的稳定分析层，负责把 qlib、回测曲线和
regime/window 分段结果转成统一的可审阅 artifact。skill 只用于提出分析角度、
拆 regime、解释模型差异或生成假设，不保存项目状态，也不作为 dashboard、
schedule 或 production 的功能入口。

## 稳定输入

- `daily_backtest` 风格表：至少包含 `date`、`model`、`nav`，可选
  `benchmark_nav`、`position`、`turnover`、`regime`、`vix`、`trend`、
  `drawdown`。
- qlib recorder 或本地 qlib artifact 目录：通过
  `load_qlib_recorder_analysis` 读取 `report_normal_1day`、`risk_analysis_1day`、
  positions、orders、indicator analysis 等常见 artifact。
- window 定义：通过 `build_analysis_windows` 生成 YTD、trailing window、
  历史压力年份和自定义窗口，所有窗口会裁剪到实际数据范围。

## 稳定输出

- `model_comparison.csv`：全区间模型横向比较。
- `regime_metrics.csv`：不同 regime 下的收益、回撤、稳定性和换手。
- `window_metrics.csv`：不同时间窗口的模型表现。
- `window_regime_metrics.csv`：时间窗口和 regime 交叉后的模型表现。
- `qlib_return_frame.csv` / `qlib_risk_metrics.csv`：qlib 风格收益序列和风险指标。
- `segment_effect_matrix.csv`：每个 segment 的最强、最弱、回撤最高、换手最高模型。
- `model_effect_deltas.csv`：相对基准模型的分段差异。
- `model_review_briefing.csv`：中文 review 行动项，用于 dashboard 和 schedule。
- `failure_modes.csv`：从分段差异派生的失败模式。
- `analysis_summary.json` / `analysis_manifest.json`：公开 artifact 和 schema 入口。

## qlib 接入方式

优先使用 qlib 自带的 recorder artifact 作为输入，再在 `src/analysis` 内统一成
框架表。这样 notebook 可以复现 qlib 原始结果，dashboard 和 schedule 不需要
知道 recorder 的内部路径。

常用路径：

1. `load_qlib_recorder_analysis(...)` 标准化单个模型的 qlib artifact。
2. `build_qlib_recorder_regime_window_analysis_report(...)` 合并多个模型并生成
   regime/window 对比报告。
3. `write_analysis_report(...)` 写出 summary、CSV sidecar 和 manifest。

## 扩展原则

- 新增分析能力先落到 `src/analysis`，再用 `research/<experiment>/<experiment>.ipynb`
  留中文证据。
- dashboard 只读 `analysis_manifest.json` 暴露的公开表，不直接读取预测明细或单次实验私有文件。
- schedule 只读取 `analysis_overview`、`model_review_briefing`、`failure_modes`
  等摘要，不重新计算分析。
- skill 可以提出新的 regime 切分、异常诊断或模型差异解释，但可复用代码必须迁入
  `src/analysis`，单次参数和结论必须留在 notebook。
