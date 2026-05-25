---
name: leveraged-etf-factor-mining
description: 3x杠杆ETF单标的时序特征（因子）挖掘、IC检验、信号回测、SHAP验证与因子库管理。默认标的SOXL，支持TQQQ/UPRO/SPXL/TECL等3xETF。独立于RL训练流程，专门负责从rulebase逻辑、研报和假设中挖掘有效特征，验证通过后入库供RL/Rulebase使用。
version: 2.1.0
changelog:
  - 2.1.0: 整合 evaluate_factor_pack.py 通用评估脚本；新增 SHAP 验证步骤；更新可复用脚本清单
  - 2.0.0: 重命名并适配多ETF，新增--symbol参数说明
  - 1.0.0: SOXL专用初始版本
---

# 3x杠杆ETF特征挖掘（Factor Mining）

本skill负责3x杠杆ETF单标的择时场景的时序特征挖掘与验证子工作流。
默认标的为SOXL，通过`--symbol`参数可切换至TQQQ/UPRO/SPXL/TECL等其他3xETF。

## 与agentic-alpha的差异

| 维度 | agentic-alpha SFA | Leveraged-ETF Factor Mining |
|------|-------------------|---------------------------|
| 标的 | 股票多因子 (CSI1000) | 单ETF择时 (SOXL/TQQQ/UPRO/SPXL/TECL) |
| IC类型 | 截面Rank IC | 时序IC |
| 回测 | Top-N组合调仓 | 信号方向回测 |
| 数据 | Qlib Expression面板 | Pandas Series时间序列 |
| 验证 | IC+回测 | IC+回测+SHAP |

## 核心工作流 (R-G-E-V-D)

Notebook-first rule:
- 单次因子假设、IC/回测/相关性结论默认写入
  `runtime/factor/card/<factor_or_pack>/<factor_or_pack>.ipynb`。
- CSV/Markdown/JSON 只作为 notebook 的 sidecar evidence。不要把一次性结论只写
  到 notes.md、knowledge.md 或 chat。
- 只有 promotion、长期路由变化、或用户明确要求时，才更新
  `runtime/factor/knowledge.md`。

### 1) Research
1. 读取`AGENTS.md`和`runtime/factor/knowledge.md`中的历史经验。
2. 运行`scripts/extract_rulebase_signals.py`从`soxl_backtest/`提取rulebase信号假设。
3. 生成特征假设卡片（YAML）。

### 2) Generate
1. 将假设转化为`src/soxl_quant/data.py`中的特征表达式（新增 feature pack）。
2. 使用`engineer_feature_frame(symbol=...)`生成特征序列。
3. 避免与已有`CORE_FEATURE_COLUMNS`高度相关的构造。

### 3) Evaluate（通用评估脚本）
使用`scripts/evaluate_factor_pack.py`完成一站式评估：

```bash
python scripts/evaluate_factor_pack.py --pack <pack_name> \
    [--features FEAT1 FEAT2] \
    [--horizons 1 5 10 20] \
    [--thresholds -1.0 -0.5 0.0 0.5 1.0 1.5]
```

该脚本自动执行：
1. **时序IC快筛** — 计算1d/5d/10d/20d时序IC
2. **信号回测** — 自动尝试 long-flat 和 short-flat，选择 Sharpe 更高的方向
3. **正交预算** — 与`CORE_FEATURE_COLUMNS`的相关系数矩阵
4. **全量判定** — 输出 CSV + 因子卡片 Markdown + 写入 `soxl_factor_library.db`

Notebook 里需要复用 gate 判定时，使用：

```python
from src.soxl_quant.app.factor_diagnostics import (
    build_factor_gate_summary,
    render_factor_gate_summary,
)
```

`build_factor_gate_summary(ic_df, backtest_df, corr_df)` 会把 IC、standalone
回测和 core 相关性合并成 `full_gate` / `primary_failure`，避免只凭单项
Sharpe 或单项 IC 下结论。

Label-discovery addendum:
- 从 fold-native target / teacher 表挖二元标签时，重复日期会跨 fold/window
  出现；不要把重复行数当作独立样本数。
- notebook 必须同时记录 `positive_rows`、`unique_positive_dates`、
  `positive_years`、`positive_span_days` 和 duplicate-date conflict rate。
- 进入训练 handoff 前，至少做一次 full-history panel sanity check；只在
  target 表里过 gate、但 full-history 只集中在单一 crisis 的候选，保留为诊断，
  不直接开 PPO。
- standalone 回测 MaxDD 过大时，结论应是 risk-capped target lift /
  execution overlay candidate，而不是 core feature 或无约束目标。

### 4) Validate
1. **SHAP 基线分析**：`python scripts/feature_analysis.py --profile <profile>`
   - 在现有模型上运行 KernelSHAP，获取各特征的 `mean_abs_shap`
   - 识别模型已经饱和的维度（SHAP 高但 IC 低的特征 = 模型已学会非线性组合）
   - 识别模型 blind spot（IC 高但 SHAP 低的特征 = 模型未充分利用）
2. **IC衰减分析**：`scripts/evaluate_factor_pack.py` 的 `--horizons` 参数覆盖多 horizon
3. **体制切片**：由`leveraged-etf-signal-backtest` skill执行

### 5) Distill
1. 入库：`scripts/evaluate_factor_pack.py` 自动写入 `data/etf_factor_library.db`
2. 更新`docs/factor_research/`和`runtime/factor/knowledge.md`
3. 新Accepted特征加入`CORE_FEATURE_COLUMNS`需单独审批

## 时序IC判定标准

| 指标 | 通过门槛 | 说明 |
|------|---------|------|
| mean_ic | > 0.03 | 时序IC均值正向（方向正确且显著）|
| ic_positive_ratio | > 0.55 | IC>0的天数占比 |
| ic_t | > 2.0 | t统计量显著 |
| signal_accuracy | > 0.55 | 信号方向正确率 |
| profit_factor | > 1.2 | 盈亏比 |
| max_correlation_with_existing | ≤ 0.80 | 与已有特征最大相关 |

## SHAP 验证标准

| 检查项 | 目的 | 通过标准 |
|--------|------|---------|
| SHAP 基线 | 了解当前模型依赖哪些特征 | 运行 `feature_analysis.py` 获取 `runtime/factor/analysis/shap_importance.csv` |
| IC vs SHAP 对比 | 识别模型 blind spot | IC 高但 SHAP 低的特征说明模型未充分利用该信息 |
| 相关性检查 | 避免与模型已高度依赖的特征冗余 | 新因子与 SHAP Top-5 特征的 |ρ| ≤ 0.80 |
| 增量预期 | 判断新因子是否能填补空白 | 新因子所属维度在 SHAP 中排名低于 median |

## 可复用脚本

| 脚本 | 功能 | 输出 |
|------|------|------|
| `evaluate_factor_pack.py` | **通用因子包评估**（IC+回测+相关性+入库） | CSV + DB + 卡片 |
| `feature_analysis.py` | **SHAP 特征重要性分析** | `shap_importance.csv` + `ic.csv` + `summary.json` |
| `extract_rulebase_signals.py` | 从rulebase提取候选特征 | YAML假设卡片 |
| `factor_db_cli.py` | 因子库CLI | 查询/管理 |

## 废弃脚本说明

以下脚本在 skill 2.0.0 中被描述，但实际从未创建，现已被 `evaluate_factor_pack.py` 统一替代：
- `test_factor_ic.py` → 功能合并到 `evaluate_factor_pack.py`
- `backtest_signal.py` → 功能合并到 `evaluate_factor_pack.py`
- `analyze_correlation.py` → 功能合并到 `evaluate_factor_pack.py`
- `test_factor_decay.py` → 使用 `evaluate_factor_pack.py --horizons` 参数
- `test_factor_stability.py` → 待实现（rolling IC 尚未合并）
- `mine_factors.py` → 待实现（全流程入口尚未合并）

## 临时脚本边界

1. 一次性脚本放在项目根目录`./scripts/temp/`，完成后删除。
2. skills下不允许存在`__pycache__` / `.pyc`。

## 参考资料

- `references/etf_thresholds.md` — 历史阈值与基准分布（按标的分类）
- `runtime/factor/analysis/shap_importance.csv` — 当前模型的 SHAP 基线
