---
name: soxl-factor-mining
description: SOXL 单标的时序特征（因子）挖掘、IC 检验、信号回测与因子库管理。独立于 RL 训练流程，专门负责从 rulebase 逻辑和假设中挖掘有效特征，验证通过后入库供 RL / Rulebase 使用。
---

# SOXL 特征挖掘（Factor Mining）

本 skill 负责 SOXL 单标的择时场景的时序特征挖掘与验证子工作流。

## 与 agentic-alpha 的差异

| 维度 | agentic-alpha SFA | SOXL Factor Mining |
|------|-------------------|-------------------|
| 标的 | 股票多因子 (CSI1000) | 单 ETF 择时 (SOXL) |
| IC 类型 | 截面 Rank IC | 时序 IC |
| 回测 | Top-N 组合调仓 | 信号方向回测 |
| 数据 | Qlib Expression 面板 | Pandas Series 时间序列 |

## 核心工作流 (R-G-E-V-D)

### 1) Research
1. 读取 `AGENTS.md` 和 `memory/` 中的历史经验。
2. 运行 `scripts/extract_rulebase_signals.py` 从 `soxl_backtest/` 提取 rulebase 信号假设。
3. 生成特征假设卡片（YAML）。

### 2) Generate
1. 将假设转化为 `src/soxl_quant/data.py` 中的特征表达式。
2. 使用 `engineer_feature_frame()` 生成特征序列。
3. 避免与已有 `CORE_FEATURE_COLUMNS` 高度相关的构造。

### 3) Evaluate
1. **时序 IC 快筛**：`scripts/test_factor_ic.py` — 计算 1d/5d/10d/20d 时序 IC。
2. **信号回测**：`scripts/backtest_signal.py` — 验证信号的实际交易效果。
3. **正交预算**：`scripts/analyze_correlation.py` — 与已有特征相关系数 ≤ 0.80。
4. **全量判定**：`ic_positive_ratio > 0.55` 且 `signal_accuracy > 0.55` 且 `profit_factor > 1.2`。

### 4) Validate
1. **IC 衰减分析**：`scripts/test_factor_decay.py` — 多 horizon 时序 IC。
2. **稳定性分析**：`scripts/test_factor_stability.py` — rolling 时序 IC。
3. **体制切片**：由 `soxl-signal-backtest` skill 执行。

### 5) Distill
1. 入库：`scripts/factor_db_cli.py` 写入 `data/soxl_factor_library.db`。
2. 更新 `docs/factor_research/` 和 `memory/`。
3. 新 Accepted 特征加入 `CORE_FEATURE_COLUMNS` 需单独审批。

## 时序 IC 判定标准

| 指标 | 通过门槛 | 说明 |
|------|---------|------|
| mean_ic | > 0.03 | 时序 IC 均值正向 |
| ic_positive_ratio | > 0.55 | IC>0 的天数占比 |
| ic_t | > 2.0 | t 统计量显著 |
| signal_accuracy | > 0.55 | 信号方向正确率 |
| profit_factor | > 1.2 | 盈亏比 |
| max_correlation_with_existing | ≤ 0.80 | 与已有特征最大相关 |

## 可复用脚本

| 脚本 | 功能 | 输出 |
|------|------|------|
| `test_factor_ic.py` | 时序 IC 检验 | CSV + DB |
| `test_factor_decay.py` | IC 衰减分析 | CSV + DB |
| `test_factor_stability.py` | 滚动稳定性 | CSV + DB |
| `backtest_signal.py` | 信号独立回测 | CSV + DB |
| `analyze_correlation.py` | 特征相关性矩阵 | CSV |
| `extract_rulebase_signals.py` | 从 rulebase 提取候选特征 | YAML 假设卡片 |
| `factor_db_cli.py` | 因子库 CLI | 查询/管理 |
| `mine_factors.py` | 全流程入口 | 实验记录 |

## 临时脚本边界

1. 一次性脚本放在项目根目录 `./scripts/temp/`，完成后删除。
2. skills 下不允许存在 `__pycache__` / `.pyc`。

## 参考资料

- `references/soxl_thresholds.md` — 历史阈值与基准分布
