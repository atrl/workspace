---
name: leveraged-etf-factor-mining
description: 3x杠杆ETF单标的时序特征（因子）挖掘、IC检验、信号回测与因子库管理。默认标的SOXL，支持TQQQ/UPRO/SPXL/TECL等3xETF。独立于RL训练流程，专门负责从rulebase逻辑和假设中挖掘有效特征，验证通过后入库供RL/Rulebase使用。
version: 2.0.0
changelog:
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

## 核心工作流 (R-G-E-V-D)

### 1) Research
1. 读取`AGENTS.md`和`runtime/factor/knowledge.md`中的历史经验。
2. 运行`scripts/extract_rulebase_signals.py`从`soxl_backtest/`提取rulebase信号假设。
3. 生成特征假设卡片（YAML）。

### 2) Generate
1. 将假设转化为`src/soxl_quant/data.py`中的特征表达式。
2. 使用`engineer_feature_frame(symbol=...)`生成特征序列。
3. 避免与已有`CORE_FEATURE_COLUMNS`高度相关的构造。

### 3) Evaluate
1. **时序IC快筛**：`scripts/test_factor_ic.py --symbol SOXL` — 计算1d/5d/10d/20d时序IC。
2. **信号回测**：`scripts/backtest_signal.py --symbol SOXL` — 验证信号的实际交易效果。
3. **正交预算**：`scripts/analyze_correlation.py` — 与已有特征相关系数≤0.80。
4. **全量判定**：`ic_positive_ratio > 0.55`且`signal_accuracy > 0.55`且`profit_factor > 1.2`。

### 4) Validate
1. **IC衰减分析**：`scripts/test_factor_decay.py --symbol SOXL` — 多horizon时序IC。
2. **稳定性分析**：`scripts/test_factor_stability.py --symbol SOXL` — rolling时序IC。
3. **体制切片**：由`leveraged-etf-signal-backtest` skill执行。

### 5) Distill
1. 入库：`scripts/factor_db_cli.py`写入`data/{symbol}_factor_library.db`（或统一库`data/etf_factor_library.db`）。
2. 更新`docs/factor_research/`和`runtime/factor/knowledge.md`。
3. 新Accepted特征加入`CORE_FEATURE_COLUMNS`需单独审批。

## 时序IC判定标准

| 指标 | 通过门槛 | 说明 |
|------|---------|------|
| mean_ic | > 0.03 | 时序IC均值正向 |
| ic_positive_ratio | > 0.55 | IC>0的天数占比 |
| ic_t | > 2.0 | t统计量显著 |
| signal_accuracy | > 0.55 | 信号方向正确率 |
| profit_factor | > 1.2 | 盈亏比 |
| max_correlation_with_existing | ≤ 0.80 | 与已有特征最大相关 |

## 可复用脚本

| 脚本 | 功能 | 输出 |
|------|------|------|
| `test_factor_ic.py` | 时序IC检验 | CSV + DB |
| `test_factor_decay.py` | IC衰减分析 | CSV + DB |
| `test_factor_stability.py` | 滚动稳定性 | CSV + DB |
| `backtest_signal.py` | 信号独立回测 | CSV + DB |
| `analyze_correlation.py` | 特征相关性矩阵 | CSV |
| `extract_rulebase_signals.py` | 从rulebase提取候选特征 | YAML假设卡片 |
| `factor_db_cli.py` | 因子库CLI | 查询/管理 |
| `mine_factors.py` | 全流程入口 | 实验记录 |

## 临时脚本边界

1. 一次性脚本放在项目根目录`./scripts/temp/`，完成后删除。
2. skills下不允许存在`__pycache__` / `.pyc`。

## 参考资料

- `references/etf_thresholds.md` — 历史阈值与基准分布（按标的分类）
