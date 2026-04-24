---
date: '2026-04-14'
title: "SOXL \u6A21\u578B\u4F18\u5316\u8DEF\u7EBF\u56FE\uFF08\u57FA\u4E8E\u6CDB\u5316\
  \u6D4B\u8BD5\u7ED3\u679C\uFF09"
kind: review
profiles:
- soxl
status: reference
tags:
- checkpoint-selection
- eval-2022
- reference
- review
- soxl
- walk-forward
artifacts:
- runtime/execution/generalization/detailed/
- runtime/execution/generalization/detailed/summary.json
workpad: memory/index.json
summary: "Legacy memory note normalized into the repo-wide schema. > \u6D4B\u8BD5\u65E5\
  \u671F\uFF1A2026-04-14 > \u6D4B\u8BD5\u6A21\u578B\uFF1Asoxl_balanced_actor.pt\uFF08\
  PPO\uFF0C\u96F6\u6837\u672C\uFF09 > \u6CDB\u5316\u6807\u7684\uFF1ASOXL\u3001TQQQ\u3001\
  UPRO\u3001SPXL\u3001TECL\uFF085\u53EA3\u500D\u6760\u6746ETF\uFF09 | ETF | \u7B56\
  \u7565\u6536\u76CA | B&H \u6536\u76CA | \u8D85\u989D\u6536\u76CA | \u7B56\u7565\u56DE\
  \u64A4 | B&H \u56DE\u64A4 | \u4EA4\u6613\u6B21\u6570 | |-----|----------|----------|"
---

# SOXL 模型优化路线图（基于泛化测试结果）

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

> 测试日期：2026-04-14  
> 测试模型：soxl_balanced_actor.pt（PPO，零样本）  
> 泛化标的：SOXL、TQQQ、UPRO、SPXL、TECL（5只3倍杠杆ETF）

---

## 一、当前表现复盘

### 1.1 总体结果

| ETF | 策略收益 | B&H 收益 | 超额收益 | 策略回撤 | B&H 回撤 | 交易次数 |
|-----|----------|----------|----------|----------|----------|----------|
| SOXL | **374.85%** | 154.26% | **+86.75%** | 51.48% | 90.46% | 578 |
| TQQQ | 230.51% | 284.31% | **-14.00%** | 43.99% | 81.66% | 476 |
| UPRO | 176.53% | 180.49% | **-1.41%** | 35.22% | 76.82% | 381 |
| SPXL | 172.53% | 183.15% | **-3.75%** | 35.42% | 76.86% | 397 |
| TECL | 223.01% | 270.08% | **-12.72%** | 42.02% | 77.96% | 512 |

### 1.2 关键洞察

1. **回撤控制极强**：所有 ETF 上策略回撤都只有 B&H 的一半左右（35% ~ 51% vs 76% ~ 90%）。
2. **只在训练域 SOXL 跑赢 B&H**：其他 4 只 ETF 均小幅跑输，说明模型对 SOXL 的分布过拟合。
3. **科技类 ETF（TQQQ、TECL）跑输最多**：2020-2021 和 2023-2024 的两轮科技牛市中，模型仓位偏保守，存在明显**踏空**。
4. **宽基指数（UPRO、SPXL）几乎打平**：收益只差 1~4%，但回撤砍半，夏普/卡玛比率非常优秀。

---

## 二、优化方向

### 方向 1：多资产联合训练（Multi-ETF Training）

**问题**：模型只在 SOXL 上训练，没见过 TQQQ/TECL 的波动路径。  
**方案**：
- 把 SOXL、TQQQ、UPRO、SPXL、TECL 的数据拼接成一个统一的训练池。
- 在 `data.py` 的 `SoxlDataHandler` 中增加 `instrument_id` 特征（one-hot 或 embedding）。
- 每个 episode 随机抽取不同 ETF 的片段，让策略学习跨品种共性。
- 预期效果：TQQQ/TECL 的收益有望追上 B&H，同时保持低回撤优势。

**改动点**：
- `src/soxl_quant/data.py`：特征里加 `instrument_id`
- `src/soxl_quant/cli.py`：训练时从多 ETF 生成 initial_states
- `src/soxl_quant/rl.py`：`obs_dim` 需要 +1（或 +N）

---

### 方向 2：动态仓位上限（Adaptive Position Cap）

**问题**：当前最大仓位仅 ~0.59（59%），在强牛市中错失主升浪。  
**方案**：
- 将 `drawdown_position_cap` 的 `min_cap` 从固定 0.4 改为**根据趋势强度动态调整**。
- 当 `trend_gap_21 > 0`、`anchor_gap_180 > 0`、`realized_vol < 0.40` 同时满足时，允许仓位上限提升到 0.85 ~ 1.0。
- 当 `regime_risk > 0.65` 或 `vix_percentile_252 > 0.80` 时，再压回 0.4。

**改动点**：
- `src/soxl_quant/rl.py`：`action_band_from_obs()` 和 `drawdown_position_cap()`
- `configs/soxl/*.yaml`：新增 `PositionCapConfig` 参数（`bull_cap`、`bull_vol_threshold`）

---

### 方向 3：踏空惩罚 / 机会成本奖励（Opportunity Cost Penalty）

**问题**：模型在趋势确立后仍然长时间低仓位，没有“追涨”动机。  
**方案**：
- 在 `ContinuousReward.reward()` 中加入**踏空惩罚项**：
  - 当 `anchor_gap > +0.03` 且 `pos < 0.30` 时，每日扣减小额奖励。
  - 当 `trend_slope_21 > 0` 连续 5 天且 `pos < 0.40` 时，加大惩罚。
- 类似地，对**逆势高仓位**也保持现有惩罚不变。

**注意**：惩罚权重必须远小于回撤惩罚，否则会导致牛市末期追高被套。

**改动点**：
- `src/soxl_quant/rl.py`：`ContinuousReward.reward()`

---

### 方向 4：品种/行业特征嵌入（Sector-Aware Encoding）

**问题**：模型不知道自己在交易半导体还是科技股还是宽基指数。  
**方案**：
- 为每个 ETF 预计算一个**静态风险画像**：
  - 年化波动率中位数（`median_vol_252`）
  - 行业 beta（对 SPY 的滚动 63 天 beta）
  - 最大历史回撤分位数
- 把这些作为额外特征输入给 actor，让模型自己学会“科技股波动更大，需要更宽的止损”。

**预期效果**：泛化到 unseen ETF 时，模型能根据品种特征自动调整激进/保守程度。

**改动点**：
- `src/soxl_quant/data.py`：`engineer_feature_frame()` 增加品种画像特征
- `scripts/fetch_3x_etf_data.py`：下载时预计算并写入 CSV

---

### 方向 5：交易成本与换手率再优化

**问题**：SOXL 交易 578 次，TQQQ 476 次，平均每 2.7 ~ 3.3 天就调一次仓。  
**方案**：
- 提高 `rebalance_threshold` 的基准值，从当前 0.015 ~ 0.03 提升到 0.02 ~ 0.04。
- 或者引入**冷却期**：单次调仓后至少持仓 2 个交易日才能再次调仓（已有 `last_step` 机制，可放宽）。
- 在奖励函数中加大 `turnover_penalty` 和 `smoothness_penalty` 权重。

**预期效果**：交易次数下降 20% ~ 30%，滑点成本降低，夏普提升。

**改动点**：
- `src/soxl_quant/rl.py`：`rebalance_threshold()`
- `configs/soxl/*.yaml`：`turnover_penalty`、`smoothness_penalty`

---

### 方向 6：网络结构升级（MLP → LSTM/Transformer）

**问题**：当前 actor 是简单的两层 MLP（128 → 64），只能看到单日的 48 维向量。  
**方案**：
- 把 `LOOKBACK=60` 天的原始特征序列直接喂给 **LSTM 或 GRU**。
- 或者使用轻量级 **Transformer Encoder**（2 层，4 head）。
- 这样模型可以自动捕捉“连续下跌后的反弹模式”、“VIX 突增后的恐慌底”等时序规律。

**注意**：时序网络训练更慢、更容易过拟合，需要配合 Dropout 和更小的学习率。

**改动点**：
- `src/soxl_quant/rl.py`：`build_policy()` 中替换 `Net` 为自定义 LSTM/Transformer
- `src/soxl_quant/data.py`：`make_observation()` 需要输出序列而非展平向量

---

### 方向 7：Walk-Forward 模型集成（Ensemble）

**问题**：单个模型在特定窗口可能表现不佳。  
**方案**：
- 把已经训练好的 `wf_2020`、`wf_2022`、`wf_2023_2024` 三个 walk-forward 模型做**投票集成**。
- 每天让 3 个 actor 分别输出目标仓位，取中位数（或按最近窗口 Sharpe 加权）。
- 这样可以在不同市场体制（牛市、熊市、震荡市）间平滑切换。

**预期效果**：降低单一窗口过拟合风险，提升泛化稳健性。

**改动点**：
- `src/soxl_quant/backtest.py`：新增 `EnsembleRLStrategy`
- `scripts/generalization_backtest.py`：加载多个 checkpoint 做集成回测

---

## 三、优先级建议

| 优先级 | 优化方向 | 预期收益 | 改动量 | 风险 |
|--------|----------|----------|--------|------|
| P0 | 多资产联合训练 | 高（解决泛化） | 中 | 低 |
| P0 | 踏空惩罚 / 动态 cap | 高（解决牛市踏空） | 小 | 中 |
| P1 | 交易成本优化 | 中（提升夏普） | 小 | 低 |
| P1 | 品种特征嵌入 | 中（提升泛化） | 中 | 低 |
| P2 | 模型集成 | 中（提升稳健性） | 小 | 低 |
| P2 | LSTM/Transformer | 高（捕捉时序） | 大 | 高 |

---

## 四、下一步行动

建议按以下顺序执行：

1. **先落地 P0**：修改 `rl.py` 的奖励函数和仓位上限逻辑，加入踏空惩罚和动态 cap。
2. **同步修改 `data.py`**：把多 ETF 数据池打通，为联合训练做准备。
3. **跑一轮联合训练**：用 SOXL + UPRO + TQQQ 做训练集，SOXL 做验证集。
4. **评估集成效果**：把 3 个 walk-forward 模型加权集成，看是否能同时提升收益和夏普。
5. **再考虑网络升级**：如果前三步收益瓶颈明显，再投入 LSTM/Transformer。

---

## 五、相关文件

- 泛化报告：`memory/2026-04-13-generalization.md`
- 详细回测数据：`runtime/execution/generalization/detailed/*.csv`
- 汇总统计：`runtime/execution/generalization/detailed/summary.json`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
