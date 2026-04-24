# Zero-Shot Migration v1

**方向**: multi_asset
**类别**: zero_shot_transfer
**状态**: ✅ Completed

## 假设

SOXL-trained actor 在不修改任何参数的情况下，直接回测其他 3x ETF（TQQQ/UPRO/SPXL/TECL），量化零样本迁移 gap。

## 方法/配置

- Actor: `soxl_balanced_actor.pt`（canonical balanced，GRU-64 + [128,64] MLP）
- 特征: 50 core features（含 VIX，TQQQ/UPRO/SPXL/TECL 的 qlib store 已有 VIX 对齐）
- 回测窗口: 2020-01-02 ~ 2026-03-31（1569 个交易日）
- 脚本: `scripts/generalization_backtest.py`

## 结果

| ETF | Sector | Return | Sharpe | MaxDD | Trades | Sharpe vs SOXL |
|-----|--------|--------|--------|-------|--------|----------------|
| **SOXL** | 半导体 | 531% | **0.846** | 53.3% | 330 | — |
| **TQQQ** | 科技 | 341% | **0.867** | 50.8% | 285 | **+2.5%** |
| **UPRO** | 宽基(S&P) | 219% | **0.842** | 38.2% | 248 | -0.5% |
| **SPXL** | 宽基(S&P) | 212% | **0.830** | 38.0% | 257 | -1.9% |
| **TECL** | 科技 | 245% | **0.716** | 51.8% | 290 | **-15.4%** |

## 关键发现

1. **TQQQ 迁移效果甚至优于 SOXL 本身**（Sharpe 0.867 vs 0.846）。原因可能是：
   - TQQQ 的科技股趋势性更强，SOXL 的半导体周期性更重
   - 特征中的 momentum/trend 信号在 TQQQ 上更有效
   
2. **宽基 ETF（UPRO/SPXL）MaxDD 显著更低**（~38% vs ~53%）。宽基的分散性天然降低了极端 drawdown。

3. **TECL 是迁移效果最弱的**（Sharpe 0.716）。虽然同为科技，但 TECL 的成份股集中度（前10大科技股）可能与 TQQQ（纳斯达克100）不同，导致特征适配度下降。

4. **交易次数与标的波动性正相关**：SOXL 330 笔 > TECL 290 > TQQQ 285 > SPXL 257 > UPRO 248。

## 结论

零样本迁移是**可行的**。SOXL actor 在所有 3x ETF 上都获得了正 Sharpe（0.72-0.87），没有灾难性失败。TQQQ 的表现甚至优于 SOXL，说明核心特征（momentum, trend, VIX）具有跨资产的通用性。

## 与基线/相关实验的差异

- 这是第一次系统性的多资产零样本迁移测试
- 之前仅停留在假设阶段，现在有了量化数据

## 下一步

- [ ] 对比各 ETF 的 B&H 基准，计算超额收益
- [ ] 测试联合训练（shared backbone + asset-specific head）vs 零样本的差距
- [ ] 分析 TECL 表现较弱的具体原因（成份股差异？特征适配？）
