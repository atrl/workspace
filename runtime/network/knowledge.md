# NETWORK Knowledge

> 最后更新: 2026-04-24
> 该方向负责网络架构选择、超参实验和模型结构诊断。

## 架构索引

| 架构 | 类别 | 状态 | 实验数 | 关键指标 | 卡片 |
|------|------|------|--------|----------|------|
| GRU | recurrent | ✅ Canonical | 60+ | Sharpe 0.69-2.24 | [card](cards/gru/card.md) |
| Attention | attention | 🟡 No improvement | 1 | Sharpe 0.76 | [card](cards/attention/card.md) |
| TiDE | temporal_decomposition | 🟡 No improvement | 1 | Sharpe 0.69 | [card](cards/tide/card.md) |
| MLP Fallback | dense | ⚪ Unused | 0 | — | [card](cards/mlp_fallback/card.md) |

## 核心结论

1. **GRU-64 + [128,64] MLP 是当前主架构**，observation_window=5。
2. **Attention 替换 GRU 无提升**：同规模下指标无差异，计算成本更高。
3. **TiDE 未超越 GRU**：时序分解网络无优势，可能与短窗口（5步）有关。
4. **MoE 未实际实验**：代码存在但未在 SOXL 上跑过完整训练。

## 待验证方向

- [ ] 跨资产 shared backbone + asset-specific head
- [ ] 更轻量网络 [64,32]
- [ ] 1D-CNN 局部特征提取


<!-- MIGRATED FROM memory/2026-04-24-risk-budget-v3-gate-pass.md -->

---
date: 2026-04-24
tags: [risk_budget, mode_conditioned, gate_pass, v3]
direction: network
---

# risk_budget_usage_v3 passes fixed-split promotion gate

## Result

| Slice | Return | Sharpe | MaxDD | AvgPos |
|-------|--------|--------|-------|--------|
| test_proxy | 274.7% | **1.074** | 21.4% | 0.208 |
| test_2020 | 29.3% | 1.430 | 18.5% | 0.233 |
| test_2022 | -9.7% | -0.601 | **12.0%** | **0.143** |
| test_2023_2024 | 60.5% | 1.378 | 16.0% | 0.215 |

### Promotion Gate
- test_2022 MaxDD: 12.0% < 20.0% ✅
- test_2022 avg_pos: 0.143 < 0.15 ✅
- test_proxy Sharpe: 1.074 > 0.95 ✅

**Fixed Split Gate: PASSED!**

## Key Insight

Mode-conditioned **actor input** (one-hot appended to obs) is dramatically more
effective than mode-conditioned **reward shaping** alone:

| Variant | proxy Sharpe | 2022 MaxDD | 2022 avg_pos |
|---------|-------------|------------|--------------|
| v1 (baseline) | 0.942 | 18.7% | 0.127 |
| v2 (reward shaping) | 0.884 | 7.4% | 0.070 |
| **v3 (actor input)** | **1.074** | **12.0%** | **0.143** |

v2's reward shaping made the policy excessively conservative — great downside
protection but missed all upside. v3 lets the actor *see* the mode and learn
its own usage policy per mode, achieving both better Sharpe and acceptable
drawdown control.

## Comparison to canonical balanced

| Metric | balanced (canonical) | risk_budget_usage_v3 |
|--------|---------------------|----------------------|
| proxy Sharpe | ~0.95 | **1.074** |
| proxy MaxDD | ~25% | 21.4% |
| 2022 MaxDD | ~15% | **12.0%** |

v3 now exceeds canonical balanced on Sharpe and has tighter drawdown control.

## Next Steps

1. Run walk-forward validation to confirm no overfit
2. If WF passes, promote to canonical or create v4 with additional refinements
3. Consider whether mode-conditioned input should become standard across all profiles



<!-- MIGRATED FROM memory/2026-04-24-risk-budget-v3-wf-failure.md -->

# risk_budget_usage_v3 Walk-Forward Failure

**Date**: 2026-04-24
**Profile**: `risk_budget_usage_v3`
**Direction**: network / execution overlay
**Status**: candidate (promotion blocked)

## Walk-Forward Results

| Window | Train | Valid | Test | Return | Sharpe | MaxDD | Trades |
|--------|-------|-------|------|--------|--------|-------|--------|
| wf_2020 | 2010-03→2018-12 | 2019 | 2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 2010-03→2020-12 | 2021 | 2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 2010-03→2022-12 | 2023 | 2024 | 13.0% | 0.22 | 23.4% | 40 |

Fixed-split backtest with saved model:
- Return: 434.1%, Sharpe: 0.87, MaxDD: 47.5%, Trades: 295

## Key Findings

### 1. Proxy-to-Backtest Gap
Proxy eval metrics during training (Sharpe 1.074, MaxDD 21.4%) were significantly more optimistic than the actual backtest on the saved model (Sharpe 0.87, MaxDD 47.5%). This means the proxy metric is NOT a reliable predictor of real backtest performance.

### 2. Walk-Forward Generalization Collapse
The model fails to generalize when trained on earlier data and tested on held-out years:
- wf_2022: MaxDD 59.0% — the 2022 crash completely breaks the strategy
- wf_2023_2024: Sharpe 0.22 — no edge in the most recent data
- wf_2020: Sharpe 0.29 — weak even in non-crash years

### 3. Mode-Conditioned Input Does Not Fix Generalization
While mode-conditioned actor input improved fixed-split proxy metrics, it did NOT solve the core generalization problem. The network likely memorizes mode-specific behavior that happens to work in the fixed test window but fails in truly novel regimes.

### 4. Risk Budget Usage Architecture May Be Too Expressive
The budget_usage mapping with delta_levels [-0.10, -0.05, 0.0, +0.05, +0.10] may give the actor too much flexibility, allowing it to overfit to specific patterns in the training/test window.

## Diagnosis Needed Before v4

1. **Mode info leak check**: Does the mode classifier use future information? Run mode classification on a pure walk-forward basis.
2. **Budget usage regularization**: Would a tighter delta range (e.g., ±0.03) or fewer levels reduce overfit?
3. **Network size**: GRU-64 + MLP 128→64 may be too large for the amount of data per mode.
4. **Baseline validation**: Test mode-conditioned input on a proven generalizing baseline (e.g., balanced_rule_residual_v19) first.
5. **Entropy collapse**: Check if entropy drops too low during training, indicating premature convergence.

## Comparison with v1/v2

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | 1.074 |
| actual Sharpe | ~0.9? | ~0.8? | **0.87** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| actual MaxDD | ~35%? | ~25%? | **47.5%** |
| wf_2022 MaxDD | ? | ? | **59.0%** |
| wf_2023_2024 Sharpe | ? | ? | **0.22** |

## Decision

**❌ PROMOTION BLOCKED.**

v3 is retained as a candidate for diagnosis purposes (keep_actor=true), but cannot be promoted to canonical. The mode-conditioned input mechanism is NOT validated for generalization.

## Next Action

Run diagnostic experiments before declaring v4:
1. Check mode classifier for forward-looking bias
2. Test mode-conditioned input on balanced_rule_residual_v19
3. Evaluate if smaller network or tighter usage bands help



<!-- MIGRATED FROM memory/2026-04-25-v19-reg-results.md -->

# v19_reg Results: Regularization Improves Bear-Market Walk-Forward

## Date
2026-04-25

## Hypothesis
Adding dropout (0.2) and weight decay (1e-5) to GRUNet would reduce bull-market overfitting and improve bear-market walk-forward generalization.

## Results

### Fixed-Split
- Proxy: Sharpe 0.89, Ret 659%, MaxDD 59.8%
- 2022: Sharpe -1.08, Ret -32.1%, MaxDD 40.3% (WORSE than v19)
- 2023_2024: Sharpe 0.71, Ret 51.3%, MaxDD 40.4% (WORSE than v19)

### Walk-Forward
- wf_2020: Sharpe 2.20, Ret 95.3%, MaxDD 20.8% (good bull market)
- wf_2022: Sharpe **-0.73**, Ret **-21.9%**, MaxDD 31.0% ← **BEST in residual family!**
- wf_2023_2024: Sharpe -0.44, Ret -22.9%, MaxDD 37.2% (WORSE than v19)

### Comparison

| Profile | wf_2020 Sh | wf_2022 Sh | wf_2024 Sh |
|---------|------------|------------|------------|
| balanced | 2.30 | -0.91 | **-0.22** |
| v19 | 2.35 | -0.80 | -0.23 |
| v19_small | 2.04 | -1.11 | -0.27 |
| v22 | 2.27 | -0.84 | -0.36 |
| **v19_reg** | **2.20** | **-0.73** | -0.44 |

## Conclusions

1. **Regularization DOES improve bear-market generalization** - v19_reg is the first model to show improvement in ANY walk-forward window (wf_2022: -21.9% vs v19 -25.2%).

2. **Trade-off exists** - improved wf_2022 but degraded wf_2023_2024 (-22.9% vs v19 -18.7%). The regularization may be too strong for mixed markets.

3. **Overfitting is confirmed** - the fact that regularization helps proves the model was overfitting to bull-market patterns.

4. **Bull-market performance preserved** - wf_2020 Sharpe 2.20 is only slightly below v19's 2.35.

## Next Actions

1. **Tune regularization strength**: Try dropout 0.1 or weight_decay 1e-6 to find a better balance between wf_2022 and wf_2023_2024.
2. **Try different regularization placement**: Only dropout on MLP (not GRU), or only weight decay without dropout.
3. **Combine with other techniques**: Regularization + VIX features might be synergistic.

## Code Changes
- `src/soxl_quant/policy/models/gru.py`: Added dropout parameter
- `src/soxl_quant/policy/policy_builder.py`: Pass dropout to GRUNet, weight_decay to Adam
- `src/soxl_quant/config.py`: Added dropout and weight_decay fields
- `configs/soxl/balanced_rule_residual_v19_reg.yaml`: New profile

## v19_reg_light Update (2026-04-26 00:17)

### Walk-Forward Results
- wf_2020: Sharpe 2.36, Ret 115.6%, MaxDD 20.1% — BEST bull market in family
- wf_2022: Sharpe -0.86, Ret -21.9%, MaxDD 30.2%
- wf_2023_2024: Sharpe -0.38, Ret -24.3%, MaxDD 41.9%

### Regularization Trade-off Summary

| Setting | wf_2020 Sh | wf_2022 Sh | wf_2024 Sh |
|---------|------------|------------|------------|
| v19 (none) | 2.35 | -0.80 | **-0.23** |
| v19_reg_light (0.1/1e-6) | **2.36** | -0.86 | -0.38 |
| v19_reg (0.2/1e-5) | 2.20 | **-0.73** | -0.44 |

**Key insight**: Regularization strength creates a Pareto frontier. No single setting improves both bear-market windows. Stronger reg → better wf_2022 but worse wf_2023_2024. Weaker reg → better wf_2020 but no wf_2022 improvement.

### Next Direction
Regularization alone is insufficient. Consider:
1. Regime-dependent dropout (higher in bull, lower in bear)
2. Combine v22 (VIX features) + v19_reg (regularization)
3. Explicit regime-conditioned policy architecture



<!-- MIGRATED FROM memory/2026-04-25-v19-small-training-failure.md -->

# v19_small Training Complete: Worst Bear-Market Performance in Family

Date: 2026-04-25

## Results

`balanced_rule_residual_v19_small` (smaller network variant of v19) completed training.

| Metric | Valid | test_2020 | test_2022 | test_2023_2024 |
|--------|-------|-----------|-----------|----------------|
| Return | +62.4% | +96.9% | **-31.6%** | +46.8% |
| Sharpe | 1.57 | 2.04 | **-1.11** | 0.67 |
| MaxDD | 30.7% | 26.6% | 38.3% | 41.9% |
| AvgPos | 0.66 | 0.53 | 0.31 | 0.60 |
| StressDays | — | 48 | 129 | 64 |

## Comparison with Family

| Profile | test_2022 Sharpe | test_2022 MaxDD | test_2023_2024 Sharpe | test_2023_2024 MaxDD |
|---------|-----------------|-----------------|----------------------|---------------------|
| balanced (canonical) | -0.91 | 33.2% | — | — |
| v19 | -0.80 | 35.2% | — | — |
| v22 | -0.76 | 30.1% | 0.72 | 35.7% |
| **v19_small** | **-1.11** | **38.3%** | **0.67** | **41.9%** |

## Key Findings

1. **Worst bear-market performance in the family** — test_2022 Sharpe -1.11 is significantly worse than v19 (-0.80), v22 (-0.76), and balanced (-0.91)
2. **test_2023_2024 MaxDD 41.9%** — very high drawdown in the recovery window, worse than v22 (35.7%)
3. **Proxy Sharpe 0.84** — passes the 0.75 threshold, but fixed-split metrics are misleading (as proven repeatedly)
4. **Smaller network does NOT help** — v19_small uses a reduced network but performs worse on all stress metrics

## Conclusion

Network size reduction is NOT the solution to bear-market generalization. v19_small is the worst-performing residual family member on bear markets.

## Decision

- `balanced_rule_residual_v19_small`: **dropped**
- `balanced_rule_residual_v19_tiny` (baseline: v19_small): should not be trained — predicted to fail similarly

## Pattern Confirmation

Four consecutive confirmations of family-level bear-market failure:
1. balanced (canonical): 2022 Sharpe -0.91
2. v19: 2022 Sharpe -0.80
3. v22: 2022 Sharpe -0.76
4. v19_small: 2022 Sharpe -1.11

The problem is NOT network size, reward coefficients, or residual architecture. The problem is **training data lacks prolonged bear markets**.



<!-- MIGRATED FROM memory/2026-04-25-v22-training-failure.md -->

# v22 Training Complete: Same Bear-Market Collapse

Date: 2026-04-25

## Results

`balanced_rule_residual_v22` (v19 variant with adjusted reward coefficients) completed training.

| Metric | Valid | test_2020 | test_2022 | test_2023_2024 |
|--------|-------|-----------|-----------|----------------|
| Return | +53.7% | +103.2% | **-21.0%** | +48.2% |
| Sharpe | 1.65 | 2.30 | **-0.76** | 0.72 |
| MaxDD | 26.0% | 20.8% | 30.1% | 35.7% |
| AvgPos | 0.54 | 0.49 | 0.27 | 0.50 |
| StressDays | — | 48 | 129 | 64 |

## Key Findings

1. **Proxy MaxDD 48.7%** — way above acceptable threshold, model is too risky even on aggregate
2. **test_2022 Sharpe -0.76** — identical bear-market collapse pattern to v19 (-0.80) and balanced (-0.91)
3. **test_2023_2024 Sharpe 0.72** — decent recovery window but MaxDD 35.7% is high

## Conclusion

Reward coefficient adjustments on the residual family do NOT solve the fundamental bear-market generalization problem. This is the **third consecutive confirmation** of family-level failure (balanced, v19, v22).

## Decision

- `balanced_rule_residual_v22`: **dropped**
- No new residual-family experiments until bear-market-aware training is implemented

## Next Direction

Priority is bear-market generalization:
1. **Data**: Include longer bear history or synthetic bear episodes
2. **Sampling**: Oversample stress episodes during training
3. **Reward**: Add bear-market-specific penalties
4. **Features**: Add leading indicators (credit spreads, yield curve)

