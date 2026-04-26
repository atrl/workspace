# Factor Knowledge

> 最后更新: 2026-04-24  
> 该方向负责特征假设、IC 检验、独立信号回测和因子库管理。

## 因子索引

| 因子 | 来源 | 类别 | 状态 | 关键指标 | 卡片 |
|------|------|------|------|----------|------|
| RB_PANIC_SCORE_PROXY | rulebase | sentiment | 🟡 Weak | Sharpe=+0.172 | [card](cards/rb_panic_score_proxy/card.md) |
| RB_DAILY_DROP_SEVERITY | rulebase | drawdown | ⚪ Unverified | — | [card](cards/rb_daily_drop_severity/card.md) |
| RB_CUMULATIVE_DROP_SEVERITY | rulebase | drawdown | ⚪ Unverified | — | [card](cards/rb_cumulative_drop_severity/card.md) |
| RB_RSI6_OVERSOLD | rulebase | momentum | ❌ Invalid | Sharpe=-0.158 | [card](cards/rb_rsi6_oversold/card.md) |
| RB_RSI6_OVERBOUGHT | rulebase | momentum | ⚪ Unverified | — | [card](cards/rb_rsi6_overbought/card.md) |
| RB_VOL_REGIME | rulebase | volatility | ⚪ Unverified | — | [card](cards/rb_vol_regime/card.md) |
| RB_BUY_COMPOSITE | rulebase | composite | 🟡 Weak | Sharpe=+0.115 | [card](cards/rb_buy_composite/card.md) |
| RB_SELL_COMPOSITE | rulebase | composite | ⚪ Unverified | — | [card](cards/rb_sell_composite/card.md) |
| RB_PROFIT_TAKE_PROXY | rulebase | drawdown | ⚪ Unverified | — | [card](cards/rb_profit_take_proxy/card.md) |
| RB_COOLDOWN_PROXY | rulebase | timing | ✅ Strong | Sharpe=+0.938 | [card](cards/rb_cooldown_proxy/card.md) |
| RB_STOP_LOSS_PROXY | rulebase | drawdown | ⚪ Unverified | — | [card](cards/rb_stop_loss_proxy/card.md) |
| RB_VOLUME_STRESS | rulebase | liquidity | 🟡 Weak | Sharpe=+0.198 | [card](cards/rb_volume_stress/card.md) |
| RB_TREND_REVERSAL | rulebase | trend | ⚪ Unverified | — | [card](cards/rb_trend_reversal/card.md) |
| RB_VIX_TERM_STRESS | rulebase | sentiment | ⚪ Unverified | — | [card](cards/rb_vix_term_stress/card.md) |
| ACAD_AMIHUD_ZSCORE | academic | mean-reversion | 🟡 Weak | Sharpe=+0.284 | [card](cards/acad_amihud_zscore/card.md) |
| ACAD_INTRADAY_REVERSAL_21 | academic | mean-reversion | 🟡 Weak | Sharpe=+0.326 | [card](cards/acad_intraday_reversal_21/card.md) |
| ACAD_VOLATILITY_DRAG | academic | momentum | ❌ Invalid | Sharpe=-0.032 | [card](cards/acad_volatility_drag/card.md) |
| ACAD_DOWN_UP_RATIO | academic | mean-reversion | 🟡 Weak | Sharpe=+0.046 | [card](cards/acad_down_up_ratio/card.md) |
| ACAD_DOWN_UP_RATIO_63 | academic | mean-reversion | 🟡 Weak | Sharpe=+0.129 | [card](cards/acad_down_up_ratio_63/card.md) |
| ACAD_DOWN_RUNS | academic | mean-reversion | ❌ Invalid | Sharpe=-0.289 | [card](cards/acad_down_runs/card.md) |
| TRANS_VIX_DIVERGENCE | transition | mean-reversion | ✅ Strong | Sharpe=+1.041 | [card](cards/trans_vix_divergence/card.md) |
| TRANS_TE_ZSCORE | transition | mean-reversion | 🟡 IC only | IC=+0.072 | [card](cards/trans_te_zscore/card.md) |
| SIGNED_TE_REVERSION | crypto analogue | transition | 🟡 Candidate / blocked | Sharpe=+0.561, IC=+0.014 | [card](card/signed_te_reversion/card.md) |
| INTRADAY_REJECTION_EVENT | crypto analogue | transition | ❌ Blocked | Sharpe=+0.261, PF=1.078 | [card](card/intraday_rejection_event/card.md) |
| regime_risk | core | composite | ✅ Core | 训练主特征 | [card](cards/regime_risk/card.md) |
| regime_risk_no_vix | core | composite | 🟡 Ablated | VIX-free | [card](cards/regime_risk_no_vix/card.md) |
| vol_stress_score | core | composite | 🟡 VIX-free proxy | 纯价格 stress | [card](cards/vol_stress_score/card.md) |

### 状态图例

- ✅ **Strong** — Sharpe > 0.5
- 🟡 **Weak/IC only/Moderate** — 0 < Sharpe < 0.5 或仅 IC 验证
- ❌ **Invalid** — Sharpe <= 0
- ⚪ **Unverified** — 未验证
- ✅ **Core** — 训练核心特征

## 核心结论

1. **TRANS_VIX_DIVERGENCE** 是目前最强的 standalone 信号（Sharpe=+1.041）。
2. **RB_COOLDOWN_PROXY** 是 rulebase 中最强的信号（Sharpe=+0.938）。
3. **RB_RSI6_OVERSOLD** 和 **ACAD_DOWN_RUNS**  standalone 无效，但可能在组合中有价值。
4. **mom_21 ↔ relative_strength_21** 等高度冗余对需要处理（|rho|=1.0）。
5. 大部分 rulebase 和 academic 因子尚未进行 standalone 验证。
6. **SIGNED_TE_REVERSION** 是 crypto funding/basis 类比因子：10d standalone 有正收益（Sharpe=+0.561, PF=1.228），但 IC mean=+0.014、positive ratio=0.516 未过门槛；不加入 core RL schema。
7. **INTRADAY_REJECTION_EVENT** 是 crypto liquidation-wick 类比因子：正交性尚可（max core rho=0.444），但 direction stability 和 standalone backtest 都失败；不要再裸测 wick rejection，除非增加明确 regime gate。

## 特征冗余（|rho| > 0.85）

详见各卡片内的 Max Correlation 指标。全局最高冗余对：
- mom_21 ↔ relative_strength_21: |rho| = 1.000
- drawdown_21 ↔ cum_drop_20: |rho| = 0.993
- vix_zscore ↔ vix_percentile_252: |rho| = 0.986

## 50 Core Features 主题分类

| 主题 | 特征数 | 关键驱动 |
|------|--------|----------|
| Trend / Anchor | 6 | 长期趋势锚定 |
| Momentum | 7 | 短期方向 |
| Volatility | 6 | 风险体制 |
| Drawdown | 7 | 压力深度与持续时间 |
| Structure | 4 | 结构性弱势 |
| Downside | 4 | 下行不对称性 |
| VIX | 4 | 市场恐惧指标 |
| Tracking Error | 4 | LETF 特有摩擦 |
| Velocity | 5 | 体制变化速率 |
| Proxy | 4 | 衍生绩效指标 |
| Composite | 3 | 多信号 stress 指数 |


<!-- MIGRATED FROM memory/2026-04-24-multi-factor-discovery.md -->

# 2026-04-24: Multi-Factor Overlay Tests

## Question

Does combining multiple predictive features into a composite score outperform
regime-separated single-factor overlays?

## Answer: No

### Multi-factor within same regime → underperforms

| Composite | Return vs Baseline |
|-----------|-------------------|
| vol + vix (stress) | +25pp |
| vol + vix + te (stress) | +28pp |
| vol_z + vix_z (stress) | **-8pp** |

### Multi-factor across regimes (dual_kurtosis) → outperforms

| Overlay | Return vs Baseline |
|---------|-------------------|
| vol_stress (stress only) | +128pp |
| kurtosis_bull (bull only) | +96pp |
| **dual_kurtosis (both)** | **+245pp** |

## Why

- **Within-regime features are correlated**: vol_of_vol_21 ρ with VIX divergence ≈ 0.65
- **Cross-regime features are uncorrelated**: vol (stress) and kurtosis (bull) capture different phenomena
- **Correlation kills composite alpha**: Combining correlated signals dilutes rather than amplifies

## Lesson

For overlay design:
1. Find the single best signal PER regime
2. Apply regime gating
3. Do NOT combine multiple signals within the same regime

The "multi-factor" intuition (more features = better) is wrong for execution overlays
when features are correlated. Regime separation is the key edge.

<!-- 2026-04-26: crypto-style signed tracking-error factor -->

# 2026-04-26: SIGNED_TE_REVERSION Candidate

## Question

Can a crypto futures basis / funding-rate analogue improve SOXL single-asset timing?

## Answer: Not as a direct standalone promotion

`SIGNED_TE_REVERSION_* = -zscore(rolling_sum(SOXL_return - 3 * SOXX_return, window), 252d)`
was added to the `transition_signals` pack and evaluated over 5d / 10d / 21d windows.

| Check | Best result | Gate |
|-------|-------------|------|
| IC | `SIGNED_TE_REVERSION_21`, h=20, mean=+0.014, positive_ratio=0.516 | Fail (`mean_ic > 0.03`, positive_ratio > 0.55 required) |
| Standalone backtest | `SIGNED_TE_REVERSION_10`, threshold=0.5, Sharpe=+0.561, PF=1.228, accuracy=0.546 | Mixed |
| Correlation | `SIGNED_TE_REVERSION_10` max core rho=0.577 | Pass |
| Family correlation | `SIGNED_TE_REVERSION_5` max core rho=0.811 | Fail for 5d variant |

## Lesson

The crypto-style basis/funding analogy has some executable signal in the 10d variant,
but the direction is not stable enough for feature promotion. If revisited, test only
`SIGNED_TE_REVERSION_10` under regime gating; do not add the full 5/10/21 family to core features.

<!-- 2026-04-26: crypto-style intraday rejection factor -->

# 2026-04-26: INTRADAY_REJECTION_EVENT Blocked

## Question

Can crypto-style liquidation wick / panic rejection improve SOXL single-asset timing?

## Answer: No as a naked standalone signal

`INTRADAY_REJECTION_EVENT = lower_wick_ratio * close_location * positive_intraday_range_zscore * (1 + positive_relative_volume_zscore)`
was added to the `transition_signals` pack and evaluated as an event-style intraday reversal factor.

| Check | Best result | Gate |
|-------|-------------|------|
| IC | h=20, mean=+0.0157, positive_ratio=0.215 | Fail |
| Standalone backtest | threshold=0.0, Sharpe=+0.261, PF=1.078, accuracy=0.529, total_return=-53.9% | Fail |
| Correlation | max core rho=0.444 vs `volume_shock` | Pass |

## Lesson

The wick/rejection pattern is not enough by itself for SOXL daily timing. The factor
is orthogonal enough to be a building block, but it should not be promoted or retested
without a clear regime gate such as stress-release, VIX divergence, or post-crash compression.
