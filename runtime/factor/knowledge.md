# Factor Knowledge

> 最后更新: 2026-05-23
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
| SIGNED_TE_REVERSION_10_STRESS_GATE | crypto analogue | transition | 🟡 Candidate / blocked | PF=1.351, Acc=0.590 | [card](card/signed_te_stress_gate/card.md) |
| INTRADAY_REJECTION_EVENT | crypto analogue | transition | ❌ Blocked | Sharpe=+0.261, PF=1.078 | [card](card/intraday_rejection_event/card.md) |
| REL_VOLUME_RATIO | SOXX | liquidity | 🟡 IC candidate | IC=+0.058, IC+=0.591, rho=0.136 | [card](card/soxx_signals/card_REL_VOLUME_RATIO.md) |
| TUSHARE_US_DAILY_ADJ | Tushare | data intake | ❌ Blocked / weak | best IC=+0.012, IC+=0.485, rho=0.280 | [card](card/tushare_soxx_factor_intake_v1/notes.md) |
| FUTU_SMALL_ORDER_FLOW | Futu | order flow | ❌ Short-history / gate fail | active best IC=+0.034, IC+=0.468, PF=1.657, rho=0.258 | [card](card/futu_small_order_flow/notes.md) |
| SOXX_SPREAD_OSCILLATION | SOXX/SOXL | relative-value | ❌ Blocked | best IC=+0.0156, PF=1.178, rho=0.917; chop rho=0.081 but IC=-0.0016 | [card](card/soxx_spread_oscillation/notes.md) |
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
7. **SIGNED_TE_REVERSION_10_STRESS_GATE** 证明 stress-release gate 能提高执行质量（PF=1.351、accuracy=0.590、max core rho=0.388），但 IC positive ratio 因信号稀疏仅 0.036，仍不进入 RL feature promotion。
8. **INTRADAY_REJECTION_EVENT** 是 crypto liquidation-wick 类比因子：正交性尚可（max core rho=0.444），但 direction stability 和 standalone backtest 都失败；不要再裸测 wick rejection，除非增加明确 regime gate。
9. **SOXX direct actor feature retrain 不应重复**：`balanced_rule_residual_v22_soxx` 已经失败并被 dropped；2026-05-23 复核显示 `REL_VOLUME_RATIO` 有 IC 候选价值（h20 IC=+0.058、positive_ratio=0.591、max core rho=0.136），但 PF/accuracy 不过门槛；`PRICE_RATIO_MOM` 与 `SOXX_BIAS_10` 分别与 `mom_5` 高相关（0.993/0.916），不要作为 core feature 直接重训。
10. **Tushare 原始 us_daily_adj pack 不足以直接重训**：2026-05-23 新 token 已拉取 SOXX/SOXL cache 到 `2026-05-22`；`TUSHARE_SOXX_TURNOVER_ZSCORE_21` 是最佳原始候选，但 h20 IC 只有 `+0.012020`、positive ratio `0.485`，低于 promotion gate。虽然部分 long threshold 的 PF/accuracy 接近或略过 `1.2/0.55`，但 IC 稳定性失败，不能据此开 Tushare feature retrain。
11. **Futu 小单资金流已做成 lagged feature pack，但不过 promotion gate**：2026-05-23 初次 probe 被 OpenD 图形验证阻断；完成本地 phone/UI 验证后，已缓存 `US.SOXL` / `US.SOXX` daily capital flow 各 `251` rows（`2025-05-23` 至 `2026-05-22`）、intraday 各 `391` rows，以及 snapshot/distribution。后续全历史测试显示 Futu K 线可覆盖长历史（SOXL `2010-03-31` 起，SOXX `2010-03-11` 起），但 `get_capital_flow` 对 `2010-2011`、`2016`、`2020`、`2024` 均返回 `0` rows，全范围也只返回最近 `251` rows。新增 `futu_small_order_flow` pack 全部 T+1 lag；active-window 最佳 `FUTU_SOXL_SMALL_FLOW_SHARE` h20 IC `+0.034`，但 positive ratio `0.468` 低于 `0.55`，不进训练 profile，只保留短历史/live overlay 候选。
12. **SOXX/SOXL 价差扩张与震荡差不适合直接训练**：`soxx_spread_oscillation` 测了 ratio zscore、ratio widen、spread vol zscore、relative choppiness。前三个与 core trend/momentum/volatility 高相关（max rho `0.885`/`0.922`/`0.917`），且 IC/PF/accuracy 均未过 gate；`SOXL_SOXX_CHOP_DIVERGENCE_21` 很正交（rho `0.081`），但 h20 IC `-0.0016`、IC+ `0.425`，不具备方向稳定性。不要基于这组直接开 SOXX/SOXL spread feature retrain。

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

<!-- 2026-04-26: signed TE stress-release gate -->

# 2026-04-26: SIGNED_TE_REVERSION_10_STRESS_GATE Candidate

## Question

Can stress-release gating fix the direction instability of `SIGNED_TE_REVERSION_10`?

## Answer: Partially, but not enough for promotion

The gated variant keeps `SIGNED_TE_REVERSION_10` only when either VIX divergence is high
or recent elevated `regime_risk` is dropping over 5 days.

| Check | Best result | Gate |
|-------|-------------|------|
| IC | h=20, mean=+0.0130, positive_ratio=0.036 | Fail |
| Standalone backtest | PF=1.351, accuracy=0.590, Sharpe=+0.373, maxDD=-49.9%, signals=210 | Mixed / executable |
| Correlation | max core rho=0.388 vs `realized_vol_10` | Pass |

## Lesson

The gate improves trade quality versus the naked signal, but the signal is too sparse
and does not meet the IC promotion gate. Keep as an execution-overlay candidate only;
do not promote to core RL features without a dedicated sparse-event validation rule.

<!-- 2026-04-28: CICC price-volume factors adapted for SOXL -->

## Question

Can low-frequency price-volume factors from CICC cross-sectional research
(动量&反转, 波动率, 流动性, 量价相关性) be adapted into useful
SOXL single-asset timing features?

## Answer: Mostly No, with one IC-only Candidate

Four factors were constructed and evaluated via `scripts/evaluate_factor_pack.py`:

| Factor | IC Mean (best h) | IC+Ratio | Max Core ρ | Sharpe (best) | PF | Acc | Verdict |
|--------|-----------------|----------|-----------|--------------|----|-----|---------|
| CICC_RETURN_VOLUME_SYNC | +0.031 (h=20) | **0.584** | 0.349 | +0.81 | 1.15 | 0.54 | 🟡 IC only |
| CICC_VOLUME_SHOCK_STD | +0.021 (h=20) | 0.589 | 0.306 | +0.66 | 1.14 | 0.53 | ❌ IC<0.03 |
| CICC_VOLUME_LEAD_RETURN | -0.008 (h=20) | 0.482 | 0.065 | +0.81 | 1.15 | 0.54 | ❌ Wrong dir |
| CICC_MOM_RANGE_M | +0.006 (h=10) | 0.510 | **0.981** | +0.75 | 1.17 | 0.53 | ❌ Redundant |

### Key Findings

1. **CICC_RETURN_VOLUME_SYNC** is the only factor with meaningful IC stability
   (mean=+0.031, positive_ratio=0.584, t=14.6).  The direction is **positive**
   (higher return-volume correlation predicts higher future returns), which is
   opposite to the A-stock finding (negative).  This may reflect LETF-specific
   microstructure: high volume-return correlation in SOXL indicates informed
   participation rather than late-stage distribution.

2. **CICC_MOM_RANGE_M** is linearly redundant with `sharpe_proxy_21` (|ρ|=0.981).
   The volatility-scaled momentum concept is already captured by existing core
   features.  Do not retry without a genuinely orthogonal construction.

3. **CICC_VOLUME_LEAD_RETURN** (lead-lag volume-return correlation) has the
   wrong sign and weak stability.  The 5-day lead-lag window may be too short
   for SOXL's daily dynamics.

4. **CICC_VOLUME_SHOCK_STD** has decent positive_ratio (0.589) but IC mean
   (0.021) below the 0.03 gate.  Direction is correct but strength is marginal.

### SHAP 验证（balanced profile 基线）

运行 `scripts/feature_analysis.py` 获取当前模型的 SHAP 重要性：

| SHAP Rank | Feature | mean_abs_shap | Spearman IC |
|-----------|---------|---------------|-------------|
| 1 | sharpe_proxy_21 | 0.049 | -0.032 |
| 2 | realized_skew_21 | 0.028 | — |
| 3 | anchor_gap_180 | 0.022 | — |
| 4 | vix_momentum_10 | 0.022 | — |
| 5 | episode_progress | 0.020 | — |
| 6 | decay_63 | 0.019 | — |
| 7 | realized_kurtosis_21 | 0.015 | — |
| 8 | vol_velocity_5 | 0.014 | — |
| 9 | relative_volume_zscore | 0.013 | +0.035 |
| 10 | below_anchor_ratio_63 | 0.011 | — |

**CICC_RETURN_VOLUME_SYNC 与 SHAP Top-10 的相关性**：

| SHAP Top Feature | |ρ| |
|------------------|-----|
| sharpe_proxy_21 (SHAP #1) | **0.069** |
| vix_momentum_10 (SHAP #4) | **0.001** |
| relative_volume_zscore (SHAP #9) | **0.023** |
| realized_kurtosis_21 (SHAP #7) | 0.349 |

**结论**: CICC_RETURN_VOLUME_SYNC 与模型当前最依赖的特征几乎完全正交。
`sharpe_proxy_21` 的 SHAP 重要性是 #1，但新因子与它只有 0.069 的相关性。
这意味着模型目前没有通过任何现有特征来捕捉"量价同步程度"这个维度。

### Lesson

- Cross-sectional factor ideas do not automatically translate to LETF timing.
- The most promising adaptation (return-volume sync) passes IC gates but fails
  standalone backtest gates (PF<1.2, Acc<0.55).  Treat as **regime-gated or
  composite building block**, not a direct standalone signal.
- **SHAP 基线 + 相关性检查**是判断新因子是否有增量价值的有效前置步骤：
  如果新因子与 SHAP Top-3 特征都低相关（|ρ|<0.20），即使 standalone backtest
  未过门槛，也值得尝试 regime-gated 或加入 RL feature schema 看 SHAP 变化。
- Before retrying "momentum / reversal" adaptations, check correlation with
  `sharpe_proxy_21`, `mom_21`, and `relative_strength_21` — these dimensions
  are already well-covered by the core feature stack.

## Next Valid Experiment Type

Test CICC_RETURN_VOLUME_SYNC under explicit regime gating (e.g. only when
`regime_risk` > 0.6 or `vol_stress_score` > 0.7) to see if the signal becomes
executable in stress periods.  Alternatively, test whether adding it to an
existing composite (e.g. `RB_BUY_COMPOSITE`) improves any metric.

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

<!-- 2026-05-10: release-signals RL pack ablation -->

# 2026-05-10: RELEASE_SIGNALS_V1 RL Injection Ablation

## Question

If the retained release-style factors are injected together into PPO state,
does the actor learn a cleaner recovery classifier?

## Answer: It learns recovery upside, but not selectivity

`release_signals_v1 = RB_COOLDOWN_PROXY + TRANS_VIX_DIVERGENCE + CICC_RETURN_VOLUME_SYNC`
was added as a narrow built-in pack and retrained on top of `balanced_rule_residual_v22`
as `balanced_rule_residual_v22_release_v1`.

Corrected replay under the current harness showed:

| Window | `v22` | `v22_release_v1` | Verdict |
|--------|-------|------------------|---------|
| `test_proxy` | `1099.1% / 0.976 / 53.21% / 0.498` | `1350.3% / 1.010 / 53.26% / 0.520` | Upside improves |
| `test_2022` | `-41.8% / -1.335 / 43.63% / 0.241` | `-46.3% / -1.480 / 48.45% / 0.252` | Defense worsens |
| `test_2023_2024` | `180.2% / 1.336 / 31.91% / 0.486` | `234.3% / 1.479 / 32.80% / 0.509` | Recovery improves |

## Lesson

- The retained release factors are real enough to move policy behavior.
- Direct RL injection makes them act like **ungated bullish pressure**, not a conditional release rule.
- The next valid line is **gated use**:
  fixed-actor overlay, low-churn route, or explicit release classifier.
- Do not widen this into a larger release pack or assume more training epochs will fix it.
