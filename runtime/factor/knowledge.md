# Factor Knowledge

Generated: auto | Scope: SOXL feature stack (core + rulebase + academic + transition)

## Factor Comparison Table

All features with **backtest evidence** or **IC evidence** are listed. Core features without standalone validation are summarized in the [Core Features Summary](#core-features-summary) section below.

| Factor | Source | Category | Formula (simplified) | Dir | IC(h=5) | Best Sharpe | Max |rho| | Verdict |
|--------|--------|----------|----------------------|-----|---------|-------------|------------|---------|
| RB_PANIC_SCORE_PROXY | rulebase | sentiment | sigmoid_vix_zscore * relative_volume_zscore + r... | negative | — | +0.172 | — | 🟡 Weak |
| RB_DAILY_DROP_SEVERITY | rulebase | drawdown | daily_return / (vol_coef_21 + 1e-8) | negative | — | — | — | ⚪ Unverified |
| RB_CUMULATIVE_DROP_SEVERITY | rulebase | drawdown | cum_drop_30 / (vol_coef_21 + 1e-8) | negative | — | — | — | ⚪ Unverified |
| RB_RSI6_OVERSOLD | rulebase | momentum | (30 - rsi_6 * 100).clip(lower=0) / 30.0 | negative | — | -0.158 | — | ❌ Invalid |
| RB_RSI6_OVERBOUGHT | rulebase | momentum | (rsi_6 * 100 - 70).clip(lower=0) / 30.0 | positive | — | — | — | ⚪ Unverified |
| RB_VOL_REGIME | rulebase | volatility | vol_coef_21 | negative | — | — | — | ⚪ Unverified |
| RB_BUY_COMPOSITE | rulebase | composite | panic_score_proxy * 0.3 + daily_drop_severity *... | negative | — | +0.115 | — | 🟡 Weak |
| RB_SELL_COMPOSITE | rulebase | composite | rsi6_overbought * 0.4 + profit_position_proxy *... | positive | — | — | — | ⚪ Unverified |
| RB_PROFIT_TAKE_PROXY | rulebase | drawdown | (cum_drop_30 + 0.30).clip(lower=0, upper=0.60) ... | positive | — | — | — | ⚪ Unverified |
| RB_COOLDOWN_PROXY | rulebase | timing | days_since_high_63 * (1 - below_anchor_ratio_63) | negative | — | +0.938 | — | ✅ Strong |
| RB_STOP_LOSS_PROXY | rulebase | drawdown | (cum_drop_30 + 0.20).clip(lower=0, upper=0.20) ... | positive | — | — | — | ⚪ Unverified |
| RB_VOLUME_STRESS | rulebase | liquidity | volume_shock * vol_stress_score | negative | — | +0.198 | — | 🟡 Weak |
| RB_TREND_REVERSAL | rulebase | trend | anchor_gap_180 * (1 + realized_vol_21) | negative | — | — | — | ⚪ Unverified |
| RB_VIX_TERM_STRESS | rulebase | sentiment | vix_term_slope * vix_short_ratio | negative | — | — | — | ⚪ Unverified |
| ACAD_AMIHUD_ZSCORE | academic | mean-reversion | price / (volume * spread) | negative | — | +0.284 | — | 🟡 Weak |
| ACAD_INTRADAY_REVERSAL_21 | academic | mean-reversion | (close-open)/(high-low) rolling 21 | negative | — | +0.326 | 0.73 | 🟡 Weak |
| ACAD_VOLATILITY_DRAG | academic | momentum | decay_63 vs vol drag estimate | negative | — | -0.032 | — | ❌ Invalid |
| ACAD_DOWN_UP_RATIO | academic | mean-reversion | down/up vol ratio 21d | negative | — | +0.046 | — | 🟡 Weak |
| ACAD_DOWN_UP_RATIO_63 | academic | mean-reversion | down/up vol ratio 63d | negative | — | +0.129 | — | 🟡 Weak |
| ACAD_DOWN_RUNS | academic | mean-reversion | consecutive down days | negative | — | -0.289 | — | ❌ Invalid |
| TRANS_VIX_DIVERGENCE | transition | mean-reversion | vix_fall_pct * price_near_low | negative | — | +1.041 | — | ✅ Strong |
| TRANS_TE_ZSCORE | transition | mean-reversion | tracking_error z-score 21d | negative | +0.072 | — | — | 🟡 IC only |
| regime_risk | core | composite | vix% + vol + dd + downside + ... | negative | — | — | 0.90 | ✅ Core |
| regime_risk_no_vix | core | composite | vol_stress + vol + dd + ... (no VIX) | negative | — | — | 0.90 | 🟡 Ablated |
| vol_stress_score | core | composite | rv_z + dd_z + ir_z + vol_shock + kurt | negative | — | — | 0.54 | 🟡 VIX-free proxy |

### Verdict Legend

- ✅ **Strong** — Sharpe > 0.5 and profit factor > 1.1
- 🟡 **Weak** — 0 < Sharpe < 0.5, or IC-only evidence
- ❌ **Invalid** — Sharpe <= 0 or IC < 0 with high confidence
- ⚪ **Unverified** — No standalone backtest or IC run yet

## Key Findings

### Best Standalone Signals (by Sharpe)

1. **TRANS_VIX_DIVERGENCE**: Sharpe=+1.041, PF=1.43, n=58
1. **RB_COOLDOWN_PROXY**: Sharpe=+0.938, PF=1.40, n=32
1. **ACAD_INTRADAY_REVERSAL_21**: Sharpe=+0.326, PF=1.21, n=28
1. **ACAD_AMIHUD_ZSCORE**: Sharpe=+0.284, PF=1.21, n=6
1. **RB_VOLUME_STRESS**: Sharpe=+0.198, PF=1.13, n=120

### Highest Redundancy (|rho| > 0.85)

- **mom_21** ↔ **relative_strength_21**: |rho| = 1.000
- **relative_strength_21** ↔ **mom_21**: |rho| = 1.000
- **drawdown_21** ↔ **cum_drop_20**: |rho| = 0.993
- **cum_drop_20** ↔ **drawdown_21**: |rho| = 0.993
- **vix_zscore** ↔ **vix_percentile_252**: |rho| = 0.986
- **vix_percentile_252** ↔ **vix_zscore**: |rho| = 0.986
- **sharpe_proxy_21** ↔ **relative_strength_21**: |rho| = 0.974
- **mom_10** ↔ **anchor_gap_velocity_10**: |rho| = 0.947
- **anchor_gap_velocity_10** ↔ **mom_10**: |rho| = 0.947
- **volume_shock** ↔ **relative_volume_zscore**: |rho| = 0.941

### IC-Verified Only

- **TRANS_TE_ZSCORE**: IC=+0.0724, pos_ratio=0.563 — needs backtest

## Core Features Summary

50 core features organized by theme. See `src/soxl_quant/data.py` for exact formulas.

| Theme | Features | Count | Key Driver |
|-------|----------|-------|------------|
| Trend / Anchor | anchor_gap_180, anchor_slope_180, trend_gap_21/63, trend_slope_21/63 | 6 | Long-term regime |
| Momentum | mom_5/10/21/63, relative_strength_21, rsi_14/6 | 7 | Short-term direction |
| Volatility | realized_vol_10/21/63, vol_of_vol_21, vol_coef_21, intraday_range | 6 | Risk regime |
| Drawdown | drawdown_21/63, cum_drop_20/30, days_since_high_63, loss_streak_21 | 7 | Stress depth & duration |
| Structure | below_ma63_days, below_anchor_ratio_63, new_low_ratio_21, rolling_new_low_ratio_63 | 4 | Structural weakness |
| Downside | downside_ratio_10/21, downside_semivol_21, negative_gap_frequency_21 | 4 | Downside asymmetry |
| VIX | vix_level, vix_zscore, vix_percentile_252, vix_momentum_10 | 4 | Market fear gauge |
| Tracking Error | tracking_error_5/21, tracking_error_vol_21, decay_63 | 4 | LETF-specific friction |
| Velocity | anchor_gap_velocity_10, trend_gap_21_velocity, rsi_14_velocity_5, regime_risk_velocity_5, volume_shock_velocity_5 | 5 | Rate-of-change in regime |
| Proxy | expectancy_proxy_21, sharpe_proxy_21, realized_skew_21, gain_streak_10 | 4 | Derived performance metrics |
| Composite | regime_risk, regime_risk_no_vix, vol_stress_score | 3 | Multi-signal stress index |

## Rulebase Hypotheses Detail

### RB_PANIC_SCORE_PROXY
- **Category**: sentiment
- **Expected direction**: negative (horizon: 5d)
- **Priority**: high
- **Expression**: `sigmoid_vix_zscore * relative_volume_zscore + rsi6_adjustment`
- **Logic**: VIX驱动的恐慌情绪(sigmoid映射) × 成交量放大器 + RSI6超卖调节
- **Risks**: VIX数据缺失时退化为vol_stress_score
- **Backtest**: Sharpe=+0.172, PF=1.13, accuracy=0.549, signals=123

### RB_DAILY_DROP_SEVERITY
- **Category**: drawdown
- **Expected direction**: negative (horizon: 3d)
- **Priority**: high
- **Expression**: `daily_return / (vol_coef_21 + 1e-8)`
- **Logic**: 日跌幅经波动率标准化，极端值暗示反转
- **Risks**: 高波动期可能连续下跌

### RB_CUMULATIVE_DROP_SEVERITY
- **Category**: drawdown
- **Expected direction**: negative (horizon: 10d)
- **Priority**: high
- **Expression**: `cum_drop_30 / (vol_coef_21 + 1e-8)`
- **Logic**: 30日累积跌幅经波动率标准化，深度累积后反弹概率高
- **Risks**: 熊市中可能持续累积下跌

### RB_RSI6_OVERSOLD
- **Category**: momentum
- **Expected direction**: negative (horizon: 5d)
- **Priority**: medium
- **Expression**: `(30 - rsi_6 * 100).clip(lower=0) / 30.0`
- **Logic**: RSI6超卖程度量化，0-1区间，越接近1超卖越严重
- **Risks**: 暴跌中RSI可能长期钝化
- **Backtest**: Sharpe=-0.158, PF=1.42, accuracy=0.592, signals=78

### RB_RSI6_OVERBOUGHT
- **Category**: momentum
- **Expected direction**: positive (horizon: 5d)
- **Priority**: medium
- **Expression**: `(rsi_6 * 100 - 70).clip(lower=0) / 30.0`
- **Logic**: RSI6超买程度量化，0-1区间，越接近1超买越严重
- **Risks**: 强趋势中RSI可能长期高位

### RB_VOL_REGIME
- **Category**: volatility
- **Expected direction**: negative (horizon: 5d)
- **Priority**: medium
- **Expression**: `vol_coef_21`
- **Logic**: 波动率系数 = HV_20 / 历史中位数，>1高波动 <1低波动
- **Risks**: 高波动可能持续

### RB_BUY_COMPOSITE
- **Category**: composite
- **Expected direction**: negative (horizon: 5d)
- **Priority**: high
- **Expression**: `panic_score_proxy * 0.3 + daily_drop_severity * 0.4 + cum_drop_severity * 0.3`
- **Logic**: rulebase买入条件的综合信号：恐慌 + 日跌 + 累积跌
- **Risks**: 复合信号可能过度拟合历史买入点
- **Backtest**: Sharpe=+0.115, PF=1.11, accuracy=0.552, signals=290

### RB_SELL_COMPOSITE
- **Category**: composite
- **Expected direction**: positive (horizon: 3d)
- **Priority**: high
- **Expression**: `rsi6_overbought * 0.4 + profit_position_proxy * 0.3 + stop_loss_proxy * 0.3`
- **Logic**: rulebase卖出条件的综合信号：超买 + 止盈 + 止损
- **Risks**: 需要持仓状态信息，非纯价格特征

### RB_PROFIT_TAKE_PROXY
- **Category**: drawdown
- **Expected direction**: positive (horizon: 3d)
- **Priority**: medium
- **Expression**: `(cum_drop_30 + 0.30).clip(lower=0, upper=0.60) / 0.60`
- **Logic**: 从深度回撤中恢复的程度，接近1暗示获利了结
- **Risks**: 仅反映价格恢复，非真实持仓盈亏

### RB_COOLDOWN_PROXY
- **Category**: timing
- **Expected direction**: negative (horizon: 10d)
- **Priority**: low
- **Expression**: `days_since_high_63 * (1 - below_anchor_ratio_63)`
- **Logic**: 距离高点天数 × 脱离底部程度，高值暗示冷却期结束
- **Risks**: 间接信号，噪声较大
- **Backtest**: Sharpe=+0.938, PF=1.40, accuracy=0.588, signals=32

### RB_STOP_LOSS_PROXY
- **Category**: drawdown
- **Expected direction**: positive (horizon: 1d)
- **Priority**: medium
- **Expression**: `(cum_drop_30 + 0.20).clip(lower=0, upper=0.20) / 0.20`
- **Logic**: 接近-20%止损线的程度，高值暗示止损风险
- **Risks**: 止损后可能立即反弹

### RB_VOLUME_STRESS
- **Category**: liquidity
- **Expected direction**: negative (horizon: 5d)
- **Priority**: medium
- **Expression**: `volume_shock * vol_stress_score`
- **Logic**: 成交量异动 × 波动压力得分，高值暗示情绪极端
- **Risks**: 高成交量可能伴随趋势延续
- **Backtest**: Sharpe=+0.198, PF=1.13, accuracy=0.552, signals=120

### RB_TREND_REVERSAL
- **Category**: trend
- **Expected direction**: negative (horizon: 10d)
- **Priority**: medium
- **Expression**: `anchor_gap_180 * (1 + realized_vol_21)`
- **Logic**: 长期均线偏离 × 波动率放大，极端负值暗示反转
- **Risks**: 趋势延续时间不确定

### RB_VIX_TERM_STRESS
- **Category**: sentiment
- **Expected direction**: negative (horizon: 5d)
- **Priority**: medium
- **Expression**: `vix_term_slope * vix_short_ratio`
- **Logic**: VIX期限结构斜率 × 短期VIX比率，异常结构暗示压力
- **Risks**: 需要VIX3M/VIX9D数据
