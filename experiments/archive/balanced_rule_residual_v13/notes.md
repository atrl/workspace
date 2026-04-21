# v13 Design Notes

## v12 Breakthrough

v12 achieved the **first drawdown improvement over v2**:
- v2: 521.3% return, Sharpe 0.86, MaxDD 50.4%
- v12: 492.3% return, Sharpe 0.85, MaxDD **48.8%**

The GRU architecture + terminal penalty enabled the agent to learn temporal credit assignment for drawdown control.

## v13: Tune for Return/Drawdown Balance

v12 was too conservative — it sacrificed ~50pp return for ~1.6pp drawdown improvement. v13 seeks a better balance.

### Changes from v12

| Parameter | v12 | v13 | Rationale |
|-----------|-----|-----|-----------|
| episode_drawdown_penalty | 6.0 | **4.0** | Less aggressive penalty → more risk-taking |
| episode_drawdown_threshold | 0.10 | **0.15** | Higher tolerance before penalty kicks in |
| downside_asymmetry_coef | 1.5 | **1.3** | Less loss aversion → more balanced behavior |

### Hypothesis

v12 proved the GRU can learn drawdown control. By reducing the penalty strength, the agent should:
1. Still keep drawdown under 50% (GRU memory + smaller penalty)
2. Recover 20-40pp of lost return
3. Maintain or improve Sharpe ratio

### Target

- MaxDD < 50% (keep the breakthrough)
- Return > 510% (recover half the lost return)
- Sharpe > 0.85
