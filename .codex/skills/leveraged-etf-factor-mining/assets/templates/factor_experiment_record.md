# Factor Experiment Record

## Basic Information

- **Feature Name**: {{ name }}
- **Category**: {{ category }}
- **Source**: {{ source }}
- **Created At**: {{ created_at }}
- **Status**: {{ status }}  <!-- Candidate / Accepted / Rejected -->

## Hypothesis

{{ market_logic }}

- Expected Direction: {{ expected_direction }}
- Expected Horizon: {{ expected_horizon }}d
- Priority: {{ priority }}

## Expression

```python
{{ expression }}
```

## IC Test Results

| Horizon | IC Mean | IC t-stat | Positive Ratio |
|---------|---------|-----------|----------------|
{% for r in ic_results %}
| {{ r.horizon_days }}d | {{ r.ic_mean:+.4f }} | {{ r.ic_t:+.2f }} | {{ r.ic_positive_ratio:.2% }} |
{% endfor %}

## Signal Backtest Results

| thr_long | thr_short | Accuracy | PF | Sharpe | WinRate | MDD | n_signals |
|----------|-----------|----------|----|--------|---------|-----|----------|
{% for r in backtest_results %}
| {{ r.threshold_long:.3f }} | {{ r.threshold_short:.3f }} | {{ r.signal_accuracy:.2% }} | {{ r.profit_factor:.2f }} | {{ r.sharpe:.2f }} | {{ r.win_rate:.1% }} | {{ r.max_drawdown:.2% }} | {{ r.num_signals }} |
{% endfor %}

## Stability Analysis

- Stability Score: {{ stability.stability_score:.2% }}
- IC Recent vs Full: {{ stability.ic_recent_vs_full:.2f }}
- Max DD Days: {{ stability.max_dd_days }}

## Decision

- [ ] **Accept** — All gates passed, ready for production
- [ ] **Iterate** — Partial pass, needs refinement
- [ ] **Drop** — Failed gates, not suitable

## Notes

{{ notes }}

## Next Action

{{ next_action }}
