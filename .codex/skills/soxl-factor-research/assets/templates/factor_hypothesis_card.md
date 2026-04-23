# Factor Hypothesis Card

## {{ name }}

### Expression

```python
{{ expression }}
```

### Metadata

| Field | Value |
|-------|-------|
| Category | {{ category }} |
| Source | {{ source }} |
| Expected Direction | {{ expected_direction }} |
| Expected Horizon | {{ expected_horizon }}d |
| Priority | {{ priority }} |

### Market Logic

{{ market_logic }}

### Risks

{{ risks }}

### Validation Plan

1. [ ] Time-series IC test (1d/5d/10d/20d)
2. [ ] Signal backtest with threshold sweep
3. [ ] Correlation check vs existing features
4. [ ] Regime slice analysis
5. [ ] Stability analysis (rolling 63d IC)

### References

- {{ source }}
