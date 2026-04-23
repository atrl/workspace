# Signal Validation Report

## Feature

- **Name**: {{ feature_name }}
- **Horizon**: {{ horizon }}d
- **Test Period**: {{ start }} ~ {{ end }}

## Backtest Summary

| Metric | Value |
|--------|-------|
| Annualized Return | {{ annualized_return:.2% }} |
| Max Drawdown | {{ max_drawdown:.2% }} |
| Sharpe Ratio | {{ sharpe:.2f }} |
| Signal Accuracy | {{ signal_accuracy:.2% }} |
| Profit Factor | {{ profit_factor:.2f }} |
| Win Rate | {{ win_rate:.2% }} |
| Number of Signals | {{ num_signals }} |
| Turnover | {{ turnover:.2% }} |

## Regime Slice Performance

| Regime | N | IC Mean | Win Rate |
|--------|----|---------|----------|
{% for r in regime_results %}
| {{ r.regime }} | {{ r.n }} | {{ r.ic_mean:+.4f }} | {{ r.win_rate:.2% }} |
{% endfor %}

## Comparison to Rulebase

| Metric | Signal | Rulebase |
|--------|--------|----------|
| Annualized Return | {{ signal.annualized_return:.2% }} | {{ rulebase.annualized_return:.2% }} |
| Max Drawdown | {{ signal.max_drawdown:.2% }} | {{ rulebase.max_drawdown:.2% }} |

## Conclusion

{{ conclusion }}
