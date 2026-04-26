# Factor Knowledge

## Retained Insight

Rulebase-derived timing signals can be useful as independent factors when judged by time-series IC, directional accuracy, profit factor, and correlation budget rather than strategy Sharpe alone.

## Failed Pattern

Do not promote a feature because it improves a local validation window if it increases fixed-split `test_2022` exposure or duplicates existing core features above the correlation budget.

Crypto futures basis / funding-rate analogues should be treated as transition
candidates, not direct core features. `SIGNED_TE_REVERSION_10` showed standalone
Sharpe/PF promise but failed IC direction-stability gates, so it needs regime
gating before any RL promotion.

Stress-release gating improves the executable quality of `SIGNED_TE_REVERSION_10`
(PF=1.351, accuracy=0.590), but the resulting signal is sparse and still fails
the standard IC positive-ratio gate. Treat it as an execution-overlay candidate,
not a core feature, unless the harness adds an explicit sparse-event validation rule.

Crypto liquidation-wick analogues are weaker: `INTRADAY_REJECTION_EVENT` passed
the correlation budget but failed both IC direction stability and standalone
backtest. Do not retry naked wick rejection without a regime gate.

## Use With Skills

- `leveraged-etf-factor-research` for hypothesis generation.
- `leveraged-etf-factor-mining` for IC / stability / correlation checks.
- `soxl-signal-backtest` for independent signal validation before RL changes.

## Evidence Path

- `runtime/factor/`
- `data/soxl_factor_library.db`
- `runtime/factor/knowledge.md`

## Reusable Method

Use the leveraged ETF factor workflow: hypothesize from rulebase or literature, generate the feature, run IC / decay / stability / correlation checks, then run independent signal backtests before touching RL profiles.

## Next Valid Experiment Type

Generalize accepted SOXL factor checks to another 3x ETF with explicit asset metadata and compare whether signal direction, horizon, and correlation budget survive outside semiconductors.
