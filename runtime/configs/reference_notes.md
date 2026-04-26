# SOXL Optimization Reference Notes

## Local legacy strategy

- `archive/legacy_workspace/soxl_backtest/strategy/buy_strategy.py`
  - Panic score was only a gate.
  - The real trigger was daily or cumulative drops, scaled by a volatility coefficient.
- `archive/legacy_workspace/soxl_backtest/strategy/sell_strategy.py`
  - The system did not just reduce risk.
  - It actively monetized the latest add-on tranche when panic normalized.
- `archive/legacy_workspace/soxl_backtest/config/backtest_config.py`
  - The strategy kept a core position and traded around it.
  - `MAX_POSITION=0.9`, `MIN_POSITION=0.5`, `SELL_AMOUNT=0.1`.
- `archive/legacy_workspace/soxl_backtest/indicators/indicator_calculator.py`
  - The misnamed `MA20` was actually a 180-day anchor.
  - That long anchor is more useful for SOXL than a short moving average when separating bull trend from decay regime.

## External references

- Qlib official repo
  - https://github.com/microsoft/qlib
  - Useful mainly for keeping data, handler, dataset, RL trainer, and official backtest in one consistent pipeline.
- FinRL official repo
  - https://github.com/AI4Finance-Foundation/FinRL
  - Useful for workflow ideas: separate environment, validation, and trading evaluation instead of one monolithic script.
- FinRL-Meta official repo
  - https://github.com/AI4Finance-Foundation/FinRL-Meta
  - Useful idea: benchmark the same agent across different market environments and avoid optimizing only a single aggregate metric.
- Jiang et al., "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"
  - https://arxiv.org/abs/1706.10059
  - Useful takeaway: the action definition matters as much as reward design; portfolio-control structure should match the market microstructure you actually trade.
- Ye et al., "Adversarial Deep Reinforcement Learning in Portfolio Management"
  - https://arxiv.org/abs/1808.09940
  - Useful takeaway: forcing the policy to survive stressed paths can improve robustness more than simply increasing drawdown penalties.
- Leveraged ETF path-dependence literature
  - Representative topic search: https://arxiv.org/search/?query=leveraged+ETF+path+dependence&searchtype=all
  - Useful takeaway: LETF returns depend on path and volatility drag, so long-trend anchor and tracking-error state are first-class features, not optional add-ons.

## What was carried into the new implementation

- Added long-anchor features around the 180-day trend.
- Added explicit cumulative-drop and volatility-coefficient features.
- Switched from a fixed absolute action grid to a dynamic core-and-satellite action band.
- Added training oversampling for 2020, 2022, and 2023 stress-transition windows.

## Next experiments

- Add tranche memory to the simulator: average entry, last add price, and overlay PnL.
- Compare checkpoint selection on full OOS versus 2022-only and 2023-2024-only slices.
- If daily data remains the base frequency, optimize "selective daily rebalancing" rather than trying to simulate true intraday T.
