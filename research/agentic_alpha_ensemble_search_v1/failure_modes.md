# Failure Modes

- The selected blend is still a research candidate, not a broker execution rule.
- The macro leg has one-day shorter realized next-return coverage because the macro qlib store ends at the baseline safe date.
- TSMOM/stop legs depend on next-day OHLC stop simulation and may differ from live fill quality around gaps.
- The ensemble was selected after prior branch searches; production promotion should re-run out-of-sample and paper-sim checks.
