# Failure Modes

- 2024 walk-forward remains negative, so this is an optimized candidate rather than a promoted production strategy.
- Regime gate excludes opening and closing risk bars, but it is still a bar-close to next-bar open/close simulation without bid/ask replay.
- 20bp stress is positive for the selected 2026 test window, but live spread, queue, PDT, latency, and volatility-halt behavior are not proven.
- qlib PortAnaRecord is not used for this high-frequency single-asset execution probe; large predictions and labels are recorder-only artifacts.
