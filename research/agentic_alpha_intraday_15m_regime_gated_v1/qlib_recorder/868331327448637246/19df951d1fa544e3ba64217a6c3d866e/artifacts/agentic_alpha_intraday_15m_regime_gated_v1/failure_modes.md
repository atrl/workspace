# Failure Modes

- The v2 dislocation gate improves historical walk-forward, but this is still a model-bar replay rather than bid/ask or order-book replay.
- Regime gate excludes opening and closing risk bars, but it still assumes bar-close signal calculation, next-bar open entry, and next-bar close exit.
- 20bp stress is positive for the selected 2026 test window, but live spread, queue, PDT, latency, and volatility-halt behavior are not proven.
- qlib PortAnaRecord is not used for this high-frequency single-asset execution probe; large predictions and labels are recorder-only artifacts.
