# Failure Modes

- Hindsight oracle can satisfy the asset-level target only by knowing many future worst SOXX days.
- LGB/XGB tail classifiers improve with lagged 15m factors, but the improvement does not close the return/drawdown frontier.
- Drawdown-safe signal-capacity return remains 58.0310% versus SOXX benchmark 413.6565%.
- Excess-return candidates still have shallowest drawdown -23.1993%, which is far below the -10% target.
- The current feature set is more suitable as a risk overlay or execution context than a standalone crash-avoidance engine.
