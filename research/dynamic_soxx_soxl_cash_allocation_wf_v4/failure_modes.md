# Failure Modes

- Hindsight oracle can satisfy the asset-level target only by knowing many future worst SOXX days.
- LGB/XGB action classifiers can rank some SOXL-upside and tail-risk days, but the signal is not sharp enough to both hold high SOXL exposure in trend rallies and exit before drawdown clusters.
- Refreshed VIX/VVIX/SKEW tail-risk fields and survivorship-biased SOXX constituent diffusion features improve diagnostics but do not close the return/drawdown frontier.
- Drawdown-safe signal-capacity return remains 127.4562% versus SOXX benchmark 413.6565%.
- Excess-return candidates still have shallowest drawdown -26.8476%, which is far below the -10% target.
- The current feature set is more suitable as a risk overlay or execution context than a standalone crash-avoidance engine.
