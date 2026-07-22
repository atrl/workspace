# Failure modes

- The 21-day attack classifier can drift across regimes; tune performance is not proof of validation stability.
- SOXS is a daily-reset inverse leveraged ETF, so even a three-day overlay is path dependent and can decay in chop.
- Cash has zero return because a point-in-time Treasury return series is not in this provider.
- A high defense threshold can miss the beginning of a crash; a low threshold can cause chronic cash drag.
- The 2023-2026 outer interval was previously observed and remains diagnostic only.
