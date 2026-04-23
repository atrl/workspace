---
date: '2026-04-22'
title: "2026-04-22: CRITICAL CORRECTION \u2014 test_2022 Diagnosis"
kind: decision
profiles:
- soxl
status: reference
tags:
- decision
- eval-2022
- reference
- soxl
- walk-forward
artifacts: []
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: "Legacy memory note normalized into the repo-wide schema. Initial diagnosis\
  \ claimed the model went 100% position right before the Oct 2022 crash: ``` 2022-10-07:\
  \ model_pos=1.00, market=-17.69% \u2192 single day -17.7% ``` Root cause identified\
  \ as"
---

# 2026-04-22: CRITICAL CORRECTION — test_2022 Diagnosis

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## Original (Wrong) Conclusion

Initial diagnosis claimed the model went 100% position right before the Oct 2022 crash:
```
2022-10-07: model_pos=1.00, market=-17.69% → single day -17.7%
```

Root cause identified as: `panic_buy_bonus` causing the model to "catch falling knives".

## The Bug

The diagnosis script had a **critical bug**: it passed discrete action indices (0-10) directly to `sim.step()`, but `SoxlSimulator.step()` expects continuous positions (0.0-1.0). This caused the simulator to interpret action=3 as position=3.0 (clipped to 1.0), making the model appear to be always 100% positioned.

## Corrected Diagnosis (after fixing script)

| Metric | v2 | v2_fix |
|--------|-----|--------|
| Avg position | **0.15** (fixed) | 0.25 (fixed) |
| Position range | [0.15, 0.20] | [0.15, 0.25] |
| Days with pos > 0.5 | **0** | 0 |
| "High pos + crash" days | **0** | 0 |
| Diagnosis MaxDD | **19.5%** | 29.1% |

**The model does NOT go 100% before crashes. It maintains a fixed low position (0.15-0.25) throughout 2022.**

## Why does 15-25% position produce 20-30% MaxDD?

SOXL is a 3x leveraged ETF. In 2022, realized_vol_10 averaged **1.25** (vs normal ~0.3-0.5).

Single-day market drops of -15% to -18% were common:
```
2022-06-16: market=-18.30%, pos=0.15 → portfolio drop = 2.75%
2022-09-13: market=-18.20%, pos=0.15 → portfolio drop = 2.73%
2022-10-07: market=-17.69%, pos=0.15 → portfolio drop = 2.65%
```

Multiple such events compound over the year. Even with very low (15%) position, the cumulative drawdown reaches ~20% (diagnosis) or ~30% (qlib backtest with costs).

## Corrected Conclusion

1. **v2's MaxDD=30.2% is NOT caused by bad timing or excessive position**
2. **It is caused by SOXL's inherent 3x leverage volatility in 2022**
3. **30% MaxDD for SOXL in 2022 is actually quite good** (rulebase had 54.2%)
4. **The panic_buy_bonus removal (v2_fix) was unnecessary** — the model never used it

## Implications

- The focus should shift from "tightening MaxDD below 30%" to **"maximizing walk-forward 2023-2024 returns"**
- v4 (test_2023_2024: 51.8% return, 31.1% MaxDD) is the better candidate than v2 (34.0% return, 29.4% MaxDD)
- The 1.7pp MaxDD difference between v2 and v4 is negligible compared to the 17.8pp return difference

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
