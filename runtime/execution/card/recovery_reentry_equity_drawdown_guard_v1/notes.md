# Recovery Reentry Equity Drawdown Guard v1

Date: 2026-05-24

## Governance

- Baseline: `v19reg_v22_budget_usage_bull_guard_v1`
- Direct comparison: `recovery_reentry_anchor_confirm_v1`
- Role: `diagnose`
- Changed layer: `execution`
- Target window: strict OOS `2020-01-02..2026-05-21`, with saved
  walk-forward checks for `wf_2020`, `wf_2022`, and `wf_2023_2024`
- Accepted trade-off: reduce strict-OOS drawdown while preserving at least
  `+30pp` excess versus buy-and-hold
- Failure mode: the drawdown governor cuts bull participation or bear-market
  recovery enough that walk-forward evidence regresses
- Comparison-table row key: `recovery_reentry_equity_drawdown_guard_v1`

## External Basis

- [Drawdown Measure in Portfolio Optimization](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID544742_code375170.pdf?abstractid=544742&mirid=1):
  drawdown should be treated as a path-dependent risk measure, not only as a
  final summary statistic.
- [Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912):
  separates trend estimation and position sizing, which matches a fixed-actor
  sizing governor instead of a new PPO branch.
- [Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322):
  repeated parameter trials require strict OOS and walk-forward checks before
  promotion.
- [Qlib](https://arxiv.org/abs/2009.11189): keep the loop inside the qlib
  research, backtest, and deployment harness.

## Root-Cause Readout

`recovery_reentry_anchor_confirm_v1` does not cause the strict-OOS max drawdown.
The max drawdown occurs before the reentry branch is active:

```text
anchor_confirm strict-OOS max drawdown:
peak=2024-03-07 trough=2025-04-08 max_dd=41.59%

anchor_confirm reentry gate:
first=2026-02-25 last=2026-05-21 gate_days=22
```

That means a post-reentry-only profit lock cannot lower the global drawdown
target. This loop therefore tests a default-off account-equity drawdown
governor in the shared two-speed composer.

## Mechanism

The new `EquityDrawdownControlConfig` tracks realized strategy equity inside
`compose_budget_usage_policy_frame()` and linearly caps the target after the
strategy enters drawdown:

```text
start_drawdown = 0.12
full_drawdown = 0.22
min_target = 0.25
sell_trade_threshold = 0.0
sell_trade_lock_days = 0
```

The trade-lock relaxation is sell-only. Buy decisions still use the normal
composer threshold and lock.

## Result

| window | anchor_confirm_v1 | equity_drawdown_guard_v1 | readout |
| --- | ---: | ---: | --- |
| strict OOS return | 1894.61% | 909.85% | still positive excess, but large giveback |
| strict OOS excess vs buyhold | +1078.10pp | +93.33pp | keeps `+30pp` target |
| strict OOS Sharpe | 1.165 | 1.075 | worse |
| strict OOS MaxDD | 41.59% | 34.37% | improves but far above `15%` |
| strict OOS avg_pos | 0.467 | 0.390 | lower exposure |
| `wf_2020` return / MaxDD | 113.89% / 26.14% | 42.68% / 24.37% | bull upside cut too much |
| `wf_2022` return / MaxDD | -20.68% / 26.65% | -26.36% / 28.25% | defensive regression |
| `wf_2023_2024` return / MaxDD | 58.19% / 30.18% | 47.91% / 26.56% | drawdown improves, return gives back |

The quick parameter sweep found that combinations capable of pushing strict
OOS MaxDD near `20%` or lower destroyed buy-and-hold-relative performance. The
selected parameters are the lowest drawdown sweep point that still preserved
more than `+30pp` strict-OOS excess.

Poor-window diagnostics now show lower local drawdowns but persistent negative
excess in violent rebound windows:

```text
2026-02-05..2026-05-06: strategy=28.81%, buyhold=211.46%, excess=-182.65pp, max_dd=4.54%
2025-04-08..2025-07-09: strategy=51.79%, buyhold=223.03%, excess=-171.24pp, max_dd=7.95%
2026-02-10..2026-05-11: strategy=27.24%, buyhold=200.63%, excess=-173.39pp, max_dd=4.54%
```

## Decision

Diagnostic keep, no runtime promotion.

The equity drawdown governor is useful as shared harness capability because it
directly targets the path risk that the reentry branch cannot touch. It is not
yet a valid runtime candidate: it still misses the `15%` MaxDD target, gives
back too much bull participation, and regresses `wf_2022` versus the
anchor-confirmed branch.

Next valid loop: do not keep tightening the global equity cap. The evidence
points to a two-state risk controller: keep the anchor-confirmed reentry for
violent rebounds, but add a regime-aware drawdown controller that only de-risks
when drawdown coincides with broken long-anchor / high-volatility downtrend
conditions. It must be tested against the same strict OOS and walk-forward
windows.

## Artifacts

```text
runtime/execution/card/recovery_reentry_equity_drawdown_guard_v1/recovery_reentry_equity_drawdown_guard_v1.json
runtime/execution/card/recovery_reentry_equity_drawdown_guard_v1/recovery_reentry_equity_drawdown_guard_v1.md
runtime/execution/card/recovery_reentry_equity_drawdown_guard_v1/recovery_reentry_equity_drawdown_guard_v1_soxl_daily.csv
runtime/execution/card/recovery_reentry_equity_drawdown_guard_v1/oos_poor_window_diagnostics/
runtime/execution/card/recovery_reentry_equity_drawdown_guard_v1/walkforward/
```
