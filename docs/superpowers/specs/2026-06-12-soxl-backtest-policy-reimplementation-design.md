# SOXL Backtest Policy Reimplementation Design

Date: 2026-06-12
Status: approved design, pending implementation plan
Direction: production

## Objective

Reimplement the legacy non-qlib `soxl_backtest` rule model inside the current
SOXL harness and validate that inference produces approximately the same trade
timing and window returns as the legacy implementation.

The goal is not to train a new actor and not to optimize returns. The goal is a
repo-owned deterministic rule policy whose features and stateful trading logic
match the existing non-qlib `soxl_backtest` behavior closely enough to make it
replayable in the current framework.

## Acceptance Criteria

- Use existing local repo data only. Do not fetch new market data for this
  migration.
- Recompute legacy-compatible features in the new implementation, rather than
  using qlib-native approximations when they differ.
- Keep the legacy `soxl_backtest` package as an oracle for tests and notebook
  validation, not as the new implementation path.
- Validate 2026 YTD fresh-start inference against the legacy evidence:
  - window: `2026-01-02` through `2026-06-09`
  - approximate legacy return: `132.08%`
  - expected trade dates near `2026-03-30`, `2026-05-27`, and `2026-06-05`
- Validate June 2026 fresh-start inference against the legacy evidence:
  - window: `2026-06-01` through `2026-06-09`
  - approximate legacy return: `3.14%`
  - expected trade date near `2026-06-05`
- Persist proof in an executable runtime notebook under
  `runtime/production/card/soxl_backtest_policy_reimplementation_v1/`.
- Include sidecar CSV/JSON artifacts for new-policy daily NAV, trades, legacy
  comparison metrics, and mismatch diagnostics.

## Baseline Evidence

Existing baseline artifacts live in
`runtime/production/card/soxl_backtest_jan_june_windows_v1/`.

Relevant legacy metrics:

- YTD fresh-start: rulebase return `132.07797795135497%`, trade count `3`, end
  position `15.91%`.
- June fresh-start: rulebase return `3.1389599999999973%`, trade count `1`, end
  position `29.94%`.

Relevant legacy trades:

- `2026-03-30`: buy, triggered by cumulative drop.
- `2026-05-27`: take profit.
- `2026-06-05`: buy, triggered by daily drop.
- June fresh-start `2026-06-05`: buy.

These values are the comparison oracle for the new implementation.

## Proposed Architecture

Add a small deterministic policy module under `src/soxl_quant/policy/` or
`src/soxl_quant/app/` with these units:

- `LegacySoxlFeatureBuilder`
  - builds the same columns used by non-qlib `soxl_backtest`: `price`, `vix`,
    `RSI6`, `MA5`, misnamed `MA20` as a 180-day moving average, `HV_20`,
    `HV_30`, `VOL_COEF`, and `panic_score`.
  - uses the existing local CSV/panel data surfaces already present in the
    repo.
- `LegacySoxlPolicyState`
  - stores cash, shares, average price, tranche stack, current position,
    last-buy date, cumulative-buy cooldown, and whether initial build is
    complete.
- `LegacySoxlRulePolicy`
  - implements the buy and sell conditions from non-qlib `soxl_backtest`:
    initial build, overvalued/undervalued buy step handling, daily and
    cumulative drop triggers, panic gates, minimum trade size, stop loss,
    take-profit, and last-tranche profit taking.
- `replay_legacy_soxl_policy`
  - runs a fresh-start replay for a date window and returns daily NAV, position,
    trade rows, metrics, and diagnostics.

The implementation should be independent of qlib actor inference. It may emit
the same dashboard daily CSV shape as actor replays, but it should not be forced
through `RLStrategy` because qlib order execution and 100-share lot semantics
would make exact legacy logic harder to preserve.

## Data Flow

1. Load existing SOXL price and VIX data from local repo files.
2. Build legacy-compatible feature frame with the same warmup behavior as
   `soxl_backtest`.
3. Run the deterministic policy over the requested window as a fresh-start
   account.
4. Compute returns, drawdown, average/end position, trade count, trade date
   matching, and return deltas versus legacy artifacts.
5. Write notebook conclusion plus sidecars under the runtime production card.

## Error Handling

- If required local SOXL/VIX columns are missing, fail with a clear message that
  identifies the missing column and checked path.
- If the new replay cannot cover the requested window, fail closed rather than
  silently shortening the comparison.
- If the legacy oracle artifacts are missing, the notebook should regenerate the
  oracle through `SOXLBacktest` and record that path in the conclusion.

## Testing

Follow test-first implementation:

- Unit test feature parity on a fixed local window:
  - assert `MA20` is a 180-day moving average.
  - assert `RSI6`, volatility, `VOL_COEF`, and `panic_score` match the legacy
    feature builder within numeric tolerance.
- Unit test rule decisions on known 2026 trigger rows:
  - YTD fresh-start buys around `2026-03-30`.
  - take-profit around `2026-05-27`.
  - daily-drop buy around `2026-06-05`.
- Integration test replay metrics:
  - YTD return within a small tolerance of `132.08%`.
  - June return within a small tolerance of `3.14%`.
  - trade-date overlap meets the accepted near-match threshold.

## Deliverables

- New deterministic rule policy module in `src/soxl_quant/`.
- Focused tests for feature parity, rule timing, and replay metrics.
- Runtime notebook:
  `runtime/production/card/soxl_backtest_policy_reimplementation_v1/soxl_backtest_policy_reimplementation_v1.ipynb`.
- Sidecars:
  - `new_policy_daily.csv`
  - `new_policy_trades.csv`
  - `legacy_comparison_metrics.csv`
  - `trade_timing_diagnostics.csv`
  - `summary.json`

## Out Of Scope

- Fetching fresh data.
- Training a neural actor to imitate the rule policy.
- Replacing the current production composer.
- Changing scheduler, broker execution, or Vercel dashboard behavior unless a
  later implementation step explicitly chooses to expose this rule policy there.
- Updating `docs/active_workpad.md` unless the result becomes durable
  production coordination state.

## Implementation Notes

The current worktree already contains unrelated data and replay CSV changes.
Implementation should avoid staging or modifying those files unless the user
explicitly asks to include them.

The proof notebook remains the primary single-run conclusion. Markdown or JSON
sidecars should support reproducibility but should not replace the notebook.
