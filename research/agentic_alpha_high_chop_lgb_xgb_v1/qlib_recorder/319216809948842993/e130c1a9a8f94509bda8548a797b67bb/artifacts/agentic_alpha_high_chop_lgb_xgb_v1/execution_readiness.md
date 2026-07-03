# Execution Readiness

## Position Rule Assessment

- current_behavior: selected candidate is not a standalone ML position forecast. `base_constrained_ref` supplies nearly all exposure and XGB probability only throttles that base between 0.85 and 1.0 after a validation near-tie implementation tiebreaker.
- formula: `base_constrained_ref = (2.894 * canary_held + 0.0503 * tsmom + 0.0005 * tsmom252).clip(0, 1)`.
- effective_driver: the canary is active when SOXX 60-day return is above 6%, VIX is below 26, and SOXL 10-day return autocorrelation is positive. This makes the rule a trend/risk-on prior, not an ML-discovered target position.
- 2026_evidence: selected latest signal after final 2026-07-02 data is `target_position=0.0004977`, while the repaired test return is 542.007% with max drawdown -14.577%. `base_constrained_ref` alone returns 568.517%, so the ML overlay improves the prior selected LGBM overlay but has not proven superiority over the base prior.
- standalone_lgb_xgb: pure probability/binary LGB/XGB whole-position candidates are retained as diagnostic-only. The best validation standalone candidate returned 199.166% in validation but -17.102% in the 2026 report window, so direct whole-position control is not stable enough for promotion.
- judgment: this is acceptable as a research baseline or prior, but not as a normal production position model. A production model should either predict calibrated expected return/probability and map it through a risk budget, or explicitly use the base rule as a benchmark with a hard risk gate and turnover/holding constraints.
- failure_mode: binary jumps between near-cash and near-full exposure are not acceptable without execution validation. They are especially fragile when the trigger is a small boolean change in `canary_held`.

## Daily Data Repair

- source_chain: Futu daily bars enter `data/raw/external/futu_history_kline_US_*.csv`, merge into `data/processed/panels/soxl_soxx_vix_full.csv`, then rebuild `data/qlib/soxl` via `refresh_qlib_store`.
- root_cause: the stale 2026-07-02 row was written around 2026-07-02 13:00 America/New_York, before the US session had settled. A later append-only refresh did not replace the overlap row.
- repaired_values: qlib `D.features` now reads 2026-07-02 SOXL open 219.23, high 226.13, low 168.88, close 181.47, volume 92,078,832; SOXX close 566.32, volume 16,150,385; VIX 16.15.
- guardrail_added: Futu daily refresh now refuses an end date later than the settled US daily date and defaults to the latest settled US date using an America/New_York 20:15 cutoff. `--replace-overlap` also replaces provider raw rows, not only the canonical panel.
- historical_scope: after a 2026-06-01 through 2026-07-02 overlap replacement, OpenD final daily bars and the canonical panel have no SOXL/SOXX OHLCV differences above tolerance for that window. Full maintenance should still periodically rerun a rolling overlap replacement after US settlement and diff against OpenD final daily bars.

## Minute Data Sources

- Futu OpenD: already works locally for SOXL/SOXX 1m. Official `request_history_kline` supports paging and `extended_time`; docs also expose US session handling, with `ALL` for 24H historical k-line and no `OVERNIGHT` input. Best immediate source for recent execution validation, but entitlement/history depth must be treated as account-dependent.
- Polygon/Massive stocks aggregates: official custom bars endpoint supports `multiplier/timespan` minute-style bars across pre-market, regular, and after-hours sessions. Docs list Stocks Developer with 10 years and Stocks Advanced with all history back to 2003. Best candidate for durable full-history SOXL/SOXX 1m backfill if API credentials are available.
- Databento: official historical API supports client libraries and schemas including `ohlcv-1m`; metadata endpoints expose dataset availability/condition. Best institutional-quality candidate, especially if execution-quality direct-feed data is required.
- Alpaca: official historical bars endpoint supports stock aggregates with `sip`, `iex`, `boats`, and `otc` feeds, adjustment options, pagination, and over 7 years of data on its market-data product page. Useful if an Alpaca data key is available; SIP is preferable to IEX for execution validation.
- Tiingo IEX: official docs provide historical intraday endpoint, but it is IEX-only. This is useful as a cheap sanity reference, not sufficient as the main execution-quality source for SOXL.
- Nasdaq Data Link: official Bars endpoint advertises 10+ years for all US-listed and OTCBB securities with interval OHLCV. Viable as a paid alternative, but subscription and field semantics must be checked before integration.

## Required Next Implementation

- build_intraday_store: create a qlib-backed intraday feature store from normalized 1m bars using the existing schema `provider,symbol,interval,timestamp,date,session,open,high,low,close,volume,turnover,source_tz,adjusted,fetched_at`.
- train_features: add lagged day-level features derived from minute bars first: overnight/extended-hours return, premarket return, first-30-minute return/range, RTH realized volatility, close location, close-volume concentration, ETH liquidity, next-open gap risk, and intraday adverse excursion labels.
- validation_gate: before production review, report close-to-close, after-hours-fill, next-open, and next-session intraday MAE/MFE results. Promotion requires the selected model to pass max drawdown and intraday adverse-excursion gates under executable timing.
- data_source_gate: Futu is sufficient for recent June/July 2026 execution validation; full 2019-present model training needs a longer source such as Polygon/Massive, Databento, Alpaca SIP, or Nasdaq Data Link.

## References

- Qlib API reference: https://qlib.readthedocs.io/en/v0.8.3/reference/api.html
- Futu historical candlesticks: https://openapi.futunn.com/futu-api-doc/en/quote/request-history-kline.html
- Polygon/Massive custom stock bars: https://polygon.io/docs/rest/stocks/aggregates/custom-bars
- Databento historical API: https://databento.com/docs/api-reference-historical
- Alpaca historical stock bars: https://docs.alpaca.markets/us/reference/stockbars
- Alpaca market data product page: https://alpaca.markets/data
- Tiingo IEX intraday docs: https://www.tiingo.com/documentation/iex
- Nasdaq Data Link API Bars: https://www.nasdaq.com/solutions/data/nasdaq-data-link/api
