# Failure Modes

- intraday_t1_lgbm_replay: qlib DatasetH/LGBM candidate search uses the T+1 intraday feature store but the best 20% drawdown-gated path remains below the full production return baseline.
- intraday_t1_overlay: direct intraday open-range/close-location/realized-volatility gates reduce risk but still fail the return gate versus production/qlib_paper_sim_v1.
- coverage: the intraday cache begins in 2024 and therefore cannot close the full 2019-2026 promotion gate without either older intraday data or a separate validated daily fallback.
