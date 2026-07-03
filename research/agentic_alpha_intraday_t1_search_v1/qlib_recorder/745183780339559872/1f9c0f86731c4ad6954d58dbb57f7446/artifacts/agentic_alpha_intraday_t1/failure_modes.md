# Failure Modes

- intraday_t1_lgbm_replay: qlib DatasetH/LGBM candidate search uses expanded T+1 intraday factors but the best 20% drawdown-gated path remains below the full production return baseline.
- intraday_t1_xgb_replay: qlib DatasetH/XGB candidate search uses the same expanded intraday factor set and is kept as a model-family cross-check, not a production proof.
- intraday_t1_overlay: direct intraday open-range/close-location/realized-volatility/amplitude/drawdown gates reduce risk but still fail the return gate versus production/qlib_paper_sim_v1.
- coverage: Futu regular-session intraday cache now covers 2018-12-31 through 2026-07-02 and T+1 features cover 2019-01-02 through 2026-07-02; the remaining gap is promotion quality, execution validation, and optional extended-hours/provider cross-checking.
