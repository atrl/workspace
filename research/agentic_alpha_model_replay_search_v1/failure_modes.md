# Failure Modes

- production_signal_overlay: existing production signal can exceed the baseline return only while retaining deep drawdowns, so the drawdown gate fails.
- qlib_daily_lgbm_replay: walk-forward daily LGBM selectors that pass MaxDD < 20% retain too little exposure to beat the production baseline.
- cross_asset_lgbm_replay: auxiliary qlib stores add TQQQ/QQQ, TECL/XLK, SPXL/SPY, and UPRO/SPY context but only cover through 2026-04-10 for the aligned forward-return test, so it cannot prove the full goal and still fails the return gate.
