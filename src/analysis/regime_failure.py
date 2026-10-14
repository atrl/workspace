"""Fixed ex-ante regimes and descriptive accounting; no regime parameter search."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def regime_features(frame):
    x = frame[['soxx_ret5', 'soxx_ret20', 'feature_available_at', 'decision_at']].copy()
    if x[['soxx_ret5', 'soxx_ret20']].isna().any().any():
        raise ValueError('missing fixed regime input')
    if not (pd.to_datetime(x.feature_available_at) < pd.to_datetime(x.decision_at)).all():
        raise ValueError('regime input is not known before decision')
    positive5, positive20 = x.soxx_ret5.ge(0), x.soxx_ret20.ge(0)
    x['regime'] = np.select([positive20 & positive5, ~positive20 & positive5,
                             positive20 & ~positive5],
                            ['uptrend', 'rebound', 'pullback'], default='decline')
    return x


def account_groups(daily, states):
    merged = daily.merge(states[['regime']], left_on='trade_date', right_index=True,
                         how='left', validate='one_to_one')
    if merged.regime.isna().any():
        raise ValueError('missing regime on account calendar')
    merged['quarter'] = pd.to_datetime(merged.signal_date).dt.to_period('Q').astype(str)
    terms = ['reference_cash', 'additional_cash', 'soxl_relative', 'recovery_gross',
             'execution_cost', 'benchmark_residual', 'native_reconciliation_residual', 'relative_log']
    outputs = {}
    for grouping in ['quarter', 'regime']:
        result = merged.groupby(grouping)[terms].sum()
        result['sessions'] = merged.groupby(grouping).size()
        result['mean_cash'] = merged.groupby(grouping).cash_weight_after_open.mean()
        result['relative_wealth_change_pct'] = 100 * np.expm1(result.relative_log)
        outputs[grouping] = result.reset_index()
        np.testing.assert_allclose(result[terms].sum().to_numpy(), merged[terms].sum().to_numpy(), atol=1e-12)
    return merged, outputs


def conditional_information(daily, states, *, repetitions=1000, seed=20261014):
    frame = daily.merge(states[['regime']], left_on='trade_date', right_index=True,
                        how='left', validate='one_to_one')
    if frame.regime.isna().any():
        raise ValueError('missing regime on forecast calendar')
    rng = np.random.default_rng(seed); n = len(frame)
    samples = [(np.concatenate([(start + np.arange(63)) % n for start in
                rng.integers(0, n, int(np.ceil(n / 63)))])[:n]) for _ in range(repetitions)]
    rows = []
    for state in ['uptrend', 'rebound', 'pullback', 'decline']:
        for score in ['frozen_current_cash_weight', 'base_cash_weight']:
            def statistic(part):
                part = part.loc[part.regime.eq(state)].dropna(subset=[score, 'cash_advantage_h5'])
                return spearmanr(part[score], part.cash_advantage_h5).statistic if len(part)>20 else np.nan
            part = frame.loc[frame.regime.eq(state)].dropna(subset=[score, 'cash_advantage_h5'])
            draws = np.array([statistic(frame.iloc[index]) for index in samples])
            low, high = np.nanquantile(draws, [.025, .975])
            annual = [statistic(group) for _, group in frame.groupby(pd.to_datetime(frame.signal_date).dt.year)]
            rows.append({'regime':state,'score':score,'sessions':len(part),'spearman':statistic(frame),
                         'lower':low,'upper':high,'positive_years':sum(v>0 for v in annual),
                         'eligible_years':sum(np.isfinite(v) for v in annual),
                         'mean_cash':part[score].mean(),'mean_cash_advantage_h5':part.cash_advantage_h5.mean()})
    return pd.DataFrame(rows)
