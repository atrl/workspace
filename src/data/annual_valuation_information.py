"""Monthly research inputs from native Qlib marks and qualified basket fields."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MARKET = ['soxx_logmom63', 'soxx_logvol63', 'soxx_drawdown126', 'relative_logvol21']
VALUATION = ['signed_annual_ep', 'annual_earnings_change', 'loss_weight',
             'annual_ep_rank', 'growth_missing', 'rank_missing']
TARGETS = ['soxx_log_return', 'relative_soxl_log_return']


def monthly_intervals(calendar):
    days = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    rows = []
    for last, following in zip(days[:-1], days[1:]):
        if last.to_period('M') != following.to_period('M'):
            rows.append({'signal_date': last,
                         'signal_at': (last+pd.Timedelta('16h15min')).tz_localize('America/New_York'),
                         'entry_at': (following+pd.Timedelta('9h47min')).tz_localize('America/New_York')})
    result = pd.DataFrame(rows).set_index('signal_date')
    result['exit_at'] = result.entry_at.shift(-1)
    result['execution_year'] = result.entry_at.dt.year
    return result


def add_optional_rank(frame, *, lookback_months=60, minimum_prior_snapshots=36):
    result = frame.copy()
    result['annual_ep_rank'] = np.nan
    result['rank_prior_count'] = 0
    for date, row in result.iterrows():
        if not row.source_qualified:
            continue
        prior = result.loc[(result.index < date) & (result.index >= date-pd.DateOffset(months=lookback_months))
                           & result.source_qualified & result.method_id.eq(row.method_id), 'signed_annual_ep'].dropna()
        result.loc[date, 'rank_prior_count'] = len(prior)
        if len(prior) >= minimum_prior_snapshots:
            result.loc[date, 'annual_ep_rank'] = float((prior.lt(row.signed_annual_ep).sum()+.5*prior.eq(row.signed_annual_ep).sum())/len(prior))
    result.loc[~result.growth_source_qualified, 'annual_earnings_change'] = np.nan
    result['growth_missing'] = result.annual_earnings_change.isna().astype(float)
    result['rank_missing'] = result.annual_ep_rank.isna().astype(float)
    return result


def read_market_inputs(root, plan):
    import qlib
    from qlib.data import D
    root = Path(root)
    for relative, digest in plan['provider_hashes'].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen market provider changed: '+relative)
    qlib.init(provider_uri=str(root/plan['provider_uri']), region='us', kernels=1)
    calendar = D.calendar(freq='1min')
    prices = D.features(['SOXL', 'SOXX'], ['$execution_price'], freq='1min',
                        start_time='2018-10-12', end_time=plan['price_as_of']+' 16:00:00')
    assets = {s: prices.xs(s, level='instrument')['$execution_price'].sort_index() for s in ['SOXL', 'SOXX']}
    closes = pd.DataFrame({s: a.groupby(a.index.normalize()).last() for s, a in assets.items()})
    if not np.isfinite(closes.to_numpy()).all() or not closes.gt(0).all().all():
        raise ValueError('invalid native closing marks')
    log_returns = np.log(closes).diff()
    daily = pd.DataFrame(index=closes.index)
    daily['soxx_logmom63'] = np.log(closes.SOXX/closes.SOXX.shift(63))
    daily['soxx_logvol63'] = log_returns.SOXX.rolling(63).std(ddof=0)
    daily['soxx_drawdown126'] = closes.SOXX/closes.SOXX.rolling(126).max()-1
    daily['relative_logvol21'] = log_returns.SOXL.rolling(21).std(ddof=0)/(3*log_returns.SOXX.rolling(21).std(ddof=0))
    schedule = monthly_intervals(calendar)
    schedule = schedule.loc[plan['signal_start']:plan['signal_end']].copy()
    schedule = schedule.join(daily[MARKET])
    schedule['market_complete'] = np.isfinite(schedule[MARKET]).all(axis=1)
    if schedule.exit_at.isna().any():
        raise ValueError('terminal monthly exit absent; do not shorten evaluation')
    for symbol, marks in assets.items():
        for column in ['entry_at', 'exit_at']:
            times = pd.DatetimeIndex(schedule[column]).tz_localize(None)
            if not times.isin(marks.index).all():
                raise ValueError('missing '+symbol+' '+column+' native marks')
    return schedule, assets, prices


def count_selection(frame, plan):
    """Decide the evaluation calendar from source coverage, never label values."""
    counts = []
    first_year = None
    for year in plan['first_execution_year_candidates']:
        selected = frame.loc[frame.execution_year.eq(year)]
        if selected.empty or selected.index[0].month != 12:
            raise ValueError('missing December signal for January execution')
        cutoff = selected.signal_at.iloc[0]
        eligible = frame.source_qualified & frame.market_complete & frame.exit_at.lt(cutoff)
        count = int(eligible.sum())
        counts.append({'execution_year': year, 'cutoff': str(cutoff), 'mature_source_qualified_train_rows': count})
        if first_year is None and count >= plan['minimum_train_rows']:
            first_year = year
    return {'selected_first_execution_year': first_year, 'counts': counts,
            'selection_used_return_values': False, 'minimum_train_rows': plan['minimum_train_rows']}


def attach_labels(frame, assets):
    """Called only after design/source/count receipts have been frozen."""
    result = frame.copy()
    log_returns = {}
    for symbol, marks in assets.items():
        entry = marks.reindex(pd.DatetimeIndex(frame.entry_at).tz_localize(None)).to_numpy(float)
        exit_ = marks.reindex(pd.DatetimeIndex(frame.exit_at).tz_localize(None)).to_numpy(float)
        if not np.isfinite(entry).all() or not np.isfinite(exit_).all() or (entry <= 0).any() or (exit_ <= 0).any():
            raise ValueError('invalid monthly label marks')
        log_returns[symbol] = np.log(exit_/entry)
    result[TARGETS[0]] = log_returns['SOXX']
    result[TARGETS[1]] = log_returns['SOXL']-log_returns['SOXX']
    return result
