"""Cross-account descriptive attribution on immutable native account evidence."""
import numpy as np
import pandas as pd
from src.analysis.decision_accounting import attribute_account
from src.analysis.integrated_allocation import paired_metrics

COMPONENTS = ['reference_cash', 'additional_cash', 'soxl_relative', 'recovery_gross',
              'execution_cost', 'benchmark_residual', 'native_reconciliation_residual']


def analyze(frame, books, benchmarks, *, initial_cash, cash_reference, rolling_sessions, pairs):
    daily, totals, periods, states, rolling, rolling_summary = {}, [], [], [], {}, []
    dates = frame[['signal_date', 'trade_date', 'exit_date']].reset_index(drop=True)
    if dates.trade_date.duplicated().any() or not dates.trade_date.is_monotonic_increasing:
        raise ValueError('duplicate or unordered source calendar')
    if not dates.exit_date.iloc[:-1].reset_index(drop=True).equals(dates.trade_date.iloc[1:].reset_index(drop=True)):
        raise ValueError('source holding intervals are not contiguous')
    if not dates.signal_date.lt(dates.trade_date).all():
        raise ValueError('source signal date is not before trade')
    if not frame.feature_available_at.le(frame.decision_at).all():
        raise ValueError('source features unavailable at source decision')
    for (family, cost), book in books.items():
        path, bench = book['authority'].reset_index(drop=True), benchmarks[cost]['authority'].reset_index(drop=True)
        for account in (path, bench):
            pd.testing.assert_frame_equal(account[list(dates)], dates)
            np.testing.assert_allclose((1 + account.strategy_return).cumprod(), account.equity, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(account.native_expected_nav.iloc[0], account.equity.iloc[0] * initial_cash, rtol=1e-9, atol=1e-6)
        for order in book['orders'].itertuples():
            mark = frame.loc[frame.trade_date.eq(order.trade_date)].iloc[0]
            expected_price = mark[f'{order.symbol.lower()}_{order.slot}']
            np.testing.assert_allclose(order.adjusted_price, expected_price, rtol=1e-12, atol=1e-12)
        result = attribute_account(frame, book, benchmarks[cost], initial_cash=initial_cash, cash_reference=cash_reference)
        result['exit_date'] = dates.exit_date
        result['prior_trend20'] = np.where(frame.soxx_trend20.to_numpy() > 0, 'positive', 'nonpositive')
        if not np.isfinite(frame.soxx_trend20).all():
            raise ValueError('source trend is unknown; do not silently classify missing')
        result['net_return'] = path.strategy_return
        result['benchmark_return'] = bench.strategy_return
        daily[(family, cost)] = result
        meta = {'family': family, 'slippage_bps': cost}
        def attribution(part):
            return {'sessions': len(part), 'mean_cash_after_open_pct': 100*part.cash_weight_after_open.mean(),
                    **part[COMPONENTS + ['relative_log']].sum().to_dict()}
        totals.append({**meta, **attribution(result), **paired_metrics(path, bench),
                       'monitored_drawdown_pct': -100*path.monitored_drawdown.max(),
                       'native_vs_independent_nav_max_usd': float((path.native_expected_nav-path.equity*initial_cash).abs().max())})
        for frequency, kind in [('Y', 'year'), ('Q', 'quarter')]:
            for label, part in result.groupby(result.trade_date.dt.to_period(frequency)):
                index = part.index
                # The first account begins Jan4; the final period ends in Oct2026.
                boundary = (label == dates.trade_date.iloc[0].to_period(frequency) or
                            label == dates.trade_date.iloc[-1].to_period(frequency))
                periods.append({**meta, 'kind': kind, 'period': str(label), 'complete_period': not boundary,
                    'first_trade': str(part.trade_date.min().date()), 'last_exit': str(part.exit_date.max().date()),
                    **attribution(part), **paired_metrics(path.loc[index], bench.loc[index]),
                    'monitored_drawdown_global_peak_pct': -100*path.loc[index].monitored_drawdown.max()})
        for column in ['risk_state', 'execute_open_target', 'recovery_action', 'prior_trend20']:
            for label, part in result.groupby(column):
                states.append({**meta, 'stratum': column, 'value': str(label),
                    'noncontiguous_accounting_only': True, **attribution(part)})
        windows = result[COMPONENTS + ['relative_log']].rolling(rolling_sessions).sum().iloc[rolling_sessions-1:].copy()
        windows['first_trade'] = dates.trade_date.iloc[:len(windows)].to_numpy()
        windows['last_exit'] = dates.exit_date.iloc[rolling_sessions-1:].to_numpy()
        windows['relative_return_pct'] = 100*np.expm1(windows.relative_log)
        rolling[(family, cost)] = windows.reset_index(drop=True)
        rolling_summary.append({**meta, 'window_sessions': rolling_sessions, 'windows': len(windows),
            'positive_fraction': windows.relative_log.gt(0).mean(),
            **{f'relative_return_{k}_pct': v for k, v in windows.relative_return_pct.agg(['min','median','max']).items()}})
    pair_rows=[]
    for candidate, control in pairs:
        for cost in benchmarks:
            left, right = daily[(candidate,cost)], daily[(control,cost)]
            for period, mask in [('all', np.ones(len(left),dtype=bool)), *[(str(y),left.trade_date.dt.year.eq(y).to_numpy()) for y in sorted(left.trade_date.dt.year.unique())]]:
                delta=left.loc[mask,COMPONENTS+['relative_log']].sum()-right.loc[mask,COMPONENTS+['relative_log']].sum()
                pair_rows.append({'candidate':candidate,'control':control,'slippage_bps':cost,'trade_year':period,
                    'sessions':int(mask.sum()), **delta.to_dict(), 'relative_wealth_ratio':float(np.exp(delta.relative_log)),
                    'interpretation':'paired realized package difference; endogenous paths, not isolated causal component'})
    return {'daily':daily,'rolling':rolling,'accounting_attribution':pd.DataFrame(totals),
            'annual_attribution':pd.DataFrame(periods).query("kind == 'year'"),
            'quarterly_attribution':pd.DataFrame(periods).query("kind == 'quarter'"),
            'state_attribution':pd.DataFrame(states),'rolling_summary':pd.DataFrame(rolling_summary),
            'paired_attribution':pd.DataFrame(pair_rows)}
