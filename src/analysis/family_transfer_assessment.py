"""Native paired evaluation of a fixed pooled-versus-single-market experiment."""
import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics, stationary_bootstrap_excess


def analyze_transfer(books, frame, *, initial_cash, costs):
    families = ['raw_candidate', 'raw_control', 'raw_candidate27baa', 'raw_SOXX']
    comparators = ['raw_control', 'raw_candidate27baa', 'raw_SOXX']
    if set(books) != {(f, c) for f in families for c in costs}:
        raise ValueError('missing or additional fixed evaluation book')
    dates = frame[['signal_date', 'trade_date', 'exit_date']].reset_index(drop=True)
    if len(dates) != 944 or not np.isfinite(frame.soxx_trend20).all():
        raise ValueError('fixed calendar or trend coverage differs')
    metrics, pair_rows, periods, roll_rows, states, intervals = [], [], [], [], [], []
    rolling = {}
    for (family, cost), book in books.items():
        path = book['authority'].reset_index(drop=True)
        pd.testing.assert_frame_equal(path[list(dates)], dates)
        np.testing.assert_allclose((1 + path.strategy_return).cumprod(), path.equity, rtol=1e-9, atol=1e-9)
        points = book['points']
        if not pd.to_datetime(points.stamp).is_monotonic_increasing:
            raise ValueError('unordered native marks')
        nav = np.r_[float(initial_cash), points.nav.to_numpy(float)]
        if not np.isfinite(nav).all() or (nav <= 0).any():
            raise ValueError('invalid native point wealth')
        point_dd = float(100 * (nav / np.maximum.accumulate(nav) - 1).min())
        base = books[('raw_SOXX', cost)]['authority'].reset_index(drop=True)
        meta = {'family': family, 'slippage_bps': cost}
        metrics.append({**meta, **paired_metrics(path, base),
                        'sampled_point_max_drawdown_pct': point_dd,
                        'final_spendable_cash': float(path.ending_spendable_cash.iloc[-1]),
                        'final_nonspendable_receivable': float(path.ending_receivable.iloc[-1]),
                        'fullminute_risk_qualified': False})
        references = comparators if family == 'raw_candidate' else ['raw_SOXX']
        for reference in references:
            bench = books[(reference, cost)]['authority'].reset_index(drop=True)
            pair_meta = {**meta, 'comparator': reference}
            pair_rows.append({**pair_meta, **paired_metrics(path, bench)})
            excess = np.log1p(path.strategy_return) - np.log1p(bench.strategy_return)
            for freq, kind in [('Y', 'year'), ('Q', 'quarter')]:
                labels = path.trade_date.dt.to_period(freq)
                for label in labels.unique():
                    mask = labels.eq(label)
                    periods.append({**pair_meta, 'kind': kind, 'period': str(label),
                                    'complete_period': bool(label != labels.iloc[0] and label != labels.iloc[-1]),
                                    **paired_metrics(path.loc[mask], bench.loc[mask])})
            windows = pd.DataFrame({'first_trade': dates.trade_date.iloc[:-125].to_numpy(),
                                    'last_exit': dates.exit_date.iloc[125:].to_numpy(),
                                    'relative_log': excess.rolling(126).sum().iloc[125:].to_numpy()})
            if len(windows) != 819 or windows.relative_log.isna().any():
                raise ValueError('incomplete contiguous rolling windows')
            windows['relative_return_pct'] = 100 * np.expm1(windows.relative_log)
            rolling[f'{family}_vs_{reference}_{cost}'] = windows
            roll_rows.append({**pair_meta, 'window_sessions': 126, 'windows': len(windows),
                              'positive_fraction': float(windows.relative_log.gt(0).mean()),
                              'minimum_relative_return_pct': float(windows.relative_return_pct.min()),
                              'median_relative_return_pct': float(windows.relative_return_pct.median())})
            if family == 'raw_candidate':
                intervals.append({**pair_meta, **stationary_bootstrap_excess(
                    path, bench, mean_block=63, resamples=2000, seed=20261014, alpha=.025)})
            if reference == 'raw_SOXX':
                strata = {'risk_state': path.risk_state, 'prior_trend20': pd.Series(
                    np.where(frame.soxx_trend20.to_numpy() > 0, 'positive', 'nonpositive'))}
                for stratum, labels in strata.items():
                    for label in labels.unique():
                        mask = labels.eq(label)
                        states.append({**meta, 'stratum': stratum, 'value': str(label),
                                       'sessions': int(mask.sum()), 'relative_log': float(excess.loc[mask].sum()),
                                       'noncontiguous_description_not_causal': True})
    table, paired = pd.DataFrame(metrics), pd.DataFrame(pair_rows)
    period, rolls, uncertainty = pd.DataFrame(periods), pd.DataFrame(roll_rows), pd.DataFrame(intervals)
    candidate = table.loc[table.family.eq('raw_candidate')]
    annual = period.loc[period.family.eq('raw_candidate') & period.comparator.eq('raw_SOXX') &
                        period.kind.eq('year') & period.complete_period]
    if set(annual.period) != {'2024', '2025'} or len(annual) != 2 * len(costs):
        raise ValueError('missing fixed complete annual coverage')
    annual_rates = annual.groupby('slippage_bps').relative_wealth.apply(lambda values: values.gt(1).mean())
    pair_candidate = paired.loc[paired.family.eq('raw_candidate')]
    roll_candidate = rolls.loc[rolls.family.eq('raw_candidate') & rolls.comparator.eq('raw_SOXX')]
    gates = {
        'beat_matched_control_all_costs': bool(pair_candidate.loc[pair_candidate.comparator.eq('raw_control'), 'relative_wealth'].gt(1 + 1e-8).all()),
        'beat_frozen_raw27baa_all_costs': bool(pair_candidate.loc[pair_candidate.comparator.eq('raw_candidate27baa'), 'relative_wealth'].gt(1 + 1e-8).all()),
        'net_soxx_excess_all_costs': bool(candidate.relative_wealth.gt(1 + 1e-8).all()),
        'sampled_drawdown_30pct': bool(candidate.sampled_point_max_drawdown_pct.ge(-30).all() and candidate.max_drawdown_pct.ge(-30).all()),
        'cvar_no_worse_soxx': bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),
        'rolling_70pct_soxx': bool(roll_candidate.positive_fraction.ge(.7).all()),
        'complete_annual_70pct_soxx': bool(annual_rates.ge(.7).all()),
        'bootstrap_positive_all_three': bool(len(uncertainty) == 3 * len(costs) and uncertainty.lower_bound_positive.all()),
        'fullminute_risk_qualified': False, 'independent_development': False, 'qualified_forward': False,
    }
    return {'metrics': table, 'paired_comparison': paired,
            'annual_metrics': period.loc[period.kind.eq('year')].reset_index(drop=True),
            'quarterly_metrics': period.loc[period.kind.eq('quarter')].reset_index(drop=True),
            'rolling_summary': rolls, 'rolling': rolling, 'state_summary': pd.DataFrame(states),
            'uncertainty': uncertainty, 'gates': gates}
