"""Paired native raw-book results; accounting migration is not a new alpha claim."""
import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics, stationary_bootstrap_excess


def analyze_raw(books, proxy_books, frame, plan):
    families = ['raw_candidate27baa', 'raw_strong9097', 'raw_SOXX']
    costs = plan['slippage_bps']
    if set(books) != {(f, c) for f in families for c in costs}:
        raise ValueError('nine fixed raw books required')
    dates = frame[['signal_date', 'trade_date', 'exit_date']].reset_index(drop=True)
    metrics, periods, rolling_rows, uncertainty, proxy_rows, differences, states = [], [], [], [], [], [], []
    rolling = {}
    for (family, cost), book in books.items():
        path = book['authority'].reset_index(drop=True)
        bench = books[('raw_SOXX', cost)]['authority'].reset_index(drop=True)
        pd.testing.assert_frame_equal(path[list(dates)], dates)
        np.testing.assert_allclose((1 + path.strategy_return).cumprod(), path.equity, rtol=1e-9, atol=1e-9)
        points = book['points']
        if not pd.to_datetime(points.stamp).is_monotonic_increasing:
            raise ValueError('point NAV is not chronological')
        nav = np.r_[float(plan['initial_cash']), points.nav.to_numpy(float)]
        if not np.isfinite(nav).all() or (nav <= 0).any():
            raise ValueError('invalid point NAV')
        point_dd = float(100 * (nav / np.maximum.accumulate(nav) - 1).min())
        meta = {'family': family, 'slippage_bps': cost}
        metrics.append({**meta, **paired_metrics(path, bench), 'sampled_point_max_drawdown_pct': point_dd,
                        'final_spendable_cash': float(path.ending_spendable_cash.iloc[-1]),
                        'final_nonspendable_receivable': float(path.ending_receivable.iloc[-1]),
                        'fullminute_risk_qualified': False})
        for freq, kind in [('Y', 'year'), ('Q', 'quarter')]:
            labels = path.trade_date.dt.to_period(freq)
            for label in labels.unique():
                mask = labels.eq(label)
                periods.append({**meta, 'kind': kind, 'period': str(label),
                                'complete_period': bool(label != labels.iloc[0] and label != labels.iloc[-1]),
                                **paired_metrics(path.loc[mask], bench.loc[mask])})
        excess = np.log1p(path.strategy_return) - np.log1p(bench.strategy_return)
        win = pd.DataFrame({'first_trade': dates.trade_date.iloc[:-125].to_numpy(),
                            'last_exit': dates.exit_date.iloc[125:].to_numpy(),
                            'relative_log': excess.rolling(126).sum().iloc[125:].to_numpy()})
        win['relative_return_pct'] = 100 * np.expm1(win.relative_log)
        rolling[f'{family}_{cost}'] = win
        rolling_rows.append({**meta, 'window_sessions': 126, 'windows': len(win),
                             'positive_fraction': float(win.relative_log.gt(0).mean()),
                             'minimum_relative_return_pct': float(win.relative_return_pct.min()),
                             'median_relative_return_pct': float(win.relative_return_pct.median())})
        strata = {'risk_state': path.risk_state, 'prior_trend20': pd.Series(
            np.where(frame.soxx_trend20.to_numpy() > 0, 'positive', 'nonpositive'))}
        if not np.isfinite(frame.soxx_trend20).all():
            raise ValueError('unknown trend cannot be classified as nonpositive')
        for name, labels in strata.items():
            for label in labels.unique():
                mask = labels.eq(label)
                states.append({**meta, 'stratum': name, 'value': str(label), 'sessions': int(mask.sum()),
                               'relative_log': float(excess.loc[mask].sum()),
                               'noncontiguous_description_not_causal': True})
        if family == 'raw_candidate27baa':
            for comparator in plan['bootstrap']['comparators']:
                other = books[(comparator, cost)]['authority']
                uncertainty.append({**meta, 'comparator': comparator,
                    **stationary_bootstrap_excess(path, other, **{k: v for k, v in plan['bootstrap'].items() if k != 'comparators'})})
        if (family, cost) in proxy_books:
            old = proxy_books[(family, cost)]['authority'].reset_index(drop=True)
            pd.testing.assert_frame_equal(path[list(dates)], old[list(dates)])
            comparison = paired_metrics(path, old)
            proxy_rows.append({**meta, 'raw_terminal_wealth': comparison['terminal_wealth'],
                               'proxy_terminal_wealth': float((1 + old.strategy_return).prod()),
                               'raw_over_proxy_wealth': comparison['relative_wealth'],
                               'raw_minus_proxy_return_percentage_points': comparison['return_pct'] - comparison['benchmark_return_pct'],
                               'interpretation': 'accounting and endogenous policy-path migration; not incremental alpha'})
            for column in ['risk_state', 'risk_clock', 'current_beta_cap', 'recovery_action']:
                if column not in old or column not in path:
                    raise ValueError('missing frozen decision comparison field ' + column)
                changed = ~np.isclose(path[column].to_numpy(float), old[column].to_numpy(float), rtol=0, atol=1e-10)
                differences.append({**meta, 'field': column, 'changed_sessions': int(changed.sum()),
                                    'first_changed_trade': str(path.trade_date[changed].iloc[0].date()) if changed.any() else None})
    table = pd.DataFrame(metrics)
    period = pd.DataFrame(periods)
    rolls = pd.DataFrame(rolling_rows)
    unc = pd.DataFrame(uncertainty)
    candidate = table.loc[table.family.eq('raw_candidate27baa')]
    annual = period.loc[period.family.eq('raw_candidate27baa') & period.kind.eq('year') & period.complete_period]
    annual_rates = annual.groupby('slippage_bps').relative_wealth.apply(lambda v: (v > 1).mean())
    paired = pd.DataFrame([{'family': 'raw_candidate27baa', 'comparator': comparator, 'slippage_bps': c,
                           **paired_metrics(books[('raw_candidate27baa', c)]['authority'], books[(comparator, c)]['authority'])}
                          for c in costs for comparator in ['raw_SOXX', 'raw_strong9097']])
    strong_pass = paired.loc[paired.comparator.eq('raw_strong9097'), 'relative_wealth'].gt(1 + 1e-8).all()
    gates = {'net_soxx_excess_all_costs': bool(candidate.relative_wealth.gt(1 + 1e-8).all()),
             'beat_raw_strong_all_costs': bool(strong_pass),
             'sampled_drawdown_30pct': bool(candidate.sampled_point_max_drawdown_pct.ge(-30).all() and candidate.max_drawdown_pct.ge(-30).all()),
             'cvar_no_worse': bool(candidate.cvar5_pct.ge(candidate.benchmark_cvar5_pct).all()),
             'rolling_70pct': bool(rolls.loc[rolls.family.eq('raw_candidate27baa'), 'positive_fraction'].ge(.7).all()),
             'complete_annual_70pct': bool(set(annual_rates.index) == set(costs) and annual_rates.ge(.7).all()),
             'bootstrap_positive': bool(len(unc) == len(costs) * 2 and unc.lower_bound_positive.all()),
             'fullminute_risk_qualified': False, 'independent_development': False, 'qualified_forward': False}
    return {'metrics': table, 'annual_metrics': period.loc[period.kind.eq('year')].reset_index(drop=True),
            'quarterly_metrics': period.loc[period.kind.eq('quarter')].reset_index(drop=True),
            'rolling_summary': rolls, 'rolling': rolling, 'uncertainty': unc,
            'proxy_comparison': pd.DataFrame(proxy_rows), 'decision_divergence': pd.DataFrame(differences),
            'state_summary': pd.DataFrame(states), 'paired_comparison': paired, 'gates': gates}
