"""Exact ex-post relative-return attribution from reconciled native accounts.

Components are accounting identities, not feasible counterfactual portfolios.
Log linking makes their sums equal observed relative log terminal wealth.
"""
import numpy as np
import pandas as pd


def attribute_account(frame, book, benchmark, *, initial_cash, cash_reference):
    path = book['authority'].reset_index(drop=True)
    bench = benchmark['authority'].reset_index(drop=True)
    prices = frame.set_index('trade_date', drop=False).loc[pd.DatetimeIndex(path.trade_date)]
    if not pd.to_datetime(path.trade_date).equals(pd.to_datetime(bench.trade_date)):
        raise ValueError('account and benchmark dates differ')
    orders = book['orders']
    q, cash = np.zeros(2), float(initial_cash)
    rows = []
    for i, (_, row) in enumerate(prices.iterrows()):
        p = row[['soxl_open','soxx_open']].to_numpy(float)
        end = row[['soxl_exit_open','soxx_exit_open']].to_numpy(float)
        recovery_price = row[['soxl_recovery','soxx_recovery']].to_numpy(float)
        nav = float(q @ p + cash)
        fees = {'open':0., 'recovery':0.}
        delta = np.zeros(2)
        session = orders.loc[pd.to_datetime(orders.trade_date).eq(row.trade_date)]
        open_q, open_cash = q.copy(), cash
        for order in session.itertuples():
            if order.slot not in fees: raise ValueError('unknown execution slot')
            j = {'SOXL':0,'SOXX':1}[order.symbol]
            amount = (-1 if order.sell else 1)*order.adjusted_amount
            fees[order.slot] += order.total
            if order.slot == 'open':
                open_q[j] += amount
                open_cash -= amount*order.adjusted_price + order.total
            else:
                delta[j] += amount
        native = path.iloc[i]
        q = native[['ending_soxl_quantity','ending_soxx_quantity']].to_numpy(float)
        cash = float(native.ending_cash)
        np.testing.assert_allclose(open_q+delta,q,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(open_cash-delta@recovery_price-fees['recovery'],cash,rtol=1e-10,atol=1e-7)
        returns = end/p-1
        cash_weight = open_cash/nav
        base_cash = min(cash_reference,cash_weight)
        components = {
            'reference_cash':-base_cash*returns[1],
            'additional_cash':-(cash_weight-base_cash)*returns[1],
            'soxl_relative':open_q[0]*p[0]/nav*(returns[0]-returns[1]),
            'recovery_gross':float(delta@(end-recovery_price))/nav,
            'execution_cost':-(fees['open']*(1+returns[1])+fees['recovery'])/nav,
            'benchmark_residual':returns[1]-bench.iloc[i].strategy_return,
            'native_reconciliation_residual':native.strategy_return-book['independent_replay'].iloc[i].strategy_return,
        }
        difference = native.strategy_return-bench.iloc[i].strategy_return
        error = sum(components.values())-difference
        if abs(error)>1e-9: raise ValueError(f'accounting mismatch {row.trade_date}: {error}')
        relative_log = np.log1p(native.strategy_return)-np.log1p(bench.iloc[i].strategy_return)
        factor = relative_log/difference if abs(difference)>1e-12 else 1/(1+bench.iloc[i].strategy_return)
        rows.append({'trade_date':row.trade_date,'signal_date':row.signal_date,'risk_state':native.risk_state,
            'execute_open_target':native.execute_open_target,'recovery_action':native.recovery_action,
            'cash_weight_after_open':cash_weight,'relative_log':relative_log,'reconciliation_error':error,
            **{k:v*factor for k,v in components.items()}})
    result=pd.DataFrame(rows)
    np.testing.assert_allclose(result[list(components)].sum(axis=1),result.relative_log,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(np.exp(result.relative_log.sum()),float(path.equity.iloc[-1]/bench.equity.iloc[-1]),rtol=1e-9)
    return result
