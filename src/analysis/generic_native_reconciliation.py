"""Read actual native cash and stock quantities; reconcile next-open NAV and fees."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.analysis.convex_qlib import native_open_reconciliation


def audit_actual_positions(frame, path, native, opening, *, initial_cash, tolerance=1e-7):
    """Use native positions and native opens, never report weight defaults or close marks."""
    symbols = ('SOXL', 'SOXX')
    expected = pd.DatetimeIndex(path.trade_date)
    if not expected.equals(pd.DatetimeIndex(frame.trade_date)) or set(native['positions']) != set(expected):
        raise ValueError('blocked: native position dates differ from the complete schedule')
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError('blocked: invalid initial account cash')
    quantities, cash, rows = np.zeros(2), float(initial_cash), []
    for row in path.itertuples():
        prices = opening.loc[row.trade_date, list(symbols)].to_numpy(float)
        exit_prices = opening.loc[row.exit_date, list(symbols)].to_numpy(float)
        if not np.isfinite([*prices, *exit_prices]).all() or (prices <= 0).any() or (exit_prices <= 0).any():
            raise ValueError('blocked: incomplete native entry/exit marks')
        before = float(cash + quantities @ prices)
        position = native['positions'][row.trade_date]
        holdings = position.get_stock_amount_dict()
        after_quantities = np.array([holdings.get(s, 0.) for s in symbols], dtype=float)
        after_cash = float(position.get_cash())
        after = float(after_cash + after_quantities @ prices)
        ending_nav = float(after_cash + after_quantities @ exit_prices)
        actual_weights = np.r_[after_quantities * exit_prices, after_cash] / ending_nav
        expected_weights = np.array([row.ending_soxl_weight, row.ending_soxx_weight, row.ending_cash_weight])
        fee_fraction = (before - after) / before
        turnover = float(np.abs(after_quantities - quantities) @ prices / before)
        if not np.isfinite([*after_quantities, after_cash, ending_nav, *actual_weights]).all() or ending_nav <= 0:
            raise ValueError('blocked: invalid actual native account')
        errors = dict(ending_weight_error=float(np.abs(actual_weights - expected_weights).max()),
                      fee_fraction_error=abs(fee_fraction - row.fees), turnover_error=abs(turnover - row.turnover),
                      normalized_nav_error=abs(ending_nav / initial_cash - row.equity) / max(1., abs(row.equity)))
        if max(errors.values()) > tolerance:
            raise ValueError(f'blocked: actual native cash/quantity reconciliation differs at {row.trade_date}: {errors}')
        if not row.execute_open_target and (not np.array_equal(quantities, after_quantities) or cash != after_cash):
            raise ValueError('blocked: native HOLD changed quantities or cash')
        rows.append(dict(trade_date=row.trade_date, exit_date=row.exit_date,
                         native_cash=after_cash, native_soxl_quantity=after_quantities[0],
                         native_soxx_quantity=after_quantities[1], native_next_open_nav=ending_nav,
                         native_fee_fraction=fee_fraction, native_turnover=turnover,
                         native_ending_soxl_weight=actual_weights[0], native_ending_soxx_weight=actual_weights[1],
                         native_ending_cash_weight=actual_weights[2], **errors))
        quantities, cash = after_quantities, after_cash
    result = pd.DataFrame(rows)
    audit = dict(status='passed', sessions=len(result), actual_cash_quantity_checked=True,
                 all_three_ending_weights_checked=True, native_fee_and_turnover_checked=True,
                 tolerance=tolerance, maximum_ending_weight_error=float(result.ending_weight_error.max()),
                 maximum_fee_fraction_error=float(result.fee_fraction_error.max()),
                 maximum_turnover_error=float(result.turnover_error.max()),
                 maximum_normalized_nav_error=float(result.normalized_nav_error.max()),
                 mark='actual native stock quantities and cash marked at next native open',
                 real_broker_fills=False)
    return result, audit


def reconcile_native_horizon(frame, path, *, provider_uri, cost_bps, account):
    """Native Qlib execution followed by independent cash/quantity/fee/weight readback."""
    schedule = path.copy()
    columns = ['open_soxl_weight','open_soxx_weight','open_cash_weight']
    values = schedule[columns].to_numpy(float)
    negative = (values < 0).any(axis=1)
    if (negative & schedule.execute_open_target.to_numpy(bool)).any() or values.min() < -1e-12:
        raise ValueError('blocked: materially negative or executed negative native target')
    # Native HOLD consumes no target. Adapt only its unused validation columns;
    # preserve the research holdings, returns and risk trace without any trade.
    cleaned = np.maximum(values[negative],0.)
    if len(cleaned):
        cleaned /= cleaned.sum(axis=1,keepdims=True)
        schedule.loc[negative,columns] = cleaned
    native = native_open_reconciliation(frame, schedule, provider_uri=provider_uri, cost_bps=cost_bps, account=account)
    if native['audit']['status'] != 'passed':
        raise ValueError('blocked: native return reconciliation failed')
    from qlib.data import D
    opening = D.features(['SOXL', 'SOXX'], ['$open'], start_time=path.trade_date.iloc[0],
                        end_time=path.exit_date.iloc[-1], freq='day')['$open'].unstack('instrument').astype(float)
    result, audit = audit_actual_positions(frame, path, native, opening, initial_cash=account)
    native['actual_position_reconciliation'], native['actual_position_audit'] = result, audit
    native['hold_validation_adapter'] = dict(unused_hold_rows_adapted=int(negative.sum()),
        maximum_unused_weight_adjustment=float(np.abs(schedule[columns].to_numpy()-values).max()),
        executed_targets_changed=False, research_path_changed=False)
    return native
