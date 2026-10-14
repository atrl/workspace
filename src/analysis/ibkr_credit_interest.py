"""Credit-interest audit around verified Qlib books, not a trading simulator.

The caller supplies value dates and date-specific credit contracts. No historical
rates, settlement calendars, or payment calendars are inferred. Fixed native
orders do not respond to interest; outputs cannot certify a new policy return.
"""
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import numpy as np
import pandas as pd


def _days(values, name):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    if result.tz is not None or result.hasnans or not result.equals(result.normalize()):
        raise ValueError(f'{name} requires explicit naive local calendar dates')
    return result


def native_cash_flows(book, *, initial_cash):
    """Decompose native-order cash and reconcile every original trading day."""
    if book['audit']['status'] != 'passed':
        raise ValueError('native account reconciliation is not passed')
    orders, path = book['orders'], book['authority']
    if not orders.index.is_unique:
        raise ValueError('duplicate native order identity')
    records = []
    for identity, row in orders.iterrows():
        if not isinstance(row.sell, (bool, np.bool_)):
            raise ValueError('native order direction must be boolean')
        amount, fee = float(row.notional), float(row.total)
        if not np.isfinite([amount, fee]).all() or amount < 0 or fee < 0:
            raise ValueError('invalid native cash amount')
        stamp = pd.Timestamp(row.stamp)
        if stamp.tzinfo is not None or pd.isna(stamp):
            raise ValueError('native stamp must use the declared local clock')
        for leg, delta in [('principal', amount if row.sell else -amount), ('fee', -fee)]:
            records.append({'order_id': identity, 'leg': leg, 'trade_date': stamp.normalize(), 'amount': delta})
    flows = pd.DataFrame(records, columns=['order_id', 'leg', 'trade_date', 'amount'])
    dates = _days(path.trade_date, 'account dates')
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('account dates must be ordered and unique')
    if len(flows) and not flows.trade_date.isin(dates).all():
        raise ValueError('order outside native account calendar')
    changes = flows.groupby('trade_date').amount.sum().reindex(dates, fill_value=0.).astype(float)
    expected = float(initial_cash) + changes.cumsum().to_numpy()
    np.testing.assert_allclose(expected, path.ending_cash.to_numpy(float), rtol=0, atol=1e-7)
    return flows


def audit_credit_interest(book, *, initial_cash, daily_nav_without_interest,
                          value_dates, contracts, payment_dates):
    """Fixed-order diagnostic; opening account is entirely settled cash.

    contracts: one row per calendar day, fields annual_credit_rate, free_cash,
    full_rate_nav, day_basis, rounding ('half_up' or 'half_even'). Rates are the
    eligible credit rates AFTER any spread, before NAV scaling. payment_dates
    maps accrual YYYY-MM to a separately verified payment date. Both principal
    and fee value dates are mandatory per order, even for zero amounts.
    """
    flows = native_cash_flows(book, initial_cash=initial_cash)
    days = _days(contracts.index, 'contract dates')
    if not len(days) or not days.is_unique or not days.equals(pd.date_range(days[0], days[-1], freq='D')):
        raise ValueError('complete consecutive calendar-day contracts required')
    if not daily_nav_without_interest.index.equals(contracts.index):
        raise ValueError('daily NAV dates differ from credit contracts')
    if not np.isfinite(daily_nav_without_interest.to_numpy(float)).all() or daily_nav_without_interest.lt(0).any():
        raise ValueError('invalid daily NAV')
    required = ['annual_credit_rate', 'free_cash', 'full_rate_nav', 'day_basis']
    if not np.isfinite(contracts[required].to_numpy(float)).all():
        raise ValueError('missing or nonfinite credit contract')
    if (contracts[['annual_credit_rate', 'free_cash']] < 0).any().any() or (contracts[['full_rate_nav', 'day_basis']] <= 0).any().any():
        raise ValueError('unsupported credit contract')
    modes = {'half_up': ROUND_HALF_UP, 'half_even': ROUND_HALF_EVEN}
    if not contracts['rounding'].isin(modes).all():
        raise ValueError('explicit supported rounding contract required')
    order_ids = book['orders'].index
    if not value_dates.index.is_unique or set(value_dates.index) != set(order_ids):
        raise ValueError('value dates must cover exactly every native order')
    mapping = {}
    for leg in ['principal', 'fee']:
        stamps = _days(value_dates[leg], f'{leg} value dates')
        mapping[leg] = dict(zip(value_dates.index, stamps))
    if len(flows):
        flows = flows.copy()
        flows['value_date'] = [mapping[r.leg][r.order_id] for r in flows.itertuples()]
        if flows.trade_date.lt(days[0]).any() or flows.trade_date.gt(days[-1]).any():
            raise ValueError('native orders outside audit span')
        if (flows.value_date < flows.trade_date).any():
            raise ValueError('settlement before trade')
    payments = {str(month): _days([date], 'payment date')[0] for month, date in payment_dates.items()}
    periods = set(days.strftime('%Y-%m'))
    if not periods.issubset(payments):
        raise ValueError('missing explicit monthly payment date')
    for period, date in payments.items():
        if date <= pd.Period(period, freq='M').end_time.normalize():
            raise ValueError('payment must follow its accrual month')
    if not np.isfinite(initial_cash) or initial_cash < 0:
        raise ValueError('invalid opening cash')
    cash = settled = Decimal(str(initial_cash))
    accrued, posted, earned = {}, Decimal('0'), Decimal('0')
    rows = []
    for day in days:
        paid_today = Decimal('0')
        for month in list(accrued):
            if payments[month] == day:
                paid_today += accrued.pop(month)
        posted += paid_today
        if len(flows):
            cash += sum((Decimal(str(v)) for v in flows.loc[flows.trade_date.eq(day), 'amount']), Decimal('0'))
            settled += sum((Decimal(str(v)) for v in flows.loc[flows.value_date.eq(day), 'amount']), Decimal('0'))
        if min(cash + posted, settled + posted) < Decimal('-0.0000001'):
            raise ValueError('needs debit-interest model for negative cash')
        rule = contracts.loc[day]
        nav = Decimal(str(daily_nav_without_interest.loc[day])) + posted + sum(accrued.values(), Decimal('0'))
        scale = min(Decimal('1'), nav / Decimal(str(rule.full_rate_nav)))
        eligible = max(Decimal('0'), settled + posted - Decimal(str(rule.free_cash)))
        interest = (eligible * Decimal(str(rule.annual_credit_rate)) * scale / Decimal(str(rule.day_basis))).quantize(Decimal('.01'), rounding=modes[rule['rounding']])
        month = day.strftime('%Y-%m')
        accrued[month] = accrued.get(month, Decimal('0')) + interest
        earned += interest
        pending_interest = sum(accrued.values(), Decimal('0'))
        assert earned == posted + pending_interest
        rows.append({'date': day, 'trade_cash_without_interest': float(cash),
            'settled_cash_without_interest': float(settled), 'unsettled_cash_flow': float(cash-settled),
            'settled_cash_with_posted_interest': float(settled+posted),
            'interest_paid_today': float(paid_today), 'interest_accrued_today': float(interest),
            'interest_posted_total': float(posted), 'interest_pending': float(pending_interest),
            'interest_earned_total': float(earned), 'nav_scaling': float(scale)})
    return pd.DataFrame(rows).set_index('date')
