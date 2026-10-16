"""Material contracts for executable-weight training and dividend-aware drift."""
import torch
import pandas as pd
import pytest

from src.model.monthly_profit_utility import beta_raw_weights, compose_weights, monthly_turnover


def test_training_composition_preserves_beta_and_removes_cancellable_leverage():
    beta = torch.tensor([0.0, 0.4, 0.85, 1.0, 1.45])
    weights = compose_weights(beta_raw_weights(beta))
    torch.testing.assert_close(weights.sum(1), torch.ones_like(beta))
    torch.testing.assert_close(3 * weights[:, 0] + weights[:, 1], beta)
    assert (weights >= -1e-7).all()
    assert ((weights[:, 0] <= 1e-7) | (weights[:, 2] <= 1e-7)).all()
    torch.testing.assert_close(weights[3], torch.tensor([0.0, 1.0, 0.0]))


def test_dividends_drift_to_cash_and_discontinuous_months_reset():
    weights = torch.tensor([[0.2, 0.6, 0.2], [0.1, 0.7, 0.2], [0.3, 0.7, 0.0]])
    capital = torch.tensor([[1.1, 0.9], [1.0, 1.0], [1.0, 1.0]])
    income = torch.tensor([[0.01, 0.02], [0.0, 0.0], [0.0, 0.0]])
    turnover, marked = monthly_turnover(weights, capital, income, torch.tensor([0, 0, 1]))
    expected = torch.tensor([0.22, 0.54, 0.214]) / 0.974
    torch.testing.assert_close(marked[1], expected)
    torch.testing.assert_close(marked[[0, 2]], torch.tensor([[0.0, 0.0, 1.0]]).expand(2, -1))
    torch.testing.assert_close(turnover[1], (weights[1, :2] - expected[:2]).abs().sum())
    torch.testing.assert_close(turnover[[0, 2]], torch.tensor([0.8, 1.0]))


def test_physical_month_label_respects_ex_dates_and_split_share_units():
    from src.workflow.quarterly_profit_allocation import physical_month_labels

    entry = pd.Timestamp('2024-01-02 09:47', tz='America/New_York')
    exit_at = pd.Timestamp('2024-02-01 09:47', tz='America/New_York')
    monthly = pd.DataFrame({'entry_at': [entry], 'exit_at': [exit_at]}, index=[pd.Timestamp('2023-12-29')])
    endpoints = pd.DataFrame({'SOXL': [10.0, 6.0], 'SOXX': [20.0, 22.0]},
                             index=[entry.tz_localize(None), exit_at.tz_localize(None)])
    cash = pd.DataFrame({'instrument': ['SOXL'] * 3,
                         'ex_date': pd.to_datetime(['2024-01-02', '2024-01-17', '2024-02-01']),
                         'original_total_decimal': [1.0, 0.2, 0.1]})
    splits = pd.DataFrame({'instrument': ['SOXL'], 'effective_trading_date': [pd.Timestamp('2024-01-16')],
                           'new_shares_per_old_share': [2.0]})
    row = physical_month_labels(monthly, endpoints, {'data/cash_events.pkl': cash, 'data/split_events.pkl': splits}).iloc[0]
    assert row.capital_growth_soxl == pytest.approx(1.2)
    assert row.income_yield_soxl == pytest.approx(0.06)
    assert row.capital_growth_soxx == pytest.approx(1.1)
    assert row.income_yield_soxx == 0.0
