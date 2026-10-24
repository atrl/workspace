import json
import pytest

from src.dashboard.stage1_mobile import build_stage1_mobile_summary
from src.schedule.stage1_futu import build_stage1_bark_message


def test_mobile_summary_uses_paired_qlib_marks_and_confirmed_fills(tmp_path):
    runtime = tmp_path / "runtime/soxl/stage1_futu_sim"
    runtime.mkdir(parents=True)
    (runtime / "paper_cycle.json").write_text(json.dumps({"cycle_id": "test", "ledger_file": "ledger.json"}))
    (runtime / "ledger.json").write_text(json.dumps({
        "cycle_id": "test", "execution_environment": "SIMULATE", "updated_at": "2026-10-22T10:00:00+08:00",
        "equity": 110, "drawdown": -0.1, "positions": {"SOXL": 1, "SOXX": 2},
        "settled_marks": [
            {"date": "2026-10-20", "source": "qlib.data.D.features", "equity": 100,
             "prices": {"SOXL": 50, "SOXX": 100}},
            {"date": "2026-10-21", "source": "qlib.data.D.features", "equity": 110,
             "prices": {"SOXL": 55, "SOXX": 105}},
        ],
        "fills": [{"side": "SELL", "symbol": "SOXL", "quantity": 1,
                   "price": 54, "updated_time": "2026-10-20 09:31:00"},
                  {"side": "BUY", "symbol": "SOXX", "quantity": 2,
                   "price": 101, "updated_time": "2026-10-21 09:31:00"}],
    }))
    summary = build_stage1_mobile_summary(tmp_path)
    assert summary["status"] == "ready"
    assert summary["performance"]["returns_pct"] == pytest.approx({"Stage1": 10.0, "SOXX": 5.0, "SOXL": 10.0})
    assert summary["performance"]["difference_vs_soxx_pp"] == pytest.approx(5.0)
    assert summary["performance"]["series_basis"] == "first_paired_settled_mark_equals_one"
    assert summary["performance"]["series"] == [
        {"date": "2026-10-20", "stage1_nav": 1.0, "soxx_nav": 1.0, "soxl_nav": 1.0},
        {"date": "2026-10-21", "stage1_nav": pytest.approx(1.1),
         "soxx_nav": pytest.approx(1.05), "soxl_nav": pytest.approx(1.1)},
    ]
    assert summary["recent_fills"][-1]["side"] == "BUY"
    assert {row["side"] for row in summary["latest_by_side"]} == {"BUY", "SELL"}
    message = build_stage1_bark_message({"status": "heartbeat", "slot": "daily_heartbeat"},
                                        mobile_summary=summary)
    assert "SOXX +5.00% / SOXL +10.00%" in message["body"]
    assert "最近成交 2026-10-21 09:31:00 纽约 BUY 2 SOXX" in message["body"]
    assert "最近成交 2026-10-20 09:31:00 纽约 SELL 1 SOXL" in message["body"]


def _write_account(tmp_path, **overrides):
    runtime = tmp_path / 'runtime/soxl/stage1_futu_sim'
    runtime.mkdir(parents=True)
    (runtime / 'paper_cycle.json').write_text(json.dumps({
        'cycle_id': 'test', 'ledger_file': 'ledger.json'}))
    ledger = {
        'cycle_id': 'test', 'execution_environment': 'SIMULATE',
        'inception_date': '2026-08-03', 'updated_at': '2026-10-24T10:09:27+08:00',
        'initial_equity': 100000, 'equity': 106856.87,
        'reconciliation': {'status': 'matched'},
        'settled_marks': [
            {'date': '2026-08-03', 'source': 'qlib.data.D.features',
             'equity': 104949.30, 'prices': {'SOXX': 100, 'SOXL': 100}},
            {'date': '2026-10-23', 'source': 'qlib.data.D.features',
             'equity': 106885.78, 'prices': {'SOXX': 94, 'SOXL': 72},
             'fee_basis': 'unknown_cash_adjustment_timing_not_allocated'},
        ],
    }
    ledger.update(overrides)
    (runtime / 'ledger.json').write_text(json.dumps(ledger))
    return runtime / 'ledger.json'


def test_inception_capital_growth_does_not_drop_first_day_or_rewrite_close(tmp_path):
    path = _write_account(tmp_path)
    original = path.read_bytes()
    summary = build_stage1_mobile_summary(tmp_path)
    account = summary['account_performance']
    assert account['capital_growth_pct'] == pytest.approx(6.85687)
    assert account['capital_change'] == pytest.approx(6856.87)
    assert account['nav'] == pytest.approx(1.0685687)
    assert account['twr_pct'] is None
    assert account['inception_excess_vs_soxx_pp'] is None
    comparison = summary['performance']
    assert comparison['returns_pct']['Stage1'] == pytest.approx(1.8451576)
    assert comparison['start_equity'] == 104949.30
    assert comparison['end_equity'] == 106885.78
    assert comparison['relative_wealth_vs_soxx_pct'] == pytest.approx(
        100 * ((106885.78 / 104949.30) / .94 - 1))
    assert summary['valuation_reconciliation']['settled_minus_account_equity'] == pytest.approx(28.91)
    assert path.read_bytes() == original
    message = build_stage1_bark_message({'status': 'heartbeat', 'slot': 'daily_heartbeat'},
                                       mobile_summary=summary)['body']
    assert '成立以来本金净值变动 +6.86%' in message
    assert '不含首日' in message
    assert 'Stage1 +1.85%' in message
    assert '差额 $+28.91' in message
    assert '非费后alpha' in message


@pytest.mark.parametrize('overrides', [
    {'initial_equity': None}, {'initial_equity': 0},
    {'initial_equity': float('nan')}, {'equity': float('inf')},
    {'reconciliation': {'status': 'mismatch'}},
])
def test_uncertified_account_does_not_fabricate_inception_return(tmp_path, overrides):
    _write_account(tmp_path, **overrides)
    account = build_stage1_mobile_summary(tmp_path)['account_performance']
    assert account['status'] == 'needs_data'
    assert account['capital_growth_pct'] is None


def test_account_growth_available_without_paired_benchmark_marks(tmp_path):
    _write_account(tmp_path, settled_marks=[])
    summary = build_stage1_mobile_summary(tmp_path)
    assert summary['account_performance']['capital_growth_pct'] == pytest.approx(6.85687)
    assert summary['performance'] is None
    assert summary['valuation_reconciliation'] is None
