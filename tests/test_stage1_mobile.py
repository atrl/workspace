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
        "fills": [{"side": "BUY", "symbol": "SOXX", "quantity": 2,
                   "price": 101, "updated_time": "2026-10-21 09:31:00"}],
    }))
    summary = build_stage1_mobile_summary(tmp_path)
    assert summary["status"] == "ready"
    assert summary["performance"]["returns_pct"] == pytest.approx({"Stage1": 10.0, "SOXX": 5.0, "SOXL": 10.0})
    assert summary["performance"]["difference_vs_soxx_pp"] == pytest.approx(5.0)
    assert summary["recent_fills"][0]["side"] == "BUY"
    message = build_stage1_bark_message({"status": "heartbeat", "slot": "daily_heartbeat"},
                                        mobile_summary=summary)
    assert "SOXX +5.00% / SOXL +10.00%" in message["body"]
    assert "最近成交 2026-10-21 09:31:00 纽约 BUY 2 SOXX" in message["body"]
