from datetime import datetime
from zoneinfo import ZoneInfo
import fcntl

from src.schedule import stage1_futu


def test_scheduler_serializes_account_reconciliation_and_submission(tmp_path, monkeypatch):
    runtime = tmp_path / stage1_futu.RUNTIME_RELATIVE
    runtime.mkdir(parents=True)
    called = []
    monkeypatch.setattr(stage1_futu, "_run_stage1_futu_task", lambda **kw: called.append(kw))
    with (runtime / "task.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = stage1_futu.run_stage1_futu_task(root=tmp_path, slot="open")
    assert result["reason"] == "stage1_task_already_running"
    assert called == []


def test_preopen_retries_failed_refresh_then_reuses_success(tmp_path, monkeypatch):
    now = datetime(2026, 9, 28, 8, 15, tzinfo=ZoneInfo("America/New_York"))
    results = iter([{"status": "needs_data"}, {"status": "ok"}])
    calls = []

    def run(**kw):
        calls.append(kw)
        return next(results)

    monkeypatch.setattr(stage1_futu, "_run_stage1_futu_task", run)
    assert stage1_futu.run_stage1_futu_task(root=tmp_path, now=now)["status"] == "needs_data"
    assert stage1_futu.run_stage1_futu_task(root=tmp_path, now=now)["status"] == "ok"
    result = stage1_futu.run_stage1_futu_task(root=tmp_path, now=now)
    assert result["reason"] == "settled_data_already_refreshed"
    assert [c["slot"] for c in calls] == ["postclose", "postclose"]
    # The next settled date must refresh, including across a DST change.
    assert stage1_futu.detect_slot(datetime(2026, 11, 2, 21, 15, tzinfo=ZoneInfo("Asia/Shanghai"))) == "postclose"


def test_skip_refresh_does_not_suppress_later_real_refresh(tmp_path, monkeypatch):
    now = datetime(2026, 9, 28, 8, 15, tzinfo=ZoneInfo("America/New_York"))
    calls = []
    monkeypatch.setattr(stage1_futu, "_run_stage1_futu_task", lambda **kw: calls.append(kw) or {"status": "ok"})
    stage1_futu.run_stage1_futu_task(root=tmp_path, slot="postclose", skip_refresh=True, now=now)
    stage1_futu.run_stage1_futu_task(root=tmp_path, now=now)
    assert len(calls) == 2


def test_settled_signal_date_skips_exchange_holidays():
    assert stage1_futu.settled_date(datetime(2026, 9, 8, 9, 31, tzinfo=ZoneInfo("America/New_York"))) == "2026-09-04"
    assert stage1_futu.settled_date(datetime(2026, 4, 6, 9, 31, tzinfo=ZoneInfo("America/New_York"))) == "2026-04-02"


def test_hold_preserves_exact_shares_and_the_recovery_open_mark(tmp_path):
    from src.api.futu_sim import record_hold_decision
    now = datetime(2026, 9, 28, 9, 31, tzinfo=ZoneInfo("America/New_York"))
    path = tmp_path / "execution.json"
    record = record_hold_decision(
        state_path=path, signal_date="2026-09-25", slot="open",
        prices={"SOXL": 151.45, "SOXX": 572.68}, quantities={"SOXL": 200, "SOXX": 98},
        decision_context={"risk_clock": 2}, now=now,
    )
    assert record["orders"] == []
    assert record["target_quantities"] == {"SOXL": 200, "SOXX": 98}
    assert stage1_futu._latest_open_settlement_prices(path, session_date="2026-09-28") == record["prices"]


def test_missing_model_does_not_repeat_successful_data_refresh(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(stage1_futu, "refresh_stage1_data", lambda **kw: calls.append(kw) or {"status": "ok"})
    monkeypatch.setattr(stage1_futu, "opend_reachable", lambda *_args: False)
    monkeypatch.setattr(stage1_futu, "run_stage1_futu_sim_card", lambda **kw: (_ for _ in ()).throw(FileNotFoundError("missing frozen model")))
    now = datetime(2026, 9, 28, 20, 20, tzinfo=ZoneInfo("America/New_York"))
    for _ in range(2):
        result = stage1_futu.run_stage1_futu_task(root=tmp_path, slot="postclose", now=now)
        assert result["status"] == "blocked"
        assert "missing frozen model" in result["reason"]
    assert len(calls) == 1


def test_execution_quote_freshness_is_independent_of_saved_open_settlement_mark():
    now = datetime(2026, 9, 28, 10, 31, tzinfo=ZoneInfo("America/New_York"))
    assert stage1_futu._execution_quote_error("2026-09-28T10:30:30-04:00", now=now) is None
    assert stage1_futu._execution_quote_error("2026-09-28T09:31:00-04:00", now=now) == "stale_or_future_execution_quote"
    assert stage1_futu._execution_quote_error(None, now=now) == "missing_execution_quote_timestamp"


def test_account_marks_must_cover_both_assets_through_settled_date(tmp_path, monkeypatch):
    import pandas as pd
    import pytest
    import qlib
    import qlib.data
    from types import SimpleNamespace
    index = pd.MultiIndex.from_tuples(
        [("SOXL", pd.Timestamp("2026-09-24")), ("SOXX", pd.Timestamp("2026-09-24")),
         ("SOXL", pd.Timestamp("2026-09-25")), ("SOXX", pd.Timestamp("2026-09-25"))],
        names=["instrument", "datetime"],
    )
    monkeypatch.setattr(qlib, "init", lambda **kw: None)
    monkeypatch.setattr(qlib.data, "D", SimpleNamespace(features=lambda *_a, **_kw: pd.DataFrame({"$close": [146.33, 566.07, 151.45, float("nan")]}, index=index)))
    with pytest.raises(RuntimeError, match="latest settled session"):
        stage1_futu._load_qlib_settled_marks(tmp_path, start="2026-09-24", end="2026-09-25")
