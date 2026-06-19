from __future__ import annotations

import json

import pytest

from src.schedule import (
    ScheduleJob,
    classify_market_calendar_day,
    load_market_calendar_decision,
    run_job_once,
    run_task_with_retries,
)


def test_classify_market_calendar_day_identifies_normal_half_day_and_closed_sessions():
    trading_days = ("2026-01-02", "2026-01-05", "2026-01-06")
    half_days = ("2026-01-02",)
    holiday_reasons = {"2026-01-01": "New Year's Day"}

    closed = classify_market_calendar_day(
        "2026-01-01",
        trading_calendar=trading_days,
        half_days=half_days,
        holiday_reasons=holiday_reasons,
    )
    half_day = classify_market_calendar_day("2026-01-02", trading_calendar=trading_days, half_days=half_days)
    normal = classify_market_calendar_day("2026-01-05", trading_calendar=trading_days, half_days=half_days)

    assert closed["schema_version"] == "schedule.market_calendar_decision.v1"
    assert closed["market_status"] == "closed"
    assert closed["should_run"] is False
    assert closed["skip_reason"] == "New Year's Day"
    assert closed["next_trading_date"] == "2026-01-02"
    assert closed["previous_trading_date"] == ""

    assert half_day["market_status"] == "half_day"
    assert half_day["should_run"] is True
    assert half_day["run_mode"] == "half_day"
    assert half_day["next_trading_date"] == "2026-01-02"

    assert normal["market_status"] == "normal"
    assert normal["should_run"] is True
    assert normal["run_mode"] == "regular"
    assert normal["previous_trading_date"] == "2026-01-02"


def test_load_market_calendar_decision_reads_qlib_calendar_and_metadata_sidecar(tmp_path):
    calendar_path = tmp_path / "data" / "qlib" / "soxl" / "calendars" / "day.txt"
    calendar_path.parent.mkdir(parents=True)
    calendar_path.write_text("2026-01-02\n2026-01-05\n2026-01-06\n", encoding="utf-8")
    metadata_path = tmp_path / "data" / "market_calendar" / "xnys_2026.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "schedule.market_calendar_source.v1",
                "exchange": "XNYS",
                "timezone": "America/New_York",
                "holidays": [{"date": "2026-01-01", "reason": "New Year's Day"}],
                "half_days": [{"date": "2026-01-02", "reason": "post-holiday half day"}],
            }
        ),
        encoding="utf-8",
    )

    closed = load_market_calendar_decision(
        "2026-01-01",
        calendar_path=calendar_path,
        metadata_path=metadata_path,
    )
    half_day_disabled = load_market_calendar_decision(
        "2026-01-02",
        calendar_path=calendar_path,
        metadata_path=metadata_path,
        allow_half_day=False,
    )

    assert closed["market_status"] == "closed"
    assert closed["skip_reason"] == "New Year's Day"
    assert closed["next_trading_date"] == "2026-01-02"
    assert closed["calendar_source"].endswith("data/qlib/soxl/calendars/day.txt")
    assert closed["metadata_source"].endswith("data/market_calendar/xnys_2026.json")
    assert closed["exchange"] == "XNYS"
    assert closed["timezone"] == "America/New_York"

    assert half_day_disabled["market_status"] == "half_day"
    assert half_day_disabled["should_run"] is False
    assert half_day_disabled["run_mode"] == "skipped_half_day"
    assert half_day_disabled["skip_reason"] == "half_day_disabled"


def test_run_job_once_skips_closed_market_calendar_decision_without_lock_task_or_notify(tmp_path):
    events: list[str] = []
    calls = {"count": 0}
    decision = classify_market_calendar_day(
        "2026-01-01",
        trading_calendar=("2026-01-02",),
        holiday_reasons={"2026-01-01": "New Year's Day"},
    )

    def task():
        calls["count"] += 1
        return {"should_not_run": True}

    job = ScheduleJob(name="paper_execution", task="paper_sim", cron="manual")
    result = run_job_once(
        job,
        task=task,
        lock_dir=tmp_path,
        notify=events.append,
        market_calendar_decision=decision,
    )

    assert result.status == "skipped"
    assert result.detail == "New Year's Day"
    assert result.payload["market_calendar_decision"]["market_status"] == "closed"
    assert result.payload["market_calendar_decision"]["next_trading_date"] == "2026-01-02"
    assert calls["count"] == 0
    assert events == []
    assert not (tmp_path / "paper_execution.lock").exists()


def test_run_job_once_can_notify_closed_market_calendar_skip_with_calendar_context(tmp_path):
    events: list[str] = []
    calls = {"count": 0}
    decision = classify_market_calendar_day(
        "2026-01-01",
        trading_calendar=("2026-01-02",),
        holiday_reasons={"2026-01-01": "New Year's Day"},
    )

    def task():
        calls["count"] += 1
        return {"should_not_run": True}

    job = ScheduleJob(name="paper_execution", task="paper_sim", cron="manual")
    result = run_job_once(
        job,
        task=task,
        lock_dir=tmp_path,
        notify=events.append,
        market_calendar_decision=decision,
        notify_skipped=True,
    )

    assert result.status == "skipped"
    assert calls["count"] == 0
    assert events == [
        "paper_execution:skipped | market_status=closed | "
        "skip_reason=New Year's Day | next_trading_date=2026-01-02"
    ]
    assert not (tmp_path / "paper_execution.lock").exists()


def test_run_task_with_retries_records_attempts_and_retry_error_history():
    calls = {"count": 0}

    def task():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary provider failure")
        return {"schema_version": "unit.retry_payload.v1", "status": "ok"}

    payload = run_task_with_retries(task, retries=2, failure_message="unit retry failed")

    assert payload["status"] == "ok"
    assert payload["attempts"] == 2
    assert payload["retry_schema_version"] == "schedule.retry_attempts.v1"
    assert payload["retry_error_count"] == 1
    assert payload["retry_errors"] == [
        {
            "attempt": 1,
            "error_type": "RuntimeError",
            "message": "temporary provider failure",
        }
    ]


def test_run_task_with_retries_raises_last_error_after_retry_budget():
    calls = {"count": 0}

    def task():
        calls["count"] += 1
        raise RuntimeError(f"permanent failure {calls['count']}")

    with pytest.raises(RuntimeError, match="permanent failure 2"):
        run_task_with_retries(task, retries=1, failure_message="unit retry failed")

    assert calls["count"] == 2
