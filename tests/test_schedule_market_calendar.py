from __future__ import annotations

from src.schedule import ScheduleJob, classify_market_calendar_day, run_job_once


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
