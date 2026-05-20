from __future__ import annotations

import subprocess
import sys

import pytest

from production import soxl_daily_scheduler


def test_run_error_preserves_child_stderr_for_retry_classification(monkeypatch: pytest.MonkeyPatch):
    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="[paper] ERROR: Data update failed: No overlapping dates for SOXL/SOXX/VIX\n",
        )

    monkeypatch.setattr(soxl_daily_scheduler.subprocess, "run", fake_run)

    with pytest.raises(soxl_daily_scheduler.SchedulerStepError) as exc_info:
        soxl_daily_scheduler._run(["python", "production/soxl_paper_trading.py"], "paper trading")

    assert "No overlapping dates for SOXL/SOXX/VIX" in exc_info.value.stderr
    assert soxl_daily_scheduler._is_transient_data_source_failure(exc_info.value)


def test_scheduler_retries_data_source_failure_as_cached_dry_run(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str):
        calls.append((cmd, description))
        if len(calls) == 1:
            raise soxl_daily_scheduler.SchedulerStepError(
                description,
                1,
                stderr="[paper] ERROR: Data update failed: No overlapping dates for SOXL/SOXX/VIX\n",
            )

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify"])

    soxl_daily_scheduler.main()

    assert [description for _cmd, description in calls] == [
        "paper trading",
        "paper trading (cached dry-run)",
        "dashboard build",
    ]
    assert "--skip-data-update" not in calls[0][0]
    assert "--skip-data-update" in calls[1][0]
    assert "--dry-run" in calls[1][0]


def test_scheduler_retries_stale_post_update_data_as_cached_dry_run(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str):
        calls.append((cmd, description))
        if len(calls) == 1:
            raise soxl_daily_scheduler.SchedulerStepError(
                description,
                1,
                stderr=(
                    "[paper] ERROR: Data update completed but data is stale: "
                    "last_date=2026-05-15 expected=2026-05-20 stale_days=5\n"
                ),
            )

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify"])

    soxl_daily_scheduler.main()

    assert [description for _cmd, description in calls] == [
        "paper trading",
        "paper trading (cached dry-run)",
        "dashboard build",
    ]
    assert "--skip-data-update" in calls[1][0]
    assert "--dry-run" in calls[1][0]


def test_scheduler_smoke_mode_runs_cached_dry_run_with_dashboard(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str):
        calls.append((cmd, description))

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--smoke"])

    soxl_daily_scheduler.main()

    assert [description for _cmd, description in calls] == [
        "paper trading",
        "dashboard build",
    ]
    paper_cmd = calls[0][0]
    assert "--dry-run" in paper_cmd
    assert "--skip-data-update" in paper_cmd
    assert "--no-notify" in paper_cmd
