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


def test_scheduler_retries_paper_trading_with_cached_data(monkeypatch: pytest.MonkeyPatch):
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
        "paper trading (skip data update)",
        "dashboard build",
    ]
    assert "--skip-data-update" not in calls[0][0]
    assert "--skip-data-update" in calls[1][0]
