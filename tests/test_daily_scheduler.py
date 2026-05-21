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


def test_run_replays_child_stderr_on_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="provider=polygon\n",
            stderr="[paper] data provider yfinance failed: No overlapping dates\n",
        )

    monkeypatch.setattr(soxl_daily_scheduler.subprocess, "run", fake_run)

    soxl_daily_scheduler._run(["python", "production/soxl_paper_trading.py"], "paper trading")

    captured = capsys.readouterr()
    assert "provider=polygon" in captured.out
    assert "data provider yfinance failed" in captured.err


def test_run_allowed_returncode_does_not_print_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="vercel_dashboard status=blocked_no_credentials\n",
            stderr="",
        )

    monkeypatch.setattr(soxl_daily_scheduler.subprocess, "run", fake_run)

    returncode = soxl_daily_scheduler._run(
        ["python", "scripts/deploy_vercel_dashboard.py", "--deploy"],
        "vercel dashboard deploy",
        allowed_returncodes={2},
    )

    captured = capsys.readouterr()
    assert returncode == 2
    assert "blocked_no_credentials" in captured.out
    assert "ERROR in vercel dashboard deploy" not in captured.err


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


def test_scheduler_prints_run_start_marker(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def fake_run(cmd: list[str], description: str):
        return None

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify", "--dry-run"])

    soxl_daily_scheduler.main()

    captured = capsys.readouterr()
    assert "[scheduler] run started at " in captured.out
    assert "broker=paper dry_run=True" in captured.out


def test_scheduler_prints_goal_gates_when_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    calls: list[str] = []

    def fake_run(cmd: list[str], description: str):
        calls.append(description)

    def fake_health_summary():
        print("--- Production Goal Gates ---")
        print("  overall: pending scheduled_paper_proof")

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(soxl_daily_scheduler, "_run_health_summary", fake_health_summary, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["soxl_daily_scheduler.py", "--no-notify", "--dry-run", "--health-summary"],
    )

    soxl_daily_scheduler.main()

    captured = capsys.readouterr()
    assert calls == ["paper trading", "dashboard build"]
    assert "[scheduler] all done" in captured.out
    assert "--- Production Goal Gates ---" in captured.out
    assert captured.out.index("[scheduler] all done") < captured.out.index("--- Production Goal Gates ---")


def test_health_summary_suppresses_successful_stderr_noise(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                "=== SOXL Scheduler Health Check ===\n"
                "--- Production Goal Gates ---\n"
                "  overall: pending external_vercel_setup\n"
            ),
            stderr="Gym has been unmaintained since 2022\n",
        )

    monkeypatch.setattr(soxl_daily_scheduler.subprocess, "run", fake_run)

    soxl_daily_scheduler._run_health_summary()

    captured = capsys.readouterr()
    assert "[scheduler] production goal gates" in captured.out
    assert "--- Production Goal Gates ---" in captured.out
    assert "overall: pending external_vercel_setup" in captured.out
    assert "Gym has been unmaintained" not in captured.err


def test_health_summary_replays_stderr_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def fake_run(cmd, capture_output, text, check):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="",
            stderr="health script failed\n",
        )

    monkeypatch.setattr(soxl_daily_scheduler.subprocess, "run", fake_run)

    soxl_daily_scheduler._run_health_summary()

    captured = capsys.readouterr()
    assert "health script failed" in captured.err
    assert "health summary failed with exit code 1" in captured.err


def test_scheduler_prints_runtime_composer_in_run_start_marker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def fake_run(cmd: list[str], description: str):
        return None

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "soxl_daily_scheduler.py",
            "--no-notify",
            "--runtime-composer",
            "budget_usage",
            "--composer-profile",
            "v19reg_v22_budget_usage_bull_guard_v1",
            "--slow-profile",
            "balanced_rule_residual_v19_reg",
            "--fast-profile",
            "balanced_rule_residual_v22",
            "--bull-participation-guard",
        ],
    )

    soxl_daily_scheduler.main()

    captured = capsys.readouterr()
    assert "runtime_composer=budget_usage" in captured.out
    assert "composer_profile=v19reg_v22_budget_usage_bull_guard_v1" in captured.out
    assert "slow_profile=balanced_rule_residual_v19_reg" in captured.out
    assert "fast_profile=balanced_rule_residual_v22" in captured.out
    assert "bull_guard=True" in captured.out


def test_scheduler_passes_data_provider_to_paper_runner(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str):
        calls.append((cmd, description))

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify", "--data-provider", "yahoo_api"])

    soxl_daily_scheduler.main()

    paper_cmd = calls[0][0]
    assert "--data-provider" in paper_cmd
    assert paper_cmd[paper_cmd.index("--data-provider") + 1] == "yahoo_api"


def test_scheduler_runs_vercel_deploy_after_dashboard_when_requested(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str, allowed_returncodes=None):
        calls.append((cmd, description))
        return 0

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify", "--vercel-deploy"])

    soxl_daily_scheduler.main()

    assert [description for _cmd, description in calls] == [
        "paper trading",
        "dashboard build",
        "vercel dashboard deploy",
    ]
    vercel_cmd = calls[2][0]
    assert "scripts/deploy_vercel_dashboard.py" in vercel_cmd
    assert "--deploy" in vercel_cmd


def test_scheduler_treats_missing_vercel_credentials_as_nonfatal(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str, allowed_returncodes=None):
        calls.append((cmd, description))
        if description == "vercel dashboard deploy":
            assert allowed_returncodes == {2}
            return 2
        return 0

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify", "--vercel-deploy"])

    soxl_daily_scheduler.main()

    captured = capsys.readouterr()
    assert "Vercel deploy blocked by missing credentials" in captured.out
    assert "Vercel deploy blocked by missing credentials" not in captured.err
    assert "[scheduler] all done" in captured.out
    assert [description for _cmd, description in calls] == [
        "paper trading",
        "dashboard build",
        "vercel dashboard deploy",
    ]


def test_scheduler_accepts_polygon_data_provider(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str):
        calls.append((cmd, description))

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["soxl_daily_scheduler.py", "--no-notify", "--data-provider", "polygon"])

    soxl_daily_scheduler.main()

    paper_cmd = calls[0][0]
    assert "--data-provider" in paper_cmd
    assert paper_cmd[paper_cmd.index("--data-provider") + 1] == "polygon"


def test_scheduler_passes_runtime_composer_to_paper_runner(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[list[str], str]] = []

    def fake_run(cmd: list[str], description: str):
        calls.append((cmd, description))

    monkeypatch.setattr(soxl_daily_scheduler, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "soxl_daily_scheduler.py",
            "--no-notify",
            "--runtime-composer",
            "budget_usage",
            "--composer-profile",
            "v19reg_v22_budget_usage_bull_guard_v1",
            "--slow-profile",
            "balanced_rule_residual_v19_reg",
            "--fast-profile",
            "balanced_rule_residual_v22",
            "--bull-participation-guard",
        ],
    )

    soxl_daily_scheduler.main()

    paper_cmd = calls[0][0]
    for arg in (
        "--runtime-composer",
        "budget_usage",
        "--composer-profile",
        "v19reg_v22_budget_usage_bull_guard_v1",
        "--slow-profile",
        "balanced_rule_residual_v19_reg",
        "--fast-profile",
        "balanced_rule_residual_v22",
        "--bull-participation-guard",
    ):
        assert arg in paper_cmd
