from __future__ import annotations

import os
import plistlib
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_WRAPPER = ROOT / "production" / "run_soxl_scheduler.sh"
WEEKLY_WRAPPER = ROOT / "production" / "run_soxl_weekly_wf.sh"
CURRENT_COMPOSER_ARGS = [
    "--runtime-composer",
    "budget_usage",
    "--composer-profile",
    "v19reg_v22_budget_usage_bull_guard_v1",
    "--slow-profile",
    "balanced_rule_residual_v19_reg",
    "--fast-profile",
    "balanced_rule_residual_v22",
    "--bull-participation-guard",
]


def _assert_current_runtime_composer_args(args: list[str]) -> None:
    for arg in CURRENT_COMPOSER_ARGS:
        assert arg in args
    assert args[args.index("--runtime-composer") + 1] == "budget_usage"
    assert args[args.index("--composer-profile") + 1] == "v19reg_v22_budget_usage_bull_guard_v1"
    assert args[args.index("--slow-profile") + 1] == "balanced_rule_residual_v19_reg"
    assert args[args.index("--fast-profile") + 1] == "balanced_rule_residual_v22"


def _soxl_cron_lines() -> list[str]:
    lines = (ROOT / "production" / "crontab.txt").read_text(encoding="utf-8").splitlines()
    return [
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
        and (
            "production/soxl_daily_scheduler.py" in line
            or "production/run_soxl_scheduler.sh" in line
        )
    ]


def test_crontab_uses_shanghai_safe_us_market_windows():
    lines = _soxl_cron_lines()

    post_close = next(line for line in lines if "scheduler_postclose.log" in line)
    pre_open = next(line for line in lines if "scheduler_preopen.log" in line)

    # Asia/Shanghai local time: Tue-Sat 06:15 covers Mon-Fri US post-close
    # and leaves a provider-availability buffer before the daily data refresh.
    assert post_close.split(maxsplit=5)[:5] == ["15", "6", "*", "*", "2-6"]
    assert str(SCHEDULER_WRAPPER) in post_close
    assert "/Users/rziiiii/.openclaw/workspace" not in post_close
    assert "--skip-data-update" not in post_close
    assert "--skip-dashboard" not in post_close
    assert "--vercel-deploy" in post_close
    assert "--health-summary" in post_close
    assert "--schedule-slot postclose" in post_close
    assert "--broker futu" in post_close
    assert "--account-notional 10000" in post_close
    for arg in CURRENT_COMPOSER_ARGS:
        assert arg in post_close
    assert "--broker ibkr" not in post_close
    assert "SOXL_SCHEDULE_TRIGGER=cron_postclose" in post_close

    # Mon-Fri 21:00 China time is a cached-data pre-open inference/dashboard pass.
    assert pre_open.split(maxsplit=5)[:5] == ["0", "21", "*", "*", "1-5"]
    assert str(SCHEDULER_WRAPPER) in pre_open
    assert "/Users/rziiiii/.openclaw/workspace" not in pre_open
    assert "--dry-run" in pre_open
    assert "--skip-data-update" in pre_open
    assert "--vercel-deploy" in pre_open
    assert "--health-summary" in pre_open
    assert "--schedule-slot preopen" in pre_open
    assert "--broker futu" in pre_open
    assert "--account-notional 10000" in pre_open
    for arg in CURRENT_COMPOSER_ARGS:
        assert arg in pre_open
    assert "--broker ibkr" not in pre_open
    assert "SOXL_SCHEDULE_TRIGGER=cron_preopen" in pre_open

    stale_local_times = {tuple(line.split(maxsplit=5)[:2]) for line in lines}
    assert ("30", "9") not in stale_local_times
    assert ("35", "16") not in stale_local_times


def test_crontab_weekly_walkforward_uses_repo_wrapper():
    lines = (ROOT / "production" / "crontab.txt").read_text(encoding="utf-8").splitlines()
    weekly = next(
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
        and "wf_weekly.log" in line
    )

    assert weekly.split(maxsplit=5)[:5] == ["0", "7", "*", "*", "6"]
    assert str(WEEKLY_WRAPPER) in weekly
    assert "balanced_trade_lock" not in weekly
    assert "/Users/rziiiii/.openclaw/workspace" not in weekly


def test_operator_doc_scheduler_examples_include_current_production_flags():
    text = (ROOT / "production" / "FUTU_SEMI_AUTO.md").read_text(encoding="utf-8")
    scheduler_lines = [
        line
        for line in text.splitlines()
        if "production/run_soxl_scheduler.sh" in line
        and ("scheduler_preopen.log" in line or "scheduler_postclose.log" in line)
    ]

    pre_open = next(line for line in scheduler_lines if "scheduler_preopen.log" in line)
    post_close = next(line for line in scheduler_lines if "scheduler_postclose.log" in line)
    assert "--vercel-deploy" in pre_open
    assert "--health-summary" in pre_open
    assert "--schedule-slot preopen" in pre_open
    assert "SOXL_SCHEDULE_TRIGGER=cron_preopen" in pre_open
    assert "--vercel-deploy" in post_close
    assert "--health-summary" in post_close
    assert "--schedule-slot postclose" in post_close
    assert "SOXL_SCHEDULE_TRIGGER=cron_postclose" in post_close

    assert "export VERCEL_TOKEN='REPLACE_ME'" in text
    assert "export VERCEL_PROJECT_ID='REPLACE_ME'" in text
    assert "export VERCEL_ORG_ID='REPLACE_ME'" in text


def test_launchagent_runs_post_close_in_shanghai_local_time():
    with (ROOT / "production" / "com.soxl.daily.scheduler.plist").open("rb") as f:
        plist = plistlib.load(f)

    intervals = plist["StartCalendarInterval"]
    schedule = {
        (entry["Weekday"], entry["Hour"], entry["Minute"])
        for entry in intervals
    }

    assert schedule == {
        (2, 6, 15),
        (3, 6, 15),
        (4, 6, 15),
        (5, 6, 15),
        (6, 6, 15),
    }
    assert plist["ProgramArguments"][0] == str(SCHEDULER_WRAPPER)
    assert plist["WorkingDirectory"] == str(ROOT)
    assert "--vercel-deploy" in plist["ProgramArguments"]
    assert "--health-summary" in plist["ProgramArguments"]
    assert plist["ProgramArguments"][plist["ProgramArguments"].index("--broker") + 1] == "futu"
    assert plist["ProgramArguments"][plist["ProgramArguments"].index("--account-notional") + 1] == "10000"
    _assert_current_runtime_composer_args(plist["ProgramArguments"])
    assert "--schedule-slot" in plist["ProgramArguments"]
    assert plist["ProgramArguments"][plist["ProgramArguments"].index("--schedule-slot") + 1] == "postclose"
    assert plist["EnvironmentVariables"]["SOXL_SCHEDULE_TRIGGER"] == "launchagent_postclose"
    assert "--skip-data-update" not in plist["ProgramArguments"]
    assert plist["StandardOutPath"] == str(ROOT / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.log")
    assert plist["StandardErrorPath"] == str(ROOT / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.err")


def test_launchagent_runs_preopen_cached_dry_run_in_shanghai_local_time():
    with (ROOT / "production" / "com.soxl.preopen.scheduler.plist").open("rb") as f:
        plist = plistlib.load(f)

    intervals = plist["StartCalendarInterval"]
    schedule = {
        (entry["Weekday"], entry["Hour"], entry["Minute"])
        for entry in intervals
    }

    assert plist["Label"] == "com.soxl.preopen.scheduler"
    assert schedule == {
        (1, 21, 0),
        (2, 21, 0),
        (3, 21, 0),
        (4, 21, 0),
        (5, 21, 0),
    }
    assert plist["ProgramArguments"][0] == str(SCHEDULER_WRAPPER)
    assert plist["WorkingDirectory"] == str(ROOT)
    assert "--dry-run" in plist["ProgramArguments"]
    assert "--skip-data-update" in plist["ProgramArguments"]
    assert "--skip-dashboard" not in plist["ProgramArguments"]
    assert "--vercel-deploy" in plist["ProgramArguments"]
    assert "--health-summary" in plist["ProgramArguments"]
    assert plist["ProgramArguments"][plist["ProgramArguments"].index("--broker") + 1] == "futu"
    assert plist["ProgramArguments"][plist["ProgramArguments"].index("--account-notional") + 1] == "10000"
    _assert_current_runtime_composer_args(plist["ProgramArguments"])
    assert "--schedule-slot" in plist["ProgramArguments"]
    assert plist["ProgramArguments"][plist["ProgramArguments"].index("--schedule-slot") + 1] == "preopen"
    assert plist["EnvironmentVariables"]["SOXL_SCHEDULE_TRIGGER"] == "launchagent_preopen"
    assert plist["StandardOutPath"] == str(ROOT / "runtime" / "soxl" / "paper_trading" / "scheduler_preopen.log")
    assert plist["StandardErrorPath"] == str(ROOT / "runtime" / "soxl" / "paper_trading" / "scheduler_preopen.err")


def test_launchagent_runs_weekly_walkforward_after_postclose_refresh():
    with (ROOT / "production" / "com.soxl.weekly.wf.plist").open("rb") as f:
        plist = plistlib.load(f)

    intervals = plist["StartCalendarInterval"]
    schedule = {
        (entry["Weekday"], entry["Hour"], entry["Minute"])
        for entry in intervals
    }

    assert plist["Label"] == "com.soxl.weekly.wf"
    assert schedule == {(6, 7, 0)}
    assert plist["ProgramArguments"][0] == str(WEEKLY_WRAPPER)
    assert plist["WorkingDirectory"] == str(ROOT)
    assert plist["StandardOutPath"] == str(ROOT / "runtime" / "soxl" / "wf_weekly.log")
    assert plist["StandardErrorPath"] == str(ROOT / "runtime" / "soxl" / "wf_weekly.err")


def test_launchagent_wrapper_exists_and_loads_private_env_file():
    wrapper = ROOT / "production" / "run_soxl_scheduler.sh"

    assert wrapper.exists()
    assert wrapper.stat().st_mode & 0o111

    text = wrapper.read_text(encoding="utf-8")
    assert "SOXL_LAUNCH_ENV_FILE" in text
    assert "/opt/homebrew/bin" in text
    assert ". \"$ENV_FILE\"" in text
    assert "soxl_daily_scheduler.py" in text


def test_launchagent_wrapper_can_enable_runtime_composer_from_private_env_file():
    wrapper = ROOT / "production" / "run_soxl_scheduler.sh"

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        root.mkdir(exist_ok=True)
        env_file = root / "soxl.env"
        capture_file = root / "captured_args.txt"
        fake_python = root / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$SOXL_CAPTURE_ARGS\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env_file.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
                    "export SOXL_RUNTIME_BULL_PARTICIPATION_GUARD='true'",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        env = {
            **os.environ,
            "SOXL_SCHEDULER_ROOT": str(root),
            "SOXL_SCHEDULER_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
            "SOXL_CAPTURE_ARGS": str(capture_file),
        }
        result = subprocess.run(
            ["bash", str(wrapper), "--broker", "paper", "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

        captured_args = capture_file.read_text(encoding="utf-8").splitlines()

    assert result.returncode == 0
    assert captured_args[0] == f"{root}/production/soxl_daily_scheduler.py"
    assert captured_args[-9:] == [
        "--runtime-composer",
        "budget_usage",
        "--composer-profile",
        "v19reg_v22_budget_usage_bull_guard_v1",
        "--slow-profile",
        "balanced_rule_residual_v19_reg",
        "--fast-profile",
        "balanced_rule_residual_v22",
        "--bull-participation-guard",
    ]


def test_launchagent_wrapper_defaults_to_its_own_repo_root():
    wrapper = ROOT / "production" / "run_soxl_scheduler.sh"

    with TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        capture_file = temp_dir / "captured_args.txt"
        fake_python = temp_dir / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$SOXL_CAPTURE_ARGS\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env_file = temp_dir / "missing.env"

        env = {
            **os.environ,
            "SOXL_SCHEDULER_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
            "SOXL_CAPTURE_ARGS": str(capture_file),
        }
        env.pop("SOXL_SCHEDULER_ROOT", None)
        result = subprocess.run(
            ["bash", str(wrapper), "--broker", "paper", "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

        captured_args = capture_file.read_text(encoding="utf-8").splitlines()

    assert result.returncode == 0
    assert captured_args[0] == str(ROOT / "production" / "soxl_daily_scheduler.py")


def test_weekly_walkforward_wrapper_reviews_current_runtime_profile():
    wrapper = ROOT / "production" / "run_soxl_weekly_wf.sh"

    assert wrapper.exists()
    assert wrapper.stat().st_mode & 0o111

    text = wrapper.read_text(encoding="utf-8")
    assert "SOXL_LAUNCH_ENV_FILE" in text
    assert "balanced_rule_residual_v22" in text
    assert "scripts/walkforward_backtest_validate.py" in text
    assert "scripts/review_runtime_selection.py" in text
    assert "--persist-model-promotion" in text
    assert "balanced_trade_lock" not in text
    assert "/home/rziiiii/dev/workspace" not in text


def test_weekly_walkforward_wrapper_reviews_enabled_runtime_composer():
    wrapper = ROOT / "production" / "run_soxl_weekly_wf.sh"

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        env_file = root / "soxl.env"
        capture_file = root / "captured_args.txt"
        fake_python = root / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' '---cmd---' \"$@\" >> \"$SOXL_CAPTURE_ARGS\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env_file.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
                    "export SOXL_RUNTIME_BULL_PARTICIPATION_GUARD='true'",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        env = {
            **os.environ,
            "SOXL_WEEKLY_WF_ROOT": str(root),
            "SOXL_WEEKLY_WF_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
            "SOXL_CAPTURE_ARGS": str(capture_file),
        }
        result = subprocess.run(
            ["bash", str(wrapper)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

        captured = capture_file.read_text(encoding="utf-8").splitlines()

    assert result.returncode == 0
    assert "mode=runtime_composer" in result.stdout
    assert f"{root}/scripts/review_budget_usage_runtime_promotion.py" in captured
    assert "--profile" in captured
    assert "v19reg_v22_budget_usage_bull_guard_v1" in captured
    assert "--slow-profile" in captured
    assert "balanced_rule_residual_v19_reg" in captured
    assert "--fast-profile" in captured
    assert "balanced_rule_residual_v22" in captured
    assert f"{root}/scripts/walkforward_backtest_validate.py" not in captured
    assert f"{root}/scripts/review_runtime_selection.py" not in captured
