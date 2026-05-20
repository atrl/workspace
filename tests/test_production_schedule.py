from __future__ import annotations

import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _soxl_cron_lines() -> list[str]:
    lines = (ROOT / "production" / "crontab.txt").read_text(encoding="utf-8").splitlines()
    return [
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith("#")
        and "production/soxl_daily_scheduler.py" in line
    ]


def test_crontab_uses_shanghai_safe_us_market_windows():
    lines = _soxl_cron_lines()

    post_close = next(line for line in lines if "scheduler_postclose.log" in line)
    pre_open = next(line for line in lines if "scheduler_preopen.log" in line)

    # Asia/Shanghai local time: Tue-Sat 05:35 covers Mon-Fri US post-close
    # in both EDT and EST, so the data refresh does not run before the close.
    assert post_close.split(maxsplit=5)[:5] == ["35", "5", "*", "*", "2-6"]
    assert "--skip-data-update" not in post_close
    assert "--skip-dashboard" not in post_close

    # Mon-Fri 21:00 China time is a cached-data pre-open inference/dashboard pass.
    assert pre_open.split(maxsplit=5)[:5] == ["0", "21", "*", "*", "1-5"]
    assert "--dry-run" in pre_open
    assert "--skip-data-update" in pre_open

    stale_local_times = {tuple(line.split(maxsplit=5)[:2]) for line in lines}
    assert ("30", "9") not in stale_local_times
    assert ("35", "16") not in stale_local_times


def test_launchagent_runs_post_close_in_shanghai_local_time():
    with (ROOT / "production" / "com.soxl.daily.scheduler.plist").open("rb") as f:
        plist = plistlib.load(f)

    intervals = plist["StartCalendarInterval"]
    schedule = {
        (entry["Weekday"], entry["Hour"], entry["Minute"])
        for entry in intervals
    }

    assert schedule == {
        (2, 5, 35),
        (3, 5, 35),
        (4, 5, 35),
        (5, 5, 35),
        (6, 5, 35),
    }
    assert plist["ProgramArguments"][0].endswith("/production/run_soxl_scheduler.sh")
    assert "--skip-data-update" not in plist["ProgramArguments"]
    assert plist["StandardOutPath"].endswith("/runtime/soxl/paper_trading/scheduler_postclose.log")
    assert plist["StandardErrorPath"].endswith("/runtime/soxl/paper_trading/scheduler_postclose.err")


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
    assert plist["ProgramArguments"][0].endswith("/production/run_soxl_scheduler.sh")
    assert "--dry-run" in plist["ProgramArguments"]
    assert "--skip-data-update" in plist["ProgramArguments"]
    assert "--skip-dashboard" not in plist["ProgramArguments"]
    assert plist["StandardOutPath"].endswith("/runtime/soxl/paper_trading/scheduler_preopen.log")
    assert plist["StandardErrorPath"].endswith("/runtime/soxl/paper_trading/scheduler_preopen.err")


def test_launchagent_wrapper_exists_and_loads_private_env_file():
    wrapper = ROOT / "production" / "run_soxl_scheduler.sh"

    assert wrapper.exists()
    assert wrapper.stat().st_mode & 0o111

    text = wrapper.read_text(encoding="utf-8")
    assert "SOXL_LAUNCH_ENV_FILE" in text
    assert ". \"$ENV_FILE\"" in text
    assert "soxl_daily_scheduler.py" in text
