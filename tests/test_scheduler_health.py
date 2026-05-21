from __future__ import annotations

import os
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scheduler_health_reports_macos_launchagent_and_logs():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        (root / "runtime" / "soxl").mkdir(parents=True, exist_ok=True)
        (runtime_dir / "scheduler_preopen.log").write_text(
            "[scheduler] all done\n",
            encoding="utf-8",
        )
        (runtime_dir / "scheduler_postclose.log").write_text(
            "[scheduler] all done (with cached dry-run)\n",
            encoding="utf-8",
        )
        (root / "runtime" / "soxl" / "wf_weekly.log").write_text(
            "[weekly-wf] all done\n",
            encoding="utf-8",
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n789\\t0\\tcom.soxl.weekly.wf\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":false,\"last_date\":\"2026-05-15\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":5},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":false}\n"
            "JSON\n"
            "exit 1\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Scheduler Service ---" in result.stdout
    assert "LaunchAgent com.soxl.preopen.scheduler: loaded" in result.stdout
    assert "LaunchAgent com.soxl.daily.scheduler: loaded" in result.stdout
    assert "LaunchAgent com.soxl.weekly.wf: loaded" in result.stdout
    assert "scheduler_preopen: [scheduler] all done" in result.stdout
    assert "wf_weekly: [weekly-wf] all done" in result.stdout
    assert "--- Last Completion Marker ---" in result.stdout
    assert "--- Last Successful Run ---" not in result.stdout
    assert "scheduler_postclose: [scheduler] all done (with cached dry-run)" in result.stdout
    assert "--- Paper Trading Health ---" in result.stdout
    assert "data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5" in result.stdout
    assert "model: ok balanced_rule_residual_v19" in result.stdout
    assert "broker: ok paper broker ready" in result.stdout
    assert "--- Dashboard Static Export ---" in result.stdout
    assert "unavailable: integrity script missing" in result.stdout


def test_scheduler_health_reports_weekly_logs_pending_before_first_run():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        (root / "runtime" / "soxl").mkdir(parents=True, exist_ok=True)

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n789\\t0\\tcom.soxl.weekly.wf\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "1300",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "runtime/soxl/wf_weekly.log: pending (not created yet)" in result.stdout
    assert "runtime/soxl/wf_weekly.err: pending (not created yet)" in result.stdout
    assert "wf_weekly: pending until first weekly run" in result.stdout
    assert "runtime/soxl/wf_weekly.log: NOT FOUND" not in result.stdout
    assert "runtime/soxl/wf_weekly.err: NOT FOUND" not in result.stdout


def test_scheduler_health_reports_vercel_remote_deployment_status():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        vercel_dir = root / "runtime" / "production" / "card" / "vercel_remote_deploy_v1"
        vercel_dir.mkdir(parents=True)
        (vercel_dir / "vercel_deployment_check.json").write_text(
            """
{
  "checked_at": "2026-05-21T01:00:00+00:00",
  "status": "blocked_no_credentials",
  "deployment_url": null,
  "static_export": {
    "latest": "paper_trading",
    "paper_trading_end": "2026-05-20"
  },
  "auth": {
    "status": "blocked_no_credentials"
  },
  "project": {
    "status": "missing_project_link"
  },
  "detail": "Set VERCEL_TOKEN or run vercel login."
}
""",
            encoding="utf-8",
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Vercel Remote Deployment ---" in result.stdout
    assert "status: blocked_no_credentials" in result.stdout
    assert "auth: blocked_no_credentials" in result.stdout
    assert "project: missing_project_link" in result.stdout
    assert "deployment_url: none" in result.stdout
    assert "dashboard_latest=paper_trading paper_trading_end=2026-05-20" in result.stdout
    assert "detail: Set VERCEL_TOKEN or run vercel login." in result.stdout


def test_scheduler_health_reports_stale_vercel_evidence_when_dashboard_is_newer():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        data_dir = root / "dashboard_data"
        data_dir.mkdir()
        (data_dir / "index.json").write_text(
            """
{
  "generated_at": "2026-05-21 12:01:45",
  "latest_version": "paper_trading"
}
""",
            encoding="utf-8",
        )
        vercel_dir = root / "runtime" / "production" / "card" / "vercel_remote_deploy_v1"
        vercel_dir.mkdir(parents=True)
        (vercel_dir / "vercel_deployment_check.json").write_text(
            """
{
  "checked_at": "2026-05-21T03:07:35+00:00",
  "status": "blocked_no_credentials",
  "deployment_url": null,
  "static_export": {
    "generated_at": "2026-05-21 11:07:30",
    "latest": "paper_trading",
    "paper_trading_end": "2026-05-20"
  },
  "auth": {
    "status": "blocked_no_credentials"
  },
  "detail": "Set VERCEL_TOKEN or run vercel login."
}
""",
            encoding="utf-8",
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert (
        "freshness: stale local_generated_at=2026-05-21 12:01:45 "
        "evidence_generated_at=2026-05-21 11:07:30"
    ) in result.stdout


def test_scheduler_health_reports_missing_launch_env_file_for_vercel_token():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        env_file = root / "private" / "soxl.env"

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
        }
        env.pop("VERCEL_TOKEN", None)
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Scheduler Private Env ---" in result.stdout
    assert f"launch env file: missing path={env_file}" in result.stdout
    assert "VERCEL_TOKEN: missing in current environment and launch env file" in result.stdout


def test_scheduler_health_reports_launch_env_vercel_token_without_leaking_secret():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        env_file = root / "private" / "soxl.env"
        env_file.parent.mkdir()
        env_file.write_text("export VERCEL_TOKEN='secret-token-123'\n", encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
        }
        env.pop("VERCEL_TOKEN", None)
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert f"launch env file: present path={env_file}" in result.stdout
    assert "VERCEL_TOKEN: configured in launch env file" in result.stdout
    assert "secret-token-123" not in result.stdout


def test_scheduler_health_reports_launch_env_vercel_project_without_leaking_ids():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        env_file = root / "private" / "soxl.env"
        env_file.parent.mkdir()
        env_file.write_text(
            "\n".join(
                [
                    "export VERCEL_PROJECT_ID='project_secretish'",
                    "export VERCEL_ORG_ID='team_secretish'",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
        }
        env.pop("VERCEL_PROJECT_ID", None)
        env.pop("VERCEL_ORG_ID", None)
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "VERCEL_PROJECT: configured in launch env file" in result.stdout
    assert "project_secretish" not in result.stdout
    assert "team_secretish" not in result.stdout


def test_scheduler_health_reports_launch_env_runtime_composer_configuration():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        env_file = root / "private" / "soxl.env"
        env_file.parent.mkdir()
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

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
        }
        for key in (
            "SOXL_RUNTIME_COMPOSER",
            "SOXL_RUNTIME_COMPOSER_PROFILE",
            "SOXL_RUNTIME_SLOW_PROFILE",
            "SOXL_RUNTIME_FAST_PROFILE",
            "SOXL_RUNTIME_BULL_PARTICIPATION_GUARD",
        ):
            env.pop(key, None)
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert (
        "runtime composer: enabled composer=budget_usage "
        "profile=v19reg_v22_budget_usage_bull_guard_v1 "
        "slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22 "
        "bull_guard=true"
    ) in result.stdout


def test_scheduler_health_reports_incomplete_launch_env_runtime_composer_configuration():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        env_file = root / "private" / "soxl.env"
        env_file.parent.mkdir()
        env_file.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_LAUNCH_ENV_FILE": str(env_file),
        }
        for key in (
            "SOXL_RUNTIME_COMPOSER",
            "SOXL_RUNTIME_COMPOSER_PROFILE",
            "SOXL_RUNTIME_SLOW_PROFILE",
            "SOXL_RUNTIME_FAST_PROFILE",
            "SOXL_RUNTIME_BULL_PARTICIPATION_GUARD",
        ):
            env.pop(key, None)
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert (
        "runtime composer: incomplete composer=budget_usage "
        "missing=SOXL_RUNTIME_FAST_PROFILE"
    ) in result.stdout
    assert "runtime composer: enabled composer=budget_usage" not in result.stdout


def test_scheduler_health_reports_installed_launchagent_plist_drift():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        repo_preopen = production_dir / "com.soxl.preopen.scheduler.plist"
        repo_daily = production_dir / "com.soxl.daily.scheduler.plist"
        repo_preopen.write_text("<plist>preopen-current</plist>\n", encoding="utf-8")
        repo_daily.write_text("<plist>daily-current</plist>\n", encoding="utf-8")
        (launchagents_dir / repo_preopen.name).write_text(repo_preopen.read_text(encoding="utf-8"), encoding="utf-8")
        (launchagents_dir / repo_daily.name).write_text("<plist>daily-stale</plist>\n", encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-19\","
            "\"expected_last_trading_day\":\"2026-05-19\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "LaunchAgent com.soxl.preopen.scheduler plist: matches repo" in result.stdout
    assert "LaunchAgent com.soxl.daily.scheduler plist: differs from repo" in result.stdout


def test_scheduler_health_reports_latest_signal_artifact():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-19"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-19",
  "profile": "balanced_rule_residual_v19",
  "dry_run": true,
  "executed": false,
  "signal": {"target_position": 0.4},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": "manual_review"}
}
""",
            encoding="utf-8",
        )
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-19\","
            "\"expected_last_trading_day\":\"2026-05-19\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Latest Signal Artifact ---" in result.stdout
    assert "latest signal: date=2026-05-19 profile=balanced_rule_residual_v19 dry_run=true executed=false" in result.stdout
    assert "target=0.4 risk_gate=approved reconcile=manual_review" in result.stdout


def test_scheduler_health_reports_pending_runtime_profile_proof():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-20"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "balanced_rule_residual_v19",
  "dry_run": false,
  "executed": true,
  "signal": {"target_position": 0.48},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": null}
}
""",
            encoding="utf-8",
        )
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "latest signal: date=2026-05-20 profile=balanced_rule_residual_v19 dry_run=false executed=true" in result.stdout
    assert (
        "runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 "
        "selected_model=balanced_rule_residual_v22"
    ) in result.stdout
    assert (
        "runtime proof detail: latest executed date already matches expected data date; "
        "next non-dry-run execution must use selected model"
    ) in result.stdout


def test_scheduler_health_accepts_selected_profile_duplicate_skip_as_runtime_proof():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-20"
        duplicate_dir = runtime_dir / "duplicate_skip" / "balanced_rule_residual_v22"
        duplicate_dir.mkdir(parents=True)
        (root / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.log").write_text(
            "[scheduler] all done\n",
            encoding="utf-8",
        )
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "balanced_rule_residual_v19",
  "dry_run": false,
  "executed": true,
  "signal": {"target_position": 0.48},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": "none"}
}
""",
            encoding="utf-8",
        )
        (duplicate_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "balanced_rule_residual_v22",
  "dry_run": false,
  "executed": false,
  "execution_result": {"submitted": false, "status": "duplicate_skipped"},
  "signal": {"target_position": 0.50},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": "none"}
}
""",
            encoding="utf-8",
        )
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert (
        "runtime proof: ok latest_non_dry_profile=balanced_rule_residual_v22 "
        "selected_model=balanced_rule_residual_v22 execution=duplicate_skipped"
    ) in result.stdout
    assert (
        "ordinary paper proof: pending latest_non_dry_profile=balanced_rule_residual_v22 "
        "selected_model=balanced_rule_residual_v22 execution=duplicate_skipped"
    ) in result.stdout
    assert (
        "postclose proof: ordinary paper proof pending selected_model=balanced_rule_residual_v22 "
        "execution=duplicate_skipped"
    ) in result.stdout
    assert "runtime proof: pending latest_executed_profile=balanced_rule_residual_v19" not in result.stdout
    assert "artifact=runtime/soxl/paper_trading/2026-05-20/duplicate_skip/balanced_rule_residual_v22/summary.json" in result.stdout


def test_scheduler_health_targets_enabled_composer_for_model_and_runtime_proof():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-20"
        duplicate_dir = runtime_dir / "duplicate_skip" / "balanced_rule_residual_v22"
        duplicate_dir.mkdir(parents=True)
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "balanced_rule_residual_v19",
  "dry_run": false,
  "executed": true,
  "signal": {"target_position": 0.48},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": "none"}
}
""",
            encoding="utf-8",
        )
        (duplicate_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "balanced_rule_residual_v22",
  "dry_run": false,
  "executed": false,
  "execution_result": {"submitted": false, "status": "duplicate_skipped"},
  "signal": {"target_position": 0.50},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": "none"}
}
""",
            encoding="utf-8",
        )
        launch_env = root / "soxl.env"
        launch_env.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
                    "export SOXL_RUNTIME_BULL_PARTICIPATION_GUARD='true'",
                ]
            ),
            encoding="utf-8",
        )
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "case \" $* \" in\n"
            "  *\" --runtime-composer budget_usage \"*)\n"
            "    cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n"
            "    ;;\n"
            "  *)\n"
            "    cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22 "
            "(soxl_balanced_rule_residual_v22_actor.pt)\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n"
            "    ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert (
        "runtime composer: enabled composer=budget_usage "
        "profile=v19reg_v22_budget_usage_bull_guard_v1 "
        "slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22 "
        "bull_guard=true"
    ) in result.stdout
    assert (
        "model: ok v19reg_v22_budget_usage_bull_guard_v1 "
        "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)"
    ) in result.stdout
    assert (
        "runtime proof: pending latest_executed_profile=balanced_rule_residual_v19 "
        "selected_model=v19reg_v22_budget_usage_bull_guard_v1"
    ) in result.stdout
    assert "selected_model=balanced_rule_residual_v22" not in result.stdout


def test_scheduler_health_accepts_selected_profile_non_dry_skipped_order_as_runtime_proof():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-21"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-21",
  "profile": "balanced_rule_residual_v22",
  "dry_run": false,
  "executed": false,
  "execution_result": {"submitted": false, "status": "skipped"},
  "signal": {"target_position": 0.50},
  "risk_gate": {"approved": true},
  "reconcile": {"action_required": "none"}
}
""",
            encoding="utf-8",
        )
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-21\","
            "\"expected_last_trading_day\":\"2026-05-21\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22 (soxl_balanced_rule_residual_v22_actor.pt)\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "latest signal: date=2026-05-21 profile=balanced_rule_residual_v22 dry_run=false executed=false" in result.stdout
    assert (
        "runtime proof: ok latest_non_dry_profile=balanced_rule_residual_v22 "
        "selected_model=balanced_rule_residual_v22 execution=skipped"
    ) in result.stdout
    assert (
        "ordinary paper proof: ok latest_non_dry_profile=balanced_rule_residual_v22 "
        "selected_model=balanced_rule_residual_v22 execution=skipped"
    ) in result.stdout


def test_scheduler_health_reports_postclose_due_without_success_marker():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-19\","
            "\"expected_last_trading_day\":\"2026-05-19\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0630",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: due but no success marker found" in result.stdout


def test_scheduler_health_keeps_postclose_pending_during_provider_buffer():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0600",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: pending until 06:15 Tue-Sat Asia/Shanghai" in result.stdout
    assert "postclose status: due" not in result.stdout


def test_scheduler_health_ignores_old_postclose_success_marker_after_due():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        old_log = runtime_dir / "scheduler_postclose.log"
        old_log.write_text("[scheduler] all done\n", encoding="utf-8")
        os.utime(old_log, (100, 100))
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-19\","
            "\"expected_last_trading_day\":\"2026-05-19\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0630",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "200",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "scheduler_postclose: [scheduler] all done" in result.stdout
    assert "postclose status: due but no current success marker found" in result.stdout


def test_scheduler_health_requires_success_after_current_postclose_run_start():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        postclose_log = runtime_dir / "scheduler_postclose.log"
        postclose_log.write_text(
            "\n".join(
                [
                    "[scheduler] run started at 2026-05-20T06:15:00+08:00",
                    "[scheduler] all done",
                    "[scheduler] run started at 2026-05-21T06:15:00+08:00",
                    "[scheduler] FAILED: paper trading failed with exit code 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.utime(postclose_log, (300, 300))
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-19\","
            "\"expected_last_trading_day\":\"2026-05-19\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0630",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "200",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: due but no current success marker found" in result.stdout


def test_scheduler_health_does_not_count_cached_dry_run_as_postclose_success():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        postclose_log = runtime_dir / "scheduler_postclose.log"
        postclose_log.write_text(
            "\n".join(
                [
                    "[scheduler] run started at 2026-05-21T06:15:04+08:00 broker=paper dry_run=False",
                    "[scheduler] FAILED: paper trading failed with exit code 1",
                    "[scheduler] Fresh data unavailable or stale. Retrying cached data as dry-run...",
                    "[scheduler] all done (with cached dry-run)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.utime(postclose_log, (300, 300))
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":false,\"last_date\":\"2026-05-19\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":1},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v19\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":false}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0630",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "200",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "scheduler_postclose: [scheduler] all done (with cached dry-run)" in result.stdout
    assert "postclose status: due but only cached dry-run fallback completed" in result.stdout
    assert "postclose status: success marker found" not in result.stdout


def test_scheduler_health_next_runs_skip_elapsed_postclose_window():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        production_dir = root / "production"
        production_dir.mkdir()
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
        for label in ("com.soxl.preopen.scheduler", "com.soxl.daily.scheduler"):
            repo_plist = production_dir / f"{label}.plist"
            repo_plist.write_text(f"<plist>{label}</plist>\n", encoding="utf-8")
            (launchagents_dir / repo_plist.name).write_text(repo_plist.read_text(encoding="utf-8"), encoding="utf-8")

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text(
            "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
            encoding="utf-8",
        )
        launchctl.chmod(0o755)
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"paper broker ready\"},"
            "\"all_ok\":true}\n"
            "JSON\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        env = {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
            "SOXL_SCHEDULER_HEALTH_ROOT": str(root),
            "SOXL_SCHEDULER_HEALTH_PYTHON": str(fake_python),
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "4",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0758",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Next Scheduled Runs ---" in result.stdout
    assert "pre-open: today 21:00 Mon-Fri Asia/Shanghai" in result.stdout
    assert "US pre-open dry-run + dashboard + Vercel deploy attempt" in result.stdout
    assert "post-close: next eligible day 06:15 Tue-Sat Asia/Shanghai" in result.stdout
    assert "US post-close data + paper trading + dashboard + Vercel deploy attempt" in result.stdout
    assert "post-close: today 06:15 Tue-Sat Asia/Shanghai" not in result.stdout
