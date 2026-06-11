from __future__ import annotations

import os
import plistlib
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_scheduler_plist(
    path: Path,
    *,
    broker: str = "paper",
    account_notional: str = "",
    extra_args: list[str] | None = None,
) -> None:
    args = [
        str(path.parents[1] / "production" / "run_soxl_scheduler.sh"),
        "--broker",
        broker,
        "--schedule-slot",
        "postclose",
    ]
    if account_notional:
        args.extend(["--account-notional", account_notional])
    if extra_args:
        args.extend(extra_args)
    payload = {
        "Label": "com.soxl.daily.scheduler",
        "ProgramArguments": args,
        "WorkingDirectory": str(path.parents[1]),
    }
    with path.open("wb") as f:
        plistlib.dump(payload, f)


def test_scheduler_health_uses_configured_launchagent_broker_for_healthcheck():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        production_dir = root / "production"
        launchagents_dir = root / "LaunchAgents"
        production_dir.mkdir()
        launchagents_dir.mkdir()
        _write_scheduler_plist(
            production_dir / "com.soxl.daily.scheduler.plist",
            broker="futu",
            account_notional="10000",
        )
        _write_scheduler_plist(
            launchagents_dir / "com.soxl.daily.scheduler.plist",
            broker="futu",
            account_notional="10000",
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text("#!/bin/sh\nprintf '456\\t0\\tcom.soxl.daily.scheduler\\n'\n", encoding="utf-8")
        launchctl.chmod(0o755)
        captured_args = root / "health_args.txt"
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$SOXL_CAPTURE_HEALTH_ARGS\"\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-22\","
            "\"expected_last_trading_day\":\"2026-05-22\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1\"},"
            "\"broker\":{\"ok\":true,\"msg\":\"futu quote ok\"},"
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
            "SOXL_CAPTURE_HEALTH_ARGS": str(captured_args),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        args = captured_args.read_text(encoding="utf-8").splitlines()

    assert "--broker" in args
    assert args[args.index("--broker") + 1] == "futu"
    assert "--account-notional" in args
    assert args[args.index("--account-notional") + 1] == "10000"
    assert "broker target: futu" in result.stdout
    assert "account notional: 10000" in result.stdout
    assert "broker: ok futu quote ok" in result.stdout


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


def test_scheduler_health_reports_stale_data_registry_check():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        scripts_dir = root / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "update_data_registry.py").write_text("# test stub\n", encoding="utf-8")

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
            "  *\" scripts/update_data_registry.py --check \"*)\n"
            "    cat <<'TEXT'\n"
            "update_data_registry: FAIL\n"
            "- stale registry processed_panels.soxl_soxx_vix_full.end: stored=2026-05-19 current=2026-05-20\n"
            "TEXT\n"
            "    exit 1\n"
            "    ;;\n"
            "  *)\n"
            "    cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"balanced_rule_residual_v22\"},"
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
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Data Registry Health ---" in result.stdout
    assert "update_data_registry: FAIL" in result.stdout
    assert (
        "stale registry processed_panels.soxl_soxx_vix_full.end: "
        "stored=2026-05-19 current=2026-05-20"
    ) in result.stdout
    assert "data registry: failed update_data_registry: FAIL" in result.stdout
    assert "overall: pending" in result.stdout
    assert "data_registry_freshness" in result.stdout


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
  "env_file": {
    "present": true,
    "applied_keys": ["VERCEL_ORG_ID", "VERCEL_PROJECT_ID", "VERCEL_TOKEN"]
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
    assert "env_file: present applied_keys=VERCEL_ORG_ID,VERCEL_PROJECT_ID,VERCEL_TOKEN" in result.stdout
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
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
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
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
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


def test_scheduler_health_treats_vercel_placeholders_as_missing():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        env_file = root / "private" / "soxl.env"
        env_file.parent.mkdir()
        env_file.write_text(
            "\n".join(
                [
                    "export VERCEL_TOKEN='REPLACE_ME'",
                    "export VERCEL_PROJECT_ID='replace-me'",
                    "export VERCEL_ORG_ID=''",
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
        env.pop("VERCEL_TOKEN", None)
        env.pop("VERCEL_PROJECT_ID", None)
        env.pop("VERCEL_ORG_ID", None)
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "VERCEL_TOKEN: missing in current environment and launch env file" in result.stdout
    assert "VERCEL_PROJECT: missing VERCEL_PROJECT_ID or VERCEL_ORG_ID in current environment and launch env file" in result.stdout


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


def test_scheduler_health_reads_runtime_composer_from_launchagent_args():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        production_dir = root / "production"
        launchagents_dir = root / "LaunchAgents"
        production_dir.mkdir()
        launchagents_dir.mkdir()
        composer_args = [
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
        _write_scheduler_plist(
            production_dir / "com.soxl.daily.scheduler.plist",
            broker="futu",
            account_notional="10000",
            extra_args=composer_args,
        )
        _write_scheduler_plist(
            launchagents_dir / "com.soxl.daily.scheduler.plist",
            broker="futu",
            account_notional="10000",
            extra_args=composer_args,
        )

        fake_bin = root / "bin"
        fake_bin.mkdir()
        launchctl = fake_bin / "launchctl"
        launchctl.write_text("#!/bin/sh\nprintf '456\\t0\\tcom.soxl.daily.scheduler\\n'\n", encoding="utf-8")
        launchctl.chmod(0o755)
        captured_args = root / "health_args.txt"
        fake_python = fake_bin / "python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$SOXL_CAPTURE_HEALTH_ARGS\"\n"
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-06-10\","
            "\"expected_last_trading_day\":\"2026-06-10\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_LAUNCH_ENV_FILE": str(root / "missing.env"),
            "SOXL_CAPTURE_HEALTH_ARGS": str(captured_args),
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
        args = captured_args.read_text(encoding="utf-8").splitlines()

    assert (
        "runtime composer: enabled composer=budget_usage "
        "profile=v19reg_v22_budget_usage_bull_guard_v1 "
        "slow=balanced_rule_residual_v19_reg fast=balanced_rule_residual_v22 "
        "bull_guard=true"
    ) in result.stdout
    assert "--runtime-composer" in args
    assert args[args.index("--runtime-composer") + 1] == "budget_usage"
    assert "--composer-profile" in args
    assert args[args.index("--composer-profile") + 1] == "v19reg_v22_budget_usage_bull_guard_v1"


def test_scheduler_health_reports_incomplete_launch_env_runtime_composer_configuration():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "runtime" / "soxl" / "paper_trading").mkdir(parents=True)
        launchagents_dir = root / "LaunchAgents"
        launchagents_dir.mkdir()
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
            "SOXL_SCHEDULER_HEALTH_LAUNCHAGENTS_DIR": str(launchagents_dir),
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
        vercel_dir = root / "runtime" / "production" / "card" / "vercel_remote_deploy_v1"
        vercel_dir.mkdir(parents=True)
        (vercel_dir / "vercel_deployment_check.json").write_text(
            """
{
  "checked_at": "2026-05-21T01:00:00+00:00",
  "status": "deployed",
  "deployment_url": "https://soxl-dashboard-example.vercel.app",
  "static_export": {
    "latest": "paper_trading",
    "paper_trading_end": "2026-05-20"
  },
  "auth": {
    "status": "token"
  },
  "project": {
    "status": "env_configured"
  }
}
""",
            encoding="utf-8",
        )
        (root / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.log").write_text(
            "\n".join(
                [
                    "[scheduler] run started at 2026-05-21T06:15:00+08:00 broker=paper dry_run=False",
                    "[scheduler] all done",
                    "",
                ]
            ),
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
            "SOXL_SCHEDULER_HEALTH_DATE": "2026-05-21",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "1",
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
    assert (
        "postclose target: pending latest_success_marker lacks schedule_slot=postclose"
    ) in result.stdout
    assert "postclose status: due but current success marker lacks schedule_slot=postclose" in result.stdout
    assert "postclose status: success marker found" not in result.stdout
    assert "--- Production Goal Gates ---" in result.stdout
    assert (
        "post-close composer target: pending latest_success_marker lacks schedule_slot=postclose"
    ) in result.stdout
    assert (
        "ordinary paper proof: unknown selected_model=v19reg_v22_budget_usage_bull_guard_v1 "
        "execution=unknown"
    ) in result.stdout
    assert (
        "overall: pending data_registry_freshness,dashboard_static_export,"
        "scheduled_postclose_target,scheduled_paper_proof"
    ) in result.stdout
    assert "selected_model=balanced_rule_residual_v22" not in result.stdout


def test_scheduler_health_requires_postclose_schedule_slot_for_composer_target():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-20"
        runtime_dir.mkdir(parents=True)
        (root / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.log").write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-21T06:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto "
                        "runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] all done",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "v19reg_v22_budget_usage_bull_guard_v1",
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
        launch_env = root / "soxl.env"
        launch_env.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_DATE": "2026-05-21",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "1",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose target: pending latest_success_marker lacks schedule_slot=postclose" in result.stdout
    assert "post-close composer target: pending latest_success_marker lacks schedule_slot=postclose" in result.stdout


def test_scheduler_health_requires_postclose_schedule_trigger_for_composer_target():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-20"
        runtime_dir.mkdir(parents=True)
        (root / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.log").write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-21T06:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto "
                        "schedule_slot=postclose runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] all done",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "v19reg_v22_budget_usage_bull_guard_v1",
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
        launch_env = root / "soxl.env"
        launch_env.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "1",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: due but current success marker lacks schedule_trigger=postclose" in result.stdout
    assert "postclose target: pending latest_success_marker lacks schedule_trigger=postclose" in result.stdout
    assert "post-close composer target: pending latest_success_marker lacks schedule_trigger=postclose" in result.stdout


def test_scheduler_health_rejects_postclose_marker_outside_schedule_window():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading" / "2026-05-20"
        runtime_dir.mkdir(parents=True)
        (root / "runtime" / "soxl" / "paper_trading" / "scheduler_postclose.log").write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-21T21:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto "
                        "schedule_slot=postclose schedule_trigger=launchagent_postclose "
                        "runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] all done",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "v19reg_v22_budget_usage_bull_guard_v1",
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
        launch_env = root / "soxl.env"
        launch_env.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_HHMM": "2115",
            "SOXL_SCHEDULER_HEALTH_DATE": "2026-05-21",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "1",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: due but current success marker is outside postclose window" in result.stdout
    assert "postclose target: pending latest_success_marker outside postclose window" in result.stdout
    assert "post-close composer target: pending latest_success_marker outside postclose window" in result.stdout
    assert "postclose status: success marker found" not in result.stdout


def test_scheduler_health_keeps_prior_postclose_slot_success_when_later_manual_run_appended():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paper_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir = paper_dir / "2026-05-20"
        runtime_dir.mkdir(parents=True)
        (paper_dir / "scheduler_postclose.log").write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-21T06:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto "
                        "schedule_slot=postclose schedule_trigger=launchagent_postclose "
                        "runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] all done",
                    (
                        "[scheduler] run started at 2026-05-21T10:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto"
                    ),
                    "[scheduler] all done",
                    (
                        "[scheduler] run started at 2026-05-21T11:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto"
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-20",
  "profile": "v19reg_v22_budget_usage_bull_guard_v1",
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
        launch_env = root / "soxl.env"
        launch_env.write_text(
            "\n".join(
                [
                    "export SOXL_RUNTIME_COMPOSER='budget_usage'",
                    "export SOXL_RUNTIME_COMPOSER_PROFILE='v19reg_v22_budget_usage_bull_guard_v1'",
                    "export SOXL_RUNTIME_SLOW_PROFILE='balanced_rule_residual_v19_reg'",
                    "export SOXL_RUNTIME_FAST_PROFILE='balanced_rule_residual_v22'",
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_DATE": "2026-05-21",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "1",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: success marker found" in result.stdout
    assert (
        "postclose target: ok runtime_composer=budget_usage "
        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1"
    ) in result.stdout
    assert "post-close composer target: ok runtime_composer=budget_usage" in result.stdout
    assert "latest_success_marker lacks schedule_slot=postclose" not in result.stdout


def test_scheduler_health_accepts_core_done_marker_for_in_run_goal_gate_export():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paper_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir = paper_dir / "2026-05-21"
        runtime_dir.mkdir(parents=True)
        (paper_dir / "scheduler_postclose.log").write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-22T06:15:00+08:00 "
                        "broker=paper dry_run=False skip_data_update=False data_provider=auto "
                        "schedule_slot=postclose schedule_trigger=launchagent_postclose "
                        "runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] core done",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (runtime_dir / "summary.json").write_text(
            """
{
  "date": "2026-05-21",
  "profile": "v19reg_v22_budget_usage_bull_guard_v1",
  "dry_run": false,
  "executed": true,
  "execution_result": {"submitted": true, "status": "filled"},
  "signal": {"target_position": 0.775},
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-21\","
            "\"expected_last_trading_day\":\"2026-05-21\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_DATE": "2026-05-22",
            "SOXL_SCHEDULER_HEALTH_WEEKDAY": "5",
            "SOXL_SCHEDULER_HEALTH_HHMM": "0617",
            "SOXL_SCHEDULER_HEALTH_POSTCLOSE_CUTOFF_EPOCH": "1",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: success marker found" in result.stdout
    assert "post-close composer target: ok runtime_composer=budget_usage" in result.stdout
    assert "ordinary paper proof: ok selected_model=v19reg_v22_budget_usage_bull_guard_v1 execution=filled" in result.stdout
    assert "scheduled_postclose_target" not in result.stdout


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
    assert "US post-close data +" in result.stdout
    assert "simulated trading + dashboard + Vercel deploy attempt" in result.stdout
    assert "US post-close data + paper trading + dashboard + Vercel deploy attempt" not in result.stdout
    assert "post-close: today 06:15 Tue-Sat Asia/Shanghai" not in result.stdout


def test_scheduler_health_reports_preopen_due_without_success_marker():
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
            "SOXL_SCHEDULER_HEALTH_HHMM": "2110",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Pre-open Run Status ---" in result.stdout
    assert "preopen status: due but no success marker found" in result.stdout


def test_scheduler_health_reports_preopen_target_for_current_success_marker():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        preopen_log = runtime_dir / "scheduler_preopen.log"
        preopen_log.write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-21T21:00:00+08:00 "
                        "broker=paper dry_run=True skip_data_update=True data_provider=auto "
                        "schedule_slot=preopen schedule_trigger=launchagent_preopen "
                        "runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] all done",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.utime(preopen_log, (300, 300))
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_HHMM": "2110",
            "SOXL_SCHEDULER_HEALTH_PREOPEN_CUTOFF_EPOCH": "200",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Pre-open Run Status ---" in result.stdout
    assert "preopen status: success marker found" in result.stdout
    assert (
        "preopen target: ok schedule_slot=preopen schedule_trigger=preopen dry_run=True "
        "skip_data_update=True runtime_composer=budget_usage "
        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1"
    ) in result.stdout


def test_scheduler_health_rejects_preopen_marker_outside_schedule_window():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        runtime_dir = root / "runtime" / "soxl" / "paper_trading"
        runtime_dir.mkdir(parents=True)
        preopen_log = runtime_dir / "scheduler_preopen.log"
        preopen_log.write_text(
            "\n".join(
                [
                    (
                        "[scheduler] run started at 2026-05-21T18:00:00+08:00 "
                        "broker=paper dry_run=True skip_data_update=True data_provider=auto "
                        "schedule_slot=preopen schedule_trigger=launchagent_preopen "
                        "runtime_composer=budget_usage "
                        "composer_profile=v19reg_v22_budget_usage_bull_guard_v1 "
                        "slow_profile=balanced_rule_residual_v19_reg "
                        "fast_profile=balanced_rule_residual_v22 bull_guard=True"
                    ),
                    "[scheduler] all done",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        os.utime(preopen_log, (300, 300))
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
            "cat <<'JSON'\n"
            "{\"data\":{\"ok\":true,\"last_date\":\"2026-05-20\","
            "\"expected_last_trading_day\":\"2026-05-20\",\"stale_days\":0},"
            "\"model\":{\"ok\":true,\"name\":\"v19reg_v22_budget_usage_bull_guard_v1 "
            "(slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)\"},"
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
            "SOXL_SCHEDULER_HEALTH_HHMM": "2110",
            "SOXL_SCHEDULER_HEALTH_DATE": "2026-05-21",
            "SOXL_SCHEDULER_HEALTH_PREOPEN_CUTOFF_EPOCH": "200",
            "SOXL_LAUNCH_ENV_FILE": str(launch_env),
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "--- Pre-open Run Status ---" in result.stdout
    assert "preopen status: due but current success marker is outside preopen window" in result.stdout
    assert "preopen target: pending latest_success_marker outside preopen window" in result.stdout
    assert "preopen status: success marker found" not in result.stdout
