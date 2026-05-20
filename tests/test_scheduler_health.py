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
        (runtime_dir / "scheduler_preopen.log").write_text(
            "[scheduler] all done\n",
            encoding="utf-8",
        )
        (runtime_dir / "scheduler_postclose.log").write_text(
            "[scheduler] all done (with cached dry-run)\n",
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
    assert "scheduler_preopen: [scheduler] all done" in result.stdout
    assert "--- Last Completion Marker ---" in result.stdout
    assert "--- Last Successful Run ---" not in result.stdout
    assert "scheduler_postclose: [scheduler] all done (with cached dry-run)" in result.stdout
    assert "--- Paper Trading Health ---" in result.stdout
    assert "data: stale last_date=2026-05-15 expected=2026-05-20 stale_days=5" in result.stdout
    assert "model: ok balanced_rule_residual_v19" in result.stdout
    assert "broker: ok paper broker ready" in result.stdout
    assert "--- Dashboard Static Export ---" in result.stdout
    assert "unavailable: integrity script missing" in result.stdout


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
            "SOXL_SCHEDULER_HEALTH_HHMM": "0600",
        }
        result = subprocess.run(
            ["bash", str(ROOT / "production" / "check_scheduler_health.sh")],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

    assert "postclose status: due but no success marker found" in result.stdout


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
            "SOXL_SCHEDULER_HEALTH_HHMM": "0600",
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
                    "[scheduler] run started at 2026-05-20T05:35:00+08:00",
                    "[scheduler] all done",
                    "[scheduler] run started at 2026-05-21T05:35:00+08:00",
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
            "SOXL_SCHEDULER_HEALTH_HHMM": "0600",
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
                    "[scheduler] run started at 2026-05-21T05:35:04+08:00 broker=paper dry_run=False",
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
            "SOXL_SCHEDULER_HEALTH_HHMM": "0600",
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
