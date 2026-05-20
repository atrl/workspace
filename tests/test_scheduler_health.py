from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scheduler_health_reports_macos_launchagent_and_logs(tmp_path: Path):
    runtime_dir = tmp_path / "runtime" / "soxl" / "paper_trading"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "scheduler_preopen.log").write_text(
        "[scheduler] all done\n",
        encoding="utf-8",
    )
    (runtime_dir / "scheduler_postclose.log").write_text(
        "[scheduler] all done (with cached data)\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl = fake_bin / "launchctl"
    launchctl.write_text(
        "#!/bin/sh\nprintf '123\\t0\\tcom.soxl.preopen.scheduler\\n456\\t0\\tcom.soxl.daily.scheduler\\n'\n",
        encoding="utf-8",
    )
    launchctl.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SOXL_SCHEDULER_HEALTH_OS": "Darwin",
        "SOXL_SCHEDULER_HEALTH_ROOT": str(tmp_path),
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
    assert "scheduler_postclose: [scheduler] all done (with cached data)" in result.stdout
