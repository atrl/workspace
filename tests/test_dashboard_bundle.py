from __future__ import annotations

import subprocess
import sys

import scripts.build_dashboard_bundle as build_dashboard_bundle


def test_dashboard_bundle_refreshes_retention_gate_before_v2_data(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["build_dashboard_bundle.py"])

    assert build_dashboard_bundle.main() == 0

    assert calls == [
        [sys.executable, "scripts/build_backtest_dashboard.py"],
        [sys.executable, "scripts/review_model_retention_gates.py"],
        [sys.executable, "scripts/build_dashboard_v2_data.py", "--detect-launchd"],
        ["npm", "--prefix", "web/dashboard-v2", "run", "build"],
        [sys.executable, "scripts/check_dashboard_v2_data.py"],
    ]


def test_dashboard_bundle_can_skip_retention_gate_refresh(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["build_dashboard_bundle.py", "--skip-retention-gate"])

    assert build_dashboard_bundle.main() == 0

    assert "scripts/review_model_retention_gates.py" not in [part for cmd in calls for part in cmd]
    assert ["npm", "--prefix", "web/dashboard-v2", "run", "build"] in calls
