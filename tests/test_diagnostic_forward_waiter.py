"""Only the new one-shot supervision contract; no trading environment imports."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest


def test_wait_then_one_attempt_or_missed_window(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "src/schedule/diagnostic_forward_waiter.py"
    spec = importlib.util.spec_from_file_location("waiter_contract", path)
    waiter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(waiter)
    start = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
    latest, stop = start + timedelta(minutes=1), start + timedelta(hours=26)
    permit = {"registration_sha256": "a" * 64}
    monkeypatch.setattr(waiter, "verified_contract", lambda *a: ({}, permit, start, latest, stop))
    launches = []

    class Child:
        pid = 123
        returncode = 0
        def poll(self):
            return self.returncode

    def launch(*args, **kwargs):
        launches.append(args[0])
        return Child()

    monkeypatch.setattr(waiter.subprocess, "Popen", launch)

    def card_for(name):
        root = tmp_path / name
        card = root / "research/card"
        card.mkdir(parents=True)
        (card / "supervision_approval.json").write_text("{}")
        return root, card

    # Start before the window: only the standard-library wait may run.
    root, card = card_for("waiting")
    times = [start - timedelta(seconds=30)]
    monkeypatch.setattr(waiter, "now", lambda: times[0])
    def wake_after_wait(seconds):
        assert not launches
        assert seconds <= 30
        times[0] = start + timedelta(seconds=1)
    monkeypatch.setattr(waiter.time, "sleep", wake_after_wait)
    result = waiter.supervise(root, card)
    assert result["phase"] == "child_finished" and result["child_attempts"] == 1
    assert len(launches) == 1 and "run_pilot" in launches[0][3]
    assert result["pilot_started"] and not result["claims_strategy_qualification"]
    with pytest.raises(FileExistsError):
        waiter.supervise(root, card)
    assert len(launches) == 1  # A second invocation cannot silently retry.

    # A sleeping/killed host that misses startup must not catch up later.
    root, card = card_for("missed")
    times[0] = latest + timedelta(seconds=1)
    result = waiter.supervise(root, card)
    assert result["phase"] == "missed_start_window" and not result["pilot_started"]
    assert len(launches) == 1
    state = json.loads(next(root.glob("runtime/research/*/supervisor_*/status.json")).read_text())
    assert state["child_attempts"] == 0
