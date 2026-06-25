from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POSITION_CARD = ROOT / "research" / "cognitive_alpha_position_model_v1"
PROVIDER_URI = ROOT / "data" / "qlib" / "soxl"


def test_cognitive_alpha_signal_refresh_fails_closed_when_card_lags_data(tmp_path):
    from src.workflow import CognitiveAlphaSignalRefreshJob, CognitiveAlphaSignalRefreshSpec

    card = tmp_path / "card"
    card.mkdir()
    (card / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "stale_cogalpha",
                "instrument": "SOXL",
                "recorder_id": "recorder-stale",
                "provider_uri": PROVIDER_URI.as_posix(),
                "latest_signal": {
                    "action": "HOLD",
                    "as_of": "2026-01-05",
                    "target_position": 0.25,
                    "previous_target_position": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    job = CognitiveAlphaSignalRefreshJob(
        CognitiveAlphaSignalRefreshSpec(
            card_dir=card,
            provider_uri=PROVIDER_URI,
            output_dir=tmp_path / "signals",
            paper_cash=10000.0,
        )
    )

    with pytest.raises(ValueError, match="stale cognitive alpha signal"):
        job.run([{"source": "qlib.data.D.features", "max_datetime": "2026-01-06"}], [])


def test_cognitive_alpha_signal_refresh_builds_paper_bark_notification_from_latest_card(tmp_path):
    from src.schedule import BarkNotifier, BarkPushConfig
    from src.workflow import CognitiveAlphaSignalRefreshJob, CognitiveAlphaSignalRefreshSpec

    sent: list[tuple[str, dict, float]] = []
    job = CognitiveAlphaSignalRefreshJob(
        CognitiveAlphaSignalRefreshSpec(
            card_dir=POSITION_CARD,
            provider_uri=PROVIDER_URI,
            output_dir=tmp_path / "signals",
            paper_cash=10000.0,
        ),
        notifier=BarkNotifier(
            BarkPushConfig(server_url="https://bark.example", device_key="secret-device", dry_run=False),
            sender=lambda url, payload, timeout: sent.append((url, payload, timeout)) or {"status": 200},
        ),
    )
    manifest = json.loads((PROVIDER_URI / "qlib_store_manifest.json").read_text(encoding="utf-8"))

    notifications = job.run([{"source": "qlib.data.D.features", "max_datetime": manifest["calendar_end"]}], [])

    assert len(notifications) == 1
    notification = notifications[0]
    latest_signal = json.loads((POSITION_CARD / "latest_signal.json").read_text(encoding="utf-8"))
    assert notification["source"] == "qlib_recorder.latest_signal"
    assert notification["symbol"] == "SOXL"
    assert notification["action"] == latest_signal["action"]
    assert notification["as_of"] == latest_signal["as_of"]
    assert notification["target_percent"] == pytest.approx(latest_signal["target_position"])
    assert notification["real_broker_order"] is False
    assert notification["push_result"]["status"] == "sent"
    assert sent[0][0] == "https://bark.example/push"
    assert sent[0][1]["title"] == f"SOXL paper/sim {latest_signal['action']}"
    assert "real_broker_order=False" in sent[0][1]["body"]
    written = json.loads((tmp_path / "signals" / "latest_signal_push.json").read_text(encoding="utf-8"))
    assert written["notification"]["recorder_id"] == notification["recorder_id"]
    assert written["data_gate"]["max_datetime"] == manifest["calendar_end"]
