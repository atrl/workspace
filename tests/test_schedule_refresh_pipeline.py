from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data import QlibProviderConfig, build_qlib_store_from_csv


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_URI = ROOT / "data" / "qlib" / "soxl"


def _write_raw(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_qlib_data_refresh_job_runs_store_refresh_before_auditing(tmp_path):
    from src.schedule import QlibDataRefreshJob, QlibDataRefreshSpec

    raw = tmp_path / "raw.csv"
    provider_uri = tmp_path / "qlib_provider"
    _write_raw(
        raw,
        [
            {"instrument": "SOXL", "datetime": "2026-01-02", "close": 10.0, "vix": 18.0, "volume": 1000},
            {"instrument": "SOXL", "datetime": "2026-01-05", "close": 11.0, "vix": 19.0, "volume": 1200},
            {"instrument": "SOXL", "datetime": "2026-01-06", "close": 12.0, "vix": 17.5, "volume": 900},
        ],
    )

    def refresh_store():
        return build_qlib_store_from_csv(
            raw,
            provider_uri=provider_uri,
            as_of="2026-01-06",
            field_map={"close": "close", "vix": "vix", "volume": "volume"},
        )

    job = QlibDataRefreshJob(
        QlibDataRefreshSpec(
            config=QlibProviderConfig(provider_uri=provider_uri, region="us", market="all"),
            instruments=["SOXL"],
            fields=["$close", "$vix", "$volume"],
            start="2026-01-02",
            end="2026-01-06",
            as_of="2026-01-06",
            required_max_datetime="2026-01-06",
            output_dir=tmp_path / "manifests",
        ),
        refresh_store=refresh_store,
    )

    manifests = job.run()
    manifest = manifests[0]
    written = json.loads((tmp_path / "manifests" / "snapshot_manifest.json").read_text(encoding="utf-8"))

    assert manifest["operation"] == "qlib_data_update"
    assert manifest["source"] == "qlib.data.D.features"
    assert manifest["status"] == "ok"
    assert manifest["max_datetime"] == "2026-01-06"
    assert manifest["required_max_datetime"] == "2026-01-06"
    assert manifest["refresh"]["operation"] == "qlib_store_build"
    assert manifest["manifest_path"].endswith("snapshot_manifest.json")
    assert written["snapshot_hash"] == manifest["snapshot_hash"]


def test_qlib_data_refresh_job_fails_closed_when_data_stays_stale(tmp_path):
    from src.data import QlibUnavailableError
    from src.schedule import QlibDataRefreshJob, QlibDataRefreshSpec

    raw = tmp_path / "raw.csv"
    provider_uri = tmp_path / "qlib_provider"
    _write_raw(
        raw,
        [
            {"instrument": "SOXL", "datetime": "2026-01-02", "close": 10.0, "vix": 18.0, "volume": 1000},
            {"instrument": "SOXL", "datetime": "2026-01-05", "close": 11.0, "vix": 19.0, "volume": 1200},
        ],
    )
    build_qlib_store_from_csv(
        raw,
        provider_uri=provider_uri,
        as_of="2026-01-05",
        field_map={"close": "close", "vix": "vix", "volume": "volume"},
    )

    job = QlibDataRefreshJob(
        QlibDataRefreshSpec(
            config=QlibProviderConfig(provider_uri=provider_uri, region="us", market="all"),
            instruments=["SOXL"],
            fields=["$close", "$vix"],
            start="2026-01-02",
            end="2026-01-08",
            as_of="2026-01-08",
            required_max_datetime="2026-01-08",
            output_dir=tmp_path / "manifests",
        )
    )

    with pytest.raises(QlibUnavailableError, match="stale qlib data"):
        job.run()


def test_command_data_refresh_runs_configured_command_without_shell(tmp_path):
    from src.schedule import CommandDataRefresh

    marker = tmp_path / "marker.txt"
    script = tmp_path / "refresh.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('updated', encoding='utf-8')\n"
        "print('refresh complete')\n",
        encoding="utf-8",
    )

    result = CommandDataRefresh([sys.executable, script.as_posix()], cwd=tmp_path)()

    assert marker.read_text(encoding="utf-8") == "updated"
    assert result["operation"] == "command_data_refresh"
    assert result["returncode"] == 0
    assert result["command"][0] == sys.executable
    assert "refresh complete" in result["stdout_tail"]


def test_walk_forward_model_refresh_uses_qlib_calendar_and_calls_workflow(tmp_path):
    from src.schedule import WalkForwardModelRefreshJob, WalkForwardModelRefreshSpec

    calls = []

    class Result:
        recorder_id = "recorder-fold"
        recorder_local_dir = tmp_path / "recorder"
        summary = {"artifact_policy": {"policy": "slim_card_recorder_first"}}
        analysis_summary = {"source": "qlib_recorder"}

    def runner(spec):
        calls.append(spec)
        return Result()

    job = WalkForwardModelRefreshJob(
        WalkForwardModelRefreshSpec(
            experiment_prefix="scheduled_soxl",
            provider_uri=PROVIDER_URI,
            recorder_uri=tmp_path / "qlib_recorder",
            output_root=tmp_path / "cards",
            market="all",
            region="us",
            instrument="SOXL",
            start="2026-01-02",
            end="2026-01-15",
            train_size=4,
            valid_size=1,
            test_size=2,
            step_size=2,
            max_folds=2,
            fixed_model_name="linear_ic",
            promoted_from="research/qlib_validated_signal_weight_v1",
        ),
        runner=runner,
    )

    results = job.run([{"source": "qlib.data.D.features", "max_datetime": "2026-01-15"}])

    assert len(calls) == 2
    assert results[0]["source"] == "qlib_recorder"
    assert results[0]["operation"] == "walk_forward_model_refresh"
    assert results[0]["segments"]["train"] == ["2026-01-02", "2026-01-07"]
    assert results[0]["segments"]["valid"] == ["2026-01-08", "2026-01-08"]
    assert results[0]["segments"]["test"] == ["2026-01-09", "2026-01-12"]
    assert calls[0].fixed_model_name == "linear_ic"
    assert calls[0].promoted_from == "research/qlib_validated_signal_weight_v1"
    assert calls[0].output_dir == tmp_path / "cards" / "scheduled_soxl_wf001_20260112"


def test_bark_notifier_builds_paper_trade_push_without_exposing_key():
    from src.api import OrderIntent
    from src.schedule import BarkNotifier, BarkPushConfig, trade_notification_from_order_intents

    sent = []
    current = OrderIntent(
        symbol="SOXL",
        target_percent=0.95,
        quantity=47,
        price=201.68,
        as_of="2026-06-09",
        recorder_id="abc123",
    )
    previous = OrderIntent(
        symbol="SOXL",
        target_percent=0.25,
        quantity=12,
        price=198.0,
        as_of="2026-06-08",
        recorder_id="old123",
    )
    notification = trade_notification_from_order_intents(current=current, previous=previous)
    notifier = BarkNotifier(
        BarkPushConfig(server_url="https://bark.example", device_key="secret-device", dry_run=False),
        sender=lambda url, payload, timeout: sent.append((url, payload, timeout))
        or {"status": 200, "body": '{"code":200,"message":"success"}'},
    )

    result = notifier.send_trade_notification(notification)

    assert notification.action == "BUY"
    assert notification.quantity_delta == 35
    assert result["status"] == "sent"
    assert result["response_message"] == "success"
    assert sent[0][0] == "https://bark.example/push"
    assert sent[0][1]["device_key"] == "secret-device"
    assert sent[0][1]["title"] == "SOXL paper/sim BUY"
    assert "target 95.00%" in sent[0][1]["body"]
    assert "recorder abc123" in sent[0][1]["body"]
    assert "secret-device" not in sent[0][1]["body"]


def test_bark_push_config_requires_env_key_and_server():
    from src.schedule import BarkPushConfig

    config = BarkPushConfig.from_env(
        {"BARK_SERVER_URL": "https://bark.example", "BARK_DEVICE_KEY": "device-from-env"},
        dry_run=False,
    )

    assert config.server_url == "https://bark.example"
    assert config.device_key == "device-from-env"
    assert config.dry_run is False
    with pytest.raises(ValueError, match="BARK_SERVER_URL and BARK_DEVICE_KEY"):
        BarkPushConfig.from_env({})


def test_bark_push_config_accepts_persistent_remote_env_aliases():
    from src.schedule import BarkPushConfig

    config = BarkPushConfig.from_env(
        {
            "BARK_REMOTE_BASE_URL": "https://bark.example",
            "BARK_REMOTE_DEVICE_KEY": "persistent-device",
        },
        dry_run=False,
    )

    assert config.server_url == "https://bark.example"
    assert config.device_key == "persistent-device"
    assert config.dry_run is False


def test_refresh_plan_writes_state_and_passes_trade_notifications(tmp_path):
    from src.schedule import RefreshPlan, run_refresh_plan

    events = []

    def data_refresh():
        events.append("data")
        return [{"source": "qlib.data.D.features", "max_datetime": "2026-01-15"}]

    def model_refresh(manifests):
        events.append("model")
        return [{"source": "qlib_recorder", "recorder_id": "abc123", "manifests": len(manifests)}]

    def signal_refresh(manifests, model_results):
        events.append("signals")
        return [{"action": "BUY", "symbol": "SOXL", "recorder_id": model_results[0]["recorder_id"]}]

    def dashboard_export(manifests, model_results):
        events.append("dashboard")
        return {"source": "qlib_recorder", "artifacts": len(manifests), "models": len(model_results)}

    def notify(summary):
        events.append(f"notify:{summary['trade_notifications']}")
        return {"status": "dry_run"}

    summary = run_refresh_plan(
        RefreshPlan(
            name="scheduled-paper-sim",
            lock_path=tmp_path / "refresh.lock",
            state_path=tmp_path / "last_run.json",
        ),
        data_refresh=data_refresh,
        model_refresh=model_refresh,
        signal_refresh=signal_refresh,
        dashboard_export=dashboard_export,
        notify=notify,
    )

    state = json.loads((tmp_path / "last_run.json").read_text(encoding="utf-8"))
    assert events == ["data", "model", "signals", "dashboard", "notify:1"]
    assert summary["trade_notifications"] == 1
    assert summary["notification_result"] == {"status": "dry_run"}
    assert state["status"] == "ok"
    assert state["trade_notification_details"][0]["action"] == "BUY"
