from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.schedule import stage1_regime_paper as schedule
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec
from src.workflow.stage1_regime_action import RegimeActionSpec


NY = ZoneInfo("America/New_York")
NOW = datetime(2026, 10, 1, 9, 31, 30, tzinfo=NY)


class Action:
    def predict(self, frame):
        return np.full(len(frame), 1.0)


class Classifier:
    def predict_proba(self, frame):
        return np.tile([.1, .9], (len(frame), 1))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    recovery = asdict(IntradayRecoveryDirectAllocationSpec(root=tmp_path)); recovery.pop("root")
    action = asdict(Stage1OpenActionValueSpec(root=tmp_path)); action.pop("root")
    manifest = {"release_id": "paired_test", "not_before": "2026-10-01T00:00:00-04:00",
        "label_available_at_max": "2025-12-31T00:00:00-05:00", "initial_cash": 1000., "cost_bps": [10, 25],
        "regime_action_spec": asdict(RegimeActionSpec(name="test", use_regime=False)),
        "recovery_spec": recovery, "baseline_action_spec": action,
        "selected_recovery_model": {"action_beta": .6, "minimum_source_alpha_probability": .55, "probability_threshold": .55},
        "intraday_provider_uri": "data/qlib/intraday"}
    objects = {"action_model": Action(), "recovery_model": Classifier(), "baseline_action_model": Classifier()}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"release_manifest": "release.json", "expected_manifest_sha256": "a" * 64}))
    monkeypatch.setattr(schedule, "_load_release", lambda *_args: (manifest, objects))
    calls, evidence = [], []

    def quotes(**kwargs):
        calls.append(kwargs)
        return {"SOXL": 10., "SOXX": 20.}, kwargs["now"].isoformat()

    monkeypatch.setattr(schedule, "read_session_quotes", quotes)
    monkeypatch.setattr(schedule, "build_open_inputs", lambda **kwargs: {
        "features": {"soxl_weight": .2, "soxx_weight": .5, "cash_weight": .3, "tail_probability": .1, "alpha_probability": .7},
        "signal_date": "2026-09-30", "previous_session": "2026-09-30", "source": {"fixture": True},
        "feature_available_at": "2026-09-30T16:00:00-04:00"})

    def write_factory(_runtime, _manifest, book):
        def write(event, receipt, state):
            evidence.append({"book": book, "event": event, "state": state})
            return {"recorder_id": f"test_{len(evidence)}"}
        return write

    monkeypatch.setattr(schedule, "_evidence_writer", write_factory)
    return dict(root=tmp_path, config=config, manifest=manifest, calls=calls, evidence=evidence)


def test_off_session_scheduler_does_not_open_configuration_or_quotes(tmp_path):
    result = schedule.run(root=tmp_path, config_path=tmp_path / "absent.json", now=NOW.replace(hour=8))
    assert result["status"] == "waiting" and result["orders_submitted"] == 0


def test_paired_open_uses_one_quote_and_independent_cost_paths(runtime):
    result = schedule.run(root=runtime["root"], config_path=runtime["config"], now=NOW)
    assert result["status"] == "recorded", result
    assert len(runtime["calls"]) == 1 and len(runtime["evidence"]) == 4
    assert set(result["books"]) == {"candidate_10bps", "baseline_10bps", "candidate_25bps", "baseline_25bps"}
    assert len({row["event"]["input_sha256"] for row in runtime["evidence"]}) == 1
    cash = {row["book"]: row["state"]["cash"] for row in runtime["evidence"]}
    assert cash["baseline_10bps"] != cash["baseline_25bps"]
    duplicate = schedule.run(root=runtime["root"], config_path=runtime["config"], now=NOW)
    assert duplicate["status"] == "already_recorded"
    assert len(runtime["calls"]) == 1 and len(runtime["evidence"]) == 4


def test_partial_commit_retries_same_group_without_recapturing_quote(runtime, monkeypatch):
    normal = schedule._evidence_writer
    failed = {"once": False}

    def factory(path, manifest, book):
        if book == "baseline_10bps" and not failed["once"]:
            def fail(*args):
                failed["once"] = True
                raise RuntimeError("test recorder unavailable")
            return fail
        return normal(path, manifest, book)

    monkeypatch.setattr(schedule, "_evidence_writer", factory)
    first = schedule.run(root=runtime["root"], config_path=runtime["config"], now=NOW)
    assert first["status"] == "blocked" and len(runtime["evidence"]) == 1
    second = schedule.run(root=runtime["root"], config_path=runtime["config"], now=NOW.replace(second=40))
    assert second["status"] == "recorded", second
    assert len(runtime["evidence"]) == 4 and len(runtime["calls"]) == 1
    assert len({row["event"]["quote_as_of"] for row in runtime["evidence"]}) == 1


def test_recovery_requires_exact_current_bars_then_reuses_frozen_open_features(runtime, monkeypatch):
    result = schedule.run(root=runtime["root"], config_path=runtime["config"], now=NOW)
    assert result["status"] == "recorded"
    recovery_at = NOW.replace(hour=10)

    def absent(**kwargs):
        raise ValueError("needs_data: SOXL exact first-hour bars are unavailable")

    monkeypatch.setattr(schedule, "read_first_hour_features", absent)
    result = schedule.run(root=runtime["root"], config_path=runtime["config"], now=recovery_at)
    assert result["status"] == "needs_data" and len(runtime["evidence"]) == 4
    fields = {f"{name}_{symbol}": value for symbol in ("soxl", "soxx")
              for name, value in [("early_ret", .01), ("early_range", -.02), ("early_from_low", .01), ("early_volume", 1000.), ("early_bar_count", 104.)]}
    monkeypatch.setattr(schedule, "read_first_hour_features", lambda **kwargs: {
        "features": fields, "feature_available_at": "2026-10-01T10:30:00-04:00", "source": {"fixture": True}})
    result = schedule.run(root=runtime["root"], config_path=runtime["config"], now=recovery_at.replace(second=40))
    assert result["status"] == "recorded", result
    assert len(runtime["evidence"]) == 8
    for record in runtime["evidence"][4:]:
        assert record["event"]["features"]["alpha_probability"] == .7
        assert record["event"]["features"]["early_ret_soxx"] == .01
        assert record["state"]["pending_session"]["recovery_observed"]


def test_first_hour_reads_exact_four_close_stamps_and_rejects_late_substitutes(tmp_path, monkeypatch):
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "qlib_store_manifest.json").write_text("{}")
    monkeypatch.setattr(schedule, "_initialize_provider", lambda *_: None)
    tuples = [(symbol, pd.Timestamp(f"2026-10-01 {minute}")) for symbol in ("SOXL", "SOXX") for minute in schedule.FIRST_HOUR_CLOSES]
    data = pd.DataFrame({"$open": 10., "$high": 12., "$low": 9., "$close": 11., "$volume": 100., "$bar_count": 26.},
                        index=pd.MultiIndex.from_tuples(tuples, names=["instrument", "datetime"])).astype("float32")
    # A real float32 failure mode: Series.sum gives 211019152 here, while the
    # historical groupby aggregation gives 211019168. Preserve trained inputs.
    volumes = np.array([52196672., 56379300., 66721064., 35722124.], dtype="float32")
    data.loc[("SOXL", slice(None)), "$volume"] = volumes
    calls = []
    monkeypatch.setattr("qlib.data.D", SimpleNamespace(features=lambda *args, **kwargs: calls.append(kwargs) or data))
    result = schedule.read_first_hour_features(provider_uri=provider, session="2026-10-01", observed_at=NOW.replace(hour=10))
    expected_volume = data.reset_index().groupby("instrument").agg(total=("$volume", "sum")).loc["SOXL", "total"]
    assert expected_volume != pd.Series(volumes).sum()
    assert result["features"]["early_volume_soxl"] == float(expected_volume) == 211019168.
    assert calls[0]["end_time"] == "2026-10-01 10:30:00"
    data.index = pd.MultiIndex.from_tuples([(symbol, stamp + pd.Timedelta(minutes=15) if stamp.strftime("%H:%M") == "10:30" else stamp)
                                           for symbol, stamp in data.index], names=data.index.names)
    with pytest.raises(ValueError, match="exact first-hour bars"):
        schedule.read_first_hour_features(provider_uri=provider, session="2026-10-01", observed_at=NOW.replace(hour=10))
