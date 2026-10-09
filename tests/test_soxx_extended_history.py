"""Synthetic capture and native single-asset source qualification contracts."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.price_basis import FIELDS
from src.data.published_price_basis import build_published_basis
from src.data.qlib_native import build_qlib_store_from_frame
from src.data.soxx_extended_history import (
    capture_soxx_history, verify_soxx_capture, qualify_soxx_extension,
    build_and_verify_soxx_store, load_parent_evidence,
    run_soxx_history_qualification, _archive_and_publish_completion,
)


SDK = SimpleNamespace(RET_OK=0, KLType=SimpleNamespace(K_DAY="K_DAY"),
    AuType=SimpleNamespace(NONE="NONE"), KL_FIELD=SimpleNamespace(ALL="ALL"),
    Session=SimpleNamespace(RTH="RTH"))


@pytest.fixture
def source(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw = pd.DataFrame({"code": "US.SOXX", "time_key": dates.strftime("%Y-%m-%d"),
        "open": np.arange(10) + 100., "high": np.arange(10) + 102.,
        "low": np.arange(10) + 99., "close": np.arange(10) + 101., "volume": 10000.})
    rehab = pd.DataFrame({"ex_div_date": [str(dates[5].date())],
        "forward_adj_factorA": [.5], "forward_adj_factorB": [.1],
        "split_base": [1.], "split_ert": [2.]})
    config = {"raw_dir": "capture", "host": "127.0.0.1", "port": 11111,
        "code": "US.SOXX", "instrument": "SOXX", "ktype": "K_DAY", "autype": "NONE",
        "fields": "KL_FIELD.ALL", "session": "RTH", "extended_time": False, "max_count": 1000,
        "max_pages": 10, "requested_start": str(dates[0].date()), "requested_end": str(dates[-1].date()),
        "as_of": str(dates[-1].date()), "expected_rows": 10, "expected_early_rows": 4,
        "sdk_logs_dir": str(tmp_path / "logs")}
    calls = []
    class Quote:
        def __init__(self, **kwargs):
            self.page = 0
        def get_rehab(self, **kwargs):
            calls.append(("get_rehab", kwargs))
            return 0, rehab.copy(deep=True)
        def request_history_kline(self, **kwargs):
            calls.append(("history", kwargs))
            self.page += 1
            if self.page == 1:
                assert kwargs["page_req_key"] is None
                return 0, raw.iloc[:6].copy(), b"synthetic-page-two"
            assert kwargs["page_req_key"] == b"synthetic-page-two"
            return 0, raw.iloc[6:].copy(), None
        def close(self):
            calls.append(("close", {}))
    plan = {"capture": config, "parent": {"overlap_start": str(dates[4].date()), "overlap_rows": 6}}
    manifest = capture_soxx_history(tmp_path, plan, quote_context_factory=Quote, sdk=SDK)
    return tmp_path, plan, dates, raw, rehab, manifest, calls


def _verified(source):
    root, plan, dates, *_ = source
    return verify_soxx_capture(root / "capture", plan["capture"], dates)


def _rewrite_response(source, index, mutate):
    root, _, _, _, _, _, _ = source
    path = root / "capture/manifest.json"
    manifest = json.loads(path.read_text())
    receipt = manifest["raw_responses"][index]
    response_path = path.parent / receipt["response_path"]
    response = pickle.loads(response_path.read_bytes())
    response = mutate(response)
    response_path.write_bytes(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
    receipt["response_sha256"] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    receipt["rows"] = len(response[1])
    path.write_text(json.dumps(manifest))


def _parent(source, capture):
    _, plan, dates, raw, rehab, *_ = source
    secondary = pd.DataFrame(columns=["instrument", "date", *FIELDS])
    secondary_id = {"source": "Yahoo", "adjustment": "SPLIT_ONLY", "volume_unit": "shares",
                    "volume_basis": "split_adjusted_shares"}
    frame, _ = build_published_basis(raw, secondary, {"SOXX": rehab, "SOXL": rehab},
                                     capture["identity"], secondary_id, plan["capture"]["as_of"])
    return {"raw_none": raw.iloc[4:].copy(), "qualified": frame.iloc[4:].copy(),
            "rehab": {"SOXX": rehab.copy(), "SOXL": rehab.copy()}}, frame


def test_complete_paginated_capture_has_bracketing_receipts_and_no_hidden_requests(source):
    _, _, _, raw, _, manifest, calls = source
    capture = _verified(source)
    assert manifest["status"] == "captured_unqualified"
    assert [call[0] for call in calls] == ["get_rehab", "history", "history", "get_rehab", "close"]
    assert capture["audit"]["rows"] == 10 and capture["audit"]["pages"] == 2
    pd.testing.assert_frame_equal(capture["frame"], raw, check_exact=True)
    assert capture["audit"]["missing_rows_filled"] == capture["audit"]["invalid_rows_dropped"] == 0


@pytest.mark.parametrize("case", ["missing_row", "duplicate_row", "wrong_symbol", "nan_price", "bad_ohlc", "future_date", "rehab_change", "continuation"])
def test_invalid_original_observations_and_rehab_cannot_be_repaired_silently(source, case):
    def changed(response):
        ret, frame, *tail = response
        frame = frame.copy()
        if case == "missing_row": frame = frame.iloc[1:]
        elif case == "duplicate_row": frame.iloc[1] = frame.iloc[0]
        elif case == "wrong_symbol": frame.loc[frame.index[0], "code"] = "US.SOXL"
        elif case == "nan_price": frame.loc[frame.index[0], "open"] = np.nan
        elif case == "bad_ohlc": frame.loc[frame.index[0], "low"] = 10000.
        elif case == "future_date": frame.loc[frame.index[-1], "time_key"] = "2030-01-01"
        elif case == "rehab_change": frame.loc[0, "forward_adj_factorA"] = .6
        elif case == "continuation": tail = [None]
        return (ret, frame, *tail)
    _rewrite_response(source, -1 if case == "rehab_change" else 1, changed)
    with pytest.raises((ValueError, AssertionError)):
        _verified(source)


@pytest.mark.parametrize("case", ["sha", "wrong_args", "wrong_time", "reordered"])
def test_receipt_identity_and_chronology_are_binding(source, case):
    root, *_ = source
    path = root / "capture/manifest.json"
    manifest = json.loads(path.read_text())
    if case == "sha": manifest["raw_responses"][1]["response_sha256"] = "0" * 64
    elif case == "wrong_args": manifest["raw_responses"][1]["params"]["autype"] = "QFQ"
    elif case == "wrong_time": manifest["raw_responses"][1]["requested_at_utc"] = "2000-01-01T00:00:00Z"
    else: manifest["raw_responses"][1:3] = manifest["raw_responses"][1:3][::-1]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="blocked:"):
        _verified(source)


def test_failed_capture_is_retained_and_never_qualifies(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=3)
    config = {"raw_dir": "failed", "host": "127.0.0.1", "port": 11111,
        "code": "US.SOXX", "instrument": "SOXX", "ktype": "K_DAY", "autype": "NONE",
        "fields": "KL_FIELD.ALL", "session": "RTH", "extended_time": False, "max_count": 1000,
        "max_pages": 10, "requested_start": "2024-01-02", "requested_end": "2024-01-04",
        "as_of": "2024-01-04", "expected_rows": 3, "sdk_logs_dir": str(tmp_path / "logs")}
    class Refused:
        def __init__(self, **kwargs): pass
        def get_rehab(self, **kwargs): return -1, "permission required"
        def close(self): pass
    manifest = capture_soxx_history(tmp_path, {"capture": config}, quote_context_factory=Refused, sdk=SDK)
    assert manifest["status"] == "needs auth" and len(manifest["raw_responses"]) == 1
    assert (tmp_path / "failed/SOXX_rehab_before.pkl").exists()
    with pytest.raises(ValueError, match="failed"):
        verify_soxx_capture(tmp_path / "failed", config, dates)


def test_published_formula_preserves_overlap_and_single_native_store_is_immutable(source):
    import qlib
    from qlib.data import D
    root, plan, dates, *_ = source
    capture = _verified(source)
    parent, expected = _parent(source, capture)
    frame, qualification = qualify_soxx_extension(capture, parent, dates, plan)
    assert qualification["rows"] == 10 and qualification["early_rows"] == 4
    assert qualification["parent_overlap_rows"] == 6 and qualification["secondary_rows"] == 0
    np.testing.assert_array_equal(frame[list(FIELDS)], expected[list(FIELDS)])
    # Build a genuinely native parent subset, then read it through Qlib.
    old_provider = root / "parent_provider"
    build_qlib_store_from_frame(parent["qualified"], provider_uri=old_provider,
        as_of=plan["capture"]["as_of"], field_map={field: field for field in FIELDS})
    qlib.init(provider_uri=str(old_provider), region="us", kernels=1)
    parent_native = D.features(["SOXX"], ["$" + field for field in FIELDS], freq="day")
    destination = root / "extended_provider"
    kwargs = {"provider_uri": destination, "as_of": plan["capture"]["as_of"], "calendar": dates,
              "parent_native": parent_native, "overlap_start": plan["parent"]["overlap_start"], "qualification": qualification}
    first = build_and_verify_soxx_store(frame, **kwargs)
    second = build_and_verify_soxx_store(frame, **kwargs)
    assert first["parent_native_overlap_bitwise_equal"] is True
    assert first["expected_float32_bitwise_equal"] is True
    assert first["provider_file_sha256"] == second["provider_file_sha256"]
    assert second["immutable_provider_reused"] is True
    assert D.list_instruments(D.instruments("all"), as_list=True) == ["SOXX"]
    altered = frame.copy(); altered.loc[0, "close"] += .1
    with pytest.raises(ValueError, match="qualification differs"):
        build_and_verify_soxx_store(altered, **kwargs)


@pytest.mark.parametrize("case", ["raw", "published", "formula"])
def test_parent_overlap_and_rehab_formula_drift_fail_closed(source, case):
    _, plan, dates, *_ = source
    capture = _verified(source)
    parent, _ = _parent(source, capture)
    if case == "raw": parent["raw_none"].iloc[0, parent["raw_none"].columns.get_loc("open")] += .01
    elif case == "published": parent["qualified"].iloc[0, parent["qualified"].columns.get_loc("volume")] += 1.
    else: parent["rehab"]["SOXX"].loc[0, "forward_adj_factorB"] += .01
    with pytest.raises((ValueError, AssertionError)):
        qualify_soxx_extension(capture, parent, dates, plan)


def test_capture_destination_cannot_be_overwritten(source):
    root, plan, *_ = source
    with pytest.raises(ValueError, match="already exists"):
        capture_soxx_history(root, plan, quote_context_factory=lambda **kwargs: None, sdk=SDK)


def test_legacy_parent_hash_is_checked_before_any_qlib_object_is_loaded(tmp_path, monkeypatch):
    root = tmp_path
    card = root / "parent"
    local = card / "qlib_recorder/1/recorder"
    artifact = local / "artifacts/inputs/evidence.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"untrusted pickle must never be opened")
    summary = {"recorder_id": "recorder", "recorder_uri": str(card / "qlib_recorder"),
        "recorder_local_dir": str(local), "data_as_of": "2024-01-01", "provider_uri": "missing",
        "source_qualification_pass": True}
    (card / "summary.json").write_text(json.dumps(summary))
    (card / "artifact_manifest.json").write_text(json.dumps({"recorder_id": "recorder", "entries": [
        {"path": artifact.relative_to(root).as_posix(), "sha256": "0" * 64, "bytes": artifact.stat().st_size}]}))
    plan = {"parent": {"summary": "parent/summary.json", "recorder_id": "recorder", "experiment_id": "1",
        "recorder_uri": "parent/qlib_recorder", "as_of": "2024-01-01", "provider_uri": "missing",
        "artifact_sha256": {"inputs/evidence.pkl": "0" * 64}}}
    with pytest.raises(ValueError, match="SHA differs"):
        load_parent_evidence(root, plan)
    assert not (card / "verification_manifest.json").exists()


def test_completed_card_guard_precedes_source_reads_or_initialization(tmp_path, monkeypatch):
    from src.data import soxx_extended_history as module
    card = tmp_path / "completed"
    card.mkdir()
    original = b'{"recorder_id":"existing-completed-run"}\n'
    (card / "summary.json").write_bytes(original)
    monkeypatch.setattr(module, "load_parent_evidence", lambda *args: pytest.fail("completed card read its sources"))
    with pytest.raises(ValueError, match="completed card"):
        run_soxx_history_qualification(tmp_path, card)
    assert (card / "summary.json").read_bytes() == original


def test_active_recorder_is_rejected_before_any_qlib_init(tmp_path, monkeypatch):
    import qlib
    from qlib.workflow import R
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    qlib.init(provider_uri=str(tmp_path / "unused_provider"), region="us", kernels=1)
    with R.start(experiment_name="synthetic_active_guard", uri=str(tmp_path / "recorder")):
        recorder_id = R.get_recorder().id
        monkeypatch.setattr(qlib, "init", lambda *args, **kwargs: pytest.fail("active Recorder was reset by init"))
        with pytest.raises(ValueError, match="before qlib.init"):
            load_parent_evidence(tmp_path, {})
        assert R.get_recorder().id == recorder_id


@pytest.mark.parametrize("failure", ["artifact", "context_exit", None])
def test_summary_is_atomically_published_only_after_archival_and_context_success(tmp_path, failure):
    state, archived = {"active": False}, {}
    (tmp_path / "small.json").write_text('{"verified":true}')
    summary = {"recorder_id": "synthetic", "rows": 10, "new_early_rows": 4}
    class Recorder:
        id = "synthetic"
        def save_objects(self, **kwargs):
            assert state["active"] and not (tmp_path / "summary.json").exists()
        def log_metrics(self, **kwargs): pass
        def log_artifact(self, source, **kwargs):
            assert state["active"] and not (tmp_path / "summary.json").exists()
            if failure == "artifact":
                raise RuntimeError("synthetic archive failure")
            archived[Path(source).name] = Path(source).read_bytes()
    @contextmanager
    def context():
        state["active"] = True
        try:
            yield
        finally:
            state["active"] = False
        if failure == "context_exit":
            raise RuntimeError("synthetic Recorder completion failure")
    def execute():
        _archive_and_publish_completion(tmp_path, summary, recorder_context=context(),
            recorder_loader=Recorder, data_objects={"audit.pkl": {}}, artifact_names=["small.json"])
    if failure:
        with pytest.raises(RuntimeError):
            execute()
        assert not (tmp_path / "summary.json").exists()
    else:
        execute()
        assert not state["active"]
        assert (tmp_path / "summary.json").read_bytes() == archived["summary.json"]
        assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert not list(tmp_path.glob(".summary-*.tmp"))
