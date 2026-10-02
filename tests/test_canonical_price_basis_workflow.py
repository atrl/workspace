import copy
from contextlib import contextmanager
import json
import pickle
from pathlib import Path
import runpy
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest

from src.data.price_basis import PriceBasisError, build_qualified_price_store, normalize_futu_candles, validate_price_basis
from src.workflow import canonical_price_basis as workflow


@pytest.fixture
def sdk_snapshot(tmp_path):
    folder = tmp_path / "sdk"
    folder.mkdir()
    start, end = "2011-01-03", "2011-01-05"
    dates = ["2011-01-03", "2011-01-04", "2011-01-05"]
    receipts = []
    rehab = pd.DataFrame({"ex_div_date": ["2010-06-01"], "forward_adj_factorA": [1.], "forward_adj_factorB": [0.]})

    def add(name, response, params, second):
        path = folder / f"{name}.pkl"
        path.write_bytes(pickle.dumps(response, protocol=4))
        instant = pd.Timestamp("2011-01-06T00:00:00Z") + pd.Timedelta(seconds=second)
        receipts.append({"name": name, "params": params, "ret": 0, "rows": len(response[1]),
            "response_path": str(path), "response_sha256": workflow.sha(path),
            "requested_at_utc": instant.isoformat(), "received_at_utc": (instant + pd.Timedelta(seconds=1)).isoformat()})

    for symbol_index, symbol in enumerate(("SOXL", "SOXX")):
        offset = symbol_index * 20
        add(f"{symbol}_rehab_before_attempt1", (0, rehab), {"code": "US." + symbol}, offset)
        for number, date in enumerate(dates, start=1):
            data = pd.DataFrame({"code": ["US." + symbol], "time_key": [date + " 00:00:00"],
                                 "open": [10.], "high": [11.], "low": [9.], "close": [10.5], "volume": [1000.]})
            params = {"code": "US." + symbol, "start": start, "end": end, "ktype": "K_DAY", "autype": "None",
                      "max_count": 1000, "extended_time": False, "session": "RTH", "page_number": number,
                      "has_prior_page_key": number > 1}
            add(f"{symbol}_history_page{number}_attempt1", (0, data, b"next" if number < 3 else None), params, offset + 2 * number)
        add(f"{symbol}_rehab_after_attempt1", (0, rehab), {"code": "US." + symbol}, offset + 8)
    manifest = {"requested_start": start, "requested_end": end, "adjustment": "NONE", "session": "RTH",
                "extended_time": False, "read_only": True, "sdk_version": "test", "raw_responses": receipts}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder, manifest, {"adjustment": "NONE", "start": start, "as_of": end}


def _rewrite_snapshot(folder, manifest):
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_original_sdk_pages_and_rehabilitation_bracketing_verify(sdk_snapshot):
    directory, _, kwargs = sdk_snapshot
    result = workflow.verify_futu_snapshot(directory, **kwargs)
    assert len(result["frame"]) == 6
    assert set(result["frame"].code) == {"US.SOXL", "US.SOXX"}
    assert result["identity"]["adjustment"] == "NONE"


@pytest.mark.parametrize("mode", ["hash", "receipt_ret", "original_ret", "future_date", "continuation", "duplicate_name"])
def test_sdk_bad_status_lineage_or_pagination_is_not_accepted(sdk_snapshot, mode):
    directory, manifest, kwargs = sdk_snapshot
    receipt = manifest["raw_responses"][1]
    path = Path(receipt["response_path"])
    if mode == "hash":
        path.write_bytes(path.read_bytes() + b"changed")
    elif mode == "receipt_ret":
        receipt["ret"] = -1
    elif mode == "duplicate_name":
        manifest["raw_responses"].append(copy.deepcopy(manifest["raw_responses"][0]))
    else:
        response = list(pickle.loads(path.read_bytes()))
        if mode == "original_ret": response[0] = -1
        elif mode == "future_date": response[1].loc[0, "time_key"] = "2011-01-06 00:00:00"
        else: response[2] = None
        path.write_bytes(pickle.dumps(tuple(response), protocol=4))
        receipt["response_sha256"] = workflow.sha(path)
    _rewrite_snapshot(directory, manifest)
    with pytest.raises(PriceBasisError):
        workflow.verify_futu_snapshot(directory, **kwargs)


@pytest.mark.parametrize("parameter,value", [("autype", "qfq"), ("session", "ALL"), ("extended_time", True),
                                              ("ktype", "K_1M"), ("end", "2011-01-06")])
def test_each_sdk_page_must_match_the_header_contract(sdk_snapshot, parameter, value):
    directory, manifest, kwargs = sdk_snapshot
    manifest["raw_responses"][2]["params"][parameter] = value
    _rewrite_snapshot(directory, manifest)
    with pytest.raises(PriceBasisError):
        workflow.verify_futu_snapshot(directory, **kwargs)


def test_every_page_not_only_endpoints_must_be_inside_rehabilitation_bracket(sdk_snapshot):
    directory, manifest, kwargs = sdk_snapshot
    receipt = manifest["raw_responses"][2]  # Intermediate page, after the stated after-rehab snapshot.
    receipt["requested_at_utc"] = "2011-01-06T01:00:00Z"
    receipt["received_at_utc"] = "2011-01-06T01:00:01Z"
    _rewrite_snapshot(directory, manifest)
    with pytest.raises(PriceBasisError):
        workflow.verify_futu_snapshot(directory, **kwargs)


@pytest.fixture
def yahoo_snapshot(tmp_path):
    folder = tmp_path / "yahoo"
    folder.mkdir()
    dates = pd.to_datetime(["2011-01-03", "2011-01-04", "2011-01-05"])
    timestamps = (dates.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
    payload = {"chart": {"error": None, "result": [{"meta": {"symbol": "SOXL", "exchangeTimezoneName": "America/New_York",
        "currency": "USD", "regularMarketPrice": 999999.}, "timestamp": [int(t.timestamp()) for t in timestamps],
        "indicators": {"quote": [{"open": [10., 11., 12.], "high": [11., 12., 13.], "low": [9., 10., 11.],
                                    "close": [10.5, 11.5, 12.5], "volume": [1000., 1100., 1200.]}],
                       "adjclose": [{"adjclose": [-999., -999., -999.]}]}}]}}
    (folder / "original_response.json").write_text(json.dumps(payload))
    query = urlencode({"period1": int(pd.Timestamp("2011-01-03T00:00:00Z").timestamp()),
                       "period2": int(pd.Timestamp("2011-01-06T00:00:00Z").timestamp()), "interval": "1d"})
    receipt = {"url": "https://query2.finance.yahoo.com/v8/finance/chart/SOXL?" + query,
               "requested_at_utc": "2011-01-06T01:00:00Z", "received_at_utc": "2011-01-06T01:00:01Z",
               "status_code": 200, "sha256": workflow.sha(folder / "original_response.json")}
    (folder / "manifest.json").write_text(json.dumps(receipt))
    primary = pd.DataFrame([{"code": "US." + symbol, "time_key": str(date.date()) + " 00:00:00"}
                            for symbol in ("SOXL", "SOXX") for date in dates
                            if symbol == "SOXX" or date == dates[1]])
    kwargs = {"primary": primary, "calendar": {"dates": [str(d.date()) for d in dates]}, "as_of": "2011-01-05",
              "rehabilitation": {}, "expected_secondary_dates": ["2011-01-03", "2011-01-05"]}
    return folder, payload, receipt, kwargs


def test_yahoo_only_fills_registered_whole_bar_gaps_and_ignores_current_metadata(yahoo_snapshot):
    directory, _, _, kwargs = yahoo_snapshot
    result = workflow.yahoo_missing_bars(directory, **kwargs)
    assert result["frame"].open.tolist() == [10., 12.]
    assert result["frame"].close.tolist() == [10.5, 12.5]
    assert result["frame"].original_payload_ordinal.tolist() == [0, 2]


@pytest.mark.parametrize("mode", ["hash", "status", "timestamp_order", "wrong_host", "request_bounds", "different_gaps"])
def test_yahoo_receipt_and_registered_gap_contract_fail_closed(yahoo_snapshot, mode):
    directory, _, receipt, kwargs = yahoo_snapshot
    if mode == "hash": receipt["sha256"] = "0" * 64
    elif mode == "status": receipt["status_code"] = 500
    elif mode == "timestamp_order": receipt["received_at_utc"] = "2011-01-06T00:00:59Z"
    elif mode == "wrong_host": receipt["url"] = receipt["url"].replace("query2.finance.yahoo.com", "not-yahoo.invalid")
    elif mode == "request_bounds": receipt["url"] = receipt["url"].replace("period1=1294012800", "period1=1294272000")
    else: kwargs["expected_secondary_dates"] = ["2011-01-03"]
    (directory / "manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(PriceBasisError):
        workflow.yahoo_missing_bars(directory, **kwargs)


def test_existing_provider_cannot_be_accepted_with_failed_status(tmp_path):
    fixture = runpy.run_path(str(Path(__file__).with_name("test_price_basis.py")))["source_fixture"].__wrapped__(tmp_path)
    raw, identity, calendar = fixture
    frame = normalize_futu_candles(raw, identity)
    qualification = validate_price_basis(frame, calendar, "2011-01-04")
    provider = tmp_path / "qualified"
    manifest = {"calendar": calendar, "qualification": qualification, "source_identities": frame.attrs["source_identities"]}
    build_qualified_price_store(frame, provider, "2011-01-04", manifest)
    (provider / "price_basis_status.json").write_text(json.dumps({"status": "failed", "error": "interrupted audit fixture"}))
    with pytest.raises(PriceBasisError):
        workflow._existing_snapshot(frame, provider, qualification)


def test_existing_provider_cannot_hide_extra_future_calendar_rows(tmp_path):
    fixture = runpy.run_path(str(Path(__file__).with_name("test_price_basis.py")))["source_fixture"].__wrapped__(tmp_path)
    raw, identity, calendar = fixture
    frame = normalize_futu_candles(raw, identity)
    qualification = validate_price_basis(frame, calendar, "2011-01-04")
    provider = tmp_path / "qualified"
    manifest = {"calendar": calendar, "qualification": qualification, "source_identities": frame.attrs["source_identities"]}
    build_qualified_price_store(frame, provider, "2011-01-04", manifest)
    file = provider / "calendars/day.txt"
    file.write_text(file.read_text() + "2011-01-05\n")
    with pytest.raises(PriceBasisError):
        workflow._existing_snapshot(frame, provider, qualification)


def test_recorder_rerun_checks_bundle_bytes_before_deserializing(tmp_path, monkeypatch):
    import qlib
    import qlib.workflow

    root = tmp_path
    card = root / "research" / workflow.EXPERIMENT
    card.mkdir(parents=True)
    plan = card / "data_plan.json"
    plan.write_text(json.dumps({"data_as_of": "2011-01-04"}))
    for name in ("src/data/price_basis.py", "src/data/published_price_basis.py", "src/workflow/canonical_price_basis.py", "src/data/qlib_native.py"):
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture\n")
    recorder_dir = root / "recorder/run"
    artifact = recorder_dir / "artifacts/inputs/verified_raw_inputs.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"changed cached source bundle")
    (card / "summary.json").write_text(json.dumps({"plan_sha256": workflow.sha(plan),
        "recorder_uri": str(recorder_dir.parent), "recorder_id": "run", "verified_raw_inputs_sha256": "0" * 64}))

    class Recorder:
        def get_local_dir(self):
            return str(recorder_dir)

        def load_object(self, *args):
            pytest.fail("Changed source bundle was deserialized before its independent hash check")

    class Workflow:
        @contextmanager
        def uri_context(self, uri):
            yield

        def get_recorder(self, **kwargs):
            return Recorder()

    monkeypatch.setattr(qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(qlib.workflow, "R", Workflow())
    with pytest.raises(PriceBasisError, match="input bundle identity changed"):
        workflow.run_data_qualification(root=root)
