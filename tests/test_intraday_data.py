from __future__ import annotations

import pandas as pd
import pytest


def _minute_rows(date: str, *, symbol: str = "SOXL", base: float = 100.0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    times = ["09:31", "09:32", "10:00", "15:31", "16:00"]
    closes = [base + 1.0, base + 2.0, base + 3.0, base - 2.0, base + 4.0]
    for idx, (hhmm, close) in enumerate(zip(times, closes, strict=True), start=1):
        rows.append(
            {
                "provider": "futu",
                "symbol": symbol,
                "interval": "1m",
                "timestamp": pd.Timestamp(f"{date} {hhmm}", tz="America/New_York"),
                "date": date,
                "session": "regular",
                "open": base if idx == 1 else close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100 * idx,
                "turnover": close * 100 * idx,
                "source_tz": "America/New_York",
                "adjusted": True,
                "fetched_at": "2026-07-03T00:00:00+00:00",
            }
        )
    return rows


def test_intraday_merge_replaces_overlap_by_symbol_timestamp():
    from src.data.intraday import merge_intraday_bars

    existing = pd.DataFrame(_minute_rows("2026-07-01", base=100.0))
    fetched = pd.DataFrame(
        [
            {
                **_minute_rows("2026-07-01", base=100.0)[0],
                "close": 111.0,
                "volume": 999,
            },
            *_minute_rows("2026-07-02", base=120.0),
        ]
    )

    merged = merge_intraday_bars(existing, fetched)

    overlap = merged.loc[merged["timestamp"] == pd.Timestamp("2026-07-01 09:31", tz="America/New_York")]
    assert len(overlap) == 1
    assert overlap["close"].item() == 111.0
    assert overlap["volume"].item() == 999
    assert len(merged) == len(existing) + len(_minute_rows("2026-07-02", base=120.0))


def test_resample_intraday_bars_returns_qlib_ready_frequency_rows():
    from src.data.intraday import resample_intraday_bars

    minute_frame = pd.DataFrame(_minute_rows("2026-07-02", base=100.0))

    bars = resample_intraday_bars(minute_frame, freq="5min", symbols=["SOXL"])

    first = bars.iloc[0]
    assert first["instrument"] == "SOXL"
    assert first["interval"] == "5min"
    assert first["datetime"] == pd.Timestamp("2026-07-02 09:35", tz="America/New_York")
    assert first["open"] == pytest.approx(100.0)
    assert first["close"] == pytest.approx(102.0)
    assert first["bar_index"] == 0
    assert first["bar_count"] == len(bars)
    assert {"open", "high", "low", "close", "volume", "turnover"}.issubset(bars.columns)


def test_build_t1_intraday_features_aligns_to_next_qlib_calendar_day():
    from src.data.intraday import build_t1_intraday_feature_frame

    minute_frame = pd.DataFrame(
        [
            *_minute_rows("2026-07-02", base=100.0),
            *_minute_rows("2026-07-06", base=120.0),
        ]
    )
    calendar = ["2026-07-02", "2026-07-06", "2026-07-07"]

    features = build_t1_intraday_feature_frame(minute_frame, calendar=calendar, symbol="SOXL")

    assert features["date"].tolist() == ["2026-07-06", "2026-07-07"]
    first = features.iloc[0]
    assert first["INTRADAY_SOXL_BAR_COUNT_1M"] == 5
    assert first["INTRADAY_SOXL_OPEN_RANGE_RETURN_30M"] == pytest.approx(0.03)
    assert first["INTRADAY_SOXL_AMPLITUDE"] == pytest.approx(105.0 / 97.0 - 1.0)
    assert first["INTRADAY_SOXL_DRAWDOWN_FROM_OPEN"] == pytest.approx(-0.03)
    assert "INTRADAY_SOXL_LATE_RETURN_30M" in features.columns
    assert "INTRADAY_SOXL_CLOSE_VWAP_DEVIATION" in features.columns


def test_minute_audit_detects_old_holes_without_rejecting_early_close_or_live_session():
    from src.data.intraday_quality import audit_minute_coverage, plan_minute_repairs

    schedule = pd.DataFrame({
        "open": pd.to_datetime(["2026-07-24 13:30Z", "2026-07-27 13:30Z", "2026-07-28 13:30Z"]),
        "close": pd.to_datetime(["2026-07-24 17:00Z", "2026-07-27 20:00Z", "2026-07-28 20:00Z"]),
    }, index=pd.to_datetime(["2026-07-24", "2026-07-27", "2026-07-28"]))
    rows = []
    for day, count in [("2026-07-24", 210), ("2026-07-27", 61), ("2026-07-28", 61)]:
        for stamp in pd.date_range(f"{day} 09:31", periods=count, freq="min", tz="America/New_York"):
            rows.append(dict(symbol="SOXL", timestamp=stamp, open=100, high=101,
                             low=99, close=100, volume=0, turnover=0))
    frame = pd.DataFrame(rows)
    audit = audit_minute_coverage(frame, schedule, symbols=["SOXL", "SOXX"],
        timestamp_convention="right_edge", observed_at=pd.Timestamp("2026-07-28 10:31", tz="America/New_York"))
    indexed = audit.set_index(["symbol", "date"])
    assert indexed.loc[("SOXL", "2026-07-24"), "training_eligible"]
    assert indexed.loc[("SOXL", "2026-07-27"), "missing_minutes"] == 329
    assert indexed.loc[("SOXX", "2026-07-27"), "missing_minutes"] == 390
    assert indexed.loc[("SOXL", "2026-07-28"), "status"] == "in_progress"
    repairs = plan_minute_repairs(audit)
    assert set(repairs.start) == {"2026-07-24", "2026-07-27"}
    with pytest.raises(ValueError, match="right_edge"):
        audit_minute_coverage(frame, schedule, symbols=["SOXL"],
            timestamp_convention="unknown", observed_at=pd.Timestamp("2026-07-28 21:00Z"))


def test_native_minute_variance_uses_session_open_and_matched_five_minute_path(tmp_path):
    import numpy as np
    from src.data import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame
    from src.data.intraday_quality import audit_minute_coverage
    from src.analysis.intraday_volatility import minute_volatility_reports

    stamps = pd.date_range("2026-07-27 09:31", periods=390, freq="min", tz="America/New_York")
    rows = []
    for symbol, step in [("SOXX", 0.001), ("SOXL", 0.003)]:
        prices = 100 * np.exp(step * np.arange(391))
        for i, stamp in enumerate(stamps):
            rows.append(dict(symbol=symbol, timestamp=stamp, open=prices[i], close=prices[i + 1],
                             high=prices[i + 1], low=prices[i], volume=100))
    raw = pd.DataFrame(rows)
    schedule = pd.DataFrame({"open": pd.to_datetime(["2026-07-27 13:30Z"]),
        "close": pd.to_datetime(["2026-07-27 20:00Z"])}, index=pd.to_datetime(["2026-07-27"]))
    coverage = audit_minute_coverage(raw, schedule, symbols=["SOXL", "SOXX"],
        timestamp_convention="right_edge", observed_at=pd.Timestamp("2026-07-27 21:00Z"))
    provider = tmp_path / "minute_provider"
    fields = ["open", "high", "low", "close", "volume"]
    build_qlib_store_from_frame(raw.rename(columns={"symbol": "instrument", "timestamp": "datetime"}),
        provider_uri=provider, freq="1min", field_map={"$" + name: name for name in fields},
        as_of="2026-07-27", source="analytical constant-log-return fixture")
    features = load_qlib_feature_frame(QlibProviderConfig(provider, freq="1min"),
        instruments=["SOXL", "SOXX"], fields=["$" + name for name in fields],
        start="2026-07-27 09:31", end="2026-07-27 16:00")
    reports = minute_volatility_reports(features, coverage, segments=[("opening", 570, 600), ("rest", 600, 960)])
    soxx = reports["daily"].set_index("instrument").loc["SOXX"]
    assert soxx.realized_vol_pct == pytest.approx(100 * np.sqrt(390 * 0.001 ** 2), rel=1e-4)
    assert soxx.rv1_to_rv5_ratio == pytest.approx(0.2, rel=1e-4)
    opening = reports["profile"].set_index(["instrument", "segment"]).loc[("SOXX", "opening")]
    assert opening.mean_variance_share == pytest.approx(30 / 390, rel=1e-4)
    features.attrs.clear()
    with pytest.raises(ValueError, match="native Qlib"):
        minute_volatility_reports(features, coverage, segments=[("full", 570, 960)])


def test_published_intraday_turnover_matches_price_and_split_share_basis():
    from src.data.minute_price_basis import normalize_none_history, published_intraday_basis

    source = pd.DataFrame([dict(code="US.SOXL", time_key="2026-06-30 09:31:00",
        open=100., high=102., low=100., close=102., volume=100., turnover=10100.)])
    events = pd.DataFrame([dict(ex_div_date="2026-07-01", forward_adj_factorA=.5,
        forward_adj_factorB=-1., split_base=1., split_ert=2.)])
    raw = normalize_none_history(source, symbol="SOXL", interval="1m", fetched_at="2026-07-02T22:00:00Z")
    research, audit = published_intraday_basis(raw, {"SOXL": events}, as_of="2026-07-02")
    assert research.iloc[0].open == pytest.approx(49.)
    assert research.iloc[0].volume == pytest.approx(200.)
    assert research.iloc[0].turnover == pytest.approx(9900.)
    assert research.iloc[0].vwap == pytest.approx(49.5)
    assert raw.iloc[0].turnover == 10100.
    assert audit["vwap_outside_rows"] == 0


def test_history_resume_replaces_explicit_partial_date_instead_of_reusing_connection_cursor(tmp_path):
    from futu import RET_OK, RET_ERROR
    from src.data.minute_intake import collect_futu_minute_history

    def response(times):
        return pd.DataFrame([dict(code="US.SOXL", time_key=stamp, open=100., high=101., low=99.,
            close=100., volume=10., turnover=1000.) for stamp in times])

    class InterruptedConnection:
        calls = 0
        def request_history_kline(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return RET_OK, response(["2026-07-27 09:31:00", "2026-07-27 09:32:00", "2026-07-28 09:31:00"]), b"connection-cursor"
            return RET_ERROR, "transient disconnected connection", None

    parameters = dict(symbol="SOXL", start="2026-07-27", end="2026-07-28", output_dir=tmp_path/"source",
                      checkpoint_dir=tmp_path/"checkpoints", adjustment="NONE", session="RTH")
    with pytest.raises(RuntimeError, match="needs_data"):
        collect_futu_minute_history(context=InterruptedConnection(), **parameters)

    class FreshConnection:
        def request_history_kline(self, *args, **kwargs):
            assert kwargs["page_req_key"] is None and kwargs["start"] == "2026-07-28"
            return RET_OK, response(["2026-07-28 09:31:00", "2026-07-28 09:32:00", "2026-07-28 09:33:00"]), None

    receipt = collect_futu_minute_history(context=FreshConnection(), **parameters)
    observed = pd.read_parquet(tmp_path/"source"/receipt["path"])
    assert receipt["explicit_whole_date_replaced_rows"] == 1
    assert len(observed) == 5 and not observed.time_key.duplicated().any()


def test_full_day_native_snapshots_exclude_afternoon_from_morning_features(tmp_path):
    from src.data import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame
    from src.data.full_day_minute_features import build_full_day_snapshots

    days = pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29"])
    rows = []
    for symbol in ["SOXL", "SOXX", "QQQ"]:
        for day in days:
            for i, stamp in enumerate(pd.date_range(day+pd.Timedelta(hours=9,minutes=31), periods=390, freq="min")):
                price = 100. + .01 * i
                rows.append(dict(instrument=symbol, datetime=stamp, open=price, high=price+.02,
                    low=price-.02, close=price+.01, volume=100., turnover=(price+.01)*100, price_valid=1.))
    fields = ["open", "high", "low", "close", "volume", "turnover", "price_valid"]
    raw = pd.DataFrame(rows)
    provider = tmp_path/"causal_native"
    build_qlib_store_from_frame(raw, provider_uri=provider, freq="1min", field_map={"$"+n:n for n in fields}, as_of="2026-07-29")
    native = load_qlib_feature_frame(QlibProviderConfig(provider,freq="1min"), instruments=["SOXL","SOXX","QQQ"],
        fields=["$"+n for n in fields], start="2026-07-27 09:31", end="2026-07-29 16:00")
    parent = pd.DataFrame(dict(signal_date=days[:2], soxl_weight=.2, soxx_weight=.5, cash_weight=.3,
        tail_probability=.1, alpha_probability=.5, train_end="2025-12-31"))
    arguments = dict(context=native, execution=native, parent=parent, calendar=days, clocks=[570,630,720,840,930],
                     execution_delay_minutes=15, context_symbols=["QQQ"])
    original, families, _ = build_full_day_snapshots(native, **arguments)
    changed = native.copy()
    times = changed.index.get_level_values("datetime")
    changed.loc[times>pd.Timestamp("2026-07-28 12:00"), "$close"] *= 2
    modified, _, _ = build_full_day_snapshots(changed, **arguments)
    at = (pd.Timestamp("2026-07-28 10:30"), "ALLOCATION")
    pd.testing.assert_series_equal(original.loc[at,families["full_day"]], modified.loc[at,families["full_day"]])
    assert original.loc[at,"source_cutoff_at"] <= at[0] < original.loc[at,"entry_at"]


def test_native_interval_ridge_excludes_unmatured_labels():
    import numpy as np
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from src.model.native_interval_forecast import NativeIntervalRidge

    days = pd.date_range("2021-01-01", periods=230, freq="D")
    index = pd.MultiIndex.from_arrays([days, ["ALLOCATION"] * len(days)], names=["datetime", "instrument"])
    raw = pd.DataFrame({("feature", "past_state"): np.arange(len(days), dtype=float),
        ("label", "soxl"): np.arange(len(days), dtype=float)/1000,
        ("label", "soxx"): np.arange(len(days), dtype=float)/2000,
        ("meta", "label_available_at"): [pd.Timestamp("2021-09-01")] * 224 + [pd.Timestamp("2022-01-04")] * 6}, index=index)
    raw.loc[raw.index[-6:], [("label", "soxl"), ("label", "soxx")]] = 1e6
    dataset = DatasetH(handler=DataHandlerLP.from_df(raw), segments={"train": ("2021-01-01", "2021-12-31"),
        "test": ("2021-08-01", "2021-08-05")})
    model = NativeIntervalRidge(features=["past_state"], alpha=10., mature_before="2021-12-31")
    model.fit(dataset)
    assert model.training_rows == 224
    assert model.predict(dataset).abs().max().max() < 1


@pytest.mark.parametrize("nas_enabled", [False, True])
def test_canonical_none_refresh_backfills_old_date_and_tail(monkeypatch, tmp_path, nas_enabled):
    import futu
    import src.data.minute_refresh as refresh
    from src.data.minute_price_basis import normalize_none_history

    def day(symbol, date, count):
        source = pd.DataFrame(dict(code="US."+symbol, time_key=pd.date_range(date+" 09:31", periods=count,
            freq="min").strftime("%Y-%m-%d %H:%M:%S"), open=100., high=101., low=99., close=100., volume=100., turnover=10000.))
        return normalize_none_history(source, symbol=symbol, interval="1m", fetched_at="2026-07-30T22:00:00Z")

    existing = pd.concat([day(symbol, date, count) for symbol in ["SOXL","SOXX"]
                          for date,count in [("2026-07-27",61),("2026-07-28",390)]], ignore_index=True)
    path = tmp_path/"data/processed/intraday/soxl_soxx_none_1m.parquet"
    path.parent.mkdir(parents=True);existing.to_parquet(path,index=False)
    delivery = []
    if nas_enabled:
        from src.data import nas_market_data as nas
        (tmp_path / "data/nas_sources.json").write_text("{}")
        prior = {"source_id": "prior", "snapshot_root": "/nas/prior", "as_of": "2026-07-27",
                 "metadata": {"context_raw_cache": "data/context.parquet", "context_invalid_rows": 2},
                 "objects": [{"local_path": "data/context.parquet", "snapshot_path": "context.parquet"}]}
        monkeypatch.setattr(nas, "load_current_nas_binding", lambda **kwargs: prior)
        monkeypatch.setattr(nas, "materialize_nas_snapshot", lambda *args, **kwargs: delivery.append("restore"))
        def publish_nas(**kwargs):
            assert delivery == ["restore"]
            assert kwargs["inherited_objects"][0]["snapshot_root"] == "/nas/prior"
            assert kwargs["source_metadata"]["context_as_of"] == "2026-07-27"
            assert kwargs["source_metadata"]["context_raw_cache"] == "data/context.parquet"
            assert kwargs["source_metadata"]["quality_status"] == "partial"
            assert pd.read_parquet(path).groupby(["symbol", "date"]).size().eq(390).all()
            delivery.append("publish")
            return {**prior, "source_id": "updated"}
        def register_nas(binding, **kwargs):
            assert delivery == ["restore", "publish"] and binding["source_id"] == "updated"
            assert kwargs["error_count"] == 2
            delivery.append("register")
            return {"name": "us_semis_intraday"}
        monkeypatch.setattr(nas, "publish_nas_snapshot", publish_nas)
        monkeypatch.setattr(nas, "register_nas_source", register_nas)
    calls=[]
    def fetch(symbol, **kwargs):
        assert kwargs["adjust"] == "none" and kwargs["session"] == "RTH"
        calls.append((symbol,kwargs["start"],kwargs["end"]))
        return day(symbol,kwargs["start"],390)
    monkeypatch.setattr(refresh,"fetch_futu_intraday_bars",fetch)
    class Context:
        def __init__(self,**kwargs):pass
        def get_rehab(self,code):return futu.RET_OK,pd.DataFrame()
        def close(self):pass
    monkeypatch.setattr(futu,"OpenQuoteContext",Context)
    def publish(raw, *args, **kwargs):
        assert raw.groupby(["symbol","date"]).size().eq(390).all()
        if nas_enabled:
            raw.to_parquet(path, index=False)
            raw.to_parquet(path.with_name("soxl_soxx_1m.parquet"), index=False)
        return {"cache_rows":len(raw)}
    monkeypatch.setattr(refresh,"publish_core_stores",publish)
    result=refresh.refresh_published_core(root=tmp_path,session_date="2026-07-28",start_session_date=None,
                                           host="127.0.0.1",port=11111)
    assert set(calls)=={(s,d,d) for s in ["SOXL","SOXX"] for d in ["2026-07-27","2026-07-28"]}
    assert not result["remaining_source_anomalies"]
    if nas_enabled:
        assert delivery == ["restore", "publish", "register"]
        assert result["nas_source"]["name"] == "us_semis_intraday"


def test_nas_restore_rejects_corrupt_remote_bytes_and_unsafe_cache_paths(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from src.data import nas_market_data as nas

    expected = b"verified source bytes"
    entry = {"local_path": "data/raw/source.parquet", "snapshot_path": "source.parquet",
             "kind": "source", "sha256": hashlib.sha256(expected).hexdigest(), "bytes": len(expected)}
    binding = {"authority": "yanxuan_nas", "verified": True, "source_id": "frozen",
               "ssh": {"host": "test-nas", "port": 911}, "snapshot_root": "/nas/snapshot", "objects": [entry]}
    destination = tmp_path / entry["local_path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stale local bytes")

    def corrupt(command, **kwargs):
        kwargs["stdout"].write(b"corrupt NAS bytes")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(nas.subprocess, "run", corrupt)
    with pytest.raises(ValueError, match="content hash"):
        nas.materialize_nas_snapshot(binding, workspace_root=tmp_path)
    assert destination.read_bytes() == b"stale local bytes"
    assert not destination.with_suffix(".parquet.nas-partial").exists()
    binding["objects"] = [{**entry, "local_path": "../outside.parquet"}]
    with pytest.raises(ValueError, match="unsafe"):
        nas.materialize_nas_snapshot(binding, workspace_root=tmp_path)


def test_nas_source_registration_preserves_other_state_and_recovers_replaced_index(tmp_path, monkeypatch):
    import copy
    import json
    from src.data import nas_market_data as nas

    binding = {"source_id": "frozen", "snapshot_root": "/nas/snapshot", "as_of": "2026-10-09", "verified": True}
    monkeypatch.setattr(nas, "_settings", lambda _: {"origin": "test", "token": "private",
        "ssh": {}, "state_key": "sync.current", "source_name": "us_semis_intraday", "snapshot_parent": "/nas"})
    monkeypatch.setattr(nas, "_ssh", lambda *args, **kwargs: json.dumps(binding) + "\n")
    other = {"name": "us_equities", "required": False, "status": "warning", "error_count": 1}
    store = {"status": "warning", "data_release": "existing-release", "source_updates": [other],
             "_state": {"revision": 100}}
    def api(**kwargs):
        if kwargs.get("method") == "PUT":
            assert "_state" not in kwargs["payload"]["payload"]
            store.update(copy.deepcopy(kwargs["payload"]["payload"]))
        return copy.deepcopy(store)
    monkeypatch.setattr(nas, "nas_api_call", api)
    registered = nas.register_nas_source(binding, workspace_root=tmp_path,
        observation={"quality_status": "partial"}, error_count=1)
    assert other in store["source_updates"] and registered in store["source_updates"]
    assert store["data_release"] == "existing-release" and store["status"] == "warning"
    assert registered["observation"]["data_owner"] == "yanxuan_nas"
    store["source_updates"] = [other]  # Daily updater replaces its status snapshot.
    recovered = nas.load_current_nas_binding(workspace_root=tmp_path)
    assert recovered["source_id"] == "frozen"
    assert other in store["source_updates"]
    source = next(item for item in store["source_updates"] if item["name"] == "us_semis_intraday")
    assert source["observation"]["index_reconstructed_from_verified_nas_snapshots"]


def test_nas_snapshot_fallback_ignores_other_sources_in_shared_parent(tmp_path, monkeypatch):
    import hashlib
    import json
    import subprocess
    import sys
    from src.data import nas_market_data as nas

    parent = tmp_path / "snapshots"
    minute_path = "data/raw/external/semis_minute_refresh/source_manifest.json"
    for kind, as_of, manifest_path in (
        ("minute", "2026-10-21", minute_path),
        ("treasury", "2026-10-23", "data/snapshots/treasury/source_manifest.json"),
    ):
        root = parent / kind
        source = root / manifest_path
        source.parent.mkdir(parents=True)
        source.write_text(kind)
        binding = {
            "authority": "yanxuan_nas",
            "verified": True,
            "snapshot_root": str(root),
            "source_manifest_path": manifest_path,
            "source_id": hashlib.sha256(source.read_bytes()).hexdigest(),
            "as_of": as_of,
            "verified_at": as_of + "T00:00:00Z",
        }
        (root / "nas_binding.json").write_text(json.dumps(binding))

    def run_remote(_ssh_config, _command, *, input):
        result = subprocess.run([sys.executable, "-c", input], text=True, capture_output=True, check=True)
        return result.stdout

    monkeypatch.setattr(nas, "_ssh", run_remote)
    selected = nas._latest_verified_snapshot(
        {"ssh": {}, "snapshot_parent": str(parent), "source_manifest_path": minute_path}
    )
    assert selected["snapshot_root"] == str(parent / "minute")


def test_nas_snapshot_transport_restores_pinned_bytes_and_rejects_corrupt_published_object(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from src.data import nas_market_data as nas

    root = tmp_path / "workspace"
    source = root / "data/raw/source.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original source bytes")
    manifest = source.with_name("source_manifest.json")
    manifest.write_text(json.dumps({"source": "source.bin"}))
    settings = {"ssh": {"host": "test-nas", "port": 911}, "snapshot_parent": str(tmp_path / "nas")}
    monkeypatch.setattr(nas, "_settings", lambda _: settings)
    original_run = nas.subprocess.run
    def local_ssh(command, **kwargs):
        if command[0] == "ssh":
            command = ["sh", "-c", command[-1]]
        return original_run(command, **kwargs)
    monkeypatch.setattr(nas.subprocess, "run", local_ssh)
    files = {str(p.relative_to(root)): "source" for p in [source, manifest]}
    binding = nas.publish_nas_snapshot(workspace_root=root, source_manifest=manifest,
                                       files=files, as_of="2026-10-09")
    assert binding["verified"]
    cache = tmp_path / "consumer"
    receipt = nas.materialize_nas_snapshot(binding, workspace_root=cache)
    assert receipt["restored_objects"] == 2
    assert (cache / "data/raw/source.bin").read_bytes() == source.read_bytes()
    published = Path(binding["snapshot_root"]) / "data/raw/source.bin"
    published.write_bytes(b"corrupt published source")
    with pytest.raises(ValueError, match="NAS SSH operation failed"):
        nas.publish_nas_snapshot(workspace_root=root, source_manifest=manifest,
                                 files=files, as_of="2026-10-09")
