import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.alfred_yield_information import read_initial_yield_archive, qualify_yield_change, build_yield_information_store


def _archive(tmp_path):
    directory = tmp_path / "DFII10"
    directory.mkdir()
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as z:
        z.writestr("README.txt", "Series ID: DFII10\nOutput Format: Observations, Initial Release Only\n")
        z.writestr("data.csv", "period_start_date,DFII10,realtime_start_date\n2020-01-02,-0.5,2020-01-03\n2020-01-03,.,2020-01-06\n")
    data = packed.getvalue()
    (directory / "batch.zip").write_bytes(data)
    manifest = dict(series_id="DFII10", as_of="2020-01-10", output_format="Observations, Initial Release Only",
                    entries=[dict(path="DFII10/batch.zip", sha256=hashlib.sha256(data).hexdigest(), csv_name="data.csv")])
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_negative_real_yield_is_valid_and_unknown_is_preserved(tmp_path):
    directory = _archive(tmp_path)
    raw = read_initial_yield_archive(directory, as_of="2020-01-10", series_id="DFII10")
    assert raw.value.iloc[0] == -.5 and np.isnan(raw.value.iloc[1])
    with pytest.raises(ValueError, match="cutoff"):
        read_initial_yield_archive(directory, as_of="2020-01-09", series_id="DFII10")
    (directory / "batch.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        read_initial_yield_archive(directory, as_of="2020-01-10", series_id="DFII10")


def test_no_early_use_unknown_latest_no_old_fill_and_native_roundtrip(tmp_path):
    calendar = pd.bdate_range("2020-01-02", periods=9)
    raw = pd.DataFrame(dict(observation_date=calendar[:6], release_date=calendar[:6],
                            value=[-.5, -.4, -.3, np.nan, -.1, .1]))
    state, events, audit = qualify_yield_change(raw, calendar, change_sessions=1)
    assert state.known.iloc[0] == 0 and state.known.iloc[1] == 0
    assert state.known.iloc[2] == 1 and state.real_yield_change.iloc[2] == pytest.approx(.1)
    assert state.state_known.iloc[4] == 0 and np.isnan(state.real_yield.iloc[4])
    assert state.known.iloc[5] == 0  # previous endpoint was unknown
    assert audit["negative_yield_observations"] == 4 and audit["missing_values_filled"] is False
    assert len(state) == len(calendar)
    manifest, native, digest = build_yield_information_store(state, provider_uri=tmp_path / "qlib",
        as_of="2020-01-14", source="synthetic_source_contract_fixture", instrument="REAL_YIELD_INFO")
    assert len(native) == len(calendar) and len(digest) == 64
    assert manifest["source_deployment_gate"] is False
    assert native["$real_yield"].min() < 0


def test_premature_release_is_quarantined_at_its_observation_clock():
    calendar = pd.bdate_range("2020-01-02", periods=5)
    raw = pd.DataFrame(dict(observation_date=calendar[:2], release_date=[calendar[0],calendar[0]], value=[-.5, -.4]))
    state, events, audit = qualify_yield_change(raw, calendar, change_sessions=1)
    assert events.premature_release.tolist() == [False,True]
    assert state.state_known.iloc[2] == 0 and np.isnan(state.real_yield.iloc[2])
    assert audit["premature_market_observations"] == 1


def test_unknown_publication_date_and_nonfinite_yield_block_before_state_creation():
    calendar = pd.bdate_range("2020-01-02", periods=5)
    raw = pd.DataFrame(dict(observation_date=calendar[:2], release_date=[calendar[0], pd.NaT], value=[-.5, -.4]))
    with pytest.raises(ValueError, match="source dates"):
        qualify_yield_change(raw, calendar, change_sessions=1)
    raw["release_date"] = calendar[:2]
    raw.loc[0, "value"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        qualify_yield_change(raw, calendar, change_sessions=1)
