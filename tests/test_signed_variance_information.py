"""One bounded synthetic/native regression; never reads campaign price inputs."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from src.workflow import signed_variance_information as screen


def test_native_rth_lags_model_seal_smearing_and_qlike(tmp_path, monkeypatch):
    import exchange_calendars as xc
    import qlib
    from qlib.data.storage.file_storage import FileFeatureStorage
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    exchange = xc.get_calendar("XNYS", start="2020-11-02", end="2021-01-08")
    sessions = exchange.sessions_in_range("2020-11-02", "2021-01-08")
    schedule = pd.DataFrame({"open_et": [exchange.session_open(s).tz_convert("America/New_York") for s in sessions],
                             "close_et": [exchange.session_close(s).tz_convert("America/New_York") for s in sessions]},
                            index=pd.DatetimeIndex(sessions, name="datetime"))
    schedule["minutes"] = ((schedule.close_et - schedule.open_et).dt.total_seconds() / 60).astype(int)
    rng = np.random.default_rng(2015)
    parts = []
    for day, session in schedule.iterrows():
        times = pd.date_range(session.open_et.tz_localize(None) + pd.Timedelta(minutes=1),
                              session.close_et.tz_localize(None), freq="min")
        changes = rng.normal(0, .0004, len(times))
        # Each day's unrelated opening level demonstrates that no overnight
        # jump belongs in that day's RV.
        opening = 50. if day.day % 2 else 5000.
        closes = opening * np.exp(np.cumsum(changes))
        opens = np.r_[opening, closes[:-1]]
        parts.append(pd.DataFrame({"open": opens, "close": closes, "price_valid": 1.}, index=times))
    bars = pd.concat(parts)
    provider = tmp_path / "synthetic_provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars/1min.txt").write_text("\n".join(str(t) for t in bars.index) + "\n")
    (provider / "instruments/all.txt").write_text(f"SOXX\t{bars.index[0]}\t{bars.index[-1]}\n")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    for field in ("open", "close", "price_valid"):
        storage = FileFeatureStorage("SOXX", field, "1min")
        storage.uri.parent.mkdir(parents=True, exist_ok=True)
        storage.write(bars[field].to_numpy(np.float32), index=0)
    journal = tmp_path / "synthetic_attempt.json"
    journal.write_text(json.dumps({"operations": []}))
    daily, daily_dataset = screen._read_phase(tmp_path, {"source": {"provider_uri": "synthetic_provider"}}, schedule, "training", journal)
    assert daily.session_known.eq(1).all()
    halfday = pd.Timestamp("2020-11-27")
    assert daily.loc[halfday, "expected_minutes"] == daily.loc[halfday, "observed_minutes"] == 210
    assert daily.loc[halfday, "completed_available_at"].hour == 13
    expected = np.log(np.r_[bars.loc[bars.index.normalize() == halfday].open.iloc[0],
                            bars.loc[bars.index.normalize() == halfday].close.to_numpy(float)])
    assert daily.loc[halfday, "rv"] == pytest.approx(np.square(np.diff(expected)).sum(), rel=1e-4)
    missing = bars.drop(bars.loc[bars.index.normalize() == halfday].index[50])
    broken = screen.measure_sessions(missing, schedule)
    assert broken.loc[halfday, "session_known"] == 0 and np.isnan(broken.loc[halfday, "rv"])
    targets = screen.target_rows(daily, schedule, start="2020-12-04", end="2021-01-08")
    assert targets.feature_known.eq(1).all()
    assert (targets.feature_available_at < targets.decision_at).all()
    assert (targets.label_mature_at > targets.decision_at).all()
    broken_targets = screen.target_rows(broken, schedule, start="2020-12-04", end="2021-01-08")
    assert broken_targets.feature_known.eq(0).any()  # no compacting the bad lag
    dataset, native_targets = screen._dataset(targets, features=screen.FEATURES, labels=("target_rv",))
    assert native_targets.index.equals(targets.index)
    assert native_targets.feature_available_at.equals(targets.feature_available_at)

    # Fit precisely two synthetic heads; model and smearing use training only.
    dates = pd.bdate_range("2020-01-02", periods=230, name="datetime")
    x = rng.normal(size=(len(dates), 5))
    x[:, -1] = rng.uniform(.1, .9, len(dates))
    training = pd.DataFrame(x, columns=screen.FEATURES, index=dates)
    training["target_rv"] = np.exp(-10 + .2*x[:, 0] + .1*x[:, 1] + .15*x[:, 2] + .04*x[:, 3] + .8*x[:, 4] + rng.normal(0, .2, len(dates)))
    training["feature_known"], training["label_known"] = 1, 1
    training_dataset, training = screen._dataset(training, features=screen.FEATURES, labels=("target_rv",))
    models = {arm: screen.fit_model(training, arm=arm) for arm in ("B", "C")}
    for model in models.values():
        z = (training[list(model.features)].to_numpy(float) - model.mean) / model.std
        prediction_log = np.column_stack([np.ones(len(z)), z]) @ model.coefficients
        assert model.smearing == pytest.approx(np.exp(np.log(training.target_rv) - prediction_log).mean())
        assert (model.predict(training) > 0).all()
    # Native Recorder must be finished and the exact tail/models readable
    # before a held callback is allowed to return its source-derived values.
    tail_dataset, tail = screen._dataset(daily.tail(22), features=("rv", "negative_share", "signed_return"))
    with R.start(experiment_name="synthetic_signed_variance", uri=str(tmp_path / "recorder")):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="models", **{arm: asdict(m) for arm, m in models.items()})
        recorder.save_objects(artifact_path="freeze", tail=tail_dataset, receipt_sha256=screen._frame_hash(tail))
        recorder.log_metrics(synthetic_model_count=2)
    assert recorder.status == Recorder.STATUS_FI
    screen._no_writer()
    readback = screen.prepare_groups(recorder.load_object("freeze/tail"), "all")
    readback.columns = readback.columns.get_level_values(1)
    readback.index = readback.index.droplevel("instrument")
    assert screen._frame_hash(readback) == recorder.load_object("freeze/receipt_sha256")
    assert recorder.load_object("models/C") == asdict(models["C"])
    held = training.iloc[:5].copy()
    held.index = pd.bdate_range("2023-01-03", periods=5, name="datetime")
    prediction_before = models["C"].predict(held)
    held.target_rv *= 100
    np.testing.assert_array_equal(prediction_before, models["C"].predict(held))
    assert screen.qlike([2.], [3.])[0] == pytest.approx(np.log(3) + 2/3)
    called = []
    def bootstrap(**kwargs):
        called.append(kwargs)
        return np.tile(np.arange(kwargs["observation_count"]), (kwargs["path_count"], 1))
    monkeypatch.setattr(screen, "stationary_block_bootstrap_indices", bootstrap)
    result = screen.evaluate(held, models, held.index, bootstrap={**screen.BOOTSTRAP, "replicates": 7})
    assert len(called) == 1 and called[0]["expected_block_length"] == 63
    assert result["yearly"]["2023"]["sessions"] == 5
    held.loc[held.index[0], "feature_known"] = 0
    with pytest.raises(ValueError, match="no QLIKE/bootstrap"):
        screen.evaluate(held, models, held.index)
    assert len(called) == 1
    with pytest.raises(ValueError, match="calendar changed or compacted"):
        screen.evaluate(held.iloc[1:], models, held.index)
    with pytest.raises(ValueError, match="training target dates"):
        screen.fit_model(held, arm="C")
