"""One synthetic P1 contract; no real source, model, query or account is loaded."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.path_area_source_preflight import (
    AUX_LABELS, COMMON_FEATURES, MINUTE_FIELDS, PIT_FIELDS, TARGETS,
    build_fold_views, corporate_metadata_qualification, frame_digest,
    qualify_minute_coverage, qualify_pit_pool, qualify_saved_model,
    validate_training_frames,
)
from src.data.minute_allocation_features import label_maturity
from src.model.minute_pit_allocator import MinutePitAllocator
from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationSpec
from src.workflow.path_area_source_preflight import bounded_raw_slices, _dataset


def test_p1_alignment_static_qualification_and_causal_coverage(tmp_path, monkeypatch):
    # Full production column schema with fabricated data and a small row count.
    calendar = pd.bdate_range("2019-05-01", "2020-01-20").difference(pd.DatetimeIndex(["2020-01-01"]))
    dates = calendar[:-6]
    base = [f"x{i}" for i in range(39)]
    values = np.arange(len(dates) * 61, dtype=float).reshape(len(dates), 61) / 100
    feature = pd.DataFrame(values, index=dates, columns=base + [f"unused{i}" for i in range(22)])
    labels = pd.DataFrame(np.zeros((len(dates), 5)), index=dates, columns=TARGETS)
    common = pd.DataFrame(values[:, :16], index=dates, columns=COMMON_FEATURES)
    source_dates = pd.DataFrame({"trade_date": calendar[1:len(dates)+1],
                                 "exit_date": calendar[2:len(dates)+2]}, index=dates)
    meta = label_maturity(source_dates, calendar)
    primary = pd.concat({"feature": feature, "label": labels, "common": common, "meta": meta}, axis=1)
    primary.index = pd.MultiIndex.from_arrays([dates, ["SOXL_SOXX_CASH"] * len(dates)],
                                             names=["datetime", "instrument"])
    aux = pd.concat({"feature": common.iloc[:3].copy(),
                     "label": pd.DataFrame(np.zeros((3, 3)), index=dates[:3], columns=AUX_LABELS),
                     "meta": pd.DataFrame({"observations_end": pd.Timestamp("2022-12-30")}, index=dates[:3])}, axis=1)
    aux.index = pd.MultiIndex.from_arrays([dates[:3], ["TEST"] * 3], names=["datetime", "instrument"])
    original, original_aux = frame_digest(primary), frame_digest(aux)
    audit = validate_training_frames(primary, aux, base_features=base,
                                     expected_primary_rows=len(primary), expected_auxiliary_rows=3)
    assert audit["primary_shape"] == [len(primary), 85] and audit["auxiliary_shape"] == [3, 20]
    views = build_fold_views(primary, calendar, base_features=base, years=(2020,))
    timing, fold = views["timing"], views["folds"][2020]
    assert timing.loc["2019-12-31", "trade_year"] == 2020
    assert pd.Timestamp("2019-12-31") in fold["prediction_index"].get_level_values("datetime")
    assert fold["train_end"] == calendar[calendar.searchsorted(pd.Timestamp("2020-01-01")) - 21]
    assert fold["train"].index.get_level_values("datetime").max() <= fold["train_end"]
    assert fold["train"][("meta", "label_available_at")].lt(fold["cutoff"]).all()
    assert not timing.iloc[-1].next_target_present  # Missing reset target remains a missing target.
    bad_clock = primary.copy()
    bad_clock[("meta", "label_available_at")] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="calendar/clock mismatch"):
        build_fold_views(bad_clock, calendar, base_features=base, years=(2020,))
    bad_index = primary.copy(); bad_index.index = primary.index[::-1]
    with pytest.raises(ValueError, match="index/order"):
        build_fold_views(bad_index, calendar, base_features=base, years=(2020,))

    # Synthetic original class, static network only; no model forward/fit allowed.
    spec = PitAuxiliaryAllocationSpec(root=Path("."))
    masks = {"unused0": "minute_core_ready"}
    model = MinutePitAllocator(base_features=base, extra_features=[], feature_masks=masks,
        spec=asdict(spec), seed=71 + 2020, cutoff=str(fold["cutoff"]),
        train_end=str(fold["train_end"]), minimum_observations=200)
    with torch.random.fork_rng():
        model.network = model._initialize()
    model.primary_scaler, model.common_scaler = fold["primary_scaler"], fold["common_scaler"]
    train_dates = fold["train"].index.get_level_values("datetime")
    model.audit = {"training_rows": len(fold["train"]), "first_training_signal": str(train_dates.min().date()),
        "last_training_signal": str(train_dates.max().date()),
        "last_mature_label_at": str(fold["train"][("meta", "label_available_at")].max()),
        "cutoff": str(fold["cutoff"]), "epochs": 300, "checkpoint_policy": "fixed_final_epoch", "auxiliary_rows": 0}
    def prohibited(*args, **kwargs):
        raise AssertionError("P1 cannot invoke a model")
    monkeypatch.setattr(model.network, "forward", prohibited)
    monkeypatch.setattr(model, "predict", prohibited)
    monkeypatch.setattr(model, "fit", prohibited)
    kwargs = dict(base_seed=71, base_features=base, expected_spec=asdict(spec), feature_masks=masks)
    result = qualify_saved_model(model, fold, auxiliary_rows=0, **kwargs)
    assert result["static_qualified"] and result["forward_calls"] == 0 and not result["P2_admitted"]
    assert qualify_saved_model(model, fold, **kwargs)["status"] == "unknown"
    model.seed = 71
    assert "seed_plus_trade_year" in qualify_saved_model(model, fold, auxiliary_rows=0, **kwargs)["failures"]
    model.seed += 2020
    model.primary_scaler = {k: dict(v) for k, v in model.primary_scaler.items()}
    model.primary_scaler["mean"][base[0]] += .1
    assert "primary_scaler_exact" in qualify_saved_model(model, fold, auxiliary_rows=0, **kwargs)["failures"]

    # The observation mask depends only on the closed prefix, never afternoon bars.
    day, next_day = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")
    stamps = list(pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=15, freq="min"))
    stamps += [day + pd.Timedelta(hours=9, minutes=48), day + pd.Timedelta(hours=10, minutes=33),
               day + pd.Timedelta(hours=13), next_day + pd.Timedelta(hours=9, minutes=48)]
    index = pd.MultiIndex.from_product([stamps, ["SOXL", "SOXX"]], names=["datetime", "instrument"])
    bars = pd.DataFrame(10., index=index, columns=MINUTE_FIELDS, dtype=np.float32)
    bars["$price_valid"] = 1.
    one_timing = timing.loc[[pd.Timestamp("2019-12-31")]].copy()
    schedule = pd.DataFrame({"open": pd.DatetimeIndex(["2020-01-02 09:30"], tz="America/New_York"),
                             "close": pd.DatetimeIndex(["2020-01-02 13:00"], tz="America/New_York")}, index=[day])
    coverage = qualify_minute_coverage(bars, one_timing, schedule, source_clock_qualified=True)
    assert coverage.iloc[0].prefix_mask and coverage.iloc[0].session_close
    assert coverage.iloc[0].fill_0947_open and coverage.iloc[0].next_0947_open
    short = bars.drop(index=day + pd.Timedelta(hours=13), level="datetime")
    short.loc[(day + pd.Timedelta(hours=10, minutes=33), "SOXL"), "$open"] = np.nan
    changed = qualify_minute_coverage(short, one_timing, schedule, source_clock_qualified=True)
    assert changed.iloc[0].prefix_mask and not changed.iloc[0].session_close and not changed.iloc[0].recovery_1032_open
    wrong_fill = bars.drop(index=day + pd.Timedelta(hours=9, minutes=48), level="datetime")
    assert not qualify_minute_coverage(wrong_fill, one_timing, schedule, source_clock_qualified=True).iloc[0].fill_0947_open
    unqualified = qualify_minute_coverage(bars, one_timing, schedule, source_clock_qualified=False).iloc[0]
    assert unqualified.prefix_observed_complete and not unqualified.prefix_mask and not unqualified.prefix_qualification_known
    with pytest.raises(ValueError, match="schema"):
        qualify_minute_coverage(pd.concat([bars, bars.iloc[:1]]), one_timing, schedule, source_clock_qualified=True)

    # Every expected flag cell matters: zero members, unavailable data and a
    # positive pool are three different conclusions. No union backfill is used.
    pit_dates = pd.bdate_range("2019-01-02", periods=3)
    pit_index = pd.MultiIndex.from_product([pit_dates, ["A", "B"]], names=["datetime", "instrument"])
    flags = pd.DataFrame(0., index=pit_index, columns=PIT_FIELDS)
    pit_kwargs = dict(calendar=pit_dates, instruments=["A", "B"], start=pit_dates[0], end=pit_dates[-1], source_qualified=True)
    assert qualify_pit_pool(flags, **pit_kwargs)["empty_pool_proven"]
    assert not qualify_pit_pool(flags.iloc[:-1], **pit_kwargs)["empty_pool_proven"]
    missing = flags.copy(); missing.iloc[0, 0] = np.nan
    assert qualify_pit_pool(missing, **pit_kwargs)["status"] == "unknown"
    assert qualify_pit_pool(None, **pit_kwargs)["status"] == "unknown"
    present = flags.copy(); present.iloc[0, :2] = 1.
    assert qualify_pit_pool(present, **pit_kwargs)["status"] == "nonempty_needs_raw_auxiliary"
    assert qualify_pit_pool(flags, **{**pit_kwargs, "source_qualified": False})["status"] == "unknown"
    from qlib.data.dataset.handler import DataHandler
    empty_aux = aux.iloc[:0].copy()
    empty_dataset = _dataset(empty_aux, "train")
    pd.testing.assert_frame_equal(empty_dataset.prepare("train", col_set=DataHandler.CS_RAW), empty_aux,
                                  check_exact=True)

    cash = pd.DataFrame([{"event_id": "cash0", "instrument": "SOXX", "ex_date": "2020-01-02",
                          "pay_date": "2020-01-03", "original_total_decimal": "0.2"},
                         {"event_id": "future", "instrument": "SOXX", "ex_date": "2023-01-02",
                          "pay_date": "2023-01-03", "original_total_decimal": "0.1"}])
    split = pd.DataFrame(columns=["event_id", "instrument", "effective_trading_date", "new_shares_per_old_share"])
    corp_kwargs = dict(sessions=pd.bdate_range("2020-01-02", "2020-01-10"), start="2020-01-02",
                       end="2020-01-10", source_qualified=True)
    corp = corporate_metadata_qualification(cash, split, **corp_kwargs)
    assert corp["status"] == "qualified_metadata_only" and corp["residency"]["cash"]["post_train_rows"] == 1
    assert corp["events"].iloc[0].available_date == pd.Timestamp("2020-01-06")
    split = pd.DataFrame([{"event_id": "split0", "instrument": "SOXX", "effective_trading_date": "2020-01-02",
                           "new_shares_per_old_share": 2}])
    assert corporate_metadata_qualification(cash, split, **corp_kwargs)["status"] == "unknown"

    # The workflow's raw-bin guard is exercised on a temporary synthetic file,
    # without D.features, a real provider, or any source object deserialization.
    from qlib.data.storage.file_storage import FileFeatureStorage
    provider = tmp_path / "provider"
    path = provider / "features/soxl/open.day.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(np.array([0., 10., 11., 12.], dtype="<f4").tobytes())
    calendar_file = provider / "calendars/day.txt"
    calendar_file.parent.mkdir(parents=True)
    calendar_file.write_text("".join(str(day.date()) + "\n" for day in pit_dates))
    import qlib
    qlib.init(provider_uri=str(provider), region="us", kernels=1, clear_mem_cache=True)
    storage = FileFeatureStorage(instrument="soxl", field="open", freq="day", provider_uri=str(provider))
    bounded_kwargs = dict(freq="day", fields=["$open"], instruments=["SOXL"],
                          start=pit_dates[0], end=pit_dates[1], calendar=pit_dates)
    with bounded_raw_slices(provider, **bounded_kwargs) as observed:
        assert len(storage[0:2]) == 2 and len(observed["reads"]) == 1
        with pytest.raises(ValueError, match="bounds"):
            storage[0:3]
    assert frame_digest(primary) == original and frame_digest(aux) == original_aux
    # Necessary sealed implementation authority, not another old-model test.
    original_module = Path("src/model/minute_pit_allocator.py")
    assert hashlib.sha256(original_module.read_bytes()).hexdigest() == "d06962ac2739633c0af34e3843365f1ad4c0a9bbc9dff0ba38e1636db6a175c8"
