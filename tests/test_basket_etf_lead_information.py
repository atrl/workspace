"""One synthetic contract test for the frozen basket information screen."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import basket_etf_lead_information as workflow


def test_closed_prefix_full_basket_training_barrier_and_paired_bootstrap(monkeypatch, tmp_path):
    # A fully hashed weight panel must have all 30 source members; the missing
    # WOLF price makes the whole basket unknown while preserving that day row.
    day = pd.Timestamp("2024-01-02")
    members = [f"S{i:02d}" for i in range(29)] + ["WOLF"]
    times = pd.date_range(day + pd.Timedelta(hours=9, minutes=35), periods=12, freq="5min")
    ix = pd.MultiIndex.from_product([times, members[:-1]], names=["datetime", "instrument"])
    stocks = pd.DataFrame({"open": 100., "close": 101., "volume": 10., "price_valid": 1.}, index=ix)
    etf_times = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=91, freq="min")
    etf_ix = pd.MultiIndex.from_product([etf_times, ["SOXX"]], names=["datetime", "instrument"])
    etf = pd.DataFrame({"open": 100., "close": 100.1, "volume": 20., "price_valid": 1.}, index=etf_ix)
    # Qlib D.features exposes instrument before datetime; normalize that API
    # shape once and select all intraday bars by the session's normalized day.
    qlib_order = stocks.swaplevel("datetime", "instrument").sort_index()
    normalized = workflow._normalize_intraday_features(qlib_order)
    assert list(normalized.index.names) == ["datetime", "instrument"]
    assert len(workflow._bars_for_day(normalized, day)) == 29 * 12
    weights = pd.DataFrame({"weight_pct": 100 / 30, "filing_date": day - pd.Timedelta(days=5),
                            "available_date": day - pd.Timedelta(days=4), "accession": "A"},
                           index=pd.MultiIndex.from_product([[day], members], names=["datetime", "instrument"]))
    built = workflow.build_daily_rows([day], stocks, etf, weights)
    assert len(built) == 1 and built.iloc[0].basket_member_count == 30
    assert built.iloc[0].feature_known == 0
    dataset = workflow.as_native_dataset(built)
    from qlib.data.dataset import DatasetH
    assert isinstance(dataset, DatasetH)
    prepared = workflow.read_native_daily_dataset(dataset, built.index)
    assert prepared.index.equals(built.index)
    assert prepared.attrs["native_qlib_dataset"] == "DatasetH"
    # Arithmetic basket aggregation prevents Jensen dispersion from creating
    # a false lead when the ETF exactly matches the weighted simple return.
    dispersed = pd.Series([0.2, -0.2] + [0.] * 28, index=members)
    equal_weights = pd.Series(1 / 30, index=members)
    assert abs(workflow.basket_residual(equal_weights, dispersed, 0.0)) < 1e-15
    # The shared closed-window helper excludes later bars and rejects any
    # incomplete prefix (including its two volume endpoints).
    part = pd.DataFrame({"datetime": times, "open": 100., "close": 101., "volume": 10., "price_valid": 1.})
    first = workflow._window_return(part, day, frequency=5, cutoff="10:30")
    part.loc[len(part)] = [day + pd.Timedelta(hours=10, minutes=35), 1., 500., 1., 1.]
    assert workflow._window_return(part, day, frequency=5, cutoff="10:30") == first
    assert np.isnan(workflow._window_return(part.iloc[1:], day, frequency=5, cutoff="10:30"))

    rng = np.random.default_rng(87)
    train_dates = pd.date_range("2024-01-02", periods=220, freq="B")
    x0, x1 = rng.normal(size=220), rng.normal(size=220)
    y = 0.3 * x0 + 0.5 * x1 + rng.normal(0, 0.02, 220)
    train = pd.DataFrame({workflow.FEATURE_COLUMNS[0]: x0,
                          workflow.FEATURE_COLUMNS[1]: x1,
                          workflow.LABEL_COLUMNS[0]: y},
                         index=pd.MultiIndex.from_arrays([train_dates, ["SOXX"] * 220], names=["datetime", "instrument"]))
    models = workflow.fit_frozen_models(train)
    held_dates = pd.date_range("2025-01-02", periods=220, freq="B")
    held = train.copy()
    held.index = pd.MultiIndex.from_arrays([held_dates, ["SOXX"] * 220], names=["datetime", "instrument"])
    frozen_prediction = models["C"].predict(held)
    held.loc[:, workflow.LABEL_COLUMNS[0]] = 1e9
    np.testing.assert_array_equal(models["C"].predict(held), frozen_prediction)
    with pytest.raises(ValueError, match="2024 rows only"):
        workflow.fit_fixed_ols(held, arm="C")

    called = {}
    def bootstrap(**kwargs):
        called.update(kwargs)
        return np.tile(np.arange(kwargs["observation_count"]), (kwargs["path_count"], 1))
    monkeypatch.setattr(workflow, "stationary_block_bootstrap_indices", bootstrap)
    held = train.copy()
    held.index = pd.MultiIndex.from_arrays([held_dates, ["SOXX"] * 220], names=["datetime", "instrument"])
    evaluated = workflow.paired_evaluation(held, models, bootstrap={"mean_block_length": 63,
        "replicates": 11, "seed": 20261014, "lower_quantile": 0.025 / 15})
    assert called == {"observation_count": 220, "path_length": 220, "path_count": 11,
                      "expected_block_length": 63, "seed": 20261014}
    assert evaluated["eligible_rows"] == 220 and np.isfinite(evaluated["mean_paired_loss_improvement"])

    # A tiny *real* Qlib provider exercises the actual API contracts without
    # touching campaign sources. The 31st historical-union name appears in
    # 2025 only, so annual metadata cannot be equated with the whole union.
    import qlib
    from qlib.data.storage.file_storage import FileFeatureStorage
    days = [pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")]
    def tiny_provider(path, freq, symbols):
        step = 5 if freq == "5min" else 1
        stamps = pd.DatetimeIndex([t for d in days for t in pd.date_range(
            d + pd.Timedelta(hours=9, minutes=30 + step),
            d + pd.Timedelta(hours=11, minutes=1 if step == 1 else 0), freq=f"{step}min")])
        (path / "calendars").mkdir(parents=True)
        (path / "instruments").mkdir()
        (path / "calendars" / f"{freq}.txt").write_text("\n".join(str(t) for t in stamps) + "\n")
        lines = []
        for symbol in symbols:
            begin = stamps[stamps.year == 2025][0] if symbol == "LATE" else stamps[0]
            lines.append(f"{symbol}\t{begin}\t{stamps[-1]}\n")
        (path / "instruments/all.txt").write_text("".join(lines))
        qlib.init(provider_uri=str(path), region="us", kernels=1)
        for symbol in symbols:
            for field, value in {"open": 100., "close": 101., "volume": 10., "price_valid": 1.}.items():
                storage = FileFeatureStorage(symbol, field, freq)
                storage.uri.parent.mkdir(parents=True, exist_ok=True)
                storage.write(np.full(len(stamps), value, dtype=np.float32), index=0)
    tiny_provider(tmp_path / "stocks", "5min", [*members, "LATE"])
    tiny_provider(tmp_path / "etf", "1min", ["SOXX", "SOXL"])
    daily = tmp_path / "daily"
    (daily / "calendars").mkdir(parents=True)
    (daily / "calendars/day.txt").write_text("2024-01-02\n2025-01-02\n")
    holdings = pd.DataFrame({"ticker": members, "weight_pct": 100 / 30,
        "filing_date": "2023-12-28", "report_date": "2023-11-30", "accession": "PRE2024"})
    for column in ("country", "cusip", "isin", "name", "series_id"):
        holdings[column] = "SYNTHETIC"
    holdings["market_value_usd"] = 1.0
    holdings.to_csv(daily / "holdings.csv", index=False)
    (daily / "manifest.json").write_text("{}")
    source_files = {str(p.relative_to(tmp_path)): workflow._sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    plan = {"data": {"basket_provider": {"instruments": [*members, "LATE"]},
        "pit_weights": {"source_csv": "daily/holdings.csv", "source_csv_sha256": workflow._sha(daily / "holdings.csv"),
            "source_manifest": "daily/manifest.json", "source_manifest_sha256": workflow._sha(daily / "manifest.json")},
        "present_day_file_snapshot": {"basket_prices_5min": {"files_sha256": source_files},
                                      "soxx_prices_1min": {"files_sha256": {}}},
        "provider_access": {"prices_5min": {"uri": "stocks", "fields": ["$open", "$close", "$volume", "$price_valid"]},
                            "soxx_prices_1min": {"uri": "etf", "fields": ["$open", "$close", "$volume", "$price_valid"]}}}}
    journal = tmp_path / "attempt.json"
    workflow._journal(journal, status="synthetic_only", operations=[])
    first_phase = workflow.read_native_phase(tmp_path, plan, year=2024, attempt_journal=journal)
    assert len(first_phase) == 1 and first_phase.feature_known.iloc[0] == 1
    assert "LATE" not in first_phase.attrs["phase_source_counts"]["phase_price_instruments"]
    second_phase = workflow.read_native_phase(tmp_path, plan, year=2025, attempt_journal=journal)
    assert second_phase.available_date_max.iloc[0] == pd.Timestamp("2023-12-29")
    assert "LATE" in second_phase.attrs["phase_source_counts"]["phase_price_instruments"]
    assert sum(e["operation"].startswith("qlib_features") and e["state"] == "started"
               for e in json.loads(journal.read_text())["operations"]) == 4
    # Verify the coordinator cannot ask for 2025 until the caller has sealed.
    barrier = []
    def phase(year):
        if year == 2025:
            assert barrier == ["sealed"]
        f = train.copy() if year == 2024 else held.copy()
        f["feature_known"], f["label_known"] = 1, 1
        f.attrs["native_qlib_dataset"] = "DatasetH"
        return f
    monkeypatch.setattr(workflow, "fit_frozen_models", lambda frame: models)
    outcome = workflow.run_two_phase(phase, lambda *args: barrier.append("sealed"),
        plan={"experiment": workflow.EXPERIMENT, "status": "static_plan_only_not_executed"}, plan_sha256="synthetic")
    assert outcome["fit_count"] == 2 and outcome["refit_count"] == 0
