"""Frozen, two phase SOXX basket lead information screen.

This module contains the mechanics only.  The notebook calls ``run`` only after the owner seals the exact GO identity.
Source reads, two OLS fits and the one bootstrap have durable attempt records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.core.native_evidence import native_dataset, finalize_evidence
from src.data.closed_intraday_window import closed_window_return
from src.objective.relative_wealth import stationary_block_bootstrap_indices

EXPERIMENT = "basket_etf_lead_information_20261020"
TRAIN_YEAR, EVALUATION_YEAR = 2024, 2025
MINIMUM_ELIGIBLE_DAYS = 200
MEMBERS_PER_DAY = 30
BOOTSTRAP = {"mean_block_length": 63, "replicates": 10_000,
             "seed": 20261014, "lower_quantile": 0.025 / 15}
FEATURE_COLUMNS = ("r_soxx_0931_1030", "basket_minus_soxx_0935_1030")
LABEL_COLUMNS = ("y_soxx_1032_1100",)
METADATA_COLUMNS = ("basket_member_count", "basket_weight_sum", "filing_date_max",
                    "available_date_max", "accession_set_sha256", "feature_known",
                    "label_known", "preentry_soxx_log_return_1031_1033")


def _require(ok: bool, message: str) -> None:
    if not bool(ok):
        raise ValueError("blocked: " + message)


def _atomic_json(path: str | Path, value: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _journal(path: str | Path, *, status: str, **changes) -> None:
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(status=status, **changes)
    _atomic_json(path, state)


def _tracked(path: str | Path, year: int, operation: str, callback):
    state = json.loads(Path(path).read_text())
    events = list(state.get("operations", []))
    events.append({"year": int(year), "operation": operation, "state": "started"})
    state["operations"] = events
    _atomic_json(path, state)
    value = callback()
    events.append({"year": int(year), "operation": operation, "state": "completed"})
    state["operations"] = events
    _atomic_json(path, state)
    return value


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


IMPLEMENTATION_DEPENDENCIES = (
    "src/core/native_evidence.py", "src/objective/relative_wealth.py",
    "src/data/nas_market_data.py",
)


def normalized_notebook_identity(path: str | Path) -> str:
    notebook = json.loads(Path(path).read_text())
    source = [{"cell_type": cell["cell_type"], "source": "".join(cell.get("source", []))}
              for cell in notebook["cells"]
              if cell.get("metadata", {}).get("role") != "conclusion"]
    return hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def implementation_dependencies(root: str | Path) -> dict:
    return {name: _sha(Path(root) / name) for name in IMPLEMENTATION_DEPENDENCIES}


def validate_execution_go(root: str | Path, execution_go: Mapping, *, state_path: str | Path) -> dict:
    """Require owner GO and exact independent-review identities before a run."""
    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    _require(isinstance(execution_go, Mapping) and execution_go.get("root_go") is True,
             "explicit root GO is required")
    plan_path = card / "plan.json"
    design_path = card / "design_review.json"
    implementation_path = root / "src/workflow/basket_etf_lead_information.py"
    notebook_path = card / "basket_etf_lead_information_20261020.ipynb"
    test_path = root / "tests/test_basket_etf_lead_information.py"
    review_path = card / "implementation_review.json"
    actual = {"plan_sha256": _sha(plan_path), "design_review_sha256": _sha(design_path),
              "implementation_sha256": _sha(implementation_path), "notebook_sha256": normalized_notebook_identity(notebook_path),
              "test_sha256": _sha(test_path)}
    actual["implementation_dependencies_sha256"] = implementation_dependencies(root)
    _require(execution_go.get("owner") == "root" and execution_go.get("approved") is True,
             "root owner approval is required")
    _require(execution_go.get("budget") == json.loads(plan_path.read_text())["budget_and_exclusions"],
             "root GO budget differs from frozen plan")
    for key, digest in actual.items():
        _require(execution_go.get(key) == digest, f"root GO {key} differs from current bytes")
    design = json.loads(design_path.read_text())
    _require(design.get("verdict") == "ACCEPT_STATIC_BOUNDED_INFORMATION_DESIGN"
             and design.get("plan_sha256") == actual["plan_sha256"],
             "static design review is absent, stale, or not accepted")
    _require(review_path.is_file(), "independent implementation review is missing")
    _require(execution_go.get("implementation_review_sha256") == _sha(review_path),
             "implementation review hash differs from root GO")
    review = json.loads(review_path.read_text())
    _require(review.get("verdict") == "APPROVE_BOUNDED_INFORMATION_IMPLEMENTATION",
             "independent implementation review has not approved this implementation")
    for key in ("plan_sha256", "implementation_sha256", "notebook_sha256", "test_sha256",
                "implementation_dependencies_sha256"):
        _require(review.get(key) == actual[key], f"implementation review {key} differs")
    state_path = Path(state_path)
    _require(not state_path.exists(), "single-attempt run state already exists; automatic rerun is forbidden")
    _require(state_path.resolve() == (card / "run_state.json").resolve(),
             "attempt journal must use the card's canonical run_state.json")
    return {**actual, "implementation_review_sha256": _sha(review_path)}


def _finite_positive(value) -> bool:
    return bool(np.isfinite(value) and float(value) > 0)


def _bars_for_day(bars: pd.DataFrame, day, instrument: str | None = None) -> pd.DataFrame:
    if not isinstance(bars.index, pd.MultiIndex) or list(bars.index.names) != ["datetime", "instrument"]:
        raise ValueError("Qlib bars must retain (datetime, instrument) index")
    dates = pd.DatetimeIndex(bars.index.get_level_values("datetime"))
    part = bars.loc[dates.normalize() == pd.Timestamp(day).normalize()]
    if part.empty:
        return pd.DataFrame(columns=["datetime", "instrument", "open", "close", "volume", "price_valid"])
    if instrument is not None:
        part = part.loc[part.index.get_level_values("instrument").str.upper() == instrument.upper()]
    out = part.reset_index()
    return out[["datetime", "instrument", "open", "close", "volume", "price_valid"]]


def _normalize_intraday_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Qlib D.features output to the documented daily-first grain."""
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {"datetime", "instrument"}:
        raise ValueError("Qlib D.features must return instrument and datetime index levels")
    if list(frame.index.names) == ["instrument", "datetime"]:
        frame = frame.swaplevel("instrument", "datetime")
    elif list(frame.index.names) != ["datetime", "instrument"]:
        raise ValueError("Qlib D.features returned duplicate or unexpected index levels")
    frame = frame.sort_index().copy()
    frame.columns = [str(column).lstrip("$").lower() for column in frame.columns]
    return frame


def _window_return(bars: pd.DataFrame, day, *, frequency: int, cutoff: str) -> float:
    if bars.empty:
        return float("nan")
    return closed_window_return(bars, day=day, cutoff=cutoff, frequency_minutes=frequency)


def _label_from_soxx(soxx: pd.DataFrame, day, *, entry_minute: int = 33,
                     exit_hour: int = 11, exit_minute: int = 1) -> float:
    """Use right-edge labels 10:33 and 11:01 for 10:32/11:00 open fills."""
    if soxx.empty:
        return float("nan")
    frame = soxx.reset_index()
    frame = frame.loc[frame.instrument.astype(str).str.upper().eq("SOXX")]
    frame = frame.set_index("datetime").sort_index()
    day = pd.Timestamp(day).normalize()
    entry_at = day + pd.Timedelta(hours=10, minutes=entry_minute)
    exit_at = day + pd.Timedelta(hours=exit_hour, minutes=exit_minute)
    if entry_at not in frame.index or exit_at not in frame.index:
        return float("nan")
    entry, exit_ = frame.loc[entry_at], frame.loc[exit_at]
    if isinstance(entry, pd.DataFrame) or isinstance(exit_, pd.DataFrame):
        return float("nan")
    valid_flags = np.isfinite([entry.price_valid, exit_.price_valid]).all() \
        and float(entry.price_valid) == 1.0 and float(exit_.price_valid) == 1.0
    if not (valid_flags
            and _finite_positive(entry.open) and _finite_positive(exit_.open)
            and np.isfinite([entry.volume, exit_.volume]).all()
            and float(entry.volume) > 0 and float(exit_.volume) > 0):
        return float("nan")
    return float(np.log(float(exit_.open) / float(entry.open)))


def basket_residual(weights: pd.Series, simple_returns: pd.Series, soxx_simple_return: float) -> float:
    """Use an arithmetic, whole-basket return before taking the log residual."""
    w = pd.to_numeric(weights, errors="coerce")
    r = pd.to_numeric(simple_returns.reindex(w.index), errors="coerce")
    if (len(w) != MEMBERS_PER_DAY or len(r) != MEMBERS_PER_DAY
            or not np.isfinite(w.to_numpy(float)).all() or not w.gt(0).all()
            or not np.isfinite(r.to_numpy(float)).all() or (r <= -1).any()
            or not _finite_positive(1.0 + soxx_simple_return) or soxx_simple_return <= -1):
        return float("nan")
    basket = float(np.average(r.to_numpy(float), weights=w.to_numpy(float)))
    return float(np.log1p(basket) - np.log1p(float(soxx_simple_return)))


def build_daily_rows(calendar, stock_bars: pd.DataFrame, soxx_bars: pd.DataFrame,
                     weight_panel: pd.DataFrame) -> pd.DataFrame:
    """Build complete daily observations while retaining every calendar row.

    ``weight_panel`` comes only from SEC CSV plus the phase Qlib daily calendar.
    A row is eligible only when all 30 disclosed members have valid prices;
    weights are never renormalized over a surviving subset.
    """
    days = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    _require(days.is_unique and days.is_monotonic_increasing, "phase Qlib calendar is not unique and ordered")
    _require(isinstance(weight_panel.index, pd.MultiIndex)
             and list(weight_panel.index.names) == ["datetime", "instrument"],
             "PIT panel must be keyed by (datetime, instrument)")
    required = {"weight_pct", "filing_date", "accession"}
    _require(required.issubset(weight_panel.columns), "PIT panel metadata is incomplete")
    rows = []
    for day in days:
        key_day = pd.Timestamp(day)
        try:
            disclosed = weight_panel.xs(key_day, level="datetime").copy()
        except KeyError:
            disclosed = pd.DataFrame(columns=sorted(required))
        disclosed.index = disclosed.index.astype(str).str.upper()
        if disclosed.index.has_duplicates:
            raise ValueError("blocked: duplicate SEC constituent weights for a session")
        filing = pd.to_datetime(disclosed.get("filing_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
        # build_pit_soxx_weight_panel deliberately omits this derived field;
        # its configured one-calendar-day lag is part of the frozen contract.
        available = (pd.to_datetime(disclosed["filing_date"], errors="coerce")
                     + pd.Timedelta(days=1)) if len(disclosed) else pd.Series(dtype="datetime64[ns]")
        weights = pd.to_numeric(disclosed.get("weight_pct", pd.Series(dtype=float)), errors="coerce")
        valid_full = (len(disclosed) == MEMBERS_PER_DAY and filing.notna().all()
                      and available.notna().all() and filing.lt(key_day).all()
                      and available.le(key_day).all() and weights.gt(0).all()
                      and np.isfinite(weights.to_numpy(float)).all())
        weight_sum = float(weights.sum()) if len(weights) else float("nan")
        accession_hash = hashlib.sha256("|".join(sorted(disclosed.get("accession", pd.Series(dtype=str)).astype(str))).encode()).hexdigest() if len(disclosed) else ""
        stock_returns = {}
        if valid_full:
            day_bars = _bars_for_day(stock_bars, key_day)
            for symbol in disclosed.index:
                part = day_bars.loc[day_bars.instrument.astype(str).str.upper().eq(symbol)]
                ret = _window_return(part, key_day, frequency=5, cutoff="10:30")
                stock_returns[symbol] = float(ret) if np.isfinite(ret) and ret > -1 else np.nan
        basket_known = (valid_full and len(stock_returns) == MEMBERS_PER_DAY
                        and np.isfinite(list(stock_returns.values())).all())
        soxx_day = _bars_for_day(soxx_bars, key_day, "SOXX")
        soxx_first_hour = _window_return(soxx_day, key_day, frequency=1, cutoff="10:30")
        soxx_logret = float(np.log1p(soxx_first_hour)) if np.isfinite(soxx_first_hour) and soxx_first_hour > -1 else np.nan
        residual = (basket_residual(weights, pd.Series(stock_returns), soxx_first_hour)
                    if basket_known and np.isfinite(soxx_first_hour) else np.nan)
        y = _label_from_soxx(soxx_day, key_day)
        feature_known = bool(basket_known and np.isfinite(soxx_logret))
        label_known = bool(np.isfinite(y))
        rows.append({"datetime": key_day, "instrument": "SOXX",
                     "r_soxx_0931_1030": soxx_logret,
                     "basket_minus_soxx_0935_1030": residual if feature_known else np.nan,
                     "y_soxx_1032_1100": y,
                     "basket_member_count": int(len(disclosed)),
                     "basket_weight_sum": weight_sum,
                     "filing_date_max": filing.max() if len(filing) else pd.NaT,
                     "available_date_max": available.max() if len(available) else pd.NaT,
                     "accession_set_sha256": accession_hash,
                     "feature_known": int(feature_known), "label_known": int(label_known),
                     "preentry_soxx_log_return_1031_1033": _label_from_soxx(
                         soxx_day, key_day, entry_minute=31, exit_hour=10, exit_minute=33)})
    result = pd.DataFrame(rows).set_index(["datetime", "instrument"]).sort_index()
    return result


def as_native_dataset(rows: pd.DataFrame):
    """Expose price, label and source clocks through native Qlib DatasetH."""
    frame = rows.copy()
    frame.columns = pd.MultiIndex.from_tuples(
        [("feature", c) for c in FEATURE_COLUMNS]
        + [("label", c) for c in LABEL_COLUMNS]
        + [("metadata", c) for c in METADATA_COLUMNS]
    )
    return native_dataset(frame, segments={"all": (None, None)})


def read_native_daily_dataset(dataset, expected_index: pd.MultiIndex) -> pd.DataFrame:
    """Read the daily source frame back through native Qlib DatasetH."""
    from qlib.data.dataset import DatasetH
    _require(isinstance(dataset, DatasetH), "daily observations must use native DatasetH")
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    _require(isinstance(prepared, pd.DataFrame) and prepared.index.equals(expected_index),
             "daily DatasetH changed or dropped native calendar rows")
    _require(isinstance(prepared.columns, pd.MultiIndex) and prepared.columns.nlevels == 2,
             "daily DatasetH lost grouped source columns")
    prepared = prepared.copy()
    prepared.columns = [str(column[-1]) for column in prepared.columns]
    prepared.attrs["native_qlib_dataset"] = "DatasetH"
    return prepared


def build_native_pit_weight_dataset(holdings: pd.DataFrame, calendar, *, year: int):
    """Build a phase-only PIT weight DatasetH from canonical SEC CSV rows.

    The CSV is metadata-only and may contain later filings.  Its eligibility
    boundary is enforced before calling the established SEC holdings builder.
    The returned panel is read back through DatasetH before it can join prices.
    """
    if year not in {TRAIN_YEAR, EVALUATION_YEAR}:
        raise ValueError("PIT phase year must be 2024 or 2025")
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    _require(dates.is_unique and dates.is_monotonic_increasing,
             "phase PIT calendar is not unique and ordered")
    source = holdings.copy()
    source["filing_date"] = pd.to_datetime(source["filing_date"], errors="raise").dt.normalize()
    # Retain every earlier filing, including 2024 filings in the 2025 phase;
    # there is intentionally no lower bound for either calendar-year slice.
    source = source.loc[source.filing_date < pd.Timestamp(year + 1, 1, 1)].copy()
    from src.data.soxx_pit_holdings import build_pit_soxx_weight_panel
    panel = build_pit_soxx_weight_panel(source, calendar=dates, availability_lag_days=1)
    _require(not panel.empty, "SEC source produced no phase PIT weights")
    if not panel.empty:
        panel["available_date"] = pd.to_datetime(panel.filing_date) + pd.Timedelta(days=1)
        _require((pd.to_datetime(panel.filing_date) < pd.to_datetime(panel.datetime)).all(),
                 "PIT panel contains a filing unavailable before the session")
        _require((pd.to_datetime(panel.available_date) <= pd.to_datetime(panel.datetime)).all(),
                 "PIT panel violates the fixed one-day availability lag")
    frame = panel.set_index(["datetime", "instrument"])[
        ["weight_pct", "filing_date", "accession", "available_date", "report_date"]
    ].sort_index()
    grouped = pd.concat({"feature": frame[["weight_pct"]],
                         "metadata": frame[["filing_date", "accession", "available_date", "report_date"]]}, axis=1)
    dataset = native_dataset(grouped, segments={"all": (dates[0], dates[-1])})
    # Qlib DatasetH/DataHandlerLP is the downstream metadata authority.
    prepared = dataset.prepare("all", col_set=["feature", "metadata"])
    _require(isinstance(prepared, pd.DataFrame), "PIT DatasetH readback is not a dataframe")
    readback = prepared.copy()
    _require(readback.index.equals(grouped.index), "PIT DatasetH changed the native membership rowset")
    _require(set(readback.columns.get_level_values(0)) == {"feature", "metadata"},
             "PIT DatasetH lost feature or filing metadata")
    readback.columns = [column[-1] for column in readback.columns]
    return dataset, readback


def pinned_source_hashes(plan: Mapping) -> dict[str, str]:
    """Collect the exact plan-pinned market and PIT source bytes."""
    data = plan["data"]
    hashes = {}
    for snapshot_name in ("basket_prices_5min", "soxx_prices_1min"):
        hashes.update(data["present_day_file_snapshot"][snapshot_name]["files_sha256"])
    pit = data["pit_weights"]
    hashes[pit["source_csv"]] = pit["source_csv_sha256"]
    hashes[pit["source_manifest"]] = pit["source_manifest_sha256"]
    for relative, entry in plan.get("source_code", {}).items():
        if isinstance(entry, Mapping) and entry.get("sha256"):
            hashes[relative] = entry["sha256"]
    return hashes


def read_native_phase(root: str | Path, plan: Mapping, *, year: int,
                      attempt_journal: str | Path) -> pd.DataFrame:
    """Read one bounded year using the two frozen Qlib price providers.

    It performs three provider-native calendar requests, one instrument
    metadata request per price store and exactly two bounded ``D.features``
    calls. Call only for 2024 first; the caller must seal its freeze receipt
    before invoking this function for 2025.
    """
    if year not in {TRAIN_YEAR, EVALUATION_YEAR}:
        raise ValueError("phase year must be 2024 or 2025")
    root = Path(root).resolve()
    journal = Path(attempt_journal)
    _require(journal.is_file(), "durable attempt journal must precede each source read")
    expected = pinned_source_hashes(plan)
    _tracked(journal, year, "source_hashes_before", lambda: verify_source_snapshot(root, expected))
    _no_active_writable_recorder()
    start, end = f"{year}-01-01", f"{year}-12-31 23:59:59"
    import qlib
    from qlib.data import D

    def provider(uri: str):
        qlib.init(provider_uri=str((root / uri).resolve()), region="us", kernels=1)

    # One daily calendar for PIT alignment, with no access to daily price bins.
    daily_provider = Path(plan["data"]["pit_weights"]["source_csv"]).parent
    provider(daily_provider)
    daily_calendar = pd.DatetimeIndex(_tracked(journal, year, "qlib_calendar_day",
        lambda: D.calendar(start_time=start, end_time=end, freq="day")))
    _require(not daily_calendar.empty and (daily_calendar.year == year).all(),
             "daily PIT alignment calendar is empty or escaped its phase")
    # Read the SEC metadata CSV only after exact byte verification; never use
    # future filings to select a phase's point-in-time basket.
    holdings_path = root / plan["data"]["pit_weights"]["source_csv"]
    holdings = _tracked(journal, year, "sec_metadata_csv", lambda: pd.read_csv(holdings_path))
    weight_dataset, weight_panel = _tracked(journal, year, "pit_weight_dataset",
        lambda: build_native_pit_weight_dataset(holdings, daily_calendar, year=year))

    stock_spec = plan["data"]["provider_access"]["prices_5min"]
    provider(stock_spec["uri"])
    stock_calendar = pd.DatetimeIndex(_tracked(journal, year, "qlib_calendar_5min",
        lambda: D.calendar(start_time=start, end_time=end, freq="5min")))
    _require(not stock_calendar.empty and (stock_calendar.year == year).all(),
             "5-minute native calendar is empty or escaped its phase")
    stock_pool = D.instruments("all")
    stock_universe = _tracked(journal, year, "qlib_instruments_5min",
        lambda: D.list_instruments(stock_pool, start_time=start, end_time=end,
                                   freq="5min", as_list=True))
    _require(isinstance(stock_universe, list), "5-minute Qlib instrument list has unexpected shape")
    universe = {str(symbol).upper(): symbol for symbol in stock_universe}
    requested = sorted(plan["data"]["basket_provider"]["instruments"])
    active = set(weight_panel.index.get_level_values("instrument").str.upper())
    _require(set(universe).issubset(set(requested)), "unexpected 5-minute source instrument")
    _require(active.issubset(set(universe)), "5-minute provider lacks a phase PIT-active member")
    stock_symbols = requested  # frozen historical union, including absent-year names
    stock_raw = _tracked(journal, year, "qlib_features_5min",
        lambda: D.features(stock_symbols, stock_spec["fields"], start_time=start, end_time=end, freq="5min"))
    stock_bars = _normalize_intraday_features(stock_raw)

    etf_spec = plan["data"]["provider_access"]["soxx_prices_1min"]
    provider(etf_spec["uri"])
    etf_calendar = pd.DatetimeIndex(_tracked(journal, year, "qlib_calendar_1min",
        lambda: D.calendar(start_time=start, end_time=end, freq="1min")))
    _require(not etf_calendar.empty and (etf_calendar.year == year).all(),
             "1-minute SOXX calendar is empty or escaped its phase")
    etf_pool = D.instruments("all")
    etf_universe = _tracked(journal, year, "qlib_instruments_1min",
        lambda: D.list_instruments(etf_pool, start_time=start, end_time=end,
                                   freq="1min", as_list=True))
    _require(isinstance(etf_universe, list) and "SOXX" in {str(s).upper() for s in etf_universe},
             "1-minute core provider lacks SOXX instrument metadata")
    etf_symbol = next(symbol for symbol in etf_universe if str(symbol).upper() == "SOXX")
    etf_raw = _tracked(journal, year, "qlib_features_1min",
        lambda: D.features([etf_symbol], etf_spec["fields"], start_time=start, end_time=end, freq="1min"))
    soxx_bars = _normalize_intraday_features(etf_raw)

    _require(stock_calendar.normalize().unique().sort_values().equals(
             etf_calendar.normalize().unique().sort_values()),
             "minute providers have different session dates in this phase")
    _require(daily_calendar.equals(stock_calendar.normalize().unique().sort_values()),
             "Qlib daily calendar differs from the minute provider session dates")
    _require((stock_bars.index.get_level_values("datetime").year == year).all()
             and (soxx_bars.index.get_level_values("datetime").year == year).all(),
             "price rows escaped their phase bounds")
    rows = build_daily_rows(daily_calendar, stock_bars, soxx_bars, weight_panel)
    dataset = _tracked(journal, year, "price_label_dataset", lambda: as_native_dataset(rows))
    prepared = read_native_daily_dataset(dataset, rows.index)
    # The exact source bytes are checked again after all source reads.
    after_hashes = _tracked(journal, year, "source_hashes_after", lambda: verify_source_snapshot(root, expected))
    prepared.attrs["phase_source_counts"] = {"calendar_calls": 3, "instrument_lookups": 2,
        "features_calls": 2, "native_dataset_builds": 2, "phase_price_instruments": sorted(universe),
        "phase_pit_instruments": sorted(active)}
    prepared.attrs["source_sha256"] = after_hashes
    prepared.attrs["native_dataset_object"] = dataset
    prepared.attrs["pit_weight_dataset_object"] = weight_dataset
    prepared.attrs["phase_calendar_sha256"] = hashlib.sha256(pd.DatetimeIndex(daily_calendar).asi8.tobytes()).hexdigest()
    return prepared


@dataclass(frozen=True)
class FixedOLS:
    arm: str
    features: tuple[str, ...]
    training_mean: tuple[float, ...]
    training_std: tuple[float, ...]
    coefficients: tuple[float, ...]  # intercept followed by standardized-feature coefficients
    training_rows: int
    training_frame_sha256: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.features].to_numpy(dtype=np.float64)
        z = (x - np.asarray(self.training_mean)) / np.asarray(self.training_std)
        return np.column_stack([np.ones(len(z)), z]) @ np.asarray(self.coefficients)

    @property
    def gamma(self) -> float | None:
        if self.arm != "C":
            return None
        return float(self.coefficients[-1] / self.training_std[-1])


def fit_fixed_ols(training: pd.DataFrame, *, arm: str) -> FixedOLS:
    """Fit one ordinary least-squares head using 2024 eligible rows only."""
    if arm not in {"B", "C"}:
        raise ValueError("arm must be B or C")
    if not isinstance(training.index, pd.MultiIndex) or list(training.index.names) != ["datetime", "instrument"]:
        raise ValueError("training must be a native DatasetH-derived daily frame")
    years = pd.DatetimeIndex(training.index.get_level_values("datetime")).year
    _require(len(years) > 0 and np.all(years == TRAIN_YEAR), "model fit must contain 2024 rows only")
    features = (FEATURE_COLUMNS[0],) if arm == "B" else FEATURE_COLUMNS
    cols = [*features, LABEL_COLUMNS[0]]
    arr = training.loc[:, cols].to_numpy(dtype=np.float64)
    _require(np.isfinite(arr).all(), "eligible training rows contain non-finite values")
    x, y = arr[:, :-1], arr[:, -1]
    _require(len(x) >= MINIMUM_ELIGIBLE_DAYS, "fewer than 200 eligible training sessions")
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    _require(np.isfinite(std).all() and (std > 0).all(), "training scaler has zero or invalid scale")
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    _require(rank == design.shape[1], "fixed OLS design is rank deficient")
    canonical = training.loc[:, cols].to_numpy(dtype="<f8").tobytes(order="C")
    return FixedOLS(arm=arm, features=tuple(features), training_mean=tuple(map(float, mean)),
                    training_std=tuple(map(float, std)), coefficients=tuple(map(float, coef)),
                    training_rows=len(training), training_frame_sha256=hashlib.sha256(canonical).hexdigest())


def fit_frozen_models(training: pd.DataFrame) -> dict[str, FixedOLS]:
    """Fit and return B/C once; validates equal eligible dates by construction."""
    _require(training.index.is_unique and training.index.is_monotonic_increasing, "training rows must be ordered and unique")
    models = {arm: fit_fixed_ols(training, arm=arm) for arm in ("B", "C")}
    return models


def model_receipt(models: Mapping[str, FixedOLS], *, plan_sha256: str, phase_rows_sha256: str) -> dict:
    _require(set(models) == {"B", "C"}, "freeze receipt requires exactly models B and C")
    return {"experiment": EXPERIMENT, "status": "sealed_2024_models",
            "plan_sha256": plan_sha256, "phase_rows_sha256": phase_rows_sha256,
            "train_year": TRAIN_YEAR, "evaluation_year_not_read": EVALUATION_YEAR,
            "models": {name: asdict(model) for name, model in models.items()},
            "fits": 2, "hyperparameter_search": False}


def paired_evaluation(test: pd.DataFrame, models: Mapping[str, FixedOLS], *, bootstrap: Mapping = BOOTSTRAP) -> dict:
    """Evaluate identical known 2025 rows and the single registered paired test."""
    _require(set(models) == {"B", "C"}, "evaluation requires both frozen models")
    years = pd.DatetimeIndex(test.index.get_level_values("datetime")).year
    _require(len(years) > 0 and np.all(years == EVALUATION_YEAR), "evaluation must contain 2025 rows only")
    cols = [*FEATURE_COLUMNS, LABEL_COLUMNS[0]]
    complete = pd.Series(np.isfinite(test.loc[:, cols].to_numpy(float)).all(axis=1), index=test.index)
    if {"feature_known", "label_known"}.issubset(test.columns):
        complete &= test.feature_known.eq(1) & test.label_known.eq(1)
    sample = test.loc[complete]
    _require(len(sample) >= MINIMUM_ELIGIBLE_DAYS, "fewer than 200 eligible evaluation sessions")
    y = sample[LABEL_COLUMNS[0]].to_numpy(float)
    pred_b, pred_c = models["B"].predict(sample), models["C"].predict(sample)
    delta = np.square(y - pred_b) - np.square(y - pred_c)
    indices = stationary_block_bootstrap_indices(observation_count=len(delta), path_length=len(delta),
        path_count=int(bootstrap["replicates"]), expected_block_length=int(bootstrap["mean_block_length"]),
        seed=int(bootstrap["seed"]))
    draws = delta[indices].mean(axis=1)
    lower = float(np.quantile(draws, float(bootstrap["lower_quantile"]), method="linear"))
    gamma = float(models["C"].gamma)
    corr = float(np.corrcoef(gamma * sample[FEATURE_COLUMNS[1]].to_numpy(float), y - pred_b)[0, 1])
    preentry_correlation = None
    diagnostic_name = "preentry_soxx_log_return_1031_1033"
    if diagnostic_name in sample:
        lag = sample[diagnostic_name].to_numpy(float)
        contribution = gamma * sample[FEATURE_COLUMNS[1]].to_numpy(float)
        known = np.isfinite(lag) & np.isfinite(contribution)
        if known.sum() > 1 and np.std(lag[known]) > 0 and np.std(contribution[known]) > 0:
            preentry_correlation = float(np.corrcoef(contribution[known], lag[known])[0, 1])
    mean_delta = float(delta.mean())
    passed = bool(gamma > 0 and mean_delta > 0 and lower > 0)
    return {"eligible_rows": len(sample), "excluded_calendar_rows": int(len(test) - len(sample)),
            "gamma": gamma, "mean_paired_loss_improvement": mean_delta,
            "stationary_bootstrap_lower": lower,
            "stationary_bootstrap_quantile": float(bootstrap["lower_quantile"]),
            "stationary_bootstrap_draws": draws,
            "predictions_and_losses": pd.DataFrame({"y": y, "prediction_B": pred_b,
                "prediction_C": pred_c, "paired_loss_improvement": delta}, index=sample.index),
            "pearson_gamma_x_vs_baseline_residual_diagnostic": corr if np.isfinite(corr) else None,
            "pearson_gamma_x_vs_preentry_return_diagnostic": preentry_correlation,
            "pass": passed, "claim": "reused_history_information_screen_only"}


def run_two_phase(read_phase: Callable, seal_phase: Callable, *, plan: dict,
                  plan_sha256: str) -> dict:
    """Enforce the 2024 seal barrier before the callback may read 2025 inputs.

    ``read_phase(year)`` must make only that phase's bounded Qlib reads and
    return a DatasetH-derived frame. ``seal_phase`` must durably save the model
    receipt in a newly created, then finished Qlib Recorder before returning.
    """
    _require(plan.get("experiment") == EXPERIMENT, "plan identity differs")
    _require(plan.get("status") == "static_plan_only_not_executed", "plan is not the approved static plan")
    train = read_phase(TRAIN_YEAR)
    _require(isinstance(train, pd.DataFrame), "2024 phase reader must return DatasetH-derived frame")
    _require(train.attrs.get("native_qlib_dataset") == "DatasetH",
             "2024 fit input must be read back from native Qlib DatasetH")
    eligible = train.loc[(train.feature_known == 1) & (train.label_known == 1),
                         [*FEATURE_COLUMNS, *LABEL_COLUMNS]]
    models = fit_frozen_models(eligible)
    row_hash = hashlib.sha256(eligible.to_numpy(dtype="<f8").tobytes(order="C")).hexdigest()
    receipt = model_receipt(models, plan_sha256=plan_sha256, phase_rows_sha256=row_hash)
    seal_phase(receipt, models, eligible)
    # This is the first point at which the caller can perform any 2025 read.
    evaluation = read_phase(EVALUATION_YEAR)
    _require(isinstance(evaluation, pd.DataFrame)
             and evaluation.attrs.get("native_qlib_dataset") == "DatasetH",
             "2025 evaluation must be read back from native Qlib DatasetH")
    result = paired_evaluation(evaluation, models)
    return {"freeze_receipt": receipt, "evaluation": result, "fit_count": 2,
            "refit_count": 0, "development_data_are_reused_history": True}


def encode_models(models: Mapping[str, FixedOLS]) -> dict:
    return {name: asdict(model) for name, model in models.items()}


def verify_source_snapshot(root: str | Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Hash exact pinned bytes without opening Qlib feature arrays."""
    root = Path(root)
    actual = {}
    for relative, digest in sorted(expected.items()):
        path = root / relative
        _require(path.is_file(), f"pinned source file is missing: {relative}")
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual[relative] == digest, f"pinned source bytes differ: {relative}")
    return actual


def _no_active_writable_recorder() -> None:
    from qlib.workflow import R
    if R.__dict__.get("_provider") is None:
        return
    active = R.exp_manager.active_experiment
    _require(active is None or active.active_recorder is None,
             "source reads require no active writable Qlib Recorder")


def current_implementation_identity(root: str | Path) -> dict:
    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    return {"implementation_sha256": _sha(root / "src/workflow/basket_etf_lead_information.py"),
            "notebook_sha256": normalized_notebook_identity(card / (EXPERIMENT + ".ipynb")),
            "test_sha256": _sha(root / "tests/test_basket_etf_lead_information.py"),
            "implementation_dependencies_sha256": implementation_dependencies(root)}


def _transport(root: Path, card: Path) -> Path:
    # Transport configuration only: no pre-existing Recorder or archive is read.
    inherited = json.loads((root / "research/constituent_intraday_history_20261014/nas_binding.json").read_text())
    binding = {"schema_version": "recorder_evidence_binding_v1", "authority": inherited["authority"],
               "ssh": inherited["ssh"], "source_id": EXPERIMENT, "source_plan": "plan.json"}
    _atomic_json(card / "nas_binding.json", binding)
    return card / "nas_binding.json"


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame.attrs.get("native_qlib_dataset") == "DatasetH", "fit/evaluation source is not native DatasetH")
    mask = ((frame.feature_known == 1) & (frame.label_known == 1)
            & np.isfinite(frame.loc[:, [*FEATURE_COLUMNS, *LABEL_COLUMNS]].to_numpy(float)).all(axis=1))
    return frame.loc[mask, [*FEATURE_COLUMNS, *LABEL_COLUMNS]]


def _phase_audit(frame: pd.DataFrame) -> dict:
    return {"calendar_rows": len(frame), "qualified_rows": len(_eligible(frame)),
            "unknown_feature_rows": int(frame.feature_known.eq(0).sum()),
            "unknown_label_rows": int(frame.label_known.eq(0).sum()),
            "counts": frame.attrs["phase_source_counts"],
            "source_sha256": frame.attrs["source_sha256"],
            "calendar_sha256": frame.attrs["phase_calendar_sha256"]}


def _receipt_digest(receipt: Mapping) -> str:
    return hashlib.sha256(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run(*, root: str | Path) -> dict:
    """Single authorized attempt; finish/read back the 2024 seal before 2025 IO.

    The owner writes execution_authorization.json after independent static review.
    The notebook is executable without edits: absence of this exact GO raises
    before journal creation, source-byte hashing, feature IO or a writable run.
    """
    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    state_path = card / "run_state.json"
    go_path = card / "execution_authorization.json"
    _require(go_path.is_file(), "execution_authorization.json is required")
    go = json.loads(go_path.read_text())
    authorization = validate_execution_go(root, go, state_path=state_path)
    plan = json.loads((card / "plan.json").read_text())
    initial_state = dict(status="authorized_single_attempt", authorization=authorization,
             authorization_sha256=_sha(go_path), operations=[], fits_completed=[],
             fit_attempts=0, bootstrap_attempts=0, bootstrap_completed=0,
             accounts=0, automatic_resume_forbidden=True)
    # Exclusive creation prevents concurrent invocations from sharing a budget.
    with state_path.open("x", encoding="utf-8") as stream:
        json.dump(initial_state, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    rec = None
    try:
        _no_active_writable_recorder()
        train = read_native_phase(root, plan, year=TRAIN_YEAR, attempt_journal=state_path)
        train_audit = _phase_audit(train)
        _journal(state_path, status="2024_sources_read", training=train_audit)
        training = _eligible(train)
        _require(len(training) >= MINIMUM_ELIGIBLE_DAYS, "fewer than 200 eligible training sessions")
        # Source reads and identity checks above precede the first writable run.
        with R.start(experiment_name=EXPERIMENT, recorder_name="single_fixed_information_screen",
                     uri=str(card / "qlib_recorder")):
            rec = R.get_recorder()
            _journal(state_path, status="2024_fitting", recorder_id=str(rec.id),
                     experiment_id=str(rec.experiment_id))
            rec.save_objects(artifact_path="inputs", plan=plan, authorization=go,
                             training_dataset=train.attrs["native_dataset_object"],
                             pit_weight_dataset=train.attrs["pit_weight_dataset_object"], training_source_audit=train_audit)
            code_paths = ["src/workflow/basket_etf_lead_information.py",
                          "tests/test_basket_etf_lead_information.py", *IMPLEMENTATION_DEPENDENCIES,
                          *plan.get("source_code", {}).keys()]
            snapshot = {name: (root / name).read_bytes() for name in code_paths}
            notebook = json.loads((card / (EXPERIMENT + ".ipynb")).read_text())
            normalized = [{"cell_type": c["cell_type"], "source": "".join(c.get("source", []))}
                          for c in notebook["cells"] if c.get("metadata", {}).get("role") != "conclusion"]
            rec.save_objects(artifact_path="sources", implementation_snapshot=snapshot,
                             normalized_notebook=normalized, implementation_identity=authorization)
            models = {}
            for arm in ("B", "C"):
                _journal(state_path, status="2024_fitting", fit_attempts=len(models) + 1)
                def fit_and_save():
                    model = fit_fixed_ols(training, arm=arm)
                    # Persist each completed fit immediately; a later failure never erases it.
                    rec.save_objects(artifact_path="models", **{arm: asdict(model)})
                    return model
                models[arm] = _tracked(state_path, TRAIN_YEAR, "ols_fit_" + arm, fit_and_save)
                _journal(state_path, status="2024_fitting", fits_completed=list(models),
                         model_sha256={name: _receipt_digest(asdict(model)) for name, model in models.items()})
            row_hash = hashlib.sha256(training.to_numpy(dtype="<f8").tobytes(order="C")).hexdigest()
            receipt = model_receipt(models, plan_sha256=authorization["plan_sha256"], phase_rows_sha256=row_hash)
            receipt.update(training_source_audit=train_audit, implementation_identity=authorization)
            seal_hash = _receipt_digest(receipt)
            rec.save_objects(artifact_path="freeze", receipt=receipt, receipt_sha256=seal_hash,
                             inference_configuration=encode_models(models))
            rec.log_metrics(training_qualified_rows=len(training), information_fits=2)
        _require(rec.status == Recorder.STATUS_FI, "2024 Recorder did not finish before freeze readback")
        _no_active_writable_recorder()
        readback = rec.load_object("freeze/receipt")
        _require(_receipt_digest(readback) == seal_hash, "2024 freeze receipt failed readback")
        for arm in ("B", "C"):
            _require(rec.load_object("models/" + arm) == asdict(models[arm]), "frozen model readback differs")
        _journal(state_path, status="2024_sealed_readback_verified", freeze_receipt_sha256=seal_hash)
        # No 2025 price, label or PIT Dataset construction is reachable before this barrier.
        evaluation = read_native_phase(root, plan, year=EVALUATION_YEAR, attempt_journal=state_path)
        evaluation_audit = _phase_audit(evaluation)
        _journal(state_path, status="2025_sources_read", evaluation=evaluation_audit)
        test = _eligible(evaluation)
        _require(len(test) >= MINIMUM_ELIGIBLE_DAYS, "fewer than 200 eligible evaluation sessions")
        rec.start_run()
        rec.save_objects(artifact_path="evaluation", dataset=evaluation.attrs["native_dataset_object"],
                         pit_weight_dataset=evaluation.attrs["pit_weight_dataset_object"], source_audit=evaluation_audit)
        _journal(state_path, status="2025_bootstrap", bootstrap_attempts=1)
        result = _tracked(state_path, EVALUATION_YEAR, "paired_stationary_bootstrap",
                          lambda: paired_evaluation(evaluation, models))
        _journal(state_path, status="2025_evaluated", bootstrap_completed=1)
        rec.save_objects(artifact_path="evaluation", result=result)
        scalar = {key: value for key, value in result.items()
                  if key not in {"stationary_bootstrap_draws", "predictions_and_losses"}}
        metrics = {key: float(value) for key, value in scalar.items()
                   if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
        rec.log_metrics(**metrics)
        source_identity = _tracked(state_path, EVALUATION_YEAR, "final_source_hashes",
                                  lambda: verify_source_snapshot(root, pinned_source_hashes(plan)))
        state = json.loads(state_path.read_text())
        summary = {"experiment": EXPERIMENT, "decision": "information_screen_pass" if result["pass"] else "not_promoted",
            "promotion_decision": "not_promoted", "claim": "reused_history_information_screen_only",
            "independent_validation": False, "fee_net_performance_claim": False, "production_changed": False,
            "plan_sha256": authorization["plan_sha256"], "implementation_identity": authorization,
            "freeze_receipt_sha256": seal_hash, "information_screen": scalar,
            "training": train_audit, "evaluation": evaluation_audit, "actual_counts": state,
            "new_information_fits": 2, "new_bootstraps": 1, "new_native_accounts": 0,
            "source_snapshot_sha256": source_identity, "workflow_config": "workflow.yaml",
            "recorder_uri": str(card / "qlib_recorder"), "region": "us",
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/training_dataset", "models", "freeze", "sources",
                                             "evaluation/dataset", "evaluation/result"]},
            "source_limits": plan["qualification_gaps"]}
        rec.save_objects(artifact_path="analysis", summary=summary)
        rec.end_run(status=Recorder.STATUS_FI)
        summary = finalize_evidence(rec, card=card, summary=summary,
                                   transport_binding=_transport(root, card), workspace_root=root)
        _journal(state_path, status="completed", recorder_id=str(rec.id), decision=summary["decision"])
        return summary
    except BaseException as error:
        _journal(state_path, status="failed_single_attempt", failure={"type": type(error).__name__, "message": str(error)})
        if rec is not None:
            try:
                if rec.status != Recorder.STATUS_R:
                    rec.start_run()
                state = json.loads(state_path.read_text())
                rec.save_objects(artifact_path="execution", failure=state)
                rec.log_metrics(actual_fit_attempts=state.get("fit_attempts", 0),
                                actual_completed_fits=len(state.get("fits_completed", [])),
                                actual_bootstrap_attempts=state.get("bootstrap_attempts", 0))
                rec.end_run(status=Recorder.STATUS_FA)
                from src.data.nas_market_data import archive_nas_recorder
                evidence = archive_nas_recorder(_transport(root, card), workspace_root=root, card=card,
                    recorder_id=str(rec.id), experiment_id=str(rec.experiment_id))
                _journal(state_path, status="failed_single_attempt", failure_nas_evidence=evidence)
            except BaseException as archive_error:
                _journal(state_path, status="failed_single_attempt", failure_archive_error={
                    "type": type(archive_error).__name__, "message": str(archive_error)})
        raise
