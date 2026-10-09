"""One declared affine price basis from observed whole bars and published events.

This is deliberately not labelled SDK QFQ: rounded published event coefficients
need not reproduce server QFQ. No coefficients are fitted to prices or returns.
Yahoo quote OHLC is split-only; its adjclose field is never used. All actual
snapshot coverage and source archive gates remain in price_basis/the notebook.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.price_basis import (
    FIELDS, INSTRUMENTS, PUBLISHED_BASIS, PriceBasisError, _date, _row_payload,
    _session_date, _sha, normalize_futu_candles, price_frame_sha256,
)


def _events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ex_div_date", "forward_adj_factorA", "forward_adj_factorB"}
    if not required.issubset(frame):
        raise PriceBasisError(f"rehabilitation columns missing: {sorted(required - set(frame))}")
    events = frame.copy()
    events["event_date"] = events["ex_div_date"].map(_session_date)
    if events.event_date.isna().any() or events.event_date.duplicated().any():
        raise PriceBasisError("rehabilitation dates must be valid unique daily events")
    for column in ("forward_adj_factorA", "forward_adj_factorB"):
        events[column] = pd.to_numeric(events[column], errors="coerce")
    if (not np.isfinite(events[["forward_adj_factorA", "forward_adj_factorB"]].to_numpy(float)).all()
            or (events.forward_adj_factorA <= 0).any()):
        raise PriceBasisError("rehabilitation affine A must be positive; A/B must be finite")
    ratios = []
    for row in events.to_dict("records"):
        found = []
        for prefix in ("split", "join"):
            numerator, denominator = row.get(f"{prefix}_base", np.nan), row.get(f"{prefix}_ert", np.nan)
            if pd.isna(numerator) and pd.isna(denominator):
                continue
            try:
                values = np.array([numerator, denominator], dtype=float)
            except (ValueError, TypeError) as error:
                raise PriceBasisError("invalid split/join share counts") from error
            if not np.isfinite(values).all() or (values <= 0).any():
                raise PriceBasisError("exact split/join numerator and denominator must both be positive finite")
            found.append(float(values[0] / values[1]))
        if len(found) > 1:
            raise PriceBasisError("ambiguous simultaneous split and join event")
        # An approximate split_ratio alone cannot invent exact share counts.
        if not found and pd.notna(row.get("split_ratio", np.nan)):
            if float(row["split_ratio"]) != 1.0:
                raise PriceBasisError("split ratio lacks exact published base/ert counts")
        for prefix in ("bonus", "transfer", "allot", "add", "spin_off"):
            if pd.notna(row.get(f"{prefix}_base", np.nan)) or pd.notna(row.get(f"{prefix}_ert", np.nan)):
                raise PriceBasisError("unsupported non-split share-count event requires a separate volume contract")
        ratios.append(found[0] if found else 1.0)
    events["exact_split_price_factor"] = ratios
    return events.sort_values("event_date").reset_index(drop=True)


def _published_identity(identity: Mapping[str, Any], source: str, original: str) -> dict[str, Any]:
    result = dict(identity)
    if result.get("source") != source or result.get("adjustment") != original:
        raise PriceBasisError(f"{source} input must declare {original} source OHLC")
    if result.get("original_adjustment", original) != original:
        raise PriceBasisError("original source adjustment declaration mismatch")
    raw_volume_basis = "raw_shares" if source == "Futu" else "split_adjusted_shares"
    for key, expected in (("volume_unit", "shares"), ("volume_basis", raw_volume_basis),
                          ("original_volume_basis", raw_volume_basis)):
        if key in result and result[key] != expected:
            raise PriceBasisError(f"{source} {key} contradicts observed {original} source contract: expected {expected}")
    result.update(
        adjustment=PUBLISHED_BASIS, original_adjustment=original, basis_adapter=PUBLISHED_BASIS,
        volume_unit="shares", volume_basis="split_adjusted_shares",
        original_volume_basis=raw_volume_basis,
        original_source_identity_sha256=_sha(dict(identity)),
    )
    return result


def _normalize_yahoo(raw: pd.DataFrame, identity: Mapping[str, Any]) -> pd.DataFrame:
    required = {"date", *FIELDS}
    if not required.issubset(raw):
        raise PriceBasisError(f"Yahoo raw columns missing: {sorted(required - set(raw))}")
    source = raw.reset_index(drop=True)
    frame = pd.DataFrame(index=source.index)
    if "instrument" in source:
        frame["instrument"] = source["instrument"].astype(str)
    elif identity.get("instrument") in INSTRUMENTS:
        frame["instrument"] = identity["instrument"]
    else:
        raise PriceBasisError("Yahoo rows need instrument column or single instrument identity")
    frame["datetime"] = pd.to_datetime(source["date"].map(_session_date))
    for field in FIELDS:
        frame[field] = pd.to_numeric(source[field], errors="coerce").astype(float)
    frame["source_identity_sha256"] = _sha(identity)
    frame["source_row_number"] = np.arange(len(source), dtype=np.int64)
    frame["source_code"] = frame["instrument"]
    frame["source_time_key"] = source["date"].astype(str)
    frame["source_raw_row_sha256"] = [_sha(row) for row in source.to_dict("records")]
    frame["normalized_row_sha256"] = [_sha(_row_payload(row)) for row in frame.to_dict("records")]
    return frame


def build_published_basis(
    futu_none_raw: pd.DataFrame,
    yahoo_split_raw: pd.DataFrame,
    rehab_by_instrument: Mapping[str, pd.DataFrame],
    primary_identity: Mapping[str, Any],
    secondary_identity: Mapping[str, Any],
    as_of: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one chronological affine operator to every observed OHLC bar.

    Secondary input must contain gap rows only, never replacement/neighbor rows.
    No future market rows are truncated. Future event coefficients are excluded
    and disclosed; future splits block Yahoo conversion because quote splits are
    referenced to the retrieval vintage, not necessarily the requested as_of.
    Invalid observed OHLCV aborts the transform before an affine operator could
    conceal a source error. Final transformed bars still require qualification.
    """
    cutoff = _date(as_of)
    primary_id = _published_identity(primary_identity, "Futu", "NONE")
    secondary_id = _published_identity(secondary_identity, "Yahoo", "SPLIT_ONLY")
    primary = normalize_futu_candles(futu_none_raw, primary_id)
    secondary = _normalize_yahoo(yahoo_split_raw, secondary_id)
    for frame in (primary, secondary):
        if frame.datetime.isna().any() or not frame.instrument.isin(INSTRUMENTS).all():
            raise PriceBasisError("source bar has invalid instrument or daily session date")
        if (frame.datetime > cutoff).any():
            raise PriceBasisError("source market rows after as_of cannot be truncated")
        if frame.duplicated(["instrument", "datetime"]).any():
            raise PriceBasisError("duplicate source bar cannot be deduplicated")
        values = frame[list(FIELDS)].to_numpy(float)
        finite_positive = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        physical = ((values[:, 2] <= values[:, 0]) & (values[:, 0] <= values[:, 1])
                    & (values[:, 2] <= values[:, 3]) & (values[:, 3] <= values[:, 1]))
        invalid = np.flatnonzero(~(finite_positive & physical))
        if len(invalid):
            raise PriceBasisError(f"invalid observed source OHLCV before affine transform at row positions {invalid.tolist()}")
    primary_keys = pd.MultiIndex.from_frame(primary[["instrument", "datetime"]])
    secondary_keys = pd.MultiIndex.from_frame(secondary[["instrument", "datetime"]])
    overlap = primary_keys.intersection(secondary_keys)
    if len(overlap):
        raise PriceBasisError(f"secondary source cannot replace available primary bars: {overlap.tolist()}")
    events = {}
    for symbol in INSTRUMENTS:
        if symbol not in rehab_by_instrument:
            raise PriceBasisError(f"missing rehabilitation source: {symbol}")
        events[symbol] = _events(rehab_by_instrument[symbol])
    if len(secondary):
        for symbol in secondary.instrument.unique():
            future = events[symbol].loc[events[symbol].event_date > cutoff]
            if future.exact_split_price_factor.ne(1.0).any():
                raise PriceBasisError("post-as_of split makes Yahoo quote vintage incompatible with this undo contract")
    original_attrs = primary.attrs
    primary.attrs = {}
    combined = pd.concat([primary, secondary], ignore_index=True)
    for column in ("affine_A", "affine_B", "split_price_factor"):
        combined[column] = np.nan
    combined["basis"] = PUBLISHED_BASIS
    combined["original_adjustment"] = ["NONE"] * len(primary) + ["SPLIT_ONLY"] * len(secondary)
    combined["rehab_event_count"] = 0
    combined["rehab_frame_sha256"] = ""
    for position, row in combined.iterrows():
        symbol = row.instrument
        applicable = events[symbol].loc[(events[symbol].event_date > row.datetime) & (events[symbol].event_date <= cutoff)]
        factor_a, factor_b, split_factor = 1.0, 0.0, 1.0
        for event in applicable.itertuples(index=False):
            factor_a *= event.forward_adj_factorA
            factor_b = factor_b * event.forward_adj_factorA + event.forward_adj_factorB
            split_factor *= event.exact_split_price_factor
        if not np.isfinite([factor_a, factor_b, split_factor]).all() or factor_a <= 0 or split_factor <= 0:
            raise PriceBasisError("composed affine/split operator is not positive finite")
        quoted_prices = row[list(FIELDS[:-1])].to_numpy(float)
        unadjusted_prices = quoted_prices if row.original_adjustment == "NONE" else quoted_prices / split_factor
        combined.loc[position, list(FIELDS[:-1])] = unadjusted_prices * factor_a + factor_b
        # Yahoo volume is already split-adjusted: undo then reapply cancels
        # exactly, so preserve the observed volume without roundtrip rounding.
        combined.loc[position, "volume"] = row.volume / split_factor if row.original_adjustment == "NONE" else row.volume
        combined.loc[position, ["affine_A", "affine_B", "split_price_factor"]] = [factor_a, factor_b, split_factor]
        combined.loc[position, "rehab_event_count"] = len(applicable)
        combined.loc[position, "rehab_frame_sha256"] = price_frame_sha256(rehab_by_instrument[symbol])
    combined["normalized_row_sha256"] = [_sha(_row_payload(row)) for row in combined.to_dict("records")]
    combined.attrs["source_identities"] = {_sha(primary_id): primary_id, _sha(secondary_id): secondary_id}
    combined.attrs["raw_frame_sha256"] = {
        "Futu": original_attrs["raw_frame_sha256"], "Yahoo": price_frame_sha256(yahoo_split_raw),
    }
    audit = {
        "basis": PUBLISHED_BASIS, "as_of": as_of, "market_fits_or_pnl_used": False,
        "primary_rows": len(primary), "secondary_whole_bar_rows": len(secondary), "total_rows": len(combined),
        "primary_secondary_overlap_rows": 0, "input_rows_discarded": 0,
        "frame_sha256": price_frame_sha256(combined),
        "raw_frame_sha256": combined.attrs["raw_frame_sha256"],
        "rehab_frame_sha256": {symbol: price_frame_sha256(raw) for symbol, raw in rehab_by_instrument.items()},
        "excluded_post_asof_event_dates": {symbol: table.loc[table.event_date > cutoff, "ex_div_date"].tolist() for symbol, table in events.items()},
        "price_operator": "Chronological future events: A<-A*a; B<-B*a+b; price<-unadjusted_price*A+B",
        "volume_operator": "Futu NONE volume / product(exact split_base/split_ert or join_base/join_ert); Yahoo split-adjusted volume unchanged",
        "qualification_pass": None,
        "scope": "transformation only; requires validate_price_basis and independently verified raw archive",
        "quote_uncertainty": "secondary whole-bar quote source may differ from Futu; no prices, coefficients or source choices fitted to returns",
        "sdk_qfq_equivalence_claimed": False,
        "official_formula_reference": "https://openapi.futunn.com/futu-api-doc/quote/get-rehab.html",
    }
    return combined, audit
