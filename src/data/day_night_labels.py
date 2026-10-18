"""Fixed H5 day/night source checks; no market loading or model operations."""
from __future__ import annotations

import hashlib
import math
import sys

import numpy as np
import pandas as pd

SYMBOLS = ("SOXL", "SOXX")
TARGETS = ("soxl_forward_open_return", "soxx_forward_open_return", "tail_event_5d",
           "tail_cash_target_5d", "relative_alpha_5d")
CLOCKS = ("decision_at", "return_available_at", "label_available_at")
NY = "America/New_York"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def basis_certificate(raw_metadata, expected_coefficients):
    """Compare exact declared metadata, never infer a map from price values."""
    coefficients = {}
    for symbol in SYMBOLS:
        item = raw_metadata["symbol_audits"][symbol]
        got = {"early_segment_before": item["nasdaq_start"],
               "k_open": item["stitch_scales"]["open"],
               "k_close": item["stitch_scales"]["close"]}
        require(got == expected_coefficients[symbol], "archived basis metadata differs: " + symbol)
        require(all(math.isfinite(got[x]) and got[x] > 0 for x in ("k_open", "k_close")),
                "missing/nonpositive basis coefficient")
        boundary = pd.Timestamp(got["early_segment_before"])
        require(boundary.tz is None and boundary == boundary.normalize(), "basis boundary is not a date")
        coefficients[symbol] = got
    return {"coefficients": coefficients, "anchor": "open_unchanged", "source": "immutable_stitch_metadata",
            "rule": "ell_C=log(float64(C_stored))+(log(k_open)-log(k_close)) strictly before boundary",
            "plane": "distribution_adjusted_research_price_plane",
            "factor_semantics": "Within-day f_t cancels in D; cross-day changes in f_t enter N.",
            "original_targets_modified": False, "historical_PIT_qualified": False,
            "independent_forward_qualified": False, "cash_dividend_ledger": False}


def original_target_digest(reference):
    """Include exact target bytes/dtypes and timestamp/index identities."""
    require(list(reference.columns) == [*CLOCKS, *TARGETS], "reference schema/order differs")
    require(isinstance(reference.index, pd.DatetimeIndex) and reference.index.is_unique,
            "reference needs unique signal date index")
    digest = hashlib.sha256()
    digest.update(reference.index.asi8.tobytes())
    digest.update(str(reference.index.dtype).encode())
    for name in reference:
        values = reference[name]
        digest.update(name.encode()); digest.update(str(values.dtype).encode())
        if name in CLOCKS:
            require(isinstance(values.dtype, pd.DatetimeTZDtype), "original clock must retain timezone")
            digest.update(values.array.asi8.tobytes())
        else:
            require(pd.api.types.is_numeric_dtype(values.dtype), "original target is not numeric")
            digest.update(np.ascontiguousarray(values.to_numpy()).tobytes())
    return digest.hexdigest()


def assert_reference_unchanged(reference, expected_hash):
    require(original_target_digest(reference) == expected_hash, "original index/clocks/targets mutated")


def build_h5_mapping(calendar, reference, *, expected_rows=3014):
    """Map on the complete native session sequence before selecting the cohort."""
    calendar = pd.DatetimeIndex(calendar)
    require(calendar.tz is None and calendar.is_unique and calendar.is_monotonic_increasing,
            "native calendar must be ordered unique naive dates")
    require(calendar.equals(calendar.normalize()), "calendar contains non-date values")
    original_target_digest(reference)
    require(len(reference) == expected_rows and reference.index.is_monotonic_increasing,
            "original cohort count/order differs")
    require(reference.index.tz is None and reference.index.equals(reference.index.normalize()),
            "signal index must be original naive dates")
    require(np.isfinite(reference.loc[:, list(TARGETS)].to_numpy()).all(), "original target is nonfinite")
    loc = calendar.get_indexer(reference.index)
    require((loc >= 0).all() and (loc + 6 < len(calendar)).all(), "incomplete H5 session mapping")
    result = pd.DataFrame({"d" + str(j): calendar[loc + 1 + j] for j in range(6)}, index=reference.index.copy())
    for name, day, minute in (("decision_at", "d0", 29), ("return_available_at", "d1", 31),
                              ("label_available_at", "d5", 31)):
        expected = pd.DatetimeIndex(result[day]).tz_localize(NY) + pd.Timedelta(hours=9, minutes=minute)
        actual = pd.DatetimeIndex(reference[name]).tz_convert(NY)
        require(actual.equals(expected), "original calendar/clock mismatch: " + name)
    return result


def validate_prices(prices, mapping):
    require(isinstance(prices.index, pd.MultiIndex) and list(prices.index.names) == ["datetime", "instrument"]
            and prices.index.is_unique, "native price index differs")
    require(list(prices.columns) == ["$open", "$close"], "only the two admitted raw fields are allowed")
    require(all(dtype == np.dtype("float32") for dtype in prices.dtypes), "native raw storage dtype must remain float32")
    require(set(prices.index.get_level_values("instrument")) == set(SYMBOLS), "unexpected native instruments")
    dates = prices.index.get_level_values("datetime")
    require(isinstance(dates, pd.DatetimeIndex) and dates.tz is None, "native price dates differ")
    required = []
    for symbol in SYMBOLS:
        frame = prices.xs(symbol, level="instrument")
        opens = frame["$open"].reindex(pd.DatetimeIndex(mapping.to_numpy().ravel()))
        closes = frame["$close"].reindex(pd.DatetimeIndex(mapping.loc[:, ["d" + str(i) for i in range(5)]].to_numpy().ravel()))
        require(np.isfinite(opens).all() and opens.gt(0).all(), "missing/nonpositive H5 open: " + symbol)
        require(np.isfinite(closes).all() and closes.gt(0).all(), "missing/nonpositive H5 close: " + symbol)
        required.append({"instrument": symbol, "required_open_occurrences": len(opens),
                         "required_close_occurrences": len(closes), "missing_or_nonpositive": 0})
    return required


def h1_vintage_diagnostic(prices, mapping, reference):
    """Fixed float32 arithmetic; differences have no acceptance threshold."""
    validate_prices(prices, mapping)
    before = original_target_digest(reference)
    paired = pd.DataFrame(index=reference.index); summary = []
    for symbol, target in zip(SYMBOLS, TARGETS[:2]):
        column = prices.xs(symbol, level="instrument")["$open"]
        p0 = column.reindex(mapping.d0).to_numpy(np.float32)
        p1 = column.reindex(mapping.d1).to_numpy(np.float32)
        # Original pandas float32 open.shift(-2)/open.shift(-1)-1 ordering.
        reconstructed = np.subtract(np.divide(p1, p0), np.float32(1), dtype=np.float32)
        original = reference[target].to_numpy(copy=True)
        difference = reconstructed.astype(np.float64) - original.astype(np.float64)
        for name, values in (("original", original), ("selected_vintage", reconstructed), ("difference", difference)):
            paired[symbol + "_" + name] = values
        summary.append({"instrument": symbol, "rows": len(original),
                        "exact_equal_count": int(np.equal(reconstructed, original).sum()),
                        "finite_count": int(np.isfinite(difference).sum()),
                        "missing_count": int((~np.isfinite(difference)).sum()),
                        "max_absolute_difference": float(np.abs(difference).max()),
                        "acceptance_threshold": None, "diagnostic_only": True})
    assert_reference_unchanged(reference, before)
    return paired, summary


def binary64_environment():
    """Small arithmetic checks of nearest ties and gradual underflow, not prices."""
    require(sys.float_info.radix == 2 and sys.float_info.mant_dig == 53, "binary64 is required")
    u = float(2.0 ** -53); eta = float.fromhex("0x0.0000000000001p-1022")
    one = float(1); step = 2 * u; next_one = one + step
    require(one + u == one and next_one + u == one + 2 * step, "round-to-nearest-even check failed")
    require(eta + eta > eta and (eta + eta) - eta == eta and eta * one == eta,
            "subnormal arithmetic requires gradual underflow without FTZ/DAZ")
    require(sys.float_info.min * .5 > 0, "flush-to-zero is forbidden")
    return {"dtype": "IEEE754_binary64", "rounding": "nearest_even", "gradual_underflow": True,
            "FTZ_DAZ": False, "reassociation_FMA": False, "u": u, "eta": eta}


def common_log_plane(prices, certificate):
    binary64_environment()
    # One cached log per present primitive. A nonrequired missing close stays NaN;
    # the fixed required set is separately checked before any label construction.
    result = pd.DataFrame(index=prices.index.copy(), columns=["ell_open", "ell_close"], dtype=np.float64)
    for symbol in SYMBOLS:
        source = prices.xs(symbol, level="instrument")
        coeff = certificate["coefficients"][symbol]
        correction = math.log(coeff["k_open"]) - math.log(coeff["k_close"])
        boundary = pd.Timestamp(coeff["early_segment_before"])
        for date, row in source.iterrows():
            for field, name in (("$open", "ell_open"), ("$close", "ell_close")):
                value = float(row[field])
                if not math.isfinite(value) or value <= 0:
                    result.loc[(date, symbol), name] = np.nan
                    continue
                cached = math.log(value)
                if field == "$close" and date < boundary:
                    cached = cached + correction
                require(math.isfinite(cached), "nonfinite cached log")
                result.loc[(date, symbol), name] = cached
    return result


def _finite(value):
    require(math.isfinite(value), "nonfinite/overflow in fixed binary64 graph")
    return value


def _leftfold(values):
    iterator = iter(values); result = _finite(float(next(iterator)))
    for value in iterator:
        result = _finite(result + float(value))
    return result


def _bound(leaves):
    require(len(leaves) in (22, 44), "incorrect telescope leaf inventory")
    # Fixed nonnegative left fold: at most 43 additions. gamma then product,
    # then additive subnormal allowance; no economic floor or rescue multiplier.
    magnitude = _leftfold(abs(float(x)) for x in leaves)
    unit = float(2.0 ** -53)
    numerator = 43 * unit
    denominator = 1 - numerator
    gamma = numerator / denominator
    product = _finite(gamma * magnitude)
    allowance = 43 * float.fromhex("0x0.0000000000001p-1022")
    return _finite(product + allowance), magnitude


def construct_labels(log_plane, mapping):
    environment = binary64_environment(); rows = []; checks = []
    for signal, dates in mapping.iterrows():
        values = {}; leaves = []; per_asset = []
        for symbol in SYMBOLS:
            frame = log_plane.xs(symbol, level="instrument")
            opens = [float(frame.at[dates["d" + str(j)], "ell_open"]) for j in range(6)]
            closes = [float(frame.at[dates["d" + str(j)], "ell_close"]) for j in range(5)]
            require(all(math.isfinite(x) for x in opens + closes), "nonfinite required cached log")
            day = _leftfold(_finite(closes[j] - opens[j]) for j in range(5))
            night = _leftfold(_finite(opens[j + 1] - closes[j]) for j in range(5))
            endpoint = _finite(opens[5] - opens[0])
            residual = _finite(_finite(day + night) - endpoint)
            leg_leaves = [x for j in range(5) for x in (closes[j], opens[j])]
            leg_leaves += [x for j in range(5) for x in (opens[j + 1], closes[j])]
            per_leaves = leg_leaves + [opens[5], opens[0]]
            bound, magnitude = _bound(per_leaves)
            require(abs(residual) <= bound, "per-asset telescope exceeds frozen bound")
            values.update({"D_" + symbol: day, "N_" + symbol: night, "endpoint_" + symbol: endpoint})
            per_asset.append({"signal_date": signal, "identity": symbol, "residual": residual,
                              "bound": bound, "L1": magnitude, "leaf_count": 22, "passed": True})
            leaves.extend(leg_leaves)
        d_relative = _finite(values["D_SOXL"] - values["D_SOXX"])
        n_relative = _finite(values["N_SOXL"] - values["N_SOXX"])
        total = _finite(d_relative + n_relative)
        contrast = _finite(n_relative - d_relative)
        endpoint = _finite(values["endpoint_SOXL"] - values["endpoint_SOXX"])
        residual = _finite(total - endpoint)
        for symbol in SYMBOLS:
            frame = log_plane.xs(symbol, level="instrument")
            leaves.extend([float(frame.at[dates.d5, "ell_open"]), float(frame.at[dates.d0, "ell_open"])])
        bound, magnitude = _bound(leaves)
        require(abs(residual) <= bound, "relative telescope exceeds frozen bound")
        rows.append({"signal_date": signal, **values, "D_relative": d_relative, "N_relative": n_relative,
                     "S": total, "Z": contrast, "S_endpoint": endpoint})
        checks.extend(per_asset)
        checks.append({"signal_date": signal, "identity": "relative", "residual": residual,
                       "bound": bound, "L1": magnitude, "leaf_count": 44, "passed": True})
    labels = pd.DataFrame(rows).set_index("signal_date")
    require(np.isfinite(labels.to_numpy()).all(), "nonfinite fixed S/Z")
    audit = pd.DataFrame(checks)
    audit.attrs["binary64_environment"] = environment
    audit.attrs["contract"] = "43*u/(1-43*u), fixed nonnegative leftfold L1, product then +43*eta"
    return labels, audit
