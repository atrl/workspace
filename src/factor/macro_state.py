from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.data.macro_state import MACRO_STATE_FIELDS, MACRO_STATE_INSTRUMENT, STATE_SUFFIXES


MACRO_STATE_FACTOR_SOURCE = "qlib.data.D.features"
MACRO_STATE_LIMITATIONS = [
    "Treasury curves and the broad-dollar mirror are current-vintage history rather than revision-vintage archives.",
    "The 5Y5Y inflation feature is derived from Treasury nominal/real 5Y and 10Y par curves, not FRED T5YIFR.",
    "Raw observations are release-mapped in the qlib store before factor calculation.",
    "Factors describe macro state and are not standalone position rules.",
]
REAL_RATE_FEATURES = (
    "macro_dfii10_chg5",
    "macro_dfii10_chg20",
    "macro_dfii10_z252",
    "macro_t5y5y_inflation_proxy_chg5",
    "macro_t5y5y_inflation_proxy_chg20",
    "macro_t5y5y_inflation_proxy_z252",
)
BROAD_DOLLAR_FEATURES = (
    "macro_dtwexbgs_ret5",
    "macro_dtwexbgs_ret20",
    "macro_dtwexbgs_ret60",
    "macro_dtwexbgs_z252",
)
RATE_VOL_FEATURES = (
    "macro_dgs2_chg5",
    "macro_dgs10_chg5",
    "macro_curve_10y2y",
    "macro_curve_10y2y_chg20",
    "macro_dgs2_dvol20",
    "macro_dgs2_dvol60",
    "macro_dgs10_dvol20",
    "macro_dgs10_dvol60",
)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    rolling = numeric.rolling(window, min_periods=min(60, window))
    return (numeric - rolling.mean()) / rolling.std(ddof=0).replace(0.0, np.nan)


def build_macro_state_factor_frame(
    config: QlibProviderConfig,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Build compact macro feature groups from release-mapped qlib state."""

    manifest_path = config.resolved_provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"macro-state qlib manifest missing: {manifest_path.as_posix()}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {str(item).removeprefix("$") for item in manifest.get("fields", [])}
    raw_fields = [*MACRO_STATE_FIELDS]
    state_fields = [f"{field}_{suffix}" for field in MACRO_STATE_FIELDS for suffix in STATE_SUFFIXES]
    query = [f"${field}" for field in [*raw_fields, *state_fields] if field in available]
    frame = load_qlib_feature_frame(
        config,
        instruments=[MACRO_STATE_INSTRUMENT],
        fields=query,
        start=start,
        end=end,
    ).reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame = frame.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    frame = frame.rename(columns={column: column.removeprefix("$") for column in query})
    for field in [*raw_fields, *state_fields]:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")

    output = pd.DataFrame({"datetime": frame["datetime"]})
    output["macro_dfii10_chg5"] = frame["dfii10"].diff(5)
    output["macro_dfii10_chg20"] = frame["dfii10"].diff(20)
    output["macro_dfii10_z252"] = _zscore(frame["dfii10"], 252)
    output["macro_t5y5y_inflation_proxy_chg5"] = frame["t5y5y_inflation_proxy"].diff(5)
    output["macro_t5y5y_inflation_proxy_chg20"] = frame["t5y5y_inflation_proxy"].diff(20)
    output["macro_t5y5y_inflation_proxy_z252"] = _zscore(frame["t5y5y_inflation_proxy"], 252)
    output["macro_dtwexbgs_ret5"] = np.log(frame["dtwexbgs"] / frame["dtwexbgs"].shift(5))
    output["macro_dtwexbgs_ret20"] = np.log(frame["dtwexbgs"] / frame["dtwexbgs"].shift(20))
    output["macro_dtwexbgs_ret60"] = np.log(frame["dtwexbgs"] / frame["dtwexbgs"].shift(60))
    output["macro_dtwexbgs_z252"] = _zscore(frame["dtwexbgs"], 252)
    output["macro_dgs2_chg5"] = frame["dgs2"].diff(5)
    output["macro_dgs10_chg5"] = frame["dgs10"].diff(5)
    curve = frame["dgs10"] - frame["dgs2"]
    output["macro_curve_10y2y"] = curve
    output["macro_curve_10y2y_chg20"] = curve.diff(20)
    output["macro_dgs2_dvol20"] = frame["dgs2"].diff().rolling(20, min_periods=10).std(ddof=0)
    output["macro_dgs2_dvol60"] = frame["dgs2"].diff().rolling(60, min_periods=20).std(ddof=0)
    output["macro_dgs10_dvol20"] = frame["dgs10"].diff().rolling(20, min_periods=10).std(ddof=0)
    output["macro_dgs10_dvol60"] = frame["dgs10"].diff().rolling(60, min_periods=20).std(ddof=0)
    feature_cols = [*REAL_RATE_FEATURES, *BROAD_DOLLAR_FEATURES, *RATE_VOL_FEATURES]
    output[feature_cols] = output[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    output.attrs.update(
        {
            "source": MACRO_STATE_FACTOR_SOURCE,
            "provider_uri": config.resolved_provider_uri.as_posix(),
            "feature_cols": feature_cols,
            "feature_groups": {
                "real_rates": list(REAL_RATE_FEATURES),
                "broad_dollar": list(BROAD_DOLLAR_FEATURES),
                "rate_vol": list(RATE_VOL_FEATURES),
            },
            "availability_policy": manifest.get("availability_policy"),
            "series": manifest.get("series", {}),
            "limitations": MACRO_STATE_LIMITATIONS,
        }
    )
    return output
