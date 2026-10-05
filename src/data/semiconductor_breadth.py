"""Complete published-member semiconductor features from native Qlib data.

SEC report dates never establish availability. A filing is usable at a signal
only after the preceding Qlib session. Missing members invalidate the whole
basket; all current members require 61 consecutive closes for 60 returns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.data.qlib_native import QlibProviderConfig, init_qlib_provider, build_qlib_store_from_frame


FEATURE_COLUMNS = (
    "breadth_up_fraction", "equal_minus_disclosed_weight_return",
    "cross_section_return_std", "mean_abs_correlation_60",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_sha(frame: pd.DataFrame) -> str:
    schema = json.dumps({"columns": list(frame.columns), "index_names": list(frame.index.names),
                         "dtypes": [str(t) for t in frame.dtypes]}, sort_keys=True).encode()
    return hashlib.sha256(schema + pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def qualify_recovered_wolf(snapshot_dir: Path, reference_provider: Path) -> tuple[pd.DataFrame, dict]:
    """Qualify a bounded old-security source, including original SEC valuations.

    This neither extends the old security beyond its declared epoch nor joins
    the cancelled 2025 shares to newly issued WOLF equity.
    """
    from qlib.data import D

    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "recovery_manifest.json").read_text())
    for name, digest in manifest["hashes"].items():
        if _sha(snapshot_dir / name) != digest:
            raise ValueError(f"blocked: recovery source hash changed: {name}")
    if manifest["autype"] != "NONE" or manifest["code"] != "US.WOLF":
        raise ValueError("blocked: recovery requires observed US.WOLF NONE bars")
    start, end = pd.Timestamp(manifest["requested_start"]), pd.Timestamp(manifest["requested_end"])
    if end >= pd.Timestamp("2025-09-29"):
        raise ValueError("blocked: old WOLF equity cannot cross 2025 cancellation")
    raw = pd.read_csv(snapshot_dir / "futu_wolf_none_raw.csv")
    raw["datetime"] = pd.to_datetime(raw["time_key"])
    if (raw.empty or raw.datetime.duplicated().any() or not raw.code.eq("US.WOLF").all()
            or not raw.datetime.between(start, end).all()):
        raise ValueError("blocked: recovery source has invalid dates/security")
    if not raw.datetime.is_monotonic_increasing:
        raise ValueError("blocked: recovery dates are unordered")
    if not np.isfinite(raw.close.to_numpy(float)).all() or not raw.close.gt(0).all():
        raise ValueError("blocked: recovery close must be positive finite")
    init_qlib_provider(QlibProviderConfig(provider_uri=reference_provider))
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq="day"))
    if not pd.DatetimeIndex(raw.datetime).equals(calendar):
        raise ValueError("blocked: recovery dates differ from reference Qlib calendar")
    rehab = pd.read_csv(snapshot_dir / "futu_rehab.csv")
    if pd.to_datetime(rehab.ex_div_date).between(start, end).any():
        raise ValueError("blocked: NONE recovery requires explicit in-range corporate-action adjustment")
    anchors = []
    observed_cusips = set()
    for declared in manifest["sec_price_identity_anchors"]:
        xml = ET.fromstring((snapshot_dir / (declared["accession"] + ".xml")).read_bytes())
        local = lambda node: node.tag.split("}")[-1]
        report_date = pd.Timestamp(next(n.text for n in xml.iter() if local(n) == "repPdDate"))
        securities = []
        for node in xml.iter():
            if local(node) == "invstOrSec":
                item = {local(n): n.text for n in node.iter()}
                if item.get("cusip") == declared["cusip"]:
                    securities.append(item)
        if len(securities) != 1 or securities[0].get("units") != "NS":
            raise ValueError("blocked: SEC identity anchor is not unique share holdings")
        item = securities[0]
        implied = float(item["valUSD"]) / float(item["balance"])
        candidates = raw.loc[raw.datetime.le(report_date)]
        if candidates.empty:
            raise ValueError("blocked: SEC anchor predates recovered prices")
        bar = candidates.iloc[-1]
        if (report_date - bar.datetime).days > 4 or abs(float(bar.close) - implied) > 0.005:
            raise ValueError("blocked: recovered old-security close disagrees with SEC valuation")
        observed_cusips.add(item["cusip"])
        anchors.append({"cusip": item["cusip"], "name": item["name"],
                        "report_date": str(report_date.date()), "price_date": str(bar.datetime.date()),
                        "sec_implied_close": implied, "futu_none_close": float(bar.close),
                        "accession": declared["accession"]})
    if observed_cusips != {"225447101", "977852102"}:
        raise ValueError("blocked: both original CREE and old WOLF identity anchors are required")
    evidence = {"status": "qualified", "source_manifest_sha256": _sha(snapshot_dir / "recovery_manifest.json"),
                "source_hashes": manifest["hashes"], "identity_anchors": anchors,
                "start": str(start.date()), "end": str(end.date()), "rows": len(raw),
                "price_basis": "observed Futu NONE close; no rehabilitation event in bounded period; split-only equivalent",
                "security_epoch": "original CREE / pre-bankruptcy WOLF only",
                "new_2025_equity_joined": False, "qlib_calendar_sessions": len(calendar),
                "identity_limit": "Provider stock_id alone is not identity proof; archived SEC CUSIP valuation anchors establish old-share price scale."}
    frame = raw.loc[:, ["datetime", "close"]].copy()
    frame["instrument"] = "WOLF"
    return frame, evidence


def build_recovered_wolf_provider(snapshot_dir: Path, provider_uri: Path,
                                  reference_provider: Path) -> dict:
    frame, qualification = qualify_recovered_wolf(snapshot_dir, reference_provider)
    manifest = build_qlib_store_from_frame(
        frame, provider_uri=provider_uri, market="all", region="us", as_of=qualification["end"],
        field_map={"$close": "close"}, source=str(Path(snapshot_dir) / "recovery_manifest.json"),
    )
    manifest["source_qualification"] = qualification
    repo_root = Path(provider_uri).resolve().parents[2]
    snapshot_path = Path(snapshot_dir).resolve()
    manifest["snapshot_dir"] = str(snapshot_path.relative_to(repo_root))
    manifest["provider_uri"] = str(Path(provider_uri).resolve().relative_to(repo_root))
    manifest["source"] = str(snapshot_path.relative_to(repo_root) / "recovery_manifest.json")
    manifest["close_bin_sha256"] = _sha(Path(provider_uri) / "features/wolf/close.day.bin")
    _dump(Path(provider_uri) / "qlib_store_manifest.json", manifest)
    (Path(provider_uri) / "qlib.yaml").write_text(
        f"provider_uri: {manifest['provider_uri']}\nregion: us\nmarket: all\n", encoding="utf-8")
    return manifest


def compute_complete_basket_features(close: pd.DataFrame, holdings: pd.DataFrame,
                                     calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate only contemporaneous features; no labels, fitting or imputation."""
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Qlib calendar must be ordered and unique")
    if close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("close dates and instruments must be unique")
    close = close.reindex(calendar)
    history = holdings.copy()
    history["filing_date"] = pd.to_datetime(history.filing_date)
    if history.duplicated(["filing_date", "accession", "ticker"]).any():
        raise ValueError("duplicate published holdings")
    if not np.isfinite(history.weight_pct.to_numpy(float)).all() or not history.weight_pct.gt(0).all():
        raise ValueError("published equity weights must be positive finite")
    snapshots = [(day, group.loc[group.accession.eq(group.accession.max())])
                 for day, group in history.groupby("filing_date", sort=True)]
    filing_dates = pd.DatetimeIndex([d for d, _ in snapshots])
    output, audit = [], []
    for i, day in enumerate(calendar):
        at = filing_dates.searchsorted(calendar[i - 1], side="right") - 1 if i else -1
        row = {"datetime": day, "status": "no_pit", "expected_members": 0, "usable_members": 0,
               "filing_date": None, "report_date": None, "missing_members": ""}
        if at < 0:
            audit.append(row)
            continue
        filing, snapshot = snapshots[at]
        symbols = snapshot.ticker.astype(str).tolist()
        row.update(expected_members=len(symbols), filing_date=str(filing.date()),
                   report_date=str(snapshot.report_date.iloc[0]))
        missing = sorted(set(symbols) - set(close.columns))
        row["missing_members"] = ",".join(missing)
        if missing:
            row["status"] = "missing_members"
            audit.append(row)
            continue
        block = close.iloc[max(0, i - 60):i + 1].loc[:, symbols]
        valid = np.isfinite(block.to_numpy(float)) & (block.to_numpy(float) > 0)
        row["usable_members"] = int(valid.all(axis=0).sum()) if len(block) == 61 else 0
        if len(block) != 61 or not valid.all():
            row["status"] = "incomplete_61close_history"
            audit.append(row)
            continue
        prices = block.to_numpy(float)
        returns = prices[1:] / prices[:-1] - 1.0
        if not (returns.std(axis=0) > 0).all():
            row["status"] = "undefined_full_member_correlation"
            audit.append(row)
            continue
        corr = np.corrcoef(returns, rowvar=False)
        pairs = np.abs(corr[np.triu_indices(len(symbols), k=1)])
        if not len(pairs) or not np.isfinite(pairs).all():
            row["status"] = "undefined_full_member_correlation"
            audit.append(row)
            continue
        latest = returns[-1]
        weights = snapshot.weight_pct.to_numpy(float)
        weights = weights / weights.sum()  # Normalize full disclosed equity basket only.
        output.append({"datetime": day, "breadth_up_fraction": float((latest > 0).mean()),
                       "equal_minus_disclosed_weight_return": float(latest.mean() - latest @ weights),
                       "cross_section_return_std": float(latest.std(ddof=0)),
                       "mean_abs_correlation_60": float(pairs.mean())})
        row["status"] = "complete"
        audit.append(row)
    features = pd.DataFrame(output, columns=["datetime", *FEATURE_COLUMNS]).set_index("datetime")
    eligibility = pd.DataFrame(audit).set_index("datetime")
    return features, eligibility


def load_sector_breadth_features(provider_uri: Path, supplemental_provider_uri: Path | None = None,
                                *, start: str = "2016-01-01", end: str = "2026-07-02") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from qlib.data import D

    provider_uri = Path(provider_uri)
    manifest = json.loads((provider_uri / "qlib_store_manifest.json").read_text())
    if pd.Timestamp(end) > pd.Timestamp(manifest["calendar_end"]):
        raise ValueError("blocked: sector source does not reach requested end")
    init_qlib_provider(QlibProviderConfig(provider_uri=provider_uri))
    calendar = pd.DatetimeIndex(D.calendar(end_time=end, freq="day"))
    symbols = manifest["training_data_audit"]["constituent_instruments"]
    raw = D.features(symbols, ["$close", "$instrument_type", "$is_actual_2x"],
                     start_time=calendar[0], end_time=end, freq="day")
    present = raw["$close"].notna()
    if not (raw.loc[present, "$instrument_type"].eq(0).all() and raw.loc[present, "$is_actual_2x"].eq(0).all()):
        raise ValueError("blocked: constituent source includes a leveraged product")
    close = raw["$close"].unstack("instrument").reindex(calendar)
    hashes = {"constituent_manifest": _sha(provider_uri / "qlib_store_manifest.json"),
              "holdings_csv": _sha(provider_uri / "sec_nport_holdings_history.csv"),
              "native_constituent_close_and_types": _frame_sha(raw)}
    if supplemental_provider_uri is not None:
        supplemental_provider_uri = Path(supplemental_provider_uri)
        extra = json.loads((supplemental_provider_uri / "qlib_store_manifest.json").read_text())
        qualification = extra.get("source_qualification", {})
        if qualification.get("status") != "qualified" or qualification.get("new_2025_equity_joined") is not False:
            raise ValueError("blocked: recovered source lacks original-security qualification")
        if _sha(supplemental_provider_uri / "features/wolf/close.day.bin") != extra.get("close_bin_sha256"):
            raise ValueError("blocked: recovered Qlib prices changed after source qualification")
        snapshot_dir = Path(extra["snapshot_dir"])
        if not snapshot_dir.is_absolute():
            snapshot_dir = supplemental_provider_uri.resolve().parents[2] / snapshot_dir
        if _sha(snapshot_dir / "recovery_manifest.json") != qualification["source_manifest_sha256"]:
            raise ValueError("blocked: recovery source manifest changed after qualification")
        for name, digest in qualification["source_hashes"].items():
            if _sha(snapshot_dir / name) != digest:
                raise ValueError("blocked: archived recovery source changed after qualification")
        if "WOLF" in close.columns:
            raise ValueError("blocked: recovery must not overwrite an existing security")
        init_qlib_provider(QlibProviderConfig(provider_uri=supplemental_provider_uri))
        recovered = D.features(["WOLF"], ["$close"], start_time=extra["calendar_start"],
                               end_time=extra["calendar_end"], freq="day")
        close["WOLF"] = recovered["$close"].xs("WOLF", level="instrument").reindex(calendar)
        hashes["recovered_manifest"] = _sha(supplemental_provider_uri / "qlib_store_manifest.json")
        hashes["native_recovered_close"] = _frame_sha(recovered)
    holdings = pd.read_csv(provider_uri / "sec_nport_holdings_history.csv", dtype={"accession": str})
    expected_hash = json.loads((provider_uri / "sec_nport_holdings_manifest.json").read_text())["sha256"]
    if hashes["holdings_csv"] != expected_hash:
        raise ValueError("blocked: SEC holdings sidecar hash mismatch")
    if not holdings.groupby(["filing_date", "accession"]).size().eq(30).all():
        raise ValueError("blocked: expected full 30-equity SEC snapshots")
    features, eligibility = compute_complete_basket_features(close, holdings, calendar)
    features, eligibility = features.loc[start:end], eligibility.loc[start:end]
    if len(features) and not features.index.equals(eligibility.loc[features.index.min():].index):
        raise ValueError("blocked: complete feature history contains an interior or terminal gap")
    all_requested_complete = bool(len(features)) and features.index.equals(eligibility.index)
    audit = {"status": "qualified" if all_requested_complete else "needs_data",
             "qualified": all_requested_complete, "source_hashes": hashes,
             "feature_start": str(features.index.min().date()) if len(features) else None,
             "feature_end": str(features.index.max().date()) if len(features) else None,
             "feature_rows": len(features), "feature_columns": list(FEATURE_COLUMNS),
             "status_counts": eligibility.status.value_counts().to_dict(),
             "membership_policy": "full latest SEC equity snapshot published by preceding Qlib session",
             "price_basis": "Yahoo split-only quote close plus qualified pre-2025 Futu NONE WOLF",
             "actual_2x_or_synthetic_used": False, "imputation": "none",
             "correlation": "mean absolute off-diagonal Pearson correlation of exactly 60 current-member daily returns",
             "weights": "full disclosed equity weights normalized to sum one; not current market capitalization",
             "provider_uri": str(provider_uri), "supplemental_provider_uri": str(supplemental_provider_uri),
             "provider_manifest_paths": [str(provider_uri / "qlib_store_manifest.json")]
                 + ([str(supplemental_provider_uri / "qlib_store_manifest.json")] if supplemental_provider_uri is not None else []),
             "source_files": [str(provider_uri / "sec_nport_holdings_history.csv"),
                              str(provider_uri / "sec_nport_holdings_manifest.json")],
             "qualification_reasons": ["all disclosed equities present", "61 consecutive positive finite close observations per current member",
                                       "strict previous-session published membership", "no 2x instruments or synthetic prices"]}
    return features, eligibility, audit


def build_sector_feature_provider(provider_uri: Path, supplemental_provider_uri: Path,
                                  output_provider: Path, *, start: str, end: str) -> dict:
    features, eligibility, audit = load_sector_breadth_features(
        provider_uri, supplemental_provider_uri, start=start, end=end)
    if features.empty:
        raise ValueError("blocked: no complete sector feature observations")
    frame = features.reset_index()
    frame["instrument"] = "SOXX"
    manifest = build_qlib_store_from_frame(
        frame, provider_uri=output_provider, market="all", region="us", as_of=end,
        field_map={"$" + col: col for col in FEATURE_COLUMNS},
        source=json.dumps(audit["source_hashes"], sort_keys=True),
    )
    manifest["source_qualification"] = audit
    eligibility.to_csv(Path(output_provider) / "eligibility.csv")
    _dump(Path(output_provider) / "source_qualification.json", audit)
    _dump(Path(output_provider) / "qlib_store_manifest.json", manifest)
    return audit | {"provider_uri": str(output_provider),
                    "source_manifest_path": str(Path(output_provider) / "qlib_store_manifest.json")}
