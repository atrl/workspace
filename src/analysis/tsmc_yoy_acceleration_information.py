"""Fixed TSMC YoY-acceleration 5-session information screen (0 models, 0 strategy).

Primary statistic: signed TIME-SERIES Spearman between the fixed source score
(reported YoY acceleration) and the 5-session SOXL-minus-SOXX compound spread,
measured from the source-eligible US open t to the stock D.calendar session t+5 open.

Availability is stored as timezone-aware UTC (NY 09:30 -> UTC, DST-correct). The
statistic applies the true masks (admitted / factor-clock-cause / finite) BEFORE any
rank correlation; unknown/unmatured/out-of-scope events stay in the calendar and are
masked, never 0-filled. Nothing is fitted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_PROVIDER = "data/qlib/tsmc_monthly_revenue_20261010_v1"
PRICE_PROVIDER = "data/qlib/stage1_uniform_daily_marks_20261009"
SOURCE_RECORDER_ID = "4d1dc18e75c7489d8251af1e737c1a0f"
SOURCE_CARD = "research/tsmc_monthly_revenue_pit_qualification_20261010"
INSTRUMENT = "TSMC_RELEASE"
HORIZON = 5
BLOCK = 3
REPLICATES = 2000
SEED = 20261010
DEV = ("2016-01-04", "2022-12-31")
OUTER = ("2023-01-03", "2026-09-25")
WINDOWS = {"development": DEV, "outer_reused_diagnostic": OUTER}
NY = "America/New_York"
FREEZE_MANIFEST = "research/tsmc_yoy_acceleration_5session_information_20261010/freeze_manifest.json"
SCORE_JOIN_TOL = 1e-5


def load_freeze(root: Path) -> dict:
    return json.loads((root / FREEZE_MANIFEST).read_text())


def select_exit(calendar, entry, horizon: int = HORIZON):
    """Exit session = real D.calendar position entry+horizon; None if unavailable."""
    pos = {d: i for i, d in enumerate(calendar)}
    if entry is None or entry not in pos or pos[entry] + horizon >= len(calendar):
        return None
    return calendar[pos[entry] + horizon]


def score_is_aligned(provider_score, parent_score, *, tol: float = SCORE_JOIN_TOL) -> bool:
    """The joined provider score must equal THIS period's parent reported-YoY delta."""
    return bool(np.isfinite(provider_score) and np.isfinite(parent_score)
                and abs(float(provider_score) - float(parent_score)) <= tol)


def mark_returns(opens: pd.DataFrame, entry, exit_day):
    """SOXL/SOXX open-to-open compound returns; NaN if either mark is missing."""
    if entry is None or exit_day is None or entry not in opens.index or exit_day not in opens.index:
        return np.nan, np.nan
    p0x, p1x = opens.loc[entry, "SOXL"], opens.loc[exit_day, "SOXL"]
    p0s, p1s = opens.loc[entry, "SOXX"], opens.loc[exit_day, "SOXX"]
    if not np.isfinite([p0x, p1x, p0s, p1s]).all():
        return np.nan, np.nan
    return float(p1x / p0x - 1.0), float(p1s / p0s - 1.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(root: Path, plan: dict, *, events_sha: str, source_sha: str, price_sha: str) -> dict:
    """Fail closed unless the plan + frozen manifest match the actual source identities."""
    freeze = load_freeze(root)
    frozen = plan["frozen_sources"]
    expected_source = frozen["source_info_provider"]["manifest_sha256"]
    expected_price = frozen["price_provider"]["manifest_sha256"]
    checks = {
        "source_manifest_matches_plan": source_sha == expected_source,
        "price_manifest_matches_plan": price_sha == expected_price,
        "source_manifest_matches_freeze": source_sha == freeze["source_provider_manifest_sha256"],
        "price_manifest_matches_freeze": price_sha == freeze["price_provider_manifest_sha256"],
        "parent_events_matches_freeze": events_sha == freeze["parent_events_object_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"blocked: frozen source identity changed: {checks}")
    return {"source_manifest_sha256": source_sha, "price_manifest_sha256": price_sha,
            "parent_events_object_sha256": events_sha, "checks": checks, "verified": True}


def load_source_features(root: Path) -> pd.DataFrame:
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str((root / SOURCE_PROVIDER).resolve()), region="us", kernels=1)
    raw = D.features([INSTRUMENT], ["$acceleration_pct_points", "$event_available"], freq="day").reset_index()
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    return raw.rename(columns={"$acceleration_pct_points": "acceleration_pct_points",
                               "$event_available": "event_available"})


def load_parent_events(root: Path) -> tuple[pd.DataFrame, dict]:
    import qlib
    from qlib.workflow import R
    qlib.init(provider_uri=str((root / SOURCE_PROVIDER).resolve()), region="us", kernels=1)
    with R.uri_context(str((root / SOURCE_CARD / "qlib_recorder").resolve())):
        rec = R.get_recorder(recorder_id=SOURCE_RECORDER_ID, experiment_name=SOURCE_CARD.split("/")[-1])
        events = rec.load_object("events")
        local = Path(rec.get_local_dir()) / "artifacts"
    # bind the exact parent events bytes (freeze evidence) without modifying the source card
    events_sha = _sha(local / "events")
    return events, {"recorder_id": SOURCE_RECORDER_ID, "card": SOURCE_CARD, "events_object_sha256": events_sha}


def load_price_calendar_and_opens(root: Path) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Strict full D.calendar + opens reindexed to it; missing opens stay NaN (never dropped)."""
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str((root / PRICE_PROVIDER).resolve()), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(freq="day")).normalize()
    raw = D.features(["SOXL", "SOXX"], ["$open"], freq="day").reset_index()
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    opens = raw.pivot_table(index="datetime", columns="instrument", values="$open", dropna=False)
    opens = opens.reindex(calendar)  # never compress missing dates
    return calendar, opens


def _ny_open_utc(session: pd.Timestamp) -> pd.Timestamp:
    naive = pd.Timestamp(session).normalize()
    return (naive.tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")


def build_observations(root: Path, plan: dict) -> dict:
    events, meta = load_parent_events(root)
    source_sha = _sha(root / SOURCE_PROVIDER / "qlib_store_manifest.json")
    price_sha = _sha(root / PRICE_PROVIDER / "qlib_store_manifest.json")
    frozen = verify_frozen(root, plan, events_sha=meta["events_object_sha256"], source_sha=source_sha, price_sha=price_sha)
    features = load_source_features(root).set_index("datetime")
    calendar, opens = load_price_calendar_and_opens(root)

    rows = []
    for _, e in events.sort_values("period").iterrows():
        entry = pd.Timestamp(e["factor_eligible_session"]).normalize() if pd.notna(e.get("factor_eligible_session")) else pd.NaT
        clock = pd.Timestamp(e["factor_clock_utc"]) if pd.notna(e.get("factor_clock_utc")) else pd.NaT
        decision_at = _ny_open_utc(entry) + pd.Timedelta(minutes=1) if pd.notna(entry) else pd.NaT
        clock_ok = bool(pd.notna(clock) and pd.notna(decision_at) and clock <= decision_at)
        # score must equal THIS period's provider value AND match the parent reported-YoY delta (float32 tol)
        provider_score = np.nan
        if pd.notna(entry) and entry in features.index:
            value = features.loc[entry, "acceleration_pct_points"]
            provider_score = float(value) if pd.notna(value) else np.nan
        parent_score = float(e["acceleration_pct_points"]) if pd.notna(e.get("acceleration_pct_points")) else np.nan
        score = provider_score
        score_aligned = score_is_aligned(provider_score, parent_score)
        exit_day = select_exit(calendar, entry)
        soxl_r, soxx_r = mark_returns(opens, entry, exit_day)
        spread = float(soxl_r - soxx_r) if np.isfinite(soxl_r) and np.isfinite(soxx_r) else np.nan
        admitted = bool(e.get("admitted_to_scope", False))
        segment = "unknown"
        if pd.notna(entry):
            if pd.Timestamp(DEV[0]) <= entry <= pd.Timestamp(DEV[1]):
                segment = "development"
            elif pd.Timestamp(OUTER[0]) <= entry <= pd.Timestamp(OUTER[1]):
                segment = "outer_reused_diagnostic"
            else:
                segment = "out_of_price_scope"
        reasons = []
        if not admitted:
            reasons.append("not_admitted_scope")
        if pd.isna(clock) or not clock_ok:
            reasons.append("factor_clock_after_decision")
        if pd.isna(entry):
            reasons.append("no_factor_session")
        elif segment == "out_of_price_scope":
            reasons.append("out_of_price_scope")
        if np.isnan(provider_score):
            reasons.append("unknown_score")
        elif not score_aligned:
            reasons.append("unknown_score_mismatch")
        if exit_day is None:
            reasons.append("no_exit_session")
        elif np.isnan(spread):
            reasons.append("missing_mark")
        stat_valid = (admitted and clock_ok and pd.notna(entry) and score_aligned
                      and np.isfinite(spread) and segment != "out_of_price_scope")
        rows.append({"period": e["period"], "entry_session": entry, "exit_session": (exit_day if exit_day is not None else pd.NaT),
                     "source_factor_clock_utc": (clock.tz_convert("UTC").isoformat() if pd.notna(clock) else None),
                     "decision_at": (decision_at.isoformat() if pd.notna(decision_at) else None),
                     "label_available_at": ((_ny_open_utc(exit_day).isoformat() if exit_day is not None else None)),
                     "score": score, "parent_score": parent_score, "score_aligned": score_aligned,
                     "soxl_ret_5d": soxl_r, "soxx_ret_5d": soxx_r, "spread_5d": spread,
                     "admitted": admitted, "factor_clock_ok": clock_ok, "segment": segment,
                     "unknown_reason": ";".join(reasons), "stat_valid": stat_valid})
    frame = pd.DataFrame(rows)
    return {"observations": frame, "calendar": calendar, "source_meta": meta, "frozen_verified": frozen,
            "source_manifest_sha256": source_sha, "price_manifest_sha256": price_sha}


def _ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank().to_numpy()


def masked_spearman(section: pd.DataFrame) -> float | None:
    sel = section[section["stat_valid"]]
    if len(sel) < 5:
        return None
    x, y = sel["score"].to_numpy(float), sel["spread_5d"].to_numpy(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def masked_pearson(section: pd.DataFrame) -> float | None:
    sel = section[section["stat_valid"]]
    if len(sel) < 5:
        return None
    x, y = sel["score"].to_numpy(float), sel["spread_5d"].to_numpy(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def masked_ic(section: pd.DataFrame, column: str) -> float | None:
    sel = section[section["stat_valid"]]
    if len(sel) < 5:
        return None
    y = sel[column].to_numpy(float)
    if not np.isfinite(y).all() or np.std(y) == 0:
        return None
    return float(np.corrcoef(_ranks(sel["score"].to_numpy(float)), _ranks(y))[0, 1])


def block_bootstrap(section: pd.DataFrame, *, block: int = BLOCK, replicates: int = REPLICATES,
                    seed: int = SEED) -> dict:
    """Block bootstrap over the FULL section; the mask is applied INSIDE the statistic."""
    ordered = section.reset_index(drop=True)
    n = len(ordered)
    point = masked_spearman(ordered)
    if n < block or point is None:
        return {"point": point, "ci_low": None, "ci_high": None, "valid_replicates": 0, "status": "needs_data"}
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in chosen])[:n]
        value = masked_spearman(ordered.loc[idx])
        if value is not None:
            draws.append(value)
    if len(draws) < 50:
        return {"point": point, "ci_low": None, "ci_high": None, "valid_replicates": len(draws), "status": "needs_data"}
    return {"point": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "valid_replicates": len(draws), "blocks": int(len(starts)), "block": block, "replicates": replicates,
            "seed": seed, "status": "ok"}


def evaluate(root: Path, plan: dict) -> dict:
    built = build_observations(root, plan)
    obs = built["observations"]
    per_window = {}
    for name, (start, end) in WINDOWS.items():
        mask = (obs["entry_session"] >= pd.Timestamp(start)) & (obs["entry_session"] <= pd.Timestamp(end))
        section = obs[mask].reset_index(drop=True)
        per_window[name] = {
            "events_in_window": int(len(section)),
            "valid_n": int(section["stat_valid"].sum()),
            "unknown": {k: int(v) for k, v in section[~section["stat_valid"]]["unknown_reason"].value_counts().items()},
            "primary_ic": masked_spearman(section),
            "pearson": masked_pearson(section),
            "secondary_soxl_ic": masked_ic(section, "soxl_ret_5d"),
            "secondary_soxx_ic": masked_ic(section, "soxx_ret_5d"),
            "bootstrap": block_bootstrap(section),
            "by_year": {int(y): {"n": int(len(g)), "valid_n": int(g["stat_valid"].sum()),
                                 "ic": masked_spearman(g)} for y, g in section.groupby(pd.to_datetime(section["entry_session"]).dt.year, dropna=True)},
        }
    dev = per_window["development"]
    gate = (dev["valid_n"] >= 40 and dev["primary_ic"] is not None and dev["primary_ic"] > 0
            and dev["bootstrap"]["status"] == "ok" and dev["bootstrap"]["ci_low"] is not None and dev["bootstrap"]["ci_low"] > 0)
    results = {
        "frozen_verified": built["frozen_verified"], "source_meta": built["source_meta"],
        "source_manifest_sha256": built["source_manifest_sha256"], "price_manifest_sha256": built["price_manifest_sha256"],
        "windows": WINDOWS, "horizon_sessions": HORIZON, "per_window": per_window,
        "primary_gate": {"scope": "development", "min_valid_n": 40, "pass": bool(gate),
                         "valid_n": dev["valid_n"], "primary_ic": dev["primary_ic"],
                         "ci95_lower": dev["bootstrap"]["ci_low"]},
        "decision": "candidate_signal_diagnostic_pass_needs_independent_forward" if gate else
                    "end_fixed_monthly_information_route_no_parameter_search",
        "caveats": ["raw_association_only", "not_proven_to_beat_price_or_volatility", "no_economic_alpha",
                    "not_promotable", "requires_priced_vol_control_and_10_25bp_and_fresh_forward",
                    "route_stopped_for_insufficient_evidence_not_proven_absence_of_information"],
    }
    return {"results": results, "observations": obs}
