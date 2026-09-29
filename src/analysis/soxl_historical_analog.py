"""Pre-registered historical shape and as-of-state retrieval through Qlib."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT = "soxl_historical_analog_20260929"
COHORTS = {
    "training_2011_2022": ("2011-01-01", "2022-12-31"),
    "untrained_2023_2025": ("2023-01-01", "2025-12-31"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(value):
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_ready(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path: Path, value):
    path.write_text(json.dumps(_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def extract_market(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read prices and VIX from their authoritative Qlib stores, never raw CSV."""
    import qlib
    from qlib.data import D

    end = "2026-09-25"
    price_provider = root / "data/qlib/soxl"
    vix_provider = root / "data/qlib/soxl"
    qlib.init(provider_uri=str(price_provider), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2010-03-11", end_time=end))
    prices = D.features(["SOXL", "SOXX"], ["$open", "$close"], start_time=calendar[0], end_time=calendar[-1])
    prices.columns = [c.removeprefix("$") for c in prices.columns]
    market = prices.unstack("instrument").reindex(calendar)
    market.columns = [f"{symbol.lower()}_{field}" for field, symbol in market.columns]
    qlib.init(provider_uri=str(vix_provider), region="us", kernels=1)
    vix = D.features(["SOXX"], ["$vix"], start_time=calendar[0], end_time=calendar[-1])
    market["vix"] = vix.droplevel("instrument")["$vix"].reindex(calendar)
    market.index.name = "datetime"
    if market.index.has_duplicates or market.index.max() != pd.Timestamp(end):
        raise ValueError("blocked: invalid frozen market calendar")
    if (market.dropna() <= 0).any().any():
        raise ValueError("blocked: nonpositive prices or VIX")
    source_files = [
        price_provider / "qlib_store_manifest.json",
        root / "data/raw/soxs_defensive_v1/source_manifest.json",
    ]
    sources = {str(p.relative_to(root)): {"sha256": _sha(p), "content": json.loads(p.read_text())} for p in source_files}
    feature_files = []
    for provider, names in ((price_provider, {"open.day.bin", "close.day.bin"}), (vix_provider, {"vix.day.bin"})):
        feature_files.extend(p for p in (provider / "features").rglob("*.bin") if p.name in names)
        feature_files.extend([provider / "calendars/day.txt", provider / "instruments/all.txt"])
    audit = {
        "input_mode": "qlib_D_API_fixed_snapshot", "as_of": end,
        "source_apis": ["qlib.init", "D.calendar", "D.features"],
        "price_provider_relative": str(price_provider.relative_to(root)),
        "vix_provider_relative": str(vix_provider.relative_to(root)),
        "rows": len(market), "start": market.index.min(), "end": market.index.max(),
        "missing_by_column": market.isna().sum().to_dict(),
        "source_manifests": sources,
        "provider_file_hashes": {str(p.relative_to(root)): _sha(p) for p in sorted(feature_files)},
        "price_basis": "Existing main-provider SOXL/SOXX price paths; no extra dividend cash or total-return assertion, not actual share accounting.",
    }
    return market, audit


def load_or_extract_market(root: Path, card: Path) -> tuple[pd.DataFrame, dict]:
    retained = card / "retained_input.json"
    if not retained.exists():
        return extract_market(root)
    from qlib.data.dataset.loader import StaticDataLoader
    info = json.loads(retained.read_text())
    path = card / info["market_relative_path"]
    audit_path = card / info["source_audit_relative_path"]
    if not path.exists():
        canonical = Path(info["canonical_card_dir"])
        path, audit_path = canonical / info["market_relative_path"], canonical / info["source_audit_relative_path"]
    if _sha(path) != info["market_sha256"] or _sha(audit_path) != info["source_audit_sha256"]:
        raise ValueError("blocked: retained Qlib input identity changed")
    # This is a Recorder-produced artifact, loaded through Qlib's file-data API.
    market = StaticDataLoader(config=str(path)).load(instruments=None)
    audit = json.loads(audit_path.read_text())
    audit["input_mode"] = "hash_verified_retained_qlib_market_artifact"
    return market, audit


def load_retained_coverage(root: Path, card: Path) -> dict | None:
    """Reuse the retained audit and its necessary input rows without live reads."""
    manifest_path = card / "retained_input.json"
    if not manifest_path.exists():
        return None
    info = json.loads(manifest_path.read_text())
    if "coverage_relative_path" not in info:
        return None
    path = card / info["coverage_relative_path"]
    if not path.exists():
        path = Path(info["canonical_card_dir"]) / info["coverage_relative_path"]
    if _sha(path) != info["coverage_sha256"]:
        raise ValueError("blocked: retained training coverage identity changed")
    import qlib
    from qlib.workflow import R
    qlib.init(provider_uri=str(root / "data/qlib/soxl"), region="us", kernels=1)
    R.set_uri(str(path.parents[4]))
    recorder = R.get_recorder(recorder_id=path.parents[2].name, experiment_name=EXPERIMENT)
    coverage = recorder.load_object("analysis/training_coverage.pkl")
    if not coverage.get("inputs"):
        return None
    return coverage


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    iqr = (np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)) / 1.349
    scale = np.where(scale > 1e-12, scale, iqr)
    scale = np.where(scale > 1e-12, scale, np.std(values, axis=0))
    return center, np.where(scale > 1e-12, scale, 1.0)


def _path_drawdown(path_values: np.ndarray) -> float:
    path = np.r_[1.0, path_values]
    return float(np.min(path / np.maximum.accumulate(path) - 1.0))


def shape_features(market: pd.DataFrame, end_position: int, length: int) -> tuple[np.ndarray, np.ndarray, dict]:
    section = market.iloc[end_position - length + 1:end_position + 1]
    if len(section) != length or section.isna().any().any():
        raise ValueError("incomplete shape window")
    soxx_path = section.soxx_close.to_numpy() / float(section.soxx_open.iloc[0])
    soxl_path = section.soxl_close.to_numpy() / float(section.soxl_open.iloc[0])
    path_log = np.log(soxx_path)
    daily_sigma = np.std(np.diff(np.r_[0.0, path_log]), ddof=1)
    if daily_sigma <= 1e-12:
        raise ValueError("degenerate shape window")
    shape = np.interp(np.linspace(0, length, 21), np.arange(length + 1), np.r_[0.0, path_log]) / (daily_sigma * np.sqrt(length))
    amplitude = np.array([path_log[-1], _path_drawdown(soxx_path), daily_sigma * np.sqrt(252), section.vix.median(), section.vix.max()])
    details = {
        "soxx_window_return_pct": 100 * (soxx_path[-1] - 1),
        "soxx_window_max_drawdown_pct": 100 * _path_drawdown(soxx_path),
        "soxx_window_volatility_pct": 100 * daily_sigma * np.sqrt(252),
        "soxl_window_return_pct": 100 * (soxl_path[-1] - 1),
        "soxl_window_max_drawdown_pct": 100 * _path_drawdown(soxl_path),
        "vix_median": section.vix.median(), "vix_peak": section.vix.max(),
    }
    return shape, amplitude, details


def state_features(market: pd.DataFrame, anchor_position: int, windows: tuple[int, ...]) -> tuple[np.ndarray, list[int]]:
    values, blocks = [], []
    momentums, risks, trends = [], [], []
    for length in windows:
        closes = market.soxx_close.iloc[anchor_position - length:anchor_position + 1].to_numpy()
        if len(closes) != length + 1 or not np.isfinite(closes).all():
            raise ValueError("incomplete state window")
        momentums.append(np.log(closes[-1] / closes[0]))
        risks.extend([np.std(np.diff(np.log(closes)), ddof=1) * np.sqrt(252), closes[-1] / closes.max() - 1.0])
        trends.append(np.log(closes[-1] / closes[-length:].mean()))
    vix = market.vix.iloc[[anchor_position - 20, anchor_position]].to_numpy()
    if not np.isfinite(vix).all():
        raise ValueError("incomplete VIX state")
    for block in (momentums, risks, trends, [np.log(vix[-1]), np.log(vix[-1] / vix[0])]):
        values.extend(block)
        blocks.append(len(block))
    return np.array(values), blocks


def _cohort(table: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = map(pd.Timestamp, COHORTS[name])
    return table[(table.support_start >= start) & (table.support_end <= end)].copy()


def _reference_distances(features: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    selected = np.arange(0, len(table), 20)
    values = features[selected]
    starts = table.support_start.to_numpy()[selected]
    ends = table.support_end.to_numpy()[selected]
    disjoint = (ends[:, None] < starts[None, :]) | (ends[None, :] < starts[:, None])
    mask = np.triu(disjoint, k=1)
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)[mask]
    if len(distances) < 30:
        raise ValueError("blocked: insufficient independent-support distance reference")
    return np.sort(distances)


def cluster_nonoverlap(ranked: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Suppress overlapping full-support windows; this does not establish IID."""
    result = ranked.sort_values(["distance", "anchor_date"]).copy().reset_index(drop=True)
    accepted = []
    for index, row in result.iterrows():
        overlap = next((event for event in accepted if row.support_start <= event[2] and row.support_end >= event[1]), None)
        representative = overlap is None
        if representative:
            overlap = (len(accepted) + 1, row.support_start, row.support_end)
            accepted.append(overlap)
        result.loc[index, "representative"] = representative
        result.loc[index, "representative_rank"] = overlap[0]
        result.loc[index, "event_cluster_id"] = f"{prefix}_C{overlap[0]:03d}"
    result["representative"] = result["representative"].astype(bool)
    result["representative_rank"] = result["representative_rank"].astype(int)
    return result


def forward_outcomes(market: pd.DataFrame, anchor_position: int) -> dict:
    result = {}
    for horizon in (18, 20):
        available = min(horizon, len(market) - anchor_position - 1)
        result[f"forward_{horizon}_complete"] = available == horizon
        result[f"forward_{horizon}_observed_sessions"] = available
        for symbol in ("soxx", "soxl"):
            closes = market[f"{symbol}_close"].iloc[anchor_position:anchor_position + available + 1].to_numpy()
            complete = available == horizon and np.isfinite(closes).all()
            path = closes / closes[0]
            result[f"{symbol}_forward_{horizon}_return_pct"] = float(100 * (path[-1] - 1.0)) if complete else None
            result[f"{symbol}_forward_{horizon}_max_drawdown_pct"] = float(100 * _path_drawdown(path)) if complete else None
    return result


def state_comparison_table(market: pd.DataFrame, displayed: pd.DataFrame) -> pd.DataFrame:
    events = [{"cohort": "target_2026", "event_id": "B20_60_target", "anchor_date": pd.Timestamp("2026-08-31")}]
    events += displayed[displayed.window_spec.eq("B20_60")][["cohort", "event_id", "anchor_date"]].to_dict("records")
    rows = []
    for event in events:
        values, _ = state_features(market, market.index.get_loc(event["anchor_date"]), (20, 60))
        rows.append({**event,
            "soxx_return_20_pct": 100 * np.expm1(values[0]), "soxx_return_60_pct": 100 * np.expm1(values[1]),
            "soxx_vol_20_annual_pct": 100 * values[2], "soxx_current_drawdown_20_pct": 100 * values[3],
            "soxx_vol_60_annual_pct": 100 * values[4], "soxx_current_drawdown_60_pct": 100 * values[5],
            "soxx_ma_gap_20_pct": 100 * np.expm1(values[6]), "soxx_ma_gap_60_pct": 100 * np.expm1(values[7]),
            "vix_level": np.exp(values[8]), "vix_change_20_pct": 100 * np.expm1(values[9]),
        })
    return pd.DataFrame(rows)


def retrieve_variant(market: pd.DataFrame, kind: str, windows: tuple[int, ...]) -> dict[str, Any]:
    is_shape = kind == "A_retrospective"
    spec_name = "A" + str(windows[0]) if is_shape else "B" + "_".join(map(str, windows))
    target_date = pd.Timestamp("2026-09-25" if is_shape else "2026-08-31")
    target_position = market.index.get_loc(target_date)
    metadata, raw_vectors, amplitudes = [], [], []
    lookback = windows[0] - 1 if is_shape else max(max(windows), 20)
    for position in range(lookback, len(market)):
        try:
            if is_shape:
                shape, amplitude, details = shape_features(market, position, windows[0])
                vector = shape
            else:
                vector, blocks = state_features(market, position, windows)
                amplitude, details = None, {}
        except ValueError:
            continue
        future_position = position if is_shape else position + 20
        metadata.append({
            "event_id": f"{spec_name}_{market.index[position]:%Y%m%d}", "retrieval_kind": kind, "window_spec": spec_name,
            "anchor_date": market.index[position], "feature_start": market.index[position - lookback],
            "feature_end": market.index[position], "support_start": market.index[position - lookback],
            "support_end": market.index[future_position] if future_position < len(market) else pd.NaT,
            "outcome_start": pd.NaT if is_shape else market.index[position + 1] if position + 1 < len(market) else pd.NaT,
            "outcome_end": pd.NaT if is_shape else market.index[future_position] if future_position < len(market) else pd.NaT,
            "market_position": position, **details,
        })
        raw_vectors.append(vector)
        if is_shape:
            amplitudes.append(amplitude)
    table = pd.DataFrame(metadata)
    raw = np.asarray(raw_vectors)
    target_index = int(table.index[table.anchor_date.eq(target_date)][0])
    training_indices = _cohort(table, "training_2011_2022").index.to_numpy()
    if is_shape:
        amp = np.asarray(amplitudes)
        center, scale = _robust_scale(amp[training_indices])
        embedded = np.c_[raw * np.sqrt(.5 / raw.shape[1]), (amp - center) / scale * np.sqrt(.5 / amp.shape[1])]
    else:
        center, scale = _robust_scale(raw[training_indices])
        block_weights = np.concatenate([np.repeat(np.sqrt(.25 / size), size) for size in blocks])
        embedded = (raw - center) / scale * block_weights
    reference = _reference_distances(embedded[training_indices], table.loc[training_indices])
    threshold = float(np.quantile(reference, .05))
    target = table.loc[target_index].to_dict()
    target["cohort"] = "target_2026"
    if not is_shape:
        target.update(forward_outcomes(market, target_position))
    ranked_tables, counts = [], []
    for cohort_name, bounds in COHORTS.items():
        candidates = _cohort(table, cohort_name)
        candidates["cohort"] = cohort_name
        distances = np.linalg.norm(embedded[candidates.index] - embedded[target_index], axis=1)
        candidates["distance"] = distances
        candidates["reference_distance_percentile"] = 100 * np.searchsorted(reference, distances, side="right") / len(reference)
        candidates["below_p5_reference"] = distances <= threshold
        ranked = cluster_nonoverlap(candidates, f"{spec_name}_{cohort_name}")
        reps = ranked[ranked.representative]
        all_anchors = table.anchor_date.between(*map(pd.Timestamp, bounds))
        counts.append({
            "window_spec": spec_name, "cohort": cohort_name, "raw_candidate_count": len(candidates),
            "boundary_purged_count": int(all_anchors.sum() - len(candidates)),
            "below_p5_reference_count": int(ranked.below_p5_reference.sum()),
            "nonoverlap_representative_count": len(reps),
            "nonoverlap_below_p5_count": int(reps.below_p5_reference.sum()),
            "p5_distance_reference": threshold, "reference_pair_count": len(reference),
            "nearest_distance": float(reps.iloc[0].distance),
            "nearest_reference_percentile": float(reps.iloc[0].reference_distance_percentile),
        })
        ranked_tables.append(ranked)
    ranked = pd.concat(ranked_tables, ignore_index=True)
    # Freeze ranks and cluster assignment BEFORE touching subsequent returns.
    ranking_digest = hashlib.sha256(ranked[["event_id", "cohort", "distance", "event_cluster_id", "representative"]].to_csv(index=False).encode()).hexdigest()
    displayed = ranked[ranked.representative & ranked.representative_rank.le(3)].copy()
    if not is_shape:
        outcomes = pd.DataFrame([forward_outcomes(market, int(position)) for position in displayed.market_position], index=displayed.index)
        displayed = pd.concat([displayed, outcomes], axis=1)
    return {"ranked": ranked, "displayed": displayed, "counts": pd.DataFrame(counts), "target": target,
            "reference_distances": reference, "scaler": {"center": center.tolist(), "scale": scale.tolist(), "fit_cohort": "training_2011_2022"},
            "ranking_sha256_before_outcome_join": ranking_digest}


def render_comparisons(market: pd.DataFrame, displayed: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#292929", "text.color": "#292929"})
    colors = ["#315B8A", "#B27F20", "#737448"]
    line_styles = ["--", "-.", ":"]
    nearest = displayed[(displayed.window_spec == "A44") & (displayed.cohort == "training_2011_2022")]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True)
    for ax, symbol in zip(axes, ("soxx", "soxl"), strict=True):
        for number, row in enumerate(nearest.itertuples()):
            section = market.loc[row.feature_start:row.feature_end]
            values = np.r_[100.0, 100 * section[f"{symbol}_close"].to_numpy() / section[f"{symbol}_open"].iloc[0]]
            status = "within P5" if row.below_p5_reference else "outside P5"
            label = f"{row.feature_start:%Y-%m-%d} to {row.feature_end:%Y-%m-%d} ({status})"
            ax.plot(np.arange(45), values, color=colors[number], linestyle=line_styles[number], linewidth=1.7, label=label)
        section = market.loc["2026-07-27":"2026-09-25"]
        target = np.r_[100.0, 100 * section[f"{symbol}_close"].to_numpy() / section[f"{symbol}_open"].iloc[0]]
        ax.plot(np.arange(45), target, color="#222222", linewidth=2.7, label="Target: 2026-07-27 to 2026-09-25")
        ax.axhline(100, color="#b7b7b7", linewidth=.8)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
        ax.set_ylabel(f"{symbol.upper()} price index\nfirst open = 100")
        ax.set_xlim(0, 44)
    axes[0].legend(loc="upper left", fontsize=8.5, ncol=1, frameon=False)
    axes[1].set_xticks([0, 10, 20, 30, 40, 44])
    axes[1].set_xlabel("Trading-session position; 0 = first open, 1–44 = session closes")
    fig.suptitle("Ex-post 44-session paths: target and three training-period nearest windows", y=.98, fontsize=13)
    fig.text(.08, .025, "Price paths, not cash-dividend total returns. P5 is a geometric reference, not significance. SOXL is a separate leveraged path, not 3×SOXX.", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .95))
    path_a = output_dir / "A44_ex_post_paths.png"
    fig.savefig(path_a, dpi=160)
    plt.close(fig)

    events = displayed[displayed.window_spec == "B20_60"].sort_values(["cohort", "representative_rank"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    labels = [f"{'Train' if row.cohort.startswith('training') else 'Later'} {row.anchor_date:%Y-%m-%d}" for row in events.itertuples()]
    target_outcomes = forward_outcomes(market, market.index.get_loc(pd.Timestamp("2026-08-31")))
    positions = np.arange(len(events))
    for ax, symbol in zip(axes, ("soxx", "soxl"), strict=True):
        values = events[f"{symbol}_forward_18_return_pct"].to_numpy()
        for index, (value, cohort) in enumerate(zip(values, events.cohort, strict=True)):
            ax.scatter(value, index, color="#315B8A" if cohort.startswith("training") else "#737448", s=45, zorder=3)
            ax.annotate(f"{value:+.1f}%", (value, index), xytext=(5, 5), textcoords="offset points", fontsize=8)
        target_value = target_outcomes[f"{symbol}_forward_18_return_pct"]
        ax.axvline(target_value, color="#222222", linewidth=1.6, linestyle="--", label=f"2026 target: {target_value:+.1f}%")
        ax.axvline(0, color="#b7b7b7", linewidth=.8)
        ax.grid(axis="x", color="#e5e5e5", linewidth=.6)
        ax.set_title(symbol.upper())
        ax.set_xlabel("Next 18-session price return")
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.margins(x=.2)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Nearest B states: subsequent outcomes — all six outside the P5 reference", fontsize=13)
    fig.text(.08, .02, "Outcomes never select neighbors. Three nonoverlapping representatives per cohort, not IID samples. Target: only 18 observed sessions; session 20 unavailable.", fontsize=8)
    fig.tight_layout(rect=(0, .055, 1, .92))
    path_b = output_dir / "B_forward18_outcomes.png"
    fig.savefig(path_b, dpi=160)
    plt.close(fig)
    return [path_a, path_b]


def run_historical_analog_card(*, root: Path | str, recorder_uri: Path | str, include_training_coverage: bool = True) -> dict:
    import qlib
    from qlib.workflow import R
    from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts

    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    plan_path = card / "selection_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["A"]["main_sessions"] != 44 or plan["B"]["main_windows"] != [20, 60] or plan["description_threshold"]["quantile"] != .05:
        raise ValueError("blocked: pre-registered retrieval contract changed")
    plan_hash = _sha(plan_path)
    market, source_audit = load_or_extract_market(root, card)
    runs = [retrieve_variant(market, "A_retrospective", (length,)) for length in (44, 40, 60)]
    runs += [retrieve_variant(market, "B_asof_state", windows) for windows in ((20, 60), (20,), (60,))]
    ranked = pd.concat([run["ranked"] for run in runs], ignore_index=True)
    displayed = pd.concat([run["displayed"] for run in runs], ignore_index=True)
    counts = pd.concat([run["counts"] for run in runs], ignore_index=True)
    targets = pd.DataFrame([run["target"] for run in runs])
    state_comparison = state_comparison_table(market, displayed)
    if targets.loc[targets.window_spec.eq("A44"), "feature_start"].iloc[0] != pd.Timestamp("2026-07-27"):
        raise ValueError("blocked: 44-session target does not match the registered interval")
    coverage = {"summary": {"status": "not_requested"}, "tables": {}}
    displayed.to_csv(card / "analog_events.csv", index=False)
    counts.to_csv(card / "retrieval_counts.csv", index=False)
    targets.to_csv(card / "target_features.csv", index=False)
    state_comparison.to_csv(card / "B_state_feature_comparison.csv", index=False)
    if include_training_coverage:
        coverage = load_retained_coverage(root, card)
        if coverage is None:
            from src.analysis.soxl_training_coverage import audit_soxl_training_coverage
            coverage = audit_soxl_training_coverage(root=root, analog_events=displayed)
    if _sha(plan_path) != plan_hash:
        raise ValueError("blocked: plan changed during analysis")
    # Dataset/D API extraction and coverage reads must finish before R.start.
    qlib.init(provider_uri=str(root / "data/qlib/soxl"), region="us", kernels=1)
    recorder_uri = Path(recorder_uri).resolve()
    R.set_uri(str(recorder_uri))
    with tempfile.TemporaryDirectory(prefix="soxl_analog_artifacts_") as temporary, R.start(experiment_name=EXPERIMENT):
        recorder = R.get_recorder()
        R.log_params(plan_sha256=plan_hash, snapshot_as_of="2026-09-25", threshold_role="descriptive_geometry_not_significance", selection_uses_forward_outcomes="false")
        R.log_metrics(market_rows=float(len(market)), displayed_rows=float(len(displayed)))
        R.save_objects(artifact_path="inputs", **{"market_frame.pkl": market, "source_audit.pkl": source_audit})
        R.save_objects(artifact_path="analysis", **{"ranked_candidates.pkl": ranked, "displayed_analogs.pkl": displayed,
                       "counts.pkl": counts, "target_features.pkl": targets,
                       "B_state_feature_comparison.pkl": state_comparison,
                       "distance_references.pkl": {run["target"]["window_spec"]: run["reference_distances"] for run in runs},
                       "scalers.pkl": {run["target"]["window_spec"]: run["scaler"] for run in runs},
                       "training_coverage.pkl": coverage})
        recorder.log_artifact(str(plan_path), artifact_path="inputs")
        audit_json = Path(temporary) / "source_audit.json"
        _write_json(audit_json, source_audit)
        recorder.log_artifact(str(audit_json), artifact_path="inputs")
        figure_paths = render_comparisons(market, displayed, Path(temporary))
        for path in figure_paths:
            recorder.log_artifact(str(path), artifact_path="figures")
        pd.testing.assert_frame_equal(market, recorder.load_object("inputs/market_frame.pkl"))
        recorder_dir = Path(recorder.get_local_dir())
        recorder_info = {"recorder_id": recorder.id, "recorder_local_dir": str(recorder_dir),
                         "recorder_relative_dir": str(recorder_dir.relative_to(recorder_uri.parent))}
    _remove_qlib_code_snapshot_artifacts(recorder_dir)
    source_audit_path = recorder_dir / "artifacts/inputs/source_audit.json"
    canonical_card = recorder_uri.parent
    retained = {
        "canonical_card_dir": str(canonical_card),
        "market_relative_path": str((recorder_dir / "artifacts/inputs/market_frame.pkl").relative_to(canonical_card)),
        "market_sha256": _sha(recorder_dir / "artifacts/inputs/market_frame.pkl"),
        "source_audit_relative_path": str(source_audit_path.relative_to(canonical_card)),
        "source_audit_sha256": _sha(source_audit_path),
        "coverage_relative_path": str((recorder_dir / "artifacts/analysis/training_coverage.pkl").relative_to(canonical_card)),
        "coverage_sha256": _sha(recorder_dir / "artifacts/analysis/training_coverage.pkl"),
        "replay_reader": "qlib.data.dataset.loader.StaticDataLoader after artifact hash verification",
    }
    _write_json(card / "retained_input.json", retained)
    displayed.to_csv(card / "analog_events.csv", index=False)
    counts.to_csv(card / "retrieval_counts.csv", index=False)
    targets.to_csv(card / "target_features.csv", index=False)
    for name, table in coverage["tables"].items():
        table.to_csv(card / f"coverage_{name}.csv", index=False)
    _write_json(card / "training_coverage_summary.json", coverage["summary"])
    summary = {
        "experiment": EXPERIMENT, "decision": "descriptive_coverage_diagnostic_no_model_or_production_change",
        "provider_uri": "data/qlib/soxl", "region": "us", "freq": "day",
        "recorder_uri": str(recorder_uri), "workflow_config": "workflow.yaml",
        "model_selection": {"status": "not_applicable_descriptive_audit"},
        "benchmark": {"instrument": "SOXX", "role": "price_path_proxy_not_total_return"},
        "plan_sha256": plan_hash, "as_of": "2026-09-25", "market_input": retained,
        "data_quality": {key: source_audit[key] for key in ("rows", "start", "end", "missing_by_column")},
        "counts": counts.to_dict("records"), "targets": targets.to_dict("records"),
        "ranking_identity_before_outcomes": {run["target"]["window_spec"]: run["ranking_sha256_before_outcome_join"] for run in runs},
        "figures": ["artifacts/figures/A44_ex_post_paths.png", "artifacts/figures/B_forward18_outcomes.png"],
        "training_coverage": coverage["summary"], "model_training_performed": False, "production_change": False,
        "human_authored_training_plan": {"path": "training_data_plan.md", "status": "proposed_not_executed_or_validated"},
        "sidecars": ["selection_plan.json", "training_data_plan.md", "analog_events.csv", "retrieval_counts.csv", "target_features.csv", "B_state_feature_comparison.csv", "training_coverage_summary.json", "retained_input.json", "conclusion.md"],
        "training_optimization_assessment": {
            "evidence": "No main-window neighbor passes the descriptive P5 reference; this does not prove the model has never seen drawdowns or rebounds. Training has tail labels, but cross-asset rows and overlapping labels greatly outnumber distinct operational episodes.",
            "bounded_next_experiments_not_executed": [
                "Prioritize own-policy action-state coverage (holdings, drawdown, recovery clock) and mature-label supervision over adding rows indiscriminately; leave whole calendar episodes and related assets together.",
                "Compare current auxiliary sampling against one pre-registered conditional crash-episode concentration cap, preserving the overall label/task budget; date weights are already almost uniform.",
                "If extending the frozen 2022 training cutoff, use strictly chronological nested validation; this 2026 query and reused 2023-2025 neighbors are diagnostics, not clean performance proof."
            ],
            "prohibited_shortcuts": ["Do not oversample A's ex-post nearest paths to train prospective B.", "Do not select B neighbors or thresholds using their forward rebound.", "Do not call P5 or nonoverlap counts statistical significance or IID effective sample size."],
        },
        "limitations": ["A uses the completed path and cannot be an early trading signal.", "B's anchor was chosen retrospectively; causal features do not make this clean OOS.", "P5 describes geometry, not significance or predictive probability; a large nearest-neighbor pool entails multiple comparisons.", "Nonoverlap representatives are not IID effective samples and related assets share shocks.", "The target has only 18 observed sessions after 8/31; its 20-day outcome remains incomplete."],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/market_frame.pkl", "inputs/source_audit.pkl", "inputs/source_audit.json", "analysis/ranked_candidates.pkl", "analysis/distance_references.pkl", "analysis/scalers.pkl", "analysis/training_coverage.pkl"]},
        "qlib_api": ["qlib.init", "D.calendar", "D.features", "StaticDataLoader", "R.start", "R.save_objects", "Recorder.load_object"],
        "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html", "https://qlib.readthedocs.io/en/latest/component/recorder.html"],
        **recorder_info,
    }
    _write_json(card / "summary.json", summary)
    (card / "workflow.yaml").write_text(f"experiment: {EXPERIMENT}\nselection_plan: selection_plan.json\nsource_as_of: 2026-09-25\nretained_input_manifest: retained_input.json\nrecorder_relative_dir: {recorder_info['recorder_relative_dir']}\nmodel_training_performed: false\nproduction_change: false\n")
    lines = ["# 中文结论", "", "本卡区分完整路径的事后形态诊断A与8/31收盘已知状态检索B；不训练模型、不修改模拟盘。所有行情锁定9/25快照，通过QLib D提取并保留完整输入。", ""]
    for spec in ("A44", "B20_60"):
        for cohort in COHORTS:
            count = counts[(counts.window_spec == spec) & (counts.cohort == cohort)].iloc[0]
            first = displayed[(displayed.window_spec == spec) & (displayed.cohort == cohort)].iloc[0]
            lines.append(f"- {spec}/{cohort}：检索{int(count.raw_candidate_count)}个窗口，边界purge {int(count.boundary_purged_count)}个；P5内原始窗口{int(count.below_p5_reference_count)}个、非重叠代表{int(count.nonoverlap_below_p5_count)}个。最近窗口{first.feature_start.date()}–{first.feature_end.date()}，距离{first.distance:.3f}、历史距离分布分位{first.reference_distance_percentile:.2f}%。")
    if coverage["summary"].get("status") == "audited":
        audit = coverage["summary"]
        concentration = coverage["tables"]["auxiliary_sampling_concentration"].set_index("grouping")
        crash_share = 100 * concentration.loc["overlap_cluster_conditional_on_crash_labels", "top5_group_nominal_loss_share"]
        date_weight = concentration.loc["datetime", "inverse_weight_concentration"]
        drift = audit["feature_drift"]
        lines += ["", f"训练覆盖：实际主训练{audit['primary_actual_fit_rows']}行，{audit['primary_tail_positive_rows']}个tail阳性行合并为{audit['primary_tail_operational_overlap_clusters']}个操作性重叠簇；辅助训练{audit['auxiliary_actual_fit_rows']}行、{audit['auxiliary_unique_dates']}个日期，crash阳性只有{audit['auxiliary_operational_overlap_clusters']}个跨资产重叠簇。它们不是IID样本数。",
                  f"辅助日期名义权重本来接近均匀（逆集中度{date_weight:.2f}/{audit['auxiliary_unique_dates']}）；crash阳性名义权重的前5簇占{crash_share:.2f}%，该分母仅为crash阳性权重，不能解释成总loss或gradient份额。因此不把一般日期均衡当主要修复。",
                  f"目标末日{drift['features']}维特征有{drift['latest_features_outside_train_1_99']}个超出训练P1–P99、无缺失；44日窗口中{drift['window_feature_cells_outside_train_1_99']}/{drift['window_feature_cells_total']}个特征单元越界。单变量范围内不证明联合状态或组合控制器路径已经覆盖。HOLD训练仍基于baseline状态，验证使用自身状态，risk_clock未列入22个动作特征。",
                  "优先有界实验：①补自身策略状态/成熟标签与完整episode留出；②在保持总体任务权重下比较一个辅助crash事件集中度上限；③如扩展2022训练截止，必须严格滚动验证并另留新的前向数据。以上仅建议，本卡没有训练候选，也不据2026已经观察的反弹调权重。"]
    lines += ["", "固定40/44/60日与B20/60敏感性全部保留在sidecar，不择最漂亮的结果。B后续18/20日结果只在排名冻结之后加入；目标的20日尚不完整。P5不是p值或预测成功率，去重数量不代表IID独立事件。A不能直接决定B的训练样本，2026目标及反复查看的2023以后窗口不能冒充clean OOS。", "", "后续人工研究计划见[training_data_plan.md](training_data_plan.md)，仅为提案，尚未训练或验证。", f"QLib recorder：{recorder_info['recorder_id']}。"]
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    return summary
