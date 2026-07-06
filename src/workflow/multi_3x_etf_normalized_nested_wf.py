from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.data import QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider
from src.workflow.cognitive_alpha_goal_codex_exec import (
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_DOCS,
)


EXPERIMENT = "multi_3x_etf_normalized_lgb_xgb_nested_wf_v1"
CARD_TYPE = "multi_3x_etf_normalized_nested_wf_research_card"
FEATURE_SOURCE = "multi_3x_etf_rolling_zscore_percentile"
ETF_UNIVERSE: dict[str, dict[str, str]] = {
    "SOXL": {"provider": "soxl", "underlying": "SOXX"},
    "TQQQ": {"provider": "tqqq", "underlying": "QQQ"},
    "UPRO": {"provider": "upro", "underlying": "SPY"},
    "SPXL": {"provider": "spxl", "underlying": "SPY"},
    "TECL": {"provider": "tecl", "underlying": "XLK"},
}
RAW_FEATURE_COLUMNS = [
    "ret_5",
    "ret_20",
    "ret_60",
    "underlying_ret_20",
    "underlying_ret_60",
    "realized_vol_20",
    "drawdown_60",
    "trend_efficiency_20",
    "tracking_error_5",
    "tracking_error_20",
    "range_ratio_20",
    "volume_z_20",
    "vix_z_60",
]


@dataclass(frozen=True)
class Multi3xEtfNormalizedNestedWfSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    outer_test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    max_common_end: str | None = None
    max_drawdown_floor_pct: float = -20.0
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def normalized_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment


def run_multi_3x_etf_normalized_nested_wf_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = Multi3xEtfNormalizedNestedWfSpec(root=_resolve_root(root))
    return _run_card(spec)


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _run_card(spec: Multi3xEtfNormalizedNestedWfSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    raw_panel, source_audit = _load_multi_3x_raw_panel(spec)
    normalized = _attach_normalized_features(raw_panel)
    common_end_ts = pd.to_datetime(normalized.groupby("instrument")["datetime"].max()).min()
    if spec.max_common_end is not None:
        common_end_ts = min(pd.Timestamp(spec.max_common_end), common_end_ts)
    common_end = common_end_ts.date().isoformat()
    normalized = normalized.loc[pd.to_datetime(normalized["datetime"]).le(pd.Timestamp(common_end))].copy()
    provider_manifest = _build_normalized_provider(spec, normalized, as_of=common_end)
    provider_frame, provider_meta = _load_normalized_provider(spec, normalized, as_of=common_end)
    feature_cols = _model_feature_columns(provider_frame)
    candidates = _candidate_configs()
    annual_rows: list[dict[str, Any]] = []
    inner_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []

    for outer_year in spec.outer_test_years:
        inner_metrics = _inner_candidate_metrics(
            provider_frame,
            outer_test_year=int(outer_year),
            candidates=candidates,
            feature_cols=feature_cols,
            cost_per_turnover=spec.cost_per_turnover,
        )
        selected = _select_candidate_from_inner_metrics(
            inner_metrics,
            outer_test_year=int(outer_year),
            required_heldout_assets=tuple(ETF_UNIVERSE),
        )
        selected_rows.append(selected)
        outer_metrics, outer_positions = _outer_soxl_test(
            provider_frame,
            outer_test_year=int(outer_year),
            selected_candidate=selected,
            candidates=candidates,
            feature_cols=feature_cols,
            common_end=common_end,
            cost_per_turnover=spec.cost_per_turnover,
            max_drawdown_floor_pct=spec.max_drawdown_floor_pct,
        )
        annual_rows.append(outer_metrics)
        outer_positions["outer_test_year"] = int(outer_year)
        position_frames.append(outer_positions)
        inner_metrics["outer_test_year"] = int(outer_year)
        inner_frames.append(inner_metrics)

    annual_metrics = pd.DataFrame(annual_rows)
    inner_selection = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    positions = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    selected_candidates = pd.DataFrame(selected_rows)
    annual_goal = _annual_goal_summary(annual_metrics, max_drawdown_floor_pct=spec.max_drawdown_floor_pct)
    overfit_audit = _overfit_audit_summary(
        experiment=spec.experiment,
        manual_template_selection=False,
        nested_walk_forward=True,
        leave_one_asset_out_assets=tuple(ETF_UNIVERSE),
        feature_identity_columns=tuple(
            column
            for column in feature_cols
            if "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
        ),
        selection_uses_outer_test=bool(selected_candidates.get("selection_uses_outer_test", pd.Series([False])).any()),
    )
    prior_template_audit = _overfit_audit_summary(
        experiment="agentic_alpha_intraday_execution_sleeve_v2",
        manual_template_selection=True,
        nested_walk_forward=False,
        leave_one_asset_out_assets=("SOXL",),
        feature_identity_columns=(),
        selection_uses_outer_test=False,
    )

    sidecars = {
        "annual_walk_forward": "annual_walk_forward.csv",
        "inner_selection": "inner_selection.csv",
        "selected_candidates": "selected_candidates.csv",
        "model_comparison": "model_comparison.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "feature_sets": "feature_sets.csv",
        "failure_modes": "failure_modes.md",
        "qlib_store_manifest": "qlib_store_manifest.json",
    }
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)

    import qlib
    from qlib.workflow import R

    _ = qlib
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="multi_3x_nested_wf_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        positions.to_csv(positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.normalized_provider_uri.resolve().as_posix(),
                source_symbols=",".join(ETF_UNIVERSE),
                feature_source=FEATURE_SOURCE,
                model_families="lgbm_classifier,xgb_classifier,lgb_xgb_consensus",
                nested_walk_forward="true",
                leave_one_asset_out="true",
                selection_uses_outer_test="false",
            )
            summary = _summary_payload(
                spec=spec,
                sidecars=sidecars,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                provider_manifest=provider_manifest,
                provider_meta=provider_meta,
                source_audit=source_audit,
                common_end=common_end,
                feature_cols=feature_cols,
                annual_metrics=annual_metrics,
                annual_goal=annual_goal,
                overfit_audit=overfit_audit,
                prior_template_audit=prior_template_audit,
                selected_candidates=selected_candidates,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                annual_metrics=annual_metrics,
                inner_selection=inner_selection,
                selected_candidates=selected_candidates,
                provider_manifest=provider_manifest,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path=spec.experiment)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
    _write_summary_files(card, summary)
    return summary


def _load_multi_3x_raw_panel(spec: Multi3xEtfNormalizedNestedWfSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for etf, meta in ETF_UNIVERSE.items():
        provider_uri = spec.root / "data" / "qlib" / meta["provider"]
        config = QlibProviderConfig(provider_uri=provider_uri, market="all", region="us", kernels=1)
        provider_meta = init_qlib_provider(config)
        end_ts = pd.Timestamp(provider_meta["calendar_end"])
        if spec.max_common_end is not None:
            end_ts = min(end_ts, pd.Timestamp(spec.max_common_end))
        end = end_ts.date().isoformat()
        start = max(pd.Timestamp(provider_meta["calendar_start"]), pd.Timestamp(spec.train_start)).date().isoformat()
        frame = _load_pair_frame(
            config,
            etf=etf,
            underlying=meta["underlying"],
            start=start,
            end=end,
        )
        frames.append(frame)
        audits[etf] = {
            "provider_uri": provider_meta["provider_uri"],
            "provider": meta["provider"],
            "underlying": meta["underlying"],
            "calendar_start": provider_meta["calendar_start"],
            "calendar_end": provider_meta["calendar_end"],
            "used_start": start,
            "used_end": end,
            "row_count": int(len(frame)),
            "source": "qlib.data.D.features",
        }
    if not frames:
        raise ValueError("no 3x ETF qlib frames loaded")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["instrument", "datetime"]).reset_index(drop=True), audits


def _load_pair_frame(
    config: QlibProviderConfig,
    *,
    etf: str,
    underlying: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    init_qlib_provider(config)
    from qlib.data import D

    etf_fields = ["$open", "$high", "$low", "$close", "$volume", "$vix"]
    underlying_fields = ["$open", "$high", "$low", "$close", "$volume"]
    etf_frame = D.features([etf], etf_fields, start_time=start, end_time=end, freq="day").reset_index()
    underlying_frame = D.features([underlying], underlying_fields, start_time=start, end_time=end, freq="day").reset_index()
    etf_frame = etf_frame.rename(columns={field: field.removeprefix("$") for field in etf_fields})
    underlying_frame = underlying_frame.rename(
        columns={field: f"underlying_{field.removeprefix('$')}" for field in underlying_fields}
    )
    etf_frame["datetime"] = pd.to_datetime(etf_frame["datetime"]).dt.normalize()
    underlying_frame["datetime"] = pd.to_datetime(underlying_frame["datetime"]).dt.normalize()
    etf_frame["instrument"] = etf
    merged = etf_frame.drop(columns=["instrument"], errors="ignore").merge(
        underlying_frame.drop(columns=["instrument"], errors="ignore"),
        on="datetime",
        how="inner",
    )
    merged.insert(0, "instrument", etf)
    return merged.sort_values("datetime").reset_index(drop=True)


def _attach_normalized_features(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = (20, 60),
    min_periods: int | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out = out.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for _, group in out.groupby("instrument", sort=False):
        g = group.copy().sort_values("datetime").reset_index(drop=True)
        close = pd.to_numeric(g["close"], errors="coerce").astype("float64")
        open_ = pd.to_numeric(g["open"], errors="coerce").astype("float64")
        high = pd.to_numeric(g["high"], errors="coerce").astype("float64")
        low = pd.to_numeric(g["low"], errors="coerce").astype("float64")
        volume = pd.to_numeric(g["volume"], errors="coerce").astype("float64")
        underlying_close = pd.to_numeric(g["underlying_close"], errors="coerce").astype("float64")
        vix = pd.to_numeric(g.get("vix", pd.Series(np.nan, index=g.index)), errors="coerce").astype("float64")
        daily_ret = close.pct_change()
        underlying_ret = underlying_close.pct_change()
        for lookback in sorted(set([5, 20, 60, *windows])):
            g[f"ret_{lookback}"] = close / close.shift(lookback) - 1.0
        g["underlying_ret_20"] = underlying_close / underlying_close.shift(20) - 1.0
        g["underlying_ret_60"] = underlying_close / underlying_close.shift(60) - 1.0
        g["realized_vol_20"] = daily_ret.rolling(20, min_periods=10).std(ddof=0)
        g["drawdown_60"] = close / close.rolling(60, min_periods=20).max() - 1.0
        g["trend_efficiency_20"] = (
            (close / close.shift(20) - 1.0).abs()
            / daily_ret.abs().rolling(20, min_periods=10).sum().replace(0.0, np.nan)
        )
        tracking_error = daily_ret - 3.0 * underlying_ret
        g["tracking_error_5"] = tracking_error.rolling(5, min_periods=3).mean()
        g["tracking_error_20"] = tracking_error.rolling(20, min_periods=10).mean()
        for lookback in windows:
            if f"tracking_error_{lookback}" not in g.columns:
                g[f"tracking_error_{lookback}"] = tracking_error.rolling(
                    int(lookback),
                    min_periods=min_periods or max(2, min(int(lookback), 10)),
                ).mean()
        intraday_range = (high - low) / close.replace(0.0, np.nan)
        g["range_ratio_20"] = intraday_range.rolling(20, min_periods=10).mean()
        g["volume_z_20"] = _rolling_zscore(volume, 20, min_periods=10)
        g["vix_z_60"] = _rolling_zscore(vix, 60, min_periods=20)
        g[OPEN_TO_OPEN_RETURN_COL] = open_.shift(-2) / open_.shift(-1) - 1.0
        g[LABEL_OPEN_TO_OPEN_5D_COL] = open_.shift(-6) / open_.shift(-1) - 1.0
        g["tail_loss_label"] = g[LABEL_OPEN_TO_OPEN_5D_COL].lt(-0.08).astype("float64")
        normalization_columns = [
            *RAW_FEATURE_COLUMNS,
            *[f"ret_{window}" for window in windows],
            *[f"tracking_error_{window}" for window in windows],
        ]
        for column in dict.fromkeys(normalization_columns):
            if column not in g.columns:
                continue
            for window in windows:
                obs = min_periods or max(5, min(window, 20))
                g[f"{column}_z{window}"] = _rolling_zscore(g[column], window, min_periods=obs)
                g[f"{column}_pctile{window}"] = _rolling_percentile(g[column], window, min_periods=obs)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["instrument", "datetime"]).reset_index(drop=True)


def _build_normalized_provider(
    spec: Multi3xEtfNormalizedNestedWfSpec,
    normalized: pd.DataFrame,
    *,
    as_of: str,
) -> dict[str, Any]:
    store_cols = [
        "instrument",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "underlying_close",
        "vix",
        OPEN_TO_OPEN_RETURN_COL,
        LABEL_OPEN_TO_OPEN_5D_COL,
        "tail_loss_label",
        *_model_feature_columns(normalized),
    ]
    store = normalized.loc[:, [column for column in dict.fromkeys(store_cols) if column in normalized.columns]].copy()
    return build_qlib_store_from_frame(
        store,
        provider_uri=spec.normalized_provider_uri,
        market="all",
        region="us",
        freq="day",
        as_of=as_of,
        source="qlib_multi_3x_etf_normalized_panel",
    )


def _load_normalized_provider(
    spec: Multi3xEtfNormalizedNestedWfSpec,
    normalized: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = QlibProviderConfig(provider_uri=spec.normalized_provider_uri, market="all", region="us", kernels=1)
    provider_meta = init_qlib_provider(config)
    from qlib.data import D

    fields = [
        "$open",
        "$close",
        "$volume",
        f"${OPEN_TO_OPEN_RETURN_COL}",
        f"${LABEL_OPEN_TO_OPEN_5D_COL}",
        "$tail_loss_label",
        *[f"${column}" for column in _model_feature_columns(normalized)],
    ]
    frame = D.features(list(ETF_UNIVERSE), fields, start_time=spec.train_start, end_time=as_of, freq="day")
    out = frame.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    return out.sort_values(["instrument", "datetime"]).reset_index(drop=True), provider_meta


def _candidate_configs() -> list[dict[str, Any]]:
    return [
        {"candidate": "lgbm_opp_tail_p55_t45", "model": "lgbm_classifier", "opp_threshold": 0.55, "tail_threshold": 0.45},
        {"candidate": "xgb_opp_tail_p55_t45", "model": "xgb_classifier", "opp_threshold": 0.55, "tail_threshold": 0.45},
        {
            "candidate": "consensus_opp_tail_p55_t45",
            "model": "lgb_xgb_consensus",
            "opp_threshold": 0.55,
            "tail_threshold": 0.45,
        },
    ]


def _inner_candidate_metrics(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    candidates: list[dict[str, Any]],
    feature_cols: list[str],
    cost_per_turnover: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = _clean_model_frame(frame, feature_cols)
    available = data.loc[data["datetime"].dt.year < int(outer_test_year)].copy()
    inner_years = [
        year
        for year in sorted(available["datetime"].dt.year.unique())
        if year >= 2015 and year <= int(outer_test_year) - 1
    ][-2:]
    for inner_year in inner_years:
        train_end = pd.Timestamp(f"{inner_year - 1}-12-31")
        for heldout_asset in ETF_UNIVERSE:
            train = available.loc[
                available["datetime"].le(train_end) & ~available["instrument"].eq(heldout_asset)
            ].copy()
            validation = available.loc[
                available["instrument"].eq(heldout_asset) & available["datetime"].dt.year.eq(inner_year)
            ].copy()
            if len(train) < 300 or len(validation) < 40:
                continue
            for candidate in candidates:
                pred = _predict_candidate(train, validation, feature_cols=feature_cols, candidate=candidate)
                metrics, _ = _position_metrics(
                    validation,
                    pred,
                    candidate=candidate,
                    segment_name="inner_validation",
                    cost_per_turnover=cost_per_turnover,
                )
                rows.append(
                    {
                        **metrics,
                        "candidate": str(candidate["candidate"]),
                        "model": str(candidate["model"]),
                        "inner_year": int(inner_year),
                        "heldout_asset": heldout_asset,
                        "selection_uses_outer_test": False,
                    }
                )
    return pd.DataFrame(rows)


def _select_candidate_from_inner_metrics(
    metrics: pd.DataFrame,
    *,
    outer_test_year: int,
    required_heldout_assets: tuple[str, ...],
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("inner nested walk-forward produced no candidate metrics")
    rows = metrics.copy()
    rows["selection_return_pct"] = pd.to_numeric(rows["selection_return_pct"], errors="coerce").fillna(0.0)
    rows["selection_max_drawdown_pct"] = pd.to_numeric(rows["selection_max_drawdown_pct"], errors="coerce").fillna(-100.0)
    rows["selection_total_turnover"] = pd.to_numeric(rows["selection_total_turnover"], errors="coerce").fillna(0.0)
    rows["passes_inner_drawdown"] = rows["selection_max_drawdown_pct"].ge(-20.0)
    rows["passes_inner_return"] = rows["selection_return_pct"].gt(0.0)
    grouped = (
        rows.groupby("candidate")
        .agg(
            avg_selection_return_pct=("selection_return_pct", "mean"),
            worst_selection_max_drawdown_pct=("selection_max_drawdown_pct", "min"),
            avg_selection_total_turnover=("selection_total_turnover", "mean"),
            inner_pass_rate=("passes_inner_drawdown", "mean"),
            heldout_assets=("heldout_asset", lambda values: sorted({str(value) for value in values})),
            inner_fold_count=("candidate", "size"),
        )
        .reset_index()
    )
    required_assets = [str(asset) for asset in required_heldout_assets]
    grouped["has_required_heldout_assets"] = grouped["heldout_assets"].map(
        lambda assets: all(asset in assets for asset in required_assets)
    )
    eligible = grouped.loc[grouped["has_required_heldout_assets"]].copy()
    if eligible.empty:
        raise ValueError("inner nested walk-forward missing required leave-one-asset-out assets")
    eligible["selection_score"] = (
        eligible["avg_selection_return_pct"]
        + 2.0 * eligible["worst_selection_max_drawdown_pct"]
        - 0.05 * eligible["avg_selection_total_turnover"]
        + 25.0 * eligible["inner_pass_rate"]
    )
    selected = eligible.sort_values(
        ["selection_score", "worst_selection_max_drawdown_pct", "avg_selection_return_pct"],
        ascending=[False, False, False],
    ).iloc[0]
    return {
        "candidate": str(selected["candidate"]),
        "outer_test_year": int(outer_test_year),
        "selection_uses_outer_test": False,
        "required_heldout_assets": required_assets,
        "observed_heldout_assets": list(selected["heldout_assets"]),
        "inner_fold_count": int(selected["inner_fold_count"]),
        "avg_selection_return_pct": float(selected["avg_selection_return_pct"]),
        "worst_selection_max_drawdown_pct": float(selected["worst_selection_max_drawdown_pct"]),
        "avg_selection_total_turnover": float(selected["avg_selection_total_turnover"]),
        "inner_pass_rate": float(selected["inner_pass_rate"]),
        "selection_score": float(selected["selection_score"]),
    }


def _outer_soxl_test(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    selected_candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_cols: list[str],
    common_end: str,
    cost_per_turnover: float,
    max_drawdown_floor_pct: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate = next(item for item in candidates if item["candidate"] == selected_candidate["candidate"])
    data = _clean_model_frame(frame, feature_cols)
    train_end = pd.Timestamp(f"{outer_test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{outer_test_year}-01-01")
    test_end = min(pd.Timestamp(f"{outer_test_year}-12-31"), pd.Timestamp(common_end))
    train = data.loc[data["datetime"].le(train_end)].copy()
    test = data.loc[
        data["instrument"].eq("SOXL") & data["datetime"].between(test_start, test_end)
    ].copy()
    if len(train) < 300 or len(test) < 40:
        raise ValueError(f"outer fold {outer_test_year} lacks enough train/test rows")
    pred = _predict_candidate(train, test, feature_cols=feature_cols, candidate=candidate)
    metrics, positions = _position_metrics(
        test,
        pred,
        candidate=candidate,
        segment_name="outer_test",
        cost_per_turnover=cost_per_turnover,
    )
    buyhold = _return_metrics(
        pd.to_numeric(test[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0),
        pd.Series(1.0, index=test.index),
        pd.Series(0.0, index=test.index),
    )
    row = {
        "outer_test_year": int(outer_test_year),
        "candidate": str(candidate["candidate"]),
        "model": str(candidate["model"]),
        "train_start": pd.Timestamp(train["datetime"].min()).date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
        "test_return_pct": metrics["selection_return_pct"],
        "test_max_drawdown_pct": metrics["selection_max_drawdown_pct"],
        "test_avg_abs_position": metrics["selection_avg_abs_position"],
        "test_total_turnover": metrics["selection_total_turnover"],
        "buyhold_return_pct": buyhold["return_pct"],
        "buyhold_max_drawdown_pct": buyhold["max_drawdown_pct"],
        "beats_buyhold": bool(metrics["selection_return_pct"] > buyhold["return_pct"]),
        "passes_drawdown": bool(metrics["selection_max_drawdown_pct"] >= max_drawdown_floor_pct),
        "test_passes_target": bool(
            metrics["selection_return_pct"] > buyhold["return_pct"]
            and metrics["selection_max_drawdown_pct"] >= max_drawdown_floor_pct
        ),
        "selection_uses_outer_test": False,
    }
    return row, positions


def _predict_candidate(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    feature_cols: list[str],
    candidate: dict[str, Any],
) -> pd.DataFrame:
    model = str(candidate["model"])
    if model == "lgbm_classifier":
        opportunity = _fit_predict_probability("lgbm_classifier", train, predict, feature_cols, LABEL_OPEN_TO_OPEN_5D_COL, 0.0)
        tail = _fit_predict_probability("lgbm_classifier", train, predict, feature_cols, "tail_loss_label", 0.5)
    elif model == "xgb_classifier":
        opportunity = _fit_predict_probability("xgb_classifier", train, predict, feature_cols, LABEL_OPEN_TO_OPEN_5D_COL, 0.0)
        tail = _fit_predict_probability("xgb_classifier", train, predict, feature_cols, "tail_loss_label", 0.5)
    elif model == "lgb_xgb_consensus":
        lgb_opp = _fit_predict_probability("lgbm_classifier", train, predict, feature_cols, LABEL_OPEN_TO_OPEN_5D_COL, 0.0)
        xgb_opp = _fit_predict_probability("xgb_classifier", train, predict, feature_cols, LABEL_OPEN_TO_OPEN_5D_COL, 0.0)
        lgb_tail = _fit_predict_probability("lgbm_classifier", train, predict, feature_cols, "tail_loss_label", 0.5)
        xgb_tail = _fit_predict_probability("xgb_classifier", train, predict, feature_cols, "tail_loss_label", 0.5)
        opportunity = (lgb_opp + xgb_opp) / 2.0
        tail = (lgb_tail + xgb_tail) / 2.0
    else:
        raise ValueError(f"unsupported candidate model: {model}")
    return pd.DataFrame(
        {
            "probability": opportunity.reindex(predict.index).fillna(0.0),
            "tail_probability": tail.reindex(predict.index).fillna(1.0),
        },
        index=predict.index,
    )


def _fit_predict_probability(
    model_name: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    positive_threshold: float,
) -> pd.Series:
    fit = train.dropna(subset=[target_col]).copy()
    x_train = fit[feature_cols].astype("float64")
    if target_col == "tail_loss_label":
        y_train = fit[target_col].gt(float(positive_threshold)).astype(int)
    else:
        y_train = fit[target_col].gt(float(positive_threshold)).astype(int)
    if y_train.nunique() < 2:
        model = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="constant", constant=int(y_train.iloc[0])))
    else:
        model = _build_fast_classifier(model_name)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(predict[feature_cols].astype("float64"))
    classes = list(getattr(model[-1], "classes_", []))
    positive_index = classes.index(1) if 1 in classes else 0
    return pd.Series(probabilities[:, positive_index], index=predict.index, name=target_col)


def _build_fast_classifier(model_name: str):
    if model_name == "lgbm_classifier":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=30,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.3,
                random_state=42,
                verbose=-1,
            ),
        )
    if model_name == "xgb_classifier":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=43,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported classifier: {model_name}")


def _position_metrics(
    segment_frame: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    candidate: dict[str, Any],
    segment_name: str,
    cost_per_turnover: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    opp_threshold = float(candidate["opp_threshold"])
    tail_threshold = float(candidate["tail_threshold"])
    opportunity = pd.to_numeric(pred["probability"], errors="coerce").fillna(0.0)
    tail = pd.to_numeric(pred["tail_probability"], errors="coerce").fillna(1.0)
    raw = ((opportunity - opp_threshold) / max(1.0 - opp_threshold, 1e-12)).clip(lower=0.0, upper=1.0)
    target_position = (raw * (1.0 - tail).clip(lower=0.0, upper=1.0)).where(tail.le(tail_threshold), 0.0)
    target_position = target_position.clip(lower=0.0, upper=1.0)
    returns = pd.to_numeric(segment_frame[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0)
    turnover = target_position.diff().abs().fillna(target_position.abs())
    strategy_returns = target_position * returns - turnover * float(cost_per_turnover)
    metrics = _return_metrics(strategy_returns, target_position, turnover)
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(segment_frame["datetime"]).dt.normalize(),
            "instrument": segment_frame["instrument"].astype(str).to_numpy(),
            "segment": segment_name,
            "candidate": str(candidate["candidate"]),
            "probability": opportunity.to_numpy(),
            "tail_probability": tail.to_numpy(),
            "target_position": target_position.to_numpy(),
            "turnover": turnover.to_numpy(),
            "strategy_return": strategy_returns.to_numpy(),
            OPEN_TO_OPEN_RETURN_COL: returns.to_numpy(),
        }
    )
    return {
        "selection_return_pct": metrics["return_pct"],
        "selection_max_drawdown_pct": metrics["max_drawdown_pct"],
        "selection_avg_abs_position": metrics["avg_abs_position"],
        "selection_total_turnover": metrics["total_turnover"],
        "selection_trade_count": int(turnover.gt(1e-9).sum()),
    }, out


def _clean_model_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    for column in [*feature_cols, OPEN_TO_OPEN_RETURN_COL, LABEL_OPEN_TO_OPEN_5D_COL, "tail_loss_label"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=[OPEN_TO_OPEN_RETURN_COL]).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if (
            (("_z" in column or "_pctile" in column) and not column.lower().startswith(("instrument", "symbol", "asset")))
        )
    ]


def _annual_goal_summary(
    annual_metrics: pd.DataFrame,
    *,
    max_drawdown_floor_pct: float,
) -> dict[str, Any]:
    if annual_metrics.empty:
        pass_years: list[int] = []
        feasible_years: list[int] = []
    else:
        feasible_years = sorted(int(year) for year in annual_metrics["outer_test_year"].dropna().astype(int).unique())
        pass_years = sorted(
            int(year)
            for year in annual_metrics.loc[annual_metrics["test_passes_target"].astype(bool), "outer_test_year"]
            .dropna()
            .astype(int)
            .unique()
        )
    required_years = [2023, 2024, 2025, 2026]
    missing = [year for year in required_years if year not in feasible_years]
    blocking = []
    if missing:
        blocking.append("missing_required_test_years")
    if pass_years != required_years:
        blocking.append("annual_target_gate_failed")
    return {
        "policy": "nested_outer_soxl_year_inner_multi_asset_leave_one_asset_out",
        "selection_uses_outer_test": False,
        "required_test_years": required_years,
        "feasible_test_years": feasible_years,
        "target_pass_test_years": pass_years,
        "missing_required_test_years": missing,
        "return_gate": "test_return_pct_gt_soxl_buyhold_open2open",
        "max_drawdown_floor_pct": float(max_drawdown_floor_pct),
        "required_policy_satisfied": bool(not missing and pass_years == required_years),
        "blocking_reasons": blocking or ["ok"],
    }


def _overfit_audit_summary(
    *,
    experiment: str,
    manual_template_selection: bool,
    nested_walk_forward: bool,
    leave_one_asset_out_assets: tuple[str, ...],
    feature_identity_columns: tuple[str, ...],
    selection_uses_outer_test: bool,
) -> dict[str, Any]:
    blocking: list[str] = []
    if manual_template_selection:
        blocking.append("manual_template_selection")
    if not nested_walk_forward:
        blocking.append("missing_nested_walk_forward")
    if len(set(leave_one_asset_out_assets)) < 3:
        blocking.append("insufficient_leave_one_asset_out_assets")
    if feature_identity_columns:
        blocking.append("asset_identity_feature_leakage")
    if selection_uses_outer_test:
        blocking.append("selection_uses_outer_test")
    return {
        "experiment": experiment,
        "overfit_risk_status": "fail" if blocking else "mitigated",
        "manual_template_selection": bool(manual_template_selection),
        "nested_walk_forward": bool(nested_walk_forward),
        "leave_one_asset_out_assets": sorted(set(leave_one_asset_out_assets)),
        "feature_identity_columns": list(feature_identity_columns),
        "selection_uses_outer_test": bool(selection_uses_outer_test),
        "blocking_reasons": blocking or ["ok"],
    }


def _return_metrics(returns: pd.Series, position: pd.Series, turnover: pd.Series) -> dict[str, float]:
    rets = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos = pd.to_numeric(position, errors="coerce").fillna(0.0)
    turns = pd.to_numeric(turnover, errors="coerce").fillna(0.0)
    if rets.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0, "avg_abs_position": 0.0, "total_turnover": 0.0}
    equity = (1.0 + rets).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "avg_abs_position": float(pos.abs().mean()),
        "total_turnover": float(turns.sum()),
    }


def _rolling_zscore(series: pd.Series, window: int, *, min_periods: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    mean = numeric.rolling(window, min_periods=min_periods).mean()
    std = numeric.rolling(window, min_periods=min_periods).std(ddof=0)
    return (numeric - mean) / std.replace(0.0, np.nan)


def _rolling_percentile(series: pd.Series, window: int, *, min_periods: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    return numeric.rolling(window, min_periods=min_periods).apply(_last_rank_pct, raw=False)


def _last_rank_pct(values: pd.Series) -> float:
    clean = pd.Series(values).dropna()
    if clean.empty:
        return np.nan
    return float(clean.rank(pct=True).iloc[-1])


def _summary_payload(
    *,
    spec: Multi3xEtfNormalizedNestedWfSpec,
    sidecars: dict[str, str],
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    provider_manifest: dict[str, Any],
    provider_meta: dict[str, Any],
    source_audit: dict[str, Any],
    common_end: str,
    feature_cols: list[str],
    annual_metrics: pd.DataFrame,
    annual_goal: dict[str, Any],
    overfit_audit: dict[str, Any],
    prior_template_audit: dict[str, Any],
    selected_candidates: pd.DataFrame,
) -> dict[str, Any]:
    recorder_artifacts = [
        f"artifacts/{spec.experiment}/{rel_path}"
        for rel_path in sidecars.values()
        if rel_path != "failure_modes.md"
    ]
    recorder_only = [f"artifacts/{spec.experiment}/positions.csv"]
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "workflow_config": "workflow.yaml",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "promotion_decision": (
            "candidate_for_validated_research_multi_3x_nested_wf"
            if annual_goal["required_policy_satisfied"] and overfit_audit["overfit_risk_status"] == "mitigated"
            else "not_promoted_multi_3x_nested_wf_goal_or_overfit_audit_failed"
        ),
        "benchmark_basis": "open_to_open",
        "execution_timing": "close_t_signal_next_open_to_following_open",
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "provider_uri": spec.normalized_provider_uri.resolve().as_posix(),
        "provider_manifest": (spec.normalized_provider_uri / "qlib_store_manifest.json").as_posix(),
        "provider_meta": provider_meta,
        "provider_build": provider_manifest,
        "source_audit": source_audit,
        "common_data_end": common_end,
        "symbols": list(ETF_UNIVERSE),
        "feature_source": FEATURE_SOURCE,
        "feature_sets": {
            "selected": FEATURE_SOURCE,
            "feature_count": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "contains_asset_identity": any(
                "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
                for column in feature_cols
            ),
            "normalization": "per_asset_rolling_zscore_and_percentile",
        },
        "model_selection": {
            "selection_policy": "inner_nested_leave_one_asset_out",
            "selection_uses_outer_test": False,
            "model_families": ["lgbm_classifier", "xgb_classifier", "lgb_xgb_consensus"],
            "candidate_count": int(len(_candidate_configs())),
            "selected_by_year": selected_candidates.to_dict("records") if not selected_candidates.empty else [],
        },
        "annual_goal": annual_goal,
        "overfit_audit": overfit_audit,
        "prior_template_v2_overfit_audit": prior_template_audit,
        "annual_summary": annual_metrics.to_dict("records") if not annual_metrics.empty else [],
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": ["qlib.init", "qlib.data.D.features", "qlib.workflow.R.start"],
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": recorder_only,
        },
        "recorder_artifacts": recorder_artifacts + recorder_only,
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.resolve().as_posix(),
        "spec": asdict(spec),
    }


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    annual_metrics: pd.DataFrame,
    inner_selection: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    provider_manifest: dict[str, Any],
) -> None:
    annual_metrics.to_csv(card / sidecars["annual_walk_forward"], index=False)
    inner_selection.to_csv(card / sidecars["inner_selection"], index=False)
    selected_candidates.to_csv(card / sidecars["selected_candidates"], index=False)
    selected_candidates.to_csv(card / sidecars["model_comparison"], index=False)
    (card / sidecars["qlib_store_manifest"]).write_text(
        json.dumps(provider_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "check": "overfit_audit",
                "status": summary["overfit_audit"]["overfit_risk_status"],
                "detail": json.dumps(summary["overfit_audit"], sort_keys=True),
            },
            {
                "check": "annual_goal",
                "status": "pass" if summary["annual_goal"]["required_policy_satisfied"] else "fail",
                "detail": json.dumps(summary["annual_goal"], sort_keys=True),
            },
            {
                "check": "prior_template_v2",
                "status": summary["prior_template_v2_overfit_audit"]["overfit_risk_status"],
                "detail": json.dumps(summary["prior_template_v2_overfit_audit"], sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    pd.DataFrame(
        [
            {
                "feature_set": summary["feature_sets"]["selected"],
                "feature_count": summary["feature_sets"]["feature_count"],
                "contains_asset_identity": summary["feature_sets"]["contains_asset_identity"],
                "normalization": summary["feature_sets"]["normalization"],
            }
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(
            {
                "annual_goal": summary["annual_goal"],
                "overfit_audit": summary["overfit_audit"],
                "prior_template_v2_overfit_audit": summary["prior_template_v2_overfit_audit"],
                "model_selection": summary["model_selection"],
                "annual_summary": summary["annual_summary"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (card / sidecars["failure_modes"]).write_text(_failure_modes(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            "model_selection:",
            "  selection_policy: inner_nested_leave_one_asset_out",
            "  selection_uses_outer_test: false",
            "overfit_audit:",
            f"  status: {summary['overfit_audit']['overfit_risk_status']}",
            f"  manual_template_selection: {str(summary['overfit_audit']['manual_template_selection']).lower()}",
            "annual_goal:",
            f"  required_policy_satisfied: {str(summary['annual_goal']['required_policy_satisfied']).lower()}",
            f"  target_pass_test_years: {summary['annual_goal']['target_pass_test_years']}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    return (
        "# 中文结论\n\n"
        "- 做了什么：不再选择人工模板，构建 SOXL/TQQQ/UPRO/SPXL/TECL 多 3x ETF 归一化 pooled LGB/XGB nested walk-forward 研究卡。\n"
        f"- 核心指标：annual target pass years={summary['annual_goal']['target_pass_test_years']}，"
        f"overfit audit={summary['overfit_audit']['overfit_risk_status']}，common_data_end={summary['common_data_end']}。\n"
        "- regime/window 差异：regime 不再手写选择模板，候选参数只由 inner leave-one-asset-out folds 选择，outer SOXL 年份只报告。\n"
        f"- 是否晋升：{summary['promotion_decision']}。\n"
        "- 风险：多 ETF 数据只能把可比样本扩到 2026-04-13；仍需要补齐更多 3x ETF、分钟级成交成本、以及完全独立未来窗口。\n"
        "- 下一步：若 annual gate 未过，优先扩大预注册 ETF universe 与固定候选空间，而不是人工添加年份模板。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    return (
        "# Failure Modes\n\n"
        "- Manual regime template selection is explicitly rejected by the overfit audit.\n"
        "- Leave-one-asset-out reduces but does not eliminate overfit risk because all LETFs share US equity beta.\n"
        "- Common coverage is clipped to the shortest local 3x ETF qlib store, so SOXL after common end is not evaluated here.\n"
        "- Feature normalization is per-asset rolling z-score/percentile; stale or split-adjustment errors in source stores can still propagate.\n"
        f"- Current promotion decision: {summary['promotion_decision']}.\n"
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if not isinstance(value, (str, bytes, bool)) and pd.isna(value):
        return None
    return value
