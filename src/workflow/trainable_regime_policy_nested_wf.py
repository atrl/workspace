from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.data import QlibProviderConfig, build_qlib_store_from_frame, init_qlib_provider
from src.factor.soxx_constituent_factors import (
    SOXX_CONSTITUENT_FACTOR_SOURCE,
    SOXX_CONSTITUENT_LIMITATIONS,
    build_soxx_constituent_factor_frame,
)
from src.workflow.cognitive_alpha_goal_codex_exec import (
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_DOCS,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import (
    ETF_UNIVERSE,
    FEATURE_SOURCE,
    _attach_normalized_features,
    _json_ready,
    _load_multi_3x_raw_panel,
    _model_feature_columns,
    _overfit_audit_summary,
    _resolve_root,
    _return_metrics,
)


EXPERIMENT = "trainable_regime_policy_nested_wf_v1"
CARD_TYPE = "trainable_regime_policy_nested_wf_research_card"
SELECTION_POLICY = "nested_inner_soxl_calibrated_learned_position_no_template_grid"
CALIBRATION_ASSET = "SOXL"
REGIME_TARGET_COL = "trainable_regime_target"
REGIME_TARGET_CODE_COL = "trainable_regime_target_code"
CRASH_TARGET_COL = "crash_probability_target"
MAX_ADVERSE_TARGET_COL = "max_adverse_return_target"
CRASH_20D_TARGET_COL = "crash_probability_20d_target"
MAX_ADVERSE_20D_TARGET_COL = "max_adverse_return_20d_target"
RISK_TARGET_COL = "risk_probability_target"
VOL_ADJUSTED_TARGET_COL = "vol_adjusted_return_target"
ALPHA_TARGET_COL = "alpha_opportunity_target"
POSITION_TARGET_COL = "learned_position_target"
INTRADAY_RISK_STATE_COL = "intraday_risk_state"
INTRADAY_EXECUTION_TRIGGERED_COL = "intraday_execution_triggered"
INTRADAY_OPEN_TO_TRIGGER_RETURN_COL = "intraday_open_to_trigger_return"
INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL = "intraday_trigger_to_close_return"
REGIME_LABELS = ["defensive", "rebound", "selective", "trend"]
SOXX_CONSTITUENT_MODEL_RAW_TOKENS = ("breadth", "downside", "dispersion")


@dataclass(frozen=True)
class TrainableRegimePolicyNestedWfSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    outer_test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    inner_validation_years: int = 8
    max_common_end: str = "2026-04-13"
    max_drawdown_floor_pct: float = -20.0
    cost_per_turnover: float = 0.0003

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.experiment

    @property
    def intraday_5min_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_5min"

    @property
    def intraday_15min_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"


def run_trainable_regime_policy_nested_wf_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = TrainableRegimePolicyNestedWfSpec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: TrainableRegimePolicyNestedWfSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    raw_panel, source_audit = _load_multi_3x_raw_panel(spec)  # type: ignore[arg-type]
    normalized = _attach_normalized_features(raw_panel)
    normalized = _attach_trainable_policy_targets(normalized)
    common_end = min(
        pd.Timestamp(spec.max_common_end),
        pd.to_datetime(normalized.groupby("instrument")["datetime"].max()).min(),
    ).date().isoformat()
    normalized = normalized.loc[pd.to_datetime(normalized["datetime"]).le(pd.Timestamp(common_end))].copy()
    constituent_factor_frame, constituent_audit = _load_soxx_constituent_factor_frame(
        spec,
        start=spec.train_start,
        end=common_end,
    )
    normalized, constituent_feature_cols = _attach_soxx_constituent_model_features(
        normalized,
        constituent_factor_frame,
    )
    intraday_risk_state, intraday_audit = _build_intraday_risk_state_daily(
        spec,
        start=spec.train_start,
        end=common_end,
    )
    normalized = _attach_intraday_risk_state(normalized, intraday_risk_state)
    provider_manifest = _build_policy_provider(spec, normalized, as_of=common_end)
    provider_frame, provider_meta = _load_policy_provider(spec, normalized, as_of=common_end)
    feature_cols = _policy_feature_columns(provider_frame)
    feature_set_configs = _policy_feature_set_configs(provider_frame)
    candidates = _policy_candidate_configs()

    annual_rows: list[dict[str, Any]] = []
    inner_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    policy_diagnostics: list[dict[str, Any]] = []

    for outer_year in spec.outer_test_years:
        inner_metrics = _inner_policy_metrics(
            provider_frame,
            outer_test_year=int(outer_year),
            candidates=candidates,
            feature_set_configs=feature_set_configs,
            cost_per_turnover=spec.cost_per_turnover,
            max_inner_years=spec.inner_validation_years,
        )
        selected = _select_policy_from_inner_metrics(
            inner_metrics,
            outer_test_year=int(outer_year),
            required_heldout_assets=tuple(ETF_UNIVERSE),
        )
        selected_rows.append(selected)
        outer_metrics, outer_positions = _outer_soxl_policy_test(
            provider_frame,
            outer_test_year=int(outer_year),
            selected_candidate=selected,
            candidates=candidates,
            feature_set_configs=feature_set_configs,
            common_end=common_end,
            cost_per_turnover=spec.cost_per_turnover,
            max_drawdown_floor_pct=spec.max_drawdown_floor_pct,
        )
        annual_rows.append(outer_metrics)
        outer_positions["outer_test_year"] = int(outer_year)
        position_frames.append(outer_positions)
        inner_metrics["outer_test_year"] = int(outer_year)
        inner_frames.append(inner_metrics)
        policy_diagnostics.append(_policy_diagnostic_row(outer_metrics, selected))

    annual_metrics = pd.DataFrame(annual_rows)
    inner_selection = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    positions = pd.concat(position_frames, ignore_index=True) if position_frames else pd.DataFrame()
    selected_candidates = pd.DataFrame(selected_rows)
    policy_diagnostics_frame = pd.DataFrame(policy_diagnostics)
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
    sidecars = {
        "annual_walk_forward": "annual_walk_forward.csv",
        "inner_selection": "inner_selection.csv",
        "selected_candidates": "selected_candidates.csv",
        "model_comparison": "model_comparison.csv",
        "policy_diagnostics": "policy_diagnostics.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "feature_sets": "feature_sets.csv",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix="trainable_regime_policy_nested_wf_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        positions.to_csv(positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                source_symbols=",".join(ETF_UNIVERSE),
                feature_source=FEATURE_SOURCE,
                model_families="lgbm_classifier,xgb_classifier,lgbm_regressor,xgb_regressor,lgb_xgb_consensus",
                regime_classifier="supervised_trainable_policy_classifier",
                nested_walk_forward="true",
                leave_one_asset_out="true",
                selection_uses_outer_test="false",
                intraday_risk_state="5min,15min,risk_state,execution_replay",
                soxx_constituent_features="lagged_model_features",
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
                feature_set_configs=feature_set_configs,
                annual_metrics=annual_metrics,
                annual_goal=annual_goal,
                overfit_audit=overfit_audit,
                selected_candidates=selected_candidates,
                intraday_audit=intraday_audit,
                constituent_audit=constituent_audit,
                constituent_feature_cols=constituent_feature_cols,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                annual_metrics=annual_metrics,
                inner_selection=inner_selection,
                selected_candidates=selected_candidates,
                policy_diagnostics=policy_diagnostics_frame,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path=spec.experiment)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(Path(recorder.get_local_dir()).resolve())
    _write_summary_files(card, summary)
    return summary


def _remove_qlib_code_snapshot_artifacts(recorder_local_dir: Path) -> None:
    artifacts_dir = recorder_local_dir / "artifacts"
    for name in ["code_status.txt", "code_diff.txt", "code_cached.txt"]:
        path = artifacts_dir / name
        if path.exists():
            path.unlink()


def _attach_trainable_policy_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    future_return = pd.to_numeric(out[LABEL_OPEN_TO_OPEN_5D_COL], errors="coerce").fillna(0.0)
    realized_vol = pd.to_numeric(out.get("realized_vol_20", 0.0), errors="coerce").fillna(0.0).abs()
    ret20 = pd.to_numeric(out.get("ret_20", 0.0), errors="coerce").fillna(0.0)
    ret60 = pd.to_numeric(out.get("ret_60", 0.0), errors="coerce").fillna(0.0)
    drawdown60 = pd.to_numeric(out.get("drawdown_60", 0.0), errors="coerce").fillna(0.0)

    out[ALPHA_TARGET_COL] = future_return.gt(0.0).astype(int)
    out[CRASH_TARGET_COL] = future_return.le(-0.08).astype(int)
    if "open" in out.columns:
        max_adverse = _future_open_path_min_adverse(out, offsets=range(2, 7))
        max_adverse_20d = _future_open_path_min_adverse(out, offsets=range(2, 22))
    else:
        max_adverse = future_return.clip(upper=0.0)
        max_adverse_20d = future_return.clip(upper=0.0)
    out[MAX_ADVERSE_TARGET_COL] = pd.to_numeric(max_adverse, errors="coerce").fillna(0.0)
    out[MAX_ADVERSE_20D_TARGET_COL] = pd.to_numeric(max_adverse_20d, errors="coerce").fillna(0.0)
    out[CRASH_20D_TARGET_COL] = (
        out[CRASH_TARGET_COL].eq(1) | out[MAX_ADVERSE_20D_TARGET_COL].le(-0.20)
    ).astype(int)
    out[RISK_TARGET_COL] = (
        out[CRASH_TARGET_COL].eq(1)
        | out[MAX_ADVERSE_TARGET_COL].le(-0.12)
        | out[CRASH_20D_TARGET_COL].eq(1)
    ).astype(int)
    out[VOL_ADJUSTED_TARGET_COL] = (future_return / (realized_vol + 0.02)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive_exposure = pd.Series(
        np.where(
            future_return.gt(0.0),
            1.0,
            ((future_return + 0.04) / 0.04).clip(lower=0.0, upper=1.0) * 0.25,
        ),
        index=out.index,
        dtype="float64",
    )
    adverse_survival = ((out[MAX_ADVERSE_TARGET_COL] + 0.18) / 0.18).clip(lower=0.0, upper=1.0)
    long_adverse_survival = ((out[MAX_ADVERSE_20D_TARGET_COL] + 0.30) / 0.30).clip(lower=0.0, upper=1.0)
    crash_survival = pd.Series(np.where(future_return.le(-0.08), 0.0, 1.0), index=out.index, dtype="float64")
    vol_survival = (0.35 / (realized_vol + 0.20)).clip(lower=0.65, upper=1.0)
    out[POSITION_TARGET_COL] = (
        positive_exposure * adverse_survival * long_adverse_survival * crash_survival * vol_survival
    ).clip(lower=0.0, upper=1.0)
    regimes = np.select(
        [
            out[RISK_TARGET_COL].eq(1) | out[VOL_ADJUSTED_TARGET_COL].lt(-0.75),
            ret60.lt(-0.20) & drawdown60.lt(-0.25) & future_return.gt(0.0),
            ret60.gt(0.10) & ret20.gt(0.0) & future_return.gt(0.0),
        ],
        ["defensive", "rebound", "trend"],
        default="selective",
    )
    out[REGIME_TARGET_COL] = pd.Series(regimes, index=out.index).astype(str)
    regime_code = {label: index for index, label in enumerate(REGIME_LABELS)}
    out[REGIME_TARGET_CODE_COL] = out[REGIME_TARGET_COL].map(regime_code).astype("float64")
    return out


def _future_open_path_min_adverse(frame: pd.DataFrame, *, offsets: range) -> pd.Series:
    adverse_pieces: list[pd.Series] = []
    for _, group in frame.sort_values(["instrument", "datetime"]).groupby("instrument", sort=False):
        open_ = pd.to_numeric(group["open"], errors="coerce").astype("float64")
        entry = open_.shift(-1)
        future_path = pd.concat(
            [(open_.shift(-offset) / entry - 1.0) for offset in offsets],
            axis=1,
        )
        adverse_pieces.append(future_path.min(axis=1).reindex(group.index))
    return pd.concat(adverse_pieces).reindex(frame.index) if adverse_pieces else pd.Series(np.nan, index=frame.index)


def _load_soxx_constituent_factor_frame(
    spec: TrainableRegimePolicyNestedWfSpec,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = spec.soxx_constituent_provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing SOXX constituent qlib manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instruments = [str(item).upper() for item in manifest.get("instruments", [])]
    if not instruments:
        raise ValueError(f"SOXX constituent qlib manifest has no instruments: {manifest_path}")
    factor_frame = build_soxx_constituent_factor_frame(
        QlibProviderConfig(provider_uri=spec.soxx_constituent_provider_uri, market="all", region="us", kernels=1),
        instruments=instruments,
        start=start,
        end=end,
        lookbacks=(5, 20),
        lag_days=1,
        top_n=5,
    )
    raw_feature_cols = [
        column
        for column in list(factor_frame.attrs.get("feature_cols", []))
        if column in factor_frame.columns and column != "datetime"
    ]
    model_raw_feature_cols = [
        column
        for column in raw_feature_cols
        if any(token in column for token in SOXX_CONSTITUENT_MODEL_RAW_TOKENS)
    ]
    factor_frame, feature_cols = _append_soxx_constituent_normalized_features(
        factor_frame,
        raw_feature_cols=model_raw_feature_cols,
    )
    feature_cols = [
        column
        for column in feature_cols
        if column in factor_frame.columns and column != "datetime"
    ]
    factor_frame.attrs["raw_feature_cols"] = raw_feature_cols
    factor_frame.attrs["feature_cols"] = feature_cols
    return factor_frame, {
        "status": "ok",
        "role": "normalized_lagged_model_features_not_template_not_gate",
        "source": SOXX_CONSTITUENT_FACTOR_SOURCE,
        "provider_uri": spec.soxx_constituent_provider_uri.resolve().as_posix(),
        "manifest": manifest_path.resolve().as_posix(),
        "raw_manifest": manifest.get("raw_manifest"),
        "as_of": manifest.get("as_of"),
        "calendar_start": manifest.get("calendar_start"),
        "calendar_end": manifest.get("calendar_end"),
        "instruments": instruments,
        "instrument_count": int(len(instruments)),
        "raw_feature_cols": raw_feature_cols,
        "raw_feature_count": int(len(raw_feature_cols)),
        "model_raw_feature_cols": model_raw_feature_cols,
        "model_raw_feature_count": int(len(model_raw_feature_cols)),
        "model_raw_feature_filter": "breadth_downside_dispersion_diffusion_only",
        "excluded_raw_feature_reason": "weighted_equal_top_minus_rest_and_available_count_overlap_index_beta_or_are_less_direct_diffusion_signals",
        "feature_cols": feature_cols,
        "feature_count": int(len(feature_cols)),
        "normalization": "rolling_z60_and_pctile252_on_lagged_constituent_aggregates",
        "lookbacks": list(factor_frame.attrs.get("lookbacks", [5, 20])),
        "lag_days": int(factor_frame.attrs.get("lag_days", 1)),
        "top_n": int(factor_frame.attrs.get("top_n", 5)),
        "limitations": list(factor_frame.attrs.get("limitations", SOXX_CONSTITUENT_LIMITATIONS)),
    }


def _append_soxx_constituent_normalized_features(
    factor_frame: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = factor_frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    normalized_cols: list[str] = []
    for column in raw_feature_cols:
        values = pd.to_numeric(out[column], errors="coerce").astype("float64")
        mean60 = values.rolling(60, min_periods=20).mean()
        std60 = values.rolling(60, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        z_col = f"{column}_z60"
        pctile_col = f"{column}_pctile252"
        out[z_col] = ((values - mean60) / std60).replace([np.inf, -np.inf], np.nan)
        out[pctile_col] = values.rolling(252, min_periods=40).rank(pct=True)
        normalized_cols.extend([z_col, pctile_col])
    out.attrs.update(factor_frame.attrs)
    return out, normalized_cols


def _attach_soxx_constituent_model_features(
    frame: pd.DataFrame,
    constituent_factor_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        column
        for column in list(constituent_factor_frame.attrs.get("feature_cols", []))
        if column in constituent_factor_frame.columns and column != "datetime"
    ]
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    if not feature_cols:
        return out, []
    factors = constituent_factor_frame[["datetime", *feature_cols]].copy()
    factors["datetime"] = pd.to_datetime(factors["datetime"]).dt.normalize()
    return out.merge(factors, on="datetime", how="left"), feature_cols


def _build_intraday_risk_state_daily(
    spec: TrainableRegimePolicyNestedWfSpec,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for freq, provider_uri in [("5min", spec.intraday_5min_provider_uri), ("15min", spec.intraday_15min_provider_uri)]:
        daily, audit = _intraday_provider_daily_state(provider_uri=provider_uri, freq=freq, start=start, end=end)
        frames.append(daily)
        audits[freq] = audit
    if not frames:
        return pd.DataFrame({"datetime": [], INTRADAY_RISK_STATE_COL: []}), {"status": "missing_intraday_state"}
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="datetime", how="outer")
    risk_cols = [column for column in out.columns if column.endswith("_risk_state")]
    score_cols = [column for column in out.columns if column.endswith("_risk_score")]
    out[INTRADAY_RISK_STATE_COL] = out[risk_cols].max(axis=1).fillna(0.0) if risk_cols else 0.0
    out["intraday_risk_score"] = out[score_cols].max(axis=1).fillna(0.0) if score_cols else 0.0
    out = _attach_intraday_execution_replay_columns(out)
    return out.sort_values("datetime").reset_index(drop=True), {
        "status": "ok",
        "uses_5min": "5min" in audits and audits["5min"].get("status") == "ok",
        "uses_15min": "15min" in audits and audits["15min"].get("status") == "ok",
        "providers": audits,
        "rows": int(len(out)),
        "role": "risk_cap_and_intraday_execution_replay_not_alpha_template",
        "execution_replay_fields": True,
    }


def _attach_intraday_execution_replay_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[INTRADAY_EXECUTION_TRIGGERED_COL] = 0.0
    out[INTRADAY_OPEN_TO_TRIGGER_RETURN_COL] = 0.0
    out[INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL] = 0.0
    out["intraday_execution_source_freq"] = "none"
    for idx, row in out.iterrows():
        selected_freq = None
        for freq in ["5min", "15min"]:
            trigger_col = f"intraday_{freq}_execution_triggered"
            if trigger_col in out.columns and float(pd.to_numeric(pd.Series([row.get(trigger_col, 0.0)]), errors="coerce").fillna(0.0).iloc[0]) >= 0.5:
                selected_freq = freq
                break
        if selected_freq is None:
            selected_freq = "5min" if "intraday_5min_day_ret" in out.columns else "15min"
        day_ret = float(pd.to_numeric(pd.Series([row.get(f"intraday_{selected_freq}_day_ret", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        open_to_trigger = float(
            pd.to_numeric(
                pd.Series([row.get(f"intraday_{selected_freq}_open_to_trigger_return", day_ret)]),
                errors="coerce",
            )
            .fillna(day_ret)
            .iloc[0]
        )
        trigger_to_close = float(
            pd.to_numeric(
                pd.Series([row.get(f"intraday_{selected_freq}_trigger_to_close_return", 0.0)]),
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[0]
        )
        triggered = float(
            pd.to_numeric(
                pd.Series([row.get(f"intraday_{selected_freq}_execution_triggered", 0.0)]),
                errors="coerce",
            )
            .fillna(0.0)
            .iloc[0]
        )
        out.at[idx, INTRADAY_EXECUTION_TRIGGERED_COL] = triggered
        out.at[idx, INTRADAY_OPEN_TO_TRIGGER_RETURN_COL] = open_to_trigger
        out.at[idx, INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL] = trigger_to_close
        out.at[idx, "intraday_execution_source_freq"] = selected_freq
    return out


def _intraday_provider_daily_state(
    *,
    provider_uri: Path,
    freq: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = QlibProviderConfig(provider_uri=provider_uri, market="all", region="us", freq=freq, kernels=1)
    provider_meta = init_qlib_provider(config)
    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$bar_index", "$bar_count"]
    raw = D.features(["SOXL", "SOXX"], fields, start_time=start, end_time=end, freq=freq).reset_index()
    if raw.empty:
        return pd.DataFrame({"datetime": [], f"intraday_{freq}_risk_state": []}), {
            "status": "empty",
            "provider_uri": provider_meta["provider_uri"],
        }
    raw = raw.rename(columns={field: field.removeprefix("$") for field in fields})
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["date"] = raw["datetime"].dt.normalize()
    wide = (
        raw.pivot_table(
            index=["datetime", "date"],
            columns="instrument",
            values=["open", "high", "low", "close", "volume", "bar_index"],
            aggfunc="last",
        )
        .sort_index()
    )
    wide.columns = [f"{field}_{instrument}".lower() for field, instrument in wide.columns]
    wide = wide.reset_index()
    rows = _intraday_execution_rows_from_wide(wide, freq=freq)
    out = pd.DataFrame(rows)
    return out, {
        "status": "ok",
        "provider_uri": provider_meta["provider_uri"],
        "calendar_start": provider_meta["calendar_start"],
        "calendar_end": provider_meta["calendar_end"],
        "rows": int(len(out)),
        "lookback_only": True,
        "execution_replay_fields": True,
    }


def _intraday_execution_rows_from_wide(wide: pd.DataFrame, *, freq: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for date, group in wide.groupby("date", sort=True):
        g = group.sort_values("datetime")
        if g.empty or "open_soxl" not in g or "close_soxl" not in g:
            continue
        soxl_open_values = pd.to_numeric(g["open_soxl"], errors="coerce").dropna()
        soxl_close = pd.to_numeric(g["close_soxl"], errors="coerce")
        soxx_open_values = pd.to_numeric(g.get("open_soxx", pd.Series(dtype=float)), errors="coerce").dropna()
        soxx_close = pd.to_numeric(g.get("close_soxx", pd.Series(dtype=float)), errors="coerce")
        if soxl_open_values.empty or soxl_close.dropna().empty or soxx_open_values.empty:
            continue
        soxl_open = float(soxl_open_values.iloc[0])
        soxx_open = float(soxx_open_values.iloc[0])
        if not np.isfinite(soxl_open) or not np.isfinite(soxx_open) or soxl_open <= 0 or soxx_open <= 0:
            continue
        day_ret = float(soxl_close.dropna().iloc[-1] / soxl_open - 1.0)
        day_soxx_ret = float(soxx_close.dropna().iloc[-1] / soxx_open - 1.0) if not soxx_close.dropna().empty else 0.0
        drawdown_from_open = float(soxl_close.min() / soxl_open - 1.0)
        dislocation = day_ret - 3.0 * day_soxx_ret
        soxl_path_ret = (soxl_close / soxl_open - 1.0).replace([np.inf, -np.inf], np.nan)
        soxx_path_ret = (soxx_close / soxx_open - 1.0).replace([np.inf, -np.inf], np.nan)
        path_dislocation = (soxl_path_ret - 3.0 * soxx_path_ret).replace([np.inf, -np.inf], np.nan)
        trigger_mask = soxl_path_ret.le(-0.08) | path_dislocation.le(-0.04)
        trigger_indices = list(trigger_mask[trigger_mask.fillna(False)].index)
        trigger_bar_index = -1
        execution_triggered = 0.0
        open_to_trigger_return = day_ret
        trigger_to_close_return = 0.0
        if trigger_indices:
            trigger_idx = trigger_indices[0]
            trigger_close = float(soxl_close.loc[trigger_idx])
            final_close = float(soxl_close.dropna().iloc[-1])
            if np.isfinite(trigger_close) and trigger_close > 0:
                trigger_bar_index = int(g.index.get_loc(trigger_idx))
                execution_triggered = 1.0
                open_to_trigger_return = float(trigger_close / soxl_open - 1.0)
                trigger_to_close_return = float(final_close / trigger_close - 1.0)
        late = g.tail(3)
        late_open = pd.to_numeric(late["open_soxl"], errors="coerce").dropna()
        late_close = pd.to_numeric(late["close_soxl"], errors="coerce").dropna()
        late_ret = float(late_close.iloc[-1] / late_open.iloc[0] - 1.0) if not late_open.empty and not late_close.empty else 0.0
        ret_cc = soxl_close.pct_change().replace([np.inf, -np.inf], np.nan)
        realized_vol = float(ret_cc.std(ddof=0)) if ret_cc.notna().sum() >= 3 else 0.0
        risk_score = max(
            0.0,
            -drawdown_from_open / 0.08,
            -late_ret / 0.03,
            -dislocation / 0.04,
            realized_vol / 0.035,
        )
        rows.append(
            {
                "datetime": pd.Timestamp(date).normalize(),
                f"intraday_{freq}_day_ret": day_ret,
                f"intraday_{freq}_drawdown_from_open": drawdown_from_open,
                f"intraday_{freq}_late_ret": late_ret,
                f"intraday_{freq}_soxl_minus_3x_soxx": dislocation,
                f"intraday_{freq}_realized_vol": realized_vol,
                f"intraday_{freq}_risk_score": risk_score,
                f"intraday_{freq}_risk_state": float(risk_score >= 1.0),
                f"intraday_{freq}_execution_triggered": execution_triggered,
                f"intraday_{freq}_trigger_bar_index": trigger_bar_index,
                f"intraday_{freq}_open_to_trigger_return": open_to_trigger_return,
                f"intraday_{freq}_trigger_to_close_return": trigger_to_close_return,
            }
        )
    return rows


def _attach_intraday_risk_state(frame: pd.DataFrame, intraday_state: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    if intraday_state.empty:
        out[INTRADAY_RISK_STATE_COL] = 0.0
        out["intraday_risk_score"] = 0.0
        return out
    state = intraday_state.copy()
    state["datetime"] = pd.to_datetime(state["datetime"]).dt.normalize()
    out = out.merge(state, on="datetime", how="left")
    risk_cols = [column for column in out.columns if column.startswith("intraday_")]
    for column in risk_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    mask = out["instrument"].astype(str).str.upper().eq("SOXL")
    out.loc[~mask, [INTRADAY_RISK_STATE_COL, "intraday_risk_score"]] = 0.0
    if INTRADAY_EXECUTION_TRIGGERED_COL in out.columns:
        out.loc[~mask, INTRADAY_EXECUTION_TRIGGERED_COL] = 0.0
    if INTRADAY_OPEN_TO_TRIGGER_RETURN_COL in out.columns and OPEN_TO_OPEN_RETURN_COL in out.columns:
        out.loc[~mask, INTRADAY_OPEN_TO_TRIGGER_RETURN_COL] = pd.to_numeric(
            out.loc[~mask, OPEN_TO_OPEN_RETURN_COL],
            errors="coerce",
        ).fillna(0.0)
    if INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL in out.columns:
        out.loc[~mask, INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL] = 0.0
    return out


def _build_policy_provider(
    spec: TrainableRegimePolicyNestedWfSpec,
    frame: pd.DataFrame,
    *,
    as_of: str,
) -> dict[str, Any]:
    store_cols = [
        "instrument",
        "datetime",
        "open",
        "close",
        "volume",
        OPEN_TO_OPEN_RETURN_COL,
        LABEL_OPEN_TO_OPEN_5D_COL,
        ALPHA_TARGET_COL,
        CRASH_TARGET_COL,
        MAX_ADVERSE_TARGET_COL,
        CRASH_20D_TARGET_COL,
        MAX_ADVERSE_20D_TARGET_COL,
        RISK_TARGET_COL,
        VOL_ADJUSTED_TARGET_COL,
        POSITION_TARGET_COL,
        REGIME_TARGET_CODE_COL,
        INTRADAY_RISK_STATE_COL,
        "intraday_risk_score",
        INTRADAY_EXECUTION_TRIGGERED_COL,
        INTRADAY_OPEN_TO_TRIGGER_RETURN_COL,
        INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL,
        *_policy_feature_columns(frame),
    ]
    store = frame.loc[:, [column for column in dict.fromkeys(store_cols) if column in frame.columns]].copy()
    return build_qlib_store_from_frame(
        store,
        provider_uri=spec.provider_uri,
        market="all",
        region="us",
        freq="day",
        as_of=as_of,
        source="qlib_trainable_regime_policy_multi_3x_panel",
    )


def _load_policy_provider(
    spec: TrainableRegimePolicyNestedWfSpec,
    frame: pd.DataFrame,
    *,
    as_of: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = QlibProviderConfig(provider_uri=spec.provider_uri, market="all", region="us", kernels=1)
    provider_meta = init_qlib_provider(config)
    from qlib.data import D

    fields = list(dict.fromkeys([
        "$open",
        "$close",
        "$volume",
        f"${OPEN_TO_OPEN_RETURN_COL}",
        f"${LABEL_OPEN_TO_OPEN_5D_COL}",
        f"${ALPHA_TARGET_COL}",
        f"${CRASH_TARGET_COL}",
        f"${MAX_ADVERSE_TARGET_COL}",
        f"${CRASH_20D_TARGET_COL}",
        f"${MAX_ADVERSE_20D_TARGET_COL}",
        f"${RISK_TARGET_COL}",
        f"${VOL_ADJUSTED_TARGET_COL}",
        f"${POSITION_TARGET_COL}",
        f"${REGIME_TARGET_CODE_COL}",
        f"${INTRADAY_RISK_STATE_COL}",
        "$intraday_risk_score",
        f"${INTRADAY_EXECUTION_TRIGGERED_COL}",
        f"${INTRADAY_OPEN_TO_TRIGGER_RETURN_COL}",
        f"${INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL}",
        *[f"${column}" for column in _policy_feature_columns(frame)],
    ]))
    qframe = D.features(list(ETF_UNIVERSE), fields, start_time=spec.train_start, end_time=as_of, freq="day")
    out = qframe.reset_index().rename(columns={field: field.removeprefix("$") for field in fields})
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    return out.sort_values(["instrument", "datetime"]).reset_index(drop=True), provider_meta


def _policy_feature_columns(frame: pd.DataFrame) -> list[str]:
    base = _base_policy_feature_columns(frame)
    constituent = _soxx_constituent_policy_feature_columns(frame)
    return list(dict.fromkeys([*base, *constituent]))


def _base_policy_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in _model_feature_columns(frame)
        if not column.startswith("soxx_const_") and column not in {INTRADAY_RISK_STATE_COL, "intraday_risk_score"}
    ]


def _soxx_constituent_policy_feature_columns(frame: pd.DataFrame) -> list[str]:
    constituent = [
        column
        for column in frame.columns
        if column.startswith("soxx_const_") and ("_z" in column or "_pctile" in column)
    ]
    return [column for column in dict.fromkeys(constituent) if column not in {INTRADAY_RISK_STATE_COL, "intraday_risk_score"}]


def _policy_feature_set_configs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    base_cols = _base_policy_feature_columns(frame)
    soxx_cols = _soxx_constituent_policy_feature_columns(frame)
    configs = [
        _feature_set_config(
            feature_set="base",
            feature_cols=base_cols,
            soxx_constituent_feature_cols=[],
        )
    ]
    if soxx_cols:
        configs.append(
            _feature_set_config(
                feature_set="base_soxx_diffusion",
                feature_cols=list(dict.fromkeys([*base_cols, *soxx_cols])),
                soxx_constituent_feature_cols=soxx_cols,
            )
        )
    return configs


def _feature_set_config(
    *,
    feature_set: str,
    feature_cols: list[str],
    soxx_constituent_feature_cols: list[str],
) -> dict[str, Any]:
    selected_soxx_cols = [column for column in soxx_constituent_feature_cols if column in feature_cols]
    return {
        "feature_set": feature_set,
        "feature_cols": list(dict.fromkeys(feature_cols)),
        "feature_count": int(len(list(dict.fromkeys(feature_cols)))),
        "contains_soxx_constituents": bool(selected_soxx_cols),
        "soxx_constituent_feature_count": int(len(selected_soxx_cols)),
        "soxx_constituent_feature_cols": selected_soxx_cols,
        "contains_asset_identity": any(
            "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
            for column in feature_cols
        ),
        "normalization": "per_asset_rolling_zscore_and_percentile",
    }


def _policy_candidate_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_family in ["lgbm", "xgb", "lgb_xgb_consensus"]:
        for suffix, intraday_replay, intraday_risk_cap in [
            ("learned_position", False, 1.0),
            ("learned_position_intraday_replay", True, 0.25),
        ]:
            rows.append(
                {
                    "candidate": f"{model_family}_{suffix}",
                    "model_family": model_family,
                    "policy_id": suffix,
                    "position_mode": "learned_position",
                    "alpha_floor": 0.0,
                    "risk_cut": 2.0,
                    "risk_cap": 1.0,
                    "trend_scale": 1.0,
                    "rebound_scale": 1.0,
                    "selective_scale": 1.0,
                    "defensive_scale": 1.0,
                    "intraday_risk_cap": intraday_risk_cap,
                    "intraday_execution_replay": intraday_replay,
                    "manual_policy_grid": False,
                    "position_target": POSITION_TARGET_COL,
                }
            )
    return rows


def _inner_policy_metrics(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    candidates: list[dict[str, Any]],
    feature_set_configs: list[dict[str, Any]],
    cost_per_turnover: float,
    max_inner_years: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set in feature_set_configs:
        feature_cols = list(feature_set["feature_cols"])
        data = _clean_policy_frame(frame, feature_cols)
        available = data.loc[data["datetime"].dt.year < int(outer_test_year)].copy()
        inner_years = [
            year
            for year in sorted(available["datetime"].dt.year.unique())
            if year >= 2015 and year <= int(outer_test_year) - 1
        ][-int(max_inner_years) :]
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
                prediction_cache = _predict_trainable_policy_cache(
                    train,
                    validation,
                    feature_cols=feature_cols,
                    candidates=candidates,
                )
                for candidate in candidates:
                    pred = prediction_cache[str(candidate["model_family"])]
                    metrics, _ = _simulate_alpha_risk_policy(
                        validation,
                        pred,
                        policy=candidate,
                        segment_name="inner_validation",
                        cost_per_turnover=cost_per_turnover,
                    )
                    buyhold = _buyhold_metrics(validation)
                    excess_return = float(metrics["selection_return_pct"] - buyhold["return_pct"])
                    rows.append(
                        {
                            **metrics,
                            "candidate": str(candidate["candidate"]),
                            "model": str(candidate["model_family"]),
                            "feature_set": str(feature_set["feature_set"]),
                            "feature_count": int(feature_set["feature_count"]),
                            "contains_soxx_constituents": bool(feature_set["contains_soxx_constituents"]),
                            "soxx_constituent_feature_count": int(feature_set["soxx_constituent_feature_count"]),
                            "inner_year": int(inner_year),
                            "heldout_asset": heldout_asset,
                            "buyhold_return_pct": buyhold["return_pct"],
                            "excess_return_pct": excess_return,
                            "beats_buyhold": bool(metrics["selection_return_pct"] > buyhold["return_pct"]),
                            "selection_uses_outer_test": False,
                        }
                    )
    return pd.DataFrame(rows)


def _select_policy_from_inner_metrics(
    metrics: pd.DataFrame,
    *,
    outer_test_year: int,
    required_heldout_assets: tuple[str, ...],
) -> dict[str, Any]:
    if metrics.empty:
        raise ValueError("trainable regime policy inner nested walk-forward produced no candidate metrics")
    rows = metrics.copy()
    rows["selection_return_pct"] = pd.to_numeric(rows["selection_return_pct"], errors="coerce").fillna(0.0)
    rows["selection_max_drawdown_pct"] = pd.to_numeric(rows["selection_max_drawdown_pct"], errors="coerce").fillna(-100.0)
    rows["selection_total_turnover"] = pd.to_numeric(rows["selection_total_turnover"], errors="coerce").fillna(0.0)
    rows["selection_avg_abs_position"] = pd.to_numeric(rows["selection_avg_abs_position"], errors="coerce").fillna(0.0)
    rows["excess_return_pct"] = pd.to_numeric(rows.get("excess_return_pct", 0.0), errors="coerce").fillna(0.0)
    rows["beats_buyhold"] = rows["beats_buyhold"].fillna(False).astype(bool)
    if "feature_set" not in rows.columns:
        rows["feature_set"] = "all_features"
    rows["feature_set"] = rows["feature_set"].fillna("all_features").astype(str)
    if "feature_count" not in rows.columns:
        rows["feature_count"] = 0
    if "contains_soxx_constituents" not in rows.columns:
        rows["contains_soxx_constituents"] = False
    if "soxx_constituent_feature_count" not in rows.columns:
        rows["soxx_constituent_feature_count"] = 0
    rows["feature_count"] = pd.to_numeric(rows["feature_count"], errors="coerce").fillna(0).astype(int)
    rows["contains_soxx_constituents"] = rows["contains_soxx_constituents"].fillna(False).astype(bool)
    rows["soxx_constituent_feature_count"] = (
        pd.to_numeric(rows["soxx_constituent_feature_count"], errors="coerce").fillna(0).astype(int)
    )
    rows["passes_inner_drawdown"] = rows["selection_max_drawdown_pct"].ge(-20.0)
    rows["passes_inner_activity"] = rows["selection_avg_abs_position"].ge(0.05)
    rows["is_calibration_asset"] = rows["heldout_asset"].astype(str).eq(CALIBRATION_ASSET)
    group_keys = ["candidate", "feature_set"]
    grouped = (
        rows.groupby(group_keys)
        .agg(
            feature_count=("feature_count", "first"),
            contains_soxx_constituents=("contains_soxx_constituents", "first"),
            soxx_constituent_feature_count=("soxx_constituent_feature_count", "first"),
            avg_selection_return_pct=("selection_return_pct", "mean"),
            avg_excess_return_pct=("excess_return_pct", "mean"),
            worst_selection_max_drawdown_pct=("selection_max_drawdown_pct", "min"),
            avg_selection_avg_abs_position=("selection_avg_abs_position", "mean"),
            avg_selection_total_turnover=("selection_total_turnover", "mean"),
            inner_pass_rate=("passes_inner_drawdown", "mean"),
            inner_activity_pass_rate=("passes_inner_activity", "mean"),
            inner_beats_buyhold_rate=("beats_buyhold", "mean"),
            heldout_assets=("heldout_asset", lambda values: sorted({str(value) for value in values})),
            inner_fold_count=("candidate", "size"),
        )
        .reset_index()
    )
    calibration = (
        rows.loc[rows["is_calibration_asset"]]
        .groupby(group_keys)
        .agg(
            soxl_inner_return_pct=("selection_return_pct", "mean"),
            soxl_inner_excess_return_pct=("excess_return_pct", "mean"),
            soxl_inner_worst_drawdown_pct=("selection_max_drawdown_pct", "min"),
            soxl_inner_avg_abs_position=("selection_avg_abs_position", "mean"),
            soxl_inner_total_turnover=("selection_total_turnover", "mean"),
            soxl_inner_pass_rate=("passes_inner_drawdown", "mean"),
            soxl_inner_activity_pass_rate=("passes_inner_activity", "mean"),
            soxl_inner_beats_buyhold_rate=("beats_buyhold", "mean"),
            soxl_inner_fold_count=("candidate", "size"),
        )
        .reset_index()
    )
    stress = (
        rows.loc[~rows["is_calibration_asset"]]
        .groupby(group_keys)
        .agg(
            cross_asset_avg_excess_return_pct=("excess_return_pct", "mean"),
            cross_asset_avg_return_pct=("selection_return_pct", "mean"),
            cross_asset_worst_drawdown_pct=("selection_max_drawdown_pct", "min"),
            cross_asset_stress_pass_rate=("passes_inner_drawdown", "mean"),
            cross_asset_activity_pass_rate=("passes_inner_activity", "mean"),
            cross_asset_beats_buyhold_rate=("beats_buyhold", "mean"),
            cross_asset_fold_count=("candidate", "size"),
        )
        .reset_index()
    )
    grouped = (
        grouped.merge(calibration, on=group_keys, how="left")
        .merge(stress, on=group_keys, how="left")
    )
    required_assets = [str(asset) for asset in required_heldout_assets]
    grouped["has_required_heldout_assets"] = grouped["heldout_assets"].map(
        lambda assets: all(asset in assets for asset in required_assets)
    )
    grouped["has_calibration_asset"] = grouped["soxl_inner_fold_count"].fillna(0).gt(0)
    grouped["soxl_drawdown_shortfall_pct"] = (
        (-20.0 - grouped["soxl_inner_worst_drawdown_pct"].fillna(-100.0)).clip(lower=0.0)
    )
    grouped["cross_asset_drawdown_shortfall_pct"] = (
        (-30.0 - grouped["cross_asset_worst_drawdown_pct"].fillna(-100.0)).clip(lower=0.0)
    )
    eligible = grouped.loc[grouped["has_required_heldout_assets"] & grouped["has_calibration_asset"]].copy()
    if eligible.empty:
        raise ValueError("trainable regime policy inner nested walk-forward missing required heldout or SOXL calibration assets")
    soxl_risk_eligible = eligible.loc[
        eligible["soxl_inner_pass_rate"].fillna(0.0).ge(0.5)
        & eligible["soxl_inner_activity_pass_rate"].fillna(0.0).ge(0.5)
        & eligible["inner_activity_pass_rate"].fillna(0.0).ge(0.5)
    ].copy()
    soxl_drawdown_gate_applied = not soxl_risk_eligible.empty
    if soxl_drawdown_gate_applied:
        eligible = soxl_risk_eligible
    eligible["selection_score"] = (
        2.50 * eligible["soxl_inner_excess_return_pct"].fillna(0.0)
        + 0.75 * eligible["soxl_inner_return_pct"].fillna(0.0)
        + 55.0 * eligible["soxl_inner_beats_buyhold_rate"].fillna(0.0)
        + 40.0 * eligible["soxl_inner_pass_rate"].fillna(0.0)
        + 25.0 * eligible["soxl_inner_activity_pass_rate"].fillna(0.0)
        - 4.00 * eligible["soxl_drawdown_shortfall_pct"].fillna(0.0)
        + 0.25 * eligible["cross_asset_avg_excess_return_pct"].fillna(0.0)
        + 20.0 * eligible["cross_asset_stress_pass_rate"].fillna(0.0)
        + 10.0 * eligible["cross_asset_beats_buyhold_rate"].fillna(0.0)
        + 10.0 * eligible["cross_asset_activity_pass_rate"].fillna(0.0)
        - 0.75 * eligible["cross_asset_drawdown_shortfall_pct"].fillna(0.0)
        - 0.03 * eligible["avg_selection_total_turnover"].fillna(0.0)
    )
    selected = eligible.sort_values(
        [
            "selection_score",
            "soxl_inner_beats_buyhold_rate",
            "soxl_inner_excess_return_pct",
            "soxl_inner_worst_drawdown_pct",
            "cross_asset_stress_pass_rate",
            "avg_excess_return_pct",
        ],
        ascending=[False, False, False, False, False, False],
    ).iloc[0]
    return {
        "candidate": str(selected["candidate"]),
        "feature_set": str(selected["feature_set"]),
        "outer_test_year": int(outer_test_year),
        "selection_policy": SELECTION_POLICY,
        "calibration_asset": CALIBRATION_ASSET,
        "feature_count": int(selected["feature_count"]),
        "contains_soxx_constituents": bool(selected["contains_soxx_constituents"]),
        "soxx_constituent_feature_count": int(selected["soxx_constituent_feature_count"]),
        "soxl_drawdown_gate_applied": bool(soxl_drawdown_gate_applied),
        "selection_uses_outer_test": False,
        "required_heldout_assets": required_assets,
        "observed_heldout_assets": list(selected["heldout_assets"]),
        "inner_fold_count": int(selected["inner_fold_count"]),
        "soxl_inner_fold_count": int(selected["soxl_inner_fold_count"]),
        "soxl_inner_return_pct": float(selected["soxl_inner_return_pct"]),
        "soxl_inner_excess_return_pct": float(selected["soxl_inner_excess_return_pct"]),
        "soxl_inner_worst_drawdown_pct": float(selected["soxl_inner_worst_drawdown_pct"]),
        "soxl_inner_avg_abs_position": float(selected["soxl_inner_avg_abs_position"]),
        "soxl_inner_pass_rate": float(selected["soxl_inner_pass_rate"]),
        "soxl_inner_activity_pass_rate": float(selected["soxl_inner_activity_pass_rate"]),
        "soxl_inner_beats_buyhold_rate": float(selected["soxl_inner_beats_buyhold_rate"]),
        "soxl_drawdown_shortfall_pct": float(selected["soxl_drawdown_shortfall_pct"]),
        "cross_asset_stress_role": "overfit_audit_not_primary_selection_target",
        "cross_asset_fold_count": int(selected["cross_asset_fold_count"]),
        "cross_asset_avg_return_pct": float(selected["cross_asset_avg_return_pct"]),
        "cross_asset_avg_excess_return_pct": float(selected["cross_asset_avg_excess_return_pct"]),
        "cross_asset_worst_drawdown_pct": float(selected["cross_asset_worst_drawdown_pct"]),
        "cross_asset_stress_pass_rate": float(selected["cross_asset_stress_pass_rate"]),
        "cross_asset_activity_pass_rate": float(selected["cross_asset_activity_pass_rate"]),
        "cross_asset_beats_buyhold_rate": float(selected["cross_asset_beats_buyhold_rate"]),
        "cross_asset_drawdown_shortfall_pct": float(selected["cross_asset_drawdown_shortfall_pct"]),
        "avg_selection_return_pct": float(selected["avg_selection_return_pct"]),
        "avg_excess_return_pct": float(selected["avg_excess_return_pct"]),
        "avg_selection_avg_abs_position": float(selected["avg_selection_avg_abs_position"]),
        "worst_selection_max_drawdown_pct": float(selected["worst_selection_max_drawdown_pct"]),
        "avg_selection_total_turnover": float(selected["avg_selection_total_turnover"]),
        "inner_pass_rate": float(selected["inner_pass_rate"]),
        "inner_activity_pass_rate": float(selected["inner_activity_pass_rate"]),
        "inner_beats_buyhold_rate": float(selected["inner_beats_buyhold_rate"]),
        "selection_score": float(selected["selection_score"]),
    }


def _outer_soxl_policy_test(
    frame: pd.DataFrame,
    *,
    outer_test_year: int,
    selected_candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_set_configs: list[dict[str, Any]],
    common_end: str,
    cost_per_turnover: float,
    max_drawdown_floor_pct: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidate = next(item for item in candidates if item["candidate"] == selected_candidate["candidate"])
    feature_set = _feature_set_for_selected_candidate(feature_set_configs, selected_candidate)
    feature_cols = list(feature_set["feature_cols"])
    data = _clean_policy_frame(frame, feature_cols)
    train_end = pd.Timestamp(f"{outer_test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{outer_test_year}-01-01")
    test_end = min(pd.Timestamp(f"{outer_test_year}-12-31"), pd.Timestamp(common_end))
    train = data.loc[data["datetime"].le(train_end)].copy()
    test = data.loc[
        data["instrument"].eq("SOXL") & data["datetime"].between(test_start, test_end)
    ].copy()
    if len(train) < 300 or len(test) < 40:
        raise ValueError(f"outer fold {outer_test_year} lacks enough train/test rows")
    pred = _predict_trainable_policy(train, test, feature_cols=feature_cols, candidate=candidate)
    metrics, positions = _simulate_alpha_risk_policy(
        test,
        pred,
        policy=candidate,
        segment_name="outer_test",
        cost_per_turnover=cost_per_turnover,
    )
    buyhold = _buyhold_metrics(test)
    row = {
        "outer_test_year": int(outer_test_year),
        "candidate": str(candidate["candidate"]),
        "feature_set": str(feature_set["feature_set"]),
        "feature_count": int(feature_set["feature_count"]),
        "contains_soxx_constituents": bool(feature_set["contains_soxx_constituents"]),
        "soxx_constituent_feature_count": int(feature_set["soxx_constituent_feature_count"]),
        "selected_regime_model": str(candidate["model_family"]),
        "policy_id": str(candidate["policy_id"]),
        "position_mode": str(candidate.get("position_mode", "scaled_alpha")),
        "train_start": pd.Timestamp(train["datetime"].min()).date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": pd.Timestamp(test["datetime"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["datetime"].max()).date().isoformat(),
        "test_return_pct": metrics["selection_return_pct"],
        "test_max_drawdown_pct": metrics["selection_max_drawdown_pct"],
        "test_avg_abs_position": metrics["selection_avg_abs_position"],
        "test_total_turnover": metrics["selection_total_turnover"],
        "test_intraday_risk_cap_day_ratio": metrics["intraday_risk_cap_day_ratio"],
        "test_intraday_execution_replay_day_ratio": metrics["intraday_execution_replay_day_ratio"],
        "test_multi_target_risk_cap_day_ratio": metrics["multi_target_risk_cap_day_ratio"],
        "test_percentile_risk_cap_day_ratio": metrics["percentile_risk_cap_day_ratio"],
        "test_drawdown_cooldown_day_ratio": metrics["drawdown_cooldown_day_ratio"],
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
    positions["feature_set"] = str(feature_set["feature_set"])
    return row, positions


def _feature_set_for_selected_candidate(
    feature_set_configs: list[dict[str, Any]],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    selected_name = str(selected_candidate.get("feature_set", feature_set_configs[0]["feature_set"]))
    for config in feature_set_configs:
        if str(config["feature_set"]) == selected_name:
            return config
    raise ValueError(f"selected feature_set not found in pre-registered configs: {selected_name}")


def _predict_trainable_policy(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    feature_cols: list[str],
    candidate: dict[str, Any],
) -> pd.DataFrame:
    cache = _predict_trainable_policy_cache(
        train,
        predict,
        feature_cols=feature_cols,
        candidates=[candidate],
    )
    return cache[str(candidate["model_family"])].copy()


def _predict_trainable_policy_cache(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    feature_cols: list[str],
    candidates: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    requested = {str(candidate["model_family"]) for candidate in candidates}
    base_requested = set(requested)
    if "lgb_xgb_consensus" in requested:
        base_requested.update({"lgbm", "xgb"})
    cache = {
        family: _predict_base_trainable_policy(train, predict, feature_cols=feature_cols, model_family=family)
        for family in sorted(base_requested)
        if family in {"lgbm", "xgb"}
    }
    if "lgb_xgb_consensus" in requested:
        cache["lgb_xgb_consensus"] = _combine_consensus_predictions(cache["lgbm"], cache["xgb"], predict)
    return {family: cache[family].copy() for family in requested}


def _predict_base_trainable_policy(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_family: str,
) -> pd.DataFrame:
    family = str(model_family)
    alpha = _fit_predict_binary_probability(family, train, predict, feature_cols, ALPHA_TARGET_COL)
    risk, risk_percentile = _fit_predict_binary_probability_with_train_percentile(
        family,
        train,
        predict,
        feature_cols,
        RISK_TARGET_COL,
    )
    crash, crash_percentile = _fit_predict_binary_probability_with_train_percentile(
        family,
        train,
        predict,
        feature_cols,
        CRASH_TARGET_COL,
    )
    max_adverse, max_adverse_risk_percentile = _fit_predict_regression_with_train_risk_percentile(
        family,
        train,
        predict,
        feature_cols,
        MAX_ADVERSE_TARGET_COL,
        higher_is_risk=False,
    )
    crash_20d, crash_20d_percentile = _fit_predict_binary_probability_with_train_percentile(
        family,
        train,
        predict,
        feature_cols,
        CRASH_20D_TARGET_COL,
    )
    max_adverse_20d, max_adverse_20d_risk_percentile = _fit_predict_regression_with_train_risk_percentile(
        family,
        train,
        predict,
        feature_cols,
        MAX_ADVERSE_20D_TARGET_COL,
        higher_is_risk=False,
    )
    regime = _fit_predict_regime(family, train, predict, feature_cols)
    vol_score = _fit_predict_regression(family, train, predict, feature_cols, VOL_ADJUSTED_TARGET_COL)
    learned_position = _fit_predict_regression(family, train, predict, feature_cols, POSITION_TARGET_COL)
    return pd.DataFrame(
        {
            "alpha_probability": alpha.reindex(predict.index).fillna(0.0),
            "risk_probability": risk.reindex(predict.index).fillna(1.0),
            "risk_percentile": risk_percentile.reindex(predict.index).fillna(1.0),
            "crash_probability": crash.reindex(predict.index).fillna(1.0),
            "crash_risk_percentile": crash_percentile.reindex(predict.index).fillna(1.0),
            "max_adverse_score": max_adverse.reindex(predict.index).fillna(-1.0),
            "max_adverse_risk_percentile": max_adverse_risk_percentile.reindex(predict.index).fillna(1.0),
            "crash_20d_probability": crash_20d.reindex(predict.index).fillna(1.0),
            "crash_20d_risk_percentile": crash_20d_percentile.reindex(predict.index).fillna(1.0),
            "max_adverse_20d_score": max_adverse_20d.reindex(predict.index).fillna(-1.0),
            "max_adverse_20d_risk_percentile": max_adverse_20d_risk_percentile.reindex(predict.index).fillna(1.0),
            "vol_adjusted_score": vol_score.reindex(predict.index).fillna(0.0),
            "learned_position_score": learned_position.reindex(predict.index).fillna(0.0).clip(0.0, 1.0),
            "regime": regime.reindex(predict.index).fillna("selective"),
            "intraday_risk_state": predict.get(INTRADAY_RISK_STATE_COL, pd.Series(0.0, index=predict.index)).to_numpy(),
        },
        index=predict.index,
    )


def _combine_consensus_predictions(lgb: pd.DataFrame, xgb: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    regime = []
    for left, right in zip(lgb["regime"], xgb["regime"], strict=False):
        regime.append(left if left == right else "selective")
    return pd.DataFrame(
        {
            "alpha_probability": (lgb["alpha_probability"].to_numpy() + xgb["alpha_probability"].to_numpy()) / 2.0,
            "risk_probability": (lgb["risk_probability"].to_numpy() + xgb["risk_probability"].to_numpy()) / 2.0,
            "risk_percentile": (lgb["risk_percentile"].to_numpy() + xgb["risk_percentile"].to_numpy()) / 2.0,
            "crash_probability": (lgb["crash_probability"].to_numpy() + xgb["crash_probability"].to_numpy()) / 2.0,
            "crash_risk_percentile": (lgb["crash_risk_percentile"].to_numpy() + xgb["crash_risk_percentile"].to_numpy()) / 2.0,
            "max_adverse_score": (lgb["max_adverse_score"].to_numpy() + xgb["max_adverse_score"].to_numpy()) / 2.0,
            "max_adverse_risk_percentile": (
                lgb["max_adverse_risk_percentile"].to_numpy() + xgb["max_adverse_risk_percentile"].to_numpy()
            ) / 2.0,
            "crash_20d_probability": (
                lgb["crash_20d_probability"].to_numpy() + xgb["crash_20d_probability"].to_numpy()
            ) / 2.0,
            "crash_20d_risk_percentile": (
                lgb["crash_20d_risk_percentile"].to_numpy() + xgb["crash_20d_risk_percentile"].to_numpy()
            ) / 2.0,
            "max_adverse_20d_score": (
                lgb["max_adverse_20d_score"].to_numpy() + xgb["max_adverse_20d_score"].to_numpy()
            ) / 2.0,
            "max_adverse_20d_risk_percentile": (
                lgb["max_adverse_20d_risk_percentile"].to_numpy()
                + xgb["max_adverse_20d_risk_percentile"].to_numpy()
            ) / 2.0,
            "vol_adjusted_score": (lgb["vol_adjusted_score"].to_numpy() + xgb["vol_adjusted_score"].to_numpy()) / 2.0,
            "learned_position_score": (
                lgb["learned_position_score"].to_numpy() + xgb["learned_position_score"].to_numpy()
            ) / 2.0,
            "regime": regime,
            "intraday_risk_state": predict.get(INTRADAY_RISK_STATE_COL, pd.Series(0.0, index=predict.index)).to_numpy(),
        },
        index=predict.index,
    )


def _simulate_alpha_risk_policy(
    segment_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy: dict[str, Any],
    cost_per_turnover: float,
    segment_name: str = "test",
) -> tuple[dict[str, Any], pd.DataFrame]:
    position_mode = str(policy.get("position_mode", "scaled_alpha"))
    alpha_floor = float(policy.get("alpha_floor", 0.0))
    alpha = pd.to_numeric(predictions["alpha_probability"], errors="coerce").fillna(0.0)
    risk = pd.to_numeric(predictions["risk_probability"], errors="coerce").fillna(1.0)
    risk_percentile = pd.to_numeric(predictions.get("risk_percentile", risk), errors="coerce").fillna(1.0)
    crash = pd.to_numeric(predictions.get("crash_probability", risk), errors="coerce").fillna(1.0)
    crash_risk_percentile = pd.to_numeric(
        predictions.get("crash_risk_percentile", crash),
        errors="coerce",
    ).fillna(1.0)
    max_adverse = pd.to_numeric(
        predictions.get("max_adverse_score", pd.Series(0.0, index=predictions.index)),
        errors="coerce",
    ).fillna(-1.0)
    max_adverse_risk_percentile = pd.to_numeric(
        predictions.get("max_adverse_risk_percentile", pd.Series(0.0, index=predictions.index)),
        errors="coerce",
    ).fillna(1.0)
    crash_20d = pd.to_numeric(
        predictions.get("crash_20d_probability", pd.Series(0.0, index=predictions.index)),
        errors="coerce",
    ).fillna(0.0)
    crash_20d_risk_percentile = pd.to_numeric(
        predictions.get("crash_20d_risk_percentile", pd.Series(0.0, index=predictions.index)),
        errors="coerce",
    ).fillna(0.0)
    max_adverse_20d = pd.to_numeric(
        predictions.get("max_adverse_20d_score", pd.Series(0.0, index=predictions.index)),
        errors="coerce",
    ).fillna(0.0)
    max_adverse_20d_risk_percentile = pd.to_numeric(
        predictions.get("max_adverse_20d_risk_percentile", pd.Series(0.0, index=predictions.index)),
        errors="coerce",
    ).fillna(0.0)
    vol_score = pd.to_numeric(predictions["vol_adjusted_score"], errors="coerce").fillna(0.0)
    regime = predictions["regime"].astype(str).where(predictions["regime"].astype(str).isin(REGIME_LABELS), "selective")
    intraday_risk = pd.to_numeric(predictions.get("intraday_risk_state", 0.0), errors="coerce").fillna(0.0)
    scale = regime.map(
        {
            "trend": float(policy["trend_scale"]),
            "rebound": float(policy["rebound_scale"]),
            "selective": float(policy["selective_scale"]),
            "defensive": float(policy["defensive_scale"]),
        }
    ).fillna(float(policy["selective_scale"]))
    if position_mode == "learned_position":
        learned_position = pd.to_numeric(
            predictions.get("learned_position_score", pd.Series(0.0, index=predictions.index)),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0)
        risk_discount = (1.0 - risk.clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
        crash_discount = (1.0 - crash.clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
        adverse_survival = ((max_adverse + 0.20) / 0.20).clip(lower=0.0, upper=1.0)
        percentile_discount = (1.0 - 0.50 * risk_percentile.clip(lower=0.0, upper=1.0)).clip(lower=0.0, upper=1.0)
        target_position = (
            learned_position
            * (0.50 + 0.50 * risk_discount)
            * (0.50 + 0.50 * crash_discount)
            * adverse_survival
            * percentile_discount
        ).clip(lower=0.0, upper=1.0)
    elif position_mode == "binary_full":
        target_position = alpha.ge(alpha_floor).astype("float64") * scale.clip(lower=0.0, upper=1.0)
    else:
        raw = ((alpha - alpha_floor) / max(1.0 - alpha_floor, 1e-12)).clip(lower=0.0, upper=1.0)
        vol_boost = (1.0 + 0.20 * vol_score.clip(lower=0.0, upper=2.0)).clip(lower=0.5, upper=1.35)
        target_position = (raw * vol_boost * scale).clip(lower=0.0, upper=1.0)
    long_adverse_pressure = ((-max_adverse_20d - 0.12) / 0.18).clip(lower=0.0, upper=1.0)
    long_crash_pressure = ((crash_20d - 0.35) / 0.65).clip(lower=0.0, upper=1.0)
    long_adverse_percentile_pressure = (
        (max_adverse_20d_risk_percentile - 0.75) / 0.25
    ).clip(lower=0.0, upper=1.0)
    long_crash_percentile_pressure = ((crash_20d_risk_percentile - 0.75) / 0.25).clip(lower=0.0, upper=1.0)
    long_horizon_risk_factor = (
        (1.0 - long_adverse_pressure)
        * (1.0 - 0.75 * long_crash_pressure)
        * (1.0 - 0.50 * long_adverse_percentile_pressure)
        * (1.0 - 0.25 * long_crash_percentile_pressure)
    ).clip(lower=0.0, upper=1.0)
    target_position = (target_position * long_horizon_risk_factor).clip(lower=0.0, upper=1.0)
    risk_cap_mask = risk.ge(float(policy["risk_cut"]))
    target_position = target_position.where(~risk_cap_mask, np.minimum(target_position, float(policy["risk_cap"])))
    crash_cut = float(policy.get("crash_cut", np.inf))
    max_adverse_cut = float(policy.get("max_adverse_cut", -np.inf))
    adverse_risk_cap = float(policy.get("adverse_risk_cap", policy["risk_cap"]))
    multi_target_risk_mask = crash.ge(crash_cut) | max_adverse.le(max_adverse_cut)
    target_position = target_position.where(
        ~multi_target_risk_mask,
        np.minimum(target_position, adverse_risk_cap),
    )
    percentile_risk_cut = float(policy.get("risk_percentile_cut", np.inf))
    crash_percentile_cut = float(policy.get("crash_percentile_cut", np.inf))
    adverse_percentile_cut = float(policy.get("adverse_percentile_cut", np.inf))
    percentile_risk_cap = float(policy.get("percentile_risk_cap", adverse_risk_cap))
    percentile_risk_mask = (
        risk_percentile.ge(percentile_risk_cut)
        | crash_risk_percentile.ge(crash_percentile_cut)
        | max_adverse_risk_percentile.ge(adverse_percentile_cut)
    )
    target_position = target_position.where(
        ~percentile_risk_mask,
        np.minimum(target_position, percentile_risk_cap),
    )
    use_intraday_execution_replay = bool(policy.get("intraday_execution_replay", False))
    intraday_cap_mask = intraday_risk.ge(0.5)
    if not use_intraday_execution_replay:
        target_position = target_position.where(
            ~intraday_cap_mask,
            np.minimum(target_position, float(policy["intraday_risk_cap"])),
        )
    returns = pd.to_numeric(segment_frame[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0)
    cooldown_active = pd.Series(0, index=segment_frame.index, dtype="int64")
    if "drawdown_cooldown_trigger" in policy:
        target_position, turnover, strategy_returns, cooldown_active = _apply_drawdown_cooldown_cap(
            target_position,
            returns,
            trigger=float(policy["drawdown_cooldown_trigger"]),
            cooldown_days=int(policy.get("drawdown_cooldown_days", 1)),
            cooldown_cap=float(policy.get("drawdown_cooldown_cap", 0.0)),
            retrigger=bool(policy.get("drawdown_cooldown_retrigger", True)),
            rearm_level=float(policy.get("drawdown_cooldown_rearm_level", policy["drawdown_cooldown_trigger"] / 2.0)),
            cost_per_turnover=float(cost_per_turnover),
        )
        opening_position = target_position.copy()
        intraday_replay_triggered = pd.Series(0.0, index=segment_frame.index, dtype="float64")
    else:
        if use_intraday_execution_replay:
            (
                opening_position,
                target_position,
                turnover,
                strategy_returns,
                intraday_replay_triggered,
            ) = _apply_intraday_execution_replay(
                target_position,
                segment_frame,
                intraday_cap=float(policy["intraday_risk_cap"]),
                cost_per_turnover=float(cost_per_turnover),
            )
        else:
            opening_position = target_position.copy()
            turnover = target_position.diff().abs().fillna(target_position.abs())
            strategy_returns = target_position * returns - turnover * float(cost_per_turnover)
            intraday_replay_triggered = pd.Series(0.0, index=segment_frame.index, dtype="float64")
    metrics = _return_metrics(strategy_returns, target_position, turnover)
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(segment_frame["datetime"]).dt.normalize(),
            "instrument": segment_frame["instrument"].astype(str).to_numpy(),
            "segment": segment_name,
            "candidate": str(policy["candidate"]) if "candidate" in policy else str(policy["policy_id"]),
            "alpha_probability": alpha.to_numpy(),
            "risk_probability": risk.to_numpy(),
            "risk_percentile": risk_percentile.to_numpy(),
            "crash_probability": crash.to_numpy(),
            "crash_risk_percentile": crash_risk_percentile.to_numpy(),
            "max_adverse_score": max_adverse.to_numpy(),
            "max_adverse_risk_percentile": max_adverse_risk_percentile.to_numpy(),
            "crash_20d_probability": crash_20d.to_numpy(),
            "crash_20d_risk_percentile": crash_20d_risk_percentile.to_numpy(),
            "max_adverse_20d_score": max_adverse_20d.to_numpy(),
            "max_adverse_20d_risk_percentile": max_adverse_20d_risk_percentile.to_numpy(),
            "long_horizon_risk_factor": long_horizon_risk_factor.to_numpy(),
            "vol_adjusted_score": vol_score.to_numpy(),
            "learned_position_score": pd.to_numeric(
                predictions.get("learned_position_score", pd.Series(0.0, index=predictions.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy(),
            "regime": regime.to_numpy(),
            "intraday_risk_state": intraday_risk.to_numpy(),
            "intraday_execution_replay_used": np.full(len(segment_frame), use_intraday_execution_replay, dtype=bool),
            "intraday_execution_triggered": intraday_replay_triggered.to_numpy(),
            "opening_position": opening_position.to_numpy(),
            "target_position": target_position.to_numpy(),
            "turnover": turnover.to_numpy(),
            "strategy_return": strategy_returns.to_numpy(),
            "drawdown_cooldown_active": cooldown_active.to_numpy(),
            OPEN_TO_OPEN_RETURN_COL: returns.to_numpy(),
        }
    )
    return {
        "selection_return_pct": metrics["return_pct"],
        "selection_max_drawdown_pct": metrics["max_drawdown_pct"],
        "selection_avg_abs_position": metrics["avg_abs_position"],
        "selection_total_turnover": metrics["total_turnover"],
        "selection_trade_count": int(turnover.gt(1e-9).sum()),
        "intraday_risk_cap_day_ratio": float(intraday_cap_mask.mean()) if len(intraday_cap_mask) else 0.0,
        "intraday_execution_replay_day_ratio": (
            float(intraday_replay_triggered.mean()) if use_intraday_execution_replay and len(intraday_replay_triggered) else 0.0
        ),
        "multi_target_risk_cap_day_ratio": float(multi_target_risk_mask.mean()) if len(multi_target_risk_mask) else 0.0,
        "percentile_risk_cap_day_ratio": float(percentile_risk_mask.mean()) if len(percentile_risk_mask) else 0.0,
        "drawdown_cooldown_day_ratio": float(cooldown_active.mean()) if len(cooldown_active) else 0.0,
    }, out


def _apply_intraday_execution_replay(
    target_position: pd.Series,
    segment_frame: pd.DataFrame,
    *,
    intraday_cap: float,
    cost_per_turnover: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    daily_returns = pd.to_numeric(segment_frame[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0)
    triggered = pd.to_numeric(
        segment_frame.get(INTRADAY_EXECUTION_TRIGGERED_COL, pd.Series(0.0, index=segment_frame.index)),
        errors="coerce",
    ).fillna(0.0).ge(0.5)
    open_to_trigger = pd.to_numeric(
        segment_frame.get(INTRADAY_OPEN_TO_TRIGGER_RETURN_COL, daily_returns),
        errors="coerce",
    ).fillna(daily_returns)
    trigger_to_close = pd.to_numeric(
        segment_frame.get(INTRADAY_TRIGGER_TO_CLOSE_RETURN_COL, pd.Series(0.0, index=segment_frame.index)),
        errors="coerce",
    ).fillna(0.0)
    opening_position = target_position.astype("float64").copy()
    closing_position = opening_position.where(~triggered, np.minimum(opening_position, float(intraday_cap)))
    previous_close_position = closing_position.shift(1).fillna(0.0)
    entry_turnover = (opening_position - previous_close_position).abs()
    intraday_turnover = (closing_position - opening_position).abs().where(triggered, 0.0)
    turnover = entry_turnover + intraday_turnover
    strategy_returns = (
        opening_position * open_to_trigger
        + closing_position * trigger_to_close
        - turnover * float(cost_per_turnover)
    )
    return (
        opening_position,
        closing_position,
        turnover.astype("float64"),
        strategy_returns.astype("float64"),
        triggered.astype("float64"),
    )


def _apply_drawdown_cooldown_cap(
    target_position: pd.Series,
    returns: pd.Series,
    *,
    trigger: float,
    cooldown_days: int,
    cooldown_cap: float,
    retrigger: bool,
    rearm_level: float,
    cost_per_turnover: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    positions: list[float] = []
    turnovers: list[float] = []
    strategy_returns: list[float] = []
    cooldown_flags: list[int] = []
    equity = 1.0
    peak = 1.0
    previous_position = 0.0
    cooldown_left = 0
    cooldown_armed = True
    for desired_position, raw_return in zip(target_position.to_numpy(), returns.to_numpy(), strict=False):
        if cooldown_left > 0:
            position = min(float(desired_position), float(cooldown_cap))
            cooldown_flags.append(1)
            cooldown_left -= 1
        else:
            position = float(desired_position)
            cooldown_flags.append(0)
        turnover = abs(position - previous_position)
        strategy_return = position * float(raw_return) - turnover * float(cost_per_turnover)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        if retrigger:
            if drawdown <= float(trigger):
                cooldown_left = max(cooldown_left, int(cooldown_days))
        else:
            if not cooldown_armed and drawdown >= float(rearm_level):
                cooldown_armed = True
            if cooldown_armed and drawdown <= float(trigger):
                cooldown_left = max(cooldown_left, int(cooldown_days))
                cooldown_armed = False
        positions.append(position)
        turnovers.append(turnover)
        strategy_returns.append(strategy_return)
        previous_position = position
    index = target_position.index
    return (
        pd.Series(positions, index=index, dtype="float64"),
        pd.Series(turnovers, index=index, dtype="float64"),
        pd.Series(strategy_returns, index=index, dtype="float64"),
        pd.Series(cooldown_flags, index=index, dtype="int64"),
    )


def _fit_predict_binary_probability(
    model_family: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.Series:
    fit = train.dropna(subset=[target_col]).copy()
    x_train = fit[feature_cols].astype("float64")
    y_train = fit[target_col].astype(int)
    if y_train.nunique() < 2:
        model = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="constant", constant=int(y_train.iloc[0])))
    else:
        model = _build_classifier(model_family)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(predict[feature_cols].astype("float64"))
    classes = list(getattr(model[-1], "classes_", []))
    positive_index = classes.index(1) if 1 in classes else 0
    return pd.Series(probabilities[:, positive_index], index=predict.index, name=target_col)


def _fit_predict_binary_probability_with_train_percentile(
    model_family: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.Series, pd.Series]:
    fit = train.dropna(subset=[target_col]).copy()
    x_train = fit[feature_cols].astype("float64")
    y_train = fit[target_col].astype(int)
    if y_train.nunique() < 2:
        model = make_pipeline(SimpleImputer(strategy="median"), DummyClassifier(strategy="constant", constant=int(y_train.iloc[0])))
    else:
        model = _build_classifier(model_family)
    model.fit(x_train, y_train)
    classes = list(getattr(model[-1], "classes_", []))
    positive_index = classes.index(1) if 1 in classes else 0
    train_probabilities = model.predict_proba(x_train)[:, positive_index]
    predict_probabilities = model.predict_proba(predict[feature_cols].astype("float64"))[:, positive_index]
    probability = pd.Series(predict_probabilities, index=predict.index, name=target_col)
    percentile = _percentile_against_train(
        pd.Series(train_probabilities, index=fit.index),
        probability,
        higher_is_risk=True,
    ).rename(f"{target_col}_percentile")
    return probability, percentile


def _fit_predict_regression_with_train_risk_percentile(
    model_family: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    higher_is_risk: bool,
) -> tuple[pd.Series, pd.Series]:
    fit = train.dropna(subset=[target_col]).copy()
    x_train = fit[feature_cols].astype("float64")
    y_train = pd.to_numeric(fit[target_col], errors="coerce").fillna(0.0)
    if y_train.nunique() < 2:
        model = make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="constant", constant=float(y_train.iloc[0])))
    else:
        model = _build_regressor(model_family)
    model.fit(x_train, y_train)
    train_scores = pd.Series(model.predict(x_train), index=fit.index, name=target_col)
    predict_scores = pd.Series(
        model.predict(predict[feature_cols].astype("float64")),
        index=predict.index,
        name=target_col,
    )
    risk_percentile = _percentile_against_train(
        train_scores,
        predict_scores,
        higher_is_risk=higher_is_risk,
    ).rename(f"{target_col}_risk_percentile")
    return predict_scores, risk_percentile


def _percentile_against_train(
    train_scores: pd.Series,
    predict_scores: pd.Series,
    *,
    higher_is_risk: bool,
) -> pd.Series:
    train_values = pd.to_numeric(train_scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    predict_values = pd.to_numeric(predict_scores, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if len(train_values) == 0:
        return pd.Series(0.0, index=predict_scores.index, dtype="float64")
    sorted_train = np.sort(train_values.astype("float64"))
    side = "right" if higher_is_risk else "left"
    predict_array = predict_values.to_numpy(dtype="float64")
    valid = np.isfinite(predict_array)
    ranks = np.zeros(len(predict_array), dtype="float64")
    ranks[valid] = np.searchsorted(sorted_train, predict_array[valid], side=side)
    percentile = ranks / float(len(sorted_train))
    if not higher_is_risk:
        percentile = 1.0 - percentile
    percentile = np.where(valid & np.isfinite(percentile), percentile, 0.0)
    return pd.Series(percentile, index=predict_scores.index, dtype="float64").clip(0.0, 1.0)


def _fit_predict_regime(
    model_family: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    fit = train.dropna(subset=[REGIME_TARGET_CODE_COL]).copy()
    x_train = fit[feature_cols].astype("float64")
    y_train = pd.to_numeric(fit[REGIME_TARGET_CODE_COL], errors="coerce").fillna(2).astype(int).clip(0, len(REGIME_LABELS) - 1)
    if len(set(y_train)) < 2:
        return pd.Series([REGIME_LABELS[int(y_train.iloc[0])]] * len(predict), index=predict.index)
    model = _build_regime_classifier(model_family, num_class=len(REGIME_LABELS))
    model.fit(x_train, y_train)
    pred = model.predict(predict[feature_cols].astype("float64"))
    pred = pd.Series(pred, index=predict.index).astype(int).clip(0, len(REGIME_LABELS) - 1)
    return pred.map(lambda value: REGIME_LABELS[int(value)]).rename("regime")


def _fit_predict_regression(
    model_family: str,
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.Series:
    fit = train.dropna(subset=[target_col]).copy()
    x_train = fit[feature_cols].astype("float64")
    y_train = pd.to_numeric(fit[target_col], errors="coerce").fillna(0.0)
    if y_train.nunique() < 2:
        model = make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="constant", constant=float(y_train.iloc[0])))
    else:
        model = _build_regressor(model_family)
    model.fit(x_train, y_train)
    return pd.Series(model.predict(predict[feature_cols].astype("float64")), index=predict.index, name=target_col)


def _build_classifier(model_family: str):
    if model_family == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=25,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.4,
                random_state=42,
                verbose=-1,
            ),
        )
    if model_family == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=30,
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
    raise ValueError(f"unsupported classifier family: {model_family}")


def _build_regime_classifier(model_family: str, *, num_class: int):
    if model_family == "lgbm":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=25,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.4,
                objective="multiclass",
                num_class=int(num_class),
                random_state=46,
                verbose=-1,
            ),
        )
    if model_family == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=int(num_class),
                random_state=47,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported regime classifier family: {model_family}")


def _build_regressor(model_family: str):
    if model_family == "lgbm":
        from lightgbm import LGBMRegressor

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=25,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.4,
                random_state=44,
                verbose=-1,
            ),
        )
    if model_family == "xgb":
        from xgboost import XGBRegressor

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=35,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=45,
                n_jobs=1,
            ),
        )
    raise ValueError(f"unsupported regressor family: {model_family}")


def _clean_policy_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out["instrument"] = out["instrument"].astype(str).str.upper()
    for column in [
        *feature_cols,
        OPEN_TO_OPEN_RETURN_COL,
        LABEL_OPEN_TO_OPEN_5D_COL,
        ALPHA_TARGET_COL,
        CRASH_TARGET_COL,
        MAX_ADVERSE_TARGET_COL,
        CRASH_20D_TARGET_COL,
        MAX_ADVERSE_20D_TARGET_COL,
        RISK_TARGET_COL,
        VOL_ADJUSTED_TARGET_COL,
        POSITION_TARGET_COL,
        REGIME_TARGET_CODE_COL,
        INTRADAY_RISK_STATE_COL,
        "intraday_risk_score",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if REGIME_TARGET_CODE_COL not in out.columns and REGIME_TARGET_COL in out.columns:
        code_map = {label: index for index, label in enumerate(REGIME_LABELS)}
        out[REGIME_TARGET_CODE_COL] = out[REGIME_TARGET_COL].map(code_map)
    out[REGIME_TARGET_CODE_COL] = pd.to_numeric(out.get(REGIME_TARGET_CODE_COL, 2), errors="coerce").fillna(2).clip(
        0,
        len(REGIME_LABELS) - 1,
    )
    return out.dropna(subset=[OPEN_TO_OPEN_RETURN_COL]).sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _buyhold_metrics(frame: pd.DataFrame) -> dict[str, float]:
    position = pd.Series(1.0, index=frame.index)
    returns = pd.to_numeric(frame[OPEN_TO_OPEN_RETURN_COL], errors="coerce").fillna(0.0)
    turnover = pd.Series(0.0, index=frame.index)
    return _return_metrics(returns, position, turnover)


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
        "policy": "nested_outer_soxl_year_inner_soxl_calibrated_no_template_policy_grid",
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


def _policy_diagnostic_row(outer_metrics: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "outer_test_year": int(outer_metrics["outer_test_year"]),
        "candidate": outer_metrics["candidate"],
        "feature_set": outer_metrics.get("feature_set"),
        "feature_count": outer_metrics.get("feature_count"),
        "contains_soxx_constituents": outer_metrics.get("contains_soxx_constituents"),
        "soxx_constituent_feature_count": outer_metrics.get("soxx_constituent_feature_count"),
        "selected_regime_model": outer_metrics["selected_regime_model"],
        "policy_id": outer_metrics["policy_id"],
        "position_mode": outer_metrics.get("position_mode"),
        "selection_policy": selected.get("selection_policy"),
        "calibration_asset": selected.get("calibration_asset"),
        "soxl_drawdown_gate_applied": selected.get("soxl_drawdown_gate_applied"),
        "soxl_inner_excess_return_pct": selected.get("soxl_inner_excess_return_pct"),
        "soxl_inner_worst_drawdown_pct": selected.get("soxl_inner_worst_drawdown_pct"),
        "soxl_drawdown_shortfall_pct": selected.get("soxl_drawdown_shortfall_pct"),
        "cross_asset_avg_excess_return_pct": selected.get("cross_asset_avg_excess_return_pct"),
        "cross_asset_stress_pass_rate": selected.get("cross_asset_stress_pass_rate"),
        "cross_asset_worst_drawdown_pct": selected.get("cross_asset_worst_drawdown_pct"),
        "inner_beats_buyhold_rate": selected.get("inner_beats_buyhold_rate"),
        "inner_activity_pass_rate": selected.get("inner_activity_pass_rate"),
        "inner_pass_rate": selected.get("inner_pass_rate"),
        "test_intraday_risk_cap_day_ratio": outer_metrics.get("test_intraday_risk_cap_day_ratio"),
        "test_intraday_execution_replay_day_ratio": outer_metrics.get("test_intraday_execution_replay_day_ratio"),
        "test_multi_target_risk_cap_day_ratio": outer_metrics.get("test_multi_target_risk_cap_day_ratio"),
        "test_percentile_risk_cap_day_ratio": outer_metrics.get("test_percentile_risk_cap_day_ratio"),
        "test_drawdown_cooldown_day_ratio": outer_metrics.get("test_drawdown_cooldown_day_ratio"),
        "test_avg_abs_position": outer_metrics.get("test_avg_abs_position"),
        "beats_buyhold": outer_metrics.get("beats_buyhold"),
        "passes_drawdown": outer_metrics.get("passes_drawdown"),
    }


def _summary_payload(
    *,
    spec: TrainableRegimePolicyNestedWfSpec,
    sidecars: dict[str, str],
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    provider_manifest: dict[str, Any],
    provider_meta: dict[str, Any],
    source_audit: dict[str, Any],
    common_end: str,
    feature_cols: list[str],
    feature_set_configs: list[dict[str, Any]],
    annual_metrics: pd.DataFrame,
    annual_goal: dict[str, Any],
    overfit_audit: dict[str, Any],
    selected_candidates: pd.DataFrame,
    intraday_audit: dict[str, Any],
    constituent_audit: dict[str, Any],
    constituent_feature_cols: list[str],
) -> dict[str, Any]:
    recorder_artifacts = [
        f"artifacts/{spec.experiment}/{rel_path}"
        for rel_path in sidecars.values()
        if rel_path != "failure_modes.md"
    ]
    recorder_only = [f"artifacts/{spec.experiment}/positions.csv"]
    selected_feature_rows = selected_candidates.to_dict("records") if not selected_candidates.empty else []
    selected_soxx_count = (
        int(pd.to_numeric(selected_candidates.get("contains_soxx_constituents", pd.Series(dtype=bool)), errors="coerce").fillna(0).astype(bool).sum())
        if not selected_candidates.empty
        else 0
    )
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "workflow_config": "workflow.yaml",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "promotion_decision": (
            "candidate_for_validated_research_trainable_regime_policy"
            if annual_goal["required_policy_satisfied"] and overfit_audit["overfit_risk_status"] == "mitigated"
            else "not_promoted_trainable_regime_policy_goal_or_overfit_audit_failed"
        ),
        "benchmark_basis": "open_to_open",
        "execution_timing": "close_t_signal_next_open_to_following_open",
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "provider_manifest": (spec.provider_uri / "qlib_store_manifest.json").as_posix(),
        "provider_meta": provider_meta,
        "provider_build": provider_manifest,
        "source_audit": source_audit,
        "common_data_end": common_end,
        "symbols": list(ETF_UNIVERSE),
        "feature_source": FEATURE_SOURCE,
        "feature_sets": {
            "selection_policy": "nested_inner_feature_set_selection",
            "selected": "outer_year_selected_feature_set",
            "available": feature_set_configs,
            "selected_by_year": selected_feature_rows,
            "selected_feature_sets": (
                sorted({str(value) for value in selected_candidates["feature_set"].dropna().unique()})
                if "feature_set" in selected_candidates
                else []
            ),
            "feature_count": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "contains_soxx_constituents": bool(any(column in feature_cols for column in constituent_feature_cols)),
            "available_soxx_constituent_feature_count": int(
                sum(1 for column in constituent_feature_cols if column in feature_cols)
            ),
            "selected_outer_year_count": int(len(selected_candidates)),
            "selected_soxx_outer_year_count": selected_soxx_count,
            "soxx_constituent_feature_cols": [
                column for column in constituent_feature_cols if column in feature_cols
            ],
            "contains_asset_identity": any(
                "instrument" in column.lower() or "symbol" in column.lower() or "asset" in column.lower()
                for column in feature_cols
            ),
            "normalization": "per_asset_rolling_zscore_and_percentile",
        },
        "model_selection": {
            "selection_policy": SELECTION_POLICY,
            "selection_uses_outer_test": False,
            "final_calibration_asset": CALIBRATION_ASSET,
            "cross_asset_training_universe": list(ETF_UNIVERSE),
            "cross_asset_role": "overfit_stress_audit_not_primary_selection_target",
            "primary_selection_target": "SOXL inner-fold excess return with drawdown and activity gates",
            "inner_validation_years": int(spec.inner_validation_years),
            "search_space_type": "pre_registered_learned_position_model_family_and_feature_set_grid",
            "feature_set_selection": "nested_inner_soxl_calibrated",
            "available_feature_sets": [str(config["feature_set"]) for config in feature_set_configs],
            "manual_template_selection": False,
            "template_library_used": False,
            "manual_policy_grid_used": False,
            "selector_regime_source": "none",
            "selector_regime_matching": "disabled_manual_prior_year_regime_templates",
            "model_families": ["lgbm_classifier", "xgb_classifier", "lgbm_regressor", "xgb_regressor", "lgb_xgb_consensus"],
            "candidate_count": int(len(_policy_candidate_configs())),
            "candidate_feature_set_count": int(len(_policy_candidate_configs()) * len(feature_set_configs)),
            "selected_by_year": selected_candidates.to_dict("records") if not selected_candidates.empty else [],
        },
        "regime_classifier": {
            "type": "supervised_trainable_policy_classifier",
            "manual_template_selection": False,
            "target_col": REGIME_TARGET_COL,
            "target_code_col": REGIME_TARGET_CODE_COL,
            "target_labels": REGIME_LABELS,
            "training_target_source": "forward_return_crash_and_vol_adjusted_labels_inside_training_folds",
        },
        "policy_architecture": {
            "alpha_model": "opportunity_probability_plus_vol_adjusted_return",
            "risk_model": "independent_5d_20d_risk_crash_probability_max_adverse_and_percentile_discount",
            "risk_target_cols": [
                CRASH_TARGET_COL,
                MAX_ADVERSE_TARGET_COL,
                CRASH_20D_TARGET_COL,
                MAX_ADVERSE_20D_TARGET_COL,
                RISK_TARGET_COL,
            ],
            "position_target_col": POSITION_TARGET_COL,
            "position_policy": "learned_position_regressor_discounted_by_independent_risk_heads",
            "risk_budget": "path_dependent_drawdown_cooldown_cap_uses_realized_strategy_equity_only",
        },
        "intraday_risk_state": intraday_audit,
        "soxx_constituent_factors": constituent_audit,
        "annual_goal": annual_goal,
        "overfit_audit": overfit_audit,
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
    policy_diagnostics: pd.DataFrame,
) -> None:
    annual_metrics.to_csv(card / sidecars["annual_walk_forward"], index=False)
    inner_selection.to_csv(card / sidecars["inner_selection"], index=False)
    selected_candidates.to_csv(card / sidecars["selected_candidates"], index=False)
    selected_candidates.to_csv(card / sidecars["model_comparison"], index=False)
    policy_diagnostics.to_csv(card / sidecars["policy_diagnostics"], index=False)
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
                "check": "trainable_regime_classifier",
                "status": "pass",
                "detail": json.dumps(summary["regime_classifier"], sort_keys=True),
            },
            {
                "check": "intraday_risk_state",
                "status": "pass" if summary["intraday_risk_state"].get("status") == "ok" else "fail",
                "detail": json.dumps(summary["intraday_risk_state"], sort_keys=True),
            },
            {
                "check": "soxx_constituent_factors",
                "status": "pass" if summary["soxx_constituent_factors"].get("status") == "ok" else "fail",
                "detail": json.dumps(summary["soxx_constituent_factors"], sort_keys=True),
            },
        ]
    ).to_csv(card / sidecars["risk_analysis"], index=False)
    selected_by_feature_set = {
        str(name): sorted(int(year) for year in group["outer_test_year"].dropna().astype(int).unique())
        for name, group in selected_candidates.groupby("feature_set")
    } if not selected_candidates.empty and "feature_set" in selected_candidates else {}
    pd.DataFrame(
        [
            {
                "feature_set": str(config["feature_set"]),
                "role": "nested_selectable_model_feature_set",
                "selected_outer_years": selected_by_feature_set.get(str(config["feature_set"]), []),
                "feature_count": int(config["feature_count"]),
                "contains_asset_identity": bool(config["contains_asset_identity"]),
                "normalization": config["normalization"],
                "contains_intraday_risk_score": "intraday_risk_score" in config["feature_cols"],
                "contains_soxx_constituents": bool(config["contains_soxx_constituents"]),
                "soxx_constituent_feature_count": int(config["soxx_constituent_feature_count"]),
            }
            for config in summary["feature_sets"]["available"]
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(
            {
                "annual_goal": summary["annual_goal"],
                "overfit_audit": summary["overfit_audit"],
                "model_selection": summary["model_selection"],
                "feature_sets": summary["feature_sets"],
                "regime_classifier": summary["regime_classifier"],
                "policy_architecture": summary["policy_architecture"],
                "intraday_risk_state": summary["intraday_risk_state"],
                "soxx_constituent_factors": summary["soxx_constituent_factors"],
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
            f"  selection_policy: {SELECTION_POLICY}",
            "  selection_uses_outer_test: false",
            f"  final_calibration_asset: {CALIBRATION_ASSET}",
            "  cross_asset_role: overfit_stress_audit_not_primary_selection_target",
            f"  inner_validation_years: {summary['model_selection']['inner_validation_years']}",
            "  search_space_type: pre_registered_learned_position_model_family_and_feature_set_grid",
            "  feature_set_selection: nested_inner_soxl_calibrated",
            f"  available_feature_sets: {summary['model_selection']['available_feature_sets']}",
            "  manual_template_selection: false",
            "  template_library_used: false",
            "  manual_policy_grid_used: false",
            "  selector_regime_source: none",
            "regime_classifier:",
            "  type: supervised_trainable_policy_classifier",
            "  manual_template_selection: false",
            "policy_architecture:",
            "  alpha_model: opportunity_probability_plus_vol_adjusted_return",
            "  risk_model: independent_5d_20d_risk_crash_probability_max_adverse_and_percentile_discount",
            f"  position_target_col: {POSITION_TARGET_COL}",
            "  position_policy: learned_position_regressor_discounted_by_independent_risk_heads",
            "intraday_risk_state:",
            f"  uses_5min: {str(summary['intraday_risk_state']['uses_5min']).lower()}",
            f"  uses_15min: {str(summary['intraday_risk_state']['uses_15min']).lower()}",
            "  role: risk_cap_and_intraday_execution_replay_not_alpha_template",
            f"  execution_replay_fields: {str(summary['intraday_risk_state'].get('execution_replay_fields', False)).lower()}",
            "soxx_constituent_factors:",
            f"  provider_uri: {summary['soxx_constituent_factors']['provider_uri']}",
            "  role: normalized_lagged_model_features_not_template_not_gate",
            f"  feature_count: {summary['soxx_constituent_factors']['feature_count']}",
            f"  lag_days: {summary['soxx_constituent_factors']['lag_days']}",
            "feature_sets:",
            "  selection_policy: nested_inner_feature_set_selection",
            f"  selected_feature_sets: {summary['feature_sets']['selected_feature_sets']}",
            f"  selected_soxx_outer_year_count: {summary['feature_sets']['selected_soxx_outer_year_count']}",
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
        "- 做了什么：trainable regime policy nested WF 已移除人工模板、prior-year regime selector 和手写 policy grid；regime classifier、alpha model、risk model、learned position model 都在 inner folds 内训练，最终只用 SOXL inner 校准、drawdown/activity gate 和跨 ETF stress audit 选择模型族与特征集候选。\n"
        f"- 核心指标：annual target pass years={summary['annual_goal']['target_pass_test_years']}，"
        f"overfit audit={summary['overfit_audit']['overfit_risk_status']}，common_data_end={summary['common_data_end']}。\n"
        f"- 特征：SOXX 成分股 normalized lagged breadth/dispersion 特征可用 {summary['feature_sets']['available_soxx_constituent_feature_count']} 个，lag_days={summary['soxx_constituent_factors']['lag_days']}；它们是 nested feature-set 候选，不是强制 alpha 输入、模板或 gate。实际选中特征集={summary['feature_sets']['selected_feature_sets']}。\n"
        "- regime/window 差异：post_bear/range/source 这类人工形态模板不再参与 selector；rebound/trend/defensive/selective 仅作为监督学习标签和诊断输出，仓位由 learned_position_target 回归头给出，outer SOXL 年份只报告，不参与选择。\n"
        f"- 是否晋升：{summary['promotion_decision']}。\n"
        "- 风险：learned_position_target 的标签设计本身仍可能过拟合训练年代，尤其如果看完 2023-2026 后继续改标签；SOXX 当前成分股回填历史有 survivorship bias；多 3x ETF 仍共享权益 beta；5m/15m replay 只是盘中减仓执行近似，本轮即使被选择也不证明独立盘中 alpha。\n"
        "- 下一步：若 annual gate 未过，继续优化标签和可验证的新因子，不再按年份手工修模板或阈值 grid。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    return (
        "# Failure Modes\n\n"
        "- Manual regime template selection is rejected and is not part of the active selector or summary evidence.\n"
        "- Manual policy-grid selection is rejected; active candidates are learned-position model-family and feature-set choices only.\n"
        "- Regime classifier labels are supervised by forward outcomes inside training folds; bad label design can still overfit the training era.\n"
        "- Learned-position label design can still be data-snooped if edited after reading outer-test failures.\n"
        "- SOXX constituent diffusion features are nested-selectable, but the feature-set search space itself must stay frozen before future outer-test evaluation.\n"
        "- SOXX constituent features use current holdings backfilled through history and remain survivorship-biased until point-in-time holdings are available.\n"
        "- Leave-one-asset-out reduces single-symbol fitting but does not remove shared US equity beta risk.\n"
        "- 5m/15m intraday execution replay is available as learned-position candidate behavior, but it is not proof of same-day alpha generation.\n"
        f"- Current promotion decision: {summary['promotion_decision']}.\n"
    )
