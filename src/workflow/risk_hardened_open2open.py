from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import evaluate_ml_position_candidates
from src.backtest.ml_position import _build_model, _model_task, _positive_probability
from src.data import QlibProviderConfig, build_alpha158_legacy_dataset, init_qlib_provider, read_intraday_cache
from src.factor import build_soxx_constituent_factor_frame
from src.factor.high_chop_ml import DEFAULT_HIGH_CHOP_FEATURES, build_high_chop_ml_frame
from src.workflow.cognitive_alpha_goal_codex_exec import (
    BASE_CONSTRAINED_REF_COL,
    COGALPHA_DIRECT_ADDON_BASE_COL,
    COGALPHA_DIRECT_BLEND_BASE_COL,
    COGALPHA_DIRECT_RISKCUT_BASE_COL,
    COGALPHA_OPEN2OPEN_RISK_GATE_COL,
    DIRECT_FACTOR_BASE_POSITION_COL,
    DIRECT_FACTOR_RAW_TARGET_COL,
    EXECUTION_TIMING,
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_NATIVE_API,
    _add_return_columns,
    _attach_cogalpha_composite_base_features,
    _attach_open_to_open_execution_context,
    _attach_soxx_constituent_model_features,
    _dynamic_open_to_open_benchmarks,
    _ensure_soxx_constituent_qlib_provider,
    _merge_alpha158_features,
    _merge_high_chop_model_features,
)
from src.workflow.cognitive_alpha_position import CognitiveAlphaPositionResult, CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model


EXPERIMENT = "risk_hardened_open2open_v2"
CARD_TYPE = "risk_hardened_open2open_research_card"
RISK_HARDENED_GATE_COL = "risk_hardened_semis_risk_gate"
RISK_BUDGET_BASE_COL = "risk_budget_base"
COGALPHA_PRIMARY_COL = "vol_regime_trend_extension_reversion"
COGALPHA_MUTATION_COL = "g001_000_mutation_vol_regime_trend_e_89dafd5928"
SOXX_BREADTH_20_COL = "soxx_const_breadth_ret20_pos_lag1"
SOXX_DOWNSIDE_5_COL = "soxx_const_downside_ret5_lt_m5_lag1"


@dataclass(frozen=True)
class RiskHardenedOpen2OpenCardSpec:
    root: Path
    experiment: str = EXPERIMENT
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"
    validation_max_drawdown_floor: float = -20.0
    validation_max_daily_loss_floor: float = -12.0
    validation_cvar_5pct_floor: float = -8.0
    validation_worst_5_daily_loss_floor: float = -6.0
    max_validation_underwater_day_ratio: float = 0.95
    test_max_drawdown_floor: float = -20.0
    test_max_daily_loss_floor: float = -12.0
    test_cvar_5pct_floor: float = -8.0
    test_worst_5_daily_loss_floor: float = -6.0
    max_test_underwater_day_ratio: float = 0.95

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def soxx_constituent_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxx_constituents"


@dataclass(frozen=True)
class _FrameBundle:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    provider_meta: dict[str, Any]
    as_of: str
    test: tuple[str, str]
    alpha158_columns: list[str]
    constituent_feature_cols: list[str]
    constituent_factor_summary: dict[str, Any]


def run_risk_hardened_open2open_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = RiskHardenedOpen2OpenCardSpec(root=_resolve_root(root))
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


def _run_card(spec: RiskHardenedOpen2OpenCardSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    bundle = _build_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(bundle.frame, test=bundle.test)
    if not benchmarks:
        raise ValueError("risk-hardened card requires dynamic open-to-open benchmarks")

    sidecars = {
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
        "factor_ablation": "factor_ablation.csv",
        "training_window_comparison": "training_window_comparison.csv",
        "execution_stress": "execution_stress.csv",
        "execution_stress_summary": "execution_stress_summary.json",
        "explainability": "explainability.csv",
        "risk_analysis": "risk_analysis.csv",
        "latest_signal": "latest_signal.json",
        "position_summary": "position_summary.json",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.as_posix())

    position_spec = _position_spec_for_features(_feature_groups(bundle)["base_cogalpha_soxx"], spec=spec, test=bundle.test)
    position_result = run_cognitive_alpha_position_model(bundle.frame, spec=position_spec, benchmarks=benchmarks)
    _attach_open_to_open_execution_context(position_result.summary, position_result.latest_signal)
    feature_groups = _feature_groups(bundle)
    factor_ablation = _run_factor_ablation(bundle.frame, spec=spec, test=bundle.test, feature_groups=feature_groups, benchmarks=benchmarks)
    training_windows = _run_training_window_comparison(bundle.frame, spec=spec, test=bundle.test, feature_cols=feature_groups["base_cogalpha_soxx"], benchmarks=benchmarks)
    execution_stress, execution_summary = _run_execution_stress(bundle.frame, position_result.positions, root=spec.root)
    explainability = _permutation_importance(bundle.frame, position_result.summary, feature_groups["base_cogalpha_soxx"], spec=position_spec)
    risk_analysis = _risk_analysis_payload(position_result.summary, factor_ablation, execution_summary)

    with tempfile.TemporaryDirectory(prefix="risk_hardened_open2open_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        position_result.positions.to_csv(positions_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=bundle.provider_meta["provider_uri"],
                market=bundle.manifest["market"],
                region=bundle.manifest["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=bundle.test[0],
                test_end=bundle.test[1],
                return_col=OPEN_TO_OPEN_RETURN_COL,
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                model_families="lgbm_classifier,lgbm_regressor,xgb_classifier,xgb_regressor",
                validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
                test_max_daily_loss_floor=spec.test_max_daily_loss_floor,
                risk_gate_col=RISK_HARDENED_GATE_COL,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                position_result=position_result,
                factor_ablation=factor_ablation,
                training_windows=training_windows,
                execution_stress=execution_stress,
                execution_summary=execution_summary,
                explainability=explainability,
                risk_analysis=risk_analysis,
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path="risk_hardened_open2open")
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            recorder_id = recorder.id

    summary = _summary_payload(
        spec=spec,
        bundle=bundle,
        sidecars=sidecars,
        position_result=position_result,
        factor_ablation=factor_ablation,
        training_windows=training_windows,
        execution_summary=execution_summary,
        explainability=explainability,
        risk_analysis=risk_analysis,
        recorder_id=recorder_id,
        recorder_local_dir=recorder_local_dir,
        recorder_uri=recorder_uri,
    )
    _write_summary_files(card, summary)
    return summary


def _build_frame_bundle(spec: RiskHardenedOpen2OpenCardSpec) -> _FrameBundle:
    manifest = json.loads((spec.provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    as_of = str(manifest["calendar_end"])
    test = (spec.test_start, as_of)
    qlib_config = QlibProviderConfig(
        provider_uri=spec.provider_uri,
        region=str(manifest["region"]),
        market=str(manifest["market"]),
        kernels=1,
    )
    provider_meta = init_qlib_provider(qlib_config)

    from qlib.data import D

    fields = ["$open", "$high", "$low", "$close", "$volume", "$vix"]
    qlib_panel = D.features(["SOXL", "SOXX"], fields, start_time=spec.train[0], end_time=as_of, freq="day")
    frame = qlib_panel.reset_index().sort_values(["datetime", "instrument"]).reset_index(drop=True)
    frame = frame.loc[frame["instrument"].astype(str).str.upper().eq("SOXL")].copy()
    frame = frame.rename(columns={field: field.removeprefix("$") for field in fields})
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    _add_return_columns(frame)

    high_chop_frame = build_high_chop_ml_frame(
        qlib_panel,
        start=spec.train[0],
        end=as_of,
        label_horizon=5,
        include_incomplete_latest=True,
    )
    frame = _merge_high_chop_model_features(frame, high_chop_frame)
    alpha_dataset = build_alpha158_legacy_dataset(
        qlib_config,
        instruments=["SOXL"],
        label="Ref($open, -6) / Ref($open, -1) - 1",
        train=spec.train,
        valid=spec.validation,
        test=test,
    )
    frame, alpha158_columns = _merge_alpha158_features(frame, alpha_dataset)
    frame["realized_vol_20"] = pd.to_numeric(frame["close"], errors="coerce").pct_change().rolling(20, min_periods=10).std().shift(1)
    frame = _attach_static_cogalpha_features(frame)

    constituent_manifest = _ensure_soxx_constituent_qlib_provider(spec.root, as_of=as_of)
    constituent_end = min(pd.Timestamp(as_of), pd.Timestamp(str(constituent_manifest["calendar_end"]))).date().isoformat()
    constituent_factor_frame = build_soxx_constituent_factor_frame(
        QlibProviderConfig(
            provider_uri=spec.soxx_constituent_provider_uri,
            region=str(constituent_manifest["region"]),
            market=str(constituent_manifest["market"]),
            kernels=1,
        ),
        instruments=list(constituent_manifest["instruments"]),
        start=spec.train[0],
        end=constituent_end,
        lookbacks=(5, 20),
        lag_days=1,
        top_n=5,
    )
    frame, constituent_feature_cols = _attach_soxx_constituent_model_features(frame, constituent_factor_frame)
    frame = _attach_risk_hardened_gate(frame)
    frame[RISK_BUDGET_BASE_COL] = 1.0
    frame, composite_feature_cols = _attach_cogalpha_composite_base_features(frame)
    frame = _attach_risk_hardened_gate(frame)
    constituent_factor_summary = {
        "source": constituent_factor_frame.attrs.get("source", "qlib.data.D.features"),
        "provider_uri": constituent_factor_frame.attrs.get("provider_uri", spec.soxx_constituent_provider_uri.as_posix()),
        "raw_manifest": constituent_manifest.get("raw_manifest"),
        "qlib_store_manifest": (spec.soxx_constituent_provider_uri / "qlib_store_manifest.json").as_posix(),
        "calendar_end": constituent_manifest.get("calendar_end"),
        "feature_cols": constituent_feature_cols,
        "feature_count": len(constituent_feature_cols),
        "lookbacks": constituent_factor_frame.attrs.get("lookbacks", [5, 20]),
        "lag_days": constituent_factor_frame.attrs.get("lag_days", 1),
        "top_n": constituent_factor_frame.attrs.get("top_n", 5),
        "point_in_time_status": _soxx_point_in_time_status(spec.root),
        "limitations": [
            *list(constituent_factor_frame.attrs.get("limitations", [])),
            "single-current-holdings snapshot is marked as survivorship-biased until PIT holdings history is available",
        ],
    }
    frame.attrs["alpha158_columns"] = alpha158_columns
    frame.attrs["constituent_feature_cols"] = constituent_feature_cols
    frame.attrs["composite_feature_cols"] = composite_feature_cols
    init_qlib_provider(qlib_config)
    return _FrameBundle(
        frame=frame,
        manifest=manifest,
        provider_meta=provider_meta,
        as_of=as_of,
        test=test,
        alpha158_columns=alpha158_columns,
        constituent_feature_cols=constituent_feature_cols,
        constituent_factor_summary=constituent_factor_summary,
    )


def _rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    min_obs = min_periods or max(5, min(window, 20))
    mean = numeric.rolling(window, min_periods=min_obs).mean()
    std = numeric.rolling(window, min_periods=min_obs).std(ddof=0)
    return (numeric - mean) / std.replace(0.0, np.nan)


def _attach_static_cogalpha_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype("float64")
    ma60 = pd.to_numeric(out.get("alpha158_MA60"), errors="coerce").astype("float64")
    max30 = pd.to_numeric(out.get("alpha158_MAX30"), errors="coerce").astype("float64")
    ma20 = pd.to_numeric(out.get("alpha158_MA20"), errors="coerce").astype("float64")
    vix = pd.to_numeric(out.get("alpha158_VIX"), errors="coerce").astype("float64")
    term = (
        ((close / ma60) - 1.0).shift(1) * (1.0 + _rolling_zscore(vix, 20, 10).shift(1).abs())
        + ((close / max30) - 1.0).shift(1) * ((ma20 / ma60) - 1.0).shift(1)
    )
    out[COGALPHA_PRIMARY_COL] = np.tanh(-_rolling_zscore(term, 20, 10))
    out[COGALPHA_MUTATION_COL] = _rolling_zscore(out[COGALPHA_PRIMARY_COL], 10, 5).shift(1)
    direct = (-out[COGALPHA_MUTATION_COL]).clip(lower=0.0)
    max_abs = float(direct.abs().max() or 0.0)
    out[DIRECT_FACTOR_BASE_POSITION_COL] = (direct / max_abs).clip(0.0, 1.0) if max_abs > 1e-12 else 0.0
    out[DIRECT_FACTOR_RAW_TARGET_COL] = out[COGALPHA_MUTATION_COL].fillna(0.0)
    return out


def _attach_risk_hardened_gate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    base_gate = _numeric_column(out, COGALPHA_OPEN2OPEN_RISK_GATE_COL, default=0.0).gt(0.5)
    breadth20 = _numeric_column(out, SOXX_BREADTH_20_COL, default=np.nan)
    downside5 = _numeric_column(out, SOXX_DOWNSIDE_5_COL, default=np.nan)
    mutation = _numeric_column(out, COGALPHA_MUTATION_COL, default=np.nan)
    soxx_stress = breadth20.lt(0.45).fillna(False) | downside5.lt(-0.02).fillna(False)
    cogalpha_stress = mutation.gt(1.0).fillna(False)
    out[RISK_HARDENED_GATE_COL] = (base_gate | soxx_stress | cogalpha_stress).astype("float64")
    return out


def _numeric_column(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _soxx_point_in_time_status(root: Path) -> dict[str, Any]:
    holdings_dir = root / "data" / "raw" / "soxx_constituents" / "holdings"
    snapshots = sorted(holdings_dir.glob("soxx_holdings_*.csv"))
    return {
        "status": "single_snapshot_survivorship_biased" if len(snapshots) <= 1 else "multi_snapshot_partial_history",
        "snapshot_count": len(snapshots),
        "snapshots": [path.name for path in snapshots[-5:]],
    }


def _feature_groups(bundle: _FrameBundle) -> dict[str, list[str]]:
    frame = bundle.frame
    base = [
        *bundle.alpha158_columns[:6],
        *[column for column in DEFAULT_HIGH_CHOP_FEATURES if column in frame.columns],
        "vix",
        "realized_vol_20",
        RISK_BUDGET_BASE_COL,
        BASE_CONSTRAINED_REF_COL,
        COGALPHA_OPEN2OPEN_RISK_GATE_COL,
        RISK_HARDENED_GATE_COL,
    ]
    cogalpha = [
        COGALPHA_PRIMARY_COL,
        COGALPHA_MUTATION_COL,
        DIRECT_FACTOR_BASE_POSITION_COL,
        DIRECT_FACTOR_RAW_TARGET_COL,
        COGALPHA_DIRECT_ADDON_BASE_COL,
        COGALPHA_DIRECT_BLEND_BASE_COL,
        COGALPHA_DIRECT_RISKCUT_BASE_COL,
    ]
    soxx = bundle.constituent_feature_cols
    return {
        "base": _existing_unique(base, frame),
        "base_cogalpha": _existing_unique([*base, *cogalpha], frame),
        "base_soxx": _existing_unique([*base, *soxx], frame),
        "base_cogalpha_soxx": _existing_unique([*base, *cogalpha, *soxx], frame),
    }


def _existing_unique(columns: list[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in dict.fromkeys(columns) if column in frame.columns]


def _position_spec_for_features(
    feature_cols: list[str],
    *,
    spec: RiskHardenedOpen2OpenCardSpec,
    test: tuple[str, str],
    train: tuple[str, str] | None = None,
    validation: tuple[str, str] | None = None,
) -> CognitiveAlphaPositionSpec:
    model_specs = [
        {"model": "lgbm_classifier"},
        {"model": "lgbm_regressor", "task": "regression"},
        {"model": "xgb_classifier"},
        {"model": "xgb_regressor", "task": "regression"},
    ]
    common_objective = {
        "objective_baseline_position_col": BASE_CONSTRAINED_REF_COL,
        "objective_excess_return_weight": 0.08,
        "objective_excess_return_to_drawdown_weight": 6.0,
        "objective_return_weight": 0.01,
        "objective_return_to_drawdown_weight": 8.0,
        "objective_exposure_weight": 0.0,
        "objective_min_avg_abs_position": 0.03,
        "objective_max_drawdown_floor": spec.validation_max_drawdown_floor,
        "objective_max_daily_loss_floor": spec.validation_max_daily_loss_floor,
        "objective_cvar_5pct_floor": spec.validation_cvar_5pct_floor,
        "objective_worst_5_daily_loss_floor": spec.validation_worst_5_daily_loss_floor,
        "objective_max_underwater_day_ratio": spec.max_validation_underwater_day_ratio,
        "exposure_shortfall_penalty": 25.0,
        "drawdown_floor_penalty": 3.0,
        "daily_loss_floor_penalty": 4.0,
        "cvar_floor_penalty": 2.0,
        "worst_5_floor_penalty": 1.5,
        "underwater_penalty": 25.0,
        "turnover_penalty": 2.0,
        "reversal_penalty": 2.0,
        "trade_penalty": 0.15,
    }
    position_specs: list[dict[str, Any]] = []
    for low, high, min_throttle in [(0.40, 0.70, 0.0), (0.45, 0.70, 0.25), (0.50, 0.75, 0.35)]:
        position_specs.append(
            {
                "position_mode": "base_throttle",
                "base_position_col": RISK_BUDGET_BASE_COL,
                "threshold": 0.50,
                "low_probability": low,
                "high_probability": high,
                "min_throttle": min_throttle,
                "max_throttle": 1.0,
                "max_position": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": 0.50,
                "min_hold_days": 2,
                "risk_gate_col": RISK_HARDENED_GATE_COL,
                "risk_gate_threshold": 0.5,
                "risk_gate_direction": "above",
                "risk_gate_max_position": 0.35,
                **common_objective,
            }
        )
    for threshold, boost, risk_cap in [(0.55, 0.20, 0.35), (0.55, 0.35, 0.35), (0.65, 0.35, 0.50), (0.65, 0.50, 0.50)]:
        position_specs.append(
            {
                "position_mode": "base_plus_probability",
                "base_position_col": BASE_CONSTRAINED_REF_COL,
                "threshold": threshold,
                "base_scale": 1.0,
                "boost_scale": boost,
                "scale": 1.0,
                "max_position": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": 0.50,
                "min_hold_days": 3,
                "risk_gate_col": RISK_HARDENED_GATE_COL,
                "risk_gate_threshold": 0.5,
                "risk_gate_direction": "above",
                "risk_gate_max_position": risk_cap,
                **common_objective,
            }
        )
    return CognitiveAlphaPositionSpec(
        feature_cols=feature_cols,
        train=train or spec.train,
        validation=validation or spec.validation,
        test=test,
        model_specs=model_specs,
        position_specs=position_specs,
        return_col=OPEN_TO_OPEN_RETURN_COL,
        label_col=LABEL_OPEN_TO_OPEN_5D_COL,
        min_validation_trade_count=1,
        min_validation_avg_abs_position=0.01,
        max_validation_zero_position_day_ratio=0.98,
        validation_max_drawdown_floor=spec.validation_max_drawdown_floor,
        validation_max_daily_loss_floor=spec.validation_max_daily_loss_floor,
        validation_cvar_5pct_floor=spec.validation_cvar_5pct_floor,
        validation_worst_5_daily_loss_floor=spec.validation_worst_5_daily_loss_floor,
        max_validation_underwater_day_ratio=spec.max_validation_underwater_day_ratio,
        max_validation_total_turnover=20.0,
        max_validation_extreme_loss_day_count=7,
        max_validation_reversal_ratio=0.50,
        min_test_avg_abs_position=0.01,
        max_test_zero_position_day_ratio=0.98,
        test_max_drawdown_floor=spec.test_max_drawdown_floor,
        test_max_daily_loss_floor=spec.test_max_daily_loss_floor,
        test_cvar_5pct_floor=spec.test_cvar_5pct_floor,
        test_worst_5_daily_loss_floor=spec.test_worst_5_daily_loss_floor,
        max_test_underwater_day_ratio=spec.max_test_underwater_day_ratio,
        max_test_total_turnover=20.0,
        max_test_extreme_loss_day_count=7,
        constrained_reference_name=BASE_CONSTRAINED_REF_COL,
        extreme_loss_threshold=-0.05,
        walk_forward_folds=[
            {
                "fold": "wf_2025_h2",
                "train": ("2019-01-02", "2024-06-28"),
                "validation": ("2024-07-01", "2025-06-30"),
                "test": ("2025-07-01", "2025-12-31"),
            },
            {
                "fold": "wf_2026_ytd",
                "train": train or spec.train,
                "validation": validation or spec.validation,
                "test": test,
            },
        ],
    )


def _run_factor_ablation(
    frame: pd.DataFrame,
    *,
    spec: RiskHardenedOpen2OpenCardSpec,
    test: tuple[str, str],
    feature_groups: dict[str, list[str]],
    benchmarks: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, feature_cols in feature_groups.items():
        result = run_cognitive_alpha_position_model(
            frame,
            spec=_position_spec_for_features(feature_cols, spec=spec, test=test),
            benchmarks=benchmarks,
        )
        selected = result.summary["selected_candidate"]["test"]
        risk = result.summary["risk_gates"]
        comparison = result.summary["comparison"]
        rows.append(
            {
                "feature_group": group_name,
                "feature_count": len(feature_cols),
                "selected_candidate": result.summary["selected_candidate"]["candidate"],
                "selected_model": result.summary["selected_candidate"]["model"],
                "test_return_pct": selected["return_pct"],
                "test_max_drawdown_pct": selected["max_drawdown_pct"],
                "test_max_daily_loss_pct": selected["max_daily_loss_pct"],
                "test_cvar_5pct_pct": selected.get("cvar_5pct_pct"),
                "test_worst_5_daily_loss_pct": selected.get("worst_5_daily_loss_pct"),
                "test_underwater_day_ratio": selected.get("underwater_day_ratio"),
                "beats_base_constrained_ref": comparison.get("beats_base_constrained_ref"),
                "selected_vs_base_constrained_ref_return_gap_pct": comparison.get("selected_vs_base_constrained_ref_return_gap_pct"),
                "selected_passes_all": risk.get("selected_passes_all"),
                "walk_forward_all_gates_pass": result.summary["walk_forward"].get("all_folds_pass_risk_gates"),
                "promotion_decision": result.summary["promotion_decision"],
            }
        )
    return pd.DataFrame(rows)


def _run_training_window_comparison(
    frame: pd.DataFrame,
    *,
    spec: RiskHardenedOpen2OpenCardSpec,
    test: tuple[str, str],
    feature_cols: list[str],
    benchmarks: list[dict[str, Any]],
) -> pd.DataFrame:
    windows = [
        ("expanding_2019", ("2019-01-02", "2024-12-31"), ("2025-01-02", "2025-12-31")),
        ("post_2020", ("2020-01-02", "2024-12-31"), ("2025-01-02", "2025-12-31")),
        ("rolling_5y", ("2021-01-04", "2024-12-31"), ("2025-01-02", "2025-12-31")),
        ("rolling_3y", ("2022-01-03", "2024-12-31"), ("2025-01-02", "2025-12-31")),
    ]
    rows: list[dict[str, Any]] = []
    for name, train, validation in windows:
        window_spec = _position_spec_for_features(feature_cols, spec=spec, test=test, train=train, validation=validation)
        result = run_cognitive_alpha_position_model(frame, spec=window_spec, benchmarks=benchmarks)
        selected = result.summary["selected_candidate"]["test"]
        rows.append(
            {
                "training_window": name,
                "train_start": train[0],
                "train_end": train[1],
                "validation_start": validation[0],
                "validation_end": validation[1],
                "selected_candidate": result.summary["selected_candidate"]["candidate"],
                "selected_model": result.summary["selected_candidate"]["model"],
                "test_return_pct": selected["return_pct"],
                "test_max_drawdown_pct": selected["max_drawdown_pct"],
                "test_max_daily_loss_pct": selected["max_daily_loss_pct"],
                "test_cvar_5pct_pct": selected.get("cvar_5pct_pct"),
                "selected_passes_all": result.summary["risk_gates"]["selected_passes_all"],
                "walk_forward_all_gates_pass": result.summary["walk_forward"].get("all_folds_pass_risk_gates"),
                "promotion_decision": result.summary["promotion_decision"],
            }
        )
    return pd.DataFrame(rows)


def _run_execution_stress(
    frame: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    minute_path = root / "data" / "raw" / "intraday" / "futu" / "soxl_1m.parquet"
    minute = read_intraday_cache(minute_path)
    if minute.empty:
        empty = pd.DataFrame(
            [{"scenario": "no_intraday_cache", "coverage_days": 0, "return_pct": None, "max_drawdown_pct": None}]
        )
        return empty, {"status": "needs_data", "source": minute_path.as_posix(), "coverage_days": 0}
    minute = minute.copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"])
    minute["date"] = pd.to_datetime(minute["date"]).dt.normalize()
    minute = minute.loc[
        (minute["symbol"].astype(str).str.upper() == "SOXL")
        & (minute["session"].astype(str).str.lower() == "regular")
    ].copy()
    dates = list(pd.to_datetime(frame["datetime"]).dt.normalize())
    selected = positions.loc[positions["selected"].astype(bool) & positions["segment"].astype(str).eq("test")].copy()
    selected["datetime"] = pd.to_datetime(selected["datetime"]).dt.normalize()
    selected = selected.sort_values("datetime").reset_index(drop=True)
    rows_by_scenario: dict[str, list[dict[str, Any]]] = {name: [] for name in ["open", "vwap1", "vwap5", "vwap15"]}
    last_position = 0.0
    for _, row in selected.iterrows():
        signal_date = pd.Timestamp(row["datetime"]).normalize()
        if signal_date not in dates:
            continue
        index = dates.index(signal_date)
        if index + 2 >= len(dates):
            continue
        entry_date = dates[index + 1]
        exit_date = dates[index + 2]
        entry_bars = minute.loc[minute["date"].eq(entry_date)].sort_values("timestamp")
        exit_bars = minute.loc[minute["date"].eq(exit_date)].sort_values("timestamp")
        if entry_bars.empty or exit_bars.empty:
            continue
        exit_open = float(exit_bars.iloc[0]["open"])
        target_position = float(row["target_position"])
        turnover = abs(target_position - last_position)
        last_position = target_position
        open_price = float(entry_bars.iloc[0]["open"])
        adverse_15m = float(entry_bars.head(15)["low"].min() / open_price - 1.0) if open_price else 0.0
        for scenario, minutes in [("open", 1), ("vwap1", 1), ("vwap5", 5), ("vwap15", 15)]:
            window = entry_bars.head(minutes)
            if scenario == "open":
                entry_price = open_price
            else:
                volume = pd.to_numeric(window["volume"], errors="coerce").fillna(0.0)
                if float(volume.sum()) > 0.0:
                    entry_price = float((pd.to_numeric(window["close"], errors="coerce").fillna(open_price) * volume).sum() / volume.sum())
                else:
                    entry_price = float(window["close"].iloc[-1])
            strategy_return = target_position * (exit_open / entry_price - 1.0) - turnover * 0.0003
            rows_by_scenario[scenario].append(
                {
                    "scenario": scenario,
                    "signal_date": signal_date.date().isoformat(),
                    "entry_date": entry_date.date().isoformat(),
                    "exit_date": exit_date.date().isoformat(),
                    "target_position": target_position,
                    "entry_price": entry_price,
                    "exit_open": exit_open,
                    "strategy_return": strategy_return,
                    "entry_slippage_vs_open_pct": (entry_price / open_price - 1.0) * 100.0 if open_price else 0.0,
                    "first_15m_adverse_excursion_pct": adverse_15m * 100.0,
                }
            )
    metrics_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    for scenario, rows in rows_by_scenario.items():
        detail = pd.DataFrame(rows)
        if detail.empty:
            metrics_rows.append({"scenario": scenario, "coverage_days": 0, "return_pct": None, "max_drawdown_pct": None})
            continue
        returns = detail["strategy_return"].astype("float64")
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        metrics_rows.append(
            {
                "scenario": scenario,
                "coverage_days": int(len(detail)),
                "return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
                "max_drawdown_pct": float(drawdown.min() * 100.0),
                "max_daily_loss_pct": float(returns.min() * 100.0),
                "avg_entry_slippage_vs_open_pct": float(detail["entry_slippage_vs_open_pct"].mean()),
                "worst_first_15m_adverse_excursion_pct": float(detail["first_15m_adverse_excursion_pct"].min()),
                "adverse_5pct_day_count": int(detail["first_15m_adverse_excursion_pct"].le(-5.0).sum()),
            }
        )
        detail_rows.append(detail)
    metrics = pd.DataFrame(metrics_rows)
    summary = {
        "status": "ok" if int(metrics["coverage_days"].fillna(0).max()) > 0 else "needs_data",
        "source": minute_path.as_posix(),
        "coverage_days": int(metrics["coverage_days"].fillna(0).max()) if not metrics.empty else 0,
        "scenarios": metrics.to_dict(orient="records"),
        "detail_policy": "temporary_detail_not_committed",
    }
    return metrics, summary


def _permutation_importance(
    frame: pd.DataFrame,
    position_summary: dict[str, Any],
    feature_cols: list[str],
    *,
    spec: CognitiveAlphaPositionSpec,
    max_features: int = 40,
) -> pd.DataFrame:
    selected = position_summary["selected_candidate"]
    model_name = str(selected["model"])
    selected_test = selected["test"]
    params = json.loads(selected_test.get("params") or "{}")
    task = _model_task(model_name, "regression" if model_name.endswith("regressor") else None)
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"]).dt.normalize()
    train = data.loc[
        (data["datetime"] >= pd.Timestamp(spec.train[0])) & (data["datetime"] <= pd.Timestamp(spec.train[1]))
    ].dropna(subset=[spec.label_col])
    validation = data.loc[
        (data["datetime"] >= pd.Timestamp(spec.validation[0])) & (data["datetime"] <= pd.Timestamp(spec.validation[1]))
    ].dropna(subset=[spec.label_col])
    if train.empty or validation.empty:
        return pd.DataFrame()
    model = _build_model(model_name, random_state=42)
    x_train = train[feature_cols].astype("float64")
    if task == "classification":
        y_train = train[spec.label_col].gt(0.0).astype(int)
        model.fit(x_train, y_train)
        train_score = None
    else:
        y_train = train[spec.label_col].astype("float64")
        model.fit(x_train, y_train)
        train_score = pd.Series(model.predict(x_train), index=train.index, name="score")
    x_valid = validation[feature_cols].astype("float64")
    label = validation[spec.label_col].astype("float64")
    base_prob = _positive_probability(model, x_valid, train_score=train_score)
    base_rank_ic = float(base_prob.corr(label, method="spearman") or 0.0)
    rng = np.random.default_rng(7)
    rows: list[dict[str, Any]] = []
    candidate_cols = feature_cols[:max_features]
    for column in candidate_cols:
        shuffled = x_valid.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        prob = _positive_probability(model, shuffled, train_score=train_score)
        rank_ic = float(prob.corr(label, method="spearman") or 0.0)
        rows.append(
            {
                "method": "validation_probability_rank_ic_permutation",
                "model": model_name,
                "position_mode": params.get("position_mode"),
                "feature": column,
                "baseline_rank_ic": base_rank_ic,
                "permuted_rank_ic": rank_ic,
                "importance_rank_ic_drop": base_rank_ic - rank_ic,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_rank_ic_drop", ascending=False).reset_index(drop=True)


def _risk_analysis_payload(
    position_summary: dict[str, Any],
    factor_ablation: pd.DataFrame,
    execution_summary: dict[str, Any],
) -> pd.DataFrame:
    selected_test = position_summary["selected_candidate"]["test"]
    base_ablation = factor_ablation.loc[factor_ablation["feature_group"].eq("base")]
    both_ablation = factor_ablation.loc[factor_ablation["feature_group"].eq("base_cogalpha_soxx")]
    base_return = float(base_ablation.iloc[0]["test_return_pct"]) if not base_ablation.empty else np.nan
    both_return = float(both_ablation.iloc[0]["test_return_pct"]) if not both_ablation.empty else np.nan
    return pd.DataFrame(
        [
            {
                "check": "strict_position_risk_gates",
                "status": "pass" if position_summary["risk_gates"]["selected_passes_all"] else "fail",
                "value": position_summary["risk_gates"]["selected_passes_all"],
                "detail": json.dumps(position_summary["risk_gates"], sort_keys=True),
            },
            {
                "check": "walk_forward_risk_gates",
                "status": "pass" if position_summary["walk_forward"].get("all_folds_pass_risk_gates") else "fail",
                "value": position_summary["walk_forward"].get("all_folds_pass_risk_gates"),
                "detail": json.dumps(position_summary["walk_forward"], sort_keys=True),
            },
            {
                "check": "factor_ablation_incremental_return",
                "status": "pass" if pd.notna(base_return) and pd.notna(both_return) and both_return > base_return else "fail",
                "value": both_return - base_return if pd.notna(base_return) and pd.notna(both_return) else None,
                "detail": "base_cogalpha_soxx minus base test_return_pct",
            },
            {
                "check": "intraday_execution_stress",
                "status": execution_summary.get("status", "needs_data"),
                "value": execution_summary.get("coverage_days"),
                "detail": json.dumps(execution_summary.get("scenarios", []), sort_keys=True),
            },
            {
                "check": "selected_tail_risk",
                "status": "observe",
                "value": selected_test.get("max_daily_loss_pct"),
                "detail": json.dumps(
                    {
                        "max_daily_loss_pct": selected_test.get("max_daily_loss_pct"),
                        "cvar_5pct_pct": selected_test.get("cvar_5pct_pct"),
                        "worst_5_daily_loss_pct": selected_test.get("worst_5_daily_loss_pct"),
                        "underwater_day_ratio": selected_test.get("underwater_day_ratio"),
                    },
                    sort_keys=True,
                ),
            },
        ]
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    position_result: CognitiveAlphaPositionResult,
    factor_ablation: pd.DataFrame,
    training_windows: pd.DataFrame,
    execution_stress: pd.DataFrame,
    execution_summary: dict[str, Any],
    explainability: pd.DataFrame,
    risk_analysis: pd.DataFrame,
) -> None:
    position_result.metrics.to_csv(card / sidecars["model_selection"], index=False)
    position_result.benchmark_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    position_result.walk_forward_metrics.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    factor_ablation.to_csv(card / sidecars["factor_ablation"], index=False)
    training_windows.to_csv(card / sidecars["training_window_comparison"], index=False)
    execution_stress.to_csv(card / sidecars["execution_stress"], index=False)
    (card / sidecars["execution_stress_summary"]).write_text(
        json.dumps(execution_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    explainability.to_csv(card / sidecars["explainability"], index=False)
    risk_analysis.to_csv(card / sidecars["risk_analysis"], index=False)
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(position_result.latest_signal, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["position_summary"]).write_text(
        json.dumps(position_result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _summary_payload(
    *,
    spec: RiskHardenedOpen2OpenCardSpec,
    bundle: _FrameBundle,
    sidecars: dict[str, str],
    position_result: CognitiveAlphaPositionResult,
    factor_ablation: pd.DataFrame,
    training_windows: pd.DataFrame,
    execution_summary: dict[str, Any],
    explainability: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    recorder_id: str,
    recorder_local_dir: Path,
    recorder_uri: Path,
) -> dict[str, Any]:
    selected_test = position_result.summary["selected_candidate"]["test"]
    risk_rows = risk_analysis.set_index("check") if not risk_analysis.empty else pd.DataFrame()
    promotion_decision = _promotion_decision(position_result.summary, execution_summary)
    return {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "status": "complete",
        "promotion_decision": promotion_decision,
        "as_of": bundle.as_of,
        "provider_uri": bundle.provider_meta["provider_uri"],
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "market": bundle.manifest["market"],
        "region": bundle.manifest["region"],
        "segments": {"train": list(spec.train), "validation": list(spec.validation), "test": list(bundle.test)},
        "benchmark_basis": "open_to_open",
        "execution_timing": EXECUTION_TIMING,
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "analysis": "https://qlib.readthedocs.io/en/latest/component/report.html",
        },
        "qlib_native_api": QLIB_NATIVE_API,
        "model_selection": {
            "source": "qlib_dataset_and_D_features",
            "model_families": position_result.summary["model_families"],
            "candidate_count": int(position_result.metrics["candidate"].nunique()),
            "selection_metric": "risk_adjusted_excess_return_with_tail_penalties",
            "selection_column": "selection_objective_score",
            "selected_model": position_result.summary["selected_candidate"]["model"],
            "selected_candidate": position_result.summary["selected_candidate"]["candidate"],
        },
        "position_model": position_result.summary,
        "latest_signal": position_result.latest_signal,
        "risk_hardened_goal": {
            "target": "beat base_constrained_ref while reducing drawdown, max daily loss, execution fragility",
            "selected_test_return_pct": selected_test["return_pct"],
            "selected_test_max_drawdown_pct": selected_test["max_drawdown_pct"],
            "selected_test_max_daily_loss_pct": selected_test["max_daily_loss_pct"],
            "selected_test_cvar_5pct_pct": selected_test.get("cvar_5pct_pct"),
            "selected_test_worst_5_daily_loss_pct": selected_test.get("worst_5_daily_loss_pct"),
            "risk_gate_pass": position_result.summary["risk_gates"]["selected_passes_all"],
            "walk_forward_gate_pass": position_result.summary["walk_forward"].get("all_folds_pass_risk_gates"),
            "intraday_execution_status": execution_summary.get("status"),
            "strict_checks": risk_analysis.to_dict(orient="records"),
        },
        "factor_ablation": {
            "groups": factor_ablation.to_dict(orient="records"),
            "best_group_by_return": (
                factor_ablation.sort_values("test_return_pct", ascending=False).iloc[0].to_dict()
                if not factor_ablation.empty
                else None
            ),
        },
        "training_window_comparison": {
            "windows": training_windows.to_dict(orient="records"),
            "best_window_by_return": (
                training_windows.sort_values("test_return_pct", ascending=False).iloc[0].to_dict()
                if not training_windows.empty
                else None
            ),
        },
        "constituent_factors": bundle.constituent_factor_summary,
        "execution_stress": execution_summary,
        "explainability": {
            "method": "validation_probability_rank_ic_permutation",
            "row_count": int(len(explainability)),
            "top_features": explainability.head(10).to_dict(orient="records") if not explainability.empty else [],
        },
        "risk_analysis": risk_analysis.to_dict(orient="records"),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "recorder_only_artifacts": {
                "positions": "artifacts/risk_hardened_open2open/positions.csv",
            },
        },
        "implementation_notes": {
            "codex_factor_generation": "not_rerun_reused_v1_selected_formula",
            "lgb_xgb_required": True,
            "whole_position_budget_candidates": True,
            "base_plus_probability_candidates": True,
            "pit_soxx_holdings_status": bundle.constituent_factor_summary["point_in_time_status"],
            "risk_analysis_pass_count": int(risk_analysis["status"].eq("pass").sum()) if not risk_analysis.empty else 0,
            "risk_analysis_fail_count": int(risk_analysis["status"].eq("fail").sum()) if not risk_analysis.empty else 0,
        },
    }


def _promotion_decision(position_summary: dict[str, Any], execution_summary: dict[str, Any]) -> str:
    comparison = position_summary["comparison"]
    risk = position_summary["risk_gates"]
    walk_forward = position_summary["walk_forward"]
    if not bool(comparison.get("beats_base_constrained_ref", False)):
        return "not_promoted_under_base_constrained_ref"
    if not bool(risk.get("selected_passes_all", False)):
        return "not_promoted_failed_strict_tail_risk_gates"
    if not bool(walk_forward.get("all_folds_pass_risk_gates", False)):
        return "not_promoted_failed_walk_forward_risk_gates"
    if execution_summary.get("status") != "ok":
        return "not_promoted_needs_intraday_execution_data"
    return "candidate_for_validated_research_risk_hardened"


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            f"recorder_id: {summary['recorder_id']}",
            f"as_of: {summary['as_of']}",
            "qlib:",
            f"  market: {summary['market']}",
            f"  region: {summary['region']}",
            "model_selection:",
            f"  return_col: {summary['return_col']}",
            f"  label_col: {summary['model_target_col']}",
            "  models: [lgbm_classifier, lgbm_regressor, xgb_classifier, xgb_regressor]",
            "  objective: risk_adjusted_excess_return_with_tail_penalties",
            "risk_gates:",
            f"  risk_gate_col: {RISK_HARDENED_GATE_COL}",
            f"  selected_passes_all: {summary['position_model']['risk_gates']['selected_passes_all']}",
            f"  walk_forward_passes_all: {summary['position_model']['walk_forward'].get('all_folds_pass_risk_gates')}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "  recorder_only_artifacts:",
            "    positions: artifacts/risk_hardened_open2open/positions.csv",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    risk = summary["risk_hardened_goal"]
    lines = [
        "# Failure Modes",
        "",
        f"- promotion_decision: {summary['promotion_decision']}",
        f"- strict_position_risk_gate_pass: {risk['risk_gate_pass']}",
        f"- walk_forward_gate_pass: {risk['walk_forward_gate_pass']}",
        f"- intraday_execution_status: {risk['intraday_execution_status']}",
        f"- soxx_pit_status: {summary['constituent_factors']['point_in_time_status']['status']}",
        "- survivorship_bias: SOXX 成分股仍是 current holdings backfill，未达到 point-in-time holdings 生产标准。",
        "- execution_gap: 1m stress 只覆盖已有 Futu cache，仍不是真实订单簿/夜盘成交证明。",
        "- model_dependency: LGB/XGB 仍依赖 base/risk-gate priors；若 base_constrained_ref regime 失效，模型会同步退化。",
        "",
    ]
    return "\n".join(lines)


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary["position_model"]["selected_candidate"]["test"]
    latest = summary["latest_signal"]
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {summary['experiment']}，用 qlib `D.features`/Alpha158、SOXX 成分股 qlib store、"
        "LGB/XGB 和 open-to-open return 重新训练 risk-hardened position selector。\n"
        f"- key_metrics: selected_return={selected['return_pct']:.4f}%，maxDD={selected['max_drawdown_pct']:.4f}%，"
        f"max_daily_loss={selected['max_daily_loss_pct']:.4f}%，CVaR5={selected.get('cvar_5pct_pct', 0.0):.4f}%。\n"
        f"- regime/window: factor_ablation 最优组为 {summary['factor_ablation']['best_group_by_return']['feature_group']}；"
        f"training_window 最优为 {summary['training_window_comparison']['best_window_by_return']['training_window']}。\n"
        f"- execution: intraday_status={summary['execution_stress']['status']}，coverage_days={summary['execution_stress']['coverage_days']}，"
        "open/VWAP1/VWAP5/VWAP15 stress 已写入 sidecar。\n"
        f"- latest_signal: as_of={latest['as_of']}，action={latest['action']}，target_position={latest['target_percent']:.4f}%。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- risks: SOXX holdings 仍有 survivorship bias；分钟验证是执行压力测试，不是实盘成交保证；不构成买卖建议。\n"
        f"- next_action: 若未晋升，优先补 PIT SOXX holdings 和更长 1m/order-flow 数据，再放宽或重训风险预算模型。recorder={summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "risk-hardened-open2open-title",
                "metadata": {},
                "source": [
                    f"# {summary['experiment']}\n",
                    "\n",
                    "qlib-native risk-hardened SOXL open-to-open research card using LGB/XGB, ablation, training-window comparison, and 1m execution stress.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "risk-hardened-open2open-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from __future__ import annotations\n",
                    "import json, sys\n",
                    "from pathlib import Path\n",
                    "ROOT = Path.cwd().resolve()\n",
                    "while not (ROOT / 'AGENTS.md').exists() and ROOT != ROOT.parent:\n",
                    "    ROOT = ROOT.parent\n",
                    "if ROOT.as_posix() not in sys.path:\n",
                    "    sys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow import run_risk_hardened_open2open_card\n",
                    "# qlib.init, D.features, DatasetH/DataHandlerLP via build_alpha158_legacy_dataset, R.start\n",
                    "# evaluate_ml_position_candidates, LGBMClassifier/LGBMRegressor, XGBClassifier/XGBRegressor\n",
                    "summary = run_risk_hardened_open2open_card(root=ROOT)\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'as_of': summary['as_of']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "risk-hardened-open2open-conclusion",
                "metadata": {},
                "source": ["The execution cell writes `summary.json`, `workflow.yaml`, sidecars, recorder artifacts, and `conclusion.md`."],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{summary['experiment']}.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
