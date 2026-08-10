from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _build_teacher_training_frame,
    _load_source_frame,
    _risk_metrics,
    _simulate_policy,
    mark_to_market_weights,
    project_legal_weights,
)
from src.workflow.stage1_controller_aware_distillation import (
    CANDIDATES as DISTILLATION_CANDIDATES,
    Stage1ControllerAwareDistillationSpec,
    _annual_metrics,
    _build_qlib_dataset as _build_distillation_dataset,
    _build_teacher_targets as _build_distillation_targets,
    _candidate_row,
    _fit_models as _fit_distillation_models,
    _remove_qlib_code_snapshot_artifacts,
)
from src.workflow.stage1_transition_aware_action_value import (
    SOURCE_EXPERIMENT,
    SOURCE_RECORDER_ID,
    _baseline_contexts,
    _distillation_candidate_targets,
    _source_contract,
)


EXPERIMENT = "stage1_transition_aware_action_value_v2"
ROUTER_FEATURES = (
    "portfolio_drawdown",
    "early_ret_soxl",
    "early_dislocation",
    "tail_probability",
    "candidate_beta_delta",
    "candidate_current_turnover",
)
LABEL = "net_advantage_bps"


@dataclass(frozen=True)
class Stage1TransitionAwareActionValueV2Spec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    first_model_year: int = 2021
    teacher_start_year: int = 2019
    teacher_end_year: int = 2022
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    ridge_alpha: float = 10.0
    lower_residual_quantile: float = 0.10
    minimum_fit_rows: int = 20
    minimum_operational_rows: int = 40
    minimum_calibration_rows: int = 20

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_stage1_transition_aware_action_value_v2_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = Stage1TransitionAwareActionValueV2Spec(root=_resolve_root(root))
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


def _action_value_frame(
    contexts: pd.DataFrame,
    *,
    baseline_replay: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1TransitionAwareActionValueV2Spec,
) -> pd.DataFrame:
    """Build point-in-time features and a future-aware outcome label.

    The next-session reset is intentionally used only in the label.  The
    feature contract exposes current turnover, which is available at 10:31.
    """

    baseline = baseline_replay.set_index("signal_date")
    candidates = candidate_targets.set_index("signal_date")
    ordered = contexts.sort_values("signal_date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, row in ordered.iterrows():
        if row["signal_date"] not in candidates.index or not np.isfinite(
            row["value_at_cut"]
        ):
            continue
        baseline_row = baseline.loc[row["signal_date"]]
        baseline_target = float(baseline_row["recovery_beta_target"])
        candidate_target = max(
            baseline_target,
            float(candidates.loc[row["signal_date"], "candidate_beta_target"]),
        )
        if candidate_target <= baseline_target + 1e-12:
            continue

        baseline_weights = (
            project_legal_weights(
                row["raw_soxl_weight"],
                row["raw_soxx_weight"],
                baseline_target,
                minimum_cash_weight=source_spec.minimum_cash_weight,
                maximum_soxl_weight=source_spec.maximum_soxl_weight,
            )
            if bool(baseline_row["recovery_action"])
            else row["drifted_weights"].copy()
        )
        candidate_weights = project_legal_weights(
            row["raw_soxl_weight"],
            row["raw_soxx_weight"],
            candidate_target,
            minimum_cash_weight=source_spec.minimum_cash_weight,
            maximum_soxl_weight=source_spec.maximum_soxl_weight,
        )
        baseline_gross = float(
            row["value_at_cut"]
            * (1.0 + baseline_weights @ row["post_returns"])
            - 1.0
        )
        candidate_gross = float(
            row["value_at_cut"]
            * (1.0 + candidate_weights @ row["post_returns"])
            - 1.0
        )
        baseline_current_turnover = float(
            np.abs(baseline_weights[:2] - row["drifted_weights"][:2]).sum()
        )
        candidate_current_turnover = float(
            np.abs(candidate_weights[:2] - row["drifted_weights"][:2]).sum()
        )
        baseline_next_reset_turnover = 0.0
        candidate_next_reset_turnover = 0.0
        if index + 1 < len(ordered):
            next_open = ordered.iloc[index + 1]["open_weights"]
            baseline_end = mark_to_market_weights(
                baseline_weights, row["post_returns"]
            )
            candidate_end = mark_to_market_weights(
                candidate_weights, row["post_returns"]
            )
            baseline_next_reset_turnover = float(
                np.abs(next_open[:2] - baseline_end[:2]).sum()
            )
            candidate_next_reset_turnover = float(
                np.abs(next_open[:2] - candidate_end[:2]).sum()
            )
        incremental_transition_turnover = (
            candidate_current_turnover
            + candidate_next_reset_turnover
            - baseline_current_turnover
            - baseline_next_reset_turnover
        )
        net_advantage = candidate_gross - baseline_gross - (
            incremental_transition_turnover
            * spec.stress_cost_bps
            / 10_000.0
        )
        rows.append(
            {
                "signal_date": row["signal_date"],
                "segment": row["segment"],
                "year": int(row["year"]),
                "portfolio_drawdown": float(row["portfolio_drawdown"]),
                "early_ret_soxl": float(row["early_ret_soxl"]),
                "early_dislocation": float(row["early_dislocation"]),
                "tail_probability": float(row["tail_probability"]),
                "candidate_beta_delta": candidate_target - baseline_target,
                "candidate_current_turnover": candidate_current_turnover,
                "baseline_current_turnover": baseline_current_turnover,
                "candidate_next_reset_turnover": candidate_next_reset_turnover,
                "baseline_next_reset_turnover": baseline_next_reset_turnover,
                "incremental_transition_turnover": incremental_transition_turnover,
                "candidate_beta_target": candidate_target,
                LABEL: net_advantage * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def _build_qlib_dataset(
    training: pd.DataFrame, spec: Stage1TransitionAwareActionValueV2Spec
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat(
        {
            "feature": training.loc[:, list(ROUTER_FEATURES)].astype("float64"),
            "label": training.loc[:, [LABEL]].astype("float64"),
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(training["signal_date"]).dt.normalize(),
            np.full(len(training), "STAGE1_ACTION_VALUE_V2"),
        ],
        names=["datetime", "instrument"],
    )
    start = pd.to_datetime(training["signal_date"]).min().date().isoformat()
    segments: dict[str, tuple[str, str]] = {
        "all": (start, f"{spec.teacher_end_year}-12-31"),
        "train_outer": (start, f"{spec.teacher_end_year}-12-31"),
    }
    for year in range(spec.first_model_year, spec.teacher_end_year + 1):
        segments[f"train_{year}"] = (start, f"{year - 1}-12-31")
        segments[f"year_{year}"] = (f"{year}-01-01", f"{year}-12-31")
    dataset = DatasetH(
        handler=DataHandlerLP(
            instruments=None,
            start_time=static.index.get_level_values("datetime").min(),
            end_time=static.index.get_level_values("datetime").max(),
            data_loader=StaticDataLoader(config=static),
            infer_processors=[],
            learn_processors=[],
        ),
        segments=segments,
    )
    audit = {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "features": list(ROUTER_FEATURES),
        "label": LABEL,
        "label_uses_future_reset_cost": True,
        "feature_uses_future_reset_cost": False,
        "segments": {key: list(value) for key, value in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
    }
    return dataset, audit


def _prepared(dataset, segment: str) -> pd.DataFrame:
    frame = dataset.prepare(segment, col_set=["feature", "label"])
    if not frame.empty:
        frame.columns = [str(column[-1]) for column in frame.columns]
    return frame


def _new_model(spec: Stage1TransitionAwareActionValueV2Spec):
    return make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        Ridge(alpha=spec.ridge_alpha),
    )


def _fit_models(dataset, spec: Stage1TransitionAwareActionValueV2Spec):
    models: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    for model_year in range(
        spec.first_model_year, spec.teacher_end_year + 2
    ):
        train_segment = (
            "train_outer"
            if model_year == spec.teacher_end_year + 1
            else f"train_{model_year}"
        )
        train = _prepared(dataset, train_segment)
        residuals: list[float] = []
        fold_predictions: list[float] = []
        fold_labels: list[float] = []
        for validation_year in range(
            spec.first_model_year, min(model_year, spec.teacher_end_year + 1)
        ):
            fold_train = _prepared(dataset, f"train_{validation_year}")
            fold_test = _prepared(dataset, f"year_{validation_year}")
            if (
                len(fold_train) < spec.minimum_fit_rows
                or fold_test.empty
            ):
                continue
            fold_model = _new_model(spec)
            fold_model.fit(
                fold_train.loc[:, list(ROUTER_FEATURES)], fold_train[LABEL]
            )
            prediction = fold_model.predict(
                fold_test.loc[:, list(ROUTER_FEATURES)]
            )
            labels = fold_test[LABEL].to_numpy(dtype="float64")
            residuals.extend((labels - prediction).tolist())
            fold_predictions.extend(prediction.tolist())
            fold_labels.extend(labels.tolist())
            for index, predicted, actual in zip(
                fold_test.index, prediction, labels, strict=True
            ):
                oof_rows.append(
                    {
                        "model_year": model_year,
                        "validation_year": validation_year,
                        "datetime": index[0],
                        "predicted_net_advantage_bps": float(predicted),
                        "actual_net_advantage_bps": float(actual),
                        "residual_bps": float(actual - predicted),
                    }
                )
        ready = bool(
            len(train) >= spec.minimum_operational_rows
            and len(residuals) >= spec.minimum_calibration_rows
        )
        lower_offset = (
            float(np.quantile(residuals, spec.lower_residual_quantile))
            if residuals
            else np.nan
        )
        if ready:
            model = _new_model(spec)
            model.fit(train.loc[:, list(ROUTER_FEATURES)], train[LABEL])
            payload = {
                "model": model,
                "lower_residual_offset_bps": lower_offset,
                "training_rows": len(train),
                "calibration_rows": len(residuals),
                "training_end": f"{model_year - 1}-12-31",
            }
            if model_year == spec.teacher_end_year + 1:
                for outer_year in spec.outer_years:
                    models[outer_year] = payload
                diagnostic_year: Any = "outer_frozen_2023_2026"
            else:
                models[model_year] = payload
                diagnostic_year = model_year
        else:
            diagnostic_year = model_year
        correlation = (
            float(
                pd.Series(fold_predictions).corr(
                    pd.Series(fold_labels), method="spearman"
                )
            )
            if len(fold_predictions) >= 2
            else np.nan
        )
        diagnostics.append(
            {
                "model_year": diagnostic_year,
                "training_rows": len(train),
                "training_end": f"{model_year - 1}-12-31",
                "calibration_rows": len(residuals),
                "lower_residual_offset_bps": lower_offset,
                "oof_spearman": correlation,
                "oof_mae_bps": (
                    float(
                        np.mean(
                            np.abs(
                                np.asarray(fold_labels)
                                - np.asarray(fold_predictions)
                            )
                        )
                    )
                    if fold_predictions
                    else np.nan
                ),
                "operational_model_ready": ready,
            }
        )
    return models, pd.DataFrame(diagnostics), pd.DataFrame(oof_rows)


def _simulate_router(
    contexts: pd.DataFrame,
    *,
    baseline_replay: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    models: dict[int, dict[str, Any]],
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1TransitionAwareActionValueV2Spec,
    cost_bps: float,
) -> pd.DataFrame:
    baseline = baseline_replay.set_index("signal_date")
    candidates = candidate_targets.set_index("signal_date")
    previous = np.zeros(3, dtype="float64")
    rows: list[dict[str, Any]] = []
    for _, row in contexts.sort_values("signal_date").iterrows():
        baseline_row = baseline.loc[row["signal_date"]]
        baseline_action = bool(baseline_row["recovery_action"])
        baseline_target = float(baseline_row["recovery_beta_target"])
        candidate_target = max(
            baseline_target,
            float(candidates.loc[row["signal_date"], "candidate_beta_target"]),
        )
        candidate_weights = project_legal_weights(
            row["raw_soxl_weight"],
            row["raw_soxx_weight"],
            candidate_target,
            minimum_cash_weight=source_spec.minimum_cash_weight,
            maximum_soxl_weight=source_spec.maximum_soxl_weight,
        )
        candidate_current_turnover = float(
            np.abs(candidate_weights[:2] - row["drifted_weights"][:2]).sum()
        )
        predicted = np.nan
        lower_bound = np.nan
        override = False
        reason = "candidate_not_above_stage1"
        model_payload = models.get(int(row["year"]))
        if candidate_target > baseline_target + 1e-12:
            reason = "model_or_calibration_unavailable"
            if model_payload is not None:
                feature = pd.DataFrame(
                    [
                        {
                            "portfolio_drawdown": float(
                                row["portfolio_drawdown"]
                            ),
                            "early_ret_soxl": float(row["early_ret_soxl"]),
                            "early_dislocation": float(
                                row["early_dislocation"]
                            ),
                            "tail_probability": float(row["tail_probability"]),
                            "candidate_beta_delta": (
                                candidate_target - baseline_target
                            ),
                            "candidate_current_turnover": (
                                candidate_current_turnover
                            ),
                        }
                    ]
                )
                predicted = float(
                    model_payload["model"].predict(
                        feature.loc[:, list(ROUTER_FEATURES)]
                    )[0]
                )
                lower_bound = (
                    predicted
                    + float(model_payload["lower_residual_offset_bps"])
                )
                override = bool(lower_bound > 0.0)
                reason = (
                    "positive_lower_confidence_bound"
                    if override
                    else "lower_confidence_bound_non_positive"
                )
        action = baseline_action or override
        target_beta = candidate_target if override else baseline_target
        if action:
            close_weights = project_legal_weights(
                row["raw_soxl_weight"],
                row["raw_soxx_weight"],
                target_beta,
                minimum_cash_weight=source_spec.minimum_cash_weight,
                maximum_soxl_weight=source_spec.maximum_soxl_weight,
            )
            gross_return = float(
                row["value_at_cut"]
                * (1.0 + close_weights @ row["post_returns"])
                - 1.0
            )
            turnover = float(
                np.abs(row["open_weights"][:2] - previous[:2]).sum()
                + np.abs(
                    close_weights[:2] - row["drifted_weights"][:2]
                ).sum()
            )
            ending = mark_to_market_weights(
                close_weights, row["post_returns"]
            )
        else:
            close_weights = row["open_weights"]
            asset_returns = np.array(
                [row["soxl_return"], row["soxx_return"], 0.0],
                dtype="float64",
            )
            gross_return = float(close_weights @ asset_returns)
            turnover = float(
                np.abs(close_weights[:2] - previous[:2]).sum()
            )
            ending = mark_to_market_weights(close_weights, asset_returns)
        strategy_return = gross_return - turnover * cost_bps / 10_000.0
        rows.append(
            {
                "signal_date": row["signal_date"],
                "trade_date": row["trade_date"],
                "exit_date": row["exit_date"],
                "segment": row["segment"],
                "decision_key": f"{row['signal_date'].date().isoformat()}:recovery",
                "controller_state_source": "shadow_stage1_ledger",
                "soxl_return": row["soxl_return"],
                "soxx_return": row["soxx_return"],
                "open_soxl_weight": row["open_weights"][0],
                "open_soxx_weight": row["open_weights"][1],
                "open_cash_weight": row["open_weights"][2],
                "soxl_weight": close_weights[0],
                "soxx_weight": close_weights[1],
                "cash_weight": close_weights[2],
                "predicted_net_advantage_bps": predicted,
                "advantage_lower_bound_bps": lower_bound,
                "decision_reason": reason,
                "recovery_action": action,
                "baseline_recovery_action": baseline_action,
                "model_override_action": override,
                "current_beta_cap": row["current_beta_cap"],
                "recovery_beta_target": target_beta,
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": cost_bps,
            }
        )
        previous = ending
    return pd.DataFrame(rows)


def _run_card(spec: Stage1TransitionAwareActionValueV2Spec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_contract = _source_contract(spec)
    source_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(source_spec)
    teacher_frame = _build_teacher_training_frame(
        frame, spec=source_spec, teacher_beta=source_spec.teacher_recovery_beta
    )
    development = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    cache: dict[tuple[int, float], Any] = {}
    baseline_dev = _simulate_policy(
        development,
        spec=source_spec,
        teacher_frame=teacher_frame,
        c_value=0.03,
        threshold=0.55,
        action_beta=0.60,
        first_action_year=2020,
        model_cache=cache,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_outer = _simulate_policy(
        outer,
        spec=source_spec,
        teacher_frame=teacher_frame,
        c_value=0.03,
        threshold=0.55,
        action_beta=0.60,
        first_action_year=min(spec.outer_years),
        model_cache=cache,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_full = pd.concat([baseline_dev, baseline_outer], ignore_index=True)
    contexts = _baseline_contexts(
        frame,
        source_spec=source_spec,
        cost_bps=spec.transaction_cost_bps,
        baseline_replay=baseline_full,
    )
    distillation_spec = Stage1ControllerAwareDistillationSpec(root=spec.root)
    distillation_teacher = _build_distillation_targets(
        frame,
        teacher_frame[
            teacher_frame["year"].between(
                spec.teacher_start_year, spec.teacher_end_year
            )
        ],
        source_spec=source_spec,
        spec=distillation_spec,
    )
    distillation_dataset, _ = _build_distillation_dataset(
        distillation_teacher, distillation_spec
    )
    distillation_models, _ = _fit_distillation_models(
        distillation_dataset, DISTILLATION_CANDIDATES, distillation_spec
    )
    candidate_targets = _distillation_candidate_targets(
        contexts, models=distillation_models
    )
    training = _action_value_frame(
        contexts[contexts["segment"].eq("development_walk_forward")],
        baseline_replay=baseline_dev,
        candidate_targets=candidate_targets,
        source_spec=source_spec,
        spec=spec,
    )
    dataset, dataset_audit = _build_qlib_dataset(training, spec)
    models, diagnostics, oof = _fit_models(dataset, spec)
    challenger_dev = _simulate_router(
        contexts[contexts["segment"].eq("development_walk_forward")],
        baseline_replay=baseline_dev,
        candidate_targets=candidate_targets,
        models=models,
        source_spec=source_spec,
        spec=spec,
        cost_bps=spec.transaction_cost_bps,
    )
    challenger_outer = _simulate_router(
        contexts[contexts["segment"].eq("outer_diagnostic")],
        baseline_replay=baseline_outer,
        candidate_targets=candidate_targets,
        models=models,
        source_spec=source_spec,
        spec=spec,
        cost_bps=spec.transaction_cost_bps,
    )
    fallback_dev = _simulate_router(
        contexts[contexts["segment"].eq("development_walk_forward")],
        baseline_replay=baseline_dev,
        candidate_targets=candidate_targets,
        models={},
        source_spec=source_spec,
        spec=spec,
        cost_bps=spec.transaction_cost_bps,
    )
    comparison = pd.DataFrame(
        [
            _candidate_row(
                variant="stage1_baseline",
                candidate=None,
                replay=replay,
                baseline=replay,
                segment=segment,
                spec=spec,
            )
            for segment, replay in (
                ("development_walk_forward", baseline_dev),
                ("outer_diagnostic", baseline_outer),
            )
        ]
        + [
            _candidate_row(
                variant="net_advantage_lcb_router",
                candidate=None,
                replay=replay,
                baseline=baseline,
                segment=segment,
                spec=spec,
            )
            for segment, replay, baseline in (
                ("development_walk_forward", challenger_dev, baseline_dev),
                ("outer_diagnostic", challenger_outer, baseline_outer),
            )
        ]
    )
    comparison["selected"] = False
    selected_dev = comparison[
        comparison["variant"].eq("net_advantage_lcb_router")
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    selected_outer = comparison[
        comparison["variant"].eq("net_advantage_lcb_router")
        & comparison["segment"].eq("outer_diagnostic")
    ].iloc[0]
    development_overrides = int(challenger_dev["model_override_action"].sum())
    outer_overrides = int(challenger_outer["model_override_action"].sum())
    gates = {
        "train_serve_feature_contract_pass": bool(
            "candidate_current_turnover" in ROUTER_FEATURES
            and "candidate_next_reset_turnover" not in ROUTER_FEATURES
        ),
        "baseline_fallback_identity_gate_pass": bool(
            np.allclose(
                fallback_dev["strategy_return"].to_numpy(),
                baseline_dev["strategy_return"].to_numpy(),
                atol=1e-12,
            )
            and np.allclose(
                fallback_dev["turnover"].to_numpy(),
                baseline_dev["turnover"].to_numpy(),
                atol=1e-12,
            )
        ),
        "unique_daily_decision_key_gate_pass": bool(
            not challenger_dev["decision_key"].duplicated().any()
            and not challenger_outer["decision_key"].duplicated().any()
        ),
        "development_override_support_gate_pass": development_overrides > 0,
        "development_net_return_gate_pass": bool(
            selected_dev["return_delta_vs_stage1_pct_points"] > 0.0
        ),
        "stress_risk_gate_pass": bool(
            selected_dev["stress_25bps_max_drawdown_pct"]
            >= comparison[
                comparison["variant"].eq("stage1_baseline")
                & comparison["segment"].eq("development_walk_forward")
            ].iloc[0]["stress_25bps_max_drawdown_pct"]
        ),
        "outer_used_for_selection": False,
        "clean_unseen_holdout_available": False,
    }
    if not gates["train_serve_feature_contract_pass"]:
        decision = "blocked_train_serve_feature_contract_failed"
    elif development_overrides == 0:
        decision = "not_promoted_no_positive_lower_bound_overrides"
    elif not all(
        gates[key]
        for key in (
            "baseline_fallback_identity_gate_pass",
            "unique_daily_decision_key_gate_pass",
            "development_net_return_gate_pass",
            "stress_risk_gate_pass",
        )
    ):
        decision = "not_promoted_development_gates_failed"
    else:
        decision = "candidate_for_clean_forward_validation_not_promoted"
    annual = pd.concat(
        [
            _annual_metrics(
                replay, variant=variant, cost_bps=spec.transaction_cost_bps
            )
            for variant, replay in (
                ("stage1_baseline", baseline_dev),
                ("net_advantage_lcb_router", challenger_dev),
                ("stage1_baseline", baseline_outer),
                ("net_advantage_lcb_router", challenger_outer),
            )
        ],
        ignore_index=True,
    )
    return _record_and_write(
        spec=spec,
        source_contract=source_contract,
        source_audit=source_audit,
        dataset_audit=dataset_audit,
        training=training,
        diagnostics=diagnostics,
        oof=oof,
        models=models,
        comparison=comparison,
        annual=annual,
        baseline_dev=baseline_dev,
        baseline_outer=baseline_outer,
        challenger_dev=challenger_dev,
        challenger_outer=challenger_outer,
        selected_dev=selected_dev,
        selected_outer=selected_outer,
        gates=gates,
        decision=decision,
        development_overrides=development_overrides,
        outer_overrides=outer_overrides,
    )


def _record_and_write(**payload) -> dict[str, Any]:
    from qlib.contrib.evaluate import risk_analysis
    from qlib.workflow import R

    spec = payload["spec"]
    risk_rows = []
    for segment, replay in (
        ("development_walk_forward", payload["challenger_dev"]),
        ("outer_diagnostic", payload["challenger_outer"]),
    ):
        for cost_bps in (spec.transaction_cost_bps, spec.stress_cost_bps):
            returns = (
                replay["gross_return"]
                - replay["turnover"] * cost_bps / 10_000.0
            )
            table = risk_analysis(
                pd.Series(
                    returns.to_numpy(),
                    index=pd.to_datetime(replay["trade_date"]),
                ),
                freq="day",
            ).reset_index()
            table["segment"] = segment
            table["cost_bps"] = cost_bps
            risk_rows.append(table)
    qlib_risk = pd.concat(risk_rows, ignore_index=True)
    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temporary:
        temp = Path(temporary)
        payload["training"].to_csv(
            temp / "action_value_training_frame.csv", index=False
        )
        payload["oof"].to_csv(temp / "oof_predictions.csv", index=False)
        pd.concat(
            [payload["challenger_dev"], payload["challenger_outer"]],
            ignore_index=True,
        ).to_csv(temp / "portfolio_report.csv", index=False)
        model_path = temp / "net_advantage_lcb_router.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(payload["models"], handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_stage1_recorder=spec.source_recorder_id,
                model="ridge_continuous_net_advantage_lcb",
                objective="net_advantage_bps_after_25bps_transition_cost",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_override_count=payload["development_overrides"],
                outer_override_count=payload["outer_overrides"],
                development_return_delta_pct_points=float(
                    payload["selected_dev"][
                        "return_delta_vs_stage1_pct_points"
                    ]
                ),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(
                    artifact.as_posix(), artifact_path=spec.experiment
                )
            recorder.log_artifact(
                model_path.as_posix(),
                artifact_path=f"{spec.experiment}/models",
            )
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            summary = _write_card(
                **payload,
                qlib_risk=qlib_risk,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
            )
            for filename in summary["sidecars"].values():
                path = spec.card_dir / filename
                if path.exists():
                    recorder.log_artifact(
                        path.as_posix(), artifact_path=spec.experiment
                    )
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return summary


def _json_row(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[key] = None if pd.isna(value) else float(value)
        else:
            result[key] = value
    return result


def _write_card(**payload) -> dict[str, Any]:
    spec = payload["spec"]
    card = spec.card_dir
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "annual_metrics": "annual_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "training_summary": "training_summary.csv",
        "calibration_diagnostics": "calibration_diagnostics.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "conclusion": "conclusion.md",
        "workflow_config": "workflow.yaml",
    }
    baseline_dev = payload["comparison"][
        payload["comparison"]["variant"].eq("stage1_baseline")
        & payload["comparison"]["segment"].eq("development_walk_forward")
    ].iloc[0]
    baseline_outer = payload["comparison"][
        payload["comparison"]["variant"].eq("stage1_baseline")
        & payload["comparison"]["segment"].eq("outer_diagnostic")
    ].iloc[0]
    summary = {
        "experiment": spec.experiment,
        "card_type": "stage1_transition_aware_action_value_research_card",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "system_of_record": "qlib",
        "provider_uri": payload["source_audit"]["source_market_provider_uri"],
        "intraday_provider_uri": payload["source_audit"]["intraday_provider_uri"],
        "as_of": payload["source_audit"]["intraday_provider_as_of"],
        "source_stage1": payload["source_contract"],
        "source_v1_status": "superseded_for_research_due_to_train_serve_turnover_skew",
        "hypothesis": (
            "A continuous net-advantage model should abstain unless its "
            "walk-forward residual lower bound remains positive."
        ),
        "method": {
            "model": "fixed ridge regression on continuous net advantage",
            "objective": "net advantage in bps after 25bps current and next-reset transition cost",
            "decision": "apply the full pre-existing candidate only when the one-sided residual lower bound is positive",
            "lower_residual_quantile": spec.lower_residual_quantile,
            "ridge_alpha": spec.ridge_alpha,
            "minimum_operational_rows": spec.minimum_operational_rows,
            "minimum_calibration_rows": spec.minimum_calibration_rows,
            "controller_state_source": "shadow_stage1_ledger_explicit_dual_ledger_contract",
            "production_stage1_changed": False,
        },
        "dataset_audit": payload["dataset_audit"],
        "selected_model": {
            "name": "net_advantage_lcb_router",
            "architecture": "ridge_continuous_net_advantage_plus_walk_forward_lower_bound",
            "candidate_count": 1,
            "feature_count": len(ROUTER_FEATURES),
            "features": list(ROUTER_FEATURES),
            "label": LABEL,
            "selection_uses_outer_test": False,
        },
        "development_baseline_metrics_10bps": _json_row(baseline_dev),
        "development_candidate_metrics_10bps": _json_row(
            payload["selected_dev"]
        ),
        "outer_diagnostic_baseline_metrics_10bps": _json_row(baseline_outer),
        "outer_diagnostic_candidate_metrics_10bps": _json_row(
            payload["selected_outer"]
        ),
        "incremental_action_audit": {
            "development_override_count": payload["development_overrides"],
            "outer_override_count": payload["outer_overrides"],
            "duplicate_decision_key_count": int(
                payload["challenger_dev"]["decision_key"].duplicated().sum()
                + payload["challenger_outer"]["decision_key"].duplicated().sum()
            ),
        },
        "promotion_gates": payload["gates"],
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_clean_unseen_holdout",
        "decision": payload["decision"],
        "promotion": "blocked",
        "qlib_api": [
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R",
            "qlib.contrib.evaluate.risk_analysis",
        ],
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "https://qlib.readthedocs.io/en/latest/component/workflow.html",
        ],
        "recorder_id": payload["recorder_id"],
        "recorder_uri": payload["recorder_uri"].resolve().as_posix(),
        "recorder_local_dir": payload["recorder_local_dir"].as_posix(),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/action_value_training_frame.csv",
                f"artifacts/{spec.experiment}/oof_predictions.csv",
                f"artifacts/{spec.experiment}/portfolio_report.csv",
                f"artifacts/{spec.experiment}/models/net_advantage_lcb_router.pkl",
            ],
        },
    }
    (card / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    payload["comparison"].to_csv(card / "model_selection.csv", index=False)
    payload["comparison"].to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    payload["annual"].to_csv(card / "annual_metrics.csv", index=False)
    payload["qlib_risk"].to_csv(card / "risk_analysis.csv", index=False)
    payload["diagnostics"].to_csv(
        card / "training_summary.csv", index=False
    )
    payload["diagnostics"].to_csv(
        card / "calibration_diagnostics.csv", index=False
    )
    (card / "selected_model.json").write_text(
        json.dumps(summary["selected_model"], indent=2), encoding="utf-8"
    )
    analysis = {
        "feature_contract_repaired": summary["promotion_gates"][
            "train_serve_feature_contract_pass"
        ],
        "development_override_count": payload["development_overrides"],
        "outer_override_count": payload["outer_overrides"],
        "development_return_delta_pct_points": float(
            payload["selected_dev"]["return_delta_vs_stage1_pct_points"]
        ),
        "decision": payload["decision"],
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    failures = ["clean_unseen_holdout_unavailable"]
    if payload["development_overrides"] == 0:
        failures.append("no_positive_lower_confidence_bound_overrides")
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + "\n".join(f"- {item}" for item in failures)
        + "\n",
        encoding="utf-8",
    )
    conclusion = (
        "# 中文结论\n\n"
        "- 修复：训练和推理都只使用 10:31 当时可见的 `candidate_current_turnover`；下一交易日复位换手仅用于连续净优势标签。\n"
        f"- 开发期 override={payload['development_overrides']}，outer 复用诊断 override={payload['outer_overrides']}。\n"
        f"- 开发期相对 Stage1 收益增量={float(payload['selected_dev']['return_delta_vs_stage1_pct_points']):.4f} 个百分点。\n"
        "- 解释：固定 Ridge 加 walk-forward 残差下界没有找到可稳健覆盖 Stage1 的动作；这不是收益优化成功，而是正确的 abstain。\n"
        f"- 决策：`{payload['decision']}`；生产 Stage1 和模拟盘均不切换，promotion blocked。\n"
        "- outer 已被反复查看，不参与选择，也不是 clean holdout。\n"
        "- 下一步：冻结本合同后积累新的 forward paper 样本；不能再通过调低置信门槛制造历史动作。\n"
        f"- qlib recorder：`{payload['recorder_id']}`。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = (
        f"experiment: {spec.experiment}\n"
        f"provider_uri: {summary['provider_uri']}\n"
        f"intraday_provider_uri: {summary['intraday_provider_uri']}\n"
        f"recorder_uri: {summary['recorder_uri']}\n"
        "source_stage1_frozen: true\n"
        "stage2_enabled: false\n"
        "selection_uses_outer_test: false\n"
        "feature_contract:\n"
        "  decision_time: '10:31 America/New_York'\n"
        "  current_turnover_feature: candidate_current_turnover\n"
        "  future_reset_cost_feature: false\n"
        "  future_reset_cost_label: true\n"
        "model_selection:\n"
        "  candidate_count: 1\n"
        "  model: fixed_ridge_continuous_net_advantage_lcb\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
        "  workflow_config: workflow.yaml\n"
        "  recorder_only_artifacts:\n"
        + "\n".join(
            f"    - {path}"
            for path in summary["artifact_policy"]["recorder_only_artifacts"]
        )
        + "\n"
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    return summary
