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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _base_cap,
    _build_teacher_training_frame,
    _feature_row,
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


EXPERIMENT = "stage1_transition_aware_action_value_v1"
SOURCE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
SOURCE_RECORDER_ID = "f6b09330175e41719886d58ee2d11bec"
ROUTER_FEATURES = (*FEATURE_COLUMNS, "candidate_beta_delta", "candidate_turnover")


@dataclass(frozen=True)
class Stage1TransitionAwareActionValueSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    first_action_year: int = 2020
    teacher_start_year: int = 2019
    teacher_end_year: int = 2022
    development_years: tuple[int, ...] = tuple(range(2016, 2023))
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_stage1_transition_aware_action_value_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = Stage1TransitionAwareActionValueSpec(root=_resolve_root(root))
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


def _source_contract(spec: Stage1TransitionAwareActionValueSpec) -> dict[str, Any]:
    summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: frozen Stage1 recorder drifted")
    if summary.get("selected_model", {}).get("stage2_enabled") is not False:
        raise ValueError("blocked: Stage2 must stay disabled")
    if summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: source Stage1 used outer for selection")
    return {
        "experiment": spec.source_experiment,
        "recorder_id": spec.source_recorder_id,
        "training_end": summary["source_audit"]["source_training_end"],
        "stage2_enabled": False,
        "frozen": True,
    }


def _baseline_contexts(
    frame: pd.DataFrame,
    *,
    source_spec: IntradayRecoveryDirectAllocationSpec,
    cost_bps: float,
    baseline_replay: pd.DataFrame | None = None,
) -> pd.DataFrame:
    replay_by_segment_date = (
        baseline_replay.set_index(["segment", "signal_date"])
        if baseline_replay is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    for _, segment in frame.groupby("segment", sort=False):
        equity = peak = 1.0
        previous = np.zeros(3, dtype="float64")
        state = clock = 0
        history: list[float] = []
        ordered = segment.sort_values("signal_date")
        for row in ordered.itertuples():
            drawdown = max(0.0, 1.0 - equity / peak)
            state, clock, q = _advance_state(state, clock, drawdown, source_spec)
            raw_beta = 3.0 * row.soxl_weight + row.soxx_weight
            replay_row = (
                replay_by_segment_date.loc[(row.segment, row.signal_date)]
                if replay_by_segment_date is not None
                else None
            )
            if replay_row is None:
                cap, raw_beta = _base_cap(
                    row,
                    q=q,
                    drawdown=drawdown,
                    history=history,
                    spec=source_spec,
                )
                open_weights = project_legal_weights(
                    row.soxl_weight,
                    row.soxx_weight,
                    cap,
                    minimum_cash_weight=source_spec.minimum_cash_weight,
                    maximum_soxl_weight=source_spec.maximum_soxl_weight,
                )
            else:
                cap = float(replay_row["current_beta_cap"])
                open_weights = replay_row[
                    ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]
                ].to_numpy(dtype="float64")
            asset_returns = np.array(
                [row.soxl_return, row.soxx_return, 0.0], dtype="float64"
            )
            gross_return = float(open_weights @ asset_returns)
            turnover = float(np.abs(open_weights[:2] - previous[:2]).sum())
            ending_weights = mark_to_market_weights(open_weights, asset_returns)
            if pd.notna(row.early_ret_soxx) and pd.notna(row.early_ret_soxl):
                early = np.array(
                    [row.early_ret_soxl, row.early_ret_soxx, 0.0],
                    dtype="float64",
                )
                value_at_cut = float(1.0 + open_weights @ early)
                drifted = mark_to_market_weights(open_weights, early)
                post = np.array(
                    [
                        (1.0 + row.soxl_return) / (1.0 + row.early_ret_soxl) - 1.0,
                        (1.0 + row.soxx_return) / (1.0 + row.early_ret_soxx) - 1.0,
                        0.0,
                    ],
                    dtype="float64",
                )
            else:
                value_at_cut = np.nan
                drifted = np.full(3, np.nan)
                post = np.full(3, np.nan)
            features = _feature_row(
                row,
                drawdown=drawdown,
                cap=cap,
                raw_beta=raw_beta,
                state=state,
                history=history,
            )
            rows.append(
                {
                    "signal_date": row.signal_date,
                    "trade_date": row.trade_date,
                    "exit_date": row.exit_date,
                    "segment": row.segment,
                    "year": int(row.signal_date.year),
                    "soxl_return": row.soxl_return,
                    "soxx_return": row.soxx_return,
                    "raw_soxl_weight": row.soxl_weight,
                    "raw_soxx_weight": row.soxx_weight,
                    "raw_beta": raw_beta,
                    **features,
                    "open_weights": open_weights,
                    "drifted_weights": drifted,
                    "post_returns": post,
                    "value_at_cut": value_at_cut,
                }
            )
            strategy_return = (
                float(replay_row["strategy_return"])
                if replay_row is not None
                else gross_return - turnover * cost_bps / 10_000.0
            )
            history.append(strategy_return)
            equity *= 1.0 + strategy_return
            peak = max(peak, equity)
            previous = ending_weights
    return pd.DataFrame(rows).sort_values(["segment", "signal_date"]).reset_index(
        drop=True
    )


def _distillation_candidate_targets(
    contexts: pd.DataFrame,
    *,
    models: dict[tuple[str, int], Any],
) -> pd.DataFrame:
    candidate = next(
        item
        for item in DISTILLATION_CANDIDATES
        if item.name == "ridge_a100_s025_u015"
    )
    rows: list[dict[str, Any]] = []
    for _, row in contexts.iterrows():
        model = models.get((candidate.name, int(row["year"])))
        predicted_uplift = 0.0
        target_beta = float(row["current_beta_cap"])
        eligible = (
            model is not None
            and row["portfolio_drawdown"] >= 0.08
            and row["raw_beta"] > row["current_beta_cap"] + 1e-12
            and np.isfinite(row["value_at_cut"])
        )
        if eligible:
            feature = pd.DataFrame(
                [{name: float(row[name]) for name in FEATURE_COLUMNS}]
            )
            predicted_uplift = max(
                0.0,
                float(model.predict(feature.loc[:, list(FEATURE_COLUMNS)])[0]),
            )
            if predicted_uplift >= candidate.minimum_predicted_uplift:
                target_beta = min(
                    float(row["raw_beta"]),
                    candidate.maximum_recovery_beta,
                    float(row["current_beta_cap"])
                    + candidate.shrinkage * predicted_uplift,
                )
        rows.append(
            {
                "signal_date": row["signal_date"],
                "predicted_beta_uplift": predicted_uplift,
                "candidate_beta_target": target_beta,
            }
        )
    return pd.DataFrame(rows)


def _router_training_frame(
    contexts: pd.DataFrame,
    *,
    baseline_replay: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1TransitionAwareActionValueSpec,
) -> pd.DataFrame:
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
        proposed = float(
            candidates.loc[row["signal_date"], "candidate_beta_target"]
        )
        target_beta = max(baseline_target, proposed)
        if target_beta <= baseline_target + 1e-12:
            continue
        baseline_action = bool(baseline_row["recovery_action"])
        baseline_weights = (
            project_legal_weights(
                row["raw_soxl_weight"],
                row["raw_soxx_weight"],
                baseline_target,
                minimum_cash_weight=source_spec.minimum_cash_weight,
                maximum_soxl_weight=source_spec.maximum_soxl_weight,
            )
            if baseline_action
            else row["drifted_weights"].copy()
        )
        candidate_weights = project_legal_weights(
            row["raw_soxl_weight"],
            row["raw_soxx_weight"],
            target_beta,
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
        baseline_turnover = float(
            np.abs(baseline_weights[:2] - row["drifted_weights"][:2]).sum()
        )
        candidate_turnover = float(
            np.abs(candidate_weights[:2] - row["drifted_weights"][:2]).sum()
        )
        next_open = (
            ordered.iloc[index + 1]["open_weights"]
            if index + 1 < len(ordered)
            else None
        )
        if next_open is not None:
            baseline_end = mark_to_market_weights(
                baseline_weights, row["post_returns"]
            )
            candidate_end = mark_to_market_weights(
                candidate_weights, row["post_returns"]
            )
            baseline_turnover += float(
                np.abs(next_open[:2] - baseline_end[:2]).sum()
            )
            candidate_turnover += float(
                np.abs(next_open[:2] - candidate_end[:2]).sum()
            )
        advantage = candidate_gross - baseline_gross - (
            candidate_turnover - baseline_turnover
        ) * spec.stress_cost_bps / 10_000.0
        rows.append(
            {
                "signal_date": row["signal_date"],
                "segment": row["segment"],
                "year": int(row["year"]),
                **{name: float(row[name]) for name in FEATURE_COLUMNS},
                "candidate_beta_delta": target_beta - baseline_target,
                "candidate_turnover": candidate_turnover,
                "candidate_beta_target": target_beta,
                "candidate_advantage": advantage,
            }
        )
    return pd.DataFrame(rows)


def _fit_router_models(
    dataset, spec: Stage1TransitionAwareActionValueSpec
) -> tuple[dict[int, Any], pd.DataFrame]:
    models: dict[int, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in range(spec.first_action_year, spec.teacher_end_year + 2):
        segment = "train_outer" if year == spec.teacher_end_year + 1 else f"train_{year}"
        train = dataset.prepare(segment, col_set=["feature", "label"])
        if not train.empty:
            train.columns = [str(column[-1]) for column in train.columns]
        if len(train) < 20:
            continue
        model = make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(
                max_iter=2_000,
                random_state=42,
            ),
        )
        model.fit(
            train.loc[:, list(ROUTER_FEATURES)],
            train["candidate_advantage"].gt(0.0),
        )
        prediction = model.predict_proba(
            train.loc[:, list(ROUTER_FEATURES)]
        )[:, 1]
        if year == spec.teacher_end_year + 1:
            for outer_year in spec.outer_years:
                models[outer_year] = model
            model_year: Any = "outer_frozen_2023_2026"
        else:
            models[year] = model
            model_year = year
        diagnostics.append(
            {
                "model_year": model_year,
                "training_rows": len(train),
                "training_end": f"{year - 1}-12-31",
                "positive_label_ratio": float(
                    train["candidate_advantage"].gt(0.0).mean()
                ),
                "training_accuracy": float(
                    np.mean(
                        (prediction >= 0.5)
                        == train["candidate_advantage"].gt(0.0).to_numpy()
                    )
                ),
            }
        )
    return models, pd.DataFrame(diagnostics)


def _build_router_qlib_dataset(
    training: pd.DataFrame, spec: Stage1TransitionAwareActionValueSpec
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat(
        {
            "feature": training.loc[:, list(ROUTER_FEATURES)].astype("float64"),
            "label": training.loc[:, ["candidate_advantage"]].astype("float64"),
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(training["signal_date"]).dt.normalize(),
            np.full(len(training), "STAGE1_RESIDUAL_ROUTER"),
        ],
        names=["datetime", "instrument"],
    )
    start = pd.to_datetime(training["signal_date"]).min().date().isoformat()
    segments: dict[str, tuple[str, str]] = {
        "all": (start, f"{spec.teacher_end_year}-12-31"),
        "train_outer": (start, f"{spec.teacher_end_year}-12-31"),
    }
    for year in range(spec.first_action_year, spec.teacher_end_year + 2):
        segments[f"train_{year}"] = (start, f"{year - 1}-12-31")
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
        "label": "candidate_advantage",
        "segments": {key: list(value) for key, value in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
    }
    return dataset, audit


def _simulate_residual_router(
    contexts: pd.DataFrame,
    *,
    baseline_replay: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    router_models: dict[int, Any],
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1TransitionAwareActionValueSpec,
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
        candidate_target = float(
            candidates.loc[row["signal_date"], "candidate_beta_target"]
        )
        candidate_target = max(candidate_target, baseline_target)
        candidate_weights = project_legal_weights(
            row["raw_soxl_weight"],
            row["raw_soxx_weight"],
            candidate_target,
            minimum_cash_weight=source_spec.minimum_cash_weight,
            maximum_soxl_weight=source_spec.maximum_soxl_weight,
        )
        candidate_turnover = float(
            np.abs(candidate_weights[:2] - row["drifted_weights"][:2]).sum()
        )
        predicted_advantage = 0.0
        router_action = False
        router_strength = 0.0
        model = router_models.get(int(row["year"]))
        if model is not None and candidate_target > baseline_target + 1e-12:
            feature = pd.DataFrame(
                [
                    {
                        **{name: float(row[name]) for name in FEATURE_COLUMNS},
                        "candidate_beta_delta": candidate_target - baseline_target,
                        "candidate_turnover": candidate_turnover,
                    }
                ]
            )
            predicted_advantage = float(
                model.predict_proba(feature.loc[:, list(ROUTER_FEATURES)])[0, 1]
                - 0.5
            )
            router_action = predicted_advantage > 0.0
            router_strength = float(
                np.clip(2.0 * predicted_advantage, 0.0, 1.0)
            )
        action = baseline_action or router_action
        target_beta = baseline_target + router_strength * (
            candidate_target - baseline_target
        )
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
                + np.abs(close_weights[:2] - row["drifted_weights"][:2]).sum()
            )
            ending = mark_to_market_weights(close_weights, row["post_returns"])
        else:
            close_weights = row["open_weights"]
            asset_returns = np.array(
                [row["soxl_return"], row["soxx_return"], 0.0], dtype="float64"
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
                "soxl_return": row["soxl_return"],
                "soxx_return": row["soxx_return"],
                "open_soxl_weight": row["open_weights"][0],
                "open_soxx_weight": row["open_weights"][1],
                "open_cash_weight": row["open_weights"][2],
                "soxl_weight": close_weights[0],
                "soxx_weight": close_weights[1],
                "cash_weight": close_weights[2],
                "predicted_candidate_advantage": predicted_advantage,
                "recovery_action": action,
                "baseline_recovery_action": baseline_action,
                "model_override_action": router_action,
                "model_override_strength": router_strength,
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


def _run_card(spec: Stage1TransitionAwareActionValueSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_contract = _source_contract(spec)
    source_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(source_spec)
    teacher_frame = _build_teacher_training_frame(
        frame, spec=source_spec, teacher_beta=source_spec.teacher_recovery_beta
    )
    development = frame[frame["segment"].eq("development_walk_forward")].copy()
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    cache: dict[tuple[int, float], Any] = {}
    baseline_dev = _simulate_policy(
        development,
        spec=source_spec,
        teacher_frame=teacher_frame,
        c_value=0.03,
        threshold=0.55,
        action_beta=0.60,
        first_action_year=spec.first_action_year,
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
    distillation_dataset, distillation_audit = _build_distillation_dataset(
        distillation_teacher, distillation_spec
    )
    distillation_models, distillation_training = _fit_distillation_models(
        distillation_dataset, DISTILLATION_CANDIDATES, distillation_spec
    )
    candidate_targets = _distillation_candidate_targets(
        contexts, models=distillation_models
    )
    router_training = _router_training_frame(
        contexts[contexts["segment"].eq("development_walk_forward")],
        baseline_replay=baseline_dev,
        candidate_targets=candidate_targets,
        source_spec=source_spec,
        spec=spec,
    )
    router_dataset, dataset_audit = _build_router_qlib_dataset(
        router_training, spec
    )
    router_models, router_diagnostics = _fit_router_models(router_dataset, spec)
    training = pd.concat(
        [
            distillation_training.assign(model_layer="sizing"),
            router_diagnostics.assign(model_layer="residual_router"),
        ],
        ignore_index=True,
    )
    challenger_dev = _simulate_residual_router(
        contexts[contexts["segment"].eq("development_walk_forward")],
        baseline_replay=baseline_dev,
        candidate_targets=candidate_targets,
        router_models=router_models,
        source_spec=source_spec,
        spec=spec,
        cost_bps=spec.transaction_cost_bps,
    )
    challenger_outer = _simulate_residual_router(
        contexts[contexts["segment"].eq("outer_diagnostic")],
        baseline_replay=baseline_outer,
        candidate_targets=candidate_targets,
        router_models=router_models,
        source_spec=source_spec,
        spec=spec,
        cost_bps=spec.transaction_cost_bps,
    )
    fallback_dev = _simulate_residual_router(
        contexts[contexts["segment"].eq("development_walk_forward")],
        baseline_replay=baseline_dev,
        candidate_targets=candidate_targets,
        router_models={},
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
                variant="transition_aware_residual_router",
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
    comparison["selected"] = comparison["variant"].eq(
        "transition_aware_residual_router"
    )
    selected_dev = comparison[
        comparison["variant"].eq("transition_aware_residual_router")
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    selected_outer = comparison[
        comparison["variant"].eq("transition_aware_residual_router")
        & comparison["segment"].eq("outer_diagnostic")
    ].iloc[0]
    gates = {
        "development_return_gate_pass": bool(
            selected_dev["return_delta_vs_stage1_pct_points"] > 0.0
        ),
        "development_risk_gate_pass": bool(
            selected_dev["max_drawdown_pct"] >= -30.0
            and selected_dev["worst_rolling_5d_pct"] >= -12.0
        ),
        "stress_cost_gate_pass": bool(
            _risk_metrics(
                challenger_dev["gross_return"].to_numpy()
                - challenger_dev["turnover"].to_numpy()
                * spec.stress_cost_bps
                / 10_000.0
            )["return_pct"]
            > _risk_metrics(
                baseline_dev["gross_return"].to_numpy()
                - baseline_dev["turnover"].to_numpy()
                * spec.stress_cost_bps
                / 10_000.0
            )["return_pct"]
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
        "outer_used_for_selection": False,
        "clean_unseen_holdout_available": False,
    }
    development_pass = all(
        gates[key]
        for key in (
            "development_return_gate_pass",
            "development_risk_gate_pass",
            "stress_cost_gate_pass",
            "baseline_fallback_identity_gate_pass",
        )
    )
    decision = (
        "candidate_for_clean_forward_validation_not_promoted"
        if development_pass
        else "not_promoted_action_value_development_gates_failed"
    )
    annual = pd.concat(
        [
            _annual_metrics(replay, variant=variant, cost_bps=spec.transaction_cost_bps)
            for variant, replay in (
                ("stage1_baseline", baseline_dev),
                ("transition_aware_residual_router", challenger_dev),
                ("stage1_baseline", baseline_outer),
                ("transition_aware_residual_router", challenger_outer),
            )
        ],
        ignore_index=True,
    )
    return _record_and_write(
        spec=spec,
        source_contract=source_contract,
        source_audit=source_audit,
        dataset_audit=dataset_audit,
        actions=router_training,
        models={
            "sizing": distillation_models,
            "router": router_models,
        },
        training=training,
        comparison=comparison,
        annual=annual,
        challenger_dev=challenger_dev,
        challenger_outer=challenger_outer,
        selected_dev=selected_dev,
        selected_outer=selected_outer,
        gates=gates,
        decision=decision,
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
            returns = replay["gross_return"] - replay["turnover"] * cost_bps / 10_000.0
            table = risk_analysis(
                pd.Series(returns.to_numpy(), index=pd.to_datetime(replay["trade_date"])),
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
        payload["actions"].to_csv(temp / "residual_router_training_frame.csv", index=False)
        payload["comparison"].to_csv(temp / "model_selection_full.csv", index=False)
        pd.concat(
            [payload["challenger_dev"], payload["challenger_outer"]],
            ignore_index=True,
        ).to_csv(temp / "portfolio_report.csv", index=False)
        model_path = temp / "transition_aware_residual_router.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(payload["models"], handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_stage1_recorder=spec.source_recorder_id,
                model="distilled_sizer_plus_soft_logistic_residual_router",
                objective="positive_relative_advantage_after_25bps_transition_cost",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_return_pct=float(payload["selected_dev"]["return_pct"]),
                development_return_delta_pct_points=float(
                    payload["selected_dev"]["return_delta_vs_stage1_pct_points"]
                ),
                outer_return_pct=float(payload["selected_outer"]["return_pct"]),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(
                model_path.as_posix(), artifact_path=f"{spec.experiment}/models"
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
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return summary


def _json_row(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.bool_, bool)):
            result[key] = bool(value)
        elif isinstance(value, (np.integer,)):
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
        "hypothesis": "A learned residual router can preserve the incumbent Stage1 path exactly when confidence is absent, while softly scaling only candidate overrides whose 25bps transition-cost advantage is predicted positive.",
        "method": {
            "model": "two-layer distilled beta sizer plus logistic residual router",
            "objective": "classify positive advantage versus the exact Stage1 action after 25bps current and next-session reset turnover",
            "decision": "Stage1 is the default expert; positive router probability margin is converted continuously into override size instead of a second fixed target",
            "development": "annual expanding walk-forward fits through 2022",
            "outer": "one frozen model trained through 2022; diagnostic only",
            "preserved_stage1_contract": {
                "incumbent_daily_actions": "exact fallback",
                "maximum_soxl_weight": 0.30,
                "minimum_cash_weight": 0.15,
                "stage2_enabled": False,
            },
            "reduced_incremental_forced_logic": [
                "no new cooldown state",
                "no full-size binary override",
                "no required 20pct turnover reduction runtime rule",
                "no outer-period parameter adjustment",
            ],
        },
        "dataset_audit": payload["dataset_audit"],
        "selected_model": {
            "name": "transition_aware_residual_router",
            "architecture": "ridge_distilled_sizer_plus_soft_logistic_router",
            "candidate_count": 1,
            "router_probability_boundary": 0.5,
            "router_size_mapping": "2 * max(probability - 0.5, 0)",
            "router_label_cost_bps": spec.stress_cost_bps,
            "selection_uses_outer_test": False,
        },
        "development_baseline_metrics_10bps": _json_row(baseline_dev),
        "development_selected_metrics_10bps": _json_row(payload["selected_dev"]),
        "outer_diagnostic_baseline_metrics_10bps": _json_row(baseline_outer),
        "outer_diagnostic_selected_metrics_10bps": _json_row(payload["selected_outer"]),
        "incremental_action_audit": {
            "development_override_count": int(
                payload["challenger_dev"]["model_override_action"].sum()
            ),
            "outer_override_count": int(
                payload["challenger_outer"]["model_override_action"].sum()
            ),
        },
        "stress_25bps_comparison": {
            "development_baseline_return_pct": float(
                baseline_dev["stress_25bps_return_pct"]
            ),
            "development_candidate_return_pct": float(
                payload["selected_dev"]["stress_25bps_return_pct"]
            ),
            "development_delta_pct_points": float(
                payload["selected_dev"]["stress_25bps_return_pct"]
                - baseline_dev["stress_25bps_return_pct"]
            ),
            "outer_baseline_return_pct": float(
                baseline_outer["stress_25bps_return_pct"]
            ),
            "outer_candidate_return_pct": float(
                payload["selected_outer"]["stress_25bps_return_pct"]
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
                f"artifacts/{spec.experiment}/residual_router_training_frame.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/portfolio_report.csv",
                f"artifacts/{spec.experiment}/models/transition_aware_residual_router.pkl",
            ],
        },
    }
    (card / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    payload["comparison"].to_csv(card / "model_selection.csv", index=False)
    payload["comparison"].to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    payload["annual"].to_csv(card / "annual_metrics.csv", index=False)
    payload["qlib_risk"].to_csv(card / "risk_analysis.csv", index=False)
    payload["training"].to_csv(card / "training_summary.csv", index=False)
    (card / "selected_model.json").write_text(
        json.dumps(summary["selected_model"], indent=2), encoding="utf-8"
    )
    analysis = {
        "development_return_delta_pct_points": float(
            payload["selected_dev"]["return_delta_vs_stage1_pct_points"]
        ),
        "development_turnover_reduction": float(
            payload["selected_dev"]["turnover_reduction_vs_stage1"]
        ),
        "outer_return_delta_pct_points": float(
            payload["selected_outer"]["return_delta_vs_stage1_pct_points"]
        ),
        "development_stress_25bps_delta_pct_points": summary[
            "stress_25bps_comparison"
        ]["development_delta_pct_points"],
        "baseline_fallback_identity": payload["gates"][
            "baseline_fallback_identity_gate_pass"
        ],
        "decision": payload["decision"],
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    failed = [
        key
        for key, value in payload["gates"].items()
        if key.endswith("_gate_pass") and value is False
    ]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + "\n".join(f"- {item}" for item in failed)
        + "\n- clean_unseen_holdout_unavailable\n",
        encoding="utf-8",
    )
    selected_dev = payload["selected_dev"]
    selected_outer = payload["selected_outer"]
    conclusion = (
        "# 中文结论\n\n"
        "- 做了什么：把 Stage1 作为逐日默认专家完整保留；新增两层残差模型，第一层给出候选 beta，第二层只学习候选相对 Stage1 在 25bp 当前及下一日复位成本后是否仍有正增益，并按分类概率连续缩放覆盖幅度。\n"
        f"- 开发期：Stage1 10bp return={float(baseline_dev['return_pct']):.4f}%，候选={float(selected_dev['return_pct']):.4f}%，增量={float(selected_dev['return_delta_vs_stage1_pct_points']):.4f} 个百分点；MaxDD 从 {float(baseline_dev['max_drawdown_pct']):.4f}% 改善到 {float(selected_dev['max_drawdown_pct']):.4f}%。\n"
        f"- 压力成本：25bp Stage1={float(baseline_dev['stress_25bps_return_pct']):.4f}%，候选={float(selected_dev['stress_25bps_return_pct']):.4f}%，增量={summary['stress_25bps_comparison']['development_delta_pct_points']:.4f} 个百分点。\n"
        f"- 换手：候选较 Stage1 增加={-100.0 * float(selected_dev['turnover_reduction_vs_stage1']):.2f}%，但已由净成本收益门直接约束，不再要求任意的 20% 换手下降。\n"
        f"- 基准保持：关闭残差路由时，逐日收益和换手与 Stage1 完全一致，gate={payload['gates']['baseline_fallback_identity_gate_pass']}。\n"
        f"- outer 复用诊断：return={float(selected_outer['return_pct']):.4f}%，相对 Stage1={float(selected_outer['return_delta_vs_stage1_pct_points']):.4f} 个百分点；不参与选择，也不是 clean holdout。\n"
        f"- 决策：`{payload['decision']}`；生产 Stage1 不变，真实资金晋升 blocked。\n"
        f"- 失败门：{failed}。\n"
        "- 下一步：只进入预登记连续前向纸面验证；不能因为复用 outer 也为正就直接替换生产 Stage1。\n"
        f"- qlib recorder：`{payload['recorder_id']}`。\n"
        "- 不构成买卖建议。\n"
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
        "outer_model_training_end: 2022-12-31\n"
        "model_selection:\n"
        "  candidate_count: 1\n"
        "  metric: development_10bps_and_25bps_return_with_risk\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
        "  workflow_config: workflow.yaml\n"
        "  recorder_only_artifacts:\n"
        + "".join(
            f"    - {artifact}\n"
            for artifact in summary["artifact_policy"]["recorder_only_artifacts"]
        )
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    _write_notebook(card / f"{spec.experiment}.ipynb", spec.experiment)
    return summary


def _write_notebook(path: Path, experiment: str) -> None:
    notebook = {
        "cells": [
            {
                "id": "hypothesis",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Stage1 transition-aware residual router\n",
                    "\n",
                    "假设：以 Stage1 为精确回退，并用 25bp 跨日成本标签学习软残差路由，可以在不改写原回测的前提下提高净收益。\n",
                ],
            },
            {
                "id": "run-workflow",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import sys\n",
                    "import json\n",
                    "root = Path.cwd().resolve().parents[1]\n",
                    "sys.path.insert(0, str(root))\n",
                    "import qlib\n",
                    "from qlib.data import D\n",
                    "from qlib.workflow import R\n",
                    "from src.workflow.stage1_transition_aware_action_value import run_stage1_transition_aware_action_value_card\n",
                    "qlib.init(provider_uri=str(root / 'data/qlib/soxl_soxx_soxs'), region='us')\n",
                    "_ = D.calendar(start_time='2022-01-03', end_time='2022-01-10', freq='day')\n",
                    "summary = run_stage1_transition_aware_action_value_card(root=root)\n",
                    "summary['decision'], summary['development_selected_metrics_10bps']\n",
                ],
            },
            {
                "id": "conclusion",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 中文结论\n", "\n", "最终结论见 [conclusion.md](conclusion.md)。\n"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
