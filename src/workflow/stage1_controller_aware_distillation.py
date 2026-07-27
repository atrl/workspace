from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _base_cap,
    _build_teacher_training_frame,
    _development_stability,
    _feature_row,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    _simulate_policy,
    project_legal_weights,
)


EXPERIMENT = "stage1_controller_aware_distillation_v1"
SOURCE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
SOURCE_RECORDER_ID = "f6b09330175e41719886d58ee2d11bec"


@dataclass(frozen=True)
class DistillationCandidate:
    name: str
    ridge_alpha: float
    shrinkage: float
    minimum_predicted_uplift: float
    maximum_recovery_beta: float = 1.20


CANDIDATES = (
    DistillationCandidate("ridge_a010_s025_u015", 10.0, 0.25, 0.15),
    DistillationCandidate("ridge_a100_s025_u010", 100.0, 0.25, 0.10),
    DistillationCandidate("ridge_a100_s025_u015", 100.0, 0.25, 0.15),
    DistillationCandidate("ridge_a100_s025_u020", 100.0, 0.25, 0.20),
    DistillationCandidate("ridge_a100_s050_u015", 100.0, 0.50, 0.15),
)


@dataclass(frozen=True)
class Stage1ControllerAwareDistillationSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    development_years: tuple[int, ...] = tuple(range(2016, 2023))
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    first_action_year: int = 2020
    teacher_start_year: int = 2019
    teacher_end_year: int = 2022
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    teacher_beta_grid: tuple[float, ...] = (0.40, 0.60, 0.80, 1.00, 1.20)
    minimum_neighbor_wealth_ratio: float = 0.90
    required_turnover_reduction: float = 0.20

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_stage1_controller_aware_distillation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = Stage1ControllerAwareDistillationSpec(
        root=_resolve_root(root)
    )
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


def _source_contract(spec: Stage1ControllerAwareDistillationSpec) -> dict[str, Any]:
    summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: frozen Stage1 recorder drifted")
    selected = summary.get("selected_model", {})
    if selected.get("stage2_enabled") is not False:
        raise ValueError("blocked: controller distillation requires Stage2 off")
    if summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: source Stage1 used outer for selection")
    return {
        "experiment": spec.source_experiment,
        "recorder_id": spec.source_recorder_id,
        "training_end": summary["source_audit"]["source_training_end"],
        "stage2_enabled": False,
        "frozen": True,
    }


def _build_teacher_targets(
    frame: pd.DataFrame,
    teacher_frame: pd.DataFrame,
    *,
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1ControllerAwareDistillationSpec,
) -> pd.DataFrame:
    source = frame.set_index(["signal_date", "segment"])
    rows: list[dict[str, Any]] = []
    for row in teacher_frame.itertuples():
        market = source.loc[(row.signal_date, row.segment)]
        cap = float(row.current_beta_cap)
        raw_beta = float(row.raw_effective_beta)
        open_weights = project_legal_weights(
            market.soxl_weight,
            market.soxx_weight,
            cap,
            minimum_cash_weight=source_spec.minimum_cash_weight,
            maximum_soxl_weight=source_spec.maximum_soxl_weight,
        )
        early = np.array(
            [market.early_ret_soxl, market.early_ret_soxx, 0.0],
            dtype="float64",
        )
        value_at_cut = 1.0 + open_weights @ early
        drifted = open_weights * (1.0 + early) / value_at_cut
        post = np.array(
            [
                (1.0 + market.soxl_return)
                / (1.0 + market.early_ret_soxl)
                - 1.0,
                (1.0 + market.soxx_return)
                / (1.0 + market.early_ret_soxx)
                - 1.0,
                0.0,
            ],
            dtype="float64",
        )
        choices: list[tuple[float, float]] = []
        for proposed_beta in (cap, *spec.teacher_beta_grid):
            beta = max(cap, min(raw_beta, float(proposed_beta)))
            weights = project_legal_weights(
                market.soxl_weight,
                market.soxx_weight,
                beta,
                minimum_cash_weight=source_spec.minimum_cash_weight,
                maximum_soxl_weight=source_spec.maximum_soxl_weight,
            )
            action_return = (
                value_at_cut * (1.0 + weights @ post)
                - 1.0
                - np.abs(weights[:2] - drifted[:2]).sum()
                * spec.transaction_cost_bps
                / 10_000.0
            )
            choices.append((float(action_return), beta))
        teacher_return, teacher_beta = max(choices)
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "segment": row.segment,
                "year": int(row.year),
                **{
                    feature: float(getattr(row, feature))
                    for feature in FEATURE_COLUMNS
                },
                "teacher_target_beta": teacher_beta,
                "teacher_target_uplift": teacher_beta - cap,
                "teacher_best_return": teacher_return,
                "teacher_current_beta_cap": cap,
                "teacher_raw_effective_beta": raw_beta,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["signal_date", "segment"]
    ).reset_index(drop=True)


def _build_qlib_dataset(
    teacher: pd.DataFrame,
    spec: Stage1ControllerAwareDistillationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        "teacher_target_beta",
        "teacher_target_uplift",
        "teacher_best_return",
        "teacher_current_beta_cap",
        "teacher_raw_effective_beta",
    ]
    static = pd.concat(
        {
            "feature": teacher.loc[:, list(FEATURE_COLUMNS)],
            "label": teacher.loc[:, labels],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(teacher["signal_date"]).dt.normalize(),
            np.full(len(teacher), "STAGE1_RECOVERY"),
        ],
        names=["datetime", "instrument"],
    )
    segments: dict[str, tuple[str, str]] = {
        "all": (
            static.index.get_level_values("datetime").min().date().isoformat(),
            static.index.get_level_values("datetime").max().date().isoformat(),
        ),
        "train_outer": (
            f"{spec.teacher_start_year}-01-01",
            f"{spec.teacher_end_year}-12-31",
        ),
    }
    for year in range(spec.first_action_year, spec.teacher_end_year + 2):
        segments[f"train_{year}"] = (
            f"{spec.teacher_start_year}-01-01",
            f"{year - 1}-12-31",
        )
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
        "features": list(FEATURE_COLUMNS),
        "labels": labels,
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
    }
    return dataset, audit


def _prepare_train(dataset, segment: str) -> pd.DataFrame:
    data = dataset.prepare(segment, col_set=["feature", "label"])
    data.columns = [str(column[-1]) for column in data.columns]
    return data.reset_index(level="instrument", drop=True).sort_index()


def _fit_models(
    dataset,
    candidates: tuple[DistillationCandidate, ...],
    spec: Stage1ControllerAwareDistillationSpec,
) -> tuple[dict[tuple[str, int], Any], pd.DataFrame]:
    models: dict[tuple[str, int], Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        for year in range(spec.first_action_year, spec.teacher_end_year + 1):
            train = _prepare_train(dataset, f"train_{year}")
            if len(train) < 20:
                continue
            model = make_pipeline(
                SimpleImputer(),
                StandardScaler(),
                Ridge(alpha=candidate.ridge_alpha),
            )
            model.fit(
                train.loc[:, list(FEATURE_COLUMNS)],
                train["teacher_target_uplift"],
            )
            models[(candidate.name, year)] = model
            fitted = model.predict(train.loc[:, list(FEATURE_COLUMNS)])
            diagnostics.append(
                {
                    "candidate": candidate.name,
                    "model_year": year,
                    "training_segment": f"train_{year}",
                    "training_rows": len(train),
                    "ridge_alpha": candidate.ridge_alpha,
                    "training_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    fitted
                                    - train[
                                        "teacher_target_uplift"
                                    ].to_numpy()
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
        outer_train = _prepare_train(dataset, "train_outer")
        model = make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            Ridge(alpha=candidate.ridge_alpha),
        )
        model.fit(
            outer_train.loc[:, list(FEATURE_COLUMNS)],
            outer_train["teacher_target_uplift"],
        )
        for year in spec.outer_years:
            models[(candidate.name, year)] = model
        diagnostics.append(
            {
                "candidate": candidate.name,
                "model_year": "outer_frozen_2023_2026",
                "training_segment": "train_outer",
                "training_rows": len(outer_train),
                "ridge_alpha": candidate.ridge_alpha,
                "training_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                model.predict(
                                    outer_train.loc[
                                        :, list(FEATURE_COLUMNS)
                                    ]
                                )
                                - outer_train[
                                    "teacher_target_uplift"
                                ].to_numpy()
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
    return models, pd.DataFrame(diagnostics)


def _simulate_distilled_policy(
    frame: pd.DataFrame,
    *,
    candidate: DistillationCandidate,
    models: dict[tuple[str, int], Any],
    source_spec: IntradayRecoveryDirectAllocationSpec,
    spec: Stage1ControllerAwareDistillationSpec,
) -> pd.DataFrame:
    equity = peak = 1.0
    previous = np.zeros(3)
    state = clock = 0
    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(
            state, clock, drawdown, source_spec
        )
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
        model = models.get((candidate.name, row.signal_date.year))
        eligible = (
            row.signal_date.year >= spec.first_action_year
            and model is not None
            and drawdown >= source_spec.recovery_eligibility_drawdown
            and raw_beta > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
        )
        predicted_uplift = 0.0
        target_beta = cap
        if eligible:
            feature = pd.DataFrame(
                [
                    _feature_row(
                        row,
                        drawdown=drawdown,
                        cap=cap,
                        raw_beta=raw_beta,
                        state=state,
                        history=history,
                    )
                ]
            )
            predicted_uplift = max(
                0.0,
                float(
                    model.predict(
                        feature.loc[:, list(FEATURE_COLUMNS)]
                    )[0]
                ),
            )
            if (
                predicted_uplift
                >= candidate.minimum_predicted_uplift
            ):
                target_beta = min(
                    raw_beta,
                    candidate.maximum_recovery_beta,
                    cap + candidate.shrinkage * predicted_uplift,
                )
        action = target_beta > cap + 1e-12
        if action:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                target_beta,
                minimum_cash_weight=source_spec.minimum_cash_weight,
                maximum_soxl_weight=source_spec.maximum_soxl_weight,
            )
            early = np.array(
                [row.early_ret_soxl, row.early_ret_soxx, 0.0]
            )
            value_at_cut = 1.0 + open_weights @ early
            drifted = open_weights * (1.0 + early) / value_at_cut
            post = np.array(
                [
                    (1.0 + row.soxl_return)
                    / (1.0 + row.early_ret_soxl)
                    - 1.0,
                    (1.0 + row.soxx_return)
                    / (1.0 + row.early_ret_soxx)
                    - 1.0,
                    0.0,
                ]
            )
            gross_return = (
                value_at_cut * (1.0 + close_weights @ post) - 1.0
            )
            turnover = np.abs(
                open_weights[:2] - previous[:2]
            ).sum() + np.abs(close_weights[:2] - drifted[:2]).sum()
        else:
            close_weights = open_weights
            gross_return = (
                open_weights[0] * row.soxl_return
                + open_weights[1] * row.soxx_return
            )
            turnover = np.abs(
                open_weights[:2] - previous[:2]
            ).sum()
        strategy_return = (
            gross_return
            - turnover * spec.transaction_cost_bps / 10_000.0
        )
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = close_weights
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "segment": row.segment,
                "soxl_return": row.soxl_return,
                "soxx_return": row.soxx_return,
                "open_soxl_weight": open_weights[0],
                "open_soxx_weight": open_weights[1],
                "open_cash_weight": open_weights[2],
                "soxl_weight": close_weights[0],
                "soxx_weight": close_weights[1],
                "cash_weight": close_weights[2],
                "predicted_beta_uplift": predicted_uplift,
                "recovery_action": action,
                "current_beta_cap": cap,
                "recovery_beta_target": target_beta,
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": spec.transaction_cost_bps,
            }
        )
    return pd.DataFrame(rows)


def _metrics_for_fixed_cost(
    replay: pd.DataFrame, cost_bps: float
) -> dict[str, float]:
    returns = (
        replay["gross_return"].to_numpy()
        - replay["turnover"].to_numpy() * cost_bps / 10_000.0
    )
    return _risk_metrics(returns)


def _annual_metrics(
    replay: pd.DataFrame, *, variant: str, cost_bps: float
) -> pd.DataFrame:
    rows = []
    for year, section in replay.groupby(replay["signal_date"].dt.year):
        returns = (
            section["gross_return"].to_numpy()
            - section["turnover"].to_numpy()
            * cost_bps
            / 10_000.0
        )
        metrics = _risk_metrics(returns)
        soxx_wealth = float(
            np.prod(1.0 + section["soxx_return"].to_numpy())
        )
        rows.append(
            {
                "variant": variant,
                "year": int(year),
                "cost_bps": cost_bps,
                **metrics,
                "terminal_wealth_ratio_vs_soxx": (
                    metrics["terminal_wealth"] / soxx_wealth
                ),
                "turnover": float(section["turnover"].sum()),
                "action_count": int(section["recovery_action"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _candidate_row(
    *,
    variant: str,
    candidate: DistillationCandidate | None,
    replay: pd.DataFrame,
    baseline: pd.DataFrame,
    segment: str,
    spec: Stage1ControllerAwareDistillationSpec,
) -> dict[str, Any]:
    metrics = _metrics_for_fixed_cost(replay, spec.transaction_cost_bps)
    stress = _metrics_for_fixed_cost(replay, spec.stress_cost_bps)
    stability = (
        _development_stability(replay)
        if segment == "development_walk_forward"
        else {}
    )
    baseline_metrics = _metrics_for_fixed_cost(
        baseline, spec.transaction_cost_bps
    )
    baseline_turnover = float(baseline["turnover"].sum())
    turnover = float(replay["turnover"].sum())
    return {
        "variant": variant,
        "segment": segment,
        "selection_uses_outer_test": False,
        **(
            {
                "ridge_alpha": candidate.ridge_alpha,
                "shrinkage": candidate.shrinkage,
                "minimum_predicted_uplift": (
                    candidate.minimum_predicted_uplift
                ),
                "maximum_recovery_beta": (
                    candidate.maximum_recovery_beta
                ),
            }
            if candidate is not None
            else {
                "ridge_alpha": np.nan,
                "shrinkage": np.nan,
                "minimum_predicted_uplift": np.nan,
                "maximum_recovery_beta": 0.60,
            }
        ),
        **metrics,
        "stress_25bps_return_pct": stress["return_pct"],
        "stress_25bps_max_drawdown_pct": stress["max_drawdown_pct"],
        "stress_25bps_worst_rolling_5d_pct": (
            stress["worst_rolling_5d_pct"]
        ),
        **stability,
        "turnover": turnover,
        "turnover_reduction_vs_stage1": (
            1.0 - turnover / baseline_turnover
        ),
        "action_count": int(replay["recovery_action"].sum()),
        "return_delta_vs_stage1_pct_points": (
            metrics["return_pct"] - baseline_metrics["return_pct"]
        ),
        "max_drawdown_delta_vs_stage1_pct_points": (
            metrics["max_drawdown_pct"]
            - baseline_metrics["max_drawdown_pct"]
        ),
        "worst_5d_delta_vs_stage1_pct_points": (
            metrics["worst_rolling_5d_pct"]
            - baseline_metrics["worst_rolling_5d_pct"]
        ),
    }


def _select_development_candidate(
    table: pd.DataFrame,
    spec: Stage1ControllerAwareDistillationSpec,
) -> tuple[str, dict[str, Any]]:
    candidates = table[
        table["segment"].eq("development_walk_forward")
        & table["variant"].ne("stage1_baseline")
    ].copy()
    candidates["risk_gate_pass"] = (
        candidates["max_drawdown_pct"].ge(-30.0)
        & candidates["worst_day_pct"].ge(-12.0)
        & candidates["cvar5_pct"].ge(-8.0)
        & candidates["worst_rolling_5d_pct"].ge(-12.0)
        & candidates["risk_window_pass_count"].eq(7)
    )
    candidates["relative_window_gate_pass"] = (
        candidates["positive_relative_window_count"].ge(5)
        & candidates["relative_ratio_p25"].ge(0.95)
    )
    candidates["stage1_return_gate_pass"] = candidates[
        "return_delta_vs_stage1_pct_points"
    ].ge(0.0)
    for index, row in candidates.iterrows():
        same = candidates[
            candidates["ridge_alpha"].eq(row["ridge_alpha"])
            & candidates["shrinkage"].eq(row["shrinkage"])
            & candidates["minimum_predicted_uplift"].between(
                row["minimum_predicted_uplift"] - 0.0500001,
                row["minimum_predicted_uplift"] + 0.0500001,
            )
        ]
        candidates.loc[index, "neighbor_count"] = len(same)
        candidates.loc[index, "neighbor_min_wealth_ratio"] = (
            same["terminal_wealth"].min() / row["terminal_wealth"]
        )
    candidates["neighbor_gate_pass"] = (
        candidates["neighbor_count"].ge(3)
        & candidates["neighbor_min_wealth_ratio"].ge(
            spec.minimum_neighbor_wealth_ratio
        )
    )
    eligible = candidates[
        candidates[
            [
                "risk_gate_pass",
                "relative_window_gate_pass",
                "stage1_return_gate_pass",
                "neighbor_gate_pass",
            ]
        ].all(axis=1)
    ]
    if eligible.empty:
        selected = candidates.sort_values(
            [
                "risk_gate_pass",
                "relative_window_gate_pass",
                "terminal_wealth",
            ],
            ascending=False,
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["terminal_wealth", "turnover"],
            ascending=[False, True],
        ).iloc[0]
    audit = {
        "risk_gate_pass": bool(selected["risk_gate_pass"]),
        "relative_window_gate_pass": bool(
            selected["relative_window_gate_pass"]
        ),
        "stage1_return_gate_pass": bool(
            selected["stage1_return_gate_pass"]
        ),
        "neighbor_gate_pass": bool(selected["neighbor_gate_pass"]),
        "neighbor_count": int(selected["neighbor_count"]),
        "neighbor_min_wealth_ratio": float(
            selected["neighbor_min_wealth_ratio"]
        ),
        "turnover_gate_pass": bool(
            selected["turnover_reduction_vs_stage1"]
            >= spec.required_turnover_reduction
        ),
        "required_turnover_reduction": spec.required_turnover_reduction,
    }
    return str(selected["variant"]), audit


def _run_card(
    spec: Stage1ControllerAwareDistillationSpec,
) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_contract = _source_contract(spec)
    source_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(source_spec)
    teacher_frame = _build_teacher_training_frame(
        frame, spec=source_spec, teacher_beta=source_spec.teacher_recovery_beta
    )
    candidate_teacher_frame = teacher_frame[
        teacher_frame["year"].between(
            spec.teacher_start_year,
            spec.teacher_end_year,
        )
    ].copy()
    teacher = _build_teacher_targets(
        frame,
        candidate_teacher_frame,
        source_spec=source_spec,
        spec=spec,
    )
    dataset, dataset_audit = _build_qlib_dataset(teacher, spec)
    models, training_diagnostics = _fit_models(dataset, CANDIDATES, spec)

    development = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    baseline_cache: dict[tuple[int, float], Any] = {}
    baseline_development = _simulate_policy(
        development,
        spec=source_spec,
        teacher_frame=teacher_frame,
        c_value=0.03,
        threshold=0.55,
        action_beta=0.60,
        first_action_year=spec.first_action_year,
        model_cache=baseline_cache,
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
        model_cache=baseline_cache,
        cost_bps=spec.transaction_cost_bps,
    )
    replays: dict[str, dict[str, pd.DataFrame]] = {
        "stage1_baseline": {
            "development_walk_forward": baseline_development,
            "outer_diagnostic": baseline_outer,
        }
    }
    rows = [
        _candidate_row(
            variant="stage1_baseline",
            candidate=None,
            replay=baseline_development,
            baseline=baseline_development,
            segment="development_walk_forward",
            spec=spec,
        ),
        _candidate_row(
            variant="stage1_baseline",
            candidate=None,
            replay=baseline_outer,
            baseline=baseline_outer,
            segment="outer_diagnostic",
            spec=spec,
        ),
    ]
    for candidate in CANDIDATES:
        dev_replay = _simulate_distilled_policy(
            development,
            candidate=candidate,
            models=models,
            source_spec=source_spec,
            spec=spec,
        )
        outer_replay = _simulate_distilled_policy(
            outer,
            candidate=candidate,
            models=models,
            source_spec=source_spec,
            spec=spec,
        )
        replays[candidate.name] = {
            "development_walk_forward": dev_replay,
            "outer_diagnostic": outer_replay,
        }
        rows.extend(
            [
                _candidate_row(
                    variant=candidate.name,
                    candidate=candidate,
                    replay=dev_replay,
                    baseline=baseline_development,
                    segment="development_walk_forward",
                    spec=spec,
                ),
                _candidate_row(
                    variant=candidate.name,
                    candidate=candidate,
                    replay=outer_replay,
                    baseline=baseline_outer,
                    segment="outer_diagnostic",
                    spec=spec,
                ),
            ]
        )
    comparison = pd.DataFrame(rows)
    selected_name, selection_audit = _select_development_candidate(
        comparison, spec
    )
    comparison["selected"] = comparison["variant"].eq(selected_name)
    selected_candidate = next(
        candidate for candidate in CANDIDATES if candidate.name == selected_name
    )
    selected_development = replays[selected_name][
        "development_walk_forward"
    ]
    selected_outer = replays[selected_name]["outer_diagnostic"]
    selected_dev_row = comparison[
        comparison["variant"].eq(selected_name)
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    selected_outer_row = comparison[
        comparison["variant"].eq(selected_name)
        & comparison["segment"].eq("outer_diagnostic")
    ].iloc[0]
    baseline_dev_row = comparison[
        comparison["variant"].eq("stage1_baseline")
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    annual = pd.concat(
        [
            _annual_metrics(
                baseline_development,
                variant="stage1_baseline",
                cost_bps=spec.transaction_cost_bps,
            ),
            _annual_metrics(
                selected_development,
                variant=selected_name,
                cost_bps=spec.transaction_cost_bps,
            ),
            _annual_metrics(
                baseline_outer,
                variant="stage1_baseline",
                cost_bps=spec.transaction_cost_bps,
            ),
            _annual_metrics(
                selected_outer,
                variant=selected_name,
                cost_bps=spec.transaction_cost_bps,
            ),
        ],
        ignore_index=True,
    )
    promotion_gates = {
        **selection_audit,
        "development_return_delta_pct_points": float(
            selected_dev_row["return_delta_vs_stage1_pct_points"]
        ),
        "development_turnover_reduction": float(
            selected_dev_row["turnover_reduction_vs_stage1"]
        ),
        "clean_unseen_holdout_available": False,
        "outer_used_for_selection": False,
    }
    all_development_gates = all(
        promotion_gates[key]
        for key in [
            "risk_gate_pass",
            "relative_window_gate_pass",
            "stage1_return_gate_pass",
            "neighbor_gate_pass",
            "turnover_gate_pass",
        ]
    )
    decision = (
        "candidate_for_clean_forward_validation_not_promoted"
        if all_development_gates
        else "not_promoted_turnover_or_robustness_gate_failed"
    )

    from qlib.contrib.evaluate import risk_analysis
    from qlib.workflow import R

    risk_rows = []
    for segment, replay in [
        ("development_walk_forward", selected_development),
        ("outer_diagnostic", selected_outer),
    ]:
        for cost_bps in (
            spec.transaction_cost_bps,
            spec.stress_cost_bps,
        ):
            returns = (
                replay["gross_return"]
                - replay["turnover"] * cost_bps / 10_000.0
            )
            qlib_risk = risk_analysis(
                pd.Series(
                    returns.to_numpy(),
                    index=pd.to_datetime(replay["trade_date"]),
                ),
                freq="day",
            )
            flattened = qlib_risk.reset_index()
            flattened["segment"] = segment
            flattened["cost_bps"] = cost_bps
            risk_rows.append(flattened)
    qlib_risk_analysis = pd.concat(risk_rows, ignore_index=True)

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        teacher.to_csv(temp / "teacher_training_frame.csv", index=False)
        comparison.to_csv(temp / "model_selection_full.csv", index=False)
        pd.concat(
            [
                selected_development.assign(
                    selected_variant=selected_name
                ),
                selected_outer.assign(selected_variant=selected_name),
            ],
            ignore_index=True,
        ).to_csv(temp / "portfolio_report.csv", index=False)
        models_dir = temp / "models"
        models_dir.mkdir()
        with (models_dir / f"{selected_name}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    year: models[(selected_name, year)]
                    for year in (
                        *range(
                            spec.first_action_year,
                            spec.teacher_end_year + 1,
                        ),
                        *spec.outer_years,
                    )
                    if (selected_name, year) in models
                },
                handle,
            )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_stage1_recorder=spec.source_recorder_id,
                teacher="safe_beta_grid_after_10_30",
                selected_candidate=selected_name,
                selection_uses_outer_test="false",
                outer_model="frozen_training_end_2022",
            )
            R.log_metrics(
                development_return_pct=float(
                    selected_dev_row["return_pct"]
                ),
                development_return_delta_pct_points=float(
                    selected_dev_row[
                        "return_delta_vs_stage1_pct_points"
                    ]
                ),
                development_turnover_reduction=float(
                    selected_dev_row[
                        "turnover_reduction_vs_stage1"
                    ]
                ),
                outer_return_pct=float(selected_outer_row["return_pct"]),
                all_development_gates=float(all_development_gates),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(
                    artifact.as_posix(), artifact_path=spec.experiment
                )
            recorder.log_artifact(
                (models_dir / f"{selected_name}.pkl").as_posix(),
                artifact_path=f"{spec.experiment}/models",
            )
            recorder_id = recorder.id
            recorder_local_dir = Path(
                recorder.get_local_dir()
            ).resolve()
            summary = _write_card(
                spec=spec,
                source_contract=source_contract,
                source_audit=source_audit,
                dataset_audit=dataset_audit,
                teacher=teacher,
                selected_candidate=selected_candidate,
                comparison=comparison,
                annual=annual,
                training_diagnostics=training_diagnostics,
                qlib_risk_analysis=qlib_risk_analysis,
                promotion_gates=promotion_gates,
                decision=decision,
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


def _remove_qlib_code_snapshot_artifacts(
    recorder_local_dir: Path,
) -> None:
    artifacts_dir = recorder_local_dir / "artifacts"
    for filename in (
        "code_cached.txt",
        "code_diff.txt",
        "code_status.txt",
    ):
        (artifacts_dir / filename).unlink(missing_ok=True)


def _write_card(
    *,
    spec: Stage1ControllerAwareDistillationSpec,
    source_contract: dict[str, Any],
    source_audit: dict[str, Any],
    dataset_audit: dict[str, Any],
    teacher: pd.DataFrame,
    selected_candidate: DistillationCandidate,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    training_diagnostics: pd.DataFrame,
    qlib_risk_analysis: pd.DataFrame,
    promotion_gates: dict[str, Any],
    decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    card = spec.card_dir
    selected_dev = comparison[
        comparison["variant"].eq(selected_candidate.name)
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    selected_outer = comparison[
        comparison["variant"].eq(selected_candidate.name)
        & comparison["segment"].eq("outer_diagnostic")
    ].iloc[0]
    baseline_dev = comparison[
        comparison["variant"].eq("stage1_baseline")
        & comparison["segment"].eq("development_walk_forward")
    ].iloc[0]
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": (
            "backtest_benchmark_comparison.csv"
        ),
        "annual_metrics": "annual_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "training_summary": "training_summary.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
        "conclusion": "conclusion.md",
        "workflow_config": "workflow.yaml",
    }
    summary: dict[str, Any] = {
        "experiment": spec.experiment,
        "card_type": "stage1_controller_aware_distillation_research_card",
        "primary_notebook": f"{spec.experiment}.ipynb",
        "system_of_record": "qlib",
        "provider_uri": source_audit["source_market_provider_uri"],
        "intraday_provider_uri": source_audit["intraday_provider_uri"],
        "as_of": source_audit["intraday_provider_as_of"],
        "market": "US",
        "region": "us",
        "source_stage1": source_contract,
        "hypothesis": (
            "A regularized continuous recovery-beta teacher can align "
            "the Stage1 safety controller with its 10:30 execution path "
            "without relaxing the 30% SOXL cap."
        ),
        "method": {
            "teacher": (
                "development-only hindsight best legal beta on the fixed "
                "0.4/0.6/0.8/1.0/1.2 grid after the first four 15m bars"
            ),
            "features": list(FEATURE_COLUMNS),
            "model": "qlib DatasetH plus regularized Ridge",
            "development": "walk-forward annual fits through 2022",
            "outer": (
                "all challenger rows in 2023-2026 use one model trained "
                "only through 2022; outer is diagnostic and never "
                "selection input"
            ),
            "incumbent_baseline": (
                "the source Stage1 keeps its prescribed annual "
                "walk-forward prior-year teacher updates; challenger "
                "selection never reads the resulting outer comparison"
            ),
            "hard_constraints": {
                "maximum_soxl_weight": 0.30,
                "minimum_cash_weight": 0.15,
                "maximum_recovery_beta": (
                    selected_candidate.maximum_recovery_beta
                ),
            },
        },
        "dataset_audit": dataset_audit,
        "teacher_audit": {
            "rows": len(teacher),
            "positive_uplift_rows": int(
                teacher["teacher_target_uplift"].gt(0.01).sum()
            ),
            "development_target_end": (
                f"{spec.teacher_end_year}-12-31"
            ),
        },
        "candidate_count": len(CANDIDATES),
        "selected_model": asdict(selected_candidate),
        "development_baseline_metrics_10bps": _json_row(baseline_dev),
        "development_selected_metrics_10bps": _json_row(selected_dev),
        "outer_diagnostic_selected_metrics_10bps": _json_row(
            selected_outer
        ),
        "promotion_gates": promotion_gates,
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_clean_unseen_holdout",
        "decision": decision,
        "promotion": "blocked",
        "qlib_api": [
            "qlib.init",
            "qlib.data.D.calendar",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R",
            "qlib.contrib.evaluate.risk_analysis",
        ],
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/model.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "https://qlib.readthedocs.io/en/latest/reference/api.html",
        ],
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/teacher_training_frame.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/portfolio_report.csv",
                f"artifacts/{spec.experiment}/models/{selected_candidate.name}.pkl",
            ],
        },
    }
    (card / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    comparison.to_csv(card / "model_selection.csv", index=False)
    comparison[
        comparison["variant"].isin(
            ["stage1_baseline", selected_candidate.name]
        )
    ].to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual.to_csv(card / "annual_metrics.csv", index=False)
    qlib_risk_analysis.to_csv(card / "risk_analysis.csv", index=False)
    training_diagnostics.to_csv(
        card / "training_summary.csv", index=False
    )
    (card / "selected_model.json").write_text(
        json.dumps(
            {
                **asdict(selected_candidate),
                "selection_uses_outer_test": False,
                "development_metrics": _json_row(selected_dev),
                "promotion_gates": promotion_gates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    analysis = {
        "selected_candidate": selected_candidate.name,
        "development_return_delta_pct_points": float(
            selected_dev["return_delta_vs_stage1_pct_points"]
        ),
        "development_max_drawdown_delta_pct_points": float(
            selected_dev[
                "max_drawdown_delta_vs_stage1_pct_points"
            ]
        ),
        "development_worst_5d_delta_pct_points": float(
            selected_dev["worst_5d_delta_vs_stage1_pct_points"]
        ),
        "development_turnover_reduction": float(
            selected_dev["turnover_reduction_vs_stage1"]
        ),
        "outer_return_delta_pct_points": float(
            selected_outer["return_delta_vs_stage1_pct_points"]
        ),
        "decision": decision,
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    failed = [
        key
        for key, value in promotion_gates.items()
        if key.endswith("_gate_pass") and value is False
    ]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + "\n".join(f"- {item}" for item in failed)
        + "\n- clean_unseen_holdout_unavailable\n",
        encoding="utf-8",
    )
    conclusion = (
        "# 中文结论\n\n"
        "- 做了什么：冻结现有 Stage1、Stage2 关闭和 30% SOXL 上限，"
        "使用 qlib DatasetH 将 10:30 后合法 beta 网格的最优增量蒸馏为"
        "连续 Ridge 目标；开发期逐年 walk-forward，2023–2026 共用截至"
        " 2022 年的冻结候选模型；对照组保留现有 Stage1 的逐年更新语义。\n"
        f"- 开发期：候选 `{selected_candidate.name}` 10bp return="
        f"{float(selected_dev['return_pct']):.4f}%，相对 Stage1 增量="
        f"{float(selected_dev['return_delta_vs_stage1_pct_points']):.4f} "
        f"个百分点，MaxDD={float(selected_dev['max_drawdown_pct']):.4f}%，"
        f"worst-5d={float(selected_dev['worst_rolling_5d_pct']):.4f}%。\n"
        f"- 换手：开发期 turnover reduction="
        f"{100.0 * float(selected_dev['turnover_reduction_vs_stage1']):.2f}%，"
        f"要求={100.0 * spec.required_turnover_reduction:.2f}%，"
        f"gate={promotion_gates['turnover_gate_pass']}。\n"
        f"- outer 复用诊断：return={float(selected_outer['return_pct']):.4f}%，"
        f"相对 Stage1 增量="
        f"{float(selected_outer['return_delta_vs_stage1_pct_points']):.4f} "
        "个百分点；该区间不参与选择，也不是 clean holdout。\n"
        f"- 决策：`{decision}`；生产 Stage1 不变，真实资金晋升 blocked。\n"
        f"- 风险：失败门={failed}；连续模型明显增加动作频率，历史开发期"
        "改善不足以替代新的连续前向模拟。\n"
        "- 下一步：保持本卡为失败边界证据；若继续，只能预登记更强的"
        "换手/动作稀疏惩罚并等待 clean-forward，不用 outer 调阈值。\n"
        f"- qlib recorder：`{recorder_id}`。\n"
        "- 不构成买卖建议。\n"
    )
    (card / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    (card / "workflow.yaml").write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                f"provider_uri: {summary['provider_uri']}",
                f"intraday_provider_uri: {summary['intraday_provider_uri']}",
                f"recorder_uri: {summary['recorder_uri']}",
                "market: US",
                "region: us",
                "source_stage1_frozen: true",
                "stage2_enabled: false",
                "selection_uses_outer_test: false",
                "outer_model_training_end: 2022-12-31",
                "incumbent_baseline: prescribed_annual_walk_forward_updates",
                "model_selection:",
                "  metric: development_terminal_wealth_then_turnover",
                "  parameter_neighborhood_required: true",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  workflow_config: workflow.yaml",
                "  recorder_only_artifacts:",
                *[
                    f"    - {artifact}"
                    for artifact in summary["artifact_policy"][
                        "recorder_only_artifacts"
                    ]
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
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
