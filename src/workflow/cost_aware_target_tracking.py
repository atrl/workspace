from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _annual_metrics,
    _base_cap,
    _build_teacher_training_frame,
    _development_stability,
    _feature_row,
    _fit_recovery_model,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    project_legal_weights,
)


EXPERIMENT = "cost_aware_target_tracking_v1"
BASELINE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
CARD_TYPE = "cost_aware_asymmetric_target_tracking_card"


@dataclass(frozen=True)
class CostAwareTargetTrackingSpec(IntradayRecoveryDirectAllocationSpec):
    experiment: str = EXPERIMENT
    baseline_experiment: str = BASELINE_EXPERIMENT
    execution_candidates: tuple[tuple[str, float, float], ...] = (
        ("current_stage1_full_target_baseline", 1.0, 0.0),
        ("asymmetric_risk_up_0.5", 0.5, 0.0),
        ("asymmetric_risk_up_0.3333", 1.0 / 3.0, 0.0),
        ("aim_ahead_risk_up_0.5", 0.5, 1.0),
        ("aim_ahead_risk_up_0.3333", 1.0 / 3.0, 2.0),
    )
    expected_open_cost_bps: float = 25.0
    expected_intraday_cost_bps: float = 10.0
    tail_open_cost_bps: float = 50.0
    tail_intraday_cost_bps: float = 25.0
    minimum_turnover_reduction_pct: float = 10.0

    @property
    def baseline_card_dir(self) -> Path:
        return self.root / "research" / self.baseline_experiment


def run_cost_aware_target_tracking_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    return _run_card(CostAwareTargetTrackingSpec(root=_resolve_root(root)))


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def asymmetric_track_open_weights(
    previous_weights: Iterable[float],
    desired_weights: Iterable[float],
    *,
    risk_up_fraction: float,
) -> tuple[np.ndarray, str]:
    previous = np.asarray(list(previous_weights), dtype="float64")
    desired = np.asarray(list(desired_weights), dtype="float64")
    if previous.shape != (3,) or desired.shape != (3,):
        raise ValueError("previous and desired weights must each contain three assets")
    if not 0.0 < float(risk_up_fraction) <= 1.0:
        raise ValueError("risk_up_fraction must be in (0, 1]")
    if not np.isclose(previous.sum(), 1.0) or not np.isclose(desired.sum(), 1.0):
        raise ValueError("previous and desired weights must each sum to one")
    if np.any(previous < -1e-12) or np.any(desired < -1e-12):
        raise ValueError("weights must be non-negative")

    previous_beta = 3.0 * previous[0] + previous[1]
    desired_beta = 3.0 * desired[0] + desired[1]
    if desired_beta < previous_beta - 1e-12:
        return desired.copy(), "full_risk_reduction"
    if float(risk_up_fraction) >= 1.0 - 1e-12:
        return desired.copy(), "full_target"
    tracked = previous + float(risk_up_fraction) * (desired - previous)
    tracked[-1] = 1.0 - tracked[0] - tracked[1]
    return tracked, "partial_risk_increase_or_rotation"


def apply_split_fixed_decision_cost_stress(
    gross_returns: Iterable[float],
    open_turnover: Iterable[float],
    intraday_turnover: Iterable[float],
    *,
    open_cost_bps: float,
    intraday_cost_bps: float,
) -> np.ndarray:
    gross = np.asarray(list(gross_returns), dtype="float64")
    open_turns = np.asarray(list(open_turnover), dtype="float64")
    intraday_turns = np.asarray(list(intraday_turnover), dtype="float64")
    if gross.shape != open_turns.shape or gross.shape != intraday_turns.shape:
        raise ValueError("gross returns and turnover inputs must have identical shapes")
    return (
        gross
        - open_turns * float(open_cost_bps) / 10_000.0
        - intraday_turns * float(intraday_cost_bps) / 10_000.0
    )


def select_cost_aware_development_candidate(
    candidates: pd.DataFrame,
) -> pd.Series:
    required = {
        "segment",
        "selection_uses_outer_test",
        "development_gate_pass",
        "expected_cost_terminal_wealth",
        "expected_cost_return_pct",
        "total_turnover",
        "risk_up_fraction",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    development = candidates[
        candidates["segment"].eq("development_walk_forward")
        & ~candidates["selection_uses_outer_test"].astype(bool)
    ].copy()
    if development.empty:
        raise ValueError("candidate table has no development-only rows")
    eligible = development[
        development["development_gate_pass"].astype(bool)
    ].copy()
    if eligible.empty:
        eligible = development
    return eligible.sort_values(
        [
            "expected_cost_terminal_wealth",
            "expected_cost_return_pct",
            "total_turnover",
            "risk_up_fraction",
        ],
        ascending=[False, False, True, False],
    ).iloc[0]


def simulate_cost_aware_policy(
    frame: pd.DataFrame,
    *,
    spec: CostAwareTargetTrackingSpec,
    teacher_frame: pd.DataFrame,
    c_value: float,
    threshold: float,
    action_beta: float,
    first_action_year: int,
    model_cache: dict[tuple[int, float], Any],
    risk_up_fraction: float,
    open_cost_bps: float,
    intraday_cost_bps: float,
    aim_momentum_multiplier: float = 0.0,
) -> pd.DataFrame:
    models: dict[int, Any] = {}
    for year in sorted(frame["signal_date"].dt.year.unique()):
        key = (int(year), float(c_value))
        if key not in model_cache:
            model_cache[key] = _fit_recovery_model(
                teacher_frame,
                year=int(year),
                c_value=c_value,
            )
        models[int(year)] = model_cache[key]

    equity = peak = 1.0
    previous = np.array([0.0, 0.0, 1.0], dtype="float64")
    previous_desired = previous.copy()
    state = clock = 0
    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(state, clock, drawdown, spec)
        cap, raw_beta = _base_cap(
            row,
            q=q,
            drawdown=drawdown,
            history=history,
            spec=spec,
        )
        desired_open_weights = project_legal_weights(
            row.soxl_weight,
            row.soxx_weight,
            cap,
            minimum_cash_weight=spec.minimum_cash_weight,
            maximum_soxl_weight=spec.maximum_soxl_weight,
        )
        execution_target = desired_open_weights
        previous_beta = 3.0 * previous[0] + previous[1]
        desired_beta = (
            3.0 * desired_open_weights[0] + desired_open_weights[1]
        )
        if (
            aim_momentum_multiplier > 0.0
            and desired_beta >= previous_beta - 1e-12
        ):
            raw_aim = desired_open_weights + aim_momentum_multiplier * (
                desired_open_weights - previous_desired
            )
            execution_target = project_legal_weights(
                max(0.0, raw_aim[0]),
                max(0.0, raw_aim[1]),
                cap,
                minimum_cash_weight=spec.minimum_cash_weight,
                maximum_soxl_weight=spec.maximum_soxl_weight,
            )
        open_weights, adjustment_type = asymmetric_track_open_weights(
            previous,
            execution_target,
            risk_up_fraction=risk_up_fraction,
        )
        open_turnover = float(
            np.abs(open_weights[:2] - previous[:2]).sum()
        )

        target_beta = min(raw_beta, action_beta)
        model = models.get(row.signal_date.year)
        eligible = (
            row.signal_date.year >= first_action_year
            and model is not None
            and drawdown >= spec.recovery_eligibility_drawdown
            and target_beta > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
        )
        probability = np.nan
        recovery = False
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
            probability = float(
                model.predict_proba(feature.loc[:, FEATURE_COLUMNS])[0, 1]
            )
            recovery = probability >= threshold

        intraday_turnover = 0.0
        if recovery:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                target_beta,
                minimum_cash_weight=spec.minimum_cash_weight,
                maximum_soxl_weight=spec.maximum_soxl_weight,
            )
            early = np.array(
                [row.early_ret_soxl, row.early_ret_soxx, 0.0],
                dtype="float64",
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
                ],
                dtype="float64",
            )
            gross_return = value_at_cut * (1.0 + close_weights @ post) - 1.0
            intraday_turnover = float(
                np.abs(close_weights[:2] - drifted[:2]).sum()
            )
        else:
            close_weights = open_weights
            gross_return = float(
                open_weights[0] * row.soxl_return
                + open_weights[1] * row.soxx_return
            )

        turnover = open_turnover + intraday_turnover
        strategy_return = float(
            gross_return
            - open_turnover * open_cost_bps / 10_000.0
            - intraday_turnover * intraday_cost_bps / 10_000.0
        )
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = close_weights
        previous_desired = desired_open_weights
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "segment": row.segment,
                "soxl_return": row.soxl_return,
                "soxx_return": row.soxx_return,
                "desired_open_soxl_weight": desired_open_weights[0],
                "desired_open_soxx_weight": desired_open_weights[1],
                "desired_open_cash_weight": desired_open_weights[2],
                "aim_open_soxl_weight": execution_target[0],
                "aim_open_soxx_weight": execution_target[1],
                "aim_open_cash_weight": execution_target[2],
                "open_soxl_weight": open_weights[0],
                "open_soxx_weight": open_weights[1],
                "open_cash_weight": open_weights[2],
                "soxl_weight": close_weights[0],
                "soxx_weight": close_weights[1],
                "cash_weight": close_weights[2],
                "recovery_probability": probability,
                "recovery_action": recovery,
                "open_adjustment_type": adjustment_type,
                "risk_up_fraction": risk_up_fraction,
                "aim_momentum_multiplier": aim_momentum_multiplier,
                "current_beta_cap": cap,
                "recovery_beta_target": target_beta if recovery else cap,
                "gross_return": gross_return,
                "open_turnover": open_turnover,
                "intraday_turnover": intraday_turnover,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "open_cost_bps": open_cost_bps,
                "intraday_cost_bps": intraday_cost_bps,
            }
        )
    return pd.DataFrame(rows)


def _turnover_metrics(replay: pd.DataFrame) -> dict[str, float | int]:
    open_turnover = float(replay["open_turnover"].sum())
    intraday_turnover = float(replay["intraday_turnover"].sum())
    total = open_turnover + intraday_turnover
    return {
        "open_turnover": open_turnover,
        "intraday_turnover": intraday_turnover,
        "total_turnover": total,
        "open_turnover_share_pct": (
            100.0 * open_turnover / total if total > 0.0 else 0.0
        ),
        "open_active_day_count": int(replay["open_turnover"].gt(1e-12).sum()),
        "open_material_day_count": int(replay["open_turnover"].ge(0.02).sum()),
        "intraday_action_count": int(replay["recovery_action"].sum()),
    }


def _candidate_row(
    expected_cost_replay: pd.DataFrame,
    current_cost_replay: pd.DataFrame,
    *,
    variant: str,
    fraction: float,
    aim_momentum_multiplier: float,
) -> dict[str, Any]:
    expected_metrics = _risk_metrics(
        expected_cost_replay["strategy_return"].to_numpy()
    )
    current_metrics = _risk_metrics(
        current_cost_replay["strategy_return"].to_numpy()
    )
    current_stability = _development_stability(current_cost_replay)
    expected_stability = _development_stability(expected_cost_replay)
    gate = bool(
        _passes_risk(current_metrics)
        and current_stability["risk_window_pass_count"] == 7
        and current_stability["positive_relative_window_count"] >= 5
        and current_stability["relative_ratio_p25"] >= 0.95
    )
    turnover = _turnover_metrics(expected_cost_replay)
    return {
        "variant": variant,
        "segment": "development_walk_forward",
        "selection_uses_outer_test": False,
        "risk_up_fraction": fraction,
        "aim_momentum_multiplier": aim_momentum_multiplier,
        "expected_cost_open_bps": float(
            expected_cost_replay["open_cost_bps"].iloc[0]
        ),
        "expected_cost_intraday_bps": float(
            expected_cost_replay["intraday_cost_bps"].iloc[0]
        ),
        "expected_cost_terminal_wealth": expected_metrics["terminal_wealth"],
        "expected_cost_return_pct": expected_metrics["return_pct"],
        "expected_cost_max_drawdown_pct": expected_metrics["max_drawdown_pct"],
        "expected_cost_worst_day_pct": expected_metrics["worst_day_pct"],
        "expected_cost_cvar5_pct": expected_metrics["cvar5_pct"],
        "expected_cost_worst_rolling_5d_pct": expected_metrics[
            "worst_rolling_5d_pct"
        ],
        "current_cost_return_pct": current_metrics["return_pct"],
        "current_cost_max_drawdown_pct": current_metrics["max_drawdown_pct"],
        **current_stability,
        "expected_cost_risk_window_pass_count": expected_stability[
            "risk_window_pass_count"
        ],
        "expected_cost_positive_relative_window_count": expected_stability[
            "positive_relative_window_count"
        ],
        "expected_cost_relative_ratio_min": expected_stability[
            "relative_ratio_min"
        ],
        "expected_cost_relative_ratio_p25": expected_stability[
            "relative_ratio_p25"
        ],
        "expected_cost_relative_ratio_median": expected_stability[
            "relative_ratio_median"
        ],
        **turnover,
        "development_gate_pass": gate,
    }
def _fixed_cost_sensitivity(
    replay: pd.DataFrame,
    *,
    variant: str,
    scenarios: tuple[tuple[str, float, float], ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, open_bps, intraday_bps in scenarios:
        stressed = apply_split_fixed_decision_cost_stress(
            replay["gross_return"],
            replay["open_turnover"],
            replay["intraday_turnover"],
            open_cost_bps=open_bps,
            intraday_cost_bps=intraday_bps,
        )
        rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "decision_path": "fixed_from_10bp_open_10bp_intraday",
                "open_cost_bps": open_bps,
                "intraday_cost_bps": intraday_bps,
                **_risk_metrics(stressed),
            }
        )
    return pd.DataFrame(rows)


def _run_card(spec: CostAwareTargetTrackingSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_source_frame(spec)
    baseline_summary = json.loads(
        (spec.baseline_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if baseline_summary.get("selection_uses_outer_test") is not False:
        raise ValueError("baseline selection must exclude outer test")
    baseline_selected = baseline_summary["selected_model"]
    if baseline_selected.get("stage2_enabled") is not False:
        raise ValueError("baseline stage2 must be disabled for exact Stage1 comparison")
    stage1_params = {
        "teacher_recovery_beta": float(
            baseline_selected["teacher_recovery_beta"]
        ),
        "action_beta": float(baseline_selected["action_beta"]),
        "logistic_c": float(baseline_selected["logistic_c"]),
        "probability_threshold": float(
            baseline_selected["probability_threshold"]
        ),
    }
    teacher_frame = _build_teacher_training_frame(
        frame,
        spec=spec,
        teacher_beta=stage1_params["teacher_recovery_beta"],
    )
    development = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    model_cache: dict[tuple[int, float], Any] = {}

    candidate_rows: list[dict[str, Any]] = []
    for variant, fraction, aim_multiplier in spec.execution_candidates:
        current_cost = simulate_cost_aware_policy(
            development,
            spec=spec,
            teacher_frame=teacher_frame,
            c_value=stage1_params["logistic_c"],
            threshold=stage1_params["probability_threshold"],
            action_beta=stage1_params["action_beta"],
            first_action_year=spec.first_intraday_action_year,
            model_cache=model_cache,
            risk_up_fraction=fraction,
            open_cost_bps=spec.transaction_cost_bps,
            intraday_cost_bps=spec.transaction_cost_bps,
            aim_momentum_multiplier=aim_multiplier,
        )
        expected_cost = simulate_cost_aware_policy(
            development,
            spec=spec,
            teacher_frame=teacher_frame,
            c_value=stage1_params["logistic_c"],
            threshold=stage1_params["probability_threshold"],
            action_beta=stage1_params["action_beta"],
            first_action_year=spec.first_intraday_action_year,
            model_cache=model_cache,
            risk_up_fraction=fraction,
            open_cost_bps=spec.expected_open_cost_bps,
            intraday_cost_bps=spec.expected_intraday_cost_bps,
            aim_momentum_multiplier=aim_multiplier,
        )
        candidate_rows.append(
            _candidate_row(
                expected_cost,
                current_cost,
                variant=variant,
                fraction=fraction,
                aim_momentum_multiplier=aim_multiplier,
            )
        )
    model_selection = pd.DataFrame(candidate_rows)
    selected = select_cost_aware_development_candidate(model_selection)
    selected_variant = str(selected["variant"])
    selected_fraction = float(selected["risk_up_fraction"])
    selected_aim_multiplier = float(selected["aim_momentum_multiplier"])

    outer_replays: dict[str, pd.DataFrame] = {}
    outer_comparison_rows: list[dict[str, Any]] = []
    for variant, fraction, aim_multiplier in spec.execution_candidates:
        replay = simulate_cost_aware_policy(
            outer,
            spec=spec,
            teacher_frame=teacher_frame,
            c_value=stage1_params["logistic_c"],
            threshold=stage1_params["probability_threshold"],
            action_beta=stage1_params["action_beta"],
            first_action_year=min(spec.outer_years),
            model_cache=model_cache,
            risk_up_fraction=fraction,
            open_cost_bps=spec.transaction_cost_bps,
            intraday_cost_bps=spec.transaction_cost_bps,
            aim_momentum_multiplier=aim_multiplier,
        )
        outer_replays[variant] = replay
        metrics = _risk_metrics(replay["strategy_return"].to_numpy())
        turnover = _turnover_metrics(replay)
        soxx_wealth = float(np.prod(1.0 + replay["soxx_return"].to_numpy()))
        outer_comparison_rows.append(
            {
                "variant": variant,
                "segment": "outer_diagnostic_reused",
                "selection_uses_outer_test": False,
                "risk_up_fraction": fraction,
                "aim_momentum_multiplier": aim_multiplier,
                "open_cost_bps": spec.transaction_cost_bps,
                "intraday_cost_bps": spec.transaction_cost_bps,
                **metrics,
                "terminal_wealth_ratio_vs_soxx": (
                    metrics["terminal_wealth"] / soxx_wealth
                ),
                "avg_cash_weight_pct": float(
                    100.0 * replay["cash_weight"].mean()
                ),
                "avg_effective_beta": float(
                    (3.0 * replay["soxl_weight"] + replay["soxx_weight"]).mean()
                ),
                **turnover,
            }
        )
    outer_comparison = pd.DataFrame(outer_comparison_rows)
    baseline_variant = spec.execution_candidates[0][0]
    baseline_replay = outer_replays[baseline_variant]
    selected_replay = outer_replays[selected_variant]
    baseline_metrics = _risk_metrics(
        baseline_replay["strategy_return"].to_numpy()
    )
    selected_metrics = _risk_metrics(
        selected_replay["strategy_return"].to_numpy()
    )
    stored_baseline_metrics = baseline_summary[
        "outer_diagnostic_metrics_10bps"
    ]
    baseline_reproduction = {
        key: baseline_metrics[key] - float(stored_baseline_metrics[key])
        for key in (
            "return_pct",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
        )
    }
    baseline_reproduction_pass = all(
        abs(value) <= 1e-8 for value in baseline_reproduction.values()
    )

    selected_expected_outer = simulate_cost_aware_policy(
        outer,
        spec=spec,
        teacher_frame=teacher_frame,
        c_value=stage1_params["logistic_c"],
        threshold=stage1_params["probability_threshold"],
        action_beta=stage1_params["action_beta"],
        first_action_year=min(spec.outer_years),
        model_cache=model_cache,
        risk_up_fraction=selected_fraction,
        open_cost_bps=spec.expected_open_cost_bps,
        intraday_cost_bps=spec.expected_intraday_cost_bps,
        aim_momentum_multiplier=selected_aim_multiplier,
    )
    baseline_expected_outer = simulate_cost_aware_policy(
        outer,
        spec=spec,
        teacher_frame=teacher_frame,
        c_value=stage1_params["logistic_c"],
        threshold=stage1_params["probability_threshold"],
        action_beta=stage1_params["action_beta"],
        first_action_year=min(spec.outer_years),
        model_cache=model_cache,
        risk_up_fraction=1.0,
        open_cost_bps=spec.expected_open_cost_bps,
        intraday_cost_bps=spec.expected_intraday_cost_bps,
        aim_momentum_multiplier=0.0,
    )
    selected_expected_metrics = _risk_metrics(
        selected_expected_outer["strategy_return"].to_numpy()
    )
    baseline_expected_metrics = _risk_metrics(
        baseline_expected_outer["strategy_return"].to_numpy()
    )

    scenarios = (
        ("current_10_open_10_intraday", 10.0, 10.0),
        ("open_heavy_25_open_10_intraday", 25.0, 10.0),
        ("stress_25_open_25_intraday", 25.0, 25.0),
        ("tail_50_open_25_intraday", 50.0, 25.0),
    )
    cost_sensitivity = pd.concat(
        [
            _fixed_cost_sensitivity(
                replay,
                variant=variant,
                scenarios=scenarios,
            )
            for variant, replay in outer_replays.items()
        ],
        ignore_index=True,
    )
    selected_tail_row = cost_sensitivity[
        cost_sensitivity["variant"].eq(selected_variant)
        & cost_sensitivity["scenario"].eq("tail_50_open_25_intraday")
    ].iloc[0]
    selected_tail_metrics = {
        key: float(selected_tail_row[key])
        for key in (
            "terminal_wealth",
            "return_pct",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
        )
    }
    selected_stress_row = cost_sensitivity[
        cost_sensitivity["variant"].eq(selected_variant)
        & cost_sensitivity["scenario"].eq("stress_25_open_25_intraday")
    ].iloc[0]
    selected_stress_metrics = {
        key: float(selected_stress_row[key])
        for key in (
            "terminal_wealth",
            "return_pct",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
        )
    }

    annual_metrics = _annual_metrics(selected_replay)
    annual_relative_pass_count = int(
        annual_metrics["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    annual_risk_pass_count = int(
        annual_metrics.apply(
            lambda row: _passes_risk(row.to_dict()),
            axis=1,
        ).sum()
    )
    baseline_turnover = _turnover_metrics(baseline_replay)
    selected_turnover = _turnover_metrics(selected_replay)
    turnover_reduction_pct = float(
        100.0
        * (
            1.0
            - selected_turnover["total_turnover"]
            / baseline_turnover["total_turnover"]
        )
    )
    from src.workflow.hierarchical_four_asset_policy import (
        _benchmark_efficiency,
    )

    benchmark_efficiency = _benchmark_efficiency(
        selected_replay.assign(soxs_weight=0.0),
        cost_bps=int(spec.transaction_cost_bps),
    )
    selected_soxx_wealth = float(
        np.prod(1.0 + selected_replay["soxx_return"].to_numpy())
    )
    selected_ratio_vs_soxx = (
        selected_metrics["terminal_wealth"] / selected_soxx_wealth
    )
    replacement_gate = bool(
        selected_variant != baseline_variant
        and bool(selected["development_gate_pass"])
        and baseline_reproduction_pass
        and selected_expected_metrics["return_pct"]
        > baseline_expected_metrics["return_pct"]
        and selected_metrics["return_pct"] >= baseline_metrics["return_pct"]
        and turnover_reduction_pct >= spec.minimum_turnover_reduction_pct
        and _passes_risk(selected_metrics)
        and _passes_risk(selected_stress_metrics)
        and annual_relative_pass_count >= 3
        and annual_risk_pass_count >= 3
        and selected_ratio_vs_soxx > 1.0
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["matched_beta_return_maxdd"]
    )
    decision = (
        "promoted_research_candidate_cost_aware_target_tracking"
        if replacement_gate
        else "retain_current_stage1_baseline_cost_aware_variant_not_promoted"
    )

    turnover_analysis = outer_comparison[
        [
            "variant",
            "risk_up_fraction",
            "aim_momentum_multiplier",
            "open_turnover",
            "intraday_turnover",
            "total_turnover",
            "open_turnover_share_pct",
            "open_active_day_count",
            "open_material_day_count",
            "intraday_action_count",
            "avg_cash_weight_pct",
            "avg_effective_beta",
        ]
    ].copy()
    risk_analysis = pd.concat(
        [
            outer_comparison.assign(cost_scenario="current_10_10"),
            cost_sensitivity[
                cost_sensitivity["variant"].eq(
                    selected_variant
                )
            ].assign(
                segment="outer_diagnostic_reused",
                selection_uses_outer_test=False,
                risk_up_fraction=selected_fraction,
                aim_momentum_multiplier=selected_aim_multiplier,
                cost_scenario=lambda value: value["scenario"],
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    recorder_info = _record_qlib_artifacts(
        spec,
        stage1_params=stage1_params,
        selected_variant=selected_variant,
        selected_fraction=selected_fraction,
        selected_aim_multiplier=selected_aim_multiplier,
        model_selection=model_selection,
        outer_comparison=outer_comparison,
        selected_replay=selected_replay,
        teacher_frame=teacher_frame,
        source_audit=source_audit,
        selected_metrics=selected_metrics,
        selected_stress_metrics=selected_stress_metrics,
        selected_tail_metrics=selected_tail_metrics,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_fresh_unseen_test",
        "promotion_scope": "execution_layer_only_against_current_stage1_baseline",
        "assets": ["SOXL", "SOXX", "cash"],
        "provider_uri": str(spec.intraday_provider_uri.resolve()),
        "system_of_record": "qlib",
        "baseline": {
            "experiment": spec.baseline_experiment,
            "recorder_id": baseline_summary["recorder_id"],
            "stage2_enabled": baseline_selected["stage2_enabled"],
            "frozen_stage1_parameters": stage1_params,
            "reproduction_deltas": baseline_reproduction,
            "reproduction_pass": baseline_reproduction_pass,
        },
        "method": {
            "target_model": "frozen current Stage1 direct SOXL/SOXX/cash model",
            "execution_rule": (
                "risk reductions execute fully; risk increases and same-beta "
                "rotations move a fixed fraction toward the target"
            ),
            "intraday_rule": "existing 10:30 Stage1 recovery remains full and unchanged",
            "execution_candidates": [
                {
                    "variant": variant,
                    "risk_up_fraction": fraction,
                    "aim_momentum_multiplier": aim_multiplier,
                }
                for variant, fraction, aim_multiplier in (
                    spec.execution_candidates
                )
            ],
            "selection": (
                "development 2019-2022 windows only, ranked at 25bp open and "
                "10bp intraday; outer excluded"
            ),
            "cost_stress": "fixed 10/10 decision path repriced by open and intraday leg",
        },
        "selected_execution": {
            "variant": selected_variant,
            "risk_up_fraction": selected_fraction,
            "aim_momentum_multiplier": selected_aim_multiplier,
            "selected_from_segment": "development_walk_forward",
            "selection_open_cost_bps": spec.expected_open_cost_bps,
            "selection_intraday_cost_bps": spec.expected_intraday_cost_bps,
        },
        "development_selection": selected.to_dict(),
        "development_selection_gate_any_pass": bool(
            model_selection["development_gate_pass"].any()
        ),
        "outer_diagnostic_metrics_10bps": selected_metrics,
        "baseline_outer_metrics_10bps": baseline_metrics,
        "outer_expected_cost_metrics_25_open_10_intraday": (
            selected_expected_metrics
        ),
        "baseline_expected_cost_metrics_25_open_10_intraday": (
            baseline_expected_metrics
        ),
        "outer_tail_fixed_decision_metrics_50_open_25_intraday": (
            selected_tail_metrics
        ),
        "outer_stress_fixed_decision_metrics_25_open_25_intraday": (
            selected_stress_metrics
        ),
        "outer_effect": {
            "return_change_vs_baseline_10bps_pct_points": (
                selected_metrics["return_pct"] - baseline_metrics["return_pct"]
            ),
            "return_change_vs_baseline_expected_cost_pct_points": (
                selected_expected_metrics["return_pct"]
                - baseline_expected_metrics["return_pct"]
            ),
            "turnover_reduction_pct": turnover_reduction_pct,
            "open_turnover_reduction_pct": float(
                100.0
                * (
                    1.0
                    - selected_turnover["open_turnover"]
                    / baseline_turnover["open_turnover"]
                )
            ),
            "avg_cash_weight_pct": float(
                100.0 * selected_replay["cash_weight"].mean()
            ),
            "avg_effective_beta": float(
                (
                    3.0 * selected_replay["soxl_weight"]
                    + selected_replay["soxx_weight"]
                ).mean()
            ),
            "recovery_action_count": int(
                selected_replay["recovery_action"].sum()
            ),
        },
        "benchmark_comparison_10bps": {
            "terminal_wealth_ratio_vs_soxx": selected_ratio_vs_soxx,
            **benchmark_efficiency,
        },
        "outer_annual_gate_summary": {
            "relative_pass_count": annual_relative_pass_count,
            "risk_pass_count": annual_risk_pass_count,
            "required_year_count": 3,
        },
        "promotion_gates": {
            "replacement_gate_pass": replacement_gate,
            "selected_is_not_baseline": selected_variant != baseline_variant,
            "baseline_reproduction_pass": baseline_reproduction_pass,
            "expected_cost_return_improves": (
                selected_expected_metrics["return_pct"]
                > baseline_expected_metrics["return_pct"]
            ),
            "current_cost_return_not_lower": (
                selected_metrics["return_pct"] >= baseline_metrics["return_pct"]
            ),
            "minimum_turnover_reduction_pct": (
                spec.minimum_turnover_reduction_pct
            ),
            "turnover_reduction_pass": (
                turnover_reduction_pct >= spec.minimum_turnover_reduction_pct
            ),
            "stress_25bps_risk_pass": _passes_risk(
                selected_stress_metrics
            ),
            "tail_50_open_25_intraday_risk_diagnostic_pass": (
                _passes_risk(selected_tail_metrics)
            ),
        },
        "source_audit": source_audit,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": (
                "https://qlib.readthedocs.io/en/latest/component/workflow.html"
            ),
            "strategy": (
                "https://qlib.readthedocs.io/en/latest/component/strategy.html"
            ),
        },
        "risk_gates": {
            "max_drawdown_floor_pct": -30.0,
            "worst_day_floor_pct": -12.0,
            "cvar5_floor_pct": -8.0,
            "worst_rolling_5d_floor_pct": -12.0,
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/positions.csv",
                f"artifacts/{spec.experiment}/strategy_returns.csv",
                f"artifacts/{spec.experiment}/stage1_training_frame.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/outer_comparison_full.csv",
            ],
        },
        **recorder_info,
    }
    _write_card_sidecars(
        spec,
        summary=summary,
        model_selection=model_selection,
        risk_analysis=risk_analysis,
        annual_metrics=annual_metrics,
        cost_sensitivity=cost_sensitivity,
        turnover_analysis=turnover_analysis,
        outer_comparison=outer_comparison,
    )
    return summary


def _record_qlib_artifacts(
    spec: CostAwareTargetTrackingSpec,
    *,
    stage1_params: dict[str, float],
    selected_variant: str,
    selected_fraction: float,
    selected_aim_multiplier: float,
    model_selection: pd.DataFrame,
    outer_comparison: pd.DataFrame,
    selected_replay: pd.DataFrame,
    teacher_frame: pd.DataFrame,
    source_audit: dict[str, Any],
    selected_metrics: dict[str, float],
    selected_stress_metrics: dict[str, float],
    selected_tail_metrics: dict[str, float],
) -> dict[str, Any]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        positions = selected_replay[
            [
                "signal_date",
                "trade_date",
                "desired_open_soxl_weight",
                "desired_open_soxx_weight",
                "desired_open_cash_weight",
                "aim_open_soxl_weight",
                "aim_open_soxx_weight",
                "aim_open_cash_weight",
                "open_soxl_weight",
                "open_soxx_weight",
                "open_cash_weight",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "recovery_probability",
                "recovery_action",
                "open_adjustment_type",
            ]
        ]
        files = {
            "positions.csv": positions,
            "strategy_returns.csv": selected_replay,
            "stage1_training_frame.csv": teacher_frame,
            "model_selection_full.csv": model_selection,
            "outer_comparison_full.csv": outer_comparison,
        }
        for filename, data in files.items():
            data.to_csv(tmp_dir / filename, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                baseline_experiment=spec.baseline_experiment,
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["source_recorder_id"],
                intraday_provider_uri=source_audit["intraday_provider_uri"],
                allocation_assets="SOXL,SOXX,cash",
                selection_uses_outer_test="false",
                selected_execution_variant=selected_variant,
                selected_risk_up_fraction=str(selected_fraction),
                selected_aim_momentum_multiplier=str(
                    selected_aim_multiplier
                ),
                expected_open_cost_bps=str(spec.expected_open_cost_bps),
                expected_intraday_cost_bps=str(
                    spec.expected_intraday_cost_bps
                ),
                stage1_logistic_c=str(stage1_params["logistic_c"]),
                stage1_probability_threshold=str(
                    stage1_params["probability_threshold"]
                ),
                stage1_action_beta=str(stage1_params["action_beta"]),
            )
            R.log_metrics(
                outer_return_pct=selected_metrics["return_pct"],
                outer_max_drawdown_pct=selected_metrics["max_drawdown_pct"],
                stress_25bps_return_pct=selected_stress_metrics["return_pct"],
                stress_25bps_max_drawdown_pct=selected_stress_metrics[
                    "max_drawdown_pct"
                ],
                tail_cost_return_pct=selected_tail_metrics["return_pct"],
                tail_cost_max_drawdown_pct=selected_tail_metrics[
                    "max_drawdown_pct"
                ],
            )
            for filename in files:
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder_info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(
                    Path(recorder.get_local_dir()).resolve()
                ),
            }
    recorder_artifacts = Path(recorder_info["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return recorder_info


def _write_card_sidecars(
    spec: CostAwareTargetTrackingSpec,
    *,
    summary: dict[str, Any],
    model_selection: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    turnover_analysis: pd.DataFrame,
    outer_comparison: pd.DataFrame,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    selected_model = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward",
        "baseline_experiment": spec.baseline_experiment,
        "frozen_stage1_parameters": summary["baseline"][
            "frozen_stage1_parameters"
        ],
        **summary["selected_execution"],
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_model, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_selection.to_csv(card / "model_selection.csv", index=False)
    risk_analysis.to_csv(card / "risk_analysis.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    cost_sensitivity.to_csv(card / "cost_sensitivity.csv", index=False)
    turnover_analysis.to_csv(card / "turnover_analysis.csv", index=False)
    outer_comparison.to_csv(card / "model_comparison.csv", index=False)
    workflow = f"""experiment: {spec.experiment}
system_of_record: qlib
provider_uri: {spec.intraday_provider_uri.resolve()}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
baseline_experiment: {spec.baseline_experiment}
baseline_recorder_id: {summary['baseline']['recorder_id']}
assets: [SOXL, SOXX, cash]
benchmark: SOXX
allocation_output: direct_legal_three_asset_weights
execution_layer: asymmetric_target_tracking
risk_reduction_fraction: 1.0
execution_candidates:
  - {{variant: current_stage1_full_target_baseline, risk_up_fraction: 1.0, aim_momentum_multiplier: 0.0}}
  - {{variant: asymmetric_risk_up_0.5, risk_up_fraction: 0.5, aim_momentum_multiplier: 0.0}}
  - {{variant: asymmetric_risk_up_0.3333, risk_up_fraction: 0.3333333333333333, aim_momentum_multiplier: 0.0}}
  - {{variant: aim_ahead_risk_up_0.5, risk_up_fraction: 0.5, aim_momentum_multiplier: 1.0}}
  - {{variant: aim_ahead_risk_up_0.3333, risk_up_fraction: 0.3333333333333333, aim_momentum_multiplier: 2.0}}
selection_cost: {{open_bps: 25.0, intraday_bps: 10.0}}
selection_uses_outer_test: false
outer_status: reused_diagnostic_not_fresh_unseen_test
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/positions.csv
    - artifacts/{spec.experiment}/strategy_returns.csv
    - artifacts/{spec.experiment}/stage1_training_frame.csv
    - artifacts/{spec.experiment}/model_selection_full.csv
    - artifacts/{spec.experiment}/outer_comparison_full.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    conclusion = f"""# 中文结论

- 做了什么：冻结当前 Stage1 的原模型、21 个特征、回撤状态机、10:30 恢复模型和参数，只比较目标仓位执行速度。降风险仍一次完成；加风险和同 beta 轮换比较 100%、50%、33% 直接靠近目标，以及按动态交易论文思路用目标变化趋势构造 aim-ahead 后再执行 50%/33%。
- 基准：{spec.baseline_experiment}，recorder={summary['baseline']['recorder_id']}，Stage2 继续关闭。新实现对基准的 return 差={summary['baseline']['reproduction_deltas']['return_pct']:.10f} 个百分点，复现门={summary['baseline']['reproduction_pass']}。
- 选择：先要求当前 10/10 成本下通过 2019–2022 完整开发门，再按开盘 25bp、盘中 10bp 收益排序；2023–2026 outer 没有参与速度选择。选中 {summary['selected_execution']['variant']}，risk_up_fraction={summary['selected_execution']['risk_up_fraction']:.6f}，aim multiplier={summary['selected_execution']['aim_momentum_multiplier']:.2f}；至少一个候选通过完整开发门={summary['development_selection_gate_any_pass']}。
- outer 10/10：选中方案 return={summary['outer_diagnostic_metrics_10bps']['return_pct']:.4f}%，MaxDD={summary['outer_diagnostic_metrics_10bps']['max_drawdown_pct']:.4f}%，worst day={summary['outer_diagnostic_metrics_10bps']['worst_day_pct']:.4f}%，CVaR5={summary['outer_diagnostic_metrics_10bps']['cvar5_pct']:.4f}%，worst 5d={summary['outer_diagnostic_metrics_10bps']['worst_rolling_5d_pct']:.4f}%。
- 相对当前基准：10/10 收益变化={summary['outer_effect']['return_change_vs_baseline_10bps_pct_points']:.4f} 个百分点；开盘 25bp/盘中 10bp 的动态成本回放收益变化={summary['outer_effect']['return_change_vs_baseline_expected_cost_pct_points']:.4f} 个百分点；总换手下降={summary['outer_effect']['turnover_reduction_pct']:.4f}%，开盘换手下降={summary['outer_effect']['open_turnover_reduction_pct']:.4f}%。
- 正式压力：冻结 10/10 决策后按开盘/盘中各 25bp 重估，return={summary['outer_stress_fixed_decision_metrics_25_open_25_intraday']['return_pct']:.4f}%，MaxDD={summary['outer_stress_fixed_decision_metrics_25_open_25_intraday']['max_drawdown_pct']:.4f}%，worst 5d={summary['outer_stress_fixed_decision_metrics_25_open_25_intraday']['worst_rolling_5d_pct']:.4f}%。额外尾部诊断按开盘 50bp、盘中 25bp 重估，但不作为晋升门。
- 现金与动作：平均现金={summary['outer_effect']['avg_cash_weight_pct']:.4f}%，平均有效 beta={summary['outer_effect']['avg_effective_beta']:.4f}，10:30 恢复动作={summary['outer_effect']['recovery_action_count']} 次。
- 是否替换：{summary['decision']}。只有非基准速度同时提高 10/10 与主成本收益、换手至少下降 {spec.minimum_turnover_reduction_pct:.0f}% 且通过风险/年度/相对 SOXX 门，才替换当前基准。
- 风险：outer 已被前序研究反复查看，因此这里只是 reused diagnostic，不是新的未见样本；成本仍是线性 bps，尚未覆盖开盘价差、冲击、排队和成交不足。
- qlib recorder：{summary['recorder_id']}
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    print(
        json.dumps(
            run_cost_aware_target_tracking_card(),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    )
