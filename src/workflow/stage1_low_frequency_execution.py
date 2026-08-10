from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import (
    FEATURE_COLUMNS,
    IntradayRecoveryDirectAllocationSpec,
    _advance_state,
    _base_cap,
    _build_teacher_training_frame,
    _development_stability,
    _feature_row,
    _fit_recovery_model,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    mark_to_market_weights,
    project_legal_weights,
)


EXPERIMENT = "stage1_low_frequency_execution_v1"
SOURCE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
SOURCE_RECORDER_ID = "f6b09330175e41719886d58ee2d11bec"


@dataclass(frozen=True)
class Stage1LowFrequencyExecutionSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    candidate_bands: tuple[float, ...] = (
        0.0,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
    )
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    minimum_trade_day_reduction: float = 0.35
    minimum_terminal_wealth_ratio: float = 0.99
    maximum_drawdown_worsening_pct_points: float = 1.0

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def apply_materiality_band(
    desired_weights: np.ndarray,
    current_weights: np.ndarray,
    *,
    beta_cap: float,
    minimum_rebalance_turnover: float,
    minimum_cash_weight: float,
    maximum_soxl_weight: float,
) -> tuple[np.ndarray, bool]:
    """Suppress a small legal basket rebalance while never blocking de-risking."""

    desired = np.asarray(desired_weights, dtype="float64")
    current = np.asarray(current_weights, dtype="float64")
    if desired.shape != (3,) or current.shape != (3,):
        raise ValueError("desired and current weights must have shape (3,)")
    if minimum_rebalance_turnover <= 0.0:
        return desired.copy(), False
    current_beta = 3.0 * current[0] + current[1]
    legal_current = (
        current.sum() > 0.0
        and current.min() >= -1e-12
        and current[0] <= maximum_soxl_weight + 1e-12
        and current[2] >= minimum_cash_weight - 1e-12
        and current_beta <= beta_cap + 1e-12
    )
    risky_turnover = float(np.abs(desired[:2] - current[:2]).sum())
    suppressed = (
        legal_current
        and risky_turnover > 1e-12
        and risky_turnover < float(minimum_rebalance_turnover)
    )
    return (current.copy(), True) if suppressed else (desired.copy(), False)


def infer_holdings_aware_source_actions(
    target_weights: np.ndarray,
    asset_returns: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Recover whether a source target requests an order versus passive drift."""

    targets = np.asarray(target_weights, dtype="float64")
    returns = np.asarray(asset_returns, dtype="float64")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_weights must have shape (rows, 3)")
    if returns.shape != (len(targets), 2):
        raise ValueError("asset_returns must have shape (rows, 2)")
    actions = np.zeros(len(targets), dtype=bool)
    current = np.array([0.0, 0.0, 1.0], dtype="float64")
    for index, (target, daily_returns) in enumerate(
        zip(targets, returns, strict=True)
    ):
        actions[index] = (
            np.abs(target[:2] - current[:2]).sum() > tolerance
        )
        marked_value = target * np.array(
            [1.0 + daily_returns[0], 1.0 + daily_returns[1], 1.0],
            dtype="float64",
        )
        current = marked_value / max(marked_value.sum(), 1e-12)
    return actions


def simulate_low_frequency_stage1(
    frame: pd.DataFrame,
    *,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    teacher_frame: pd.DataFrame,
    selected_model: dict[str, Any],
    first_action_year: int,
    minimum_rebalance_turnover: float,
    cost_bps: float,
    source_action_flags: pd.Series | None = None,
) -> pd.DataFrame:
    models: dict[int, Any] = {}
    model_cache: dict[tuple[int, float], Any] = {}
    c_value = float(selected_model["logistic_c"])
    for year in sorted(frame["signal_date"].dt.year.unique()):
        key = (int(year), c_value)
        if key not in model_cache:
            model_cache[key] = _fit_recovery_model(
                teacher_frame,
                year=int(year),
                c_value=c_value,
            )
        models[int(year)] = model_cache[key]

    equity = peak = 1.0
    previous = np.zeros(3, dtype="float64")
    risk_state = risk_clock = 0
    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        pretrade = previous.copy()
        drawdown = max(0.0, 1.0 - equity / peak)
        risk_state, risk_clock, q = _advance_state(
            risk_state,
            risk_clock,
            drawdown,
            recovery_spec,
        )
        cap, raw_beta = _base_cap(
            row,
            q=q,
            drawdown=drawdown,
            history=history,
            spec=recovery_spec,
        )
        desired_open = project_legal_weights(
            row.soxl_weight,
            row.soxx_weight,
            cap,
            minimum_cash_weight=recovery_spec.minimum_cash_weight,
            maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
        )
        source_action = (
            True
            if source_action_flags is None
            else bool(source_action_flags.loc[row.signal_date])
        )
        effective_band = (
            minimum_rebalance_turnover if source_action else float("inf")
        )
        open_weights, open_suppressed = apply_materiality_band(
            desired_open,
            previous,
            beta_cap=cap,
            minimum_rebalance_turnover=effective_band,
            minimum_cash_weight=recovery_spec.minimum_cash_weight,
            maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
        )
        open_delta = open_weights - previous
        open_turnover = float(np.abs(open_delta[:2]).sum())
        open_trade = open_turnover > 1e-12
        open_order_legs = int(np.count_nonzero(np.abs(open_delta[:2]) > 1e-12))

        target_beta = min(raw_beta, float(selected_model["action_beta"]))
        model = models.get(row.signal_date.year)
        eligible = (
            row.signal_date.year >= first_action_year
            and model is not None
            and drawdown >= recovery_spec.recovery_eligibility_drawdown
            and target_beta > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
        )
        probability = np.nan
        recovery_action = False
        if eligible:
            feature = pd.DataFrame(
                [
                    _feature_row(
                        row,
                        drawdown=drawdown,
                        cap=cap,
                        raw_beta=raw_beta,
                        state=risk_state,
                        history=history,
                    )
                ]
            )
            probability = float(
                model.predict_proba(feature.loc[:, FEATURE_COLUMNS])[0, 1]
            )
            recovery_action = probability >= float(
                selected_model["probability_threshold"]
            )

        recovery_turnover = 0.0
        recovery_order_legs = 0
        if recovery_action:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                target_beta,
                minimum_cash_weight=recovery_spec.minimum_cash_weight,
                maximum_soxl_weight=recovery_spec.maximum_soxl_weight,
            )
            early = np.array(
                [row.early_ret_soxl, row.early_ret_soxx, 0.0],
                dtype="float64",
            )
            value_at_cut = 1.0 + open_weights @ early
            drifted = mark_to_market_weights(open_weights, early)
            recovery_delta = close_weights - drifted
            recovery_turnover = float(np.abs(recovery_delta[:2]).sum())
            recovery_order_legs = int(
                np.count_nonzero(np.abs(recovery_delta[:2]) > 1e-12)
            )
            post = np.array(
                [
                    (1.0 + row.soxl_return) / (1.0 + row.early_ret_soxl) - 1.0,
                    (1.0 + row.soxx_return) / (1.0 + row.early_ret_soxx) - 1.0,
                    0.0,
                ],
                dtype="float64",
            )
            gross_return = value_at_cut * (1.0 + close_weights @ post) - 1.0
            ending_weights = mark_to_market_weights(close_weights, post)
        else:
            close_weights = open_weights.copy()
            asset_returns = np.array(
                [row.soxl_return, row.soxx_return, 0.0],
                dtype="float64",
            )
            gross_return = float(open_weights @ asset_returns)
            ending_weights = mark_to_market_weights(open_weights, asset_returns)

        turnover = open_turnover + recovery_turnover
        strategy_return = float(gross_return - turnover * cost_bps / 10_000.0)
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = ending_weights
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
                "pretrade_soxl_weight": pretrade[0],
                "pretrade_soxx_weight": pretrade[1],
                "desired_open_soxl_weight": desired_open[0],
                "desired_open_soxx_weight": desired_open[1],
                "open_rebalance_suppressed": open_suppressed,
                "source_action": source_action,
                "open_trade": open_trade,
                "recovery_action": recovery_action,
                "recovery_probability": probability,
                "daily_trade_event": open_trade or recovery_turnover > 1e-12,
                "open_order_legs": open_order_legs,
                "recovery_order_legs": recovery_order_legs,
                "order_legs": open_order_legs + recovery_order_legs,
                "open_turnover": open_turnover,
                "recovery_turnover": recovery_turnover,
                "turnover": turnover,
                "gross_return": gross_return,
                "strategy_return": strategy_return,
                "cost_bps": cost_bps,
            }
        )
    return pd.DataFrame(rows)


def trade_frequency_metrics(replay: pd.DataFrame) -> dict[str, float | int]:
    return {
        "session_count": int(len(replay)),
        "open_trade_days": int(replay["open_trade"].sum()),
        "recovery_trade_days": int(
            (replay["recovery_turnover"] > 1e-12).sum()
        ),
        "daily_trade_events": int(replay["daily_trade_event"].sum()),
        "order_legs": int(replay["order_legs"].sum()),
        "open_rebalance_suppressed": int(
            replay["open_rebalance_suppressed"].sum()
        ),
        "turnover": float(replay["turnover"].sum()),
    }


def _candidate_row(
    replay: pd.DataFrame,
    *,
    baseline: pd.DataFrame,
    band: float,
    spec: Stage1LowFrequencyExecutionSpec,
) -> dict[str, Any]:
    metrics = _risk_metrics(replay["strategy_return"].to_numpy())
    baseline_metrics = _risk_metrics(baseline["strategy_return"].to_numpy())
    frequency = trade_frequency_metrics(replay)
    baseline_frequency = trade_frequency_metrics(baseline)
    trade_day_reduction = 1.0 - (
        frequency["daily_trade_events"] / baseline_frequency["daily_trade_events"]
    )
    order_leg_reduction = 1.0 - (
        frequency["order_legs"] / baseline_frequency["order_legs"]
    )
    wealth_ratio = metrics["terminal_wealth"] / baseline_metrics["terminal_wealth"]
    drawdown_change = (
        metrics["max_drawdown_pct"] - baseline_metrics["max_drawdown_pct"]
    )
    gate = bool(
        band > 0.0
        and _passes_risk(metrics)
        and trade_day_reduction >= spec.minimum_trade_day_reduction
        and wealth_ratio >= spec.minimum_terminal_wealth_ratio
        and drawdown_change >= -spec.maximum_drawdown_worsening_pct_points
    )
    return {
        "segment": "development_walk_forward",
        "selection_uses_outer_test": False,
        "minimum_rebalance_turnover": band,
        **metrics,
        **frequency,
        "trade_day_reduction_vs_baseline": trade_day_reduction,
        "order_leg_reduction_vs_baseline": order_leg_reduction,
        "terminal_wealth_ratio_vs_baseline": wealth_ratio,
        "max_drawdown_change_pct_points": drawdown_change,
        "selection_gate_pass": gate,
    }


def select_low_frequency_candidate(candidates: pd.DataFrame) -> pd.Series:
    eligible = candidates[
        candidates["selection_gate_pass"].astype(bool)
        & ~candidates["selection_uses_outer_test"].astype(bool)
    ].copy()
    if eligible.empty:
        raise ValueError("no low-frequency candidate passes development-only gates")
    return eligible.sort_values(
        [
            "trade_day_reduction_vs_baseline",
            "terminal_wealth_ratio_vs_baseline",
            "minimum_rebalance_turnover",
        ],
        ascending=[False, False, True],
    ).iloc[0]


def _record_artifacts(
    spec: Stage1LowFrequencyExecutionSpec,
    *,
    summary_metrics: dict[str, Any],
    model_selection: pd.DataFrame,
    development_replay: pd.DataFrame,
    outer_replay: pd.DataFrame,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        model_selection.to_csv(tmp_dir / "model_selection_full.csv", index=False)
        development_replay.to_csv(tmp_dir / "development_replay.csv", index=False)
        outer_replay.to_csv(tmp_dir / "outer_replay.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=spec.source_recorder_id,
                selection_uses_outer_test="false",
                execution_policy="portfolio_level_materiality_band",
                required_risk_reduction_bypass="true",
            )
            R.log_metrics(**summary_metrics)
            for filename in (
                "model_selection_full.csv",
                "development_replay.csv",
                "outer_replay.csv",
            ):
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(recorder.get_local_dir())
                .resolve()
                .as_posix(),
            }
    recorder_artifacts = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return result


def run_stage1_low_frequency_execution_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    spec = Stage1LowFrequencyExecutionSpec(root=repo_root)
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: pinned Stage1 recorder drifted")
    if source_summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: source Stage1 selection used outer test")
    selected_model = dict(source_summary["selected_model"])
    selected_model["stage2_enabled"] = False

    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=repo_root)
    frame, source_audit = _load_source_frame(recovery_spec)
    teacher_frame = _build_teacher_training_frame(
        frame,
        spec=recovery_spec,
        teacher_beta=float(selected_model["teacher_recovery_beta"]),
    )
    development = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()

    development_replays: dict[float, pd.DataFrame] = {}
    for band in spec.candidate_bands:
        development_replays[band] = simulate_low_frequency_stage1(
            development,
            recovery_spec=recovery_spec,
            teacher_frame=teacher_frame,
            selected_model=selected_model,
            first_action_year=recovery_spec.first_intraday_action_year,
            minimum_rebalance_turnover=band,
            cost_bps=spec.transaction_cost_bps,
        )
    baseline_development = development_replays[0.0]
    model_selection = pd.DataFrame(
        [
            _candidate_row(
                replay,
                baseline=baseline_development,
                band=band,
                spec=spec,
            )
            for band, replay in development_replays.items()
        ]
    )
    try:
        selected = select_low_frequency_candidate(model_selection)
    except ValueError:
        selected = None
    diagnostic = model_selection[
        model_selection["minimum_rebalance_turnover"].gt(0.0)
    ].sort_values(
        ["terminal_wealth_ratio_vs_baseline", "trade_day_reduction_vs_baseline"],
        ascending=[False, False],
    ).iloc[0]
    diagnostic_band = float(diagnostic["minimum_rebalance_turnover"])
    diagnostic_development = development_replays[diagnostic_band]
    baseline_outer = simulate_low_frequency_stage1(
        outer,
        recovery_spec=recovery_spec,
        teacher_frame=teacher_frame,
        selected_model=selected_model,
        first_action_year=min(recovery_spec.outer_years),
        minimum_rebalance_turnover=0.0,
        cost_bps=spec.transaction_cost_bps,
    )
    diagnostic_outer = simulate_low_frequency_stage1(
        outer,
        recovery_spec=recovery_spec,
        teacher_frame=teacher_frame,
        selected_model=selected_model,
        first_action_year=min(recovery_spec.outer_years),
        minimum_rebalance_turnover=diagnostic_band,
        cost_bps=spec.transaction_cost_bps,
    )
    stress_outer = diagnostic_outer.copy()
    stress_outer["strategy_return"] = (
        stress_outer["gross_return"]
        - stress_outer["turnover"] * spec.stress_cost_bps / 10_000.0
    )

    baseline_dev_metrics = _risk_metrics(
        baseline_development["strategy_return"].to_numpy()
    )
    diagnostic_dev_metrics = _risk_metrics(
        diagnostic_development["strategy_return"].to_numpy()
    )
    baseline_outer_metrics = _risk_metrics(
        baseline_outer["strategy_return"].to_numpy()
    )
    diagnostic_outer_metrics = _risk_metrics(
        diagnostic_outer["strategy_return"].to_numpy()
    )
    stress_outer_metrics = _risk_metrics(
        stress_outer["strategy_return"].to_numpy()
    )
    baseline_dev_frequency = trade_frequency_metrics(baseline_development)
    diagnostic_dev_frequency = trade_frequency_metrics(diagnostic_development)
    baseline_outer_frequency = trade_frequency_metrics(baseline_outer)
    diagnostic_outer_frequency = trade_frequency_metrics(diagnostic_outer)
    outer_trade_reduction = 1.0 - (
        diagnostic_outer_frequency["daily_trade_events"]
        / baseline_outer_frequency["daily_trade_events"]
    )
    outer_wealth_ratio = (
        diagnostic_outer_metrics["terminal_wealth"]
        / baseline_outer_metrics["terminal_wealth"]
    )
    outer_validation_pass = bool(
        selected is not None
        and _passes_risk(diagnostic_outer_metrics)
        and outer_trade_reduction >= spec.minimum_trade_day_reduction
        and outer_wealth_ratio >= spec.minimum_terminal_wealth_ratio
    )
    decision = (
        "promoted_futu_simulate_low_frequency_execution_policy"
        if outer_validation_pass
        else "not_promoted_no_low_frequency_candidate_preserves_performance"
    )

    recorder = _record_artifacts(
        spec,
        summary_metrics={
            "diagnostic_band": diagnostic_band,
            "development_trade_day_reduction": float(
                diagnostic["trade_day_reduction_vs_baseline"]
            ),
            "development_terminal_wealth_ratio": float(
                diagnostic["terminal_wealth_ratio_vs_baseline"]
            ),
            "outer_trade_day_reduction": outer_trade_reduction,
            "outer_terminal_wealth_ratio": outer_wealth_ratio,
        },
        model_selection=model_selection,
        development_replay=diagnostic_development,
        outer_replay=diagnostic_outer,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "stage1_low_frequency_execution_card",
        "decision": decision,
        "selection_uses_outer_test": False,
        "outer_status": "validation_only_not_used_for_threshold_selection",
        "source_experiment": spec.source_experiment,
        "source_recorder_id": spec.source_recorder_id,
        "model_conclusion": (
            "Stage1 is not operationally low-frequency: its reported recovery action "
            "count omitted daily open rebalances, and no tested no-trade band met both "
            "frequency reduction and performance preservation gates. The prior V2 also "
            "added no actions and no incremental alpha."
        ),
        "selected_execution_policy": None,
        "diagnostic_execution_policy": {
            "minimum_rebalance_turnover": diagnostic_band,
            "measurement": "L1 distance of SOXL and SOXX weights",
            "deployment": "rejected_not_wired_to_simulator",
            "required_risk_reduction_bypass": True,
            "recovery_rule": "10:31 executes only when recovery_action is true",
            "broker_scope": "Futu SIMULATE only",
        },
        "selection_gates": {
            "minimum_trade_day_reduction": spec.minimum_trade_day_reduction,
            "minimum_terminal_wealth_ratio": spec.minimum_terminal_wealth_ratio,
            "maximum_drawdown_worsening_pct_points": (
                spec.maximum_drawdown_worsening_pct_points
            ),
            "selected_development_gate_pass": selected is not None,
            "outer_validation_pass": outer_validation_pass,
        },
        "development": {
            "baseline_metrics_10bps": baseline_dev_metrics,
            "diagnostic_metrics_10bps": diagnostic_dev_metrics,
            "baseline_trade_frequency": baseline_dev_frequency,
            "diagnostic_trade_frequency": diagnostic_dev_frequency,
            "trade_day_reduction": float(
                diagnostic["trade_day_reduction_vs_baseline"]
            ),
            "order_leg_reduction": float(
                diagnostic["order_leg_reduction_vs_baseline"]
            ),
            "terminal_wealth_ratio": float(
                diagnostic["terminal_wealth_ratio_vs_baseline"]
            ),
            "stability": _development_stability(diagnostic_development),
        },
        "outer_validation": {
            "baseline_metrics_10bps": baseline_outer_metrics,
            "diagnostic_metrics_10bps": diagnostic_outer_metrics,
            "stress_metrics_25bps": stress_outer_metrics,
            "baseline_trade_frequency": baseline_outer_frequency,
            "diagnostic_trade_frequency": diagnostic_outer_frequency,
            "trade_day_reduction": outer_trade_reduction,
            "terminal_wealth_ratio": outer_wealth_ratio,
        },
        "source_audit": source_audit,
        "qlib_docs": {
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "backtest": "https://qlib.readthedocs.io/en/latest/component/strategy.html",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/development_replay.csv",
                f"artifacts/{spec.experiment}/outer_replay.csv",
            ],
        },
        **recorder,
    }
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "source_experiment": spec.source_experiment,
                "source_recorder_id": spec.source_recorder_id,
                "selection_uses_outer_test": False,
                "predictive_model_changed": False,
                "minimum_rebalance_turnover": None,
                "diagnostic_minimum_rebalance_turnover": diagnostic_band,
                "required_risk_reduction_bypass": True,
                "deployment": "rejected_not_wired_to_simulator",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model_selection.to_csv(spec.card_dir / "model_selection.csv", index=False)
    pd.DataFrame(
        [
            {
                "segment": "development_walk_forward",
                "variant": "baseline",
                **baseline_dev_metrics,
                **baseline_dev_frequency,
            },
            {
                "segment": "development_walk_forward",
                "variant": f"diagnostic_materiality_{diagnostic_band:.4f}",
                **diagnostic_dev_metrics,
                **diagnostic_dev_frequency,
            },
            {
                "segment": "outer_validation",
                "variant": "baseline",
                **baseline_outer_metrics,
                **baseline_outer_frequency,
            },
            {
                "segment": "outer_validation",
                "variant": f"diagnostic_materiality_{diagnostic_band:.4f}",
                **diagnostic_outer_metrics,
                **diagnostic_outer_frequency,
            },
        ]
    ).to_csv(spec.card_dir / "backtest_benchmark_comparison.csv", index=False)
    (spec.card_dir / "workflow.yaml").write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                f"source_experiment: {spec.source_experiment}",
                f"source_recorder_id: {spec.source_recorder_id}",
                f"decision: {decision}",
                "selection_uses_outer_test: false",
                "execution_policy:",
                "  selected: null",
                f"  diagnostic_minimum_rebalance_turnover: {diagnostic_band}",
                "  deployment: rejected_not_wired_to_simulator",
                "  measurement: L1 SOXL and SOXX weight distance",
                "  required_risk_reduction_bypass: true",
                "  scope: Futu SIMULATE open and manual-start basket",
                "  recovery: execute only when recovery_action is true",
                "gates:",
                f"  minimum_trade_day_reduction: {spec.minimum_trade_day_reduction}",
                f"  minimum_terminal_wealth_ratio: {spec.minimum_terminal_wealth_ratio}",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  workflow_config: workflow.yaml",
                "  recorder_only_artifacts:",
                f"    - artifacts/{spec.experiment}/model_selection_full.csv",
                f"    - artifacts/{spec.experiment}/development_replay.csv",
                f"    - artifacts/{spec.experiment}/outer_replay.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (spec.card_dir / "conclusion.md").write_text(
        "\n".join(
            [
                "# 中文结论",
                "",
                "- 先前 V2 的 0 次覆盖不是模型提升，只能说明它不应替换 Stage1；本卡不晋升新的预测模型。",
                f"- 原统计只报告恢复动作，漏掉每日开盘调仓；开发期真实开盘调仓日为 {baseline_dev_frequency['open_trade_days']}/{baseline_dev_frequency['session_count']}。",
                f"- 开发期没有候选同时达到交易日减少 {spec.minimum_trade_day_reduction:.0%} 与终值保留 {spec.minimum_terminal_wealth_ratio:.0%}；因此不晋升、不接入模拟盘。",
                f"- 最小损伤诊断门槛仅 {diagnostic_band:.2%}：交易日只减少 {float(diagnostic['trade_day_reduction_vs_baseline']):.2%}，订单腿减少 {float(diagnostic['order_leg_reduction_vs_baseline']):.2%}，终值保留 {float(diagnostic['terminal_wealth_ratio_vs_baseline']):.2%}，未达到降频目标。",
                f"- 外层只做诊断，不参与选参：交易日减少 {outer_trade_reduction:.2%}，终值保留 {outer_wealth_ratio:.2%}，晋升验证={str(outer_validation_pass).lower()}。",
                "- 当前持仓违反风控上限时，强制降风险不受 band 限制；普通小幅增减成对跳过，避免单向降仓棘轮。10:31 仍仅在 recovery_action=true 时执行。",
                "- 结论是当前 Stage1 不适合进一步机械降频；模拟盘只保留已验证的每日一次开盘篮子，以及 recovery_action=true 时的稀疏恢复动作。",
                f"- qlib recorder：{recorder['recorder_id']}",
                "- 不构成买卖建议。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "Stage1LowFrequencyExecutionSpec",
    "apply_materiality_band",
    "run_stage1_low_frequency_execution_card",
    "select_low_frequency_candidate",
    "simulate_low_frequency_stage1",
    "trade_frequency_metrics",
]
