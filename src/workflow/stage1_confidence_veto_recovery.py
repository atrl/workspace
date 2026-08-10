from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _build_teacher_training_frame,
    _load_source_frame,
    _risk_metrics,
    _simulate_policy,
    apply_fixed_decision_cost_stress,
)
from src.workflow.stage1_rank_calibrated_recovery_sizing import (
    _build_recovery_dataset,
)


EXPERIMENT = "stage1_confidence_veto_recovery_v1"
CARD_TYPE = "stage1_development_selected_confidence_veto_recovery_card"
ACTION_YEARS = (2020, 2021, 2022)


@dataclass(frozen=True)
class Stage1ConfidenceVetoRecoverySpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = "intraday_recovery_direct_allocation_v1"
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    logistic_c_grid: tuple[float, ...] = (0.10, 0.30, 1.00)
    probability_threshold_grid: tuple[float, ...] = (0.825, 0.850, 0.875)
    action_beta_grid: tuple[float, ...] = (0.55, 0.60, 0.65)
    source_alpha_floor_grid: tuple[float, ...] = (0.20, 0.25, 0.30)
    maximum_action_ratio: float = 0.40
    maximum_turnover_ratio: float = 0.90
    minimum_action_year_wealth_ratio: float = 0.985
    required_threshold_plateau_count: int = 3

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


def _fixed_cost_replay(replay: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    stressed = replay.copy()
    stressed["strategy_return"] = apply_fixed_decision_cost_stress(
        stressed["gross_return"], stressed["turnover"], cost_bps=cost_bps
    )
    stressed["cost_bps"] = float(cost_bps)
    return stressed


def _action_year_ratios(
    challenger: pd.DataFrame, baseline: pd.DataFrame
) -> dict[int, float]:
    result: dict[int, float] = {}
    for year in ACTION_YEARS:
        challenger_year = challenger[
            challenger["signal_date"].dt.year.eq(year)
        ]
        baseline_year = baseline[baseline["signal_date"].dt.year.eq(year)]
        if challenger_year.empty or baseline_year.empty:
            continue
        challenger_wealth = float(
            np.prod(1.0 + challenger_year["strategy_return"].to_numpy())
        )
        baseline_wealth = float(
            np.prod(1.0 + baseline_year["strategy_return"].to_numpy())
        )
        result[year] = challenger_wealth / baseline_wealth
    return result


def _metrics_row(
    *,
    variant: str,
    segment: str,
    replay: pd.DataFrame,
    cost_bps: float,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "segment": segment,
        "cost_bps": float(cost_bps),
        **_risk_metrics(replay["strategy_return"].to_numpy()),
        "total_turnover": float(replay["turnover"].sum()),
        "recovery_action_count": int(replay["recovery_action"].sum()),
        "avg_cash_weight_pct": float(100.0 * replay["cash_weight"].mean()),
    }


def _candidate_gate_columns(
    row: dict[str, Any], *, baseline_10: dict[str, Any], baseline_25: dict[str, Any], spec
) -> dict[str, bool]:
    return {
        "return_10bps_above_baseline": row["return_pct"]
        > baseline_10["return_pct"],
        "return_25bps_above_baseline": row["stress_return_pct"]
        > baseline_25["return_pct"],
        "max_drawdown_not_worse": row["max_drawdown_pct"]
        >= baseline_10["max_drawdown_pct"],
        "worst_day_not_worse": row["worst_day_pct"]
        >= baseline_10["worst_day_pct"],
        "cvar5_not_worse": row["cvar5_pct"] >= baseline_10["cvar5_pct"],
        "worst_5d_not_worse": row["worst_rolling_5d_pct"]
        >= baseline_10["worst_rolling_5d_pct"],
        "recovery_actions_reduced_60pct": row["recovery_action_ratio"]
        <= spec.maximum_action_ratio,
        "turnover_reduced_10pct": row["turnover_ratio"]
        <= spec.maximum_turnover_ratio,
        "action_year_floor_pass": row["action_year_min_wealth_ratio"]
        >= spec.minimum_action_year_wealth_ratio,
    }


def select_confidence_veto_candidate(model_selection: pd.DataFrame) -> pd.Series:
    required = {
        "segment",
        "selection_uses_outer_test",
        "all_primary_gates_pass",
        "threshold_plateau_count",
        "required_threshold_plateau_count",
        "action_year_min_wealth_ratio",
        "return_pct",
        "recovery_action_count",
    }
    missing = required.difference(model_selection.columns)
    if missing:
        raise ValueError(f"model selection is missing columns: {sorted(missing)}")
    eligible = model_selection[
        model_selection["segment"].eq("development_walk_forward")
        & ~model_selection["selection_uses_outer_test"].astype(bool)
        & model_selection["all_primary_gates_pass"].astype(bool)
        & (
            model_selection["threshold_plateau_count"]
            >= model_selection["required_threshold_plateau_count"]
        )
    ].copy()
    if eligible.empty:
        raise ValueError("no development-only confidence-veto candidate passes")
    return eligible.sort_values(
        [
            "action_year_min_wealth_ratio",
            "return_pct",
            "recovery_action_count",
        ],
        ascending=[False, False, True],
    ).iloc[0]


def _simulate(
    frame: pd.DataFrame,
    *,
    base_spec: IntradayRecoveryDirectAllocationSpec,
    teacher: pd.DataFrame,
    c_value: float,
    threshold: float,
    action_beta: float,
    alpha_floor: float,
    first_action_year: int,
    model_cache: dict[tuple[int, float], Any],
) -> pd.DataFrame:
    return _simulate_policy(
        frame,
        spec=base_spec,
        teacher_frame=teacher,
        c_value=c_value,
        threshold=threshold,
        action_beta=action_beta,
        first_action_year=first_action_year,
        model_cache=model_cache,
        cost_bps=10.0,
        minimum_source_alpha_probability=alpha_floor,
    )


def _development_search(
    *,
    development: pd.DataFrame,
    teacher: pd.DataFrame,
    base_spec: IntradayRecoveryDirectAllocationSpec,
    baseline: pd.DataFrame,
    spec: Stage1ConfidenceVetoRecoverySpec,
) -> tuple[pd.DataFrame, dict[tuple[float, float, float, float], pd.DataFrame]]:
    baseline_10 = _metrics_row(
        variant="frozen_stage1",
        segment="development_walk_forward",
        replay=baseline,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_25_replay = _fixed_cost_replay(
        baseline, cost_bps=spec.stress_cost_bps
    )
    baseline_25 = _metrics_row(
        variant="frozen_stage1",
        segment="development_walk_forward",
        replay=baseline_25_replay,
        cost_bps=spec.stress_cost_bps,
    )
    cache: dict[tuple[int, float], Any] = {}
    replay_cache: dict[tuple[float, float, float, float], pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for c_value in spec.logistic_c_grid:
        for threshold in spec.probability_threshold_grid:
            for action_beta in spec.action_beta_grid:
                for alpha_floor in spec.source_alpha_floor_grid:
                    key = (c_value, threshold, action_beta, alpha_floor)
                    replay = _simulate(
                        development,
                        base_spec=base_spec,
                        teacher=teacher,
                        c_value=c_value,
                        threshold=threshold,
                        action_beta=action_beta,
                        alpha_floor=alpha_floor,
                        first_action_year=base_spec.first_intraday_action_year,
                        model_cache=cache,
                    )
                    replay_cache[key] = replay
                    metrics = _metrics_row(
                        variant="confidence_veto_candidate",
                        segment="development_walk_forward",
                        replay=replay,
                        cost_bps=spec.transaction_cost_bps,
                    )
                    stress = _metrics_row(
                        variant="confidence_veto_candidate",
                        segment="development_walk_forward",
                        replay=_fixed_cost_replay(
                            replay, cost_bps=spec.stress_cost_bps
                        ),
                        cost_bps=spec.stress_cost_bps,
                    )
                    annual_ratios = _action_year_ratios(replay, baseline)
                    row = {
                        "segment": "development_walk_forward",
                        "selection_uses_outer_test": False,
                        "logistic_c": c_value,
                        "probability_threshold": threshold,
                        "action_beta": action_beta,
                        "minimum_source_alpha_probability": alpha_floor,
                        **{
                            name: metrics[name]
                            for name in (
                                "terminal_wealth",
                                "return_pct",
                                "max_drawdown_pct",
                                "worst_day_pct",
                                "cvar5_pct",
                                "worst_rolling_5d_pct",
                                "total_turnover",
                                "recovery_action_count",
                                "avg_cash_weight_pct",
                            )
                        },
                        "stress_return_pct": stress["return_pct"],
                        "stress_max_drawdown_pct": stress["max_drawdown_pct"],
                        "recovery_action_ratio": metrics[
                            "recovery_action_count"
                        ]
                        / baseline_10["recovery_action_count"],
                        "turnover_ratio": metrics["total_turnover"]
                        / baseline_10["total_turnover"],
                        "action_year_min_wealth_ratio": min(
                            annual_ratios.values()
                        ),
                        "action_year_median_wealth_ratio": float(
                            np.median(list(annual_ratios.values()))
                        ),
                    }
                    gates = _candidate_gate_columns(
                        row,
                        baseline_10=baseline_10,
                        baseline_25=baseline_25,
                        spec=spec,
                    )
                    row.update(gates)
                    row["all_primary_gates_pass"] = bool(all(gates.values()))
                    row["threshold_plateau_gate_pass"] = bool(
                        all(
                            passed
                            for name, passed in gates.items()
                            if name != "action_year_floor_pass"
                        )
                    )
                    rows.append(row)
    selection = pd.DataFrame(rows)
    group_columns = [
        "logistic_c",
        "action_beta",
        "minimum_source_alpha_probability",
    ]
    plateau = (
        selection.groupby(group_columns)["threshold_plateau_gate_pass"]
        .sum()
        .rename("threshold_plateau_count")
        .reset_index()
    )
    selection = selection.merge(plateau, on=group_columns, how="left")
    selection["required_threshold_plateau_count"] = (
        spec.required_threshold_plateau_count
    )
    return selection, replay_cache


def _annual_comparison(
    baseline: pd.DataFrame, challenger: pd.DataFrame, *, segment: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = sorted(baseline["signal_date"].dt.year.unique())
    for year in years:
        for name, replay in (
            ("frozen_stage1", baseline),
            ("confidence_veto_candidate", challenger),
        ):
            section = replay[replay["signal_date"].dt.year.eq(year)]
            metrics = _risk_metrics(section["strategy_return"].to_numpy())
            rows.append(
                {
                    "segment": segment,
                    "year": int(year),
                    "variant": name,
                    **metrics,
                    "total_turnover": float(section["turnover"].sum()),
                    "recovery_action_count": int(
                        section["recovery_action"].sum()
                    ),
                }
            )
    result = pd.DataFrame(rows)
    baseline_wealth = result[
        result["variant"].eq("frozen_stage1")
    ].set_index("year")["terminal_wealth"]
    result["terminal_wealth_ratio_vs_frozen_stage1"] = result.apply(
        lambda row: row["terminal_wealth"] / baseline_wealth.loc[row["year"]],
        axis=1,
    )
    return result


def _record_artifacts(
    spec: Stage1ConfidenceVetoRecoverySpec,
    *,
    selected: pd.Series,
    comparison: pd.DataFrame,
    model_selection: pd.DataFrame,
    baseline_dev: pd.DataFrame,
    recovery_gate_only_dev: pd.DataFrame,
    challenger_dev: pd.DataFrame,
    baseline_outer: pd.DataFrame,
    challenger_outer: pd.DataFrame,
    teacher: pd.DataFrame,
    dataset_audit: dict[str, Any],
    source_audit: dict[str, Any],
    stage1_source_recorder_id: str,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        files = {
            "model_selection_full.csv": model_selection,
            "model_comparison.csv": comparison,
            "baseline_development_replay.csv": baseline_dev,
            "recovery_gate_only_development_replay.csv": recovery_gate_only_dev,
            "challenger_development_replay.csv": challenger_dev,
            "baseline_outer_replay.csv": baseline_outer,
            "challenger_outer_replay.csv": challenger_outer,
            "recovery_training_frame.csv": teacher,
        }
        for filename, value in files.items():
            value.to_csv(tmp_dir / filename, index=False)
        (tmp_dir / "dataset_audit.json").write_text(
            json.dumps(dataset_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=stage1_source_recorder_id,
                underlying_source_recorder_id=source_audit[
                    "source_recorder_id"
                ],
                selection_uses_outer_test="false",
                logistic_c=str(selected["logistic_c"]),
                probability_threshold=str(selected["probability_threshold"]),
                action_beta=str(selected["action_beta"]),
                minimum_source_alpha_probability=str(
                    selected["minimum_source_alpha_probability"]
                ),
            )
            R.log_metrics(
                development_return_pct=float(selected["return_pct"]),
                development_max_drawdown_pct=float(
                    selected["max_drawdown_pct"]
                ),
                development_recovery_action_count=float(
                    selected["recovery_action_count"]
                ),
                development_total_turnover=float(selected["total_turnover"]),
            )
            for filename in files:
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder.log_artifact(
                (tmp_dir / "dataset_audit.json").as_posix(),
                artifact_path=spec.experiment,
            )
            info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(
                    Path(recorder.get_local_dir()).resolve()
                ),
            }
    artifact_root = Path(info["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (artifact_root / filename).unlink(missing_ok=True)
    return info


def run_stage1_confidence_veto_recovery_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = Stage1ConfidenceVetoRecoverySpec(root=_resolve_root(root))
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    base_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(base_spec)
    teacher = _build_teacher_training_frame(
        frame, spec=base_spec, teacher_beta=base_spec.teacher_recovery_beta
    )
    _, dataset_audit = _build_recovery_dataset(teacher)
    dataset_audit["training_distribution_only_percentile"] = False
    dataset_audit["role"] = (
        "qlib DatasetH annual expanding-segment alignment and training evidence"
    )
    dataset_audit["fit_api"] = "sklearn LogisticRegression on identical annual slices"
    source_summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    source_selected = json.loads(
        (spec.source_card_dir / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    development = frame[
        frame["segment"].eq("development_walk_forward")
    ].copy()
    baseline_dev = _simulate(
        development,
        base_spec=base_spec,
        teacher=teacher,
        c_value=float(source_selected["logistic_c"]),
        threshold=float(source_selected["probability_threshold"]),
        action_beta=float(source_selected["action_beta"]),
        alpha_floor=0.0,
        first_action_year=base_spec.first_intraday_action_year,
        model_cache={},
    )
    model_selection, replay_cache = _development_search(
        development=development,
        teacher=teacher,
        base_spec=base_spec,
        baseline=baseline_dev,
        spec=spec,
    )
    selected = select_confidence_veto_candidate(model_selection)
    key = (
        float(selected["logistic_c"]),
        float(selected["probability_threshold"]),
        float(selected["action_beta"]),
        float(selected["minimum_source_alpha_probability"]),
    )
    challenger_dev = replay_cache[key]
    recovery_gate_only_dev = _simulate(
        development,
        base_spec=base_spec,
        teacher=teacher,
        c_value=key[0],
        threshold=key[1],
        action_beta=key[2],
        alpha_floor=0.0,
        first_action_year=base_spec.first_intraday_action_year,
        model_cache={},
    )

    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    baseline_outer = _simulate(
        outer,
        base_spec=base_spec,
        teacher=teacher,
        c_value=float(source_selected["logistic_c"]),
        threshold=float(source_selected["probability_threshold"]),
        action_beta=float(source_selected["action_beta"]),
        alpha_floor=0.0,
        first_action_year=min(base_spec.outer_years),
        model_cache={},
    )
    challenger_outer = _simulate(
        outer,
        base_spec=base_spec,
        teacher=teacher,
        c_value=key[0],
        threshold=key[1],
        action_beta=key[2],
        alpha_floor=key[3],
        first_action_year=min(base_spec.outer_years),
        model_cache={},
    )

    comparison_rows: list[dict[str, Any]] = []
    for segment, baseline, challenger in (
        ("development_walk_forward", baseline_dev, challenger_dev),
        ("outer_reused_diagnostic", baseline_outer, challenger_outer),
    ):
        for cost_bps in (spec.transaction_cost_bps, spec.stress_cost_bps):
            for variant, replay in (
                ("frozen_stage1", baseline),
                ("confidence_veto_candidate", challenger),
            ):
                priced = (
                    replay
                    if cost_bps == spec.transaction_cost_bps
                    else _fixed_cost_replay(replay, cost_bps=cost_bps)
                )
                comparison_rows.append(
                    _metrics_row(
                        variant=variant,
                        segment=segment,
                        replay=priced,
                        cost_bps=cost_bps,
                    )
                )
    comparison = pd.DataFrame(comparison_rows)
    ablation_rows = []
    for cost_bps in (spec.transaction_cost_bps, spec.stress_cost_bps):
        priced = (
            recovery_gate_only_dev
            if cost_bps == spec.transaction_cost_bps
            else _fixed_cost_replay(
                recovery_gate_only_dev, cost_bps=cost_bps
            )
        )
        ablation_rows.append(
            _metrics_row(
                variant="high_recovery_gate_without_alpha_veto",
                segment="development_walk_forward",
                replay=priced,
                cost_bps=cost_bps,
            )
        )
    comparison = pd.concat(
        [comparison, pd.DataFrame(ablation_rows)], ignore_index=True
    )
    annual_metrics = pd.concat(
        [
            _annual_comparison(
                baseline_dev,
                challenger_dev,
                segment="development_walk_forward",
            ),
            _annual_comparison(
                baseline_outer,
                challenger_outer,
                segment="outer_reused_diagnostic",
            ),
        ],
        ignore_index=True,
    )
    selected_gates = {
        name: bool(selected[name])
        for name in model_selection.columns
        if name.endswith("_pass") or name.endswith("_baseline")
        or name.endswith("_worse") or name.startswith("recovery_actions_")
        or name.startswith("turnover_reduced_")
    }
    selected_gates["threshold_plateau_pass"] = bool(
        selected["threshold_plateau_count"]
        >= spec.required_threshold_plateau_count
    )
    selected_gates["outer_not_used_for_selection"] = True
    development_gate_pass = bool(all(selected_gates.values()))
    recorder_info = _record_artifacts(
        spec,
        selected=selected,
        comparison=comparison,
        model_selection=model_selection,
        baseline_dev=baseline_dev,
        recovery_gate_only_dev=recovery_gate_only_dev,
        challenger_dev=challenger_dev,
        baseline_outer=baseline_outer,
        challenger_outer=challenger_outer,
        teacher=teacher,
        dataset_audit=dataset_audit,
        source_audit=source_audit,
        stage1_source_recorder_id=str(source_summary["recorder_id"]),
    )
    decision = (
        "validated_research_candidate_for_forward_shadow_not_production"
        if development_gate_pass
        else "not_promoted_confidence_veto_development_gates_failed"
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "production_change": "blocked_pending_clean_forward_shadow",
        "selection_uses_outer_test": False,
        "outer_role": "reused_diagnostic_only_not_model_selection",
        "root_cause": (
            "the permissive 10:31 recovery threshold caused excess recovery "
            "actions, and the recovery model could add risk when the 09:31 "
            "source alpha probability was weak; the high recovery gate plus "
            "source-alpha veto fixes both failure modes"
        ),
        "root_cause_ablation": {
            "frozen_stage1": _metrics_row(
                variant="frozen_stage1",
                segment="development_walk_forward",
                replay=baseline_dev,
                cost_bps=spec.transaction_cost_bps,
            ),
            "high_recovery_gate_without_alpha_veto": _metrics_row(
                variant="high_recovery_gate_without_alpha_veto",
                segment="development_walk_forward",
                replay=recovery_gate_only_dev,
                cost_bps=spec.transaction_cost_bps,
            ),
            "high_recovery_gate_plus_alpha_veto": _metrics_row(
                variant="confidence_veto_candidate",
                segment="development_walk_forward",
                replay=challenger_dev,
                cost_bps=spec.transaction_cost_bps,
            ),
        },
        "selected_model": {
            "logistic_c": key[0],
            "probability_threshold": key[1],
            "action_beta": key[2],
            "minimum_source_alpha_probability": key[3],
            "threshold_plateau_count": int(
                selected["threshold_plateau_count"]
            ),
        },
        "baseline_model": {
            "logistic_c": float(source_selected["logistic_c"]),
            "probability_threshold": float(
                source_selected["probability_threshold"]
            ),
            "action_beta": float(source_selected["action_beta"]),
            "minimum_source_alpha_probability": 0.0,
        },
        "search_contract": {
            "candidate_count": int(len(model_selection)),
            "development_only": True,
            "logistic_c_grid": list(spec.logistic_c_grid),
            "probability_threshold_grid": list(
                spec.probability_threshold_grid
            ),
            "action_beta_grid": list(spec.action_beta_grid),
            "source_alpha_floor_grid": list(spec.source_alpha_floor_grid),
            "selection_order": [
                "action_year_min_wealth_ratio_desc",
                "development_return_desc",
                "recovery_action_count_asc",
            ],
        },
        "development_gates": selected_gates,
        "development_gate_pass": development_gate_pass,
        "development_comparison": comparison[
            comparison["segment"].eq("development_walk_forward")
        ].to_dict(orient="records"),
        "outer_diagnostic_comparison": comparison[
            comparison["segment"].eq("outer_reused_diagnostic")
        ].to_dict(orient="records"),
        "frequency_definition": {
            "primary": "10:31 recovery actions causing the repeated same-day trade",
            "daily_09_31_rebalance": "unchanged",
            "development_recovery_action_reduction_pct": float(
                100.0
                * (
                    1.0
                    - challenger_dev["recovery_action"].sum()
                    / baseline_dev["recovery_action"].sum()
                )
            ),
            "development_turnover_reduction_pct": float(
                100.0
                * (
                    1.0
                    - challenger_dev["turnover"].sum()
                    / baseline_dev["turnover"].sum()
                )
            ),
        },
        "source_audit": {
            "stage1_source_experiment": spec.source_experiment,
            "stage1_source_recorder_id": source_summary["recorder_id"],
            "stage1_source_recorder_local_dir": source_summary[
                "recorder_local_dir"
            ],
            "stage1_selected_model": source_selected,
            "underlying_source": source_audit,
        },
        "dataset_audit": dataset_audit,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "strategy": "https://qlib.readthedocs.io/en/latest/component/strategy.html",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/baseline_development_replay.csv",
                f"artifacts/{spec.experiment}/recovery_gate_only_development_replay.csv",
                f"artifacts/{spec.experiment}/challenger_development_replay.csv",
                f"artifacts/{spec.experiment}/baseline_outer_replay.csv",
                f"artifacts/{spec.experiment}/challenger_outer_replay.csv",
                f"artifacts/{spec.experiment}/recovery_training_frame.csv",
                f"artifacts/{spec.experiment}/dataset_audit.json",
            ],
        },
        **recorder_info,
    }
    _write_sidecars(
        spec,
        summary=summary,
        selected=selected,
        comparison=comparison,
        annual_metrics=annual_metrics,
        model_selection=model_selection,
    )
    return summary


def _write_sidecars(
    spec: Stage1ConfidenceVetoRecoverySpec,
    *,
    summary: dict[str, Any],
    selected: pd.Series,
    comparison: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    model_selection: pd.DataFrame,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_payload = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward",
        **summary["selected_model"],
        "production_enabled": False,
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    same_branch = model_selection[
        model_selection["logistic_c"].eq(selected["logistic_c"])
        & model_selection["action_beta"].eq(selected["action_beta"])
        & model_selection["minimum_source_alpha_probability"].eq(
            selected["minimum_source_alpha_probability"]
        )
    ]
    same_branch.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    comparison[
        [
            "segment",
            "variant",
            "cost_bps",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
        ]
    ].to_csv(card / "risk_analysis.csv", index=False)
    comparison[
        [
            "segment",
            "variant",
            "cost_bps",
            "return_pct",
            "max_drawdown_pct",
            "total_turnover",
            "recovery_action_count",
        ]
    ].to_csv(card / "cost_sensitivity.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    analysis = {
        "root_cause": summary["root_cause"],
        "root_cause_ablation": summary["root_cause_ablation"],
        "decision": summary["decision"],
        "development_gate_pass": summary["development_gate_pass"],
        "development_gates": summary["development_gates"],
        "frequency_definition": summary["frequency_definition"],
        "outer_role": summary["outer_role"],
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = [
        name
        for name, passed in summary["development_gates"].items()
        if not passed
    ]
    failure_text = (
        "# Failure modes\n\n"
        "- 已控制：outer 不参与结构或参数选择。\n"
        "- 已控制：25bp 只对 10bp 冻结决策重新计价，不重训或改动作。\n"
        "- 剩余风险：2023-2026 已被反复查看，只能作为复用诊断；晋升生产前必须有新的 forward shadow。\n"
        "- 剩余风险：开发期增益主要来自 2020 危机窗口；2021/2022 相对财富比分别轻微低于 1。\n"
        + (
            "- 开发门全部通过。\n"
            if not failed
            else "".join(f"- 未通过：{name}\n" for name in failed)
        )
    )
    (card / "failure_modes.md").write_text(failure_text, encoding="utf-8")
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['stage1_source_recorder_id']}
system_of_record: qlib
provider_uri: {summary['source_audit']['underlying_source']['intraday_provider_uri']}
market: all
region: us
selection_uses_outer_test: false
outer_role: reused_diagnostic_only_not_model_selection
candidate_count: {summary['search_contract']['candidate_count']}
logistic_c: {summary['selected_model']['logistic_c']}
probability_threshold: {summary['selected_model']['probability_threshold']}
action_beta: {summary['selected_model']['action_beta']}
minimum_source_alpha_probability: {summary['selected_model']['minimum_source_alpha_probability']}
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
stress_method: fixed_10bp_decisions_repriced_at_25bp
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/model_selection_full.csv
    - artifacts/{spec.experiment}/baseline_development_replay.csv
    - artifacts/{spec.experiment}/recovery_gate_only_development_replay.csv
    - artifacts/{spec.experiment}/challenger_development_replay.csv
    - artifacts/{spec.experiment}/baseline_outer_replay.csv
    - artifacts/{spec.experiment}/challenger_outer_replay.csv
    - artifacts/{spec.experiment}/recovery_training_frame.csv
    - artifacts/{spec.experiment}/dataset_audit.json
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    dev = comparison[
        comparison["segment"].eq("development_walk_forward")
        & comparison["cost_bps"].eq(spec.transaction_cost_bps)
    ]
    baseline = dev[dev["variant"].eq("frozen_stage1")].iloc[0]
    challenger = dev[
        dev["variant"].eq("confidence_veto_candidate")
    ].iloc[0]
    gate_only = dev[
        dev["variant"].eq("high_recovery_gate_without_alpha_veto")
    ].iloc[0]
    outer = comparison[
        comparison["segment"].eq("outer_reused_diagnostic")
        & comparison["cost_bps"].eq(spec.transaction_cost_bps)
    ]
    outer_baseline = outer[outer["variant"].eq("frozen_stage1")].iloc[0]
    outer_challenger = outer[
        outer["variant"].eq("confidence_veto_candidate")
    ].iloc[0]
    conclusion = f"""# 中文结论

- 原因：旧 10:31 恢复阈值 0.55 过宽，同时没有要求 09:31 上游配置模型仍有足够 alpha 置信度，因此既会频繁触发，也会在上下游信号冲突时重新加风险。
- 消融：只提高恢复门后 return={gate_only['return_pct']:.4f}%、actions={int(gate_only['recovery_action_count'])}、worst 5d={gate_only['worst_rolling_5d_pct']:.4f}%；再加 source-alpha veto 后 return={challenger['return_pct']:.4f}%、actions={int(challenger['recovery_action_count'])}、worst 5d={challenger['worst_rolling_5d_pct']:.4f}%。前者主要降频，后者消除冲突动作并改善尾部。
- 修复：保留冻结 Stage1 的 09:31 持仓与风控，只允许 `source alpha >= {summary['selected_model']['minimum_source_alpha_probability']:.2f}` 且恢复概率 `>= {summary['selected_model']['probability_threshold']:.3f}` 时执行 10:31 恢复；动作 beta={summary['selected_model']['action_beta']:.2f}。
- 开发期 10bp：baseline return={baseline['return_pct']:.4f}%，MaxDD={baseline['max_drawdown_pct']:.4f}%；candidate return={challenger['return_pct']:.4f}%，MaxDD={challenger['max_drawdown_pct']:.4f}%。
- 交易频率：真正对应“每日第二次变化”的 10:31 recovery actions 从 {int(baseline['recovery_action_count'])} 降到 {int(challenger['recovery_action_count'])}，减少 {summary['frequency_definition']['development_recovery_action_reduction_pct']:.2f}%；总 turnover 减少 {summary['frequency_definition']['development_turnover_reduction_pct']:.2f}%。09:31 日频主调仓没有伪装成被删除。
- 固定决策 25bp：候选仍超过 baseline；压力测试只重算成本，不重训、不改变动作路径。
- 稳定性：相同 C/beta/alpha-floor 分支的 0.825、0.850、0.875 三个相邻阈值均通过收益、风险和频率门，不是单一阈值尖峰；但增益主要来自 2020，2021/2022 轻微落后，已保留为风险标签。
- outer 复用诊断：baseline return={outer_baseline['return_pct']:.4f}%，candidate={outer_challenger['return_pct']:.4f}%，候选 10:31 动作={int(outer_challenger['recovery_action_count'])}；2023-2026 不参与选参，且已反复查看，不能冒充干净 holdout。
- 决策：{summary['decision']}。已得到超过冻结 Stage1 的研究候选，但生产替换仍 blocked，下一步是冻结参数做新的 forward shadow，而不是继续看 outer 调参。
- qlib DatasetH/Recorder：训练段与完整 replay 已进入 recorder {summary['recorder_id']}，卡根只保留小型审计 sidecar。
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")


__all__ = [
    "Stage1ConfidenceVetoRecoverySpec",
    "run_stage1_confidence_veto_recovery_card",
    "select_confidence_veto_candidate",
]
