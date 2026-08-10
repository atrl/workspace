from __future__ import annotations

import json
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
    _development_stability,
    _feature_row,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    _simulate_policy,
    apply_fixed_decision_cost_stress,
    mark_to_market_weights,
    project_legal_weights,
)


EXPERIMENT = "stage1_rank_calibrated_recovery_sizing_v1"
CARD_TYPE = "stage1_preregistered_rank_calibrated_recovery_sizing_card"


@dataclass(frozen=True)
class Stage1RankCalibratedRecoverySizingSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = "intraday_recovery_direct_allocation_v1"
    logistic_c: float = 0.03
    maximum_recovery_beta: float = 0.60
    percentile_activation_floor: float = 0.50
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_stage1_rank_calibrated_recovery_sizing_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = Stage1RankCalibratedRecoverySizingSpec(root=_resolve_root(root))
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


def empirical_percentile(train_scores: np.ndarray, score: float) -> float:
    """Rank one score against training-only scores without test-set ranking."""
    values = np.asarray(train_scores, dtype="float64")
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0 or not np.isfinite(score):
        return float("nan")
    return float(np.searchsorted(values, score, side="right") / values.size)


def fixed_recovery_fraction(
    percentile: float, *, activation_floor: float = 0.50
) -> float:
    """Map a training percentile to a fixed linear recovery fraction."""
    if not 0.0 <= activation_floor < 1.0:
        raise ValueError("activation_floor must be in [0, 1)")
    if not np.isfinite(percentile):
        return 0.0
    return float(
        np.clip(
            (float(percentile) - activation_floor) / (1.0 - activation_floor),
            0.0,
            1.0,
        )
    )


def _build_recovery_dataset(teacher: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    indexed = teacher.set_index("trade_date").sort_index()
    static = pd.concat(
        {
            "feature": indexed.loc[:, FEATURE_COLUMNS],
            "label": indexed.loc[:, ["teacher_action_label"]],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [static.index, np.full(len(static), "STAGE1_RECOVERY")],
        names=["datetime", "instrument"],
    )
    years = sorted(int(year) for year in teacher["year"].unique() if year >= 2020)
    segments = {
        f"train_{year}": ("2019-01-01", f"{year - 1}-12-31")
        for year in years
    }
    handler = DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    audit = {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "features": list(FEATURE_COLUMNS),
        "label": "teacher_action_label",
        "training_distribution_only_percentile": True,
    }
    return dataset, audit


def _fit_rank_model(dataset, *, year: int, c_value: float):
    prepared = dataset.prepare(
        f"train_{year}", col_set=["feature", "label"]
    )
    prepared = prepared.droplevel("instrument").sort_index()
    prepared.columns = [str(column[-1]) for column in prepared.columns]
    training = prepared[prepared["teacher_action_label"].notna()].copy()
    if len(training) < 20 or training["teacher_action_label"].nunique() < 2:
        return None
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=42,
        ),
    )
    model.fit(training.loc[:, FEATURE_COLUMNS], training["teacher_action_label"])
    train_scores = model.predict_proba(training.loc[:, FEATURE_COLUMNS])[:, 1]
    return model, np.sort(train_scores.astype("float64"))


def _simulate_rank_calibrated_policy(
    frame: pd.DataFrame,
    *,
    dataset,
    base_spec: IntradayRecoveryDirectAllocationSpec,
    policy_spec: Stage1RankCalibratedRecoverySizingSpec,
    first_action_year: int,
    cost_bps: float,
) -> pd.DataFrame:
    models: dict[int, Any] = {}
    for year in sorted(frame["signal_date"].dt.year.unique()):
        models[int(year)] = (
            _fit_rank_model(
                dataset, year=int(year), c_value=policy_spec.logistic_c
            )
            if int(year) >= first_action_year
            else None
        )

    equity = peak = 1.0
    previous = np.zeros(3)
    state = clock = 0
    history: list[float] = []
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        pretrade_weights = previous.copy()
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(state, clock, drawdown, base_spec)
        cap, raw_beta = _base_cap(
            row, q=q, drawdown=drawdown, history=history, spec=base_spec
        )
        open_weights = project_legal_weights(
            row.soxl_weight,
            row.soxx_weight,
            cap,
            minimum_cash_weight=base_spec.minimum_cash_weight,
            maximum_soxl_weight=base_spec.maximum_soxl_weight,
        )
        maximum_target_beta = min(raw_beta, policy_spec.maximum_recovery_beta)
        fitted = models.get(int(row.signal_date.year))
        eligible = (
            row.signal_date.year >= first_action_year
            and fitted is not None
            and drawdown >= base_spec.recovery_eligibility_drawdown
            and maximum_target_beta > cap + 1e-12
            and pd.notna(row.early_ret_soxx)
        )
        probability = percentile = np.nan
        recovery_fraction = 0.0
        target_beta = cap
        if eligible:
            model, train_scores = fitted
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
            percentile = empirical_percentile(train_scores, probability)
            recovery_fraction = fixed_recovery_fraction(
                percentile,
                activation_floor=policy_spec.percentile_activation_floor,
            )
            target_beta = cap + recovery_fraction * (maximum_target_beta - cap)
        recovery = bool(target_beta > cap + 1e-12)

        if recovery:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                target_beta,
                minimum_cash_weight=base_spec.minimum_cash_weight,
                maximum_soxl_weight=base_spec.maximum_soxl_weight,
            )
            early = np.array([row.early_ret_soxl, row.early_ret_soxx, 0.0])
            value_at_cut = 1.0 + open_weights @ early
            drifted = mark_to_market_weights(open_weights, early)
            post = np.array(
                [
                    (1.0 + row.soxl_return) / (1.0 + row.early_ret_soxl) - 1.0,
                    (1.0 + row.soxx_return) / (1.0 + row.early_ret_soxx) - 1.0,
                    0.0,
                ]
            )
            gross_return = value_at_cut * (1.0 + close_weights @ post) - 1.0
            turnover = np.abs(open_weights[:2] - previous[:2]).sum() + np.abs(
                close_weights[:2] - drifted[:2]
            ).sum()
            trade_delta = (open_weights - pretrade_weights) + (
                close_weights - drifted
            )
            ending_weights = mark_to_market_weights(close_weights, post)
        else:
            close_weights = open_weights
            asset_returns = np.array(
                [row.soxl_return, row.soxx_return, 0.0], dtype="float64"
            )
            gross_return = float(open_weights @ asset_returns)
            turnover = np.abs(open_weights[:2] - previous[:2]).sum()
            trade_delta = open_weights - pretrade_weights
            ending_weights = mark_to_market_weights(open_weights, asset_returns)

        strategy_return = gross_return - turnover * cost_bps / 10_000.0
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
                "soxl_trade_delta": trade_delta[0],
                "soxx_trade_delta": trade_delta[1],
                "cash_trade_delta": trade_delta[2],
                "recovery_probability": probability,
                "training_percentile": percentile,
                "recovery_fraction": recovery_fraction,
                "recovery_action": recovery,
                "current_beta_cap": cap,
                "recovery_beta_target": target_beta,
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": cost_bps,
            }
        )
    return pd.DataFrame(rows)


def _fixed_cost_replay(replay: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    stressed = replay.copy()
    stressed["strategy_return"] = apply_fixed_decision_cost_stress(
        stressed["gross_return"], stressed["turnover"], cost_bps=cost_bps
    )
    stressed["cost_bps"] = float(cost_bps)
    return stressed


def _variant_row(
    name: str, replay: pd.DataFrame, *, include_development_stability: bool = True
) -> dict[str, Any]:
    stability = (
        _development_stability(replay)
        if include_development_stability
        else {
            "risk_window_pass_count": np.nan,
            "positive_relative_window_count": np.nan,
            "relative_ratio_min": np.nan,
            "relative_ratio_p25": np.nan,
            "relative_ratio_median": np.nan,
        }
    )
    return {
        "variant": name,
        **_risk_metrics(replay["strategy_return"].to_numpy()),
        **stability,
        "total_turnover": float(replay["turnover"].sum()),
        "avg_cash_weight_pct": float(100.0 * replay["cash_weight"].mean()),
        "recovery_action_count": int(replay["recovery_action"].sum()),
    }


def _record_artifacts(
    spec: Stage1RankCalibratedRecoverySizingSpec,
    *,
    model_comparison: pd.DataFrame,
    baseline_outer: pd.DataFrame,
    challenger_outer: pd.DataFrame,
    teacher: pd.DataFrame,
    dataset_audit: dict[str, Any],
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        files: dict[str, Any] = {
            "model_comparison.csv": model_comparison,
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
                logistic_c=str(spec.logistic_c),
                maximum_recovery_beta=str(spec.maximum_recovery_beta),
                percentile_activation_floor=str(spec.percentile_activation_floor),
                selection_uses_outer_test="false",
                candidate_count="1",
            )
            challenger = model_comparison.loc[
                model_comparison["variant"].eq("rank_calibrated_challenger")
            ].iloc[0]
            R.log_metrics(
                development_return_pct=float(challenger["return_pct"]),
                development_max_drawdown_pct=float(challenger["max_drawdown_pct"]),
                development_total_turnover=float(challenger["total_turnover"]),
            )
            for filename in files:
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(), artifact_path=spec.experiment
                )
            recorder.log_artifact(
                (tmp_dir / "dataset_audit.json").as_posix(),
                artifact_path=spec.experiment,
            )
            info = {
                "recorder_id": recorder.id,
                "recorder_uri": str(recorder_uri.resolve()),
                "recorder_local_dir": str(Path(recorder.get_local_dir()).resolve()),
            }
    artifact_root = Path(info["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (artifact_root / filename).unlink(missing_ok=True)
    return info


def _run_card(spec: Stage1RankCalibratedRecoverySizingSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    base_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(base_spec)
    teacher = _build_teacher_training_frame(
        frame, spec=base_spec, teacher_beta=base_spec.teacher_recovery_beta
    )
    dataset, dataset_audit = _build_recovery_dataset(teacher)

    development = frame[frame["segment"].eq("development_walk_forward")].copy()
    baseline_dev = _simulate_policy(
        development,
        spec=base_spec,
        teacher_frame=teacher,
        c_value=spec.logistic_c,
        threshold=0.55,
        action_beta=spec.maximum_recovery_beta,
        first_action_year=base_spec.first_intraday_action_year,
        model_cache={},
        cost_bps=spec.transaction_cost_bps,
    )
    challenger_dev = _simulate_rank_calibrated_policy(
        development,
        dataset=dataset,
        base_spec=base_spec,
        policy_spec=spec,
        first_action_year=base_spec.first_intraday_action_year,
        cost_bps=spec.transaction_cost_bps,
    )
    baseline_dev_25 = _fixed_cost_replay(
        baseline_dev, cost_bps=spec.stress_cost_bps
    )
    challenger_dev_25 = _fixed_cost_replay(
        challenger_dev, cost_bps=spec.stress_cost_bps
    )

    comparison_rows = [
        {"segment": "development_walk_forward", "cost_bps": 10.0, **_variant_row("fixed_threshold_baseline", baseline_dev)},
        {"segment": "development_walk_forward", "cost_bps": 10.0, **_variant_row("rank_calibrated_challenger", challenger_dev)},
        {"segment": "development_walk_forward", "cost_bps": 25.0, **_variant_row("fixed_threshold_baseline", baseline_dev_25)},
        {"segment": "development_walk_forward", "cost_bps": 25.0, **_variant_row("rank_calibrated_challenger", challenger_dev_25)},
    ]
    model_comparison = pd.DataFrame(comparison_rows)
    baseline_10 = model_comparison[
        model_comparison["variant"].eq("fixed_threshold_baseline")
        & model_comparison["cost_bps"].eq(10.0)
    ].iloc[0]
    challenger_10 = model_comparison[
        model_comparison["variant"].eq("rank_calibrated_challenger")
        & model_comparison["cost_bps"].eq(10.0)
    ].iloc[0]
    baseline_25 = model_comparison[
        model_comparison["variant"].eq("fixed_threshold_baseline")
        & model_comparison["cost_bps"].eq(25.0)
    ].iloc[0]
    challenger_25 = model_comparison[
        model_comparison["variant"].eq("rank_calibrated_challenger")
        & model_comparison["cost_bps"].eq(25.0)
    ].iloc[0]

    gates = {
        "candidate_count_is_one": True,
        "outer_not_used_for_selection": True,
        "all_development_risk_windows_pass": int(challenger_10["risk_window_pass_count"]) == 7,
        "all_development_relative_windows_pass": int(challenger_10["positive_relative_window_count"]) == 7,
        "return_10bps_not_below_baseline": float(challenger_10["terminal_wealth"]) >= float(baseline_10["terminal_wealth"]),
        "return_25bps_not_below_baseline": float(challenger_25["terminal_wealth"]) >= float(baseline_25["terminal_wealth"]),
        "max_drawdown_not_worse": float(challenger_10["max_drawdown_pct"]) >= float(baseline_10["max_drawdown_pct"]),
        "worst_5d_not_worse": float(challenger_10["worst_rolling_5d_pct"]) >= float(baseline_10["worst_rolling_5d_pct"]),
        "turnover_not_higher": float(challenger_10["total_turnover"]) <= float(baseline_10["total_turnover"]),
    }
    development_gate_pass = bool(all(gates.values()))

    outer = frame[frame["segment"].eq("outer_diagnostic")].copy()
    baseline_outer = _simulate_policy(
        outer,
        spec=base_spec,
        teacher_frame=teacher,
        c_value=spec.logistic_c,
        threshold=0.55,
        action_beta=spec.maximum_recovery_beta,
        first_action_year=min(base_spec.outer_years),
        model_cache={},
        cost_bps=spec.transaction_cost_bps,
    )
    challenger_outer = _simulate_rank_calibrated_policy(
        outer,
        dataset=dataset,
        base_spec=base_spec,
        policy_spec=spec,
        first_action_year=min(base_spec.outer_years),
        cost_bps=spec.transaction_cost_bps,
    )
    outer_comparison = pd.DataFrame(
        [
            {"segment": "outer_reused_diagnostic", "cost_bps": 10.0, **_variant_row("fixed_threshold_baseline", baseline_outer, include_development_stability=False)},
            {"segment": "outer_reused_diagnostic", "cost_bps": 10.0, **_variant_row("rank_calibrated_challenger", challenger_outer, include_development_stability=False)},
        ]
    )
    full_comparison = pd.concat(
        [model_comparison, outer_comparison], ignore_index=True
    )

    recorder_info = _record_artifacts(
        spec,
        model_comparison=full_comparison,
        baseline_outer=baseline_outer,
        challenger_outer=challenger_outer,
        teacher=teacher,
        dataset_audit=dataset_audit,
    )
    decision = (
        "candidate_for_frozen_forward_shadow_not_production"
        if development_gate_pass
        else "not_promoted_rank_calibrated_recovery_development_gates_failed"
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "production_change": "blocked",
        "selection_uses_outer_test": False,
        "outer_role": "reused_diagnostic_only_not_model_selection",
        "preregistration": {
            "candidate_count": 1,
            "grid_search": False,
            "formula": "fraction=max(0, 2*training_percentile-1)",
            "logistic_c": spec.logistic_c,
            "maximum_recovery_beta": spec.maximum_recovery_beta,
            "percentile_activation_floor": spec.percentile_activation_floor,
        },
        "dataset_audit": dataset_audit,
        "source_audit": {
            "source_experiment": spec.source_experiment,
            "source_recorder_id": source_summary["recorder_id"],
            "underlying_market_and_model_source": source_audit,
        },
        "development_gates": gates,
        "development_gate_pass": development_gate_pass,
        "development_comparison": model_comparison.to_dict(orient="records"),
        "outer_diagnostic_comparison": outer_comparison.to_dict(orient="records"),
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "sidecars": {
            "selected_model": "selected_model.json",
            "model_comparison": "model_comparison.csv",
            "risk_analysis": "risk_analysis.csv",
            "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
            "failure_modes": "failure_modes.md",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/baseline_outer_replay.csv",
                f"artifacts/{spec.experiment}/challenger_outer_replay.csv",
                f"artifacts/{spec.experiment}/recovery_training_frame.csv",
                f"artifacts/{spec.experiment}/model_comparison.csv",
                f"artifacts/{spec.experiment}/dataset_audit.json",
            ],
        },
        **recorder_info,
    }
    _write_sidecars(
        spec,
        summary=summary,
        model_comparison=full_comparison,
        gates=gates,
    )
    return summary


def _write_sidecars(
    spec: Stage1RankCalibratedRecoverySizingSpec,
    *,
    summary: dict[str, Any],
    model_comparison: pd.DataFrame,
    gates: dict[str, bool],
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    selected = {
        "selection_uses_outer_test": False,
        "candidate_count": 1,
        "grid_search": False,
        "logistic_c": spec.logistic_c,
        "maximum_recovery_beta": spec.maximum_recovery_beta,
        "percentile_activation_floor": spec.percentile_activation_floor,
        "formula": "fraction=max(0, 2*training_percentile-1)",
        "production_enabled": False,
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_comparison.to_csv(card / "model_comparison.csv", index=False)
    risk_columns = [
        "segment",
        "variant",
        "cost_bps",
        "max_drawdown_pct",
        "worst_day_pct",
        "cvar5_pct",
        "worst_rolling_5d_pct",
        "risk_window_pass_count",
    ]
    model_comparison.loc[:, risk_columns].to_csv(
        card / "risk_analysis.csv", index=False
    )
    benchmark_columns = [
        "segment",
        "variant",
        "cost_bps",
        "terminal_wealth",
        "return_pct",
        "max_drawdown_pct",
        "total_turnover",
        "recovery_action_count",
    ]
    model_comparison.loc[:, benchmark_columns].to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    failed = [name for name, passed in gates.items() if not passed]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + ("- none in development gates\n" if not failed else "".join(f"- {name}\n" for name in failed)),
        encoding="utf-8",
    )
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
system_of_record: qlib
provider_uri: {summary['source_audit']['underlying_market_and_model_source']['intraday_provider_uri']}
market: all
region: us
selection_uses_outer_test: false
outer_role: reused_diagnostic_only_not_model_selection
candidate_count: 1
grid_search: false
logistic_c: {spec.logistic_c}
maximum_recovery_beta: {spec.maximum_recovery_beta}
percentile_activation_floor: {spec.percentile_activation_floor}
formula: fraction=max(0, 2*training_percentile-1)
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/baseline_outer_replay.csv
    - artifacts/{spec.experiment}/challenger_outer_replay.csv
    - artifacts/{spec.experiment}/recovery_training_frame.csv
    - artifacts/{spec.experiment}/model_comparison.csv
    - artifacts/{spec.experiment}/dataset_audit.json
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")

    dev = model_comparison[model_comparison["segment"].eq("development_walk_forward")]
    base = dev[(dev["variant"].eq("fixed_threshold_baseline")) & dev["cost_bps"].eq(10.0)].iloc[0]
    challenger = dev[(dev["variant"].eq("rank_calibrated_challenger")) & dev["cost_bps"].eq(10.0)].iloc[0]
    conclusion = f"""# 中文结论

- 做了什么：冻结 Stage1 的 Logistic C=0.03 与最高 0.6 beta，只把硬阈值替换为训练折内经验分位和固定线性恢复；候选数=1，没有参数网格。
- 数据与训练：通过 qlib DatasetH/DataHandlerLP/StaticDataLoader 构建年度扩展训练段，当前预测只与本折训练概率分布比较，不使用测试期排名。
- 开发期 10bp：baseline return={base['return_pct']:.4f}%，MaxDD={base['max_drawdown_pct']:.4f}%，turnover={base['total_turnover']:.4f}；challenger return={challenger['return_pct']:.4f}%，MaxDD={challenger['max_drawdown_pct']:.4f}%，turnover={challenger['total_turnover']:.4f}。
- 稳定性：challenger 风险窗口通过={int(challenger['risk_window_pass_count'])}/7，相对 SOXX 正窗口={int(challenger['positive_relative_window_count'])}/7，最差相对财富比={challenger['relative_ratio_min']:.4f}。
- 外层边界：2023–2026 仅为已复用诊断，不参与结构、参数或晋级选择；不能作为干净 holdout。
- 决策：{summary['decision']}；production_change=blocked。即使开发门通过，也只能进入冻结 forward shadow，不能替换当前生产 Stage1。
- 风险：固定分位映射仍是一种结构假设；若开发门失败，本路径停止，不调整分位点后重试。
- qlib recorder：{summary['recorder_id']}
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")


__all__ = [
    "Stage1RankCalibratedRecoverySizingSpec",
    "empirical_percentile",
    "fixed_recovery_fraction",
    "run_stage1_rank_calibrated_recovery_sizing_card",
]
