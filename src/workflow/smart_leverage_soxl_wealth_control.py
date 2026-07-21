from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.leveraged_etf_paper_replay import (
    LeveragedEtfPaperReplaySpec,
    _load_market_frame,
    _matched_beta_weights,
    _risk_gate,
    _risk_metrics,
    _simulate_weights,
)
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "smart_leverage_soxl_wealth_control_v1"
CARD_TYPE = "smart_leverage_soxl_wealth_control_research_card"
SOURCE_EXPERIMENT = "trainable_regime_policy_nested_wf_v1"
OUTER_YEARS = (2023, 2024, 2025, 2026)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
)


@dataclass(frozen=True)
class SmartLeverageSoxlWealthControlSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    outer_years: tuple[int, ...] = OUTER_YEARS
    costs_bps: tuple[int, ...] = (0, 10, 25)
    benchmark_soxx_weight: float = 0.70
    wealth_lock_strength: float = 1.0
    wealth_lock_floor: float = 0.50
    target_surplus: float = 0.0

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_smart_leverage_soxl_wealth_control_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(SmartLeverageSoxlWealthControlSpec(root=repo_root))


def asymmetric_wealth_multiplier(
    surplus: float,
    *,
    target_surplus: float,
    strength: float,
    floor: float,
) -> float:
    """De-risk after exceeding the target; never increase risk to chase losses."""
    if not 0.0 <= floor <= 1.0:
        raise ValueError("wealth lock floor must be in [0, 1]")
    if strength < 0.0:
        raise ValueError("wealth lock strength must be non-negative")
    excess = max(float(surplus) - float(target_surplus), 0.0)
    return float(np.clip(np.exp(-float(strength) * excess), floor, 1.0))


def beta_preserving_soxl_soxx_ladder(target_soxl_weight: float) -> tuple[float, float]:
    """Use SOXX first while preserving the target's SOXL-equivalent effective beta."""
    target = float(np.clip(target_soxl_weight, 0.0, 1.0))
    effective_beta = 3.0 * target
    if effective_beta <= 1.0:
        return 0.0, effective_beta
    soxl_weight = (effective_beta - 1.0) / 2.0
    soxx_weight = 1.0 - soxl_weight
    return float(soxl_weight), float(soxx_weight)


def simulate_wealth_control(
    frame: pd.DataFrame,
    opportunity_target: pd.Series | np.ndarray,
    *,
    cost_bps: int,
    allocation_mode: str,
    wealth_lock: bool,
    benchmark_soxx_weight: float,
    wealth_lock_strength: float,
    wealth_lock_floor: float,
    target_surplus: float,
) -> pd.DataFrame:
    required = {"signal_date", "trade_date", "exit_date", "soxl_return", "soxx_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"wealth control frame is missing columns: {sorted(missing)}")
    if allocation_mode not in {"soxl_only", "beta_ladder"}:
        raise ValueError("allocation_mode must be soxl_only or beta_ladder")
    if not 0.0 <= benchmark_soxx_weight <= 1.0:
        raise ValueError("benchmark_soxx_weight must be in [0, 1]")

    targets = np.asarray(opportunity_target, dtype=float)
    if len(targets) != len(frame):
        raise ValueError("opportunity target length must match frame")

    strategy_wealth = 1.0
    benchmark_wealth = 1.0
    previous_soxl = 0.0
    previous_soxx = 0.0
    rows: list[dict[str, float]] = []
    for index, market_row in enumerate(frame.itertuples(index=False)):
        surplus = strategy_wealth / benchmark_wealth - 1.0 if benchmark_wealth > 0.0 else 0.0
        multiplier = (
            asymmetric_wealth_multiplier(
                surplus,
                target_surplus=target_surplus,
                strength=wealth_lock_strength,
                floor=wealth_lock_floor,
            )
            if wealth_lock
            else 1.0
        )
        raw_target = float(np.clip(targets[index], 0.0, 1.0))
        controlled_target = float(np.clip(raw_target * multiplier, 0.0, 1.0))
        if allocation_mode == "soxl_only":
            soxl_weight, soxx_weight = controlled_target, 0.0
        else:
            soxl_weight, soxx_weight = beta_preserving_soxl_soxx_ladder(controlled_target)
        cash_weight = 1.0 - soxl_weight - soxx_weight
        turnover = abs(soxl_weight - previous_soxl) + abs(soxx_weight - previous_soxx)
        gross_return = soxl_weight * float(market_row.soxl_return) + soxx_weight * float(market_row.soxx_return)
        cost = turnover * float(cost_bps) / 10_000.0
        strategy_return = gross_return - cost
        benchmark_return = float(benchmark_soxx_weight) * float(market_row.soxx_return)
        rows.append(
            {
                "opportunity_target": raw_target,
                "wealth_surplus": surplus,
                "wealth_multiplier": multiplier,
                "controlled_target": controlled_target,
                "soxl_weight": soxl_weight,
                "soxx_weight": soxx_weight,
                "cash_weight": cash_weight,
                "effective_beta": 3.0 * soxl_weight + soxx_weight,
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "strategy_wealth_before": strategy_wealth,
                "benchmark_wealth_before": benchmark_wealth,
            }
        )
        strategy_wealth *= 1.0 + strategy_return
        benchmark_wealth *= 1.0 + benchmark_return
        previous_soxl, previous_soxx = soxl_weight, soxx_weight

    base = frame.loc[:, ["signal_date", "trade_date", "exit_date", "soxl_return", "soxx_return"]].reset_index(
        drop=True
    )
    return pd.concat([base, pd.DataFrame(rows)], axis=1)


def _load_nested_opportunity(spec: SmartLeverageSoxlWealthControlSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = spec.source_card_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing source qlib card summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection_uses_outer = summary.get("selection_uses_outer_test")
    if selection_uses_outer is None:
        selection_uses_outer = summary.get("model_selection", {}).get("selection_uses_outer_test")
    if selection_uses_outer is None:
        selection_uses_outer = summary.get("overfit_audit", {}).get("selection_uses_outer_test")
    if selection_uses_outer is not False:
        raise ValueError("source opportunity model must not use outer test for selection")
    recorder_dir = Path(str(summary["recorder_local_dir"]))
    positions_path = recorder_dir / "artifacts" / spec.source_experiment / "positions.csv"
    if not positions_path.exists():
        raise FileNotFoundError(f"missing source qlib recorder positions: {positions_path}")
    positions = pd.read_csv(positions_path)
    required = {"datetime", "target_position", "outer_test_year", "segment"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"source opportunity positions are missing columns: {sorted(missing)}")
    positions["datetime"] = pd.to_datetime(positions["datetime"]).dt.normalize()
    positions = positions.loc[positions["segment"].eq("outer_test")].copy()
    positions["target_position"] = pd.to_numeric(positions["target_position"], errors="coerce")
    positions = positions.dropna(subset=["target_position"]).sort_values("datetime")
    if positions["datetime"].duplicated().any():
        raise ValueError("source opportunity positions contain duplicate dates")
    audit = {
        "experiment": spec.source_experiment,
        "summary": summary_path.resolve().as_posix(),
        "recorder_id": summary["recorder_id"],
        "recorder_local_dir": recorder_dir.resolve().as_posix(),
        "positions_artifact": positions_path.resolve().as_posix(),
        "selection_uses_outer_test": False,
        "outer_years": sorted(int(value) for value in positions["outer_test_year"].unique()),
        "position_field": "target_position",
        "ignored_source_return_fields": ["strategy_return", "open_to_open_return"],
        "return_replay": "qlib_SOXL_SOXX_open_t_plus_1_to_open_t_plus_2",
    }
    return positions, audit


def _align_market_and_opportunity(
    market: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    outer_years: tuple[int, ...],
) -> pd.DataFrame:
    aligned = market.merge(
        positions.loc[:, ["datetime", "target_position", "outer_test_year"]],
        left_on="signal_date",
        right_on="datetime",
        how="inner",
        validate="one_to_one",
    )
    aligned = aligned.loc[aligned["outer_test_year"].astype(int).isin(outer_years)].copy()
    aligned = aligned.loc[aligned["trade_date"].dt.year.eq(aligned["outer_test_year"].astype(int))].copy()
    aligned = aligned.sort_values("signal_date").reset_index(drop=True)
    present_years = sorted(int(value) for value in aligned["outer_test_year"].unique())
    if present_years != list(outer_years):
        raise ValueError(f"aligned opportunity data must cover outer years {list(outer_years)}; got {present_years}")
    return aligned


def _strategy_definitions(spec: SmartLeverageSoxlWealthControlSpec) -> dict[str, dict[str, Any]]:
    return {
        "nested_opportunity_clean_soxl": {
            "allocation_mode": "soxl_only",
            "wealth_lock": False,
        },
        "nested_opportunity_beta_ladder": {
            "allocation_mode": "beta_ladder",
            "wealth_lock": False,
        },
        "smart_leverage_asymmetric_lock_soxl": {
            "allocation_mode": "soxl_only",
            "wealth_lock": True,
        },
        "smart_leverage_asymmetric_lock_beta_ladder": {
            "allocation_mode": "beta_ladder",
            "wealth_lock": True,
        },
    }


def _evaluate_strategies(
    frame: pd.DataFrame,
    *,
    spec: SmartLeverageSoxlWealthControlSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    positions_rows: list[pd.DataFrame] = []
    returns_rows: list[pd.DataFrame] = []
    definitions = _strategy_definitions(spec)
    for cost_bps in spec.costs_bps:
        for strategy, definition in definitions.items():
            result = simulate_wealth_control(
                frame,
                frame["target_position"],
                cost_bps=cost_bps,
                allocation_mode=str(definition["allocation_mode"]),
                wealth_lock=bool(definition["wealth_lock"]),
                benchmark_soxx_weight=spec.benchmark_soxx_weight,
                wealth_lock_strength=spec.wealth_lock_strength,
                wealth_lock_floor=spec.wealth_lock_floor,
                target_surplus=spec.target_surplus,
            )
            metrics = _risk_metrics(result)
            avg_soxl = float(result["soxl_weight"].mean())
            same = _simulate_weights(
                frame,
                soxl_weight=np.full(len(frame), avg_soxl),
                soxx_weight=np.zeros(len(frame)),
                cost_bps=cost_bps,
            )
            same_metrics = _risk_metrics(same)
            matched_soxl, matched_soxx = _matched_beta_weights(metrics["avg_effective_beta"], len(frame))
            matched = _simulate_weights(
                frame,
                soxl_weight=matched_soxl,
                soxx_weight=matched_soxx,
                cost_bps=cost_bps,
            )
            matched_metrics = _risk_metrics(matched)
            row = {
                "strategy": strategy,
                "cost_bps": cost_bps,
                **metrics,
                "same_avg_position_calmar": same_metrics["calmar"],
                "same_avg_position_return_pct": same_metrics["return_pct"],
                "matched_beta_calmar": matched_metrics["calmar"],
                "matched_beta_return_pct": matched_metrics["return_pct"],
            }
            row["risk_gate_pass"] = _risk_gate(row)
            row["participation_gate_pass"] = row["avg_soxl_position_pct"] >= 15.0
            row["upside_capture_gate_pass"] = row["upside_capture_pct"] >= 60.0
            row["calmar_same_position_gate_pass"] = row["calmar"] > row["same_avg_position_calmar"]
            row["calmar_matched_beta_gate_pass"] = row["calmar"] > row["matched_beta_calmar"]
            comparison_rows.append(row)
            stored = result.copy()
            stored["strategy"] = strategy
            stored["cost_bps"] = cost_bps
            positions_rows.append(
                stored.loc[
                    :,
                    [
                        "signal_date",
                        "trade_date",
                        "exit_date",
                        "strategy",
                        "cost_bps",
                        "opportunity_target",
                        "wealth_surplus",
                        "wealth_multiplier",
                        "controlled_target",
                        "soxl_weight",
                        "soxx_weight",
                        "cash_weight",
                        "effective_beta",
                    ],
                ]
            )
            returns_rows.append(stored)
            for year in spec.outer_years:
                year_result = result.loc[result["trade_date"].dt.year.eq(year)].copy()
                year_metrics = _risk_metrics(year_result)
                annual_rows.append(
                    {
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "outer_year": year,
                        **year_metrics,
                        "risk_gate_pass": _risk_gate(year_metrics),
                    }
                )

    comparison = pd.DataFrame(comparison_rows)
    annual = pd.DataFrame(annual_rows)
    year_passes = annual.groupby(["strategy", "cost_bps"])["risk_gate_pass"].sum().rename(
        "outer_years_risk_passed"
    )
    comparison = comparison.merge(year_passes, on=["strategy", "cost_bps"], how="left")
    comparison["outer_year_gate_pass"] = comparison["outer_years_risk_passed"].ge(3)
    gate_columns = [
        "risk_gate_pass",
        "participation_gate_pass",
        "upside_capture_gate_pass",
        "calmar_same_position_gate_pass",
        "calmar_matched_beta_gate_pass",
        "outer_year_gate_pass",
    ]
    comparison["all_required_gates_pass"] = comparison[gate_columns].all(axis=1)
    comparison["gate_count"] = comparison[gate_columns].sum(axis=1)
    comparison["failed_gates"] = comparison.apply(
        lambda row: ",".join(column.removesuffix("_pass") for column in gate_columns if not bool(row[column])),
        axis=1,
    )
    return comparison, annual, pd.concat(positions_rows, ignore_index=True), pd.concat(returns_rows, ignore_index=True)


def _controller_diagnostics(positions: pd.DataFrame) -> pd.DataFrame:
    return (
        positions.groupby(["strategy", "cost_bps"], as_index=False)
        .agg(
            rows=("trade_date", "size"),
            avg_opportunity_target=("opportunity_target", "mean"),
            avg_wealth_multiplier=("wealth_multiplier", "mean"),
            min_wealth_multiplier=("wealth_multiplier", "min"),
            avg_controlled_target=("controlled_target", "mean"),
            avg_soxl_weight=("soxl_weight", "mean"),
            avg_soxx_weight=("soxx_weight", "mean"),
            avg_effective_beta=("effective_beta", "mean"),
        )
        .sort_values(["cost_bps", "strategy"])
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _run_card(spec: SmartLeverageSoxlWealthControlSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, market_audit = _load_market_frame(LeveragedEtfPaperReplaySpec(root=spec.root))
    source_positions, source_audit = _load_nested_opportunity(spec)
    frame = _align_market_and_opportunity(market, source_positions, outer_years=spec.outer_years)
    comparison, annual, positions, strategy_returns = _evaluate_strategies(frame, spec=spec)
    diagnostics = _controller_diagnostics(positions)
    required = comparison.loc[comparison["cost_bps"].eq(10)].copy()
    primary_name = "smart_leverage_asymmetric_lock_beta_ladder"
    primary = required.loc[required["strategy"].eq(primary_name)].iloc[0].to_dict()
    baseline = required.loc[required["strategy"].eq("nested_opportunity_clean_soxl")].iloc[0].to_dict()
    required_pass_count = int(required["all_required_gates_pass"].sum())
    promotion_decision = (
        "diagnostic_pass_requires_new_sealed_forward_confirmation"
        if bool(primary["all_required_gates_pass"])
        else "research_reject_smart_leverage_controller_failed_required_gates"
    )

    sidecars = {
        "strategy_comparison": "strategy_comparison.csv",
        "annual_outer_metrics": "annual_outer_metrics.csv",
        "controller_diagnostics": "controller_diagnostics.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "selected_model": "selected_model.json",
        "failure_modes": "failure_modes.md",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        positions_path = tmp_dir / "positions.csv"
        returns_path = tmp_dir / "strategy_returns.csv"
        positions.to_csv(positions_path, index=False)
        strategy_returns.to_csv(returns_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["recorder_id"],
                outer_years=",".join(str(year) for year in spec.outer_years),
                costs_bps=",".join(str(cost) for cost in spec.costs_bps),
                selection_uses_outer_test="false",
                execution_timing=market_audit["timing"],
                wealth_lock_policy="asymmetric_de_risk_only_no_loss_chasing",
                wealth_lock_strength=spec.wealth_lock_strength,
                wealth_lock_floor=spec.wealth_lock_floor,
                benchmark_soxx_weight=spec.benchmark_soxx_weight,
            )
            R.log_metrics(
                required_cost_pass_count=float(required_pass_count),
                primary_return_pct=float(primary["return_pct"]),
                primary_max_drawdown_pct=float(primary["max_drawdown_pct"]),
                primary_worst_day_pct=float(primary["worst_day_pct"]),
                primary_cvar5_pct=float(primary["cvar5_pct"]),
                primary_upside_capture_pct=float(primary["upside_capture_pct"]),
            )
            recorder.log_artifact(positions_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(returns_path.as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()

            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "promotion_decision": promotion_decision,
                "selection_uses_outer_test": False,
                "outer_test_use": "evaluation_only; architecture was designed after prior outer results and requires future sealed confirmation",
                "outer_years": list(spec.outer_years),
                "data_audit": market_audit,
                "source_opportunity_audit": source_audit,
                "execution_timing": market_audit["timing"],
                "cash_return_assumption": 0.0,
                "bond_data_status": "needs_data_pit_tbill_tbond_not_in_current_qlib_store",
                "controller": {
                    "policy": "asymmetric_de_risk_only_no_loss_chasing",
                    "inputs": ["strategy_wealth", "benchmark_wealth", "time_order"],
                    "benchmark": f"{100.0 * spec.benchmark_soxx_weight:.0f}% SOXX plus zero-return cash",
                    "target_surplus": spec.target_surplus,
                    "strength": spec.wealth_lock_strength,
                    "multiplier_floor": spec.wealth_lock_floor,
                    "allocation_modes": ["soxl_only", "beta_preserving_soxl_soxx_ladder"],
                },
                "gates": {
                    "max_drawdown_pct_min": -20.0,
                    "worst_day_pct_min": -12.0,
                    "cvar5_pct_min": -8.0,
                    "worst_rolling_5d_pct_min": -12.0,
                    "avg_soxl_position_pct_min": 15.0,
                    "upside_capture_pct_min": 60.0,
                    "outer_years_risk_passed_min": 3,
                    "calmar_must_exceed_same_avg_soxl": True,
                    "calmar_must_exceed_matched_beta_soxl_soxx_cash": True,
                },
                "required_cost_bps": 10,
                "stress_cost_bps": 25,
                "required_cost_pass_count": required_pass_count,
                "primary_candidate": _json_ready(primary),
                "clean_opportunity_baseline": _json_ready(baseline),
                "incremental_effect": {
                    "return_pct": float(primary["return_pct"] - baseline["return_pct"]),
                    "max_drawdown_pct": float(primary["max_drawdown_pct"] - baseline["max_drawdown_pct"]),
                    "worst_day_pct": float(primary["worst_day_pct"] - baseline["worst_day_pct"]),
                    "cvar5_pct": float(primary["cvar5_pct"] - baseline["cvar5_pct"]),
                    "worst_rolling_5d_pct": float(
                        primary["worst_rolling_5d_pct"] - baseline["worst_rolling_5d_pct"]
                    ),
                    "upside_capture_pct": float(primary["upside_capture_pct"] - baseline["upside_capture_pct"]),
                },
                "comparison_preview_10bps": _json_ready(required.to_dict(orient="records")),
                "recorder_id": recorder_id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "sidecars": sidecars,
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
                },
            }
            _write_card_files(card, summary, comparison, annual, diagnostics)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _write_card_files(
    card: Path,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    comparison.to_csv(card / "strategy_comparison.csv", index=False)
    annual.to_csv(card / "annual_outer_metrics.csv", index=False)
    diagnostics.to_csv(card / "controller_diagnostics.csv", index=False)
    risk_columns = [
        "strategy",
        "cost_bps",
        "max_drawdown_pct",
        "worst_day_pct",
        "cvar5_pct",
        "worst_rolling_5d_pct",
        "annualized_vol_pct",
        "risk_gate_pass",
    ]
    comparison.loc[:, risk_columns].to_csv(card / "risk_analysis.csv", index=False)
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "required_cost_pass_count": summary["required_cost_pass_count"],
        "primary_candidate": summary["primary_candidate"],
        "clean_opportunity_baseline": summary["clean_opportunity_baseline"],
        "incremental_effect": summary["incremental_effect"],
    }
    (card / "analysis_summary.json").write_text(json.dumps(_json_ready(analysis), indent=2) + "\n", encoding="utf-8")
    selected_model = {
        "status": "diagnostic_fixed_primary_not_outer_selected",
        "candidate": summary["primary_candidate"]["strategy"],
        "source_recorder_id": summary["source_opportunity_audit"]["recorder_id"],
        "controller": summary["controller"],
        "selection_uses_outer_test": False,
    }
    (card / "selected_model.json").write_text(
        json.dumps(_json_ready(selected_model), indent=2) + "\n", encoding="utf-8"
    )
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    primary = summary["primary_candidate"]
    baseline = summary["clean_opportunity_baseline"]
    incremental = summary["incremental_effect"]
    conclusion = f"""# 结论

## 跑了什么

复用 `{summary['source_opportunity_audit']['experiment']}` 的 nested outer 目标仓位，但丢弃旧 `strategy_return`，重新使用 qlib SOXL/SOXX 开盘价按 close(t) 信号、next open 到 following open 收益计算。固定测试非对称财富锁盈和 beta-preserving SOXL/SOXX 梯形配置；outer 2023–2026 不参与控制器参数选择。

## 核心结果

10bp 主方案 `{primary['strategy']}`：总收益 {primary['return_pct']:.2f}%，CAGR {primary['cagr_pct']:.2f}%，MaxDD {primary['max_drawdown_pct']:.2f}%，单日最差 {primary['worst_day_pct']:.2f}%，CVaR5 {primary['cvar5_pct']:.2f}%，滚动五日最差 {primary['worst_rolling_5d_pct']:.2f}%，SOXL 平均仓位 {primary['avg_soxl_position_pct']:.2f}%，上涨捕获 {primary['upside_capture_pct']:.2f}%。正式全门槛通过数为 {summary['required_cost_pass_count']}。

## 增量效果

相对干净重算的 `{baseline['strategy']}`，主方案收益变化 {incremental['return_pct']:+.2f} 个百分点，MaxDD 改善 {incremental['max_drawdown_pct']:+.2f} 个百分点，单日最差改善 {incremental['worst_day_pct']:+.2f} 个百分点，CVaR5 改善 {incremental['cvar5_pct']:+.2f} 个百分点，上涨捕获变化 {incremental['upside_capture_pct']:+.2f} 个百分点。

## 决定

`{summary['promotion_decision']}`。这是论文启发的诊断实现，不是原论文忠实复刻；当前 qlib store 缺少 PIT T-bill/T-bond，债券状态为 `{summary['bond_data_status']}`。此外该架构是在看过既有 outer 结果后提出，任何看似通过的结果仍需方案冻结后的未来 paper-sim 确认。

## 风险与下一步

SOXX 当前成分股特征仍有 survivorship 风险；本卡只复用来源模型的目标仓位，不把旧盘中回放收益作为证据。下一步只在结果显示尾部改善且参与度未坍塌时，才值得补 PIT 债券并训练路径级 LFNN；否则停止该方向。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    failure_modes = """# Failure modes

- The paper's PIT T-bill and Treasury sleeves are unavailable in the current qlib store; zero-return cash is an explicitly disclosed diagnostic approximation.
- The controller is intentionally asymmetric and cannot increase risk when behind; this differs from the paper's symmetric target-tracking intuition.
- The opportunity target comes from a prior nested workflow whose constituent breadth inputs remain survivorship-biased until PIT holdings are available.
- Outer 2023-2026 is evaluation-only, but the architecture was conceived after prior results were observed, so promotion requires a newly sealed forward period.
"""
    (card / "failure_modes.md").write_text(failure_modes, encoding="utf-8")
    workflow = f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
provider_uri: {summary['data_audit']['provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
as_of: {summary['data_audit']['as_of']}
qlib:
  market: all
  region: us
  data_api: [qlib.init, qlib.data.D.calendar, qlib.data.D.features]
  analysis_api: [qlib.contrib.evaluate.risk_analysis]
execution:
  timing: {summary['execution_timing']}
  costs_bps: [0, 10, 25]
source_opportunity:
  experiment: {summary['source_opportunity_audit']['experiment']}
  recorder_id: {summary['source_opportunity_audit']['recorder_id']}
selection:
  outer_years: [2023, 2024, 2025, 2026]
  selection_uses_outer_test: false
controller:
  policy: asymmetric_de_risk_only_no_loss_chasing
  wealth_lock_strength: {summary['controller']['strength']}
  wealth_lock_floor: {summary['controller']['multiplier_floor']}
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")


if __name__ == "__main__":
    result = run_smart_leverage_soxl_wealth_control_card()
    print(json.dumps(result, indent=2))
