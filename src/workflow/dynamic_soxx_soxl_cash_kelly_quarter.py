from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v10 import (
    MAX_DRAWDOWN_FLOOR_PCT,
    SOURCE_CARD,
    SOURCE_TEMPLATE_ARTIFACT,
    SOXX_EXCESS_FLOOR_PCT,
    _attach_qlib_return_fields,
    _benchmark_metrics,
    _load_source_template_positions,
    _raw_template_metrics,
    _remove_noisy_recorder_params,
    _return_drawdown,
    _simulate_policy as _simulate_fixed_risk_budget_policy,
    _soxl_buyhold_metrics,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_kelly_quarter_v1"
CARD_TYPE = "dynamic_soxx_soxl_cash_quarter_kelly_template_card"


@dataclass(frozen=True)
class DynamicSoxxSoxlCashKellyQuarterSpec:
    root: Path
    experiment: str = EXPERIMENT
    kelly_fraction: float = 0.25
    selected_lookback: int = 63
    selected_min_observations: int = 20
    selected_max_kelly_scale: float = 0.45
    selected_soxx_floor: float = 0.1
    variance_floor: float = 1.0e-12
    cost_per_turnover: float = 0.0003
    test_years: tuple[int, ...] = (2023, 2024, 2025, 2026)

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "dynamic_soxx_soxl_cash_allocation_wf_v1"

    @property
    def raw_price_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def source_card_dir(self) -> Path:
        return self.root / SOURCE_CARD


def run_dynamic_soxx_soxl_cash_kelly_quarter_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashKellyQuarterSpec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashKellyQuarterSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    template_positions = _load_source_template_positions(spec, source_summary=source_summary)
    feature_frame, data_audit = _attach_qlib_return_fields(spec, template_positions)

    selected_policy_id = _policy_id(
        lookback=spec.selected_lookback,
        min_observations=spec.selected_min_observations,
        kelly_fraction=spec.kelly_fraction,
        max_kelly_scale=spec.selected_max_kelly_scale,
        soxx_floor=spec.selected_soxx_floor,
    )
    variants = _build_quarter_kelly_variants(feature_frame, spec=spec)
    if selected_policy_id not in variants:
        raise ValueError(f"selected policy missing from variant grid: {selected_policy_id}")

    selected = variants[selected_policy_id]
    benchmark = _benchmark_metrics(feature_frame)
    raw_template = _raw_template_metrics(feature_frame)
    soxl_buyhold = _soxl_buyhold_metrics(feature_frame)
    fixed_risk_budget = _simulate_fixed_risk_budget_policy(
        feature_frame,
        scale=0.4,
        soxx_floor=0.1,
        spec=spec,
        policy_id="v10_fixed_risk_budget_scale0p40_soxxfloor0p10",
    )
    walk_forward = _walk_forward_metrics(
        feature_frame,
        selected=selected,
        fixed_risk_budget=fixed_risk_budget,
        spec=spec,
    )
    model_comparison = _model_comparison_frame(
        variants=variants,
        benchmark=benchmark,
        raw_template=raw_template,
        soxl_buyhold=soxl_buyhold,
        fixed_risk_budget=fixed_risk_budget,
    )
    backtest_comparison = _backtest_benchmark_comparison(
        selected=selected,
        benchmark=benchmark,
        raw_template=raw_template,
        soxl_buyhold=soxl_buyhold,
        fixed_risk_budget=fixed_risk_budget,
    )
    risk_analysis = _risk_analysis_frame(
        selected=selected,
        benchmark=benchmark,
        fixed_risk_budget=fixed_risk_budget,
        source_summary=source_summary,
        spec=spec,
    )
    analysis_summary = _analysis_summary(
        selected=selected,
        benchmark=benchmark,
        raw_template=raw_template,
        soxl_buyhold=soxl_buyhold,
        fixed_risk_budget=fixed_risk_budget,
        walk_forward=walk_forward,
        model_comparison=model_comparison,
    )
    promotion_decision = (
        "candidate_for_validated_research_soxx_excess_goal_passed"
        if bool(selected["passes_target"])
        else "not_promoted_quarter_kelly_failed_current_goal"
    )

    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "failure_modes": "failure_modes.md",
        "kelly_walk_forward": "kelly_walk_forward.csv",
        "model_comparison": "model_comparison.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
    }

    import qlib
    from qlib.workflow import R

    _ = qlib
    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        selected_positions = _selected_positions_frame(feature_frame, selected=selected)
        selected_positions_path = tmp_dir / "selected_positions.csv"
        selected_positions.to_csv(selected_positions_path, index=False)
        model_comparison_path = tmp_dir / "model_comparison.csv"
        model_comparison.to_csv(model_comparison_path, index=False)
        walk_forward_path = tmp_dir / "kelly_walk_forward.csv"
        walk_forward.to_csv(walk_forward_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                raw_price_provider_uri=spec.raw_price_provider_uri.resolve().as_posix(),
                source_card=SOURCE_CARD,
                source_template_artifact=SOURCE_TEMPLATE_ARTIFACT,
                benchmark="SOXX",
                benchmark_basis="open_to_open",
                allocation_assets="SOXX,SOXL,cash",
                sizing_method="quarter_kelly_rolling_source_template_returns",
                kelly_fraction=float(spec.kelly_fraction),
                selected_lookback=int(spec.selected_lookback),
                selected_min_observations=int(spec.selected_min_observations),
                selected_max_kelly_scale=float(spec.selected_max_kelly_scale),
                selected_soxx_floor=float(spec.selected_soxx_floor),
                selection_uses_test_segment=False,
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                selected_return_pct=float(selected["return_pct"]),
                selected_max_drawdown_pct=float(selected["max_drawdown_pct"]),
                selected_excess_vs_soxx_pct=float(selected["excess_vs_soxx_pct"]),
                fixed_risk_budget_return_pct=float(fixed_risk_budget["return_pct"]),
                fixed_risk_budget_max_drawdown_pct=float(fixed_risk_budget["max_drawdown_pct"]),
            )
            recorder.log_artifact(selected_positions_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(model_comparison_path.as_posix(), artifact_path=spec.experiment)
            recorder.log_artifact(walk_forward_path.as_posix(), artifact_path=spec.experiment)
            summary = _summary_payload(
                spec=spec,
                source_summary=source_summary,
                data_audit=data_audit,
                selected=selected,
                benchmark=benchmark,
                raw_template=raw_template,
                soxl_buyhold=soxl_buyhold,
                fixed_risk_budget=fixed_risk_budget,
                walk_forward=walk_forward,
                model_comparison=model_comparison,
                analysis_summary=analysis_summary,
                promotion_decision=promotion_decision,
                recorder_id=recorder.id,
                recorder_uri=recorder_uri,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                sidecars=sidecars,
            )
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                model_comparison=model_comparison,
                backtest_comparison=backtest_comparison,
                risk_analysis=risk_analysis,
                walk_forward=walk_forward,
                analysis_summary=analysis_summary,
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            recorder_dir = Path(recorder.get_local_dir()).resolve()
            _remove_qlib_code_snapshot_artifacts(recorder_dir)
            _remove_noisy_recorder_params(recorder_dir)
    _write_summary_files(card, summary)
    return summary


def _quarter_kelly_sizing_frame(
    frame: pd.DataFrame,
    *,
    lookback: int,
    min_observations: int,
    kelly_fraction: float,
    max_kelly_scale: float,
    variance_floor: float,
) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if min_observations <= 1:
        raise ValueError("min_observations must be greater than 1")
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be in (0, 1]")
    if max_kelly_scale < 0.0:
        raise ValueError("max_kelly_scale must be non-negative")

    out = frame.copy()
    source_returns = pd.to_numeric(out["template_strategy_return"], errors="coerce").fillna(0.0)
    prior_returns = source_returns.shift(1)
    rolling = prior_returns.rolling(window=int(lookback), min_periods=int(min_observations))
    rolling_mean = rolling.mean()
    rolling_variance = rolling.var(ddof=1)
    observation_count = rolling.count()
    valid_edge = (
        rolling_mean.gt(0.0)
        & rolling_variance.gt(float(variance_floor))
        & observation_count.ge(int(min_observations))
    )
    full_kelly = (rolling_mean / rolling_variance).where(valid_edge, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    kelly_scale = (full_kelly * float(kelly_fraction)).clip(lower=0.0, upper=float(max_kelly_scale))
    target_position = pd.to_numeric(out["template_target_position"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    out["kelly_lookback"] = int(lookback)
    out["kelly_min_observations"] = int(min_observations)
    out["kelly_observation_count"] = observation_count.fillna(0).astype(int)
    out["kelly_rolling_mean"] = rolling_mean.fillna(0.0)
    out["kelly_rolling_variance"] = rolling_variance.fillna(0.0)
    out["kelly_full_fraction"] = full_kelly.astype("float64")
    out["kelly_fraction"] = float(kelly_fraction)
    out["kelly_scale"] = kelly_scale.astype("float64")
    out["template_target_position"] = target_position
    out["soxl_weight"] = target_position.to_numpy(dtype="float64") * out["kelly_scale"].to_numpy(dtype="float64")
    return out


def _simulate_quarter_kelly_policy(
    frame: pd.DataFrame,
    *,
    lookback: int,
    min_observations: int,
    kelly_fraction: float,
    max_kelly_scale: float,
    soxx_floor: float,
    cost_per_turnover: float,
    policy_id: str,
    variance_floor: float = 1.0e-12,
) -> dict[str, Any]:
    sized = _quarter_kelly_sizing_frame(
        frame,
        lookback=lookback,
        min_observations=min_observations,
        kelly_fraction=kelly_fraction,
        max_kelly_scale=max_kelly_scale,
        variance_floor=variance_floor,
    )
    position = pd.to_numeric(sized["template_target_position"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    soxl_weight = pd.to_numeric(sized["soxl_weight"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    soxx_weight = (1.0 - position.to_numpy(dtype="float64")) * float(soxx_floor)
    total_weight = soxl_weight + soxx_weight
    over = total_weight > 1.0
    if over.any():
        soxl_weight[over] = soxl_weight[over] / total_weight[over]
        soxx_weight[over] = soxx_weight[over] / total_weight[over]
        total_weight = soxl_weight + soxx_weight
    cash_weight = 1.0 - total_weight
    soxl_turnover = np.abs(soxl_weight - np.r_[0.0, soxl_weight[:-1]])
    soxx_turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]])
    kelly_scale = pd.to_numeric(sized["kelly_scale"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    returns = (
        kelly_scale
        * pd.to_numeric(sized["template_strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        + soxx_weight
        * pd.to_numeric(sized["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        - (soxl_turnover + soxx_turnover) * float(cost_per_turnover)
    )
    ret, maxdd = _return_drawdown(returns)
    benchmark_ret, _ = _return_drawdown(
        pd.to_numeric(sized["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    )
    excess = float(ret - benchmark_ret)
    return {
        "policy_id": policy_id,
        "sizing_method": "quarter_kelly_rolling_source_template_returns",
        "lookback": int(lookback),
        "min_observations": int(min_observations),
        "kelly_fraction": float(kelly_fraction),
        "max_kelly_scale": float(max_kelly_scale),
        "soxx_floor": float(soxx_floor),
        "return_pct": float(ret),
        "max_drawdown_pct": float(maxdd),
        "excess_vs_soxx_pct": float(excess),
        "passes_target": bool(excess >= SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT),
        "selection_uses_test_segment": False,
        "parameter_source": "fixed_predeclared_quarter_kelly_overlay_not_test_selected",
        "return_source": "shifted_rolling_source_template_edge_scaled_plus_qlib_soxx_floor",
        "avg_kelly_scale": float(np.mean(kelly_scale)) if len(kelly_scale) else 0.0,
        "max_observed_kelly_scale": float(np.max(kelly_scale)) if len(kelly_scale) else 0.0,
        "avg_soxl_weight": float(np.mean(soxl_weight)) if len(soxl_weight) else 0.0,
        "avg_soxx_weight": float(np.mean(soxx_weight)) if len(soxx_weight) else 0.0,
        "avg_cash_weight": float(np.mean(cash_weight)) if len(cash_weight) else 0.0,
        "max_soxl_weight": float(np.max(soxl_weight)) if len(soxl_weight) else 0.0,
        "max_soxx_weight": float(np.max(soxx_weight)) if len(soxx_weight) else 0.0,
        "active_soxl_day_count": int((soxl_weight > 0.001).sum()),
        "active_soxx_floor_day_count": int((soxx_weight > 0.001).sum()),
        "total_soxl_turnover": float(soxl_turnover.sum()),
        "total_soxx_floor_turnover": float(soxx_turnover.sum()),
        "returns": returns,
        "soxl_weight": soxl_weight,
        "soxx_weight": soxx_weight,
        "cash_weight": cash_weight,
        "kelly_scale": kelly_scale,
        "kelly_full_fraction": pd.to_numeric(sized["kelly_full_fraction"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype="float64"),
        "kelly_rolling_mean": pd.to_numeric(sized["kelly_rolling_mean"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype="float64"),
        "kelly_rolling_variance": pd.to_numeric(sized["kelly_rolling_variance"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype="float64"),
        "kelly_observation_count": pd.to_numeric(sized["kelly_observation_count"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int64"),
    }


def _build_quarter_kelly_variants(
    frame: pd.DataFrame,
    *,
    spec: DynamicSoxxSoxlCashKellyQuarterSpec,
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for lookback, min_observations in [(21, 10), (63, 20), (126, 40)]:
        for max_kelly_scale in [0.25, 0.35, 0.45]:
            for soxx_floor in [0.0, 0.1, 0.2]:
                policy_id = _policy_id(
                    lookback=lookback,
                    min_observations=min_observations,
                    kelly_fraction=spec.kelly_fraction,
                    max_kelly_scale=max_kelly_scale,
                    soxx_floor=soxx_floor,
                )
                variants[policy_id] = _simulate_quarter_kelly_policy(
                    frame,
                    lookback=lookback,
                    min_observations=min_observations,
                    kelly_fraction=spec.kelly_fraction,
                    max_kelly_scale=max_kelly_scale,
                    soxx_floor=soxx_floor,
                    cost_per_turnover=spec.cost_per_turnover,
                    policy_id=policy_id,
                    variance_floor=spec.variance_floor,
                )
    return variants


def _walk_forward_metrics(
    frame: pd.DataFrame,
    *,
    selected: dict[str, Any],
    fixed_risk_budget: dict[str, Any],
    spec: DynamicSoxxSoxlCashKellyQuarterSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_returns = np.asarray(selected["returns"], dtype="float64")
    fixed_returns = np.asarray(fixed_risk_budget["returns"], dtype="float64")
    kelly_scale = np.asarray(selected["kelly_scale"], dtype="float64")
    soxl_weight = np.asarray(selected["soxl_weight"], dtype="float64")
    for year, group in frame.assign(
        selected_return=selected_returns,
        fixed_risk_budget_return=fixed_returns,
        kelly_scale=kelly_scale,
        soxl_weight=soxl_weight,
    ).groupby("year", sort=True):
        returns = pd.to_numeric(group["selected_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        bench = pd.to_numeric(group["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        raw = pd.to_numeric(group["template_strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        fixed = pd.to_numeric(group["fixed_risk_budget_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        ret, maxdd = _return_drawdown(returns)
        bench_ret, bench_dd = _return_drawdown(bench)
        raw_ret, raw_dd = _return_drawdown(raw)
        fixed_ret, fixed_dd = _return_drawdown(fixed)
        rows.append(
            {
                "test_year": int(year),
                "start": group["datetime"].min().date().isoformat(),
                "end": group["datetime"].max().date().isoformat(),
                "return_pct": float(ret),
                "max_drawdown_pct": float(maxdd),
                "soxx_return_pct": float(bench_ret),
                "soxx_max_drawdown_pct": float(bench_dd),
                "excess_vs_soxx_pct": float(ret - bench_ret),
                "raw_template_return_pct": float(raw_ret),
                "raw_template_max_drawdown_pct": float(raw_dd),
                "fixed_risk_budget_return_pct": float(fixed_ret),
                "fixed_risk_budget_max_drawdown_pct": float(fixed_dd),
                "excess_vs_fixed_risk_budget_pct": float(ret - fixed_ret),
                "avg_kelly_scale": float(pd.to_numeric(group["kelly_scale"], errors="coerce").fillna(0.0).mean()),
                "avg_soxl_weight": float(pd.to_numeric(group["soxl_weight"], errors="coerce").fillna(0.0).mean()),
                "passes_drawdown": bool(maxdd >= MAX_DRAWDOWN_FLOOR_PCT),
                "passes_annual_soxx_excess_20pp": bool(
                    ret >= bench_ret + SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT
                ),
                "selection_uses_test_segment": False,
                "kelly_fraction": float(spec.kelly_fraction),
                "lookback": int(spec.selected_lookback),
                "max_kelly_scale": float(spec.selected_max_kelly_scale),
                "soxx_floor": float(spec.selected_soxx_floor),
            }
        )
    return pd.DataFrame(rows)


def _model_comparison_frame(
    *,
    variants: dict[str, dict[str, Any]],
    benchmark: dict[str, Any],
    raw_template: dict[str, Any],
    soxl_buyhold: dict[str, Any],
    fixed_risk_budget: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        {
            "model": "soxx_buyhold_benchmark",
            "return_pct": benchmark["return_pct"],
            "max_drawdown_pct": benchmark["max_drawdown_pct"],
            "excess_vs_soxx_pct": 0.0,
            "passes_target": False,
            "selection_uses_test_segment": False,
            "role": "benchmark",
        },
        {
            "model": "soxl_buyhold_reference",
            "return_pct": soxl_buyhold["return_pct"],
            "max_drawdown_pct": soxl_buyhold["max_drawdown_pct"],
            "excess_vs_soxx_pct": soxl_buyhold["return_pct"] - benchmark["return_pct"],
            "passes_target": False,
            "selection_uses_test_segment": False,
            "role": "risk_reference",
        },
        {
            "model": "source_regime_template_raw",
            "return_pct": raw_template["return_pct"],
            "max_drawdown_pct": raw_template["max_drawdown_pct"],
            "excess_vs_soxx_pct": raw_template["return_pct"] - benchmark["return_pct"],
            "passes_target": raw_template["max_drawdown_pct"] >= MAX_DRAWDOWN_FLOOR_PCT
            and raw_template["return_pct"] >= benchmark["return_pct"] + SOXX_EXCESS_FLOOR_PCT,
            "selection_uses_test_segment": False,
            "role": "source_alpha",
        },
        {
            "model": fixed_risk_budget["policy_id"],
            "return_pct": fixed_risk_budget["return_pct"],
            "max_drawdown_pct": fixed_risk_budget["max_drawdown_pct"],
            "excess_vs_soxx_pct": fixed_risk_budget["excess_vs_soxx_pct"],
            "passes_target": fixed_risk_budget["passes_target"],
            "selection_uses_test_segment": False,
            "role": "fixed_risk_budget_reference",
        },
    ]
    for policy in variants.values():
        rows.append(
            {
                "model": policy["policy_id"],
                "return_pct": policy["return_pct"],
                "max_drawdown_pct": policy["max_drawdown_pct"],
                "excess_vs_soxx_pct": policy["excess_vs_soxx_pct"],
                "passes_target": policy["passes_target"],
                "selection_uses_test_segment": policy["selection_uses_test_segment"],
                "lookback": policy["lookback"],
                "min_observations": policy["min_observations"],
                "kelly_fraction": policy["kelly_fraction"],
                "max_kelly_scale": policy["max_kelly_scale"],
                "soxx_floor": policy["soxx_floor"],
                "avg_kelly_scale": policy["avg_kelly_scale"],
                "avg_soxl_weight": policy["avg_soxl_weight"],
                "role": "quarter_kelly_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _backtest_benchmark_comparison(
    *,
    selected: dict[str, Any],
    benchmark: dict[str, Any],
    raw_template: dict[str, Any],
    soxl_buyhold: dict[str, Any],
    fixed_risk_budget: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for name, item in [
        ("SOXX", benchmark),
        ("SOXL", soxl_buyhold),
        ("source_regime_template_raw", raw_template),
        ("v10_fixed_risk_budget", fixed_risk_budget),
    ]:
        rows.append(
            {
                "benchmark": name,
                "benchmark_return_pct": float(item["return_pct"]),
                "benchmark_max_drawdown_pct": float(item["max_drawdown_pct"]),
                "selected_return_pct": float(selected["return_pct"]),
                "selected_max_drawdown_pct": float(selected["max_drawdown_pct"]),
                "return_gap_pct": float(selected["return_pct"] - item["return_pct"]),
                "drawdown_gap_pct": float(selected["max_drawdown_pct"] - item["max_drawdown_pct"]),
                "beats_return": bool(selected["return_pct"] > item["return_pct"]),
                "beats_drawdown": bool(selected["max_drawdown_pct"] > item["max_drawdown_pct"]),
                "passes_current_goal": bool(selected["passes_target"]) if name == "SOXX" else False,
            }
        )
    return pd.DataFrame(rows)


def _risk_analysis_frame(
    *,
    selected: dict[str, Any],
    benchmark: dict[str, Any],
    fixed_risk_budget: dict[str, Any],
    source_summary: dict[str, Any],
    spec: DynamicSoxxSoxlCashKellyQuarterSpec,
) -> pd.DataFrame:
    source_template = source_summary.get("regime_template", {})
    return pd.DataFrame(
        [
            {
                "check": "soxx_excess_and_drawdown_goal",
                "status": "pass" if selected["passes_target"] else "fail",
                "details": json.dumps(
                    {
                        "selected_return_pct": selected["return_pct"],
                        "selected_max_drawdown_pct": selected["max_drawdown_pct"],
                        "soxx_return_pct": benchmark["return_pct"],
                        "excess_vs_soxx_pct": selected["excess_vs_soxx_pct"],
                        "required_excess_pct": SOXX_EXCESS_FLOOR_PCT,
                        "required_max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            {
                "check": "kelly_no_lookahead",
                "status": "pass",
                "details": json.dumps(
                    {
                        "edge_estimator": "rolling mean(template_strategy_return.shift(1))",
                        "variance_estimator": "rolling var(template_strategy_return.shift(1), ddof=1)",
                        "lookback": spec.selected_lookback,
                        "min_observations": spec.selected_min_observations,
                        "selection_uses_test_segment": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            {
                "check": "fixed_risk_budget_reference",
                "status": "watch",
                "details": json.dumps(
                    {
                        "fixed_risk_budget_return_pct": fixed_risk_budget["return_pct"],
                        "fixed_risk_budget_max_drawdown_pct": fixed_risk_budget["max_drawdown_pct"],
                        "return_gap_pct": selected["return_pct"] - fixed_risk_budget["return_pct"],
                        "drawdown_gap_pct": selected["max_drawdown_pct"] - fixed_risk_budget["max_drawdown_pct"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            {
                "check": "source_template_dependency",
                "status": "watch",
                "details": json.dumps(
                    {
                        "source_template_selection_uses_test_segment": source_template.get("selection_uses_test_segment"),
                        "source_template_ml_confirmation": source_template.get("ml_confirmation"),
                        "remaining_risk": "quarter Kelly sizes the recorded signal; it does not prove the source template is live-ready",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
    )


def _analysis_summary(
    *,
    selected: dict[str, Any],
    benchmark: dict[str, Any],
    raw_template: dict[str, Any],
    soxl_buyhold: dict[str, Any],
    fixed_risk_budget: dict[str, Any],
    walk_forward: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> dict[str, Any]:
    return _json_ready(
        {
            "source": "qlib_recorder_quarter_kelly_template_replay",
            "selected_policy": _strip_arrays(selected),
            "benchmark": benchmark,
            "raw_template": raw_template,
            "soxl_buyhold": soxl_buyhold,
            "fixed_risk_budget_reference": _strip_arrays(fixed_risk_budget),
            "walk_forward": walk_forward.to_dict("records"),
            "diagnostic_pass_count": int(model_comparison["passes_target"].fillna(False).astype(bool).sum()),
            "best_quarter_kelly_return_pct": float(
                pd.to_numeric(
                    model_comparison.loc[model_comparison["role"].eq("quarter_kelly_diagnostic"), "return_pct"],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .max()
            ),
        }
    )


def _selected_positions_frame(frame: pd.DataFrame, *, selected: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "walk_forward_fold",
        "datetime",
        "year",
        "selected_template_id",
        "template_position_source",
        "template_ml_probability",
        "template_ml_target_position",
        "template_ml_confirmed",
        "template_ml_confirmation_used",
        "selection_uses_test_segment",
        "template_target_position",
        "template_strategy_return",
        "soxx_open_to_open_return",
        "soxl_open_to_open_return",
    ]
    out = frame[[column for column in columns if column in frame.columns]].copy()
    out["policy_id"] = str(selected["policy_id"])
    out["sizing_method"] = str(selected["sizing_method"])
    out["kelly_fraction"] = float(selected["kelly_fraction"])
    out["kelly_lookback"] = int(selected["lookback"])
    out["kelly_min_observations"] = int(selected["min_observations"])
    out["max_kelly_scale"] = float(selected["max_kelly_scale"])
    out["soxx_floor"] = float(selected["soxx_floor"])
    out["kelly_observation_count"] = np.asarray(selected["kelly_observation_count"], dtype="int64")
    out["kelly_rolling_mean"] = np.asarray(selected["kelly_rolling_mean"], dtype="float64")
    out["kelly_rolling_variance"] = np.asarray(selected["kelly_rolling_variance"], dtype="float64")
    out["kelly_full_fraction"] = np.asarray(selected["kelly_full_fraction"], dtype="float64")
    out["kelly_scale"] = np.asarray(selected["kelly_scale"], dtype="float64")
    out["soxl_weight"] = np.asarray(selected["soxl_weight"], dtype="float64")
    out["soxx_weight"] = np.asarray(selected["soxx_weight"], dtype="float64")
    out["cash_weight"] = np.asarray(selected["cash_weight"], dtype="float64")
    out["strategy_return"] = np.asarray(selected["returns"], dtype="float64")
    return out


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashKellyQuarterSpec,
    source_summary: dict[str, Any],
    data_audit: dict[str, Any],
    selected: dict[str, Any],
    benchmark: dict[str, Any],
    raw_template: dict[str, Any],
    soxl_buyhold: dict[str, Any],
    fixed_risk_budget: dict[str, Any],
    walk_forward: pd.DataFrame,
    model_comparison: pd.DataFrame,
    analysis_summary: dict[str, Any],
    promotion_decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
    sidecars: dict[str, str],
) -> dict[str, Any]:
    source_model_families = (
        source_summary.get("position_model", {}).get("model_families", [])
        or ["lgbm_classifier", "lgbm_regressor", "xgb_classifier", "xgb_regressor", "lgb_xgb_consensus"]
    )
    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": CARD_TYPE,
            "promotion_decision": promotion_decision,
            "as_of": data_audit["end"],
            "provider_uri": spec.provider_uri.resolve().as_posix(),
            "raw_price_provider_uri": spec.raw_price_provider_uri.resolve().as_posix(),
            "benchmark_asset": "SOXX",
            "benchmark_basis": "open_to_open",
            "allocation_assets": ["SOXX", "SOXL", "cash"],
            "execution_timing": "close_t_signal_next_open_to_following_open",
            "goal": {
                "excess_vs_soxx_floor_pct": SOXX_EXCESS_FLOOR_PCT,
                "max_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            },
            "source_template": {
                "card": SOURCE_CARD,
                "experiment": source_summary.get("experiment"),
                "recorder_id": source_summary.get("recorder_id"),
                "recorder_local_dir": source_summary.get("recorder_local_dir"),
                "artifact": SOURCE_TEMPLATE_ARTIFACT,
                "selection_uses_test_segment": source_summary.get("regime_template", {}).get(
                    "selection_uses_test_segment"
                ),
                "ml_confirmation": source_summary.get("regime_template", {}).get("ml_confirmation"),
                "model_families": source_model_families,
            },
            "kelly_overlay": {
                "formula": "kelly_scale = clip(0.25 * rolling_mean(template_strategy_return.shift(1)) / rolling_var(template_strategy_return.shift(1)), 0, max_kelly_scale)",
                "kelly_fraction": float(spec.kelly_fraction),
                "selected_lookback": int(spec.selected_lookback),
                "selected_min_observations": int(spec.selected_min_observations),
                "selected_max_kelly_scale": float(spec.selected_max_kelly_scale),
                "selected_soxx_floor": float(spec.selected_soxx_floor),
                "variance_floor": float(spec.variance_floor),
                "selection_uses_test_segment": False,
            },
            "selected_policy": {
                **_strip_arrays(selected),
                "uses_lgb_xgb": True,
                "source_model_families": source_model_families,
                "risk_budget_explanation": "source regime template is fixed from v2; this card replaces v10's constant SOXL scale with a shifted rolling 1/4 Kelly overlay capped at a fixed max scale",
            },
            "benchmark": benchmark,
            "raw_source_template": raw_template,
            "soxl_buyhold_reference": soxl_buyhold,
            "fixed_risk_budget_reference": _strip_arrays(fixed_risk_budget),
            "walk_forward_summary": {
                "years": walk_forward.to_dict("records"),
                "selection_uses_test_segment": False,
                "annual_pass_count": int(walk_forward["passes_annual_soxx_excess_20pp"].sum()),
            },
            "diagnostic_capacity": {
                "selection_uses_test_segment": False,
                "variant_count": int(len(model_comparison)),
                "target_pass_count": int(model_comparison["passes_target"].fillna(False).astype(bool).sum()),
                "variants": model_comparison.to_dict("records"),
            },
            "data_audit": data_audit,
            "analysis_summary_preview": analysis_summary,
            "overfit_risk": {
                "status": "material_and_not_resolved",
                "mitigations": [
                    "Kelly edge and variance estimates are shifted by one row",
                    "selected Kelly fraction, lookback, cap, and SOXX floor are fixed before reading the test result",
                    "benchmarks are recomputed from qlib provider fields",
                ],
                "remaining_risks": [
                    "source v2 regime template remains a hand-designed prior",
                    "rolling Kelly can chase recent realized edge and under-participate after cold starts",
                    "cap choice is a risk-control parameter, not proof of stable alpha",
                ],
            },
            "qlib_docs": {
                "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
                "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            },
            "qlib_native_api": [
                "qlib.init",
                "qlib.data.D.features",
                "qlib.workflow.R.start",
                "qlib.workflow.R.log_params",
                "qlib.workflow.R.log_metrics",
                "qlib.workflow.Recorder.log_artifact",
                "LGBMClassifier",
                "LGBMRegressor",
                "XGBClassifier",
                "XGBRegressor",
            ],
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "primary_artifact": f"{spec.experiment}.ipynb",
                "machine_entrypoint": "summary.json",
                "workflow_config": "workflow.yaml",
                "kept_sidecars": sorted(sidecars.values()),
                "recorder_only_artifacts": [
                    f"artifacts/{spec.experiment}/selected_positions.csv",
                ],
            },
            "recorder_artifacts": [
                f"artifacts/{spec.experiment}/selected_positions.csv",
                f"artifacts/{spec.experiment}/model_comparison.csv",
                f"artifacts/{spec.experiment}/kelly_walk_forward.csv",
                f"artifacts/{spec.experiment}/analysis_summary.json",
                f"artifacts/{spec.experiment}/backtest_benchmark_comparison.csv",
                f"artifacts/{spec.experiment}/risk_analysis.csv",
                f"artifacts/{spec.experiment}/selected_model.json",
            ],
            "workflow_config": "workflow.yaml",
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
        }
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    model_comparison: pd.DataFrame,
    backtest_comparison: pd.DataFrame,
    risk_analysis: pd.DataFrame,
    walk_forward: pd.DataFrame,
    analysis_summary: dict[str, Any],
) -> None:
    model_comparison.to_csv(card / sidecars["model_comparison"], index=False)
    backtest_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    risk_analysis.to_csv(card / sidecars["risk_analysis"], index=False)
    walk_forward.to_csv(card / sidecars["kelly_walk_forward"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(_json_ready(analysis_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected = {
        "selected_model": summary["selected_policy"]["policy_id"],
        "promotion_decision": summary["promotion_decision"],
        "selected_policy": summary["selected_policy"],
        "kelly_overlay": summary["kelly_overlay"],
        "benchmark": summary["benchmark"],
        "fixed_risk_budget_reference": summary["fixed_risk_budget_reference"],
        "source_template": summary["source_template"],
        "overfit_risk": summary["overfit_risk"],
    }
    (card / sidecars["selected_model"]).write_text(
        json.dumps(_json_ready(selected), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (card / sidecars["failure_modes"]).write_text(_failure_modes_text(summary), encoding="utf-8")


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion_text(summary), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")


def _failure_modes_text(summary: dict[str, Any]) -> str:
    selected = summary["selected_policy"]
    benchmark = summary["benchmark"]
    fixed = summary["fixed_risk_budget_reference"]
    return (
        "# Failure Modes\n\n"
        "- kelly_estimator_noise: rolling Kelly uses recent realized source-template returns; noisy edge estimates can change exposure after the regime has already moved.\n"
        "- source_template_dependency: this card sizes the v2 LGB/XGB-confirmed regime template artifact; it does not retrain an independent signal.\n"
        "- cap_dependency: the max Kelly scale is a fixed risk-control cap; without the cap, daily Kelly fractions can be unstable.\n"
        "- benchmark_scope: current evidence is versus SOXX open-to-open and the v10 fixed-risk-budget reference, not versus live execution.\n"
        f"- selected_return={_fmt_pct(selected['return_pct'])}%, selected_maxDD={_fmt_pct(selected['max_drawdown_pct'])}%, "
        f"SOXX_return={_fmt_pct(benchmark['return_pct'])}%, v10_fixed_return={_fmt_pct(fixed['return_pct'])}%.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    selected = summary["selected_policy"]
    benchmark = summary["benchmark"]
    fixed = summary["fixed_risk_budget_reference"]
    overlay = summary["kelly_overlay"]
    return (
        "# 中文结论\n\n"
        "- 做了什么：在现有 v2/v10 source regime template signal 上叠加 1/4 Kelly sizing；Kelly 的 edge/variance 只用 `template_strategy_return.shift(1)` 的过去滚动窗口估计。\n"
        f"- 参数：kelly_fraction={overlay['kelly_fraction']}，lookback={overlay['selected_lookback']}，"
        f"min_obs={overlay['selected_min_observations']}，max_kelly_scale={overlay['selected_max_kelly_scale']}，"
        f"SOXX floor={overlay['selected_soxx_floor']}。\n"
        f"- 核心指标：SOXX return={_fmt_pct(benchmark['return_pct'])}%，maxDD={_fmt_pct(benchmark['max_drawdown_pct'])}%；"
        f"quarter-Kelly selected return={_fmt_pct(selected['return_pct'])}%，maxDD={_fmt_pct(selected['max_drawdown_pct'])}%，"
        f"excess={_fmt_pct(selected['excess_vs_soxx_pct'])}%。\n"
        f"- 对比 v10 固定缩放：v10 fixed return={_fmt_pct(fixed['return_pct'])}%，maxDD={_fmt_pct(fixed['max_drawdown_pct'])}%；"
        f"Kelly return gap={_fmt_pct(selected['return_pct'] - fixed['return_pct'])}pp，"
        f"drawdown gap={_fmt_pct(selected['max_drawdown_pct'] - fixed['max_drawdown_pct'])}pp。\n"
        "- 泛化判断：Kelly 叠加解决的是仓位大小随近期 edge/variance 变化，不解决 source template 自身是否过拟合；所有参数在本卡固定，不用 test segment 选择。\n"
        f"- 是否晋升：{summary['promotion_decision']}；研究证据可用于 sizing 对比，不能直接实盘。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _workflow_yaml(summary: dict[str, Any]) -> str:
    selected = summary["selected_policy"]
    overlay = summary["kelly_overlay"]
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"raw_price_provider_uri: {summary['raw_price_provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            "benchmark: SOXX",
            "benchmark_basis: open_to_open",
            "allocation_assets: [SOXX, SOXL, cash]",
            "model_source:",
            f"  source_card: {SOURCE_CARD}",
            f"  source_template_artifact: {SOURCE_TEMPLATE_ARTIFACT}",
            "  uses_lgb_xgb: true",
            "kelly_overlay:",
            f"  selected_policy: {selected['policy_id']}",
            f"  kelly_fraction: {overlay['kelly_fraction']}",
            f"  lookback: {overlay['selected_lookback']}",
            f"  min_observations: {overlay['selected_min_observations']}",
            f"  max_kelly_scale: {overlay['selected_max_kelly_scale']}",
            f"  soxx_floor: {overlay['selected_soxx_floor']}",
            "  selection_uses_test_segment: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _policy_id(
    *,
    lookback: int,
    min_observations: int,
    kelly_fraction: float,
    max_kelly_scale: float,
    soxx_floor: float,
) -> str:
    return (
        f"quarter_kelly_lkb{lookback}_min{min_observations}_frac{kelly_fraction:.2f}_"
        f"cap{max_kelly_scale:.2f}_soxxfloor{soxx_floor:.2f}"
    ).replace(".", "p")


def _strip_arrays(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in policy.items()
        if key
        not in {
            "returns",
            "soxl_weight",
            "soxx_weight",
            "cash_weight",
            "kelly_scale",
            "kelly_full_fraction",
            "kelly_rolling_mean",
            "kelly_rolling_variance",
            "kelly_observation_count",
        }
    }


def _fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric):
        return "nan"
    if abs(numeric) >= 1_000_000.0:
        return f"{numeric:.4e}"
    return f"{numeric:.4f}"
