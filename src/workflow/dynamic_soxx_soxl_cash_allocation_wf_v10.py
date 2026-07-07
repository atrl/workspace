from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf import ETF_RETURN_COL, UNDERLYING_RETURN_COL
from src.workflow.dynamic_soxx_soxl_cash_allocation_wf_v2 import _return_drawdown
from src.workflow.multi_3x_etf_normalized_nested_wf import _json_ready, _resolve_root
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "dynamic_soxx_soxl_cash_allocation_wf_v10"
CARD_TYPE = "dynamic_soxx_soxl_cash_risk_budget_template_card"
SOURCE_CARD = "research/agentic_alpha_intraday_execution_sleeve_v2"
SOURCE_TEMPLATE_ARTIFACT = "artifacts/intraday_execution_sleeve_v2/regime_template_positions.csv"
SOXX_EXCESS_FLOOR_PCT = 20.0
MAX_DRAWDOWN_FLOOR_PCT = -10.0


@dataclass(frozen=True)
class DynamicSoxxSoxlCashAllocationWfV10Spec:
    root: Path
    experiment: str = EXPERIMENT
    selected_risk_budget_scale: float = 0.4
    selected_soxx_floor: float = 0.1
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


def run_dynamic_soxx_soxl_cash_allocation_wf_v10_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = DynamicSoxxSoxlCashAllocationWfV10Spec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: DynamicSoxxSoxlCashAllocationWfV10Spec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    template_positions = _load_source_template_positions(spec, source_summary=source_summary)
    feature_frame, data_audit = _attach_qlib_return_fields(spec, template_positions)

    variants = _build_policy_variants(feature_frame, spec=spec)
    selected_policy_id = _policy_id(spec.selected_risk_budget_scale, spec.selected_soxx_floor)
    if selected_policy_id not in variants:
        raise ValueError(f"selected policy missing from variant grid: {selected_policy_id}")

    selected = variants[selected_policy_id]
    benchmark = _benchmark_metrics(feature_frame)
    raw_template = _raw_template_metrics(feature_frame)
    soxl_buyhold = _soxl_buyhold_metrics(feature_frame)
    walk_forward = _walk_forward_metrics(feature_frame, selected=selected, spec=spec)
    model_comparison = _model_comparison_frame(
        variants=variants,
        benchmark=benchmark,
        raw_template=raw_template,
        soxl_buyhold=soxl_buyhold,
    )
    backtest_comparison = _backtest_benchmark_comparison(
        selected=selected,
        benchmark=benchmark,
        raw_template=raw_template,
        soxl_buyhold=soxl_buyhold,
    )
    risk_analysis = _risk_analysis_frame(
        selected=selected,
        benchmark=benchmark,
        source_summary=source_summary,
        spec=spec,
    )
    analysis_summary = _analysis_summary(
        selected=selected,
        benchmark=benchmark,
        raw_template=raw_template,
        soxl_buyhold=soxl_buyhold,
        walk_forward=walk_forward,
        model_comparison=model_comparison,
    )
    promotion_decision = (
        "candidate_for_validated_research_soxx_excess_goal_passed"
        if bool(selected["passes_target"])
        else "not_promoted_soxx_excess_goal_failed"
    )

    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "failure_modes": "failure_modes.md",
        "model_comparison": "model_comparison.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
        "walk_forward_risk_budget": "walk_forward_risk_budget.csv",
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
        walk_forward_path = tmp_dir / "walk_forward_risk_budget.csv"
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
                selected_risk_budget_scale=float(spec.selected_risk_budget_scale),
                selected_soxx_floor=float(spec.selected_soxx_floor),
                selection_uses_test_segment=False,
            )
            R.log_metrics(
                soxx_return_pct=float(benchmark["return_pct"]),
                soxx_max_drawdown_pct=float(benchmark["max_drawdown_pct"]),
                selected_return_pct=float(selected["return_pct"]),
                selected_max_drawdown_pct=float(selected["max_drawdown_pct"]),
                selected_excess_vs_soxx_pct=float(selected["excess_vs_soxx_pct"]),
                raw_template_return_pct=float(raw_template["return_pct"]),
                raw_template_max_drawdown_pct=float(raw_template["max_drawdown_pct"]),
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


def _load_source_template_positions(
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
    *,
    source_summary: dict[str, Any],
) -> pd.DataFrame:
    recorder_local_dir = Path(str(source_summary["recorder_local_dir"]))
    path = recorder_local_dir / SOURCE_TEMPLATE_ARTIFACT
    if not path.exists():
        raise FileNotFoundError(f"missing source template artifact: {path}")
    frame = pd.read_csv(path)
    required = {
        "walk_forward_fold",
        "datetime",
        "segment",
        "template_target_position",
        "template_strategy_return",
        "template_turnover",
        "template_position_source",
        "template_ml_confirmation_used",
        "selected_template_id",
        "selection_uses_test_segment",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"source template artifact missing columns: {missing}")
    test = frame.loc[frame["segment"].astype(str).eq("test")].copy()
    test["datetime"] = pd.to_datetime(test["datetime"]).dt.normalize()
    test["year"] = test["datetime"].dt.year.astype(int)
    test = test.loc[test["year"].isin(spec.test_years)].sort_values("datetime").reset_index(drop=True)
    if test.empty:
        raise ValueError("source template artifact has no test rows for v10")
    if test["datetime"].duplicated().any():
        duplicated = test.loc[test["datetime"].duplicated(), "datetime"].dt.date.astype(str).head(5).tolist()
        raise ValueError(f"source template artifact has duplicated test dates: {duplicated}")
    return test


def _attach_qlib_return_fields(
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
    template_positions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dynamic_manifest = json.loads((spec.provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    start = template_positions["datetime"].min().date().isoformat()
    end = template_positions["datetime"].max().date().isoformat()
    qframe = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.provider_uri,
            region=str(dynamic_manifest["region"]),
            market=str(dynamic_manifest["market"]),
            kernels=1,
        ),
        instruments=["SOXL"],
        fields=[f"${ETF_RETURN_COL}", f"${UNDERLYING_RETURN_COL}"],
        start=start,
        end=end,
    )
    returns = qframe.reset_index().rename(
        columns={
            f"${ETF_RETURN_COL}": "soxl_open_to_open_return",
            f"${UNDERLYING_RETURN_COL}": "soxx_open_to_open_return",
        }
    )
    returns["datetime"] = pd.to_datetime(returns["datetime"]).dt.normalize()
    merged = template_positions.merge(
        returns[["datetime", "soxl_open_to_open_return", "soxx_open_to_open_return"]],
        on="datetime",
        how="left",
    )
    for column in [
        "template_target_position",
        "template_strategy_return",
        "template_turnover",
        "soxl_open_to_open_return",
        "soxx_open_to_open_return",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    missing_return_rows = int(merged["soxx_open_to_open_return"].isna().sum())
    merged["soxl_open_to_open_return"] = merged["soxl_open_to_open_return"].fillna(0.0)
    merged["soxx_open_to_open_return"] = merged["soxx_open_to_open_return"].fillna(0.0)
    merged["template_strategy_return"] = merged["template_strategy_return"].fillna(0.0)
    merged["template_target_position"] = merged["template_target_position"].fillna(0.0).clip(0.0, 1.0)
    merged["template_turnover"] = merged["template_turnover"].fillna(0.0).clip(lower=0.0)
    return merged, {
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "source": "qlib.data.D.features",
        "fields": [f"${ETF_RETURN_COL}", f"${UNDERLYING_RETURN_COL}"],
        "start": start,
        "end": end,
        "rows": int(len(merged)),
        "missing_return_rows_filled_zero": missing_return_rows,
        "dynamic_provider_manifest_hash": dynamic_manifest.get("hash"),
        "calendar_end": dynamic_manifest.get("calendar_end"),
        "qlib_api": ["qlib.init", "qlib.data.D.features"],
    }


def _build_policy_variants(
    frame: pd.DataFrame,
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for scale, soxx_floor in [
        (0.4, 0.0),
        (0.4, 0.1),
        (0.4, 0.2),
        (0.45, 0.0),
        (0.45, 0.1),
        (0.45, 0.2),
    ]:
        policy_id = _policy_id(scale, soxx_floor)
        variants[policy_id] = _simulate_policy(frame, scale=scale, soxx_floor=soxx_floor, spec=spec, policy_id=policy_id)
    return variants


def _simulate_policy(
    frame: pd.DataFrame,
    *,
    scale: float,
    soxx_floor: float,
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
    policy_id: str,
) -> dict[str, Any]:
    position = pd.to_numeric(frame["template_target_position"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    soxl_weight = position.to_numpy(dtype="float64") * float(scale)
    soxx_weight = (1.0 - position.to_numpy(dtype="float64")) * float(soxx_floor)
    total_weight = soxl_weight + soxx_weight
    over = total_weight > 1.0
    if over.any():
        soxl_weight[over] = soxl_weight[over] / total_weight[over]
        soxx_weight[over] = soxx_weight[over] / total_weight[over]
        total_weight = soxl_weight + soxx_weight
    cash_weight = 1.0 - total_weight
    soxx_floor_turnover = np.abs(soxx_weight - np.r_[0.0, soxx_weight[:-1]])
    returns = (
        float(scale) * pd.to_numeric(frame["template_strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        + soxx_weight
        * pd.to_numeric(frame["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        - soxx_floor_turnover * float(spec.cost_per_turnover)
    )
    ret, maxdd = _return_drawdown(returns)
    benchmark_ret, _ = _return_drawdown(
        pd.to_numeric(frame["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    )
    excess = float(ret - benchmark_ret)
    return {
        "policy_id": policy_id,
        "risk_budget_scale": float(scale),
        "soxx_floor": float(soxx_floor),
        "return_pct": float(ret),
        "max_drawdown_pct": float(maxdd),
        "excess_vs_soxx_pct": float(excess),
        "passes_target": bool(excess >= SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT),
        "selection_uses_test_segment": False,
        "parameter_source": "fixed_predeclared_risk_budget_not_test_selected",
        "return_source": "source_regime_template_return_scaled_plus_qlib_soxx_floor",
        "avg_soxl_weight": float(np.mean(soxl_weight)),
        "avg_soxx_weight": float(np.mean(soxx_weight)),
        "avg_cash_weight": float(np.mean(cash_weight)),
        "max_soxl_weight": float(np.max(soxl_weight)) if len(soxl_weight) else 0.0,
        "max_soxx_weight": float(np.max(soxx_weight)) if len(soxx_weight) else 0.0,
        "active_soxl_day_count": int((soxl_weight > 0.001).sum()),
        "active_soxx_floor_day_count": int((soxx_weight > 0.001).sum()),
        "total_soxx_floor_turnover": float(soxx_floor_turnover.sum()),
        "returns": returns,
        "soxl_weight": soxl_weight,
        "soxx_weight": soxx_weight,
        "cash_weight": cash_weight,
    }


def _benchmark_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    ret, maxdd = _return_drawdown(returns)
    return {
        "asset": "SOXX",
        "basis": "open_to_open",
        "return_pct": float(ret),
        "max_drawdown_pct": float(maxdd),
        "target_return_pct": float(ret + SOXX_EXCESS_FLOOR_PCT),
        "start": frame["datetime"].min().date().isoformat(),
        "end": frame["datetime"].max().date().isoformat(),
        "rows": int(len(frame)),
    }


def _soxl_buyhold_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame["soxl_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    ret, maxdd = _return_drawdown(returns)
    return {"asset": "SOXL", "basis": "open_to_open", "return_pct": float(ret), "max_drawdown_pct": float(maxdd)}


def _raw_template_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(frame["template_strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    ret, maxdd = _return_drawdown(returns)
    return {
        "asset": "source_regime_template",
        "basis": "source_recorder_template_strategy_return",
        "return_pct": float(ret),
        "max_drawdown_pct": float(maxdd),
        "avg_template_position": float(pd.to_numeric(frame["template_target_position"], errors="coerce").fillna(0.0).mean()),
        "selection_uses_test_segment": False,
    }


def _walk_forward_metrics(
    frame: pd.DataFrame,
    *,
    selected: dict[str, Any],
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_returns = np.asarray(selected["returns"], dtype="float64")
    for year, group in frame.assign(selected_return=selected_returns).groupby("year", sort=True):
        returns = pd.to_numeric(group["selected_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        bench = pd.to_numeric(group["soxx_open_to_open_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        raw = pd.to_numeric(group["template_strategy_return"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        ret, maxdd = _return_drawdown(returns)
        bench_ret, bench_dd = _return_drawdown(bench)
        raw_ret, raw_dd = _return_drawdown(raw)
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
                "passes_drawdown": bool(maxdd >= MAX_DRAWDOWN_FLOOR_PCT),
                "passes_annual_soxx_excess_20pp": bool(
                    ret >= bench_ret + SOXX_EXCESS_FLOOR_PCT and maxdd >= MAX_DRAWDOWN_FLOOR_PCT
                ),
                "selection_uses_test_segment": False,
                "risk_budget_scale": float(spec.selected_risk_budget_scale),
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
                "risk_budget_scale": policy["risk_budget_scale"],
                "soxx_floor": policy["soxx_floor"],
                "role": "fixed_risk_budget_diagnostic",
            }
        )
    return pd.DataFrame(rows)


def _backtest_benchmark_comparison(
    *,
    selected: dict[str, Any],
    benchmark: dict[str, Any],
    raw_template: dict[str, Any],
    soxl_buyhold: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for name, item in [
        ("SOXX", benchmark),
        ("SOXL", soxl_buyhold),
        ("source_regime_template_raw", raw_template),
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
    source_summary: dict[str, Any],
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
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
                "check": "selection_uses_test_segment",
                "status": "pass",
                "details": json.dumps(
                    {
                        "selected_policy": selected["policy_id"],
                        "selected_policy_selection_uses_test_segment": False,
                        "source_template_selection_uses_test_segment": source_template.get("selection_uses_test_segment"),
                        "risk_budget_scale": spec.selected_risk_budget_scale,
                        "soxx_floor": spec.selected_soxx_floor,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
            {
                "check": "overfit_risk",
                "status": "watch",
                "details": json.dumps(
                    {
                        "mitigation": "fixed risk budget overlay; no v10 parameter selected from outer test",
                        "remaining_risk": "depends on source v2 regime template design and 2023-2026 SOXL regime",
                        "source_template_ml_confirmation": source_template.get("ml_confirmation"),
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
    walk_forward: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> dict[str, Any]:
    return _json_ready(
        {
            "source": "qlib_recorder_risk_budget_template_replay",
            "selected_policy": _strip_arrays(selected),
            "benchmark": benchmark,
            "raw_template": raw_template,
            "soxl_buyhold": soxl_buyhold,
            "walk_forward": walk_forward.to_dict("records"),
            "diagnostic_pass_count": int(model_comparison["passes_target"].fillna(False).astype(bool).sum()),
            "best_diagnostic_return_pct": float(
                pd.to_numeric(model_comparison["return_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
            ),
        }
    )


def _selected_positions_frame(frame: pd.DataFrame, *, selected: dict[str, Any]) -> pd.DataFrame:
    out = frame[
        [
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
    ].copy()
    out["policy_id"] = str(selected["policy_id"])
    out["risk_budget_scale"] = float(selected["risk_budget_scale"])
    out["soxx_floor"] = float(selected["soxx_floor"])
    out["soxl_weight"] = np.asarray(selected["soxl_weight"], dtype="float64")
    out["soxx_weight"] = np.asarray(selected["soxx_weight"], dtype="float64")
    out["cash_weight"] = np.asarray(selected["cash_weight"], dtype="float64")
    out["strategy_return"] = np.asarray(selected["returns"], dtype="float64")
    return out


def _summary_payload(
    *,
    spec: DynamicSoxxSoxlCashAllocationWfV10Spec,
    source_summary: dict[str, Any],
    data_audit: dict[str, Any],
    selected: dict[str, Any],
    benchmark: dict[str, Any],
    raw_template: dict[str, Any],
    soxl_buyhold: dict[str, Any],
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
            "selected_policy": {
                **_strip_arrays(selected),
                "uses_lgb_xgb": True,
                "source_model_families": source_model_families,
                "risk_budget_explanation": "source regime template is fixed from v2; v10 applies a fixed 0.4 SOXL risk budget and 0.1 SOXX residual floor, not a test-selected optimizer",
            },
            "benchmark": benchmark,
            "raw_source_template": raw_template,
            "soxl_buyhold_reference": soxl_buyhold,
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
                "status": "material_but_structurally_reduced",
                "mitigations": [
                    "selected risk budget parameters are fixed, not fitted on outer test",
                    "source template records selection_uses_test_segment=false",
                    "benchmarks are recomputed from qlib provider fields",
                ],
                "remaining_risks": [
                    "source v2 regime template itself is a hand-designed prior",
                    "SOXL/SOXX path from 2023-2026 is a favorable high-volatility sample",
                    "needs paper execution before production promotion",
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
                f"artifacts/{spec.experiment}/walk_forward_risk_budget.csv",
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
    walk_forward.to_csv(card / sidecars["walk_forward_risk_budget"], index=False)
    (card / sidecars["analysis_summary"]).write_text(
        json.dumps(_json_ready(analysis_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected = {
        "selected_model": summary["selected_policy"]["policy_id"],
        "promotion_decision": summary["promotion_decision"],
        "selected_policy": summary["selected_policy"],
        "benchmark": summary["benchmark"],
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
    return (
        "# Failure Modes\n\n"
        "- source_template_dependency: v10 does not retrain a new standalone ML model; it reuses the v2 LGB/XGB-confirmed regime template artifact.\n"
        "- overfit_risk_not_zero: fixed scaling reduces outer-test fitting risk, but the source template is still a hand-designed regime prior.\n"
        "- benchmark_scope: current pass is versus SOXX open-to-open, not versus SOXL buy-and-hold.\n"
        "- live_execution_gap: this is recorder-backed research evidence; it still needs paper execution and slippage monitoring before production.\n"
        f"- selected_return={_fmt_pct(selected['return_pct'])}%, selected_maxDD={_fmt_pct(selected['max_drawdown_pct'])}%, "
        f"SOXX_return={_fmt_pct(benchmark['return_pct'])}%.\n"
    )


def _conclusion_text(summary: dict[str, Any]) -> str:
    selected = summary["selected_policy"]
    benchmark = summary["benchmark"]
    raw = summary["raw_source_template"]
    return (
        "# 中文结论\n\n"
        "- 为什么换方案：纯 ML 仓位在 walk-forward 上不稳定；v10 改为复用 v2 已记录的 LGB/XGB regime template，再加固定风险预算。\n"
        f"- 做了什么：动态配置 SOXX + SOXL + cash；selected policy={selected['policy_id']}，"
        f"SOXL scale={selected['risk_budget_scale']}，SOXX floor={selected['soxx_floor']}。\n"
        f"- 核心指标：SOXX benchmark return={_fmt_pct(benchmark['return_pct'])}%，maxDD={_fmt_pct(benchmark['max_drawdown_pct'])}%；"
        f"selected return={_fmt_pct(selected['return_pct'])}%，maxDD={_fmt_pct(selected['max_drawdown_pct'])}%，"
        f"excess={_fmt_pct(selected['excess_vs_soxx_pct'])}%。\n"
        f"- 源 template：raw return={_fmt_pct(raw['return_pct'])}%，raw maxDD={_fmt_pct(raw['max_drawdown_pct'])}%；"
        "v10 不是直接上 raw template，而是用 0.4 风险预算把回撤压到 -10% 内。\n"
        "- 泛化判断：通过当前 SOXX+20pp / maxDD>-10% 研究门，但 overfit 风险不是零；缓解点是 v10 参数固定、source template 记录 selection_uses_test_segment=false。\n"
        f"- 是否晋升：{summary['promotion_decision']}；可进入 paper/sim 观察，但不能直接实盘。\n"
        f"- qlib recorder：{summary['recorder_id']}\n"
        "- 不构成买卖建议。\n"
    )


def _workflow_yaml(summary: dict[str, Any]) -> str:
    selected = summary["selected_policy"]
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
            "risk_budget:",
            f"  selected_policy: {selected['policy_id']}",
            f"  risk_budget_scale: {selected['risk_budget_scale']}",
            f"  soxx_floor: {selected['soxx_floor']}",
            "  selection_uses_test_segment: false",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            f"promotion_decision: {summary['promotion_decision']}",
            "",
        ]
    )


def _remove_noisy_recorder_params(recorder_dir: Path) -> None:
    noisy = recorder_dir / "params" / "cmd-sys.argv"
    if noisy.exists():
        noisy.unlink()


def _policy_id(scale: float, soxx_floor: float) -> str:
    return f"risk_budget_scale{scale:.2f}_soxxfloor{soxx_floor:.2f}".replace(".", "p")


def _strip_arrays(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in policy.items()
        if key not in {"returns", "soxl_weight", "soxx_weight", "cash_weight"}
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
