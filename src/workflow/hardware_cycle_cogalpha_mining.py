from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import (
    BLS_HARDWARE_SALES_PRICE_API_URL,
    BLS_HARDWARE_SALES_PRICE_DEFAULT_RELEASE_LAG_DAYS,
    FRED_HARDWARE_SALES_PRICE_BASE_URL,
    FRED_HARDWARE_SALES_PRICE_SERIES,
    FREE_GPU_PRICE_HISTORY_BASE_URL,
    QlibProviderConfig,
    build_hardware_cycle_qlib_store,
)
from src.factor import (
    CognitiveAlphaFactorMiningSpec,
    build_hardware_cycle_cogalpha_candidates,
    build_hardware_cycle_factor_frame,
    materialize_cognitive_alpha_features,
    run_cognitive_alpha_factor_mining,
)
from src.workflow.cognitive_alpha_evolution import CognitiveAlphaEvolutionSpec
from src.workflow.cognitive_alpha_goal_codex_exec import (
    LABEL_OPEN_TO_OPEN_5D_COL,
    OPEN_TO_OPEN_RETURN_COL,
    QLIB_NATIVE_API,
    _attach_open_to_open_execution_context,
)
from src.workflow.cognitive_alpha_position import run_cognitive_alpha_position_model
from src.workflow.probability_budget_open2open import (
    ProbabilityBudgetOpen2OpenCardSpec,
    _position_spec_for_probability_budget,
)
from src.workflow.risk_hardened_open2open import _build_frame_bundle, _dynamic_open_to_open_benchmarks, _existing_unique, _feature_groups


HARDWARE_COGALPHA_EXPERIMENT = "hardware_cycle_cogalpha_mining_v1"
HARDWARE_COGALPHA_CARD_TYPE = "hardware_cycle_cogalpha_factor_mining_research_card"
HARDWARE_COGALPHA_RECORDER_ARTIFACT_PATH = "hardware_cycle_cogalpha_mining"
HARDWARE_COGALPHA_FEATURE_GROUP = "base_cogalpha_soxx_hardware_cogalpha"
REQUIRED_HARDWARE_DAILY_STATE_FIELDS = {
    "$cpu_compute_index",
    "$enterprise_storage_index",
    "$cloud_gpu_daily_hopper_proxy_index",
    "$cloud_gpu_daily_server_supply_proxy_index",
    "$cloud_gpu_daily_supply_demand_pressure_index",
    "$cloud_gpu_h100_on_demand_source_observed",
    "$cloud_gpu_h100_on_demand_is_carried",
    "$cloud_gpu_h100_on_demand_staleness_days",
    "$free_cloud_gpu_h100_price",
    "$free_cloud_gpu_h100_price_available",
    "$free_cloud_gpu_h100_price_provider_count",
    "$storage_hdd_manufacturer_index",
    "$storage_hdd_cloud_demand_pressure_index",
    "$hardware_sales_storage_device_ppi",
    "$hardware_sales_storage_device_ppi_staleness_days",
}


@dataclass(frozen=True)
class HardwareCycleCogAlphaMiningCardSpec(ProbabilityBudgetOpen2OpenCardSpec):
    root: Path
    experiment: str = HARDWARE_COGALPHA_EXPERIMENT

    @property
    def hardware_cycle_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "hardware_cycle_v1"


def _daily_data_semantics_payload(*, provider_uri: str, raw_manifest: str | None) -> dict[str, Any]:
    return {
        "provider_uri": provider_uri,
        "raw_manifest": raw_manifest,
        "calendar_frequency": "day",
        "cloud_gpu_level_semantics": "daily_proxy_primary_free_direct_short_history_monthly_direct_price_state_diagnostic",
        "cloud_gpu_daily_proxy_semantics": "daily_market_proxy_indices_from_trading_day_closes",
        "free_gpu_direct_quote_semantics": "free_daily_provider_snapshot_prices_no_forward_fill",
        "free_gpu_direct_quote_source": f"{FREE_GPU_PRICE_HISTORY_BASE_URL}/YYYY-MM-DD/all.json",
        "hardware_sales_price_semantics": "official_bls_fred_monthly_sales_price_release_date_state_with_staleness",
        "hardware_sales_price_source": BLS_HARDWARE_SALES_PRICE_API_URL,
        "hardware_sales_price_default_release_lag_days": BLS_HARDWARE_SALES_PRICE_DEFAULT_RELEASE_LAG_DAYS,
        "hardware_sales_price_fred_reference": (
            f"{FRED_HARDWARE_SALES_PRICE_BASE_URL}?id="
            f"{FRED_HARDWARE_SALES_PRICE_SERIES['hardware_sales_storage_device_ppi']['series_id']}"
        ),
        "storage_daily_proxy_semantics": "daily_public_market_hdd_nand_networking_proxy_indices_from_trading_day_closes",
        "cloud_gpu_monthly_state_semantics": "diagnostic_last_observed_state_forward_filled_to_qlib_daily_calendar",
        "source_observed_flag": "<field>_source_observed",
        "carried_state_flag": "<field>_is_carried",
        "staleness_measure": "<field>_staleness_days",
        "period_age_measure": "<field>_period_age_days",
        "publication_lag_measure": "<field>_publication_lag_days",
        "direct_quote_gap_rule": "free direct quote gaps remain missing and are not forward-filled",
        "model_use_rule": (
            "prefer_lagged_daily_proxy_features; official_sales_price_state_is_staleness_gated; "
            "free_direct_daily_price_is_short_history_diagnostic; "
            "monthly_direct_price_state_is_diagnostic"
        ),
        "leakage_guard": "non_trading_source_dates_are_mapped_to_first_qlib_calendar_date_on_or_after_source_date",
    }


def run_hardware_cycle_cogalpha_mining_card(*, root: Path | str | None = None) -> dict[str, Any]:
    spec = HardwareCycleCogAlphaMiningCardSpec(root=_resolve_root(root))
    return _run_card(spec)


def _run_card(spec: HardwareCycleCogAlphaMiningCardSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    base_bundle = _build_frame_bundle(spec)
    benchmarks = _dynamic_open_to_open_benchmarks(base_bundle.frame, test=base_bundle.test)
    if not benchmarks:
        raise ValueError("hardware-cycle CogAlpha card requires dynamic open-to-open benchmarks")

    hardware_manifest = _ensure_hardware_cycle_qlib_provider(spec, as_of=base_bundle.as_of)
    hardware_end = min(pd.Timestamp(base_bundle.as_of), pd.Timestamp(str(hardware_manifest["calendar_end"]))).date().isoformat()
    hardware_factors = build_hardware_cycle_factor_frame(
        QlibProviderConfig(
            provider_uri=spec.hardware_cycle_provider_uri,
            region=str(hardware_manifest["region"]),
            market=str(hardware_manifest["market"]),
            kernels=1,
        ),
        start=spec.train[0],
        end=hardware_end,
        lag_days=1,
    )
    frame, hardware_feature_cols = _attach_hardware_cycle_features(base_bundle.frame, hardware_factors)
    mining_candidates = build_hardware_cycle_cogalpha_candidates()
    mining_frame = frame.loc[pd.to_datetime(frame["datetime"]).dt.normalize() <= pd.Timestamp(spec.validation[1])].copy()

    with tempfile.TemporaryDirectory(prefix="hardware_cycle_cogalpha_mining_") as tmp:
        tmp_dir = Path(tmp)
        factor_mining = run_cognitive_alpha_factor_mining(
            mining_frame,
            spec=CognitiveAlphaFactorMiningSpec(
                evolution_spec=CognitiveAlphaEvolutionSpec(
                    run_id="hardware_cycle_cogalpha_evolution",
                    profile="soxl_smoke",
                    runtime_dir=tmp_dir,
                    generations=1,
                    parent_pool_size=min(6, len(mining_candidates)),
                    children_per_generation=max(1, min(4, len(mining_candidates))),
                    initial_pool_size=len(mining_candidates),
                    qualified_percentile=0.50,
                    elite_percentile=0.75,
                    max_abs_correlation=0.98,
                    min_observations=180,
                    max_missing_ratio=0.75,
                    min_distinct_values=5,
                    fallback_top_n=4,
                    resume=False,
                ),
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                alpha158_columns=tuple(base_bundle.alpha158_columns[:6]),
                max_factor_library_size=6,
                source_layer="src.factor.hardware_cycle_cogalpha",
            ),
            initial_candidates=mining_candidates,
            candidate_provider=None,
        )
        materialized = materialize_cognitive_alpha_features(
            frame,
            list(factor_mining.evolution_result.candidate_map.values()),
            selected_names=factor_mining.selected_feature_names,
        )
        model_frame = pd.concat([frame.reset_index(drop=True), materialized.reset_index(drop=True)], axis=1)
        feature_groups = _hardware_cogalpha_feature_groups(
            _feature_groups(base_bundle),
            selected_cogalpha_feature_cols=factor_mining.selected_feature_names,
            frame=model_frame,
        )
        position_spec = _position_spec_for_probability_budget(
            feature_groups[HARDWARE_COGALPHA_FEATURE_GROUP],
            spec=spec,
            test=base_bundle.test,
        )
        position_result = run_cognitive_alpha_position_model(model_frame, spec=position_spec, benchmarks=benchmarks)
        _attach_open_to_open_execution_context(position_result.summary, position_result.latest_signal)

        sidecars = {
            "selected_model": "selected_model.json",
            "model_selection": "model_selection.csv",
            "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
            "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
            "risk_analysis": "risk_analysis.csv",
            "latest_signal": "latest_signal.json",
            "position_summary": "position_summary.json",
            "feature_sets": "feature_sets.csv",
            "factor_library": "factor_library.json",
            "generation_metrics": "generation_metrics.csv",
            "cognitive_alpha_metrics": "cognitive_alpha_metrics.csv",
            "failure_modes": "failure_modes.md",
        }

        import qlib
        from qlib.workflow import R

        _ = qlib
        recorder_uri = card / "qlib_recorder"
        recorder_uri.mkdir(parents=True, exist_ok=True)
        R.set_uri(recorder_uri.as_posix())

        positions_path = tmp_dir / "positions.csv"
        walk_forward_positions_path = tmp_dir / "walk_forward_positions.csv"
        position_result.positions.to_csv(positions_path, index=False)
        position_result.walk_forward_positions.to_csv(walk_forward_positions_path, index=False)

        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=base_bundle.provider_meta["provider_uri"],
                hardware_cycle_provider_uri=spec.hardware_cycle_provider_uri.as_posix(),
                market=base_bundle.manifest["market"],
                region=base_bundle.manifest["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=base_bundle.test[0],
                test_end=base_bundle.test[1],
                return_col=OPEN_TO_OPEN_RETURN_COL,
                label_col=LABEL_OPEN_TO_OPEN_5D_COL,
                factor_mining="cognitive_alpha_factor_mining",
                hardware_candidate_count=len(mining_candidates),
                selected_factor_count=len(factor_mining.selected_feature_names),
                hardware_feature_count=len(hardware_feature_cols),
                feature_group=HARDWARE_COGALPHA_FEATURE_GROUP,
            )
            summary = _hardware_cogalpha_summary_payload(
                spec=spec,
                provider_meta=base_bundle.provider_meta,
                hardware_manifest=hardware_manifest,
                hardware_feature_cols=hardware_feature_cols,
                factor_mining=factor_mining.summary,
                position_summary=position_result.summary,
                latest_signal=position_result.latest_signal,
                sidecars=sidecars,
                recorder_uri=recorder_uri,
                recorder_id=recorder.id,
                recorder_local_dir=Path(recorder.get_local_dir()).resolve(),
                as_of=base_bundle.as_of,
            )
            summary = _augment_summary_with_results(
                summary,
                position_result=position_result,
                feature_groups=feature_groups,
                factor_mining=factor_mining,
            )
            risk_analysis = _hardware_cogalpha_risk_analysis(summary)
            _write_sidecars(
                card,
                sidecars=sidecars,
                summary=summary,
                position_result=position_result,
                risk_analysis=risk_analysis,
                feature_groups=feature_groups,
                factor_mining=factor_mining,
            )
            _write_summary_files(card, summary, risk_analysis)
            recorder.log_artifact(positions_path.as_posix(), artifact_path=HARDWARE_COGALPHA_RECORDER_ARTIFACT_PATH)
            recorder.log_artifact(
                walk_forward_positions_path.as_posix(),
                artifact_path=HARDWARE_COGALPHA_RECORDER_ARTIFACT_PATH,
            )
            _log_factor_mining_artifacts(recorder, factor_mining.run_dir)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name != "failure_modes.md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
        _remove_transient_code_snapshots(Path(summary["recorder_local_dir"]))
    return summary


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _ensure_hardware_cycle_qlib_provider(
    spec: HardwareCycleCogAlphaMiningCardSpec,
    *,
    as_of: str,
) -> dict[str, Any]:
    manifest_path = spec.hardware_cycle_provider_uri / "qlib_store_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fields = {str(field) for field in manifest.get("fields", [])}
        if str(manifest.get("calendar_end")) >= str(as_of) and REQUIRED_HARDWARE_DAILY_STATE_FIELDS.issubset(fields):
            return manifest
    return build_hardware_cycle_qlib_store(root=spec.root, start=spec.train[0], end=as_of)


def _attach_hardware_cycle_features(frame: pd.DataFrame, hardware_factor_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [
        column
        for column in list(hardware_factor_frame.attrs.get("feature_cols", []))
        if column in hardware_factor_frame.columns and column != "datetime"
    ]
    if not feature_cols:
        raise ValueError("hardware-cycle CogAlpha factor frame produced no feature columns")
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    factors = hardware_factor_frame[["datetime", *feature_cols]].copy()
    factors["datetime"] = pd.to_datetime(factors["datetime"]).dt.normalize()
    return out.merge(factors, on="datetime", how="left"), feature_cols


def _hardware_cogalpha_feature_groups(
    base_groups: dict[str, list[str]],
    *,
    selected_cogalpha_feature_cols: list[str],
    frame: pd.DataFrame,
) -> dict[str, list[str]]:
    groups = dict(base_groups)
    groups[HARDWARE_COGALPHA_FEATURE_GROUP] = _existing_unique(
        [*groups["base_cogalpha_soxx"], *selected_cogalpha_feature_cols],
        frame,
    )
    return groups


def _hardware_cogalpha_summary_payload(
    *,
    spec: HardwareCycleCogAlphaMiningCardSpec,
    provider_meta: dict[str, Any],
    hardware_manifest: dict[str, Any],
    hardware_feature_cols: list[str],
    factor_mining: dict[str, Any],
    position_summary: dict[str, Any],
    latest_signal: dict[str, Any],
    sidecars: dict[str, str],
    recorder_uri: Path,
    recorder_id: str,
    recorder_local_dir: Path,
    as_of: str,
) -> dict[str, Any]:
    return {
        "experiment": spec.experiment,
        "card_type": HARDWARE_COGALPHA_CARD_TYPE,
        "status": "complete",
        "promotion_decision": str(position_summary.get("promotion_decision", "not_promoted_missing_position_result")),
        "as_of": as_of,
        "provider_uri": str(provider_meta.get("provider_uri", "")),
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "segments": {"train": list(spec.train), "validation": list(spec.validation), "test": [spec.test_start, as_of]},
        "benchmark_basis": "open_to_open",
        "daily_data_semantics": _daily_data_semantics_payload(
            provider_uri=str(hardware_manifest.get("provider_uri", "")),
            raw_manifest=hardware_manifest.get("raw_manifest"),
        ),
        "hardware_cycle": {
            "source": "qlib.data.D.features",
            "provider_uri": str(hardware_manifest.get("provider_uri", "")),
            "raw_manifest": hardware_manifest.get("raw_manifest"),
            "calendar_start": hardware_manifest.get("calendar_start"),
            "calendar_end": hardware_manifest.get("calendar_end"),
            "free_gpu_price_source": hardware_manifest.get("free_gpu_price_source"),
            "free_gpu_price_status": hardware_manifest.get("free_gpu_price_status"),
            "free_gpu_price_snapshot_count": hardware_manifest.get("free_gpu_price_snapshot_count"),
            "free_gpu_price_start": hardware_manifest.get("free_gpu_price_start"),
            "free_gpu_price_end": hardware_manifest.get("free_gpu_price_end"),
            "hardware_sales_price_source": hardware_manifest.get("hardware_sales_price_source"),
            "hardware_sales_price_fred_csv_source": hardware_manifest.get("hardware_sales_price_fred_csv_source"),
            "hardware_sales_price_snapshot_source": hardware_manifest.get("hardware_sales_price_snapshot_source"),
            "hardware_sales_price_status": hardware_manifest.get("hardware_sales_price_status"),
            "hardware_sales_price_field_count": hardware_manifest.get("hardware_sales_price_field_count"),
            "hardware_sales_price_observation_count": hardware_manifest.get("hardware_sales_price_observation_count"),
            "hardware_sales_price_start": hardware_manifest.get("hardware_sales_price_start"),
            "hardware_sales_price_end": hardware_manifest.get("hardware_sales_price_end"),
            "hardware_sales_price_series_ids": hardware_manifest.get("hardware_sales_price_series_ids", []),
            "feature_cols": list(hardware_feature_cols),
            "feature_count": int(len(hardware_feature_cols)),
            "role": "lagged_hardware_cycle_feature_surface_for_cognitive_alpha_mining",
        },
        "factor_mining": factor_mining,
        "position_model": position_summary,
        "latest_signal": latest_signal,
        "qlib_native_api": QLIB_NATIVE_API,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        },
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "primary_artifact": f"{spec.experiment}.ipynb",
            "recorder_only_artifacts": {
                "positions": "artifacts/hardware_cycle_cogalpha_mining/positions.csv",
                "walk_forward_positions": "artifacts/hardware_cycle_cogalpha_mining/walk_forward_positions.csv",
                "factor_evolution": "artifacts/hardware_cycle_cogalpha_evolution",
            },
        },
        "spec": _json_safe(asdict(spec)),
    }


def _augment_summary_with_results(
    summary: dict[str, Any],
    *,
    position_result: Any,
    feature_groups: dict[str, list[str]],
    factor_mining: Any,
) -> dict[str, Any]:
    out = dict(summary)
    out["factor_mining"] = {
        **out["factor_mining"],
        "selected_feature_names": list(factor_mining.selected_feature_names),
        "selected_count": int(len(factor_mining.selected_feature_names)),
        "candidate_roles": _candidate_role_counts(factor_mining.factor_library),
    }
    out["position_model"] = {
        **out["position_model"],
        "workflow": "hardware_cycle_cogalpha_position_model",
        "architecture": "lgb_xgb_consensus_probability_budget_with_mined_hardware_cogalpha_features",
    }
    out["benchmark_comparison"] = position_result.benchmark_comparison.to_dict(orient="records")
    out["walk_forward_metrics"] = position_result.walk_forward_metrics.to_dict(orient="records")
    out["feature_sets"] = {
        name: {"feature_count": len(columns), "features": columns}
        for name, columns in feature_groups.items()
    }
    return out


def _candidate_role_counts(factor_library: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in factor_library:
        role = str(item.get("metadata", {}).get("history_role", "unknown"))
        counts[role] = counts.get(role, 0) + 1
    return counts


def _hardware_cogalpha_risk_analysis(summary: dict[str, Any]) -> pd.DataFrame:
    position = summary["position_model"]
    factor_mining = summary["factor_mining"]
    walk_forward = position.get("walk_forward", {})
    return pd.DataFrame(
        [
            {
                "check": "daily_proxy_semantics_present",
                "status": "pass",
                "value": summary["daily_data_semantics"]["model_use_rule"],
                "detail": summary["daily_data_semantics"]["cloud_gpu_daily_proxy_semantics"],
            },
            {
                "check": "cognitive_alpha_selected_factor_count",
                "status": "pass" if int(factor_mining.get("selected_count", 0)) > 0 else "fail",
                "value": int(factor_mining.get("selected_count", 0)),
                "detail": json.dumps(factor_mining.get("selected_feature_names", []), sort_keys=True),
            },
            {
                "check": "selected_position_risk_gates",
                "status": "pass" if position.get("risk_gates", {}).get("selected_passes_all") else "fail",
                "value": position.get("risk_gates", {}).get("selected_passes_all"),
                "detail": json.dumps(position.get("risk_gates", {}), sort_keys=True),
            },
            {
                "check": "walk_forward_risk_gates",
                "status": "pass" if walk_forward.get("all_folds_pass_risk_gates") else "fail",
                "value": walk_forward.get("all_folds_pass_risk_gates"),
                "detail": json.dumps(walk_forward, sort_keys=True),
            },
            {
                "check": "daily_cloud_gpu_proxy_role",
                "status": "pass",
                "value": factor_mining.get("candidate_roles", {}).get("daily_cloud_gpu_proxy_promotable", 0),
                "detail": "daily cloud GPU proxy candidates are promotable only after position and walk-forward gates",
            },
            {
                "check": "free_gpu_direct_price_role",
                "status": "pass",
                "value": factor_mining.get("candidate_roles", {}).get("free_direct_gpu_price_short_history", 0),
                "detail": "free direct GPU price candidates are diagnostic short-history signals unless gates pass",
            },
            {
                "check": "official_storage_sales_price_role",
                "status": "pass",
                "value": factor_mining.get("candidate_roles", {}).get("official_storage_sales_price_state", 0),
                "detail": "official FRED/BLS storage sales price state candidates require lagging and staleness guards",
            },
        ]
    )


def _write_sidecars(
    card: Path,
    *,
    sidecars: dict[str, str],
    summary: dict[str, Any],
    position_result: Any,
    risk_analysis: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    factor_mining: Any,
) -> None:
    (card / sidecars["selected_model"]).write_text(
        json.dumps(summary["position_model"]["selected_candidate"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    position_result.metrics.to_csv(card / sidecars["model_selection"], index=False)
    position_result.benchmark_comparison.to_csv(card / sidecars["backtest_benchmark_comparison"], index=False)
    position_result.walk_forward_metrics.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    risk_analysis.to_csv(card / sidecars["risk_analysis"], index=False)
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(position_result.latest_signal, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["position_summary"]).write_text(
        json.dumps(position_result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"feature_group": name, "feature_count": len(columns), "features": json.dumps(columns)}
            for name, columns in feature_groups.items()
        ]
    ).to_csv(card / sidecars["feature_sets"], index=False)
    (card / sidecars["factor_library"]).write_text(
        json.dumps(factor_mining.factor_library, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    factor_mining.generation_metrics.to_csv(card / sidecars["generation_metrics"], index=False)
    factor_mining.final_report.metrics.to_csv(card / sidecars["cognitive_alpha_metrics"], index=False)


def _write_summary_files(card: Path, summary: dict[str, Any], risk_analysis: pd.DataFrame) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary, risk_analysis), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    _write_notebook(card, summary)


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {summary['experiment']}",
            f"card_type: {summary['card_type']}",
            f"provider_uri: {summary['provider_uri']}",
            f"hardware_cycle_provider_uri: {summary['hardware_cycle']['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            f"recorder_id: {summary['recorder_id']}",
            f"as_of: {summary['as_of']}",
            f"hardware_sales_price_status: {summary['hardware_cycle'].get('hardware_sales_price_status')}",
            f"hardware_sales_price_snapshot_source: {summary['hardware_cycle'].get('hardware_sales_price_snapshot_source')}",
            "daily_data_semantics:",
            f"  cloud_gpu_daily_proxy_semantics: {summary['daily_data_semantics']['cloud_gpu_daily_proxy_semantics']}",
            f"  free_gpu_direct_quote_semantics: {summary['daily_data_semantics']['free_gpu_direct_quote_semantics']}",
            f"  free_gpu_direct_quote_source: {summary['daily_data_semantics']['free_gpu_direct_quote_source']}",
            f"  hardware_sales_price_semantics: {summary['daily_data_semantics']['hardware_sales_price_semantics']}",
            f"  hardware_sales_price_source: {summary['daily_data_semantics']['hardware_sales_price_source']}",
            f"  hardware_sales_price_default_release_lag_days: {summary['daily_data_semantics']['hardware_sales_price_default_release_lag_days']}",
            f"  hardware_sales_price_fred_reference: {summary['daily_data_semantics']['hardware_sales_price_fred_reference']}",
            f"  storage_daily_proxy_semantics: {summary['daily_data_semantics']['storage_daily_proxy_semantics']}",
            f"  cloud_gpu_monthly_state_semantics: {summary['daily_data_semantics']['cloud_gpu_monthly_state_semantics']}",
            f"  source_observed_flag: {summary['daily_data_semantics']['source_observed_flag']}",
            f"  carried_state_flag: {summary['daily_data_semantics']['carried_state_flag']}",
            f"  staleness_measure: {summary['daily_data_semantics']['staleness_measure']}",
            f"  period_age_measure: {summary['daily_data_semantics']['period_age_measure']}",
            f"  publication_lag_measure: {summary['daily_data_semantics']['publication_lag_measure']}",
            f"  model_use_rule: {summary['daily_data_semantics']['model_use_rule']}",
            "factor_mining:",
            "  workflow: cognitive_alpha_factor_mining",
            f"  source_layer: {summary['factor_mining'].get('source_layer')}",
            f"  selected_count: {summary['factor_mining'].get('selected_count')}",
            "position_model:",
            f"  workflow: {summary['position_model']['workflow']}",
            f"  return_col: {OPEN_TO_OPEN_RETURN_COL}",
            f"  label_col: {LABEL_OPEN_TO_OPEN_5D_COL}",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any], risk_analysis: pd.DataFrame) -> str:
    failed = risk_analysis.loc[risk_analysis["status"].eq("fail"), "check"].astype(str).tolist()
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- promotion_decision: {summary['promotion_decision']}",
            f"- failed_checks: {json.dumps(failed, ensure_ascii=False)}",
            f"- selected_factor_count: {summary['factor_mining'].get('selected_count')}",
            f"- candidate_roles: {json.dumps(summary['factor_mining'].get('candidate_roles', {}), sort_keys=True)}",
            "- cloud_daily_proxy_role: 云 GPU 主特征改为日频公开市场代理；真实租赁价格月频状态只作为诊断，不作为主晋升信号。",
            "- free_direct_price_role: 免费日频 GPU 租赁报价已接入，但它是云实例小时租赁价，不是 GPU 销售价；历史覆盖短且缺口不前填，默认只作为诊断候选。",
            "- storage_sales_price_role: FRED/BLS 存储设备 PPI 是官方销售价格指数，但为月频；进入模型时只作为带 staleness 的 lagged 状态。",
            "- storage_daily_proxy_role: 硬盘、NAND/SSD、网络互联日频字段来自公开市场供应链股票代理，不是产品级硬盘报价。",
            "- rejected_storage_price_source: PricePerGig 产品级存储价格因 ML 训练和缓存限制不进入 qlib 特征。",
            "- promotion_rule: 只有 position model 和 walk-forward gates 同时通过，才可进入 validated research；单个硬件因子 IC 不作为晋升依据。",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    selected = summary.get("position_model", {}).get("selected_candidate", {}).get("test", {})
    walk_forward = summary.get("position_model", {}).get("walk_forward", {})
    latest = summary.get("latest_signal", {})
    library = summary.get("factor_mining", {}).get("factor_library", {})
    daily_semantics = summary.get("daily_data_semantics", {})
    selected_names = summary.get("factor_mining", {}).get(
        "selected_feature_names",
        library.get("selected_feature_names", []),
    )
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {summary['experiment']}，将硬件周期 qlib 日频状态特征交给 CognitiveAlpha factor mining，"
        "再把选中硬件因子输入 open-to-open probability budget position model。\n"
        f"- data: hardware_provider={summary['hardware_cycle']['provider_uri']}，"
        f"feature_count={summary['hardware_cycle']['feature_count']}，"
        f"daily_semantics={daily_semantics.get('cloud_gpu_daily_proxy_semantics', 'daily_proxy_semantics_recorded_in_summary')}，"
        f"free_gpu_snapshots={summary['hardware_cycle'].get('free_gpu_price_snapshot_count')} "
        f"({summary['hardware_cycle'].get('free_gpu_price_start')}..{summary['hardware_cycle'].get('free_gpu_price_end')})，"
        f"hardware_sales_ppi_obs={summary['hardware_cycle'].get('hardware_sales_price_observation_count')} "
        f"({summary['hardware_cycle'].get('hardware_sales_price_start')}..{summary['hardware_cycle'].get('hardware_sales_price_end')})，"
        f"sales_status={summary['hardware_cycle'].get('hardware_sales_price_status')}，"
        f"sales_snapshot={summary['hardware_cycle'].get('hardware_sales_price_snapshot_source')}。\n"
        f"- factor_mining: selected_count={summary.get('factor_mining', {}).get('selected_count', library.get('selected_count'))}，"
        f"selected={json.dumps(selected_names, ensure_ascii=False)}。\n"
        f"- key_metrics: selected_return={float(selected.get('return_pct', 0.0)):.4f}%，"
        f"maxDD={float(selected.get('max_drawdown_pct', 0.0)):.4f}%。\n"
        f"- walk_forward: folds={walk_forward.get('fold_count')}，"
        f"risk_all_pass={walk_forward.get('all_folds_pass_risk_gates')}。\n"
        f"- latest_signal: as_of={latest.get('as_of')}，action={latest.get('action')}，"
        f"target_position={float(latest.get('target_position', latest.get('target_percent', 0.0)) or 0.0):.4f}。\n"
        f"- promotion_decision: {summary['promotion_decision']}。\n"
        "- risks: free_cloud_gpu_* 是云 GPU 租赁/实例小时价，不是销售价；"
        "FRED/BLS hardware_sales_* 是按披露日/as-of 生效的月频官方销售价指数状态，不是日频现货报价；"
        "硬盘和网络日频数据是公开市场代理，不能单独晋升。\n"
        f"- recorder: {summary['recorder_id']}。\n"
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "hardware-cogalpha-title",
                "metadata": {},
                "source": [f"# {summary['experiment']}\n", "\n", _conclusion(summary)],
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "hardware-cogalpha-run",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import qlib\n",
                    "from qlib.data import D\n",
                    "from qlib.workflow import R\n",
                    "\n",
                    "ROOT = Path.cwd()\n",
                    "while not (ROOT / 'src').exists() and ROOT.parent != ROOT:\n",
                    "    ROOT = ROOT.parent\n",
                    "if ROOT.as_posix() not in sys.path:\n",
                    "    sys.path.insert(0, ROOT.as_posix())\n",
                    "from src.workflow.hardware_cycle_cogalpha_mining import run_hardware_cycle_cogalpha_mining_card\n",
                    "\n",
                    "# qlib.init, D.features and R.start are executed inside the workflow.\n",
                    "summary = run_hardware_cycle_cogalpha_mining_card(root=ROOT)\n",
                    "print(json.dumps({'recorder_id': summary['recorder_id'], 'promotion_decision': summary['promotion_decision'], 'as_of': summary['as_of']}, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "hardware-cogalpha-conclusion",
                "metadata": {},
                "source": [_conclusion(summary)],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (card / f"{summary['experiment']}.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _log_factor_mining_artifacts(recorder: Any, run_dir: Path) -> None:
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parent = path.parent.relative_to(run_dir).as_posix()
        artifact_path = (
            "hardware_cycle_cogalpha_evolution"
            if rel_parent == "."
            else f"hardware_cycle_cogalpha_evolution/{rel_parent}"
        )
        recorder.log_artifact(path.as_posix(), artifact_path=artifact_path)


def _remove_transient_code_snapshots(recorder_dir: Path) -> None:
    for name in ["code_cached.txt", "code_diff.txt", "code_status.txt"]:
        path = recorder_dir / "artifacts" / name
        if path.exists():
            path.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
