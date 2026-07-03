from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import QlibProviderConfig, build_alpha158_legacy_dataset, init_qlib_provider
from src.factor import CognitiveAlphaCandidate, materialize_cognitive_alpha_features, run_cognitive_alpha_factor_mining
from src.factor.high_chop_ml import DEFAULT_HIGH_CHOP_FEATURES, build_high_chop_ml_frame
from src.workflow.cognitive_alpha_codex_provider import (
    build_cognitive_alpha_codex_exec_candidate_provider,
    codex_cli_readiness,
)
from src.workflow.cognitive_alpha_evolution import CognitiveAlphaEvolutionSpec
from src.workflow.cognitive_alpha_goal import (
    CognitiveAlphaGoalGateSpec,
    CognitiveAlphaGoalRunner,
    CognitiveAlphaGoalRunnerSpec,
)
from src.workflow.cognitive_alpha_position import CognitiveAlphaPositionSpec, run_cognitive_alpha_position_model
from src.workflow.cognitive_alpha_weighted import (
    CognitiveAlphaWeightedFactorSearchSpec,
    run_cognitive_alpha_weighted_factor_search,
)


EXPERIMENT = "cognitive_alpha_goal_codex_exec_v1"
CARD_TYPE = "diagnostic_cognitive_alpha_codex_exec_goal_chain"
PAPER_REFERENCE = "arxiv:2511.18850"
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
}
QLIB_NATIVE_API = [
    "qlib.init",
    "qlib.data.D.features",
    "DatasetH",
    "DataHandlerLP",
    "Alpha158.get_feature_config",
    "qlib.workflow.R.start",
]
DIRECT_FACTOR_BASE_POSITION_COL = "direct_factor_base_position"
DIRECT_FACTOR_RAW_TARGET_COL = "direct_factor_raw_target"
OPEN_TO_OPEN_RETURN_COL = "open_to_open_return"
LABEL_OPEN_TO_OPEN_5D_COL = "label_open_to_open_5d"
EXECUTION_TIMING = "close_t_signal_next_open_to_following_open"
BASE_CONSTRAINED_REF_COL = "base_constrained_ref"
COGALPHA_DIRECT_ADDON_BASE_COL = "cogalpha_direct_addon_base"
COGALPHA_DIRECT_BLEND_BASE_COL = "cogalpha_direct_blend_base"
COGALPHA_DIRECT_RISKCUT_BASE_COL = "cogalpha_direct_riskcut_base"
COGALPHA_OPEN2OPEN_RISK_GATE_COL = "cogalpha_open2open_risk_gate"


@dataclass(frozen=True)
class CognitiveAlphaGoalCodexExecCardSpec:
    root: Path
    codex_bin: str = "codex"
    codex_model: str | None = None
    codex_timeout_seconds: float = 300.0
    codex_sandbox: str = "read-only"
    resume: bool = False
    attempt_index: int = 1
    max_attempts: int = 2
    request_context: dict[str, Any] | None = None
    initial_pool_size: int = 3
    codex_candidate_count: int = 2
    generations: int = 2
    parent_pool_size: int = 3
    children_per_generation: int = 4
    train: tuple[str, str] = ("2019-01-02", "2024-12-31")
    validation: tuple[str, str] = ("2025-01-02", "2025-12-31")
    test_start: str = "2026-01-02"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / EXPERIMENT

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


def run_cognitive_alpha_goal_codex_exec_card(
    *,
    root: Path | str | None = None,
    resume: bool = False,
    codex_model: str | None = None,
    codex_timeout_seconds: float = 300.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    spec = CognitiveAlphaGoalCodexExecCardSpec(
        root=_resolve_root(root),
        resume=resume,
        codex_model=codex_model,
        codex_timeout_seconds=codex_timeout_seconds,
        max_attempts=max_attempts,
    )
    return _run_card(spec)


def _run_card(spec: CognitiveAlphaGoalCodexExecCardSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    if spec.resume:
        existing_summary = _read_json_object(card / "summary.json")
        if existing_summary.get("status") == "complete":
            return existing_summary
    start_attempt = _resume_start_attempt(card) if spec.resume else 1
    effective_max_attempts = _effective_max_attempts(card, spec=spec, start_attempt=start_attempt)
    attempts: list[dict[str, Any]] = _load_attempt_history(card) if spec.resume else []
    request_context = _feedback_context_from_card(card) if spec.resume and (card / "summary.json").exists() else None
    final_summary: dict[str, Any] = {}
    for attempt_index in range(start_attempt, effective_max_attempts + 1):
        attempt_spec = replace(
            spec,
            attempt_index=attempt_index,
            max_attempts=effective_max_attempts,
            resume=False,
            request_context=request_context,
        )
        summary = _run_card_attempt(attempt_spec, attempts=attempts)
        attempt_entry = _attempt_entry(summary)
        attempts = [entry for entry in attempts if int(entry.get("attempt_index", 0) or 0) != attempt_index]
        attempts.append(attempt_entry)
        attempts = sorted(attempts, key=lambda item: int(item.get("attempt_index", 0) or 0))
        summary["attempts"] = attempts
        summary["goal_loop"]["attempts"] = attempts
        _write_summary_files(card, summary)
        _snapshot_attempt_artifacts(card, attempt_index=attempt_index, summary=summary)
        final_summary = summary
        if summary["goal_manifest"]["status"] != "needs_iteration":
            return final_summary
        request_context = _feedback_context_from_card(card)
    return final_summary


def _run_card_attempt(spec: CognitiveAlphaGoalCodexExecCardSpec, *, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    card = spec.card_dir
    recorder_uri = card / "qlib_recorder"
    card.mkdir(parents=True, exist_ok=True)
    recorder_uri.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((spec.provider_uri / "qlib_store_manifest.json").read_text(encoding="utf-8"))
    as_of = str(manifest["calendar_end"])
    test = (spec.test_start, as_of)
    qlib_config = QlibProviderConfig(
        provider_uri=spec.provider_uri,
        region=str(manifest["region"]),
        market=str(manifest["market"]),
        kernels=1,
    )
    provider_meta = init_qlib_provider(qlib_config)
    codex_readiness = codex_cli_readiness(codex_bin=spec.codex_bin, timeout_seconds=60.0)

    import qlib
    from qlib.data import D
    from qlib.workflow import R

    _ = qlib
    fields = ["$open", "$high", "$low", "$close", "$volume", "$vix"]
    qlib_panel = D.features(["SOXL", "SOXX"], fields, start_time=spec.train[0], end_time=as_of, freq="day")
    frame = qlib_panel.reset_index().sort_values(["datetime", "instrument"]).reset_index(drop=True)
    frame = frame.loc[frame["instrument"].astype(str).str.upper().eq("SOXL")].copy()
    frame = frame.rename(columns={field: field.removeprefix("$") for field in fields})
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    _add_return_columns(frame)
    high_chop_frame = build_high_chop_ml_frame(
        qlib_panel,
        start=spec.train[0],
        end=as_of,
        label_horizon=5,
        include_incomplete_latest=True,
    )
    frame = _merge_high_chop_model_features(frame, high_chop_frame)

    alpha_dataset = build_alpha158_legacy_dataset(
        qlib_config,
        instruments=["SOXL"],
        label="Ref($open, -6) / Ref($open, -1) - 1",
        train=spec.train,
        valid=spec.validation,
        test=test,
    )
    frame, alpha158_columns = _merge_alpha158_features(frame, alpha_dataset)
    frame["realized_vol_20"] = pd.to_numeric(frame["close"], errors="coerce").pct_change().rolling(20, min_periods=10).std().shift(1)

    R.set_uri(recorder_uri.as_posix())
    with tempfile.TemporaryDirectory(prefix="cogalpha_goal_codex_exec_") as tmp_dir:
        runtime_dir = Path(tmp_dir)
        sidecars: dict[str, str] = {
            "evolution_summary": "evolution_summary.json",
            "failure_modes": "failure_modes.md",
            "generation_metrics": "generation_metrics.csv",
            "selected_features": "selected_features.csv",
            "model_selection": "model_selection.csv",
            "benchmark_comparison": "backtest_benchmark_comparison.csv",
            "position_summary": "position_summary.json",
            "latest_signal": "latest_signal.json",
            "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
            "direct_factor_candidate_metrics": "direct_factor_candidate_metrics.csv",
            "direct_factor_benchmark_comparison": "direct_factor_benchmark_comparison.csv",
            "direct_factor_summary": "direct_factor_summary.json",
            "goal_manifest": "goal_manifest.json",
            "state": "state.json",
        }

        with R.start(experiment_name=EXPERIMENT):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=provider_meta["provider_uri"],
                market=manifest["market"],
                region=manifest["region"],
                instrument="SOXL",
                train_start=spec.train[0],
                train_end=spec.train[1],
                validation_start=spec.validation[0],
                validation_end=spec.validation[1],
                test_start=test[0],
                test_end=test[1],
                codex_provider="build_cognitive_alpha_codex_exec_candidate_provider",
                codex_sandbox=spec.codex_sandbox,
                initial_pool_size=spec.initial_pool_size,
                generations=spec.generations,
                parent_pool_size=spec.parent_pool_size,
                children_per_generation=spec.children_per_generation,
            )

            candidate_provider = build_cognitive_alpha_codex_exec_candidate_provider(
                codex_bin=spec.codex_bin,
                provider_id="codex_exec_candidate_provider",
                model=spec.codex_model,
                cwd=spec.root,
                timeout_seconds=spec.codex_timeout_seconds,
                sandbox=spec.codex_sandbox,
                request_context=spec.request_context,
            )
            evolution_spec = CognitiveAlphaEvolutionSpec(
                run_id="cognitive_alpha_evolution",
                profile="soxl_full",
                runtime_dir=runtime_dir,
                generations=spec.generations,
                parent_pool_size=spec.parent_pool_size,
                children_per_generation=spec.children_per_generation,
                initial_pool_size=spec.initial_pool_size,
                qualified_percentile=0.50,
                elite_percentile=0.75,
                max_abs_correlation=0.98,
                min_observations=180,
                max_missing_ratio=0.60,
                min_distinct_values=5,
                fallback_top_n=3,
                resume=False,
            )
            mining_frame = frame.loc[frame["datetime"] <= pd.Timestamp(spec.validation[1])].copy()
            factor_mining = run_cognitive_alpha_factor_mining(
                mining_frame,
                spec=_factor_mining_spec(evolution_spec, alpha158_columns),
                initial_candidates=[_quality_checker_canary()],
                candidate_provider=candidate_provider,
            )
            feature_frame = materialize_cognitive_alpha_features(
                frame,
                list(factor_mining.evolution_result.candidate_map.values()),
                selected_names=factor_mining.selected_feature_names,
            )
            model_frame = pd.concat([frame.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
            selected_factor_names = list(factor_mining.selected_feature_names)
            model_features = _model_feature_columns(selected_factor_names, alpha158_columns, model_frame)
            benchmarks = _load_benchmarks(spec.root, model_frame=model_frame, test=test)
            direct_result = run_cognitive_alpha_weighted_factor_search(
                model_frame,
                spec=_direct_factor_spec(selected_factor_names, model_features),
                benchmarks=benchmarks,
            )
            model_frame, direct_feature_cols = _attach_direct_factor_model_features(model_frame, direct_result)
            model_frame, composite_feature_cols = _attach_cogalpha_composite_base_features(model_frame)
            model_features = _model_feature_columns(
                selected_factor_names,
                alpha158_columns,
                model_frame,
                extra_columns=[*direct_feature_cols, *composite_feature_cols],
            )
            position_result = run_cognitive_alpha_position_model(
                model_frame,
                spec=_position_spec(model_features, spec=spec, test=test),
                benchmarks=benchmarks,
            )
            _attach_open_to_open_execution_context(position_result.summary, position_result.latest_signal)

            _write_evolution_sidecars(card, sidecars, factor_mining)
            _write_position_sidecars(card, sidecars, position_result)
            _write_direct_sidecars(card, sidecars, direct_result)

            positions_path = runtime_dir / "positions.csv"
            position_result.positions.to_csv(positions_path, index=False)
            direct_positions_path = runtime_dir / "direct_weighted_positions.csv"
            direct_result.positions.to_csv(direct_positions_path, index=False)

            _log_evolution_artifacts(recorder, factor_mining.run_dir)
            recorder.log_artifact(positions_path.as_posix(), artifact_path="cognitive_alpha_position")
            recorder.log_artifact(direct_positions_path.as_posix(), artifact_path="cognitive_alpha_direct_weighted")
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.name not in {"state.json", "goal_manifest.json"}:
                    recorder.log_artifact(path.as_posix(), artifact_path=EXPERIMENT)

            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            recorder_id = recorder.id
            research_card_result = {
                "card_dir": card.relative_to(spec.root).as_posix(),
                "recorder_id": recorder_id,
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "promotion_decision": position_result.summary["promotion_decision"],
            }
            goal_runner = CognitiveAlphaGoalRunner(
                CognitiveAlphaGoalRunnerSpec(
                    run_id="cognitive_alpha_goal_codex_exec",
                    objective="SOXL CogAlpha Codex exec provider data/evolution/position/direct-factor/card chain",
                    state_path=card / sidecars["state"],
                    manifest_path=card / sidecars["goal_manifest"],
                    required_max_datetime=as_of,
                    resume=False,
                    attempt_index=spec.attempt_index,
                    max_attempts=spec.max_attempts,
                    gates=CognitiveAlphaGoalGateSpec(
                        min_initial_candidate_count=spec.initial_pool_size,
                        min_completed_generations=spec.generations,
                        min_selected_count=1,
                        min_repair_attempt_count=1,
                        required_candidate_provider_ids=("codex_exec_candidate_provider",),
                        required_candidate_provider_kinds=("codex_exec_code",),
                        require_candidate_provider_no_openai_api_key=True,
                        required_metric_names=("ic", "icir", "rank_ic", "rank_icir", "mutual_information"),
                        required_evolution_ops=("mutation", "crossover", "crossover_mutation"),
                        required_position_model_families=("lgbm_classifier", "xgb_classifier"),
                        required_benchmark_names=(
                            "soxl_backtest",
                            "base_constrained_ref",
                            "constrained_weight_reference",
                            "soxl_buyhold",
                        ),
                        required_direct_factor_benchmark_names=(
                            "base_constrained_ref",
                            "constrained_weight_reference",
                            "soxl_backtest",
                            "soxl_buyhold",
                        ),
                        min_walk_forward_fold_count=2,
                    ),
                ),
                data_gate=lambda: [
                    {
                        "source": "qlib.data.D.features",
                        "provider_uri": provider_meta["provider_uri"],
                        "market": manifest["market"],
                        "region": manifest["region"],
                        "max_datetime": as_of,
                    }
                ],
                evolution=lambda manifests: factor_mining.evolution_result.summary,
                position=lambda manifests, evolution_result: position_result.summary,
                direct_factor_backtest=lambda manifests, evolution_result, position_result_payload: direct_result.summary,
                research_card=lambda manifests, evolution_result, position_result_payload: research_card_result,
                signal_push=lambda manifests, model_results: [
                    {
                        "source": "qlib_recorder.latest_signal",
                        "action": position_result.latest_signal["action"],
                        "as_of": position_result.latest_signal["as_of"],
                        "real_broker_order": False,
                    }
                ],
            )
            goal_manifest = goal_runner.run()
            recorder.log_artifact((card / sidecars["goal_manifest"]).as_posix(), artifact_path=EXPERIMENT)
            recorder.log_artifact((card / sidecars["state"]).as_posix(), artifact_path=EXPERIMENT)

        recorder_artifacts = _recorder_files(recorder_local_dir)
        summary = _summary_payload(
            spec=spec,
            manifest=manifest,
            as_of=as_of,
            test=test,
            provider_meta=provider_meta,
            codex_readiness=codex_readiness,
            factor_mining=factor_mining.summary,
            evolution=factor_mining.evolution_result.summary,
            position=position_result.summary,
            direct=direct_result.summary,
            goal_manifest=goal_manifest,
            research_card=research_card_result,
            sidecars=sidecars,
            recorder_id=recorder_id,
            recorder_local_dir=recorder_local_dir,
            recorder_uri=recorder_uri,
            recorder_artifacts=recorder_artifacts,
        )
        summary["attempts"] = attempts
        _write_summary_files(card, summary)
        return summary


def _resume_start_attempt(card: Path) -> int:
    state_path = card / "state.json"
    state = _read_json_object(state_path)
    if state.get("status") == "needs_iteration":
        return int(state.get("current_attempt", state.get("attempt_index", 0)) or 0) + 1
    summary = _read_json_object(card / "summary.json")
    if summary.get("status") == "needs_iteration":
        return int(summary.get("attempt_index", 0) or 0) + 1
    return 1


def _effective_max_attempts(card: Path, *, spec: CognitiveAlphaGoalCodexExecCardSpec, start_attempt: int) -> int:
    configured_max_attempts = max(int(spec.max_attempts), 1)
    if not spec.resume or start_attempt <= configured_max_attempts:
        return configured_max_attempts
    summary = _read_json_object(card / "summary.json")
    if summary.get("status") == "needs_iteration":
        return start_attempt
    return configured_max_attempts


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_attempt_history(card: Path) -> list[dict[str, Any]]:
    summary_path = card / "summary.json"
    if not summary_path.exists():
        return []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    attempts = summary.get("attempts", [])
    return list(attempts) if isinstance(attempts, list) else []


def _attempt_entry(summary: dict[str, Any]) -> dict[str, Any]:
    position = summary["position_model"]
    direct = summary["direct_factor_backtest"]
    return {
        "attempt_index": int(summary["attempt_index"]),
        "status": str(summary["status"]),
        "terminal_reason": str(summary.get("terminal_reason", "")),
        "promotion_decision": str(summary["promotion_decision"]),
        "recorder_id": str(summary["recorder_id"]),
        "selected_return_pct": position.get("comparison", {}).get("selected_return_pct"),
        "beats_constrained_weight_reference": position.get("comparison", {}).get(
            "beats_constrained_weight_reference"
        ),
        "walk_forward_all_gates_pass": position.get("walk_forward", {}).get("all_folds_pass_risk_gates"),
        "direct_factor_promotion_decision": direct.get("promotion_decision"),
    }


def _attach_open_to_open_execution_context(position_summary: dict[str, Any], latest_signal: dict[str, Any]) -> None:
    position_summary["execution_timing"] = EXECUTION_TIMING
    position_summary["return_col"] = OPEN_TO_OPEN_RETURN_COL
    position_summary["model_target_col"] = LABEL_OPEN_TO_OPEN_5D_COL
    latest_signal["execution_timing"] = EXECUTION_TIMING
    latest_signal["return_col"] = OPEN_TO_OPEN_RETURN_COL
    latest_signal["model_target_col"] = LABEL_OPEN_TO_OPEN_5D_COL


def _snapshot_attempt_artifacts(card: Path, *, attempt_index: int, summary: dict[str, Any]) -> None:
    attempt_dir = card / "attempts" / f"attempt_{attempt_index:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    snapshot_files = [
        "summary.json",
        "workflow.yaml",
        "conclusion.md",
        "failure_modes.md",
        *list(summary.get("sidecars", {}).values()),
    ]
    for rel_path in dict.fromkeys(snapshot_files):
        source = card / str(rel_path)
        if source.exists() and source.is_file():
            shutil.copy2(source, attempt_dir / source.name)


def _feedback_context_from_card(card: Path) -> dict[str, Any]:
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    goal_manifest = json.loads((card / "goal_manifest.json").read_text(encoding="utf-8"))
    failure_modes = (card / "failure_modes.md").read_text(encoding="utf-8")
    position = summary.get("position_model", {})
    direct = summary.get("direct_factor_backtest", {})
    next_action = goal_manifest.get("next_action", {})
    return {
        "goal_status": summary.get("status"),
        "terminal_reason": summary.get("terminal_reason"),
        "previous_attempt": summary.get("attempt_index"),
        "next_attempt": next_action.get("next_attempt"),
        "max_attempts": summary.get("max_attempts"),
        "previous_promotion_decision": summary.get("promotion_decision"),
        "failure_modes": failure_modes.splitlines(),
        "goal_next_action": next_action,
        "position_comparison": position.get("comparison", {}),
        "walk_forward": position.get("walk_forward", {}),
        "direct_factor_promotion_decision": direct.get("promotion_decision"),
        "direct_factor_role": summary.get("research_gate", {}).get("direct_factor_backtest_role"),
        "forbidden_repetition": {
            "recorder_id": summary.get("recorder_id"),
            "selected_feature_names": summary.get("evolution", {}).get("selected_feature_names", []),
        },
    }


def _factor_mining_spec(evolution_spec: CognitiveAlphaEvolutionSpec, alpha158_columns: list[str]):
    from src.factor import CognitiveAlphaFactorMiningSpec

    return CognitiveAlphaFactorMiningSpec(
        evolution_spec=evolution_spec,
        label_col=LABEL_OPEN_TO_OPEN_5D_COL,
        alpha158_columns=tuple(alpha158_columns[:8]),
        max_factor_library_size=4,
    )


def _quality_checker_canary() -> CognitiveAlphaCandidate:
    def _future_leak(frame: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(frame["close"], errors="coerce").shift(-1) / pd.to_numeric(
            frame["close"], errors="coerce"
        ) - 1.0

    return CognitiveAlphaCandidate(
        name="codex_exec_quality_canary_future_leak",
        level="quality_checker_canary",
        agent="QualityCheckerCanary",
        theme="intentional future-leak canary for CogAlpha repair loop",
        required_columns=("close",),
        lookback=1,
        function=_future_leak,
        economic_rationale="Intentional invalid candidate proving the quality checker and repair loop reject future leakage.",
        code="close.shift(-1) / close - 1.0",
        declared_lookahead=1,
    )


def _merge_alpha158_features(frame: pd.DataFrame, alpha_dataset: Any) -> tuple[pd.DataFrame, list[str]]:
    alpha_frames = [alpha_dataset.prepare(segment, col_set="feature") for segment in ["train", "valid", "test"]]
    alpha_frame = pd.concat(alpha_frames).reset_index()
    alpha_frame["datetime"] = pd.to_datetime(alpha_frame["datetime"]).dt.normalize()
    preferred = ["MA5", "MA10", "MA20", "MA30", "MA60", "MAX20", "MAX30", "VIX", "SOXL_RET_5", "VOLUME_RATIO_5"]
    alpha_cols = [column for column in preferred if column in alpha_frame.columns]
    alpha_subset = alpha_frame[["datetime", "instrument", *alpha_cols]].rename(
        columns={column: f"alpha158_{column}" for column in alpha_cols}
    )
    out = frame.merge(alpha_subset, on=["datetime", "instrument"], how="left")
    return out, [f"alpha158_{column}" for column in alpha_cols]


def _merge_high_chop_model_features(frame: pd.DataFrame, high_chop_frame: pd.DataFrame) -> pd.DataFrame:
    high_chop = high_chop_frame.copy()
    high_chop["datetime"] = pd.to_datetime(high_chop["datetime"]).dt.normalize()
    out = frame.copy()
    feature_cols = [
        column
        for column in DEFAULT_HIGH_CHOP_FEATURES
        if column in high_chop.columns and column not in {OPEN_TO_OPEN_RETURN_COL, LABEL_OPEN_TO_OPEN_5D_COL}
        and column not in out.columns
    ]
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    return out.merge(high_chop[["datetime", *feature_cols]], on="datetime", how="left")


def _add_return_columns(frame: pd.DataFrame) -> None:
    close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
    open_ = pd.to_numeric(frame["open"], errors="coerce").astype("float64")
    frame["forward_return"] = close.pct_change().shift(-1).fillna(0.0)
    frame["label_return_5d"] = close.pct_change(5).shift(-5).ffill().fillna(0.0)
    frame[OPEN_TO_OPEN_RETURN_COL] = open_.shift(-2) / open_.shift(-1) - 1.0
    frame[LABEL_OPEN_TO_OPEN_5D_COL] = open_.shift(-6) / open_.shift(-1) - 1.0
    if not frame.empty:
        latest_index = frame.index.max()
        if pd.isna(frame.loc[latest_index, OPEN_TO_OPEN_RETURN_COL]):
            frame.loc[latest_index, OPEN_TO_OPEN_RETURN_COL] = 0.0


def _model_feature_columns(
    selected_factor_names: list[str],
    alpha158_columns: list[str],
    frame: pd.DataFrame,
    *,
    extra_columns: list[str] | None = None,
) -> list[str]:
    base = [
        *selected_factor_names,
        *alpha158_columns[:6],
        *[column for column in DEFAULT_HIGH_CHOP_FEATURES if column in frame.columns],
        "vix",
        "realized_vol_20",
        *(extra_columns or []),
    ]
    return [column for column in dict.fromkeys(base) if column in frame.columns]


def _attach_direct_factor_model_features(frame: pd.DataFrame, direct_result: Any) -> tuple[pd.DataFrame, list[str]]:
    selected_candidate = str(direct_result.summary.get("selected_candidate", {}).get("candidate", ""))
    positions = direct_result.positions.copy()
    if not selected_candidate or positions.empty:
        out = frame.copy()
        out[DIRECT_FACTOR_BASE_POSITION_COL] = 0.0
        out[DIRECT_FACTOR_RAW_TARGET_COL] = 0.0
        return out, [DIRECT_FACTOR_BASE_POSITION_COL, DIRECT_FACTOR_RAW_TARGET_COL]

    selected = positions.loc[positions["candidate"].astype(str).eq(selected_candidate)].copy()
    selected["datetime"] = pd.to_datetime(selected["datetime"]).dt.normalize()
    selected = selected[["datetime", "target_position", "raw_target"]].rename(
        columns={
            "target_position": DIRECT_FACTOR_BASE_POSITION_COL,
            "raw_target": DIRECT_FACTOR_RAW_TARGET_COL,
        }
    )
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.normalize()
    out = out.merge(selected, on="datetime", how="left")
    out[DIRECT_FACTOR_BASE_POSITION_COL] = (
        pd.to_numeric(out[DIRECT_FACTOR_BASE_POSITION_COL], errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    out[DIRECT_FACTOR_RAW_TARGET_COL] = (
        pd.to_numeric(out[DIRECT_FACTOR_RAW_TARGET_COL], errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0.0)
    )
    selected_base = out[DIRECT_FACTOR_BASE_POSITION_COL]
    selected_raw = out[DIRECT_FACTOR_RAW_TARGET_COL]
    if selected_base.abs().mean() <= 1e-9 and selected_raw.abs().mean() > 1e-9:
        out[DIRECT_FACTOR_BASE_POSITION_COL] = _direct_factor_base_position_from_raw(selected_raw)
    return out, [DIRECT_FACTOR_BASE_POSITION_COL, DIRECT_FACTOR_RAW_TARGET_COL]


def _attach_cogalpha_composite_base_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    base = _numeric_series(out, BASE_CONSTRAINED_REF_COL).clip(lower=0.0, upper=1.0)
    direct = _numeric_series(out, DIRECT_FACTOR_BASE_POSITION_COL).clip(lower=0.0, upper=1.0)
    vix_chg5 = _numeric_series(out, "vix_chg5")
    rv20_expansion = _numeric_series(out, "rv20_expansion")
    breakdown = _numeric_series(out, "chop_breakdown_risk").gt(0.0)
    risk_active = breakdown | rv20_expansion.gt(1.8) | vix_chg5.gt(0.25)
    reentry_active = (
        _numeric_series(out, "chop_reentry_quality").gt(0.02)
        | _numeric_series(out, "soxx_lead_10").gt(0.02)
        | _numeric_series(out, "panic_reversal_5").gt(0.01)
    ) & ~risk_active

    direct_addon = pd.Series(0.0, index=out.index)
    direct_addon.loc[reentry_active] = direct.loc[reentry_active]
    out[COGALPHA_DIRECT_ADDON_BASE_COL] = pd.concat([base, direct_addon], axis=1).max(axis=1).clip(0.0, 1.0)
    out[COGALPHA_DIRECT_BLEND_BASE_COL] = (0.70 * base + 0.30 * direct_addon).clip(0.0, 1.0)
    out[COGALPHA_DIRECT_RISKCUT_BASE_COL] = out[COGALPHA_DIRECT_ADDON_BASE_COL].where(
        ~risk_active,
        base.clip(upper=0.35),
    ).clip(0.0, 1.0)
    out[COGALPHA_OPEN2OPEN_RISK_GATE_COL] = risk_active.astype("float64")
    return out, [
        COGALPHA_DIRECT_ADDON_BASE_COL,
        COGALPHA_DIRECT_BLEND_BASE_COL,
        COGALPHA_DIRECT_RISKCUT_BASE_COL,
        COGALPHA_OPEN2OPEN_RISK_GATE_COL,
    ]


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, name=column)
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .fillna(0.0)
        .astype("float64")
        .rename(column)
    )


def _direct_factor_base_position_from_raw(raw_target: pd.Series) -> pd.Series:
    raw = pd.to_numeric(raw_target, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    max_abs = float(raw.abs().max())
    if max_abs <= 1e-12:
        return pd.Series(0.0, index=raw.index, name=DIRECT_FACTOR_BASE_POSITION_COL)
    if float(raw.max()) <= 1e-12 and float(raw.min()) < -1e-12:
        normalized = (-raw / max_abs).clip(lower=0.0, upper=1.0)
    else:
        normalized = (raw / max_abs).clip(lower=0.0, upper=1.0)
    return normalized.rename(DIRECT_FACTOR_BASE_POSITION_COL)


def _load_benchmarks(
    root: Path,
    *,
    model_frame: pd.DataFrame | None = None,
    test: tuple[str, str] | None = None,
) -> list[dict[str, Any]]:
    if model_frame is not None and test is not None:
        rows = _dynamic_open_to_open_benchmarks(model_frame, test=test)
        if rows:
            return rows

    path = root / "research" / "cognitive_alpha_position_model_v1" / "backtest_benchmark_comparison.csv"
    frame = pd.read_csv(path)
    rows = []
    for row in frame.loc[frame["benchmark"] != "cognitive_alpha_selected"].to_dict(orient="records"):
        rows.append(
            {
                "name": str(row["benchmark"]),
                "return_pct": float(row["return_pct"]),
                "max_drawdown_pct": float(row["max_drawdown_pct"]),
                "trade_count": int(row.get("trade_count", 0) or 0),
                "source": str(row.get("source", row["benchmark"])),
            }
        )
    return rows


def _dynamic_open_to_open_benchmarks(frame: pd.DataFrame, *, test: tuple[str, str]) -> list[dict[str, Any]]:
    data = frame.copy()
    data["datetime"] = pd.to_datetime(data["datetime"]).dt.normalize()
    mask = (data["datetime"] >= pd.Timestamp(test[0])) & (data["datetime"] <= pd.Timestamp(test[1]))
    window = data.loc[mask].sort_values("datetime").reset_index(drop=True)
    if window.empty or OPEN_TO_OPEN_RETURN_COL not in window.columns:
        return []
    returns = pd.to_numeric(window[OPEN_TO_OPEN_RETURN_COL], errors="coerce").replace(
        [float("inf"), float("-inf")],
        pd.NA,
    ).fillna(0.0)
    rows = [_benchmark_row_from_returns("soxl_buyhold", returns, source="qlib_dynamic_open_to_open_buyhold")]
    if BASE_CONSTRAINED_REF_COL in window.columns:
        rows.append(
            _benchmark_row_from_position(
                "base_constrained_ref",
                window[BASE_CONSTRAINED_REF_COL],
                returns,
                source="qlib_dynamic_open_to_open_base_constrained_ref",
            )
        )
        rows.append(
            _benchmark_row_from_position(
                "constrained_weight_reference",
                window[BASE_CONSTRAINED_REF_COL],
                returns,
                source="qlib_dynamic_open_to_open_base_constrained_ref_alias",
            )
        )
    try:
        from soxl_backtest.reference import replay_soxlbacktest_reference

        legacy = replay_soxlbacktest_reference(start=test[0], end=test[1]).metrics
        rows.append(
            {
                "name": "soxl_backtest",
                "return_pct": float(legacy.get("rulebase_return_pct", 0.0)),
                "max_drawdown_pct": -abs(float(legacy.get("rulebase_max_drawdown_pct", 0.0))),
                "trade_count": int(legacy.get("trade_count", 0) or 0),
                "source": "soxl_backtest.reference.replay_soxlbacktest_reference",
            }
        )
    except Exception:
        pass
    return rows


def _benchmark_row_from_returns(name: str, returns: pd.Series, *, source: str) -> dict[str, Any]:
    strategy_returns = returns.astype("float64").fillna(0.0)
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "name": name,
        "return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if not equity.empty else 0.0,
        "max_drawdown_pct": float(drawdown.min() * 100.0) if not drawdown.empty else 0.0,
        "trade_count": 1,
        "source": source,
    }


def _benchmark_row_from_position(
    name: str,
    position: pd.Series,
    returns: pd.Series,
    *,
    source: str,
    cost_per_turnover: float = 0.0003,
) -> dict[str, Any]:
    target = pd.to_numeric(position, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype("float64")
    turnover = target.diff().abs().fillna(target.abs())
    strategy_returns = target * returns.astype("float64").fillna(0.0) - turnover * float(cost_per_turnover)
    equity = (1.0 + strategy_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "name": name,
        "return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if not equity.empty else 0.0,
        "max_drawdown_pct": float(drawdown.min() * 100.0) if not drawdown.empty else 0.0,
        "trade_count": int(turnover.gt(1e-9).sum()),
        "source": source,
    }


def _float_or_zero(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(parsed) else float(parsed)


def _int_or_zero(value: Any) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0 if pd.isna(parsed) else int(parsed)


def _position_spec(feature_cols: list[str], *, spec: CognitiveAlphaGoalCodexExecCardSpec, test: tuple[str, str]) -> CognitiveAlphaPositionSpec:
    base_throttle_specs = []
    base_columns = [
        BASE_CONSTRAINED_REF_COL,
        "base_chop_control",
        "base_recovery",
        DIRECT_FACTOR_BASE_POSITION_COL,
        COGALPHA_DIRECT_ADDON_BASE_COL,
        COGALPHA_DIRECT_BLEND_BASE_COL,
        COGALPHA_DIRECT_RISKCUT_BASE_COL,
    ]
    for base_col in [column for column in base_columns if column in feature_cols]:
        min_throttles = [0.55, 0.75, 0.85] if base_col == BASE_CONSTRAINED_REF_COL else [0.35, 0.55, 0.75]
        for min_throttle in min_throttles:
            base_throttle_specs.append(
                {
                    "position_mode": "base_throttle",
                    "base_position_col": base_col,
                    "threshold": 0.45,
                    "low_probability": 0.0 if base_col == BASE_CONSTRAINED_REF_COL else 0.35,
                    "high_probability": 0.5 if base_col == BASE_CONSTRAINED_REF_COL else 0.65,
                    "min_throttle": min_throttle,
                    "max_throttle": 1.0,
                    "scale": 1.0,
                    "max_position": 1.0,
                    "rebalance_threshold": 0.05,
                    "min_hold_days": 1 if base_col == BASE_CONSTRAINED_REF_COL else 2,
                    "objective_return_weight": 0.02,
                    "objective_return_to_drawdown_weight": 10.0,
                    "objective_exposure_weight": 0.0,
                    "objective_min_avg_abs_position": 0.03,
                    "exposure_shortfall_penalty": 25.0,
                    "objective_max_drawdown_floor": -25.0,
                    "drawdown_floor_penalty": 2.0,
                    "turnover_penalty": 1.5,
                    "reversal_penalty": 2.0,
                    "trade_penalty": 0.10,
                    "selection_disabled": base_col != BASE_CONSTRAINED_REF_COL,
                }
            )
    base_plus_specs = []
    if BASE_CONSTRAINED_REF_COL in feature_cols and COGALPHA_OPEN2OPEN_RISK_GATE_COL in feature_cols:
        base_plus_grid = [
            {"threshold": 0.35, "boost_scale": 0.10, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.35, "boost_scale": 0.20, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.45, "boost_scale": 0.20, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.45, "boost_scale": 0.35, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.45, "boost_scale": 0.50, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.55, "boost_scale": 0.35, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.55, "boost_scale": 0.50, "risk_cap": 0.35, "min_hold_days": 2},
            {"threshold": 0.65, "boost_scale": 1.00, "risk_cap": 0.75, "min_hold_days": 3},
            {"threshold": 0.65, "boost_scale": 1.50, "risk_cap": 0.75, "min_hold_days": 3},
            {"threshold": 0.65, "boost_scale": 1.50, "risk_cap": 0.55, "min_hold_days": 2},
        ]
        for item in base_plus_grid:
            base_plus_specs.append(
                {
                    "position_mode": "base_plus_probability",
                    "base_position_col": BASE_CONSTRAINED_REF_COL,
                    "threshold": item["threshold"],
                    "base_scale": 1.0,
                    "boost_scale": item["boost_scale"],
                    "scale": 1.0,
                    "max_position": 1.0,
                    "rebalance_threshold": 0.05,
                    "min_hold_days": item["min_hold_days"],
                    "risk_gate_col": COGALPHA_OPEN2OPEN_RISK_GATE_COL,
                    "risk_gate_threshold": 0.5,
                    "risk_gate_direction": "above",
                    "risk_gate_max_position": item["risk_cap"],
                    "objective_return_weight": 0.02,
                    "objective_return_to_drawdown_weight": 10.0,
                    "objective_exposure_weight": 0.0,
                    "objective_min_avg_abs_position": 0.03,
                    "exposure_shortfall_penalty": 25.0,
                    "objective_max_drawdown_floor": -25.0,
                    "drawdown_floor_penalty": 2.0,
                    "turnover_penalty": 1.5,
                    "reversal_penalty": 2.0,
                    "trade_penalty": 0.10,
                }
            )
    direct_probability_specs = []
    if DIRECT_FACTOR_BASE_POSITION_COL in feature_cols:
        direct_probability_specs.append(
            {
                "position_mode": "base_probability",
                "base_position_col": DIRECT_FACTOR_BASE_POSITION_COL,
                "threshold": 0.35,
                "scale": 1.0,
                "max_position": 1.0,
                "rebalance_threshold": 0.05,
                "max_daily_change": 0.5,
                "min_hold_days": 2,
                "objective_return_weight": 0.20,
                "objective_return_to_drawdown_weight": 4.0,
                "objective_exposure_weight": 3.0,
                "objective_min_avg_abs_position": 0.05,
                "exposure_shortfall_penalty": 25.0,
                "objective_max_drawdown_floor": -25.0,
                "drawdown_floor_penalty": 2.0,
                "turnover_penalty": 0.25,
                "reversal_penalty": 1.0,
                "trade_penalty": 0.05,
                "selection_disabled": True,
            }
        )
    return CognitiveAlphaPositionSpec(
        feature_cols=feature_cols,
        train=spec.train,
        validation=spec.validation,
        test=test,
        model_specs=[
            {"model": "lgbm_classifier"},
            {"model": "lgbm_regressor", "task": "regression"},
            {"model": "xgb_classifier"},
            {"model": "xgb_regressor", "task": "regression"},
        ],
        position_specs=[
            {"threshold": 0.45, "scale": 1.0, "min_hold_days": 3, "max_position": 1.0, "selection_disabled": True},
            {"threshold": 0.55, "scale": 1.0, "min_hold_days": 3, "max_position": 1.0, "selection_disabled": True},
            *base_throttle_specs,
            *base_plus_specs,
            *direct_probability_specs,
        ],
        return_col=OPEN_TO_OPEN_RETURN_COL,
        label_col=LABEL_OPEN_TO_OPEN_5D_COL,
        min_validation_trade_count=1,
        min_validation_avg_abs_position=0.01,
        max_validation_zero_position_day_ratio=0.98,
        validation_max_drawdown_floor=-25.0,
        validation_max_daily_loss_floor=None,
        max_validation_total_turnover=20.0,
        max_validation_extreme_loss_day_count=7,
        max_validation_reversal_ratio=0.50,
        min_test_avg_abs_position=0.01,
        max_test_zero_position_day_ratio=0.98,
        test_max_drawdown_floor=-25.0,
        test_max_daily_loss_floor=None,
        max_test_total_turnover=20.0,
        max_test_extreme_loss_day_count=7,
        walk_forward_folds=[
            {
                "fold": "wf_2025_h2",
                "train": ("2019-01-02", "2024-06-28"),
                "validation": ("2024-07-01", "2025-06-30"),
                "test": ("2025-07-01", "2025-12-31"),
            },
            {
                "fold": "wf_2026_ytd",
                "train": ("2019-01-02", "2024-12-31"),
                "validation": ("2025-01-02", "2025-12-31"),
                "test": test,
            },
        ],
        constrained_reference_name="base_constrained_ref",
    )


def _direct_factor_spec(selected_factor_names: list[str], model_features: list[str]) -> CognitiveAlphaWeightedFactorSearchSpec:
    component_cols = selected_factor_names[:3] or model_features[:3]
    if len(component_cols) == 1 and len(model_features) > 1:
        component_cols = [component_cols[0], model_features[1]]
    candidates = [
        {
            "candidate": "codex_exec_equal_weight_direct",
            "weights": {column: 1.0 / max(len(component_cols), 1) for column in component_cols},
            "policy": {"min_position": 0.0, "max_position": 1.0, "rebalance_threshold": 0.20},
        },
        {
            "candidate": "codex_exec_primary_factor_direct",
            "weights": {component_cols[0]: 1.0},
            "policy": {"min_position": 0.0, "max_position": 1.0, "rebalance_threshold": 0.30},
        },
    ]
    return CognitiveAlphaWeightedFactorSearchSpec(
        component_cols=component_cols,
        candidates=candidates,
        return_col=OPEN_TO_OPEN_RETURN_COL,
        min_effective_component_count=1,
        max_drawdown_floor=-25.0,
        constrained_reference_name="base_constrained_ref",
    )


def _write_evolution_sidecars(card: Path, sidecars: dict[str, str], factor_mining: Any) -> None:
    (card / sidecars["evolution_summary"]).write_text(
        json.dumps(factor_mining.evolution_result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    factor_mining.generation_metrics.to_csv(card / sidecars["generation_metrics"], index=False)
    selected_rows = [
        {
            "name": item["name"],
            "level": item["level"],
            "agent": item["agent"],
            "lookback": item["lookback"],
            "code": item["code"],
        }
        for item in factor_mining.factor_library
    ]
    pd.DataFrame(selected_rows).to_csv(card / sidecars["selected_features"], index=False)


def _write_position_sidecars(card: Path, sidecars: dict[str, str], position_result: Any) -> None:
    position_result.metrics.to_csv(card / sidecars["model_selection"], index=False)
    position_result.benchmark_comparison.to_csv(card / sidecars["benchmark_comparison"], index=False)
    position_result.walk_forward_metrics.to_csv(card / sidecars["backtest_walk_forward_metrics"], index=False)
    (card / sidecars["position_summary"]).write_text(
        json.dumps(position_result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (card / sidecars["latest_signal"]).write_text(
        json.dumps(position_result.latest_signal, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_direct_sidecars(card: Path, sidecars: dict[str, str], direct_result: Any) -> None:
    direct_result.metrics.to_csv(card / sidecars["direct_factor_candidate_metrics"], index=False)
    direct_result.benchmark_comparison.to_csv(card / sidecars["direct_factor_benchmark_comparison"], index=False)
    (card / sidecars["direct_factor_summary"]).write_text(
        json.dumps(direct_result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _log_evolution_artifacts(recorder: Any, run_dir: Path) -> None:
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parent = path.parent.relative_to(run_dir).as_posix()
        artifact_path = "cognitive_alpha_evolution" if rel_parent == "." else f"cognitive_alpha_evolution/{rel_parent}"
        recorder.log_artifact(path.as_posix(), artifact_path=artifact_path)


def _summary_payload(
    *,
    spec: CognitiveAlphaGoalCodexExecCardSpec,
    manifest: dict[str, Any],
    as_of: str,
    test: tuple[str, str],
    provider_meta: dict[str, Any],
    codex_readiness: dict[str, Any],
    factor_mining: dict[str, Any],
    evolution: dict[str, Any],
    position: dict[str, Any],
    direct: dict[str, Any],
    goal_manifest: dict[str, Any],
    research_card: dict[str, Any],
    sidecars: dict[str, str],
    recorder_id: str,
    recorder_local_dir: Path,
    recorder_uri: Path,
    recorder_artifacts: list[str],
) -> dict[str, Any]:
    codex_provider = dict(evolution.get("candidate_provider", {}))
    codex_provider.update(
        {
            "provider_builder": "build_cognitive_alpha_codex_exec_candidate_provider",
            "sandbox": spec.codex_sandbox,
            "output_schema": "cognitive_alpha_candidate_schema.json",
            "execution_mode": "codex_exec_chatgpt_auth_no_openai_api_key",
        }
    )
    promotion_decision = str(position["promotion_decision"])
    production_status = "candidate_for_paper_sim" if promotion_decision == "candidate_for_validated_research" else "blocked_not_promoted"
    goal_status = str(goal_manifest["status"])
    terminal_reason = str(goal_manifest.get("terminal_reason", ""))
    return {
        "experiment": EXPERIMENT,
        "card_type": CARD_TYPE,
        "workflow_config": "workflow.yaml",
        "status": goal_status,
        "terminal_reason": terminal_reason,
        "promotion_decision": promotion_decision,
        "attempt_index": int(spec.attempt_index),
        "max_attempts": int(spec.max_attempts),
        "remaining_attempts": int(goal_manifest.get("remaining_attempts", 0) or 0),
        "next_action": goal_manifest.get("next_action", {}),
        "paper_reference": {"arxiv_id": "2511.18850", "source": PAPER_REFERENCE},
        "provider_uri": provider_meta["provider_uri"],
        "recorder_uri": recorder_uri.as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "market": manifest["market"],
        "region": manifest["region"],
        "as_of": as_of,
        "execution_timing": EXECUTION_TIMING,
        "return_col": OPEN_TO_OPEN_RETURN_COL,
        "model_target_col": LABEL_OPEN_TO_OPEN_5D_COL,
        "benchmark_basis": "open_to_open",
        "segments": {
            "train": list(spec.train),
            "validation": list(spec.validation),
            "test": list(test),
        },
        "codex_provider_readiness": codex_readiness,
        "codex_provider": codex_provider,
        "factor_mining": factor_mining,
        "evolution": _evolution_summary_for_card(evolution),
        "position_model": position,
        "direct_factor_backtest": direct,
        "latest_signal": position["latest_signal"],
        "research_gate": {
            "final_promotion_source": "position_model",
            "direct_factor_backtest_role": "prior_only_not_final_promotion",
            "promotion_decision": promotion_decision,
        },
        "production_paper_sim": {
            "status": production_status,
            "required_gate": "validated_research_position_model_or_production_paper_sim",
            "direct_factor_backtest_allowed_to_promote": False,
        },
        "goal_manifest": {
            "path": sidecars["goal_manifest"],
            "status": goal_status,
            "terminal_reason": terminal_reason,
            "attempt_index": int(goal_manifest.get("attempt_index", spec.attempt_index) or spec.attempt_index),
            "max_attempts": int(goal_manifest.get("max_attempts", spec.max_attempts) or spec.max_attempts),
            "remaining_attempts": int(goal_manifest.get("remaining_attempts", 0) or 0),
            "next_action": goal_manifest.get("next_action", {}),
            "stage_order": goal_manifest["stage_order"],
        },
        "goal_loop": {
            "status": goal_status,
            "terminal_reason": terminal_reason,
            "attempt_index": int(spec.attempt_index),
            "max_attempts": int(spec.max_attempts),
            "remaining_attempts": int(goal_manifest.get("remaining_attempts", 0) or 0),
            "next_action": goal_manifest.get("next_action", {}),
            "feedback_context_included": spec.request_context is not None,
            "attempts": [],
        },
        "qlib_docs": QLIB_DOCS,
        "qlib_docs_reference": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "primary_artifact": f"{EXPERIMENT}.ipynb",
            "kept_sidecars": list(sidecars.values()),
            "recorder_only_artifacts": {
                "candidate_pool": "artifacts/cognitive_alpha_evolution/candidate_pool/generation_*.jsonl",
                "fitness": "artifacts/cognitive_alpha_evolution/fitness/generation_*.csv",
                "lineage": "artifacts/cognitive_alpha_evolution/lineage/generation_*.jsonl",
                "quality": "artifacts/cognitive_alpha_evolution/quality/generation_*.csv",
                "rejected": "artifacts/cognitive_alpha_evolution/rejected/generation_*.jsonl",
                "state": "artifacts/cognitive_alpha_evolution/state.json",
                "positions": "artifacts/cognitive_alpha_position/positions.csv",
                "direct_weighted_positions": "artifacts/cognitive_alpha_direct_weighted/direct_weighted_positions.csv",
            },
        },
    }


def _evolution_summary_for_card(evolution: dict[str, Any]) -> dict[str, Any]:
    out = dict(evolution)
    rows = []
    run_dir = evolution.get("run_dir")
    if run_dir:
        metrics_path = Path(str(run_dir)) / "fitness" / f"generation_{int(evolution.get('completed_generations', 1)) - 1:03d}.csv"
        if metrics_path.exists():
            rows = pd.read_csv(metrics_path).to_dict(orient="records")
    levels = sorted({str(row.get("level")) for row in rows if row.get("level")})
    if not levels:
        levels = sorted(set(out.get("selected_level_counts", {}).keys()))
    out["agent_levels_covered"] = levels
    return out


def _write_summary_files(card: Path, summary: dict[str, Any]) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (card / "workflow.yaml").write_text(_workflow_yaml(summary), encoding="utf-8")
    (card / "failure_modes.md").write_text(_failure_modes(summary), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _workflow_yaml(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"experiment: {EXPERIMENT}",
            f"card_type: {CARD_TYPE}",
            f"provider_uri: {summary['provider_uri']}",
            f"recorder_uri: {summary['recorder_uri']}",
            "codex_provider:",
            "  provider_builder: build_cognitive_alpha_codex_exec_candidate_provider",
            "  provider_id: codex_exec_candidate_provider",
            "  command: codex exec --output-schema cognitive_alpha_candidate_schema.json --output-last-message last_message.json",
            "  approval_policy: never",
            "  sandbox: read-only",
            "factor_mining:",
            "  workflow: cognitive_alpha_factor_mining",
            "  candidate_provider: codex_exec_candidate_provider",
            f"  label_col: {summary['model_target_col']}",
            "position_model:",
            "  workflow: cognitive_alpha_position_model",
            f"  return_col: {summary['return_col']}",
            f"  label_col: {summary['model_target_col']}",
            f"  execution_timing: {summary['execution_timing']}",
            "  constrained_reference_name: base_constrained_ref",
            "direct_factor_backtest:",
            "  workflow: cognitive_alpha_weighted_factor_search",
            f"  return_col: {summary['return_col']}",
            "  role: prior_only_not_final_promotion",
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            "  workflow_config: workflow.yaml",
            "  machine_entrypoint: summary.json",
            "",
        ]
    )


def _failure_modes(summary: dict[str, Any]) -> str:
    position = summary["position_model"]
    direct = summary["direct_factor_backtest"]
    goal_loop = summary.get("goal_loop", {})
    return (
        "# Failure Modes\n\n"
        f"- goal_loop_status: {summary.get('status')}\n"
        f"- terminal_reason: {summary.get('terminal_reason')}\n"
        f"- attempts: {summary.get('attempt_index')}/{summary.get('max_attempts')}\n"
        f"- position_model_promotion: {position['promotion_decision']}\n"
        f"- direct_factor_role: {summary['research_gate']['direct_factor_backtest_role']}\n"
        f"- direct_factor_promotion: {direct['promotion_decision']}\n"
        f"- next_action: {goal_loop.get('next_action', {}).get('action', '')}\n"
        "- direct_factor_backtest 不能作为最终晋升门，只能作为 prior/diagnostic。\n"
        "- 若未跑赢 base_constrained_ref 或 walk-forward/exposure/risk gates 且仍有 attempt 预算，保持 needs_iteration 并继续 Codex exec 因子生成。\n"
        "- 若达到当前 max_attempts 仍未晋升，保持 needs_iteration，并由外层 Goal refresh attempt budget 后继续。\n"
    )


def _conclusion(summary: dict[str, Any]) -> str:
    position = summary["position_model"]
    direct = summary["direct_factor_backtest"]
    latest = summary["latest_signal"]
    goal_loop = summary.get("goal_loop", {})
    return (
        "# 结论\n\n"
        f"- what_ran: 执行 {EXPERIMENT}，使用 Codex exec candidate provider 生成 CogAlpha 候选，"
        "再通过 qlib provider、Alpha158 DatasetH/DataHandlerLP、R.start recorder、position model 和 direct-factor prior 完成研究闭环。\n"
        f"- goal_loop: status={summary.get('status')}，terminal_reason={summary.get('terminal_reason')}，"
        f"attempt={summary.get('attempt_index')}/{summary.get('max_attempts')}，"
        f"feedback_context_included={goal_loop.get('feedback_context_included')}。\n"
        f"- data: provider={summary['provider_uri']}，as_of={summary['as_of']}，"
        f"train={summary['segments']['train'][0]} 到 {summary['segments']['train'][1]}，"
        f"validation={summary['segments']['validation'][0]} 到 {summary['segments']['validation'][1]}，"
        f"test={summary['segments']['test'][0]} 到 {summary['segments']['test'][1]}。\n"
        f"- execution: timing={summary['execution_timing']}，return_col={summary['return_col']}，"
        f"model_target_col={summary['model_target_col']}，benchmark_basis={summary['benchmark_basis']}。\n"
        f"- codex_provider: provider_id={summary['codex_provider']['provider_id']}，"
        f"prompt_count={summary['codex_provider'].get('prompt_count', 0)}，sandbox=read-only，"
        "openai_api_key_required=False。\n"
        f"- evolution: generations={summary['evolution']['completed_generations']}，"
        f"selected_count={summary['evolution']['selected_count']}，"
        f"metrics={','.join(summary['evolution']['metric_names'])}。\n"
        f"- position_model: promotion={position['promotion_decision']}，"
        f"selected_return={position['comparison'].get('selected_return_pct', 0.0):.4f}%，"
        f"walk_forward_folds={position['walk_forward']['fold_count']}，"
        f"all_gates_pass={position['walk_forward'].get('all_folds_pass_risk_gates')}。\n"
        f"- direct_factor_backtest: promotion={direct['promotion_decision']}，"
        f"return={direct['comparison'].get('selected_return_pct', 0.0):.4f}%，"
        "role=prior_only_not_final_promotion。\n"
        f"- latest_signal: as_of={latest['as_of']}，action={latest['action']}，"
        f"target_position={latest['target_position']:.4f}。\n"
        f"- promotion_decision: {summary['promotion_decision']}；production_paper_sim={summary['production_paper_sim']['status']}。\n"
        f"- next_action: {goal_loop.get('next_action', {}).get('action')}；"
        "若未通过 base_constrained_ref 或 walk-forward/exposure/risk gates，继续扩展 Codex-exec 候选和稳定性筛选；"
        "当前 max_attempts 只是本次命令预算，不作为 Goal 完成条件。\n"
        f"- recorder: {summary['recorder_id']}。\n"
        "- risk: 本研究仅用于 SOXL qlib 研究和 paper/sim 风控，不构成买卖建议。\n"
    )


def _recorder_files(recorder_local_dir: Path) -> list[str]:
    return sorted(path.relative_to(recorder_local_dir).as_posix() for path in recorder_local_dir.rglob("*") if path.is_file())


def _resolve_root(root: Path | str | None) -> Path:
    current = Path(root).resolve() if root is not None else Path.cwd().resolve()
    while not (current / "AGENTS.md").exists() and current != current.parent:
        current = current.parent
    return current
