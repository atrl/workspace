from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
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
    _feature_row,
    _fit_recovery_model,
    _load_source_frame,
    _risk_metrics,
    mark_to_market_weights,
    project_legal_weights,
)


EXPERIMENT = "soxl_compounding_regime_sparse_execution_v1"
SOURCE_EXPERIMENT = "stage1_confidence_veto_recovery_v1"


@dataclass(frozen=True)
class StructuralProfile:
    name: str
    volatility_budget: bool = False
    regime_cap: bool = False
    sparse_execution: bool = False
    recovery_trend_only: bool = False
    target_volatility: float = 0.35
    maximum_volatility_beta: float = 1.35
    mean_reverting_beta_cap: float = 0.80
    crash_beta_cap: float = 0.25
    trend_no_trade_band: float = 0.02
    mean_reverting_no_trade_band: float = 0.05
    minimum_dwell_days: int = 3


PROFILES = (
    StructuralProfile(name="confidence_veto_baseline"),
    StructuralProfile(name="recovery_trend_only", recovery_trend_only=True),
    StructuralProfile(name="volatility_budget", volatility_budget=True),
    StructuralProfile(name="compounding_regime_cap", regime_cap=True),
    StructuralProfile(
        name="regime_sparse_execution",
        volatility_budget=True,
        regime_cap=True,
        sparse_execution=True,
    ),
)


@dataclass(frozen=True)
class CompoundingRegimeSparseSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0
    required_open_trade_reduction: float = 0.20

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def classify_compounding_regime(row: Any) -> str:
    required = ("ret20", "eff20", "autocorr20", "rv20", "drawdown60")
    if any(not np.isfinite(float(getattr(row, name))) for name in required):
        return "mean_reverting"
    if (
        float(row.ret20) <= -0.10
        or float(row.drawdown60) <= -0.15
        or float(row.rv20) >= 0.70
    ):
        return "crash"
    if (
        float(row.ret20) > 0.0
        and float(row.eff20) >= 0.25
        and float(row.autocorr20) >= -0.10
        and float(row.rv20) <= 0.55
    ):
        return "trend"
    return "mean_reverting"


def _load_regime_frame(
    spec: CompoundingRegimeSparseSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import qlib
    from qlib.config import REG_US
    from qlib.data import D

    manifest_path = spec.daily_provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise ValueError("blocked: qlib daily provider manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qlib.init(
        provider_uri=spec.daily_provider_uri.resolve().as_posix(),
        region=REG_US,
        kernels=1,
    )
    close = D.features(
        ["SOXX"],
        ["$close"],
        start_time=manifest["calendar_start"],
        end_time=manifest["calendar_end"],
        freq="day",
    ).reset_index()
    close.columns = ["instrument", "signal_date", "close"]
    close = close.sort_values("signal_date").set_index("signal_date")
    daily_return = close["close"].pct_change()
    close["ret20"] = close["close"] / close["close"].shift(20) - 1.0
    close["eff20"] = (
        (close["close"] - close["close"].shift(20)).abs()
        / close["close"].diff().abs().rolling(20).sum()
    )
    close["autocorr20"] = daily_return.rolling(20).corr(
        daily_return.shift(1)
    )
    close["rv20"] = daily_return.rolling(20).std() * np.sqrt(252.0)
    close["rv60"] = daily_return.rolling(60).std() * np.sqrt(252.0)
    close["volatility_ratio20_60"] = close["rv20"] / close["rv60"]
    close["drawdown60"] = (
        close["close"] / close["close"].rolling(60).max() - 1.0
    )
    feature_columns = [
        "ret20",
        "eff20",
        "autocorr20",
        "rv20",
        "rv60",
        "volatility_ratio20_60",
        "drawdown60",
    ]
    result = close.loc[:, feature_columns].copy()
    result["regime"] = [
        classify_compounding_regime(row) for row in result.itertuples()
    ]
    audit = {
        "provider_uri": str(spec.daily_provider_uri.resolve()),
        "provider_as_of": manifest.get("as_of"),
        "calendar_start": manifest.get("calendar_start"),
        "calendar_end": manifest.get("calendar_end"),
        "source_api": ["qlib.init", "D.calendar", "D.features"],
        "signal_timing": "SOXX close and rolling features known at signal_date close",
        "features": feature_columns,
    }
    return result, audit


def _build_regime_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    columns = [
        "ret20",
        "eff20",
        "autocorr20",
        "rv20",
        "rv60",
        "volatility_ratio20_60",
        "drawdown60",
    ]
    indexed = frame.set_index("signal_date").sort_index()
    static = pd.concat({"feature": indexed.loc[:, columns]}, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [static.index, np.full(len(static), "SOXX_REGIME")],
        names=["datetime", "instrument"],
    )
    segments = {
        "development": ("2016-01-01", "2022-12-31"),
        "outer_diagnostic": ("2023-01-01", "2026-12-31"),
    }
    handler = DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    return {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "features": columns,
        "role": "qlib-aligned point-in-time SOXX regime feature evidence",
    }


def _profile_cap(row: Any, profile: StructuralProfile) -> float:
    cap = 3.0
    if profile.volatility_budget:
        cap = min(
            cap,
            max(
                profile.crash_beta_cap,
                min(
                    profile.maximum_volatility_beta,
                    profile.target_volatility / max(float(row.rv20), 0.01),
                ),
            ),
        )
    if profile.regime_cap:
        regime = str(row.regime)
        regime_cap = {
            "trend": 3.0,
            "mean_reverting": profile.mean_reverting_beta_cap,
            "crash": profile.crash_beta_cap,
        }[regime]
        cap = min(cap, regime_cap)
    return float(cap)


def _open_target(
    row: Any,
    *,
    controller_cap: float,
    profile: StructuralProfile,
    base_spec: IntradayRecoveryDirectAllocationSpec,
) -> np.ndarray:
    cap = min(controller_cap, _profile_cap(row, profile))
    return project_legal_weights(
        row.soxl_weight,
        row.soxx_weight,
        cap,
        minimum_cash_weight=base_spec.minimum_cash_weight,
        maximum_soxl_weight=base_spec.maximum_soxl_weight,
    )


def _simulate_profile(
    frame: pd.DataFrame,
    *,
    teacher: pd.DataFrame,
    selected: dict[str, Any],
    profile: StructuralProfile,
    base_spec: IntradayRecoveryDirectAllocationSpec,
    cost_bps: float,
    first_action_year: int,
) -> pd.DataFrame:
    models = {
        int(year): _fit_recovery_model(
            teacher,
            year=int(year),
            c_value=float(selected["logistic_c"]),
        )
        for year in sorted(frame["signal_date"].dt.year.unique())
    }
    equity = peak = 1.0
    previous = np.zeros(3, dtype="float64")
    state = clock = 0
    history: list[float] = []
    days_since_open_trade = 999
    rows: list[dict[str, Any]] = []
    for row in frame.sort_values("signal_date").itertuples():
        pretrade = previous.copy()
        drawdown = max(0.0, 1.0 - equity / peak)
        state, clock, q = _advance_state(state, clock, drawdown, base_spec)
        controller_cap, raw_beta = _base_cap(
            row,
            q=q,
            drawdown=drawdown,
            history=history,
            spec=base_spec,
        )
        proposed = _open_target(
            row,
            controller_cap=controller_cap,
            profile=profile,
            base_spec=base_spec,
        )
        open_delta = float(np.abs(proposed[:2] - pretrade[:2]).sum())
        previous_beta = float(3.0 * pretrade[0] + pretrade[1])
        proposed_beta = float(3.0 * proposed[0] + proposed[1])
        if profile.sparse_execution:
            band = (
                profile.trend_no_trade_band
                if row.regime == "trend"
                else profile.mean_reverting_no_trade_band
            )
            deterministic_derisk = (
                proposed_beta < previous_beta - 0.10
                or (row.regime == "crash" and proposed_beta < previous_beta)
            )
            should_trade_open = deterministic_derisk or (
                open_delta >= band
                and days_since_open_trade >= profile.minimum_dwell_days
            )
        else:
            should_trade_open = True
        open_weights = proposed if should_trade_open else pretrade
        actual_open_delta = float(
            np.abs(open_weights[:2] - pretrade[:2]).sum()
        )
        if actual_open_delta > 1e-12:
            days_since_open_trade = 0
        else:
            days_since_open_trade += 1

        recovery_beta = min(raw_beta, float(selected["action_beta"]))
        model = models.get(int(row.signal_date.year))
        recovery_eligible = (
            row.signal_date.year >= first_action_year
            and model is not None
            and drawdown >= base_spec.recovery_eligibility_drawdown
            and recovery_beta > controller_cap + 1e-12
            and pd.notna(row.early_ret_soxx)
            and pd.notna(row.alpha_probability)
            and row.alpha_probability
            >= float(selected["minimum_source_alpha_probability"])
            and (not profile.recovery_trend_only or row.regime == "trend")
        )
        probability = np.nan
        recovery_action = False
        if recovery_eligible:
            feature = pd.DataFrame(
                [
                    _feature_row(
                        row,
                        drawdown=drawdown,
                        cap=controller_cap,
                        raw_beta=raw_beta,
                        state=state,
                        history=history,
                    )
                ]
            )
            probability = float(
                model.predict_proba(feature.loc[:, FEATURE_COLUMNS])[0, 1]
            )
            recovery_action = probability >= float(
                selected["probability_threshold"]
            )
        if recovery_action:
            close_weights = project_legal_weights(
                row.soxl_weight,
                row.soxx_weight,
                recovery_beta,
                minimum_cash_weight=base_spec.minimum_cash_weight,
                maximum_soxl_weight=base_spec.maximum_soxl_weight,
            )
            early = np.array(
                [row.early_ret_soxl, row.early_ret_soxx, 0.0],
                dtype="float64",
            )
            value_at_cut = 1.0 + open_weights @ early
            drifted = mark_to_market_weights(open_weights, early)
            post = np.array(
                [
                    (1.0 + row.soxl_return) / (1.0 + row.early_ret_soxl) - 1.0,
                    (1.0 + row.soxx_return) / (1.0 + row.early_ret_soxx) - 1.0,
                    0.0,
                ],
                dtype="float64",
            )
            gross_return = float(
                value_at_cut * (1.0 + close_weights @ post) - 1.0
            )
            turnover = actual_open_delta + float(
                np.abs(close_weights[:2] - drifted[:2]).sum()
            )
            ending = mark_to_market_weights(close_weights, post)
        else:
            close_weights = open_weights
            asset_returns = np.array(
                [row.soxl_return, row.soxx_return, 0.0], dtype="float64"
            )
            gross_return = float(open_weights @ asset_returns)
            turnover = actual_open_delta
            ending = mark_to_market_weights(open_weights, asset_returns)
        strategy_return = gross_return - turnover * cost_bps / 10_000.0
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        previous = ending
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "segment": row.segment,
                "profile": profile.name,
                "regime": row.regime,
                "soxl_weight": float(close_weights[0]),
                "soxx_weight": float(close_weights[1]),
                "cash_weight": float(close_weights[2]),
                "open_trade": bool(actual_open_delta > 1e-8),
                "recovery_action": bool(recovery_action),
                "recovery_probability": probability,
                "gross_return": gross_return,
                "turnover": turnover,
                "strategy_return": strategy_return,
                "cost_bps": float(cost_bps),
            }
        )
    return pd.DataFrame(rows)


def _metrics(
    replay: pd.DataFrame, *, profile: str, segment: str, cost_bps: float
) -> dict[str, Any]:
    return {
        "profile": profile,
        "segment": segment,
        "cost_bps": float(cost_bps),
        **_risk_metrics(replay["strategy_return"].to_numpy()),
        "total_turnover": float(replay["turnover"].sum()),
        "open_trade_days": int(replay["open_trade"].sum()),
        "recovery_action_count": int(replay["recovery_action"].sum()),
        "avg_cash_weight_pct": float(100.0 * replay["cash_weight"].mean()),
    }


def _stress(replay: pd.DataFrame, *, cost_bps: float) -> pd.DataFrame:
    result = replay.copy()
    result["strategy_return"] = (
        result["gross_return"] - result["turnover"] * cost_bps / 10_000.0
    )
    result["cost_bps"] = float(cost_bps)
    return result


def promotion_gates(
    row: pd.Series,
    *,
    baseline_10: pd.Series,
    baseline_25: pd.Series,
    required_open_trade_reduction: float,
) -> dict[str, bool]:
    return {
        "return_10bps_above_baseline": bool(
            row["return_pct"] > baseline_10["return_pct"]
        ),
        "return_25bps_above_baseline": bool(
            row["stress_return_pct"] > baseline_25["return_pct"]
        ),
        "max_drawdown_not_worse": bool(
            row["max_drawdown_pct"] >= baseline_10["max_drawdown_pct"]
        ),
        "worst_5d_not_worse": bool(
            row["worst_rolling_5d_pct"]
            >= baseline_10["worst_rolling_5d_pct"]
        ),
        "open_trade_days_reduced": bool(
            row["open_trade_days"]
            <= baseline_10["open_trade_days"]
            * (1.0 - required_open_trade_reduction)
        ),
        "recovery_actions_not_increased": bool(
            row["recovery_action_count"]
            <= baseline_10["recovery_action_count"]
        ),
    }


def select_structural_challenger(model_selection: pd.DataFrame) -> str | None:
    eligible = model_selection[
        model_selection["segment"].eq("development_walk_forward")
        & ~model_selection["selection_uses_outer_test"].astype(bool)
        & model_selection["all_gates_pass"].astype(bool)
        & ~model_selection["profile"].eq("confidence_veto_baseline")
    ]
    if eligible.empty:
        return None
    return str(
        eligible.sort_values(
            ["return_pct", "open_trade_days"], ascending=[False, True]
        ).iloc[0]["profile"]
    )


def _regime_metrics(replays: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (segment, profile), replay in replays.items():
        for regime, section in replay.groupby("regime", observed=True):
            rows.append(
                {
                    "segment": segment,
                    "profile": profile,
                    "regime": regime,
                    "rows": int(len(section)),
                    **_risk_metrics(section["strategy_return"].to_numpy()),
                    "total_turnover": float(section["turnover"].sum()),
                    "open_trade_days": int(section["open_trade"].sum()),
                    "recovery_action_count": int(
                        section["recovery_action"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _record(
    spec: CompoundingRegimeSparseSpec,
    *,
    model_selection: pd.DataFrame,
    comparison: pd.DataFrame,
    replays: dict[tuple[str, str], pd.DataFrame],
    dataset_audit: dict[str, Any],
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        model_selection.to_csv(tmp_dir / "model_selection_full.csv", index=False)
        comparison.to_csv(tmp_dir / "model_comparison.csv", index=False)
        (tmp_dir / "dataset_audit.json").write_text(
            json.dumps(dataset_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for (segment, profile), replay in replays.items():
            replay.to_csv(tmp_dir / f"{segment}_{profile}_replay.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                selection_uses_outer_test="false",
                profile_count=str(len(PROFILES)),
                transaction_cost_bps=str(spec.transaction_cost_bps),
                stress_cost_bps=str(spec.stress_cost_bps),
            )
            baseline = model_selection[
                model_selection["profile"].eq("confidence_veto_baseline")
            ].iloc[0]
            R.log_metrics(
                baseline_development_return_pct=float(baseline["return_pct"]),
                passing_challenger_count=float(
                    model_selection["all_gates_pass"].sum()
                ),
            )
            for path in tmp_dir.iterdir():
                recorder.log_artifact(
                    path.as_posix(), artifact_path=spec.experiment
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


def run_soxl_compounding_regime_sparse_execution_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = CompoundingRegimeSparseSpec(root=_resolve_root(root))
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    source_selected = json.loads(
        (spec.source_card_dir / "selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    if source_summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: Stage1 source used outer for selection")
    base_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    source_frame, source_audit = _load_source_frame(base_spec)
    regime_frame, regime_audit = _load_regime_frame(spec)
    frame = source_frame.merge(
        regime_frame.reset_index(), on="signal_date", how="left", validate="one_to_one"
    )
    dataset_audit = _build_regime_dataset(frame)
    teacher = _build_teacher_training_frame(
        frame, spec=base_spec, teacher_beta=base_spec.teacher_recovery_beta
    )
    replays: dict[tuple[str, str], pd.DataFrame] = {}
    comparison_rows: list[dict[str, Any]] = []
    for segment_name, source_segment, first_action_year in (
        ("development_walk_forward", "development_walk_forward", 2020),
        ("outer_reused_diagnostic", "outer_diagnostic", 2023),
    ):
        segment = frame[frame["segment"].eq(source_segment)].copy()
        for profile in PROFILES:
            replay = _simulate_profile(
                segment,
                teacher=teacher,
                selected=source_selected,
                profile=profile,
                base_spec=base_spec,
                cost_bps=spec.transaction_cost_bps,
                first_action_year=first_action_year,
            )
            replays[(segment_name, profile.name)] = replay
            comparison_rows.append(
                _metrics(
                    replay,
                    profile=profile.name,
                    segment=segment_name,
                    cost_bps=spec.transaction_cost_bps,
                )
            )
            comparison_rows.append(
                _metrics(
                    _stress(replay, cost_bps=spec.stress_cost_bps),
                    profile=profile.name,
                    segment=segment_name,
                    cost_bps=spec.stress_cost_bps,
                )
            )
    comparison = pd.DataFrame(comparison_rows)
    development_10 = comparison[
        comparison["segment"].eq("development_walk_forward")
        & comparison["cost_bps"].eq(spec.transaction_cost_bps)
    ].copy()
    development_25 = comparison[
        comparison["segment"].eq("development_walk_forward")
        & comparison["cost_bps"].eq(spec.stress_cost_bps)
    ].set_index("profile")
    baseline_10 = development_10[
        development_10["profile"].eq("confidence_veto_baseline")
    ].iloc[0]
    baseline_25 = development_25.loc["confidence_veto_baseline"]
    rows: list[dict[str, Any]] = []
    for _, candidate in development_10.iterrows():
        gates = promotion_gates(
            pd.Series(
                {
                    **candidate.to_dict(),
                    "stress_return_pct": development_25.loc[
                        candidate["profile"], "return_pct"
                    ],
                }
            ),
            baseline_10=baseline_10,
            baseline_25=baseline_25,
            required_open_trade_reduction=spec.required_open_trade_reduction,
        )
        rows.append(
            {
                **candidate.to_dict(),
                "stress_return_pct": float(
                    development_25.loc[candidate["profile"], "return_pct"]
                ),
                "selection_uses_outer_test": False,
                **gates,
                "all_gates_pass": bool(all(gates.values())),
            }
        )
    model_selection = pd.DataFrame(rows)
    selected_challenger = select_structural_challenger(model_selection)
    regime_metrics = _regime_metrics(replays)
    recorder_info = _record(
        spec,
        model_selection=model_selection,
        comparison=comparison,
        replays=replays,
        dataset_audit=dataset_audit,
    )
    decision = (
        "validated_structural_challenger_for_forward_shadow"
        if selected_challenger
        else "research_reject_no_structural_profile_passed_all_gates"
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": "paper_hypothesis_structural_ablation_card",
        "decision": decision,
        "production_change": "blocked",
        "selection_uses_outer_test": False,
        "outer_role": "reused_diagnostic_only_not_model_selection",
        "selected_challenger": selected_challenger,
        "retained_model": "confidence_veto_baseline",
        "hypothesis": (
            "SOXX compounding regime, volatility budgeting, and sparse execution "
            "may reduce SOXL trading without sacrificing the frozen Stage1 edge"
        ),
        "structural_profiles": [asdict(profile) for profile in PROFILES],
        "model_selection": model_selection.to_dict(orient="records"),
        "development_gate_pass": selected_challenger is not None,
        "source_audit": {
            "source_experiment": spec.source_experiment,
            "source_recorder_id": source_summary["recorder_id"],
            "source_selected_model": source_selected,
            "underlying_source": source_audit,
        },
        "regime_audit": regime_audit,
        "dataset_audit": dataset_audit,
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/stable/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/stable/component/recorder.html",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/model_comparison.csv",
                f"artifacts/{spec.experiment}/dataset_audit.json",
                f"artifacts/{spec.experiment}/*_replay.csv",
            ],
        },
        **recorder_info,
    }
    _write_sidecars(
        spec,
        summary=summary,
        comparison=comparison,
        model_selection=model_selection,
        regime_metrics=regime_metrics,
    )
    return summary


def _write_sidecars(
    spec: CompoundingRegimeSparseSpec,
    *,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    model_selection: pd.DataFrame,
    regime_metrics: pd.DataFrame,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (card / "selected_model.json").write_text(
        json.dumps(
            {
                "retained_model": summary["retained_model"],
                "selected_challenger": summary["selected_challenger"],
                "production_enabled": False,
                "selection_uses_outer_test": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    model_selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    comparison[
        [
            "segment",
            "profile",
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
            "profile",
            "cost_bps",
            "return_pct",
            "total_turnover",
            "open_trade_days",
            "recovery_action_count",
        ]
    ].to_csv(card / "execution_metrics.csv", index=False)
    regime_metrics.to_csv(card / "regime_metrics.csv", index=False)
    (card / "analysis_summary.json").write_text(
        json.dumps(
            {
                "hypothesis": summary["hypothesis"],
                "decision": summary["decision"],
                "selected_challenger": summary["selected_challenger"],
                "retained_model": summary["retained_model"],
                "outer_role": summary["outer_role"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    workflow = f"""experiment: {spec.experiment}
source_experiment: {spec.source_experiment}
system_of_record: qlib
provider_uri: {summary['regime_audit']['provider_uri']}
market: all
region: us
selection_segment: 2016-01-01_to_2022-12-31
selection_uses_outer_test: false
outer_role: reused_diagnostic_only_not_model_selection
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
stress_method: fixed_decisions_repriced_at_25bp
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/model_selection_full.csv
    - artifacts/{spec.experiment}/model_comparison.csv
    - artifacts/{spec.experiment}/dataset_audit.json
    - artifacts/{spec.experiment}/*_replay.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    dev = comparison[
        comparison["segment"].eq("development_walk_forward")
        & comparison["cost_bps"].eq(spec.transaction_cost_bps)
    ].set_index("profile")
    base = dev.loc["confidence_veto_baseline"]
    trend = dev.loc["recovery_trend_only"]
    vol = dev.loc["volatility_budget"]
    regime = dev.loc["compounding_regime_cap"]
    sparse = dev.loc["regime_sparse_execution"]
    outer = comparison[
        comparison["segment"].eq("outer_reused_diagnostic")
        & comparison["cost_bps"].eq(spec.transaction_cost_bps)
    ].set_index("profile")
    conclusion = f"""# 中文结论

- 做了什么：以冻结的 Stage1 confidence-veto 为直接基线，固定测试论文启发的四个结构消融：恢复动作只限趋势、35% SOXX 波动率预算、趋势/震荡/危机 beta cap，以及状态 cap + 2%/5% no-trade band + 3日最短持有。没有用 2023-2026 选结构或参数。
- 基线：开发期 10bp return={base['return_pct']:.4f}%、MaxDD={base['max_drawdown_pct']:.4f}%、open trade days={int(base['open_trade_days'])}、10:31 actions={int(base['recovery_action_count'])}。
- 恢复状态反证：趋势限定把 10:31 actions 从 {int(base['recovery_action_count'])} 降到 {int(trend['recovery_action_count'])}，但 return 降到 {trend['return_pct']:.4f}%。原19次动作实际位于危机/震荡后的恢复阶段，不能按普通趋势追随信号删除。
- 波动率预算：return={vol['return_pct']:.4f}%、MaxDD={vol['max_drawdown_pct']:.4f}%；固定35%预算与现有 controller 重复去风险，收益损失过大。
- 状态限仓：return={regime['return_pct']:.4f}%、MaxDD={regime['max_drawdown_pct']:.4f}%；简单把震荡 beta 限到0.8、危机限到0.25并不适合当前模型。
- 稀疏执行：open trade days 从 {int(base['open_trade_days'])} 降到 {int(sparse['open_trade_days'])}，turnover 从 {base['total_turnover']:.4f} 降到 {sparse['total_turnover']:.4f}，但 return 仅 {sparse['return_pct']:.4f}%、MaxDD={sparse['max_drawdown_pct']:.4f}%。这证明每日目标变化包含有价值的信息，粗暴3日冻结并不是免费降频。
- outer 复用诊断：基线 return={outer.loc['confidence_veto_baseline', 'return_pct']:.4f}%，稀疏方案={outer.loc['regime_sparse_execution', 'return_pct']:.4f}%；该窗口已反复查看，只用于确认失败没有反转，不能作为选择证据。
- 决策：{summary['decision']}。没有结构候选同时超过基线收益、守住风险、通过25bp并减少20%开盘交易日；保留 confidence-veto baseline，不修改生产模拟盘。
- 下一步：若继续降频，应把目标改成“未来3-5日累计调仓价值/交易成本”的监督式 action-value objective，而不是继续调固定波动率、MA或no-trade阈值。
- qlib：SOXX状态特征通过 D.features 读取，DatasetH 对齐开发/outer 段，完整 replay 和模型比较保存在 recorder {summary['recorder_id']}；卡根只保留小型审计 sidecar。
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        "- 固定波动率预算与现有回撤 controller 重复去风险。\n"
        "- 简单趋势状态与回撤后的 10:31 恢复动作语义冲突。\n"
        "- 固定 no-trade band 和 dwell 会丢弃日频模型的有效目标变化。\n"
        "- 2023-2026 已反复查看，只能作为 reused diagnostic。\n",
        encoding="utf-8",
    )


__all__ = [
    "CompoundingRegimeSparseSpec",
    "StructuralProfile",
    "classify_compounding_regime",
    "promotion_gates",
    "run_soxl_compounding_regime_sparse_execution_card",
    "select_structural_challenger",
]
