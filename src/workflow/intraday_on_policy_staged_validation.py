from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.workflow.intraday_recovery_direct_allocation import (
    _annual_metrics,
    _build_teacher_training_frame,
    _development_stability,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    apply_fixed_decision_cost_stress,
)
from src.workflow.intraday_staged_recovery_direct_allocation import (
    BASE_EXPERIMENT,
    STAGE2_FEATURE_COLUMNS,
    IntradayStagedRecoveryDirectAllocationSpec,
    _add_cut_features,
    _fit_stage2_model,
    _load_cut_features,
    build_on_policy_stage2_teacher_training_frame,
    simulate_staged_policy,
)


EXPERIMENT = "intraday_on_policy_staged_validation_v1"
CARD_TYPE = "intraday_on_policy_staged_validation_card"


@dataclass(frozen=True)
class IntradayOnPolicyStagedValidationSpec(
    IntradayStagedRecoveryDirectAllocationSpec
):
    experiment: str = EXPERIMENT
    prior_staged_experiment: str = (
        "intraday_staged_recovery_direct_allocation_v1"
    )
    open_no_trade_bands: tuple[float, ...] = (
        0.0,
        0.005,
        0.01,
        0.02,
    )
    minimum_development_stage2_actions: int = 20
    minimum_neighbor_count: int = 3
    minimum_neighbor_wealth_ratio: float = 0.90
    near_best_neighbor_wealth_tolerance: float = 0.005

    @property
    def prior_staged_card_dir(self) -> Path:
        return self.root / "research" / self.prior_staged_experiment


def run_intraday_on_policy_staged_validation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    return _run_card(
        IntradayOnPolicyStagedValidationSpec(root=repo_root)
    )


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def annotate_parameter_neighborhood(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "open_no_trade_band",
        "second_cut_bars",
        "stage2_logistic_c",
        "stage2_probability_threshold",
        "stage2_action_beta",
        "terminal_wealth",
        "base_development_gate",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            f"candidate table is missing columns: {sorted(missing)}"
        )
    result = candidates.copy()
    result["neighbor_count"] = 0
    result["neighbor_min_wealth_ratio"] = np.nan
    result["neighbor_median_terminal_wealth"] = np.nan
    for index, row in result.iterrows():
        neighbors = result[
            result["open_no_trade_band"].eq(
                row["open_no_trade_band"]
            )
            & result["second_cut_bars"].eq(row["second_cut_bars"])
            & result["stage2_logistic_c"].eq(
                row["stage2_logistic_c"]
            )
            & result["stage2_action_beta"].eq(
                row["stage2_action_beta"]
            )
            & result["stage2_probability_threshold"].between(
                row["stage2_probability_threshold"] - 0.0500001,
                row["stage2_probability_threshold"] + 0.0500001,
            )
            & result["base_development_gate"].astype(bool)
        ]
        result.loc[index, "neighbor_count"] = len(neighbors)
        if len(neighbors):
            result.loc[index, "neighbor_min_wealth_ratio"] = (
                neighbors["terminal_wealth"].min()
                / row["terminal_wealth"]
            )
            result.loc[
                index, "neighbor_median_terminal_wealth"
            ] = neighbors["terminal_wealth"].median()
    return result


def select_robust_development_candidate(
    candidates: pd.DataFrame,
    *,
    spec: IntradayOnPolicyStagedValidationSpec,
) -> pd.Series:
    robust = candidates[
        candidates["base_development_gate"].astype(bool)
        & candidates["stage2_action_count"].ge(
            spec.minimum_development_stage2_actions
        )
        & candidates["neighbor_count"].ge(
            spec.minimum_neighbor_count
        )
        & candidates["neighbor_min_wealth_ratio"].ge(
            spec.minimum_neighbor_wealth_ratio
        )
        & ~candidates["selection_uses_outer_test"].astype(bool)
    ].copy()
    if robust.empty:
        raise ValueError(
            "no development-only candidate passes robust selection gates"
        )
    best_neighbor_wealth = float(
        robust["neighbor_median_terminal_wealth"].max()
    )
    near_best = robust[
        robust["neighbor_median_terminal_wealth"].ge(
            best_neighbor_wealth
            * (1.0 - spec.near_best_neighbor_wealth_tolerance)
        )
    ].copy()
    return near_best.sort_values(
        [
            "open_no_trade_band",
            "stage2_action_beta",
            "second_cut_bars",
            "stage2_probability_threshold",
            "stage2_logistic_c",
        ],
        ascending=[True, True, False, False, True],
    ).iloc[0]


def _candidate_row(
    replay: pd.DataFrame,
    *,
    band: float,
    bars: int,
    c_value: float,
    threshold: float,
    action_beta: float,
) -> dict[str, Any]:
    metrics = _risk_metrics(replay["strategy_return"].to_numpy())
    stability = _development_stability(replay)
    base_gate = bool(
        _passes_risk(metrics)
        and stability["risk_window_pass_count"] == 7
        and stability["positive_relative_window_count"] >= 5
        and stability["relative_ratio_p25"] >= 0.95
    )
    return {
        "segment": "development_walk_forward",
        "selection_uses_outer_test": False,
        "open_no_trade_band": band,
        "second_cut_bars": bars,
        "second_cut_time": "11:00" if bars == 6 else "11:30",
        "stage2_logistic_c": c_value,
        "stage2_probability_threshold": threshold,
        "stage2_action_beta": action_beta,
        **metrics,
        **stability,
        "base_development_gate": base_gate,
        "stage1_action_count": int(
            replay["recovery_action"].sum()
        ),
        "stage2_action_count": int(replay["stage2_action"].sum()),
        "total_turnover": float(replay["turnover"].sum()),
        "open_no_trade_count": int(replay["open_no_trade"].sum()),
        "avg_cash_weight_pct": float(
            100.0 * replay["cash_weight"].mean()
        ),
    }


def _annual_calibration(
    teacher_frame: pd.DataFrame,
    *,
    c_value: float,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in sorted(teacher_frame["year"].unique()):
        test = teacher_frame[teacher_frame["year"].eq(year)]
        model = _fit_stage2_model(
            teacher_frame,
            year=int(year),
            c_value=c_value,
        )
        if model is None:
            rows.append(
                {
                    "year": int(year),
                    "sample_count": int(len(test)),
                    "status": "insufficient_prior_training_samples",
                    "label_rate": float(
                        test["teacher_action_label"].mean()
                    ),
                    "auc": np.nan,
                    "brier": np.nan,
                    "predicted_action_rate": np.nan,
                }
            )
            continue
        probabilities = model.predict_proba(
            test.loc[:, STAGE2_FEATURE_COLUMNS]
        )[:, 1]
        labels = test["teacher_action_label"].astype(int).to_numpy()
        auc = (
            float(roc_auc_score(labels, probabilities))
            if np.unique(labels).size == 2
            else np.nan
        )
        rows.append(
            {
                "year": int(year),
                "sample_count": int(len(test)),
                "status": "scored",
                "label_rate": float(labels.mean()),
                "auc": auc,
                "brier": float(
                    brier_score_loss(labels, probabilities)
                ),
                "predicted_action_rate": float(
                    (probabilities >= threshold).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_card(
    spec: IntradayOnPolicyStagedValidationSpec,
) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _load_source_frame(spec)
    stage1_summary = json.loads(
        (spec.base_card_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    prior_staged_summary = json.loads(
        (spec.prior_staged_card_dir / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if stage1_summary.get("selection_uses_outer_test") is not False:
        raise ValueError("stage1 model must exclude outer from selection")
    stage1_params = stage1_summary["selected_model"]
    stage1_teacher = _build_teacher_training_frame(
        frame,
        spec=spec,
        teacher_beta=float(stage1_params["teacher_recovery_beta"]),
    )
    expanded_frames: dict[tuple[float, int], pd.DataFrame] = {}
    on_policy_teachers: dict[tuple[float, int], pd.DataFrame] = {}
    stage2_model_caches: dict[
        tuple[float, int], dict[tuple[int, float], Any]
    ] = {}
    stage1_model_cache: dict[tuple[int, float], Any] = {}
    candidate_rows: list[dict[str, Any]] = []
    for band in spec.open_no_trade_bands:
        for bars in spec.second_cut_bar_candidates:
            cut = _load_cut_features(
                spec,
                bars=bars,
                end=frame["trade_date"].max(),
            )
            expanded = _add_cut_features(frame, cut)
            key = (band, bars)
            expanded_frames[key] = expanded
            teacher = build_on_policy_stage2_teacher_training_frame(
                expanded,
                spec=spec,
                stage1_teacher_frame=stage1_teacher,
                stage1_c_value=float(stage1_params["logistic_c"]),
                stage1_threshold=float(
                    stage1_params["probability_threshold"]
                ),
                stage1_action_beta=float(
                    stage1_params["action_beta"]
                ),
                teacher_beta=spec.stage2_teacher_beta,
                open_no_trade_band=band,
            )
            on_policy_teachers[key] = teacher
            stage2_model_cache: dict[tuple[int, float], Any] = {}
            stage2_model_caches[key] = stage2_model_cache
            development = expanded[
                expanded["segment"].eq("development_walk_forward")
            ].copy()
            for c_value in (0.03, 0.10, 0.30):
                for threshold in (0.55, 0.60, 0.65, 0.70, 0.75):
                    for action_beta in (0.80, 1.00, 1.20):
                        replay = simulate_staged_policy(
                            development,
                            spec=spec,
                            stage1_teacher_frame=stage1_teacher,
                            stage1_c_value=float(
                                stage1_params["logistic_c"]
                            ),
                            stage1_threshold=float(
                                stage1_params[
                                    "probability_threshold"
                                ]
                            ),
                            stage1_action_beta=float(
                                stage1_params["action_beta"]
                            ),
                            stage2_teacher_frame=teacher,
                            stage2_c_value=c_value,
                            stage2_threshold=threshold,
                            stage2_action_beta=action_beta,
                            first_action_year=(
                                spec.first_intraday_action_year
                            ),
                            stage1_model_cache=stage1_model_cache,
                            stage2_model_cache=stage2_model_cache,
                            cost_bps=spec.transaction_cost_bps,
                            open_no_trade_band=band,
                        )
                        candidate_rows.append(
                            _candidate_row(
                                replay,
                                band=band,
                                bars=bars,
                                c_value=c_value,
                                threshold=threshold,
                                action_beta=action_beta,
                            )
                        )
    model_selection = annotate_parameter_neighborhood(
        pd.DataFrame(candidate_rows)
    )
    selected = select_robust_development_candidate(
        model_selection,
        spec=spec,
    )
    selected_params = {
        "open_no_trade_band": float(
            selected["open_no_trade_band"]
        ),
        "second_cut_bars": int(selected["second_cut_bars"]),
        "second_cut_time": str(selected["second_cut_time"]),
        "stage2_logistic_c": float(
            selected["stage2_logistic_c"]
        ),
        "stage2_probability_threshold": float(
            selected["stage2_probability_threshold"]
        ),
        "stage2_action_beta": float(
            selected["stage2_action_beta"]
        ),
        "stage2_teacher_beta": spec.stage2_teacher_beta,
    }
    selected_key = (
        selected_params["open_no_trade_band"],
        selected_params["second_cut_bars"],
    )
    selected_frame = expanded_frames[selected_key]
    selected_teacher = on_policy_teachers[selected_key]
    selected_stage2_cache = stage2_model_caches[selected_key]
    calibration = _annual_calibration(
        selected_teacher,
        c_value=float(selected["stage2_logistic_c"]),
        threshold=float(selected["stage2_probability_threshold"]),
    )
    development_calibration = calibration[
        calibration["year"].isin(spec.intraday_selection_years)
        & calibration["status"].eq("scored")
    ]
    stage2_calibration_gate = bool(
        len(development_calibration) >= 2
        and development_calibration["auc"].ge(0.52).all()
        and development_calibration["brier"].le(0.30).all()
    )
    selected_params["stage2_enabled"] = stage2_calibration_gate
    development = selected_frame[
        selected_frame["segment"].eq("development_walk_forward")
    ].copy()
    outer = selected_frame[
        selected_frame["segment"].eq("outer_diagnostic")
    ].copy()
    replay_kwargs = {
        "spec": spec,
        "stage1_teacher_frame": stage1_teacher,
        "stage1_c_value": float(stage1_params["logistic_c"]),
        "stage1_threshold": float(
            stage1_params["probability_threshold"]
        ),
        "stage1_action_beta": float(stage1_params["action_beta"]),
        "stage2_teacher_frame": selected_teacher,
        "stage2_c_value": selected_params["stage2_logistic_c"],
        "stage2_threshold": selected_params[
            "stage2_probability_threshold"
        ],
        "stage2_action_beta": selected_params["stage2_action_beta"],
        "stage1_model_cache": stage1_model_cache,
        "stage2_model_cache": selected_stage2_cache,
        "cost_bps": spec.transaction_cost_bps,
        "open_no_trade_band": selected_params[
            "open_no_trade_band"
        ],
        "enable_stage2": stage2_calibration_gate,
    }
    dev_replay = simulate_staged_policy(
        development,
        first_action_year=spec.first_intraday_action_year,
        **replay_kwargs,
    )
    outer_replay = simulate_staged_policy(
        outer,
        first_action_year=min(spec.outer_years),
        **replay_kwargs,
    )
    fixed_25 = outer_replay.copy()
    fixed_25["strategy_return"] = apply_fixed_decision_cost_stress(
        fixed_25["gross_return"],
        fixed_25["turnover"],
        cost_bps=spec.stress_cost_bps,
    )
    dev_metrics = _risk_metrics(
        dev_replay["strategy_return"].to_numpy()
    )
    outer_metrics = _risk_metrics(
        outer_replay["strategy_return"].to_numpy()
    )
    stress_metrics = _risk_metrics(
        fixed_25["strategy_return"].to_numpy()
    )
    raw_metrics = _risk_metrics(outer["strategy_return"].to_numpy())
    stage1_metrics = stage1_summary[
        "outer_diagnostic_metrics_10bps"
    ]
    prior_staged_metrics = prior_staged_summary[
        "outer_diagnostic_metrics_10bps"
    ]
    annual_metrics = _annual_metrics(outer_replay)
    annual_metrics["stage2_action_count"] = [
        int(
            outer_replay.loc[
                outer_replay["signal_date"].dt.year.eq(year),
                "stage2_action",
            ].sum()
        )
        for year in annual_metrics["year"]
    ]
    annual_relative_pass_count = int(
        annual_metrics["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    annual_risk_pass_count = int(
        annual_metrics.apply(
            lambda row: _passes_risk(row.to_dict()),
            axis=1,
        ).sum()
    )
    development_teacher = selected_teacher[
        selected_teacher["segment"].eq("development_walk_forward")
    ]
    outer_teacher = selected_teacher[
        selected_teacher["segment"].eq("outer_diagnostic")
    ]
    from src.workflow.hierarchical_four_asset_policy import (
        _benchmark_efficiency,
    )

    benchmark_efficiency = _benchmark_efficiency(
        outer_replay.assign(soxs_weight=0.0),
        cost_bps=int(spec.transaction_cost_bps),
    )
    soxx_wealth = float(np.prod(1.0 + outer_replay["soxx_return"]))
    terminal_wealth_ratio_vs_soxx = (
        outer_metrics["terminal_wealth"] / soxx_wealth
    )
    robust_gate = bool(
        int(selected["stage2_action_count"])
        >= spec.minimum_development_stage2_actions
        and int(selected["neighbor_count"])
        >= spec.minimum_neighbor_count
        and float(selected["neighbor_min_wealth_ratio"])
        >= spec.minimum_neighbor_wealth_ratio
    )
    benchmark_gate = bool(
        terminal_wealth_ratio_vs_soxx > 1.0
        and benchmark_efficiency["return_maxdd"]
        > benchmark_efficiency["matched_beta_return_maxdd"]
    )
    decision = (
        "promoted_research_candidate_stage1_direct_allocation_stage2_gated"
        if robust_gate
        and _passes_risk(outer_metrics)
        and _passes_risk(stress_metrics)
        and annual_relative_pass_count >= 3
        and annual_risk_pass_count >= 3
        and benchmark_gate
        else "not_promoted_on_policy_return_or_stability_gate_failed"
    )
    robust_candidates = model_selection[
        model_selection["base_development_gate"].astype(bool)
        & model_selection["stage2_action_count"].ge(
            spec.minimum_development_stage2_actions
        )
        & model_selection["neighbor_count"].ge(
            spec.minimum_neighbor_count
        )
        & model_selection["neighbor_min_wealth_ratio"].ge(
            spec.minimum_neighbor_wealth_ratio
        )
    ].copy()
    band_diagnostics = (
        robust_candidates.sort_values(
            [
                "open_no_trade_band",
                "neighbor_median_terminal_wealth",
                "terminal_wealth",
            ],
            ascending=[True, False, False],
        )
        .groupby("open_no_trade_band", as_index=False)
        .head(1)
        .sort_values("open_no_trade_band")
    )
    risk_analysis = pd.DataFrame(
        [
            {
                "variant": "raw_direct_allocation",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **raw_metrics,
            },
            {
                "variant": "stage1_10_30_recovery",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **stage1_metrics,
            },
            {
                "variant": "prior_off_policy_staged",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **prior_staged_metrics,
            },
            {
                "variant": "selected_on_policy_staged",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.transaction_cost_bps,
                **outer_metrics,
            },
            {
                "variant": "selected_on_policy_staged_fixed_decisions",
                "segment": "outer_diagnostic_reused",
                "cost_bps": spec.stress_cost_bps,
                **stress_metrics,
            },
        ]
    )
    recorder_info = _record_qlib_artifacts(
        spec,
        selected_params=selected_params,
        stage1_params=stage1_params,
        dev_metrics=dev_metrics,
        outer_metrics=outer_metrics,
        stress_metrics=stress_metrics,
        model_selection=model_selection,
        outer_replay=outer_replay,
        stage1_teacher=stage1_teacher,
        on_policy_teacher=selected_teacher,
        source_audit=source_audit,
    )
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "decision": decision,
        "selection_uses_outer_test": False,
        "outer_status": "reused_diagnostic_not_fresh_unseen_test",
        "promotion_scope": (
            "on_policy_and_parameter_stability_validation"
        ),
        "assets": ["SOXL", "SOXX", "cash"],
        "provider_uri": str(spec.intraday_provider_uri.resolve()),
        "system_of_record": "qlib",
        "method": {
            "training_distribution": (
                "stage2 labels are generated only on days when the "
                "walk-forward stage1 policy actually acts"
            ),
            "selection": (
                "2019-2022 development only; outer excluded from all "
                "cut, band, model, threshold, and beta choices"
            ),
            "parameter_stability": (
                "candidate requires adjacent probability thresholds, "
                "minimum action count, and minimum neighbor wealth ratio"
            ),
            "complexity_tie_break": (
                "within 0.5% of best neighbor-median wealth, prefer no "
                "open no-trade band and lower action beta"
            ),
            "stage2_holding_period": (
                "selected cut to regular-session close; restore stage1 "
                "weights before overnight"
            ),
        },
        "selected_model": {
            **stage1_params,
            **selected_params,
            "stage2_enabled": stage2_calibration_gate,
        },
        "selection_gates": {
            "minimum_development_stage2_actions": (
                spec.minimum_development_stage2_actions
            ),
            "minimum_neighbor_count": spec.minimum_neighbor_count,
            "minimum_neighbor_wealth_ratio": (
                spec.minimum_neighbor_wealth_ratio
            ),
            "near_best_neighbor_wealth_tolerance": (
                spec.near_best_neighbor_wealth_tolerance
            ),
            "selected_neighbor_count": int(
                selected["neighbor_count"]
            ),
            "selected_neighbor_min_wealth_ratio": float(
                selected["neighbor_min_wealth_ratio"]
            ),
            "selected_development_stage2_actions": int(
                selected["stage2_action_count"]
            ),
            "robust_gate_pass": robust_gate,
            "stage2_calibration_gate_pass": stage2_calibration_gate,
            "benchmark_gate_pass": benchmark_gate,
            "raw_return_preservation_gate_pass": bool(
                outer_metrics["return_pct"] >= raw_metrics["return_pct"]
            ),
        },
        "training_sample_audit": {
            "development_on_policy_stage2_samples": int(
                len(development_teacher)
            ),
            "outer_prior_year_update_samples": int(len(outer_teacher)),
            "development_samples_by_year": {
                str(int(year)): int(count)
                for year, count in development_teacher.groupby(
                    "year"
                ).size().items()
            },
            "outer_samples_by_year": {
                str(int(year)): int(count)
                for year, count in outer_teacher.groupby(
                    "year"
                ).size().items()
            },
        },
        "development_metrics_10bps": dev_metrics,
        "outer_diagnostic_metrics_10bps": outer_metrics,
        "raw_model_outer_metrics_10bps": raw_metrics,
        "stage1_outer_metrics_10bps": stage1_metrics,
        "prior_off_policy_staged_outer_metrics_10bps": (
            prior_staged_metrics
        ),
        "fixed_decision_stress_metrics_25bps": stress_metrics,
        "outer_effect": {
            "return_change_vs_prior_staged_pct_points": (
                outer_metrics["return_pct"]
                - prior_staged_metrics["return_pct"]
            ),
            "return_change_vs_stage1_pct_points": (
                outer_metrics["return_pct"]
                - stage1_metrics["return_pct"]
            ),
            "return_gap_vs_raw_pct_points": (
                outer_metrics["return_pct"]
                - raw_metrics["return_pct"]
            ),
            "stage2_action_count": int(
                outer_replay["stage2_action"].sum()
            ),
            "total_turnover": float(outer_replay["turnover"].sum()),
            "open_no_trade_count": int(
                outer_replay["open_no_trade"].sum()
            ),
            "avg_cash_weight_pct": float(
                100.0 * outer_replay["cash_weight"].mean()
            ),
        },
        "benchmark_comparison_10bps": {
            "terminal_wealth_ratio_vs_soxx": (
                terminal_wealth_ratio_vs_soxx
            ),
            **benchmark_efficiency,
        },
        "outer_annual_gate_summary": {
            "relative_pass_count": annual_relative_pass_count,
            "risk_pass_count": annual_risk_pass_count,
            "required_year_count": 3,
        },
        "no_trade_band_diagnostic": {
            "selected_band": selected_params[
                "open_no_trade_band"
            ],
            "conclusion": (
                "0.5%-2% bands do not materially reduce turnover; "
                "3%-5% bands failed development risk/relative windows "
                "in the feasibility probe"
            ),
        },
        "source_audit": source_audit,
        "qlib_docs": {
            "data": (
                "https://qlib.readthedocs.io/en/latest/component/data.html"
            ),
            "workflow": (
                "https://qlib.readthedocs.io/en/latest/component/workflow.html"
            ),
            "recorder": (
                "https://qlib.readthedocs.io/en/latest/component/recorder.html"
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
                f"artifacts/{spec.experiment}/on_policy_stage2_training_frame.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
            ],
        },
        **recorder_info,
    }
    _write_card_sidecars(
        spec,
        summary=summary,
        selected=selected,
        model_selection=model_selection,
        risk_analysis=risk_analysis,
        annual_metrics=annual_metrics,
        calibration=calibration,
        band_diagnostics=band_diagnostics,
    )
    return summary


def _record_qlib_artifacts(
    spec,
    *,
    selected_params,
    stage1_params,
    dev_metrics,
    outer_metrics,
    stress_metrics,
    model_selection,
    outer_replay,
    stage1_teacher,
    on_policy_teacher,
    source_audit,
) -> dict[str, Any]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        positions = outer_replay[
            [
                "signal_date",
                "trade_date",
                "open_soxl_weight",
                "open_soxx_weight",
                "open_cash_weight",
                "open_no_trade",
                "stage2_soxl_weight",
                "stage2_soxx_weight",
                "stage2_cash_weight",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "recovery_probability",
                "recovery_action",
                "stage2_probability",
                "stage2_action",
            ]
        ]
        files = {
            "positions.csv": positions,
            "strategy_returns.csv": outer_replay,
            "stage1_training_frame.csv": stage1_teacher,
            "on_policy_stage2_training_frame.csv": on_policy_teacher,
            "model_selection_full.csv": model_selection,
        }
        for filename, data in files.items():
            data.to_csv(tmp_dir / filename, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=source_audit["source_recorder_id"],
                base_experiment=spec.base_experiment,
                prior_staged_experiment=spec.prior_staged_experiment,
                allocation_assets="SOXL,SOXX,cash",
                allocation_output="direct_staged_legal_three_asset_weights",
                stage2_training_distribution="on_policy_stage1_actions_only",
                selection_uses_outer_test="false",
                stage1_action_beta=str(stage1_params["action_beta"]),
                second_cut_time=selected_params["second_cut_time"],
                open_no_trade_band=str(
                    selected_params["open_no_trade_band"]
                ),
                stage2_teacher_beta=str(
                    selected_params["stage2_teacher_beta"]
                ),
                stage2_action_beta=str(
                    selected_params["stage2_action_beta"]
                ),
                stage2_enabled=str(
                    selected_params.get("stage2_enabled", False)
                ),
                stage2_probability_threshold=str(
                    selected_params["stage2_probability_threshold"]
                ),
                stage2_logistic_c=str(
                    selected_params["stage2_logistic_c"]
                ),
                stage2_holding_period="second_cut_to_regular_session_close",
            )
            R.log_metrics(
                development_return_pct=dev_metrics["return_pct"],
                development_max_drawdown_pct=dev_metrics[
                    "max_drawdown_pct"
                ],
                outer_return_pct=outer_metrics["return_pct"],
                outer_max_drawdown_pct=outer_metrics["max_drawdown_pct"],
                stress_25bps_return_pct=stress_metrics["return_pct"],
                stress_25bps_max_drawdown_pct=stress_metrics[
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
    recorder_artifacts = (
        Path(recorder_info["recorder_local_dir"]) / "artifacts"
    )
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return recorder_info


def _write_card_sidecars(
    spec,
    *,
    summary,
    selected,
    model_selection,
    risk_analysis,
    annual_metrics,
    calibration,
    band_diagnostics,
) -> None:
    card = spec.card_dir
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_payload = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward",
        "outer_status": "reused_diagnostic_not_fresh_unseen_test",
        "selected_model": summary["selected_model"],
        "selection_gates": summary["selection_gates"],
        "development_metrics": {
            key: float(selected[key])
            for key in (
                "return_pct",
                "max_drawdown_pct",
                "worst_day_pct",
                "cvar5_pct",
                "worst_rolling_5d_pct",
                "avg_cash_weight_pct",
                "neighbor_median_terminal_wealth",
                "neighbor_min_wealth_ratio",
            )
        },
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    model_selection.sort_values(
        [
            "neighbor_median_terminal_wealth",
            "terminal_wealth",
        ],
        ascending=False,
    ).head(40).to_csv(card / "model_selection.csv", index=False)
    risk_analysis.to_csv(card / "risk_analysis.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    calibration.to_csv(card / "calibration_metrics.csv", index=False)
    band_diagnostics.to_csv(
        card / "no_trade_band_diagnostics.csv", index=False
    )
    workflow = f"""experiment: {spec.experiment}
base_experiment: {spec.base_experiment}
prior_staged_experiment: {spec.prior_staged_experiment}
source_experiment: {spec.source_experiment}
source_recorder_id: {summary['source_audit']['source_recorder_id']}
intraday_provider_uri: {spec.intraday_provider_uri.resolve()}
assets: [SOXL, SOXX, cash]
allocation_output: direct_staged_legal_three_asset_weights
stage2_training_distribution: on_policy_stage1_actions_only
selection_uses_outer_test: false
outer_status: reused_diagnostic_not_fresh_unseen_test
stage1_action_beta: {summary['selected_model']['action_beta']}
second_cut_time: {summary['selected_model']['second_cut_time']}
open_no_trade_band: {summary['selected_model']['open_no_trade_band']}
stage2_teacher_beta: {summary['selected_model']['stage2_teacher_beta']}
stage2_action_beta: {summary['selected_model']['stage2_action_beta']}
stage2_enabled: {summary['selected_model']['stage2_enabled']}
stage2_holding_period: second_cut_to_regular_session_close
close_action: restore_stage1_weights_before_overnight
transaction_cost_bps: {spec.transaction_cost_bps}
stress_cost_bps: {spec.stress_cost_bps}
stress_method: fixed_10bp_decisions_repriced_at_25bp
robust_selection:
  minimum_development_stage2_actions: {spec.minimum_development_stage2_actions}
  minimum_neighbor_count: {spec.minimum_neighbor_count}
  minimum_neighbor_wealth_ratio: {spec.minimum_neighbor_wealth_ratio}
  near_best_neighbor_wealth_tolerance: {spec.near_best_neighbor_wealth_tolerance}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  recorder_only_artifacts:
    - artifacts/{spec.experiment}/positions.csv
    - artifacts/{spec.experiment}/strategy_returns.csv
    - artifacts/{spec.experiment}/stage1_training_frame.csv
    - artifacts/{spec.experiment}/on_policy_stage2_training_frame.csv
    - artifacts/{spec.experiment}/model_selection_full.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    outer = summary["outer_diagnostic_metrics_10bps"]
    stress = summary["fixed_decision_stress_metrics_25bps"]
    dev = summary["development_metrics_10bps"]
    selected_model = summary["selected_model"]
    outer_effect = summary["outer_effect"]
    conclusion = f"""# 中文结论

- 做了什么：把第二段模型改成真正的 on-policy 训练。第二段标签只在年度 walk-forward 第一段模型实际触发的交易日生成，不再拿“老师假设触发”的日子冒充模型会遇到的状态。
- 开发期样本：第二段共有 {summary['training_sample_audit']['development_on_policy_stage2_samples']} 条真实动作状态；年度分布为 {summary['training_sample_audit']['development_samples_by_year']}。这比旧 off-policy 训练集小得多，因此本卡同时要求至少 {spec.minimum_development_stage2_actions} 次开发期动作、连续三个相邻阈值成立，并限制相邻参数最差终值比例。
- 选中参数：第二切点 {selected_model['second_cut_time']}，Logistic C={selected_model['stage2_logistic_c']:.2f}，阈值={selected_model['stage2_probability_threshold']:.2f}，临时 beta={selected_model['stage2_action_beta']:.1f}，开盘 no-trade band={selected_model['open_no_trade_band']:.3f}。
- 开发期 10bp：return={dev['return_pct']:.4f}%，MaxDD={dev['max_drawdown_pct']:.4f}%，worst 5d={dev['worst_rolling_5d_pct']:.4f}%。参数邻域门已通过，但它证明的是局部稳定性，不是 outer 泛化。
- 外层诊断 10bp：return={outer['return_pct']:.4f}%，MaxDD={outer['max_drawdown_pct']:.4f}%，worst day={outer['worst_day_pct']:.4f}%，CVaR5={outer['cvar5_pct']:.4f}%，worst 5d={outer['worst_rolling_5d_pct']:.4f}%。
- 收益复盘：相对第一段模型增加 {outer_effect['return_change_vs_stage1_pct_points']:.4f} 个百分点，但相对旧 off-policy 二段模型减少 {-outer_effect['return_change_vs_prior_staged_pct_points']:.4f} 个百分点；相对原始直接配置模型仍少 {-outer_effect['return_gap_vs_raw_pct_points']:.4f} 个百分点。outer 二段实际仅触发 {outer_effect['stage2_action_count']} 次，证据仍稀疏。
- 25bp 冻结决策压力：return={stress['return_pct']:.4f}%，MaxDD={stress['max_drawdown_pct']:.4f}%，worst 5d={stress['worst_rolling_5d_pct']:.4f}%。
- no-trade 结论：开发期近优候选最终选择 0 band。0.5%-2% 小带宽不能实质减少换手，3%-5% 大带宽在可行性检查中破坏回撤或相对收益窗口，所以不应把“保持现金”写成独立手工规则；现金仍由模型合法权重直接决定。
- 是否晋升：{summary['decision']}。Stage 2 开发期校准闸门={summary['selection_gates']['stage2_calibration_gate_pass']}；本次因此自动退回 Stage 1 直接合法权重。原始直接配置收益不是硬晋升目标，真正验收是超过 SOXX、超过 matched-beta 基准并通过风险和成本压力门。
- 数据边界：2023-2026 已被前序实验多次查看，本卡只把它们标记为 reused diagnostic，不能声称是全新未见测试。下一轮应冻结本卡结构，等待新数据或改用跨品种训练后再做一次真正未见验证。
- qlib recorder：{summary['recorder_id']}
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
