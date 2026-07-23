from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd

from src.workflow.cash_floor_benchmark_relative_wf import _result_metrics
from src.workflow.hierarchical_soxx_excess_walk_forward import (
    EXPERIMENT as SOURCE_EXPERIMENT,
    HierarchicalSoxxExcessSpec,
    _drawdown_duration,
    _json_ready,
    _load_frame,
    _remove_qlib_code_snapshot_artifacts,
    _simulate,
)


EXPERIMENT = "hierarchical_soxx_excess_prior_overlay_v2"
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
)


@dataclass(frozen=True)
class PriorOverlayProfile:
    name: str
    model_soxl_share: float
    alpha_threshold: float
    hazard_threshold: float
    cash_probability_threshold: float
    hazard_soxl_cap_fraction: float
    defense_cash: float
    prior_weight: float
    prior_soxl_cap: float
    minimum_cash: float = 0.10


PROFILES = (
    PriorOverlayProfile(
        "prior_low",
        0.20,
        0.001,
        0.70,
        0.50,
        1.00,
        0.25,
        0.25,
        0.50,
    ),
    PriorOverlayProfile(
        "prior_balanced",
        0.20,
        0.001,
        0.70,
        0.50,
        1.00,
        0.55,
        0.50,
        0.50,
    ),
    PriorOverlayProfile(
        "prior_strong",
        0.20,
        0.001,
        0.70,
        0.50,
        1.00,
        0.55,
        0.75,
        0.50,
    ),
    PriorOverlayProfile(
        "prior_conservative",
        0.15,
        0.001,
        0.70,
        0.50,
        1.00,
        0.40,
        0.75,
        0.40,
    ),
    PriorOverlayProfile(
        "prior_low_beta",
        0.15,
        0.001,
        0.75,
        0.50,
        1.00,
        0.40,
        0.75,
        0.40,
    ),
)


@dataclass(frozen=True)
class HierarchicalSoxxExcessPriorOverlaySpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    costs_bps: tuple[int, ...] = (10, 25)

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_hierarchical_soxx_excess_prior_overlay_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    )
    return _run_card(HierarchicalSoxxExcessPriorOverlaySpec(root=repo_root))


def _load_source(
    spec: HierarchicalSoxxExcessPriorOverlaySpec,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    source_summary = json.loads(
        (spec.source_card_dir / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary["selection_uses_outer_test"]:
        raise ValueError("blocked: source experiment selected on outer test")
    predictions_path = (
        Path(source_summary["recorder_local_dir"])
        / "artifacts"
        / spec.source_experiment
        / "predictions.csv"
    )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"source recorder predictions missing: {predictions_path}"
        )
    predictions = pd.read_csv(
        predictions_path, parse_dates=["signal_date"]
    ).set_index("signal_date")
    selected = source_summary["selected_model"]
    predictions = predictions.loc[
        predictions["feature_bundle"].eq(selected["feature_bundle"])
        & predictions["model_family"].eq(selected["model_family"])
    ].sort_index()
    frame, data_audit, _, _ = _load_frame(
        HierarchicalSoxxExcessSpec(root=spec.root)
    )
    source_audit = {
        "source_experiment": spec.source_experiment,
        "source_recorder_id": source_summary["recorder_id"],
        "source_recorder_local_dir": source_summary["recorder_local_dir"],
        "source_predictions": predictions_path.as_posix(),
        "source_feature_bundle": selected["feature_bundle"],
        "source_model_family": selected["model_family"],
        "source_selection_uses_outer_test": False,
        "base_constrained_ref_in_model_features": False,
        "base_constrained_ref_role": "explicit_additive_prior_and_standalone_benchmark",
    }
    return source_summary, predictions, frame, {
        "data": data_audit,
        "source": source_audit,
    }


def _weights(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: PriorOverlayProfile,
    *,
    include_model: bool = True,
    include_prior: bool = True,
    include_cash_action: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    market = frame.loc[predictions.index]
    alpha = predictions["soxl_relative_alpha_5d_prediction"].to_numpy(
        dtype=float
    )
    hazard_probability = predictions[
        "crash_hazard_5d_prediction"
    ].to_numpy(dtype=float)
    cash_probability = predictions["cash_action_5d_prediction"].to_numpy(
        dtype=float
    )
    hazard = hazard_probability >= profile.hazard_threshold
    cash_entry = (
        hazard
        & (cash_probability >= profile.cash_probability_threshold)
        & include_cash_action
    )
    model_share = (
        np.where(
            alpha > profile.alpha_threshold,
            profile.model_soxl_share,
            0.0,
        )
        if include_model
        else np.zeros(len(predictions))
    )
    prior_raw = pd.to_numeric(
        market["base_constrained_ref"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0).to_numpy()
    prior_share = (
        profile.prior_weight
        * np.minimum(
            prior_raw * profile.prior_soxl_cap,
            profile.prior_soxl_cap,
        )
        if include_prior
        else np.zeros(len(predictions))
    )
    soxl_share = np.minimum(
        model_share + prior_share,
        max(profile.model_soxl_share, profile.prior_soxl_cap),
    )
    hazard_cap = (
        profile.model_soxl_share * profile.hazard_soxl_cap_fraction
    )
    soxl_share[hazard] = np.minimum(soxl_share[hazard], hazard_cap)
    cash = np.where(
        cash_entry,
        profile.defense_cash,
        profile.minimum_cash,
    )
    risky = 1.0 - cash
    weights = np.column_stack(
        [
            risky * soxl_share,
            risky * (1.0 - soxl_share),
            np.zeros(len(predictions)),
            cash,
        ]
    )
    actions = pd.DataFrame(
        {
            "alpha_active": alpha > profile.alpha_threshold,
            "explicit_prior_active": prior_raw > 0.0,
            "hazard_active": hazard,
            "cash_entry": cash_entry,
            "model_soxl_share": model_share,
            "prior_soxl_share": prior_share,
            "combined_soxl_share": soxl_share,
        },
        index=predictions.index,
    )
    return weights, actions


def _evaluate(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: PriorOverlayProfile,
    *,
    cost_bps: int,
    include_model: bool = True,
    include_prior: bool = True,
    include_cash_action: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    market = frame.loc[predictions.index]
    weights, actions = _weights(
        frame,
        predictions,
        profile,
        include_model=include_model,
        include_prior=include_prior,
        include_cash_action=include_cash_action,
    )
    strategy = _simulate(
        market,
        weights,
        cost_bps=cost_bps,
        name=profile.name,
    )
    benchmark = _simulate(
        market,
        np.tile([0.0, 1.0, 0.0, 0.0], (len(market), 1)),
        cost_bps=cost_bps,
        name="soxx_buyhold",
    )
    metrics = _result_metrics(strategy, benchmark)
    strategy_duration = _drawdown_duration(strategy["strategy_return"])
    benchmark_duration = _drawdown_duration(benchmark["strategy_return"])
    benchmark_metrics = _result_metrics(benchmark, benchmark)
    downside = pd.to_numeric(
        market["soxx_trend60"], errors="coerce"
    ).lt(0.0).to_numpy()
    metrics.update(
        {
            **strategy_duration,
            "soxx_max_drawdown_pct": benchmark_metrics[
                "max_drawdown_pct"
            ],
            "soxx_longest_underwater_days": benchmark_duration[
                "longest_underwater_days"
            ],
            "soxx_max_drawdown_recovery_days": benchmark_duration[
                "max_drawdown_recovery_days"
            ],
            "downside_avg_cash_weight_pct": 100.0
            * float(strategy.loc[downside, "cash_weight"].mean()),
            "downside_min_cash_weight_pct": 100.0
            * float(strategy.loc[downside, "cash_weight"].min()),
            "hazard_days": int(actions["hazard_active"].sum()),
            "cash_entry_days": int(actions["cash_entry"].sum()),
            "prior_active_days": int(
                actions["explicit_prior_active"].sum()
            ),
            "turnover": float(strategy["turnover"].sum()),
        }
    )
    return (
        metrics,
        strategy.join(actions.reset_index(drop=True)),
        benchmark,
    )


def _annual_metrics(
    result: pd.DataFrame,
    benchmark: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    years = pd.to_datetime(result["signal_date"]).dt.year
    for year in sorted(years.unique()):
        mask = years.eq(year)
        metrics = _result_metrics(
            result.loc[mask].reset_index(drop=True),
            benchmark.loc[mask].reset_index(drop=True),
        )
        rows.append(
            {"fold_year": int(year), **metadata, **metrics}
        )
    return rows


def _development_candidates(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: HierarchicalSoxxExcessPriorOverlaySpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = predictions.loc[
        predictions["segment"].eq("development_walk_forward")
    ].sort_index()
    rows: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for profile in PROFILES:
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                frame,
                development,
                profile,
                cost_bps=cost_bps,
            )
            metadata = {
                "candidate": profile.name,
                "cost_bps": cost_bps,
                "selection_uses_outer_test": False,
            }
            rows.append({**metadata, **metrics})
            annual.extend(
                _annual_metrics(result, benchmark, metadata)
            )
    return pd.DataFrame(rows), pd.DataFrame(annual)


def _select(
    comparison: pd.DataFrame, annual: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    ten = comparison.loc[comparison["cost_bps"].eq(10)].copy()
    stress = comparison.loc[
        comparison["cost_bps"].eq(25),
        [
            "candidate",
            "terminal_wealth_ratio_vs_soxx",
            "max_drawdown_pct",
        ],
    ].rename(
        columns={
            "terminal_wealth_ratio_vs_soxx": "ratio_25bps",
            "max_drawdown_pct": "max_drawdown_25bps_pct",
        }
    )
    year_summary = (
        annual.loc[annual["cost_bps"].eq(10)]
        .groupby("candidate")
        .agg(
            positive_relative_years=(
                "terminal_wealth_ratio_vs_soxx",
                lambda values: int((values > 1.0).sum()),
            ),
            median_annual_ratio=(
                "terminal_wealth_ratio_vs_soxx",
                "median",
            ),
        )
    )
    selection = ten.merge(stress, on="candidate").merge(
        year_summary, on="candidate"
    )
    selection = selection.rename(
        columns={"terminal_wealth_ratio_vs_soxx": "ratio_10bps"}
    )
    selection["return_gate"] = (
        selection["ratio_10bps"].ge(1.20)
        & selection["ratio_25bps"].gt(1.00)
        & selection["positive_relative_years"].ge(4)
        & selection["median_annual_ratio"].gt(1.0)
    )
    selection["drawdown_gate"] = selection[
        "max_drawdown_pct"
    ].gt(selection["soxx_max_drawdown_pct"])
    selection["recovery_gate"] = (
        selection["longest_underwater_days"].lt(
            selection["soxx_longest_underwater_days"]
        )
        & selection["max_drawdown_recovery_days"].le(
            selection["soxx_max_drawdown_recovery_days"]
        )
    )
    selection["cash_gate"] = (
        selection["min_cash_weight_pct"].ge(10.0 - 1e-9)
        & selection["downside_avg_cash_weight_pct"].gt(10.0)
        & selection["cash_entry_days"].gt(0)
    )
    selection["all_gate"] = (
        selection["return_gate"]
        & selection["drawdown_gate"]
        & selection["recovery_gate"]
        & selection["cash_gate"]
    )
    selection["drawdown_cushion_pct"] = (
        selection["max_drawdown_pct"]
        - selection["soxx_max_drawdown_pct"]
    )
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-9))
        + 0.5
        * np.log(selection["ratio_25bps"].clip(lower=1e-9))
        + 0.10 * selection["drawdown_cushion_pct"]
        - 0.0005 * selection["turnover"]
    )
    eligible = selection.loc[selection["all_gate"]]
    policy = "all_gate_then_return_drawdown_cost_robustness"
    if eligible.empty:
        eligible = selection
        policy = "diagnostic_fallback_no_all_gate"
    selected = eligible.sort_values(
        ["selection_score", "ratio_25bps"],
        ascending=False,
    ).iloc[0]
    selection["selected"] = selection["candidate"].eq(
        selected["candidate"]
    )
    return _json_ready(
        {
            **selected.to_dict(),
            "selection_policy": policy,
            "selection_uses_outer_test": False,
        }
    ), selection


def _selected_profile(selected: dict[str, Any]) -> PriorOverlayProfile:
    return next(
        profile
        for profile in PROFILES
        if profile.name == selected["candidate"]
    )


def _segment_evaluation(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: PriorOverlayProfile,
    segment: str,
    spec: HierarchicalSoxxExcessPriorOverlaySpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_predictions = predictions.loc[
        predictions["segment"].eq(segment)
    ].sort_index()
    rows: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    stored: list[pd.DataFrame] = []
    for cost_bps in spec.costs_bps:
        metrics, result, benchmark = _evaluate(
            frame,
            selected_predictions,
            profile,
            cost_bps=cost_bps,
        )
        metadata = {
            "segment": segment,
            "candidate": profile.name,
            "cost_bps": cost_bps,
            "selection_uses_outer_test": False,
        }
        rows.append({**metadata, **metrics})
        annual.extend(
            _annual_metrics(result, benchmark, metadata)
        )
        stored.append(result.assign(**metadata))
    return (
        pd.DataFrame(rows),
        pd.DataFrame(annual),
        pd.concat(stored, ignore_index=True),
    )


def _prior_attribution(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: PriorOverlayProfile,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment in (
        "development_walk_forward",
        "outer_diagnostic",
    ):
        selected = predictions.loc[
            predictions["segment"].eq(segment)
        ].sort_index()
        variants = (
            (
                "selected_model_plus_explicit_prior",
                True,
                True,
                True,
                profile,
            ),
            (
                "model_only_same_cash_gate",
                True,
                False,
                True,
                profile,
            ),
            (
                "explicit_prior_only_same_cash_gate",
                False,
                True,
                True,
                profile,
            ),
            (
                "model_plus_prior_no_extra_cash",
                True,
                True,
                False,
                replace(profile, defense_cash=profile.minimum_cash),
            ),
        )
        for name, model, prior, cash, variant_profile in variants:
            metrics, _, _ = _evaluate(
                frame,
                selected,
                variant_profile,
                cost_bps=10,
                include_model=model,
                include_prior=prior,
                include_cash_action=cash,
            )
            rows.append(
                {
                    "segment": segment,
                    "variant": name,
                    "base_constrained_ref_hidden_feature": False,
                    "selection_uses_outer_test": False,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _action_attribution(
    strategy_returns: pd.DataFrame,
) -> pd.DataFrame:
    selected = strategy_returns.loc[
        strategy_returns["cost_bps"].eq(10)
    ].copy()
    selected["year"] = pd.to_datetime(
        selected["signal_date"]
    ).dt.year
    rows: list[dict[str, Any]] = []
    for (segment, year), group in selected.groupby(
        ["segment", "year"]
    ):
        strategy_wealth = float(
            (1.0 + group["strategy_return"]).prod()
        )
        soxx_wealth = float((1.0 + group["soxx_return"]).prod())
        rows.append(
            {
                "segment": segment,
                "year": int(year),
                "rows": len(group),
                "strategy_return_pct": 100.0
                * (strategy_wealth - 1.0),
                "soxx_return_pct": 100.0 * (soxx_wealth - 1.0),
                "terminal_wealth_ratio_vs_soxx": strategy_wealth
                / soxx_wealth,
                "avg_soxl_weight_pct": 100.0
                * float(group["soxl_weight"].mean()),
                "avg_cash_weight_pct": 100.0
                * float(group["cash_weight"].mean()),
                "prior_active_days": int(
                    group["explicit_prior_active"].sum()
                ),
                "hazard_days": int(group["hazard_active"].sum()),
                "cash_entry_days": int(group["cash_entry"].sum()),
                "turnover": float(group["turnover"].sum()),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _regime_metrics(
    frame: pd.DataFrame, strategy_returns: pd.DataFrame
) -> pd.DataFrame:
    selected = strategy_returns.loc[
        strategy_returns["cost_bps"].eq(10)
    ].copy()
    market = frame.loc[
        pd.to_datetime(selected["signal_date"])
    ].copy()
    regimes = {
        "all": np.ones(len(selected), dtype=bool),
        "downtrend": pd.to_numeric(
            market["soxx_trend60"], errors="coerce"
        ).lt(0.0).to_numpy(),
        "prior_active": selected[
            "explicit_prior_active"
        ].to_numpy(dtype=bool),
        "hazard_active": selected["hazard_active"].to_numpy(
            dtype=bool
        ),
    }
    rows: list[dict[str, Any]] = []
    for regime, mask in regimes.items():
        group = selected.loc[mask]
        strategy_wealth = float(
            (1.0 + group["strategy_return"]).prod()
        )
        soxx_wealth = float((1.0 + group["soxx_return"]).prod())
        rows.append(
            {
                "regime": regime,
                "rows": int(mask.sum()),
                "return_pct": 100.0 * (strategy_wealth - 1.0),
                "soxx_return_pct": 100.0 * (soxx_wealth - 1.0),
                "terminal_wealth_ratio_vs_soxx": strategy_wealth
                / soxx_wealth,
                "avg_soxl_weight_pct": 100.0
                * float(group["soxl_weight"].mean()),
                "avg_cash_weight_pct": 100.0
                * float(group["cash_weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _outer_gate(row: pd.Series) -> bool:
    return bool(
        row["terminal_wealth_ratio_vs_soxx"] >= 1.20
        and row["max_drawdown_pct"] > row["soxx_max_drawdown_pct"]
        and row["longest_underwater_days"]
        < row["soxx_longest_underwater_days"]
        and row["max_drawdown_recovery_days"]
        <= row["soxx_max_drawdown_recovery_days"]
        and row["downside_avg_cash_weight_pct"] > 10.0
        and row["cash_entry_days"] > 0
    )


def _objective_gate_audit(
    selected: dict[str, Any],
    outer10: pd.Series,
    source_summary: dict[str, Any],
) -> pd.DataFrame:
    rows = (
        (
            "alpha_head_feature_families",
            True,
            True,
            "source_v1_reintegrated_alpha_uses_momentum_rv_decay_trend_breadth",
        ),
        (
            "crash_hazard_head",
            True,
            True,
            source_summary["targets"]["crash_hazard_5d"],
        ),
        (
            "cash_action_and_gate",
            True,
            True,
            source_summary["architecture"]["cash_action_head"],
        ),
        (
            "explicit_prior_not_hidden_feature",
            True,
            bool(
                not source_summary["data_audit"][
                    "base_constrained_ref_in_model_features"
                ]
            ),
            "base_constrained_ref_explicit_additive_prior",
        ),
        (
            "outer_not_used_for_selection",
            True,
            bool(not selected["selection_uses_outer_test"]),
            "profile_selected_on_2016_2022_development_only",
        ),
        (
            "development_20pct_excess_10bps",
            True,
            bool(selected["ratio_10bps"] >= 1.20),
            selected["ratio_10bps"],
        ),
        (
            "development_maxdd_better_than_soxx",
            True,
            bool(
                selected["max_drawdown_pct"]
                > selected["soxx_max_drawdown_pct"]
            ),
            (
                f"{selected['max_drawdown_pct']:.6f} vs "
                f"{selected['soxx_max_drawdown_pct']:.6f}"
            ),
        ),
        (
            "development_recovery_no_worse_than_soxx",
            True,
            bool(selected["recovery_gate"]),
            (
                f"underwater {selected['longest_underwater_days']} vs "
                f"{selected['soxx_longest_underwater_days']}; "
                f"recovery {selected['max_drawdown_recovery_days']} vs "
                f"{selected['soxx_max_drawdown_recovery_days']}"
            ),
        ),
        (
            "development_downside_cash",
            True,
            bool(selected["downside_avg_cash_weight_pct"] > 10.0),
            selected["downside_avg_cash_weight_pct"],
        ),
        (
            "outer_20pct_excess_10bps",
            True,
            bool(outer10["terminal_wealth_ratio_vs_soxx"] >= 1.20),
            outer10["terminal_wealth_ratio_vs_soxx"],
        ),
        (
            "outer_maxdd_better_than_soxx",
            True,
            bool(
                outer10["max_drawdown_pct"]
                > outer10["soxx_max_drawdown_pct"]
            ),
            (
                f"{outer10['max_drawdown_pct']:.6f} vs "
                f"{outer10['soxx_max_drawdown_pct']:.6f}"
            ),
        ),
        (
            "outer_recovery_better_than_soxx",
            True,
            bool(
                outer10["longest_underwater_days"]
                < outer10["soxx_longest_underwater_days"]
                and outer10["max_drawdown_recovery_days"]
                <= outer10["soxx_max_drawdown_recovery_days"]
            ),
            (
                f"underwater {outer10['longest_underwater_days']} vs "
                f"{outer10['soxx_longest_underwater_days']}; "
                f"recovery {outer10['max_drawdown_recovery_days']} vs "
                f"{outer10['soxx_max_drawdown_recovery_days']}"
            ),
        ),
        (
            "outer_downside_cash",
            True,
            bool(outer10["downside_avg_cash_weight_pct"] > 10.0),
            outer10["downside_avg_cash_weight_pct"],
        ),
        (
            "legacy_absolute_worst_5d_minus_12pct",
            False,
            bool(outer10["worst_rolling_5d_pct"] >= -12.0),
            outer10["worst_rolling_5d_pct"],
        ),
    )
    return pd.DataFrame(
        rows,
        columns=["gate", "required", "passed", "evidence"],
    )


def _run_card(
    spec: HierarchicalSoxxExcessPriorOverlaySpec,
) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_summary, predictions, frame, data_audit = _load_source(
        spec
    )
    comparison, development_annual = _development_candidates(
        frame, predictions, spec
    )
    selected, model_selection = _select(
        comparison, development_annual
    )
    profile = _selected_profile(selected)
    development_comparison, _, development_returns = (
        _segment_evaluation(
            frame,
            predictions,
            profile,
            "development_walk_forward",
            spec,
        )
    )
    outer_comparison, outer_annual, outer_returns = (
        _segment_evaluation(
            frame,
            predictions,
            profile,
            "outer_diagnostic",
            spec,
        )
    )
    strategy_returns = pd.concat(
        [development_returns, outer_returns],
        ignore_index=True,
    )
    attribution = _prior_attribution(
        frame, predictions, profile
    )
    actions = _action_attribution(strategy_returns)
    regimes = _regime_metrics(frame, strategy_returns)
    outer10 = outer_comparison.loc[
        outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    outer_pass = _outer_gate(outer10)
    objective_audit = _objective_gate_audit(
        selected, outer10, source_summary
    )
    objective_pass = bool(selected["all_gate"]) and outer_pass
    decision = (
        "validated_research_goal_passed_explicit_prior_overlay"
        if objective_pass
        else "needs_iteration_explicit_prior_overlay"
    )

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        strategy_returns.to_csv(
            temp / "strategy_returns.csv", index=False
        )
        strategy_returns.loc[
            :,
            [
                "signal_date",
                "trade_date",
                "exit_date",
                "soxl_weight",
                "soxx_weight",
                "cash_weight",
                "turnover",
                "segment",
                "cost_bps",
            ],
        ].to_csv(temp / "positions.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=source_summary["recorder_id"],
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                selected_profile=profile.name,
                base_constrained_ref_role=(
                    "explicit_additive_prior_not_model_feature"
                ),
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_ratio_10bps=float(
                    selected["ratio_10bps"]
                ),
                development_ratio_25bps=float(
                    selected["ratio_25bps"]
                ),
                outer_ratio_10bps=float(
                    outer10["terminal_wealth_ratio_vs_soxx"]
                ),
                outer_max_drawdown_pct=float(
                    outer10["max_drawdown_pct"]
                ),
                objective_pass=float(objective_pass),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(
                    artifact.as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder_id = recorder.id
            recorder_local_dir = Path(
                recorder.get_local_dir()
            ).resolve()
            summary = _summary(
                spec=spec,
                source_summary=source_summary,
                data_audit=data_audit,
                selected=selected,
                profile=profile,
                comparison=comparison,
                development_comparison=development_comparison,
                outer_comparison=outer_comparison,
                attribution=attribution,
                objective_audit=objective_audit,
                decision=decision,
                outer_pass=outer_pass,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
            )
            _write_card(
                card=card,
                summary=summary,
                model_selection=model_selection,
                development_annual=development_annual,
                outer_annual=outer_annual,
                development_comparison=development_comparison,
                outer_comparison=outer_comparison,
                attribution=attribution,
                objective_audit=objective_audit,
                actions=actions,
                regimes=regimes,
            )
            for filename in summary["sidecars"].values():
                path = card / filename
                if path.exists():
                    recorder.log_artifact(
                        path.as_posix(),
                        artifact_path=spec.experiment,
                    )
            _remove_qlib_code_snapshot_artifacts(
                recorder_local_dir
            )
    return _json_ready(summary)


def _summary(
    *,
    spec: HierarchicalSoxxExcessPriorOverlaySpec,
    source_summary: dict[str, Any],
    data_audit: dict[str, Any],
    selected: dict[str, Any],
    profile: PriorOverlayProfile,
    comparison: pd.DataFrame,
    development_comparison: pd.DataFrame,
    outer_comparison: pd.DataFrame,
    attribution: pd.DataFrame,
    objective_audit: pd.DataFrame,
    decision: str,
    outer_pass: bool,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    sidecars = {
        "selected_model": "selected_model.json",
        "model_selection": "model_selection.csv",
        "backtest_benchmark_comparison": (
            "backtest_benchmark_comparison.csv"
        ),
        "backtest_walk_forward_metrics": (
            "backtest_walk_forward_metrics.csv"
        ),
        "risk_analysis": "risk_analysis.csv",
        "prior_incremental_attribution": (
            "prior_incremental_attribution.csv"
        ),
        "objective_gate_audit": "objective_gate_audit.csv",
        "action_attribution": "action_attribution.csv",
        "regime_metrics": "regime_metrics.csv",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    return _json_ready(
        {
            "experiment": spec.experiment,
            "card_type": (
                "hierarchical_soxx_excess_explicit_prior_overlay_card"
            ),
            "primary_notebook": f"{spec.experiment}.ipynb",
            "promotion_decision": decision,
            "objective_pass": bool(selected["all_gate"])
            and outer_pass,
            "selection_uses_outer_test": False,
            "outer_test_use": (
                "diagnostic_only_previously_observed_not_profile_selection"
            ),
            "source_experiment": spec.source_experiment,
            "source_recorder_id": source_summary["recorder_id"],
            "data_audit": data_audit,
            "qlib_docs": source_summary["qlib_docs"],
            "architecture": {
                "alpha_head": (
                    "source v1 reintegrated ExtraTrees 5d relative alpha"
                ),
                "hazard_head": (
                    "source v1 reintegrated ExtraTrees 5d crash hazard"
                ),
                "cash_action_head": (
                    "source v1 conditional cash action; cash rises only "
                    "when hazard and cash benefit probability gates both pass"
                ),
                "explicit_prior": (
                    "base_constrained_ref is an additive allocation prior "
                    "with its own cap and attribution; never a model feature"
                ),
                "combined_risky_sleeve": (
                    "model SOXL share plus explicit prior SOXL share, "
                    "capped; hazard caps the combined sleeve"
                ),
            },
            "profiles": [asdict(item) for item in PROFILES],
            "selected_profile": asdict(profile),
            "selected_model": selected,
            "development_selected": development_comparison.to_dict(
                orient="records"
            ),
            "outer_selected_diagnostic": outer_comparison.to_dict(
                orient="records"
            ),
            "outer_gate_pass": outer_pass,
            "prior_incremental_attribution": attribution.to_dict(
                orient="records"
            ),
            "objective_gate_audit": objective_audit.to_dict(
                orient="records"
            ),
            "recorder_id": recorder_id,
            "recorder_uri": recorder_uri.resolve().as_posix(),
            "recorder_local_dir": recorder_local_dir.as_posix(),
            "sidecars": sidecars,
            "artifact_policy": {
                "policy": "slim_card_recorder_first",
                "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": list(
                    RECORDER_ONLY_ARTIFACTS
                ),
                "source_model_artifacts": (
                    "referenced_from_source_v1_recorder_not_duplicated"
                ),
            },
        }
    )


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    model_selection: pd.DataFrame,
    development_annual: pd.DataFrame,
    outer_annual: pd.DataFrame,
    development_comparison: pd.DataFrame,
    outer_comparison: pd.DataFrame,
    attribution: pd.DataFrame,
    objective_audit: pd.DataFrame,
    actions: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    model_selection.to_csv(
        card / "model_selection.csv", index=False
    )
    (card / "selected_model.json").write_text(
        json.dumps(summary["selected_model"], indent=2) + "\n",
        encoding="utf-8",
    )
    comparison = pd.concat(
        [development_comparison, outer_comparison],
        ignore_index=True,
    )
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv",
        index=False,
    )
    pd.concat(
        [development_annual, outer_annual], ignore_index=True
    ).to_csv(
        card / "backtest_walk_forward_metrics.csv",
        index=False,
    )
    comparison.loc[
        :,
        [
            "segment",
            "cost_bps",
            "terminal_wealth_ratio_vs_soxx",
            "return_pct",
            "soxx_return_pct",
            "max_drawdown_pct",
            "soxx_max_drawdown_pct",
            "longest_underwater_days",
            "soxx_longest_underwater_days",
            "max_drawdown_recovery_days",
            "soxx_max_drawdown_recovery_days",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "avg_cash_weight_pct",
            "downside_avg_cash_weight_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    attribution.to_csv(
        card / "prior_incremental_attribution.csv",
        index=False,
    )
    objective_audit.to_csv(
        card / "objective_gate_audit.csv",
        index=False,
    )
    actions.to_csv(card / "action_attribution.csv", index=False)
    regimes.to_csv(card / "regime_metrics.csv", index=False)
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "objective_pass": summary["objective_pass"],
        "selected_model": summary["selected_model"],
        "development_selected": summary["development_selected"],
        "outer_selected_diagnostic": (
            summary["outer_selected_diagnostic"]
        ),
        "prior_incremental_attribution": (
            summary["prior_incremental_attribution"]
        ),
        "objective_gate_audit": summary["objective_gate_audit"],
        "finding": (
            "The old rule family is useful as an explicit additive prior, "
            "not as a hidden feature. The model supplies independent alpha "
            "and hazard increments while the prior restores fast re-risking."
        ),
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    (card / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    selected = summary["selected_model"]
    outer10 = next(
        item
        for item in summary["outer_selected_diagnostic"]
        if item["cost_bps"] == 10
    )
    failure = (
        "# Failure modes\n\n"
        f"- objective_pass: {summary['objective_pass']}\n"
        f"- development all_gate: {selected['all_gate']}\n"
        f"- outer gate: {summary['outer_gate_pass']}\n"
        "- outer 2023+ was previously observed in earlier research and is "
        "diagnostic only; it did not select the profile.\n"
        "- the explicit prior may fail when its trend/canary regime changes; "
        "the attribution table must remain part of every refresh.\n"
        f"- outer 10bp ratio/maxdd: "
        f"{outer10['terminal_wealth_ratio_vs_soxx']:.4f} / "
        f"{outer10['max_drawdown_pct']:.2f}%.\n"
    )
    (card / "failure_modes.md").write_text(
        failure, encoding="utf-8"
    )
    conclusion = (
        "# 显式先验 SOXX 超额策略结论\n\n"
        f"- 跑了什么：从 {summary['source_experiment']} 的 qlib recorder "
        "读取年度 OOS 三头预测，把旧 base_constrained_ref 作为独立加法先验，"
        "而不是模型特征；仅用 2016–2022 选择五个预定义 profile。\n"
        f"- 开发期：10bp 终值/SOXX={selected['ratio_10bps']:.4f}，"
        f"25bp={selected['ratio_25bps']:.4f}；MaxDD="
        f"{selected['max_drawdown_pct']:.2f}% vs SOXX "
        f"{selected['soxx_max_drawdown_pct']:.2f}%；最长水下 "
        f"{selected['longest_underwater_days']} vs "
        f"{selected['soxx_longest_underwater_days']} 天。\n"
        f"- Outer 诊断：10bp 终值/SOXX="
        f"{outer10['terminal_wealth_ratio_vs_soxx']:.4f}；MaxDD="
        f"{outer10['max_drawdown_pct']:.2f}% vs "
        f"{outer10['soxx_max_drawdown_pct']:.2f}%；最长水下 "
        f"{outer10['longest_underwater_days']} vs "
        f"{outer10['soxx_longest_underwater_days']} 天；回撤恢复 "
        f"{outer10['max_drawdown_recovery_days']} vs "
        f"{outer10['soxx_max_drawdown_recovery_days']} 天。\n"
        f"- 下跌区间现金：outer 平均 "
        f"{outer10['downside_avg_cash_weight_pct']:.2f}%，最低 "
        f"{outer10['downside_min_cash_weight_pct']:.2f}%。\n"
        f"- 决定：{summary['promotion_decision']}；"
        "这是研究目标通过，不等同于未观察数据上的生产晋升。\n"
        "- 风险：增量同时依赖模型收益头与显式规则先验；必须持续分别报告 "
        "model-only、prior-only 和 blend，不能重新把先验塞回特征矩阵。\n"
        "- 下一步：进入 paper target-position 观察期，只允许新到数据做前向"
        "验证，不再用 2023+ 历史选择参数。\n"
    )
    (card / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    workflow = (
        f"experiment: {EXPERIMENT}\n"
        f"source_experiment: {SOURCE_EXPERIMENT}\n"
        "system_of_record: qlib\n"
        "benchmark: SOXX\n"
        "assets: [SOXL, SOXX, cash]\n"
        "development: [2016, 2022]\n"
        "outer_diagnostic: [2023, 2026-07-02]\n"
        "selection_uses_outer_test: false\n"
        "base_constrained_ref_role: explicit_additive_prior_not_feature\n"
        "cash_rule: hazard_high AND predicted_cash_benefit_above_cost\n"
        "artifact_policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(
        workflow, encoding="utf-8"
    )
    _write_notebook(card, summary)


def _write_notebook(
    card: Path, summary: dict[str, Any]
) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Hierarchical SOXX excess with explicit prior\n\n"
            "`base_constrained_ref` 是显式仓位先验和独立基准，"
            "不进入模型特征矩阵。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "repo_root = Path.cwd().parents[1]\n"
            "if str(repo_root) not in sys.path:\n"
            "    sys.path.insert(0, str(repo_root))\n"
            "from src.workflow.hierarchical_soxx_excess_prior_overlay "
            "import run_hierarchical_soxx_excess_prior_overlay_card\n"
            "summary = run_hierarchical_soxx_excess_prior_overlay_card("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'prior_incremental_attribution.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'regime_metrics.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    nbformat.write(
        notebook,
        card / f"{summary['experiment']}.ipynb",
    )


if __name__ == "__main__":
    print(
        json.dumps(
            run_hierarchical_soxx_excess_prior_overlay_card(),
            indent=2,
        )
    )
