from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.soxs_defensive_hurdle import (
    ALL_FEATURES,
    BASE_FEATURES,
    _annual_metrics,
    _benchmark,
    _bootstrap_validation,
    _build_dataset,
    _feature_importance,
    _fit_classifier,
    _json_ready,
    _load_feature_frame,
    _metrics,
    _predict_probabilities,
    _prepare,
)
from src.workflow.leveraged_etf_paper_replay import (
    _matched_beta_weights,
    _risk_metrics,
    _simulate_weights,
)
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "hierarchical_four_asset_policy_v1"
VARIANTS = (
    "cash_defense_no_soxs",
    "soxs_5pct_1d",
    "soxs_5pct_3d",
    "soxs_10pct_1d",
    "soxs_10pct_3d",
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/probabilities.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/bootstrap_terminal_wealth.csv",
)


@dataclass(frozen=True)
class HierarchicalFourAssetSpec:
    root: Path
    experiment: str = EXPERIMENT
    initial_train: tuple[str, str] = ("2011-01-03", "2016-11-30")
    tune: tuple[str, str] = ("2017-01-03", "2019-11-29")
    refit: tuple[str, str] = ("2011-01-03", "2019-11-29")
    validation: tuple[str, str] = ("2020-01-02", "2022-12-28")
    outer: tuple[str, str] = ("2023-01-03", "2026-12-31")
    costs_bps: tuple[int, ...] = (10, 25, 50)
    bootstrap_paths: int = 4096
    bootstrap_block_days: int = 63

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


@dataclass(frozen=True)
class HierarchicalPolicyConfig:
    variant: str
    attack_threshold: float
    defense_threshold: float
    moderate_gap: float
    soxs_weight: float
    max_soxs_days: int
    cooldown_days: int = 2
    ordinary_soxl_weight: float = 0.20
    strong_soxl_weight: float = 0.40
    strong_attack_increment: float = 0.08
    risk_down_soxx_weight: float = 0.60


def run_hierarchical_four_asset_policy_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(HierarchicalFourAssetSpec(root=repo_root))


def _variant_overlay(variant: str) -> tuple[float, int]:
    mapping = {
        "cash_defense_no_soxs": (0.0, 0),
        "soxs_5pct_1d": (0.05, 1),
        "soxs_5pct_3d": (0.05, 3),
        "soxs_10pct_1d": (0.10, 1),
        "soxs_10pct_3d": (0.10, 3),
    }
    return mapping[variant]


def candidate_configs() -> list[HierarchicalPolicyConfig]:
    configs: list[HierarchicalPolicyConfig] = []
    for variant in VARIANTS:
        soxs_weight, max_soxs_days = _variant_overlay(variant)
        for attack_threshold in (0.55, 0.60, 0.65):
            for defense_threshold in (0.70, 0.80, 0.85):
                for moderate_gap in (0.05, 0.10):
                    configs.append(
                        HierarchicalPolicyConfig(
                            variant=variant,
                            attack_threshold=attack_threshold,
                            defense_threshold=defense_threshold,
                            moderate_gap=moderate_gap,
                            soxs_weight=soxs_weight,
                            max_soxs_days=max_soxs_days,
                        )
                    )
    return configs


def simulate_hierarchical_policy(
    market: pd.DataFrame,
    probabilities: pd.DataFrame,
    config: HierarchicalPolicyConfig,
    *,
    cost_bps: int,
) -> pd.DataFrame:
    drifted_weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    defense_streak = 0
    cooldown_remaining = 0
    was_severe = False
    rows: list[dict[str, Any]] = []
    for row in market.itertuples():
        probability = probabilities.loc[row.Index]
        severe = bool(
            probability["defense_soxs_probability"] >= config.defense_threshold
            and row.soxx_trend20 < 0.0
            and row.soxs_trend5 > 0.0
        )
        moderate = bool(
            probability["defense_soxs_probability"] >= config.defense_threshold - config.moderate_gap
            and row.soxx_trend20 < 0.0
        )
        if was_severe and not severe:
            cooldown_remaining = config.cooldown_days

        if severe:
            defense_streak += 1
            cooldown_remaining = 0
            if config.soxs_weight > 0.0 and defense_streak <= config.max_soxs_days:
                target = np.array([0.0, 0.0, config.soxs_weight, 1.0 - config.soxs_weight])
                state = "soxs_tail_overlay"
            else:
                target = np.array([0.0, 0.0, 0.0, 1.0])
                state = "cash_severe"
        elif cooldown_remaining > 0:
            defense_streak = 0
            target = np.array([0.0, 0.30, 0.0, 0.70])
            state = "post_defense_cooldown"
            cooldown_remaining -= 1
        elif moderate:
            defense_streak = 0
            target = np.array([0.0, config.risk_down_soxx_weight, 0.0, 1.0 - config.risk_down_soxx_weight])
            state = "risk_down_soxx_cash"
        else:
            defense_streak = 0
            attack_probability = float(probability["attack_soxs_probability"])
            if attack_probability >= config.attack_threshold + config.strong_attack_increment:
                target = np.array([config.strong_soxl_weight, 1.0 - config.strong_soxl_weight, 0.0, 0.0])
                state = "strong_risk_on"
            elif attack_probability >= config.attack_threshold:
                target = np.array([config.ordinary_soxl_weight, 1.0 - config.ordinary_soxl_weight, 0.0, 0.0])
                state = "ordinary_risk_on"
            else:
                target = np.array([0.0, 1.0, 0.0, 0.0])
                state = "soxx_anchor"

        if target[0] > 0.0 and target[2] > 0.0:
            raise AssertionError("SOXL and SOXS must remain mutually exclusive")
        turnover = float(0.5 * np.abs(target - drifted_weights).sum())
        transaction_cost = turnover * float(cost_bps) / 10_000.0
        asset_gross = np.array(
            [
                1.0 + row.soxl_forward_open_return,
                1.0 + row.soxx_forward_open_return,
                1.0 + row.soxs_forward_open_return,
                1.0,
            ]
        )
        portfolio_gross = float(target @ asset_gross)
        strategy_return = (1.0 - transaction_cost) * portfolio_gross - 1.0
        drifted_weights = target * asset_gross / max(portfolio_gross, 1e-12)
        rows.append(
            {
                "signal_date": row.Index,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "state": state,
                "soxl_return": row.soxl_forward_open_return,
                "soxx_return": row.soxx_forward_open_return,
                "soxs_return": row.soxs_forward_open_return,
                "strategy_return": strategy_return,
                "soxl_weight": target[0],
                "soxx_weight": target[1],
                "soxs_weight": target[2],
                "cash_weight": target[3],
                "turnover": turnover,
                "attack_probability": probability["attack_soxs_probability"],
                "defense_probability": probability["defense_soxs_probability"],
            }
        )
        was_severe = severe
    return pd.DataFrame(rows)


def _select_tune_configs(
    market: pd.DataFrame,
    probabilities: pd.DataFrame,
    spec: HierarchicalFourAssetSpec,
) -> tuple[pd.DataFrame, dict[str, HierarchicalPolicyConfig]]:
    rows: list[dict[str, Any]] = []
    benchmark10 = _benchmark(market, cost_bps=10)
    benchmark10_metrics = _metrics(benchmark10, benchmark10)
    for config in candidate_configs():
        cost_metrics: dict[int, dict[str, Any]] = {}
        for cost_bps in (10, 25):
            benchmark = _benchmark(market, cost_bps=cost_bps)
            result = simulate_hierarchical_policy(market, probabilities, config, cost_bps=cost_bps)
            cost_metrics[cost_bps] = _metrics(result, benchmark)
        ten = cost_metrics[10]
        stress = cost_metrics[25]
        score = float(
            np.log(max(ten["terminal_wealth_ratio_vs_soxx"], 1e-8))
            + np.log(max(stress["terminal_wealth_ratio_vs_soxx"], 1e-8))
            + 0.02 * (ten["active_information_ratio"] + stress["active_information_ratio"])
            + 0.002 * (ten["max_drawdown_pct"] - benchmark10_metrics["max_drawdown_pct"])
        )
        rows.append(
            {
                **asdict(config),
                "tune_score": score,
                "tune_ratio_10bps": ten["terminal_wealth_ratio_vs_soxx"],
                "tune_ratio_25bps": stress["terminal_wealth_ratio_vs_soxx"],
                "tune_active_ir_10bps": ten["active_information_ratio"],
                "tune_max_drawdown_10bps": ten["max_drawdown_pct"],
                "selection_uses_outer_test": False,
            }
        )
    candidates = pd.DataFrame(rows).sort_values(["variant", "tune_score"], ascending=[True, False])
    selected: dict[str, HierarchicalPolicyConfig] = {}
    for variant in VARIANTS:
        row = candidates.loc[candidates["variant"].eq(variant)].iloc[0]
        selected[variant] = HierarchicalPolicyConfig(
            variant=variant,
            attack_threshold=float(row["attack_threshold"]),
            defense_threshold=float(row["defense_threshold"]),
            moderate_gap=float(row["moderate_gap"]),
            soxs_weight=float(row["soxs_weight"]),
            max_soxs_days=int(row["max_soxs_days"]),
            cooldown_days=int(row["cooldown_days"]),
            ordinary_soxl_weight=float(row["ordinary_soxl_weight"]),
            strong_soxl_weight=float(row["strong_soxl_weight"]),
            strong_attack_increment=float(row["strong_attack_increment"]),
            risk_down_soxx_weight=float(row["risk_down_soxx_weight"]),
        )
    return candidates, selected


def _validation_rows(
    comparison: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    selected_configs: dict[str, HierarchicalPolicyConfig],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, config in selected_configs.items():
        ten = comparison.query("segment == 'validation' and variant == @variant and cost_bps == 10").iloc[0]
        stress = comparison.query("segment == 'validation' and variant == @variant and cost_bps == 25").iloc[0]
        bootstrap10 = bootstrap_metrics.query("variant == @variant and cost_bps == 10").iloc[0]
        years = annual_metrics.query("segment == 'validation' and variant == @variant and cost_bps == 10")
        positive_years = int(years["wealth_ratio"].gt(1.0).sum())
        relative_gate = bool(
            ten["terminal_wealth_ratio_vs_soxx"] > 1.0
            and stress["terminal_wealth_ratio_vs_soxx"] > 1.0
            and ten["active_information_ratio"] > 0.0
            and bootstrap10["outperformance_probability"] >= 0.55
            and positive_years >= 2
        )
        risk_gate = bool(
            ten["max_drawdown_pct"] >= -20.0
            and ten["worst_day_pct"] >= -12.0
            and ten["cvar5_pct"] >= -8.0
            and ten["worst_rolling_5d_pct"] >= -12.0
            and ten["avg_total_risk_position_pct"] >= 15.0
            and ten["upside_capture_pct"] >= 60.0
        )
        benchmark_efficiency_gate = bool(
            ten["return_maxdd"] > ten["same_avg_risk_soxl_return_maxdd"]
            and ten["return_maxdd"] > ten["matched_beta_return_maxdd"]
        )
        validation_score = float(
            np.log(max(ten["terminal_wealth_ratio_vs_soxx"], 1e-8))
            + 0.02 * ten["active_information_ratio"]
            + 0.002 * ten["max_drawdown_pct"]
        )
        rows.append(
            {
                **asdict(config),
                "validation_gate_pass": bool(relative_gate and risk_gate and benchmark_efficiency_gate),
                "relative_gate_pass": relative_gate,
                "risk_gate_pass": risk_gate,
                "benchmark_efficiency_gate_pass": benchmark_efficiency_gate,
                "validation_score": validation_score,
                "validation_ratio_10bps": ten["terminal_wealth_ratio_vs_soxx"],
                "validation_ratio_25bps": stress["terminal_wealth_ratio_vs_soxx"],
                "validation_active_ir_10bps": ten["active_information_ratio"],
                "validation_max_drawdown_10bps": ten["max_drawdown_pct"],
                "validation_worst_rolling_5d_10bps": ten["worst_rolling_5d_pct"],
                "validation_upside_capture_10bps": ten["upside_capture_pct"],
                "validation_bootstrap_outperformance_probability_10bps": bootstrap10[
                    "outperformance_probability"
                ],
                "positive_validation_years": positive_years,
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["validation_gate_pass", "validation_score"], ascending=[False, False]
    )


def _benchmark_efficiency(result: pd.DataFrame, *, cost_bps: int) -> dict[str, Any]:
    frame = result.loc[:, ["signal_date", "trade_date", "exit_date", "soxl_return", "soxx_return"]]
    average_risk_position = float(
        (result["soxl_weight"] + result["soxx_weight"] + result["soxs_weight"]).mean()
    )
    same_soxl = _simulate_weights(
        frame,
        soxl_weight=np.full(len(frame), np.clip(average_risk_position, 0.0, 1.0)),
        soxx_weight=np.zeros(len(frame)),
        cost_bps=cost_bps,
    )
    average_effective_beta = float(
        (3.0 * result["soxl_weight"] + result["soxx_weight"] - 3.0 * result["soxs_weight"]).mean()
    )
    matched_soxl_weight, matched_soxx_weight = _matched_beta_weights(average_effective_beta, len(frame))
    matched_beta = _simulate_weights(
        frame,
        soxl_weight=matched_soxl_weight,
        soxx_weight=matched_soxx_weight,
        cost_bps=cost_bps,
    )
    strategy_metrics = _risk_metrics(result)
    same_metrics = _risk_metrics(same_soxl)
    matched_metrics = _risk_metrics(matched_beta)

    def return_maxdd(metrics: dict[str, float]) -> float:
        drawdown = abs(float(metrics["max_drawdown_pct"]))
        return float(metrics["return_pct"]) / drawdown if drawdown > 0.0 else np.nan

    return {
        "avg_effective_beta": average_effective_beta,
        "return_maxdd": return_maxdd(strategy_metrics),
        "same_avg_risk_soxl_return_pct": same_metrics["return_pct"],
        "same_avg_risk_soxl_max_drawdown_pct": same_metrics["max_drawdown_pct"],
        "same_avg_risk_soxl_return_maxdd": return_maxdd(same_metrics),
        "matched_beta_return_pct": matched_metrics["return_pct"],
        "matched_beta_max_drawdown_pct": matched_metrics["max_drawdown_pct"],
        "matched_beta_return_maxdd": return_maxdd(matched_metrics),
    }


def _outer_year_risk_metrics(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outer = strategy_returns.loc[strategy_returns["segment"].eq("outer")].copy()
    outer["outer_year"] = pd.to_datetime(outer["trade_date"]).dt.year
    for (variant, cost_bps, outer_year), frame in outer.groupby(
        ["variant", "cost_bps", "outer_year"], sort=True
    ):
        metrics = _risk_metrics(frame.reset_index(drop=True))
        gate = bool(
            metrics["max_drawdown_pct"] >= -20.0
            and metrics["worst_day_pct"] >= -12.0
            and metrics["cvar5_pct"] >= -8.0
            and metrics["worst_rolling_5d_pct"] >= -12.0
        )
        rows.append(
            {
                "variant": variant,
                "cost_bps": int(cost_bps),
                "outer_year": int(outer_year),
                "risk_gate_pass": gate,
                "selection_uses_outer_test": False,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _run_card(spec: HierarchicalFourAssetSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, data_audit = _load_feature_frame(spec)
    dataset, dataset_audit = _build_dataset(market, spec)
    initial_train = _prepare(dataset, "initial_train")
    refit = _prepare(dataset, "refit")
    tune_market = market.loc[spec.tune[0] : spec.tune[1]].copy()
    validation = market.loc[spec.validation[0] : spec.validation[1]].copy()
    outer_end = dataset_audit["segments"]["outer"][1]
    outer = market.loc[spec.outer[0] : outer_end].copy()

    initial_models = {
        "attack_base": _fit_classifier(initial_train, BASE_FEATURES, "attack_label"),
        "attack_soxs": _fit_classifier(initial_train, ALL_FEATURES, "attack_label"),
        "defense_soxs": _fit_classifier(initial_train, ALL_FEATURES, "defense_label"),
    }
    tune_probabilities = _predict_probabilities(tune_market, **initial_models)
    candidates, selected_configs = _select_tune_configs(tune_market, tune_probabilities, spec)

    refit_models = {
        "attack_base": _fit_classifier(refit, BASE_FEATURES, "attack_label"),
        "attack_soxs": _fit_classifier(refit, ALL_FEATURES, "attack_label"),
        "defense_soxs": _fit_classifier(refit, ALL_FEATURES, "defense_label"),
    }
    segment_probabilities = {
        "validation": _predict_probabilities(validation, **refit_models),
        "outer": _predict_probabilities(outer, **refit_models),
    }

    comparison_rows: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    stored_returns: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_distributions: list[pd.DataFrame] = []
    for segment_name, segment_market in (("validation", validation), ("outer", outer)):
        probabilities = segment_probabilities[segment_name]
        for cost_bps in spec.costs_bps:
            benchmark = _benchmark(segment_market, cost_bps=cost_bps)
            comparison_rows.append(
                {
                    "segment": segment_name,
                    "variant": "soxx_buyhold",
                    "cost_bps": cost_bps,
                    **_metrics(benchmark, benchmark),
                }
            )
            for variant, config in selected_configs.items():
                result = simulate_hierarchical_policy(
                    segment_market, probabilities, config, cost_bps=cost_bps
                )
                result_metrics = _metrics(result, benchmark)
                result_metrics.update(_benchmark_efficiency(result, cost_bps=cost_bps))
                comparison_rows.append(
                    {
                        "segment": segment_name,
                        "variant": variant,
                        "cost_bps": cost_bps,
                        **result_metrics,
                    }
                )
                result = result.assign(segment=segment_name, variant=variant, cost_bps=cost_bps)
                stored_returns.append(result)
                annual_frames.append(
                    _annual_metrics(result, benchmark).assign(
                        segment=segment_name, variant=variant, cost_bps=cost_bps
                    )
                )
                if segment_name == "validation" and cost_bps in (10, 25):
                    bootstrap, distribution = _bootstrap_validation(result, benchmark, spec)
                    bootstrap_rows.append({"variant": variant, "cost_bps": cost_bps, **bootstrap})
                    bootstrap_distributions.append(
                        distribution.assign(variant=variant, cost_bps=cost_bps)
                    )

    comparison = pd.DataFrame(comparison_rows)
    annual_metrics = pd.concat(annual_frames, ignore_index=True)
    bootstrap_metrics = pd.DataFrame(bootstrap_rows)
    model_selection = _validation_rows(
        comparison, annual_metrics, bootstrap_metrics, selected_configs
    )
    selected = model_selection.iloc[0].to_dict()
    selected_variant = str(selected["variant"])
    validation_pass = bool(selected["validation_gate_pass"])
    promotion_decision = (
        "research_diagnostic_validation_pass_requires_sealed_forward"
        if validation_pass
        else "research_reject_hierarchical_four_asset_failed_validation_gates"
    )
    probabilities = pd.concat(
        [frame.assign(segment=name) for name, frame in segment_probabilities.items()]
    ).reset_index(names="signal_date")
    strategy_returns = pd.concat(stored_returns, ignore_index=True)
    outer_year_metrics = _outer_year_risk_metrics(strategy_returns)
    positions = strategy_returns.loc[
        :,
        [
            "signal_date",
            "trade_date",
            "exit_date",
            "segment",
            "variant",
            "cost_bps",
            "state",
            "soxl_weight",
            "soxx_weight",
            "soxs_weight",
            "cash_weight",
        ],
    ]
    bootstrap_distribution = pd.concat(bootstrap_distributions, ignore_index=True)
    feature_importance = _feature_importance(refit_models)

    sidecars = {
        "model_selection": "model_selection.csv",
        "candidate_selection": "candidate_selection.csv",
        "bootstrap_metrics": "bootstrap_metrics.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "annual_metrics": "annual_metrics.csv",
        "outer_year_risk_metrics": "outer_year_risk_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "feature_importance": "feature_importance.csv",
        "selected_model": "selected_model.json",
        "analysis_summary": "analysis_summary.json",
        "failure_modes": "failure_modes.md",
    }
    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temporary:
        temp = Path(temporary)
        model_dir = temp / "models"
        model_dir.mkdir()
        for name, model in refit_models.items():
            with (model_dir / f"{name}.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        probabilities.to_csv(temp / "probabilities.csv", index=False)
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(temp / "strategy_returns.csv", index=False)
        bootstrap_distribution.to_csv(temp / "bootstrap_terminal_wealth.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                initial_train=f"{spec.initial_train[0]}:{spec.initial_train[1]}",
                tune=f"{spec.tune[0]}:{spec.tune[1]}",
                refit=f"{spec.refit[0]}:{spec.refit[1]}",
                validation=f"{spec.validation[0]}:{spec.validation[1]}",
                outer=f"{spec.outer[0]}:{outer_end}",
                variants=",".join(VARIANTS),
                selection_uses_outer_test="false",
                selected_variant=selected_variant,
            )
            R.log_metrics(
                validation_gate_pass=float(validation_pass),
                validation_ratio_10bps=float(selected["validation_ratio_10bps"]),
                validation_active_ir_10bps=float(selected["validation_active_ir_10bps"]),
                validation_max_drawdown_10bps=float(selected["validation_max_drawdown_10bps"]),
            )
            for artifact in [model_dir, *temp.glob("*.csv")]:
                recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            selected_validation = comparison.query(
                "segment == 'validation' and variant == @selected_variant"
            ).to_dict(orient="records")
            selected_outer = comparison.query(
                "segment == 'outer' and variant == @selected_variant"
            ).to_dict(orient="records")
            summary = {
                "experiment": spec.experiment,
                "card_type": "hierarchical_soxl_soxx_cash_with_short_soxs_overlay",
                "primary_notebook": f"{spec.experiment}.ipynb",
                "promotion_decision": promotion_decision,
                "selection_uses_outer_test": False,
                "outer_test_use": "diagnostic_only_previously_observed_requires_new_sealed_forward",
                "data_audit": data_audit,
                "qlib_dataset": dataset_audit,
                "qlib_docs": [
                    "https://qlib.readthedocs.io/en/latest/component/data.html",
                    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                    "https://qlib.readthedocs.io/en/latest/component/strategy.html",
                ],
                "hypothesis": "cash controls gross risk, SOXX anchors the benchmark, SOXL adds conditional upside, and SOXS is only a short mutually-exclusive tail overlay",
                "constraints": {
                    "soxl_soxs_mutually_exclusive": True,
                    "soxs_weights_tested": [0.05, 0.10],
                    "soxs_max_days_tested": [1, 3],
                    "cooldown_days": 2,
                    "ordinary_soxl_weight": 0.20,
                    "strong_soxl_weight": 0.40,
                    "risk_down_soxx_weight": 0.60,
                    "severe_state": "cash plus optional SOXS overlay",
                    "signal_trade_timing": "signal at T close, weights applied T+1 open to T+2 open",
                    "return_maxdd_must_exceed_same_average_risk_soxl": True,
                    "return_maxdd_must_exceed_matched_beta_soxl_soxx_cash": True,
                    "outer_year_risk_requirement": "report at least 3 of 4 years; diagnostic only and excluded from selection",
                },
                "selected_model": _json_ready(selected),
                "selected_config": asdict(selected_configs[selected_variant]),
                "model_selection_uses_segment": "validation; candidate thresholds fixed on tune",
                "selected_validation_metrics": _json_ready(selected_validation),
                "selected_outer_metrics": _json_ready(selected_outer),
                "outer_diagnostic": {
                    "selection_uses_outer_test": False,
                    "required_years_risk_passed": 3,
                    "selected_variant_years_risk_passed_10bps": int(
                        outer_year_metrics.query(
                            "variant == @selected_variant and cost_bps == 10"
                        )["risk_gate_pass"].sum()
                    ),
                    "years": sorted(
                        int(year)
                        for year in outer_year_metrics.query(
                            "variant == @selected_variant and cost_bps == 10"
                        )["outer_year"].unique()
                    ),
                },
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
            _write_card(
                card,
                summary,
                model_selection=model_selection,
                candidates=candidates,
                bootstrap_metrics=bootstrap_metrics,
                comparison=comparison,
                annual_metrics=annual_metrics,
                outer_year_metrics=outer_year_metrics,
                feature_importance=feature_importance,
            )
            for relative_path in sidecars.values():
                artifact = card / relative_path
                if artifact.exists() and artifact.suffix != ".md":
                    recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _write_card(
    card: Path,
    summary: dict[str, Any],
    *,
    model_selection: pd.DataFrame,
    candidates: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    outer_year_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    model_selection.to_csv(card / "model_selection.csv", index=False)
    candidates.to_csv(card / "candidate_selection.csv", index=False)
    bootstrap_metrics.to_csv(card / "bootstrap_metrics.csv", index=False)
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
    outer_year_metrics.to_csv(card / "outer_year_risk_metrics.csv", index=False)
    feature_importance.to_csv(card / "feature_importance.csv", index=False)
    comparison.loc[
        :,
        [
            "segment",
            "variant",
            "cost_bps",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "annualized_vol_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    selected = summary["selected_model"]
    (card / "selected_model.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    (card / "analysis_summary.json").write_text(
        json.dumps(
            {
                "promotion_decision": summary["promotion_decision"],
                "selected_model": selected,
                "selected_validation_metrics": summary["selected_validation_metrics"],
                "selected_outer_metrics": summary["selected_outer_metrics"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    validation10 = next(row for row in summary["selected_validation_metrics"] if row["cost_bps"] == 10)
    outer10 = next(row for row in summary["selected_outer_metrics"] if row["cost_bps"] == 10)
    conclusion = f"""# 中文结论

## 跑了什么

用独立 qlib SOXL/SOXX/SOXS provider 执行分层策略：现金控制总风险，SOXX 为核心，SOXL 为 20%/40% 条件进攻仓位，SOXS 仅测试 5%/10% 与 1/3 日尾部覆盖。SOXL 与 SOXS 互斥，SOXS 退出后冷却两日。候选阈值只在 2017–2019 tune 选择；2020–2022 validation 选择变体；2023–2026 outer 只诊断。

## Validation

选中 `{selected['variant']}`。10bp：收益 {validation10['return_pct']:.2f}%，SOXX {validation10['soxx_benchmark_return_pct']:.2f}%，财富比 {validation10['terminal_wealth_ratio_vs_soxx']:.4f}，Active IR {validation10['active_information_ratio']:.4f}，MaxDD {validation10['max_drawdown_pct']:.2f}%，最差五日 {validation10['worst_rolling_5d_pct']:.2f}%，SOXS 平均仓位 {validation10['avg_soxs_position_pct']:.2f}%，现金平均仓位 {validation10['avg_cash_position_pct']:.2f}%。validation gate={selected['validation_gate_pass']}。

## Outer 诊断

10bp：收益 {outer10['return_pct']:.2f}%，SOXX {outer10['soxx_benchmark_return_pct']:.2f}%，财富比 {outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD {outer10['max_drawdown_pct']:.2f}%；四个年份中 {summary['outer_diagnostic']['selected_variant_years_risk_passed_10bps']} 个通过年度尾部风险门。outer 不参与选择。

## 决定、风险和下一步

`{summary['promotion_decision']}`。若 validation 失败，说明把四种资产做成分层状态机仍未修复进攻标签漂移；SOXS 只能保留为特征或极小尾部压力测试，不能晋升。下一步应更换 benchmark-relative 进攻目标，而不是用 outer 返调当前阈值。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        """# Failure modes

- The 21-day attack classifier can drift across regimes; tune performance is not proof of validation stability.
- SOXS is a daily-reset inverse leveraged ETF, so even a three-day overlay is path dependent and can decay in chop.
- Cash has zero return because a point-in-time Treasury return series is not in this provider.
- A high defense threshold can miss the beginning of a crash; a low threshold can cause chronic cash drag.
- The 2023-2026 outer interval was previously observed and remains diagnostic only.
""",
        encoding="utf-8",
    )
    workflow = f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
provider_uri: {summary['data_audit']['provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
qlib:
  market: all
  region: us
  dataset: [DatasetH, DataHandlerLP]
  recorder: qlib.workflow.R
selection:
  candidate_segment: tune_2017_2019
  model_selection_segment: validation_2020_2022
  selection_uses_outer_test: false
  costs_bps: [10, 25, 50]
constraints:
  soxl_soxs_mutually_exclusive: true
  soxs_weights: [0.05, 0.10]
  soxs_max_days: [1, 3]
  cooldown_days: 2
  return_maxdd_benchmarks: [same_average_risk_soxl, matched_beta_soxl_soxx_cash]
  outer_year_risk_requirement: 3_of_4_report_only
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/models/
    - artifacts/{summary['experiment']}/probabilities.csv
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
    - artifacts/{summary['experiment']}/bootstrap_terminal_wealth.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_hierarchical_four_asset_policy_card(), indent=2))
