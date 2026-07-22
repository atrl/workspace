from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.objective.relative_wealth import stationary_block_bootstrap_indices
from src.workflow.leveraged_etf_paper_replay import _risk_metrics
from src.workflow.soxl_objective_feasibility_audit import _add_targets, _json_ready, _simulate
from src.workflow.soxs_defensive_hurdle import ALL_FEATURES, SoxsDefensiveHurdleSpec, _load_feature_frame
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "cash_floor_benchmark_relative_walk_forward_v1"
DEVELOPMENT_FOLDS = tuple(range(2016, 2023))
TARGETS = ("cash_advantage_5d", "future_vol_5d")
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/bootstrap_terminal_wealth.csv",
)


@dataclass(frozen=True)
class CashFloorBenchmarkRelativeSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_folds: tuple[int, ...] = DEVELOPMENT_FOLDS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = ("2023-01-03", "2026-12-31")
    purge_rows: int = 20
    ridge_alpha: float = 10.0
    minimum_cash: float = 0.10
    maximum_cash: float = 0.60
    risky_sleeve_soxl_share: float = 0.30
    calibration_lower_quantile: float = 0.50
    calibration_upper_quantile: float = 0.90
    costs_bps: tuple[int, ...] = (10, 25)
    bootstrap_paths: int = 2048
    bootstrap_block_days: int = 63

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def requirements_path(self) -> Path:
        return self.root / "docs" / "superpowers" / "specs" / (
            "2026-07-22-cash-floor-benchmark-relative-walk-forward-requirements.md"
        )


def run_cash_floor_benchmark_relative_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(CashFloorBenchmarkRelativeSpec(root=repo_root))


def _build_dataset(frame: pd.DataFrame, spec: CashFloorBenchmarkRelativeSpec):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat(
        {"feature": frame.loc[:, ALL_FEATURES], "label": frame.loc[:, TARGETS]}, axis=1
    )
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_SOXS")],
        names=["datetime", "instrument"],
    )
    outer_end = min(spec.outer[1], frame.index.max().date().isoformat())
    segments: dict[str, tuple[str, str]] = {
        "development": (f"{spec.development_folds[0]}-01-01", spec.development_end),
        "outer": (spec.outer[0], outer_end),
    }
    for year in spec.development_folds:
        segments[f"fold_{year}"] = (f"{year}-01-01", f"{year}-12-31")
    handler = DataHandlerLP(
        instruments=None,
        start_time=frame.index.min(),
        end_time=frame.index.max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    return dataset, {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {name: int(len(dataset.prepare(name))) for name in segments},
        "purge_rows": spec.purge_rows,
        "selection_uses_outer_test": False,
    }


def _prepare(dataset, segment: str) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result = result.droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _model(spec: CashFloorBenchmarkRelativeSpec) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=spec.ridge_alpha))]
    )


def _fit_head(
    train: pd.DataFrame,
    target: str,
    spec: CashFloorBenchmarkRelativeSpec,
) -> tuple[Pipeline, dict[str, float]]:
    usable = train.dropna(subset=[*ALL_FEATURES, target])
    model = _model(spec)
    model.fit(usable.loc[:, ALL_FEATURES], usable[target])
    prediction = model.predict(usable.loc[:, ALL_FEATURES])
    lower, upper = np.quantile(
        prediction,
        [spec.calibration_lower_quantile, spec.calibration_upper_quantile],
    )
    if upper <= lower:
        raise ValueError(f"blocked: invalid training-only calibration for {target}")
    return model, {"lower": float(lower), "upper": float(upper)}


def _predict_score(
    model: Pipeline,
    calibration: dict[str, float],
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = model.predict(test.loc[:, ALL_FEATURES])
    score = np.clip(
        (prediction - calibration["lower"])
        / (calibration["upper"] - calibration["lower"]),
        0.0,
        1.0,
    )
    return prediction, score


def _target_weights(risk_score: np.ndarray, spec: CashFloorBenchmarkRelativeSpec) -> np.ndarray:
    score = np.asarray(risk_score, dtype=float)
    cash = spec.minimum_cash + (spec.maximum_cash - spec.minimum_cash) * np.clip(score, 0.0, 1.0)
    risky = 1.0 - cash
    soxl = risky * spec.risky_sleeve_soxl_share
    soxx = risky * (1.0 - spec.risky_sleeve_soxl_share)
    weights = np.column_stack([soxl, soxx, np.zeros(len(score)), cash])
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise AssertionError("cash-floor target weights must sum to one")
    return weights


def _walk_forward_predictions(
    dataset,
    market: pd.DataFrame,
    spec: CashFloorBenchmarkRelativeSpec,
):
    predictions: list[pd.DataFrame] = []
    fold_models: dict[str, Pipeline] = {}
    calibration_rows: list[dict[str, Any]] = []
    for year in spec.development_folds:
        fold = _prepare(dataset, f"fold_{year}")
        fold = fold.loc[fold.index.year == year].dropna(subset=[*ALL_FEATURES, *TARGETS])
        start_position = market.index.searchsorted(pd.Timestamp(f"{year}-01-01"))
        train_end_position = start_position - spec.purge_rows - 1
        if train_end_position < 0:
            raise ValueError(f"blocked: insufficient training history for fold {year}")
        train_end = market.index[train_end_position]
        train = market.loc[spec.train_start:train_end].dropna(subset=[*ALL_FEATURES, *TARGETS])
        output = pd.DataFrame(index=fold.index)
        output["segment"] = "development_walk_forward"
        output["fold_year"] = year
        output["train_end"] = train_end
        scores: list[np.ndarray] = []
        for target in TARGETS:
            model, calibration = _fit_head(train, target, spec)
            prediction, score = _predict_score(model, calibration, fold)
            output[f"{target}_actual"] = fold[target]
            output[f"{target}_prediction"] = prediction
            output[f"{target}_risk_score"] = score
            scores.append(score)
            fold_models[f"fold_{year}__{target}"] = model
            calibration_rows.append(
                {
                    "segment": "development_walk_forward",
                    "fold_year": year,
                    "target": target,
                    "train_start": spec.train_start,
                    "train_end": train_end.date().isoformat(),
                    "train_rows": len(train),
                    **calibration,
                    "selection_uses_outer_test": False,
                }
            )
        output["risk_score"] = np.maximum(scores[0], scores[1])
        predictions.append(output)

    outer = _prepare(dataset, "outer").dropna(subset=[*ALL_FEATURES, *TARGETS])
    development_train = market.loc[spec.train_start:spec.development_end].dropna(
        subset=[*ALL_FEATURES, *TARGETS]
    )
    output = pd.DataFrame(index=outer.index)
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = outer.index.year
    output["train_end"] = pd.Timestamp(spec.development_end)
    scores = []
    final_models: dict[str, Pipeline] = {}
    for target in TARGETS:
        model, calibration = _fit_head(development_train, target, spec)
        prediction, score = _predict_score(model, calibration, outer)
        output[f"{target}_actual"] = outer[target]
        output[f"{target}_prediction"] = prediction
        output[f"{target}_risk_score"] = score
        scores.append(score)
        final_models[f"outer_refit__{target}"] = model
        calibration_rows.append(
            {
                "segment": "outer_diagnostic",
                "fold_year": "all",
                "target": target,
                "train_start": spec.train_start,
                "train_end": spec.development_end,
                "train_rows": len(development_train),
                **calibration,
                "selection_uses_outer_test": False,
            }
        )
    output["risk_score"] = np.maximum(scores[0], scores[1])
    predictions.append(output)
    return (
        pd.concat(predictions).sort_index(),
        pd.DataFrame(calibration_rows),
        {**fold_models, **final_models},
    )


def _result_metrics(result: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    metrics = _risk_metrics(result)
    wealth = float((1.0 + result["strategy_return"]).prod())
    benchmark_wealth = float((1.0 + benchmark["strategy_return"]).prod())
    metrics.update(
        {
            "terminal_wealth": wealth,
            "terminal_wealth_ratio_vs_soxx": wealth / benchmark_wealth,
            "soxx_terminal_wealth": benchmark_wealth,
            "soxx_return_pct": 100.0 * (benchmark_wealth - 1.0),
            "avg_cash_weight_pct": 100.0 * float(result["cash_weight"].mean()),
            "min_cash_weight_pct": 100.0 * float(result["cash_weight"].min()),
            "max_cash_weight_pct": 100.0 * float(result["cash_weight"].max()),
            "zero_cash_days": int(result["cash_weight"].le(1e-12).sum()),
            "avg_soxl_weight_pct": 100.0 * float(result["soxl_weight"].mean()),
            "avg_soxx_weight_pct": 100.0 * float(result["soxx_weight"].mean()),
        }
    )
    return metrics


def _benchmarks(
    market: pd.DataFrame,
    dynamic_weights: np.ndarray,
    cost_bps: int,
    spec: CashFloorBenchmarkRelativeSpec,
) -> dict[str, pd.DataFrame]:
    rows = len(market)
    average = dynamic_weights.mean(axis=0)
    weights = {
        "dynamic_cash_floor": dynamic_weights,
        "soxx_buyhold": np.tile([0.0, 1.0, 0.0, 0.0], (rows, 1)),
        "matched_static": np.tile(average, (rows, 1)),
        "no_cash_30soxl_70soxx": np.tile(
            [spec.risky_sleeve_soxl_share, 1.0 - spec.risky_sleeve_soxl_share, 0.0, 0.0],
            (rows, 1),
        ),
        "cash_only": np.tile([0.0, 0.0, 0.0, 1.0], (rows, 1)),
    }
    return {
        name: _simulate(market, value, cost_bps=cost_bps, name=name)
        for name, value in weights.items()
    }


def _evaluate_backtest(
    market: pd.DataFrame,
    predictions: pd.DataFrame,
    spec: CashFloorBenchmarkRelativeSpec,
):
    comparison_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    stored_returns: list[pd.DataFrame] = []
    positions: list[pd.DataFrame] = []
    for segment, bounds in (
        ("development_walk_forward", (f"{spec.development_folds[0]}-01-01", spec.development_end)),
        ("outer_diagnostic", spec.outer),
    ):
        segment_predictions = predictions.loc[predictions["segment"].eq(segment)]
        segment_market = market.loc[segment_predictions.index].copy()
        dynamic_weights = _target_weights(segment_predictions["risk_score"].to_numpy(), spec)
        for cost_bps in spec.costs_bps:
            results = _benchmarks(segment_market, dynamic_weights, cost_bps, spec)
            soxx = results["soxx_buyhold"]
            for strategy, result in results.items():
                comparison_rows.append(
                    {
                        "segment": segment,
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "selection_uses_outer_test": False,
                        **_result_metrics(result, soxx),
                    }
                )
                stored_returns.append(
                    result.assign(segment=segment, cost_bps=cost_bps)
                )
                positions.append(
                    result.loc[
                        :,
                        [
                            "signal_date",
                            "trade_date",
                            "exit_date",
                            "strategy",
                            "soxl_weight",
                            "soxx_weight",
                            "soxs_weight",
                            "cash_weight",
                            "turnover",
                        ],
                    ].assign(segment=segment, cost_bps=cost_bps)
                )
                years = pd.to_datetime(result["signal_date"]).dt.year
                for year in sorted(years.unique()):
                    mask = years.eq(year)
                    fold_result = result.loc[mask].reset_index(drop=True)
                    fold_soxx = soxx.loc[mask].reset_index(drop=True)
                    fold_rows.append(
                        {
                            "segment": segment,
                            "fold_year": int(year),
                            "strategy": strategy,
                            "cost_bps": cost_bps,
                            "selection_uses_outer_test": False,
                            **_result_metrics(fold_result, fold_soxx),
                        }
                    )
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(fold_rows),
        pd.concat(stored_returns, ignore_index=True),
        pd.concat(positions, ignore_index=True),
    )


def _bootstrap(
    strategy_returns: pd.DataFrame,
    spec: CashFloorBenchmarkRelativeSpec,
) -> tuple[dict[str, Any], pd.DataFrame]:
    development = strategy_returns.loc[
        strategy_returns["segment"].eq("development_walk_forward")
        & strategy_returns["cost_bps"].eq(10)
        & strategy_returns["strategy"].isin(["dynamic_cash_floor", "soxx_buyhold"])
    ]
    pivot = development.pivot(
        index="signal_date", columns="strategy", values="strategy_return"
    ).dropna()
    indices = stationary_block_bootstrap_indices(
        observation_count=len(pivot),
        path_length=len(pivot),
        path_count=spec.bootstrap_paths,
        expected_block_length=spec.bootstrap_block_days,
        seed=17,
    )
    strategy_log = np.log1p(pivot["dynamic_cash_floor"].to_numpy()[indices]).sum(axis=1)
    benchmark_log = np.log1p(pivot["soxx_buyhold"].to_numpy()[indices]).sum(axis=1)
    ratios = np.exp(strategy_log - benchmark_log)
    distribution = pd.DataFrame(
        {
            "path": np.arange(spec.bootstrap_paths),
            "strategy_terminal_wealth": np.exp(strategy_log),
            "soxx_terminal_wealth": np.exp(benchmark_log),
            "terminal_wealth_ratio": ratios,
        }
    )
    return (
        {
            "method": "stationary_block_bootstrap",
            "paths": spec.bootstrap_paths,
            "path_days": len(pivot),
            "expected_block_days": spec.bootstrap_block_days,
            "outperformance_probability": float((ratios > 1.0).mean()),
            "ratio_p05": float(np.quantile(ratios, 0.05)),
            "ratio_p10": float(np.quantile(ratios, 0.10)),
            "ratio_p50": float(np.quantile(ratios, 0.50)),
            "ratio_p90": float(np.quantile(ratios, 0.90)),
            "ratio_p95": float(np.quantile(ratios, 0.95)),
        },
        distribution,
    )


def _gate_evaluation(
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    development = comparison.loc[comparison["segment"].eq("development_walk_forward")]

    def row(strategy: str, cost: int) -> pd.Series:
        return development.loc[
            development["strategy"].eq(strategy) & development["cost_bps"].eq(cost)
        ].iloc[0]

    dynamic10 = row("dynamic_cash_floor", 10)
    dynamic25 = row("dynamic_cash_floor", 25)
    soxx10 = row("soxx_buyhold", 10)
    static10 = row("matched_static", 10)
    no_cash10 = row("no_cash_30soxl_70soxx", 10)
    annual = folds.loc[
        folds["segment"].eq("development_walk_forward")
        & folds["strategy"].eq("dynamic_cash_floor")
        & folds["cost_bps"].eq(10)
    ]
    annual_ratios = annual["terminal_wealth_ratio_vs_soxx"]
    gates = {
        "relative_wealth_10bps": bool(dynamic10["terminal_wealth_ratio_vs_soxx"] >= 1.10),
        "relative_wealth_25bps": bool(dynamic25["terminal_wealth_ratio_vs_soxx"] >= 1.05),
        "positive_fold_count": bool(annual_ratios.gt(1.0).sum() >= 5),
        "median_fold_ratio": bool(annual_ratios.median() > 1.0),
        "bootstrap_outperformance_probability": bool(
            bootstrap["outperformance_probability"] >= 0.60
        ),
        "minimum_daily_cash": bool(dynamic10["min_cash_weight_pct"] >= 10.0 - 1e-9),
        "average_cash": bool(dynamic10["avg_cash_weight_pct"] >= 15.0),
        "zero_cash_days": bool(dynamic10["zero_cash_days"] == 0),
        "timing_value_vs_matched_static": bool(
            dynamic10["terminal_wealth"] > static10["terminal_wealth"]
        ),
        "maxdd_vs_soxx": bool(
            dynamic10["max_drawdown_pct"] >= soxx10["max_drawdown_pct"] - 10.0
        ),
        "worst5_vs_soxx": bool(
            dynamic10["worst_rolling_5d_pct"] >= soxx10["worst_rolling_5d_pct"] - 5.0
        ),
        "maxdd_vs_matched_static": bool(
            dynamic10["max_drawdown_pct"] >= static10["max_drawdown_pct"] - 2.0
        ),
        "worst5_vs_matched_static": bool(
            dynamic10["worst_rolling_5d_pct"] >= static10["worst_rolling_5d_pct"] - 2.0
        ),
        "maxdd_improvement_vs_no_cash": bool(
            dynamic10["max_drawdown_pct"] - no_cash10["max_drawdown_pct"] >= 5.0
        ),
        "worst5_improvement_vs_no_cash": bool(
            dynamic10["worst_rolling_5d_pct"] - no_cash10["worst_rolling_5d_pct"] >= 3.0
        ),
    }
    relative_names = {
        "relative_wealth_10bps",
        "relative_wealth_25bps",
        "positive_fold_count",
        "median_fold_ratio",
        "bootstrap_outperformance_probability",
    }
    cash_names = {"minimum_daily_cash", "average_cash", "zero_cash_days"}
    all_pass = all(gates.values())
    if all_pass:
        decision = "research_candidate_requires_sealed_forward"
    elif not all(gates[name] for name in relative_names):
        decision = "needs_iteration_relative_wealth"
    elif not all(gates[name] for name in cash_names):
        decision = "blocked_inactive"
    else:
        decision = "needs_iteration_cash_timing"
    return {
        "gates": gates,
        "all_development_gates_pass": all_pass,
        "passed_gate_count": int(sum(gates.values())),
        "total_gate_count": len(gates),
        "promotion_decision": decision,
        "development_positive_fold_count_10bps": int(annual_ratios.gt(1.0).sum()),
        "development_fold_count": int(len(annual_ratios)),
        "development_median_fold_ratio_10bps": float(annual_ratios.median()),
        "selection_uses_outer_test": False,
    }


def _feature_coefficients(models: dict[str, Pipeline]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        coefficients = model.named_steps["ridge"].coef_
        for feature, coefficient in zip(ALL_FEATURES, coefficients):
            rows.append(
                {"model": model_name, "feature": feature, "scaled_coefficient": coefficient}
            )
    return pd.DataFrame(rows)


def _run_card(spec: CashFloorBenchmarkRelativeSpec) -> dict[str, Any]:
    if not spec.requirements_path.exists():
        raise FileNotFoundError("blocked: cash-floor requirements contract missing")
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, data_audit = _load_feature_frame(SoxsDefensiveHurdleSpec(root=spec.root))
    market = _add_targets(market)
    dataset, dataset_audit = _build_dataset(market, spec)
    predictions, calibrations, models = _walk_forward_predictions(dataset, market, spec)
    comparison, folds, strategy_returns, positions = _evaluate_backtest(
        market, predictions, spec
    )
    bootstrap, bootstrap_distribution = _bootstrap(strategy_returns, spec)
    gate_evaluation = _gate_evaluation(comparison, folds, bootstrap)
    coefficients = _feature_coefficients(models)

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as temporary:
        temp = Path(temporary)
        model_dir = temp / "models"
        model_dir.mkdir()
        for name, model in models.items():
            with (model_dir / f"{name}.pkl").open("wb") as handle:
                pickle.dump(model, handle)
        predictions.reset_index(names="signal_date").to_csv(temp / "predictions.csv", index=False)
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(temp / "strategy_returns.csv", index=False)
        bootstrap_distribution.to_csv(temp / "bootstrap_terminal_wealth.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                train_start=spec.train_start,
                development_folds=",".join(map(str, spec.development_folds)),
                outer=f"{spec.outer[0]}:{dataset_audit['segments']['outer'][1]}",
                ridge_alpha=spec.ridge_alpha,
                cash_bounds=f"{spec.minimum_cash}:{spec.maximum_cash}",
                risky_sleeve_soxl_share=spec.risky_sleeve_soxl_share,
                selection_uses_outer_test="false",
            )
            development10 = comparison.query(
                "segment == 'development_walk_forward' and strategy == 'dynamic_cash_floor' and cost_bps == 10"
            ).iloc[0]
            R.log_metrics(
                development_ratio_10bps=float(development10["terminal_wealth_ratio_vs_soxx"]),
                development_maxdd_10bps=float(development10["max_drawdown_pct"]),
                development_avg_cash_pct=float(development10["avg_cash_weight_pct"]),
                passed_gate_count=gate_evaluation["passed_gate_count"],
            )
            for artifact in [model_dir, *temp.glob("*.csv")]:
                recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            summary = {
                "experiment": spec.experiment,
                "card_type": "cash_floor_benchmark_relative_walk_forward_research_card",
                "primary_notebook": f"{spec.experiment}.ipynb",
                "requirements_contract": spec.requirements_path.resolve().as_posix(),
                "promotion_decision": gate_evaluation["promotion_decision"],
                "selection_uses_outer_test": False,
                "outer_test_use": "diagnostic_only_previously_observed",
                "data_audit": data_audit,
                "qlib_dataset": dataset_audit,
                "qlib_docs": [
                    "https://qlib.readthedocs.io/en/latest/component/data.html",
                    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                    "https://qlib.readthedocs.io/en/latest/component/strategy.html",
                ],
                "model": {
                    "family": "two_head_fixed_ridge",
                    "ridge_alpha": spec.ridge_alpha,
                    "features": list(ALL_FEATURES),
                    "targets": list(TARGETS),
                    "risk_score": "max(cash_advantage_5d_tail_score, future_vol_5d_tail_score)",
                    "calibration": {
                        "lower_quantile": spec.calibration_lower_quantile,
                        "upper_quantile": spec.calibration_upper_quantile,
                        "fit_scope": "training_fold_only",
                    },
                },
                "portfolio_contract": {
                    "tradable_assets": ["SOXL", "SOXX", "cash"],
                    "soxs_feature_only": True,
                    "minimum_cash": spec.minimum_cash,
                    "maximum_cash": spec.maximum_cash,
                    "risky_sleeve_soxl_share": spec.risky_sleeve_soxl_share,
                    "risky_sleeve_soxx_share": 1.0 - spec.risky_sleeve_soxl_share,
                    "maximum_total_soxl_weight": (1.0 - spec.minimum_cash)
                    * spec.risky_sleeve_soxl_share,
                    "cash_return": 0.0,
                    "timing": "signal T close; target T+1 open to T+2 open",
                    "costs_bps": list(spec.costs_bps),
                },
                "gate_evaluation": _json_ready(gate_evaluation),
                "bootstrap": _json_ready(bootstrap),
                "recorder_id": recorder_id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "sidecars": {
                    "model_selection": "model_selection.csv",
                    "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
                    "backtest_walk_forward_metrics": "backtest_walk_forward_metrics.csv",
                    "risk_analysis": "risk_analysis.csv",
                    "bootstrap_metrics": "bootstrap_metrics.csv",
                    "calibration_summary": "calibration_summary.csv",
                    "feature_coefficients": "feature_coefficients.csv",
                    "selected_model": "selected_model.json",
                    "analysis_summary": "analysis_summary.json",
                    "failure_modes": "failure_modes.md",
                },
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
                },
            }
            _write_card(
                card,
                _json_ready(summary),
                comparison,
                folds,
                calibrations,
                coefficients,
            )
            for filename in summary["sidecars"].values():
                artifact = card / filename
                if artifact.exists():
                    recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _write_card(
    card: Path,
    summary: dict[str, Any],
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
    calibrations: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    folds.to_csv(card / "backtest_walk_forward_metrics.csv", index=False)
    comparison.loc[
        :,
        [
            "segment",
            "strategy",
            "cost_bps",
            "return_pct",
            "annualized_vol_pct",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "avg_cash_weight_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    pd.DataFrame([summary["bootstrap"]]).to_csv(card / "bootstrap_metrics.csv", index=False)
    calibrations.to_csv(card / "calibration_summary.csv", index=False)
    coefficients.to_csv(card / "feature_coefficients.csv", index=False)
    gate = summary["gate_evaluation"]
    model_selection = pd.DataFrame(
        [
            {
                "model": summary["model"]["family"],
                "selected": True,
                "selection_policy": "fixed_by_requirements_no_candidate_search",
                "all_development_gates_pass": gate["all_development_gates_pass"],
                "passed_gate_count": gate["passed_gate_count"],
                "total_gate_count": gate["total_gate_count"],
                "promotion_decision": summary["promotion_decision"],
                "selection_uses_outer_test": False,
            }
        ]
    )
    model_selection.to_csv(card / "model_selection.csv", index=False)
    selected = {
        "model": summary["model"],
        "portfolio_contract": summary["portfolio_contract"],
        "selection_policy": "fixed_by_requirements_no_candidate_search",
        "selection_uses_outer_test": False,
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    development = comparison.query(
        "segment == 'development_walk_forward' and strategy == 'dynamic_cash_floor'"
    )
    outer = comparison.query("segment == 'outer_diagnostic' and strategy == 'dynamic_cash_floor'")
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "gate_evaluation": gate,
        "development_dynamic": _json_ready(development.to_dict(orient="records")),
        "outer_dynamic_diagnostic": _json_ready(outer.to_dict(orient="records")),
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (card / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    dynamic10 = development.loc[development["cost_bps"].eq(10)].iloc[0]
    dynamic25 = development.loc[development["cost_bps"].eq(25)].iloc[0]
    outer10 = outer.loc[outer["cost_bps"].eq(10)].iloc[0]
    passed = [name for name, value in gate["gates"].items() if value]
    failed = [name for name, value in gate["gates"].items() if not value]
    conclusion = f"""# 中文结论

## 跑了什么

按冻结的 cash-floor benchmark-relative 合同，使用 qlib SOXL/SOXX/SOXS provider、DatasetH/DataHandlerLP 和 recorder，训练两个固定 Ridge 头：五日现金优势与五日未来波动。2016–2022 逐年 expanding walk-forward，20 日 purge；风险资产固定 30% SOXL / 70% SOXX，模型只控制 10%–60% 现金。2023–2026 outer 仅诊断。

## 开发期结果

10bp：收益 {dynamic10['return_pct']:.2f}%，SOXX {dynamic10['soxx_return_pct']:.2f}%，相对财富 {dynamic10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD {dynamic10['max_drawdown_pct']:.2f}%，最差五日 {dynamic10['worst_rolling_5d_pct']:.2f}%，平均现金 {dynamic10['avg_cash_weight_pct']:.2f}%。25bp 相对财富 {dynamic25['terminal_wealth_ratio_vs_soxx']:.4f}。正相对收益年份 {gate['development_positive_fold_count_10bps']}/{gate['development_fold_count']}，年度中位财富比 {gate['development_median_fold_ratio_10bps']:.4f}，bootstrap 跑赢概率 {summary['bootstrap']['outperformance_probability']:.2%}。

通过门：`{passed}`。失败门：`{failed}`。

## Outer 诊断

10bp：收益 {outer10['return_pct']:.2f}%，SOXX {outer10['soxx_return_pct']:.2f}%，相对财富 {outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD {outer10['max_drawdown_pct']:.2f}%。outer 不参与模型、映射或阈值选择。

## 决定、风险和下一步

`{summary['promotion_decision']}`。双头 max 映射改善了尾部风险，但平均现金过高且换手成本显著，未达到相对 SOXX 的主目标。不能用 outer 回调；下一步若继续，应在开发期内部重新设计现金聚合函数，例如让波动头限制现金上限或采用平滑/滞后，而不是改变固定风险袖套后宣称通过。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        """# Failure modes

- Taking the maximum of two tail scores can keep cash chronically high and create cash drag.
- Daily continuous target changes can create high turnover even when the risky-sleeve mix is fixed.
- The volatility head can improve drawdown while still reducing relative terminal wealth below SOXX.
- Cash earns zero in v1; a PIT T-bill series could change the drag but cannot be backfilled with a current yield.
- 2023-2026 was previously observed and is diagnostic only.
""",
        encoding="utf-8",
    )
    workflow = f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
requirements_contract: {summary['requirements_contract']}
provider_uri: {summary['data_audit']['provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
qlib:
  market: all
  region: us
  dataset: [DatasetH, DataHandlerLP, StaticDataLoader]
  recorder: qlib.workflow.R
  risk_analyzer: qlib.contrib.evaluate.risk_analysis
walk_forward:
  train_start: 2011-01-03
  development_folds: [2016, 2017, 2018, 2019, 2020, 2021, 2022]
  purge_rows: 20
  outer: 2023-2026_diagnostic_only
  selection_uses_outer_test: false
model:
  family: fixed_two_head_ridge
  targets: [cash_advantage_5d, future_vol_5d]
  ridge_alpha: 10.0
portfolio:
  cash_bounds: [0.10, 0.60]
  risky_sleeve: {{SOXL: 0.30, SOXX: 0.70}}
  costs_bps: [10, 25]
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/models/
    - artifacts/{summary['experiment']}/predictions.csv
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
    - artifacts/{summary['experiment']}/bootstrap_terminal_wealth.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_cash_floor_benchmark_relative_card(), indent=2))
