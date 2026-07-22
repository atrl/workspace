from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from src.model.relative_wealth_lfnn import (
    RelativeWealthPolicy,
    RelativeWealthTrainingSpec,
    simulate_policy_paths,
    train_relative_wealth_policy,
)
from src.objective.relative_wealth import (
    aggregate_bootstrap_paths,
    stationary_block_bootstrap_indices,
    terminal_relative_wealth_metrics,
)
from src.workflow.leveraged_etf_paper_replay import (
    LeveragedEtfPaperReplaySpec,
    _load_market_frame,
    _risk_metrics,
    _simulate_weights,
)
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "relative_wealth_lfnn_soxx_v1"
CARD_TYPE = "relative_wealth_lfnn_soxx_research_card"
MARKET_FEATURES = ("soxx_trend60", "soxx_vol20", "relative_momentum20", "soxx_drawdown60")
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/training_history.csv",
    f"artifacts/{EXPERIMENT}/bootstrap_terminal_wealth.csv",
    f"artifacts/{EXPERIMENT}/diagnostic_seed0_terminal_wealth.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
)


@dataclass(frozen=True)
class RelativeWealthLfnnSoxxSpec:
    root: Path
    experiment: str = EXPERIMENT
    train: tuple[str, str] = ("2011-01-03", "2019-12-31")
    valid: tuple[str, str] = ("2020-01-02", "2022-12-30")
    outer: tuple[str, str] = ("2023-01-03", "2026-12-31")
    annual_excess_targets: tuple[float, ...] = (0.02, 0.04, 0.06)
    variants: tuple[str, ...] = ("paper_minimal", "market_state_augmented")
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    horizon_days: int = 756
    horizon_years: float = 3.0
    rebalance_days: int = 63
    expected_block_length: int = 63
    epochs: int = 25
    paths_per_epoch: int = 2048
    validation_paths: int = 4096
    hidden_size: int = 32
    learning_rate: float = 0.003
    training_cost_bps: int = 10
    stress_cost_bps: int = 25

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def run_relative_wealth_lfnn_soxx_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(RelativeWealthLfnnSoxxSpec(root=repo_root))


def _build_market_states(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy().sort_values("signal_date").reset_index(drop=True)
    frame["soxx_close_return"] = frame["soxx_close"].pct_change()
    frame["soxx_trend60"] = frame["soxx_close"] / frame["soxx_close"].rolling(60).mean() - 1.0
    frame["soxx_vol20"] = frame["soxx_close_return"].rolling(20).std(ddof=1)
    frame["relative_momentum20"] = frame["soxl_close"].pct_change(20) - frame["soxx_close"].pct_change(20)
    frame["soxx_drawdown60"] = frame["soxx_close"] / frame["soxx_close"].rolling(60).max() - 1.0
    return frame.dropna(subset=[*MARKET_FEATURES, "soxl_return", "soxx_return"]).reset_index(drop=True)


def _build_qlib_dataset(frame: pd.DataFrame, spec: RelativeWealthLfnnSoxxSpec):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    features = frame.loc[:, MARKET_FEATURES].copy()
    labels = frame.loc[:, ["soxl_return", "soxx_return"]].rename(
        columns={"soxl_return": "soxl_forward_open_return", "soxx_return": "soxx_forward_open_return"}
    )
    static = pd.concat({"feature": features, "label": labels}, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(frame["signal_date"]), np.full(len(frame), "SOXL_SOXX_PAIR")],
        names=["datetime", "instrument"],
    )
    segments = {
        "train": spec.train,
        "valid": spec.valid,
        "test": (spec.outer[0], min(spec.outer[1], frame["signal_date"].max().date().isoformat())),
    }
    handler = DataHandlerLP(
        instruments=None,
        start_time=frame["signal_date"].min(),
        end_time=frame["signal_date"].max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    audit = {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {name: int(len(dataset.prepare(name))) for name in segments},
        "feature_columns": list(MARKET_FEATURES),
        "label_columns": ["soxl_forward_open_return", "soxx_forward_open_return"],
    }
    return dataset, audit


def _dataset_segment(dataset, segment: str) -> pd.DataFrame:
    prepared = dataset.prepare(segment, col_set=["feature", "label"])
    frame = prepared.droplevel("instrument").sort_index().copy()
    frame.columns = [str(column[-1]) for column in frame.columns]
    return frame


def _standardize_features(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, dict[str, float]]]:
    mean = train.loc[:, MARKET_FEATURES].mean()
    std = train.loc[:, MARKET_FEATURES].std(ddof=1).replace(0.0, 1.0)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        normalized = ((result.loc[:, MARKET_FEATURES].astype(float) - mean) / std).clip(-5.0, 5.0)
        for column in MARKET_FEATURES:
            result[column] = normalized[column].to_numpy(dtype=float)
        return result

    audit = {
        column: {"train_mean": float(mean[column]), "train_std": float(std[column])} for column in MARKET_FEATURES
    }
    return transform(train), [transform(frame) for frame in others], audit


def _arrays(frame: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    returns = frame.loc[:, ["soxl_forward_open_return", "soxx_forward_open_return"]].to_numpy(dtype=float)
    if variant == "paper_minimal":
        features = np.empty((len(frame), 0), dtype=float)
    elif variant == "market_state_augmented":
        features = frame.loc[:, MARKET_FEATURES].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown LFNN variant: {variant}")
    return returns, features


def _bootstrap_paths(
    returns: np.ndarray,
    features: np.ndarray,
    *,
    spec: RelativeWealthLfnnSoxxSpec,
    path_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = stationary_block_bootstrap_indices(
        observation_count=len(returns),
        path_length=spec.horizon_days,
        path_count=path_count,
        expected_block_length=spec.expected_block_length,
        seed=seed,
    )
    return aggregate_bootstrap_paths(
        returns,
        features,
        indices,
        rebalance_days=spec.rebalance_days,
    )


def _evaluate_policies_on_paths(
    policies: Sequence[RelativeWealthPolicy],
    returns: np.ndarray,
    features: np.ndarray,
    *,
    cost_bps: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    for policy in policies:
        policy.eval()
    with torch.no_grad():
        result = simulate_policy_paths(
            policies,
            torch.as_tensor(returns, dtype=torch.float32),
            torch.as_tensor(features, dtype=torch.float32),
            cost_bps=cost_bps,
        )
    strategy = result["strategy_wealth"].numpy()
    benchmark = result["benchmark_wealth"].numpy()
    metrics = terminal_relative_wealth_metrics(strategy, benchmark)
    weights = result["weights"].numpy()
    metrics.update(
        {
            "cost_bps": int(cost_bps),
            "avg_soxl_weight": float(weights[:, :, 0].mean()),
            "avg_soxx_weight": float(weights[:, :, 1].mean()),
            "avg_cash_weight": float(weights[:, :, 2].mean()),
            "avg_active_exposure": float(np.abs(3.0 * weights[:, :, 0] + weights[:, :, 1] - 1.0).mean()),
            "avg_turnover": float(result["turnover"].sum(dim=1).mean()),
        }
    )
    distribution = pd.DataFrame(
        {
            "strategy_terminal_wealth": strategy,
            "benchmark_terminal_wealth": benchmark,
            "terminal_wealth_ratio": strategy / benchmark,
            "cost_bps": int(cost_bps),
        }
    )
    return metrics, distribution


def _train_candidates(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    spec: RelativeWealthLfnnSoxxSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, float, int], RelativeWealthPolicy], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    policies: dict[tuple[str, float, int], RelativeWealthPolicy] = {}
    common_distributions: list[pd.DataFrame] = []
    for variant in spec.variants:
        train_returns, train_features = _arrays(train, variant)
        valid_returns, valid_features = _arrays(valid, variant)
        common_validation = _bootstrap_paths(
            valid_returns,
            valid_features,
            spec=spec,
            path_count=spec.validation_paths,
            seed=90_001,
        )
        for seed in spec.seeds:
            batches = [
                _bootstrap_paths(
                    train_returns,
                    train_features,
                    spec=spec,
                    path_count=spec.paths_per_epoch,
                    seed=seed * 10_000 + epoch,
                )
                for epoch in range(spec.epochs)
            ]
            seeded_validation = _bootstrap_paths(
                valid_returns,
                valid_features,
                spec=spec,
                path_count=spec.validation_paths,
                seed=seed * 100 + 7,
            )
            for target in spec.annual_excess_targets:
                training_spec = RelativeWealthTrainingSpec(
                    annual_excess_target=target,
                    horizon_years=spec.horizon_years,
                    cost_bps=spec.training_cost_bps,
                    hidden_size=spec.hidden_size,
                    epochs=spec.epochs,
                    paths_per_epoch=spec.paths_per_epoch,
                    learning_rate=spec.learning_rate,
                )
                policy, training_metrics = train_relative_wealth_policy(
                    training_batches=batches,
                    validation_returns=seeded_validation[0],
                    validation_features=seeded_validation[1],
                    spec=training_spec,
                    seed=seed,
                )
                key = (variant, target, seed)
                policies[key] = policy
                history = pd.DataFrame(training_metrics.pop("history"))
                history["variant"] = variant
                history["annual_excess_target"] = target
                history["seed"] = seed
                histories.append(history)
                for cost_bps in (spec.training_cost_bps, spec.stress_cost_bps):
                    metrics, distribution = _evaluate_policies_on_paths(
                        [policy], common_validation[0], common_validation[1], cost_bps=cost_bps
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "annual_excess_target": target,
                            "seed": seed,
                            **{key: value for key, value in training_metrics.items() if key != "history"},
                            **metrics,
                            "selection_uses_outer_test": False,
                        }
                    )
                    if seed == spec.seeds[0]:
                        distribution["variant"] = variant
                        distribution["annual_excess_target"] = target
                        distribution["seed"] = seed
                        common_distributions.append(distribution)
    return (
        pd.DataFrame(rows),
        pd.concat(histories, ignore_index=True),
        policies,
        pd.concat(common_distributions, ignore_index=True),
    )


def _select_group(seed_metrics: pd.DataFrame, spec: RelativeWealthLfnnSoxxSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregates = (
        seed_metrics.groupby(["variant", "annual_excess_target", "cost_bps"], as_index=False)
        .agg(
            median_terminal_wealth_ratio=("median_terminal_wealth_ratio", "median"),
            outperformance_probability=("outperformance_probability", "median"),
            active_information_ratio=("active_information_ratio", "median"),
            avg_active_exposure=("avg_active_exposure", "median"),
            avg_soxl_weight=("avg_soxl_weight", "median"),
            positive_seed_count=("median_terminal_wealth_ratio", lambda values: int((values > 1.0).sum())),
        )
    )
    pivot = aggregates.pivot(index=["variant", "annual_excess_target"], columns="cost_bps")
    group_rows: list[dict[str, Any]] = []
    for variant, target in pivot.index:
        ten = aggregates.loc[
            aggregates["variant"].eq(variant)
            & aggregates["annual_excess_target"].eq(target)
            & aggregates["cost_bps"].eq(spec.training_cost_bps)
        ].iloc[0]
        stress = aggregates.loc[
            aggregates["variant"].eq(variant)
            & aggregates["annual_excess_target"].eq(target)
            & aggregates["cost_bps"].eq(spec.stress_cost_bps)
        ].iloc[0]
        gate = bool(
            ten["median_terminal_wealth_ratio"] > 1.0
            and stress["median_terminal_wealth_ratio"] > 1.0
            and ten["outperformance_probability"] >= 0.55
            and stress["outperformance_probability"] >= 0.525
            and ten["active_information_ratio"] >= 0.25
            and stress["active_information_ratio"] > 0.0
            and ten["positive_seed_count"] >= 4
            and stress["positive_seed_count"] >= 3
            and ten["avg_active_exposure"] >= 0.10
        )
        score = float(
            ten["active_information_ratio"]
            + stress["active_information_ratio"]
            + 5.0 * (ten["median_terminal_wealth_ratio"] - 1.0)
            + 2.0 * (ten["outperformance_probability"] - 0.5)
        )
        group_rows.append(
            {
                "variant": variant,
                "annual_excess_target": target,
                "validation_gate_pass": gate,
                "selection_score": score,
                "median_ratio_10bps": ten["median_terminal_wealth_ratio"],
                "outperformance_probability_10bps": ten["outperformance_probability"],
                "active_ir_10bps": ten["active_information_ratio"],
                "positive_seeds_10bps": int(ten["positive_seed_count"]),
                "median_ratio_25bps": stress["median_terminal_wealth_ratio"],
                "outperformance_probability_25bps": stress["outperformance_probability"],
                "active_ir_25bps": stress["active_information_ratio"],
                "positive_seeds_25bps": int(stress["positive_seed_count"]),
                "avg_active_exposure_10bps": ten["avg_active_exposure"],
                "avg_soxl_weight_10bps": ten["avg_soxl_weight"],
                "selection_uses_outer_test": False,
            }
        )
    groups = pd.DataFrame(group_rows).sort_values(
        ["validation_gate_pass", "selection_score"], ascending=[False, False]
    )
    selected = groups.iloc[0].to_dict()
    selected["status"] = "validation_pass" if selected["validation_gate_pass"] else "diagnostic_fallback"
    return groups.reset_index(drop=True), selected


def _simulate_historical_ensemble(
    policies: Sequence[RelativeWealthPolicy],
    frame: pd.DataFrame,
    *,
    variant: str,
    cost_bps: int,
    rebalance_days: int,
) -> pd.DataFrame:
    for policy in policies:
        policy.eval()
    wealth = 1.0
    benchmark_wealth = 1.0
    weights = np.array([0.0, 0.0, 1.0], dtype=float)
    rows: list[dict[str, Any]] = []
    step_count = int(np.ceil(len(frame) / rebalance_days))
    for index, row in enumerate(frame.itertuples()):
        before = wealth
        benchmark_before = benchmark_wealth
        turnover = 0.0
        if index % rebalance_days == 0:
            step = index // rebalance_days
            feature_values = (
                np.empty(0, dtype=np.float32)
                if variant == "paper_minimal"
                else np.asarray([getattr(row, column) for column in MARKET_FEATURES], dtype=np.float32)
            )
            state = np.concatenate(
                [
                    [step / max(step_count - 1, 1), np.log(wealth), np.log(benchmark_wealth)],
                    [np.log(wealth / benchmark_wealth), wealth / benchmark_wealth - 1.0],
                    weights,
                    feature_values,
                ]
            ).astype(np.float32)
            with torch.no_grad():
                target = np.mean(
                    [policy(torch.from_numpy(state).unsqueeze(0)).numpy()[0] for policy in policies], axis=0
                )
            turnover = float(0.5 * np.abs(target - weights).sum())
            wealth *= 1.0 - turnover * float(cost_bps) / 10_000.0
            weights = target
        if index == 0:
            benchmark_wealth *= 1.0 - float(cost_bps) / 10_000.0
        asset_gross = np.array([1.0 + row.soxl_forward_open_return, 1.0 + row.soxx_forward_open_return, 1.0])
        portfolio_gross = float(np.dot(weights, asset_gross))
        wealth *= portfolio_gross
        benchmark_wealth *= 1.0 + row.soxx_forward_open_return
        target_weights = weights.copy()
        weights = weights * asset_gross / max(portfolio_gross, 1e-12)
        rows.append(
            {
                "signal_date": row.Index,
                "trade_date": row.trade_date,
                "exit_date": row.exit_date,
                "soxl_return": row.soxl_forward_open_return,
                "soxx_return": row.soxx_forward_open_return,
                "strategy_return": wealth / before - 1.0,
                "benchmark_return": benchmark_wealth / benchmark_before - 1.0,
                "soxl_weight": target_weights[0],
                "soxx_weight": target_weights[1],
                "cash_weight": target_weights[2],
                "effective_beta": 3.0 * target_weights[0] + target_weights[1],
                "turnover": turnover,
                "strategy_wealth": wealth,
                "benchmark_wealth": benchmark_wealth,
            }
        )
    return pd.DataFrame(rows)


def _outer_evaluation(
    selected_policies: Sequence[RelativeWealthPolicy],
    outer: pd.DataFrame,
    *,
    selected_variant: str,
    spec: RelativeWealthLfnnSoxxSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    return_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    raw_frame = pd.DataFrame(
        {
            "signal_date": outer.index,
            "trade_date": outer["trade_date"].to_numpy(),
            "exit_date": outer["exit_date"].to_numpy(),
            "soxl_return": outer["soxl_forward_open_return"].to_numpy(),
            "soxx_return": outer["soxx_forward_open_return"].to_numpy(),
        }
    )
    for cost_bps in (spec.training_cost_bps, spec.stress_cost_bps):
        strategies = {
            "relative_wealth_lfnn_ensemble": _simulate_historical_ensemble(
                selected_policies,
                outer,
                variant=selected_variant,
                cost_bps=cost_bps,
                rebalance_days=spec.rebalance_days,
            ),
            "soxx_buyhold": _simulate_weights(
                raw_frame,
                soxl_weight=np.zeros(len(raw_frame)),
                soxx_weight=np.ones(len(raw_frame)),
                cost_bps=cost_bps,
            ),
            "one_third_soxl_cash": _simulate_weights(
                raw_frame,
                soxl_weight=np.full(len(raw_frame), 1.0 / 3.0),
                soxx_weight=np.zeros(len(raw_frame)),
                cost_bps=cost_bps,
            ),
            "half_soxl_cash": _simulate_weights(
                raw_frame,
                soxl_weight=np.full(len(raw_frame), 0.5),
                soxx_weight=np.zeros(len(raw_frame)),
                cost_bps=cost_bps,
            ),
        }
        benchmark = strategies["soxx_buyhold"]
        benchmark_wealth = float((1.0 + benchmark["strategy_return"]).prod())
        for strategy, result in strategies.items():
            metrics = _risk_metrics(result)
            strategy_wealth = float((1.0 + result["strategy_return"]).prod())
            active = result["strategy_return"].to_numpy() - benchmark["strategy_return"].to_numpy()
            active_std = float(active.std(ddof=1))
            active_ir = float(active.mean() / active_std * np.sqrt(252.0)) if active_std > 0.0 else 0.0
            comparison_rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost_bps,
                    **metrics,
                    "soxx_benchmark_return_pct": (benchmark_wealth - 1.0) * 100.0,
                    "terminal_wealth_ratio_vs_soxx": strategy_wealth / benchmark_wealth,
                    "active_information_ratio_daily": active_ir,
                    "beats_soxx": bool(strategy_wealth > benchmark_wealth),
                }
            )
            stored = result.copy()
            stored["strategy"] = strategy
            stored["cost_bps"] = cost_bps
            return_frames.append(stored)
            position_frames.append(
                stored.loc[
                    :,
                    [
                        "signal_date",
                        "trade_date",
                        "exit_date",
                        "strategy",
                        "cost_bps",
                        "soxl_weight",
                        "soxx_weight",
                        "cash_weight",
                    ],
                ]
            )
            for year in sorted(pd.to_datetime(stored["signal_date"]).dt.year.unique()):
                mask = pd.to_datetime(stored["signal_date"]).dt.year.eq(year)
                year_result = stored.loc[mask]
                year_benchmark = benchmark.loc[mask]
                annual_rows.append(
                    {
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "outer_year": int(year),
                        **_risk_metrics(year_result),
                        "soxx_return_pct": float(((1.0 + year_benchmark["strategy_return"]).prod() - 1.0) * 100.0),
                    }
                )
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(annual_rows),
        pd.concat(position_frames, ignore_index=True),
        pd.concat(return_frames, ignore_index=True),
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


def _run_card(spec: RelativeWealthLfnnSoxxSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, market_audit = _load_market_frame(LeveragedEtfPaperReplaySpec(root=spec.root))
    market = _build_market_states(market)
    dataset, dataset_audit = _build_qlib_dataset(market, spec)
    train = _dataset_segment(dataset, "train")
    valid = _dataset_segment(dataset, "valid")
    outer = _dataset_segment(dataset, "test")
    train, [valid, outer], scaler_audit = _standardize_features(train, valid, outer)
    execution_dates = market.set_index(pd.to_datetime(market["signal_date"]))[["trade_date", "exit_date"]]
    outer = outer.join(execution_dates, how="left")
    if outer[["trade_date", "exit_date"]].isna().any().any():
        raise ValueError("outer rows must retain next-open trade and exit dates")

    seed_metrics, training_history, policies, diagnostic_distributions = _train_candidates(train, valid, spec)
    model_selection, selected = _select_group(seed_metrics, spec)
    selected_variant = str(selected["variant"])
    selected_target = float(selected["annual_excess_target"])
    selected_policies = [policies[(selected_variant, selected_target, seed)] for seed in spec.seeds]

    valid_returns, valid_features = _arrays(valid, selected_variant)
    common_validation = _bootstrap_paths(
        valid_returns,
        valid_features,
        spec=spec,
        path_count=spec.validation_paths,
        seed=90_001,
    )
    bootstrap_rows: list[dict[str, Any]] = []
    selected_distributions: list[pd.DataFrame] = []
    for cost_bps in (spec.training_cost_bps, spec.stress_cost_bps):
        metrics, distribution = _evaluate_policies_on_paths(
            selected_policies,
            common_validation[0],
            common_validation[1],
            cost_bps=cost_bps,
        )
        bootstrap_rows.append(
            {
                "variant": selected_variant,
                "annual_excess_target": selected_target,
                **metrics,
                "selection_uses_outer_test": False,
            }
        )
        distribution["variant"] = selected_variant
        distribution["annual_excess_target"] = selected_target
        selected_distributions.append(distribution)
    bootstrap_metrics = pd.DataFrame(bootstrap_rows)
    bootstrap_distribution = pd.concat(selected_distributions, ignore_index=True)
    comparison, annual, positions, strategy_returns = _outer_evaluation(
        selected_policies,
        outer,
        selected_variant=selected_variant,
        spec=spec,
    )
    selected_outer = comparison.loc[
        comparison["strategy"].eq("relative_wealth_lfnn_ensemble")
        & comparison["cost_bps"].eq(spec.training_cost_bps)
    ].iloc[0].to_dict()
    validation_pass = bool(selected["validation_gate_pass"])
    promotion_decision = (
        "research_diagnostic_validation_pass_outer_previously_observed_requires_sealed_forward"
        if validation_pass
        else "research_reject_relative_wealth_validation_distribution_gates_failed"
    )

    sidecars = {
        "model_selection": "model_selection.csv",
        "seed_metrics": "seed_metrics.csv",
        "bootstrap_metrics": "bootstrap_metrics.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "annual_outer_metrics": "annual_outer_metrics.csv",
        "risk_analysis": "risk_analysis.csv",
        "analysis_summary": "analysis_summary.json",
        "selected_model": "selected_model.json",
        "failure_modes": "failure_modes.md",
    }

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        model_dir = tmp_dir / "models"
        model_dir.mkdir()
        for seed, policy in zip(spec.seeds, selected_policies, strict=True):
            torch.save(policy.state_dict(), model_dir / f"{selected_variant}_target{selected_target:.2f}_seed{seed}.pt")
        training_history.to_csv(tmp_dir / "training_history.csv", index=False)
        bootstrap_distribution.to_csv(tmp_dir / "bootstrap_terminal_wealth.csv", index=False)
        diagnostic_distributions.to_csv(tmp_dir / "diagnostic_seed0_terminal_wealth.csv", index=False)
        positions.to_csv(tmp_dir / "positions.csv", index=False)
        strategy_returns.to_csv(tmp_dir / "strategy_returns.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                provider_uri=spec.provider_uri.resolve().as_posix(),
                train=f"{spec.train[0]}:{spec.train[1]}",
                valid=f"{spec.valid[0]}:{spec.valid[1]}",
                outer=f"{spec.outer[0]}:{dataset_audit['segments']['test'][1]}",
                selection_uses_outer_test="false",
                variants=",".join(spec.variants),
                annual_excess_targets=",".join(str(value) for value in spec.annual_excess_targets),
                seeds=",".join(str(seed) for seed in spec.seeds),
                horizon_days=spec.horizon_days,
                rebalance_days=spec.rebalance_days,
                expected_block_length=spec.expected_block_length,
                paths_per_candidate=spec.epochs * spec.paths_per_epoch,
                selected_variant=selected_variant,
                selected_target=selected_target,
            )
            R.log_metrics(
                validation_gate_pass=float(validation_pass),
                selected_validation_active_ir_10bps=float(
                    bootstrap_metrics.loc[bootstrap_metrics["cost_bps"].eq(10), "active_information_ratio"].iloc[0]
                ),
                selected_validation_outperformance_probability_10bps=float(
                    bootstrap_metrics.loc[bootstrap_metrics["cost_bps"].eq(10), "outperformance_probability"].iloc[0]
                ),
                outer_return_pct=float(selected_outer["return_pct"]),
                outer_terminal_wealth_ratio_vs_soxx=float(selected_outer["terminal_wealth_ratio_vs_soxx"]),
            )
            for artifact in [model_dir, *tmp_dir.glob("*.csv")]:
                recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()

            summary = {
                "experiment": spec.experiment,
                "card_type": CARD_TYPE,
                "primary_notebook": f"{spec.experiment}.ipynb",
                "promotion_decision": promotion_decision,
                "selection_uses_outer_test": False,
                "outer_test_use": "diagnostic_only_previously_observed_requires_new_sealed_forward",
                "provider_uri": spec.provider_uri.resolve().as_posix(),
                "data_audit": market_audit,
                "qlib_dataset": dataset_audit,
                "qlib_native_api": [
                    "qlib.init",
                    "qlib.data.D.features",
                    "DatasetH",
                    "DataHandlerLP",
                    "qlib.workflow.R.start",
                    "qlib.contrib.evaluate.risk_analysis",
                ],
                "qlib_docs": [
                    "https://qlib.readthedocs.io/en/latest/component/data.html",
                    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                    "https://qlib.readthedocs.io/en/latest/component/strategy.html",
                ],
                "objective": {
                    "name": "target_relative_terminal_wealth_mse",
                    "formula": "mean((W_strategy_T/W_SOXX_T-(1+g)^horizon_years)^2)",
                    "benchmark": "100% SOXX buyhold",
                    "target_grid": list(spec.annual_excess_targets),
                    "selection_metric": "validation terminal active information ratio plus distribution gates",
                    "max_drawdown_in_training_loss": False,
                },
                "model": {
                    "class": "RelativeWealthPolicy",
                    "method": "global_in_time_leverage_feasible_neural_network_not_reinforcement_learning",
                    "variants": list(spec.variants),
                    "hidden_size": spec.hidden_size,
                    "output": ["SOXL", "SOXX", "cash"],
                    "constraint": "long_only_softmax_weights_sum_to_one_no_borrowing",
                    "seeds": list(spec.seeds),
                },
                "bootstrap": {
                    "method": "stationary_block_bootstrap",
                    "horizon_days": spec.horizon_days,
                    "horizon_years": spec.horizon_years,
                    "expected_block_length": spec.expected_block_length,
                    "rebalance_days": spec.rebalance_days,
                    "paths_per_candidate": spec.epochs * spec.paths_per_epoch,
                    "validation_paths": spec.validation_paths,
                },
                "feature_scaler": scaler_audit,
                "validation_gates": {
                    "median_ratio_10bps_min_exclusive": 1.0,
                    "median_ratio_25bps_min_exclusive": 1.0,
                    "outperformance_probability_10bps_min": 0.55,
                    "outperformance_probability_25bps_min": 0.525,
                    "active_ir_10bps_min": 0.25,
                    "active_ir_25bps_min_exclusive": 0.0,
                    "positive_seeds_10bps_min": 4,
                    "positive_seeds_25bps_min": 3,
                    "avg_active_exposure_10bps_min": 0.10,
                },
                "selected_model": _json_ready(selected),
                "selected_bootstrap_metrics": _json_ready(bootstrap_metrics.to_dict(orient="records")),
                "selected_outer_metrics_10bps": _json_ready(selected_outer),
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
            _write_card_files(card, summary, model_selection, seed_metrics, bootstrap_metrics, comparison, annual)
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _write_card_files(
    card: Path,
    summary: dict[str, Any],
    model_selection: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
) -> None:
    model_selection.to_csv(card / "model_selection.csv", index=False)
    seed_metrics.to_csv(card / "seed_metrics.csv", index=False)
    bootstrap_metrics.to_csv(card / "bootstrap_metrics.csv", index=False)
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual.to_csv(card / "annual_outer_metrics.csv", index=False)
    comparison.loc[
        :,
        [
            "strategy",
            "cost_bps",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "annualized_vol_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    selected = summary["selected_model"]
    outer = summary["selected_outer_metrics_10bps"]
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "selected_model": selected,
        "selected_bootstrap_metrics": summary["selected_bootstrap_metrics"],
        "selected_outer_metrics_10bps": outer,
    }
    (card / "analysis_summary.json").write_text(json.dumps(_json_ready(analysis), indent=2) + "\n", encoding="utf-8")
    (card / "selected_model.json").write_text(json.dumps(_json_ready(selected), indent=2) + "\n", encoding="utf-8")
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8")
    bootstrap10 = next(row for row in summary["selected_bootstrap_metrics"] if row["cost_bps"] == 10)
    bootstrap25 = next(row for row in summary["selected_bootstrap_metrics"] if row["cost_bps"] == 25)
    conclusion = f"""# 中文结论

## 跑了什么

用 qlib DatasetH/DataHandlerLP 读取 SOXL/SOXX 因果状态与 next-open returns，在 2011–2019 上训练 paper-inspired global-in-time feasible network。模型直接输出 SOXL/SOXX/cash long-only 权重，以 100% SOXX 为基准，按 63 交易日调仓，并在 stationary block bootstrap 三年路径上优化相对终值目标。2020–2022 只用于 validation；2023–2026-07 只做已经看过的诊断 outer。

## Bootstrap validation

选中 `{selected['variant']}`、年化目标 {100.0 * selected['annual_excess_target']:.1f}%。10bp：终值财富比中位数 {bootstrap10['median_terminal_wealth_ratio']:.4f}，跑赢概率 {100.0 * bootstrap10['outperformance_probability']:.2f}%，Active IR {bootstrap10['active_information_ratio']:.4f}。25bp：中位数 {bootstrap25['median_terminal_wealth_ratio']:.4f}，跑赢概率 {100.0 * bootstrap25['outperformance_probability']:.2f}%，Active IR {bootstrap25['active_information_ratio']:.4f}。validation gate={selected['validation_gate_pass']}。

## Outer诊断

10bp 策略收益 {outer['return_pct']:.2f}%，SOXX {outer['soxx_benchmark_return_pct']:.2f}%，终值财富比 {outer['terminal_wealth_ratio_vs_soxx']:.4f}，Active IR {outer['active_information_ratio_daily']:.4f}，MaxDD {outer['max_drawdown_pct']:.2f}%。

## 决定

`{summary['promotion_decision']}`。无论 outer 是否好看，这个架构是在 outer 已被观察后提出，不能据此晋升；需要 2026-07-21 之后 sealed forward。当前 cash 收益按 0，且 SOXL/SOXX 是集中半导体映射，不等同论文宽基 LETF+债券。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        """# Failure modes

- Actual SOXL history is short relative to the paper; the primary horizon is three years rather than ten.
- Cash has zero return because PIT T-bill/T-bond data is not present in the current qlib store.
- SOXL/SOXX is a concentrated semiconductor mapping, not the paper's broad-market LETF/VETF setting.
- Bootstrap paths reuse a limited return history and cannot create unseen structural regimes.
- The 2023-2026 outer path was already observed before this architecture was designed and is diagnostic only.
""",
        encoding="utf-8",
    )
    workflow = f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
provider_uri: {summary['provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
qlib:
  market: all
  region: us
  dataset: [DatasetH, DataHandlerLP]
  recorder: qlib.workflow.R
segments:
  train: [2011-01-03, 2019-12-31]
  valid: [2020-01-02, 2022-12-30]
  outer: [2023-01-03, {summary['qlib_dataset']['segments']['test'][1]}]
selection:
  selection_uses_outer_test: false
  objective: target_relative_terminal_wealth_mse
  benchmark: SOXX
  costs_bps: [10, 25]
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/models/
    - artifacts/{summary['experiment']}/training_history.csv
    - artifacts/{summary['experiment']}/bootstrap_terminal_wealth.csv
    - artifacts/{summary['experiment']}/diagnostic_seed0_terminal_wealth.csv
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
"""
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")


if __name__ == "__main__":
    result = run_relative_wealth_lfnn_soxx_card()
    print(json.dumps(result, indent=2))
