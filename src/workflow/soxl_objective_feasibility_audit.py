from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.objective.relative_wealth import stationary_block_bootstrap_indices
from src.workflow.hierarchical_four_asset_policy import _benchmark_efficiency
from src.workflow.leveraged_etf_paper_replay import _risk_metrics
from src.workflow.soxs_defensive_hurdle import ALL_FEATURES, BASE_FEATURES, SoxsDefensiveHurdleSpec, _load_feature_frame
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "soxl_objective_feasibility_and_predictability_audit_v1"
HORIZONS = (1, 5, 10, 20)
FOLD_YEARS = tuple(range(2016, 2023))
FEATURE_SETS = {"base": BASE_FEATURES, "base_plus_soxs": ALL_FEATURES}
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/bootstrap_metrics.csv",
)


@dataclass(frozen=True)
class ObjectiveFeasibilityAuditSpec:
    root: Path
    experiment: str = EXPERIMENT
    development: tuple[str, str] = ("2011-01-03", "2022-12-30")
    outer: tuple[str, str] = ("2023-01-03", "2026-12-31")
    fold_years: tuple[int, ...] = FOLD_YEARS
    costs_bps: tuple[int, ...] = (10, 25)
    bootstrap_paths: int = 512
    bootstrap_block_days: int = 20
    oracle_dwell_days: int = 3

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_soxx_soxs"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def run_objective_feasibility_audit_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(ObjectiveFeasibilityAuditSpec(root=repo_root))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for horizon in HORIZONS:
        soxl = np.log(result["soxl_open"].shift(-(horizon + 1)) / result["soxl_open"].shift(-1))
        soxx = np.log(result["soxx_open"].shift(-(horizon + 1)) / result["soxx_open"].shift(-1))
        result[f"leverage_advantage_{horizon}d"] = soxl - soxx
        result[f"cash_advantage_{horizon}d"] = -soxx
    forward_soxx = pd.concat(
        [result["soxx_forward_open_return"].shift(-offset) for offset in range(5)], axis=1
    )
    result["future_vol_5d"] = forward_soxx.std(axis=1, ddof=1)
    path = (1.0 + forward_soxx).cumprod(axis=1) - 1.0
    complete = forward_soxx.notna().all(axis=1)
    result["tail_event_5d"] = np.where(complete, path.min(axis=1).le(-0.08).astype(float), np.nan)
    return result


def _build_dataset(frame: pd.DataFrame, spec: ObjectiveFeasibilityAuditSpec):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        *[f"leverage_advantage_{h}d" for h in HORIZONS],
        *[f"cash_advantage_{h}d" for h in HORIZONS],
        "future_vol_5d",
        "tail_event_5d",
    ]
    static = pd.concat({"feature": frame.loc[:, ALL_FEATURES], "label": frame.loc[:, labels]}, axis=1)
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_SOXS")], names=["datetime", "instrument"]
    )
    outer_end = min(spec.outer[1], frame.index.max().date().isoformat())
    segments: dict[str, tuple[str, str]] = {
        "development": spec.development,
        "outer": (spec.outer[0], outer_end),
    }
    for year in spec.fold_years:
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
        "segments": {key: list(value) for key, value in segments.items()},
        "segment_rows": {key: int(len(dataset.prepare(key))) for key in segments},
        "purge_embargo": "20 trading rows removed before every fold test year",
    }


def _dataset_frame(dataset, segment: str) -> pd.DataFrame:
    frame = dataset.prepare(segment, col_set=["feature", "label"]).droplevel("instrument").sort_index()
    frame.columns = [str(column[-1]) for column in frame.columns]
    return frame


def _continuous_model() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])


def _tail_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=1_000, random_state=17),
            ),
        ]
    )


def _safe_ic(actual: np.ndarray, prediction: np.ndarray) -> float:
    value = spearmanr(actual, prediction, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else 0.0


def _walk_forward_predictability(dataset, frame: pd.DataFrame, spec: ObjectiveFeasibilityAuditSpec):
    targets = [
        *[f"leverage_advantage_{h}d" for h in HORIZONS],
        *[f"cash_advantage_{h}d" for h in HORIZONS],
        "future_vol_5d",
        "tail_event_5d",
    ]
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    final_models: dict[str, Pipeline] = {}
    for year in spec.fold_years:
        test = _dataset_frame(dataset, f"fold_{year}")
        test = test.loc[test.index.year == year]
        train_end_position = frame.index.searchsorted(pd.Timestamp(f"{year}-01-01")) - 20
        train_end = frame.index[max(train_end_position - 1, 0)]
        train = frame.loc[(frame.index >= pd.Timestamp(spec.development[0])) & (frame.index <= train_end)]
        for feature_set, features in FEATURE_SETS.items():
            for target in targets:
                usable_train = train.dropna(subset=[*features, target])
                usable_test = test.dropna(subset=[*features, target])
                if usable_test.empty:
                    continue
                is_tail = target == "tail_event_5d"
                if is_tail and usable_train[target].nunique() < 2:
                    continue
                model = _tail_model() if is_tail else _continuous_model()
                model.fit(usable_train.loc[:, features], usable_train[target].astype(int if is_tail else float))
                prediction = (
                    model.predict_proba(usable_test.loc[:, features])[:, 1]
                    if is_tail
                    else model.predict(usable_test.loc[:, features])
                )
                actual = usable_test[target].to_numpy(dtype=float)
                if is_tail:
                    auc = roc_auc_score(actual, prediction) if len(np.unique(actual)) == 2 else np.nan
                    metric = float(auc) if np.isfinite(auc) else 0.5
                    row = {
                        "metric_name": "roc_auc",
                        "metric": metric,
                        "brier": float(brier_score_loss(actual, prediction)),
                        "base_rate": float(actual.mean()),
                    }
                else:
                    metric = _safe_ic(actual, prediction)
                    row = {
                        "metric_name": "spearman_ic",
                        "metric": metric,
                        "sign_accuracy": float((np.sign(actual) == np.sign(prediction)).mean()),
                        "brier": np.nan,
                        "base_rate": np.nan,
                    }
                fold_rows.append(
                    {
                        "fold_year": year,
                        "feature_set": feature_set,
                        "target": target,
                        "train_end": train_end.date().isoformat(),
                        "test_rows": len(usable_test),
                        **row,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "signal_date": usable_test.index,
                            "segment": "development_walk_forward",
                            "fold_year": year,
                            "feature_set": feature_set,
                            "target": target,
                            "actual": actual,
                            "prediction": prediction,
                        }
                    )
                )

    development = frame.loc[spec.development[0] : spec.development[1]]
    outer = _dataset_frame(dataset, "outer")
    for feature_set, features in FEATURE_SETS.items():
        for target in targets:
            usable_train = development.dropna(subset=[*features, target])
            usable_outer = outer.dropna(subset=[*features, target])
            is_tail = target == "tail_event_5d"
            if usable_outer.empty or (is_tail and usable_train[target].nunique() < 2):
                continue
            model = _tail_model() if is_tail else _continuous_model()
            model.fit(usable_train.loc[:, features], usable_train[target].astype(int if is_tail else float))
            prediction = (
                model.predict_proba(usable_outer.loc[:, features])[:, 1]
                if is_tail
                else model.predict(usable_outer.loc[:, features])
            )
            actual = usable_outer[target].to_numpy(dtype=float)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "signal_date": usable_outer.index,
                        "segment": "outer_diagnostic",
                        "fold_year": usable_outer.index.year,
                        "feature_set": feature_set,
                        "target": target,
                        "actual": actual,
                        "prediction": prediction,
                    }
                )
            )
            final_models[f"{feature_set}__{target}"] = model
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary, bootstrap = _summarize_predictability(folds, predictions, spec)
    outer_metrics = _outer_predictability_metrics(predictions)
    return folds, summary, outer_metrics, predictions, bootstrap, final_models


def _summarize_predictability(
    folds: pd.DataFrame, predictions: pd.DataFrame, spec: ObjectiveFeasibilityAuditSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    dev = predictions.loc[predictions["segment"].eq("development_walk_forward")]
    for (feature_set, target), group in folds.groupby(["feature_set", "target"], sort=True):
        metric_name = str(group["metric_name"].iloc[0])
        values = group["metric"].to_numpy(dtype=float)
        pred = dev.loc[(dev["feature_set"] == feature_set) & (dev["target"] == target)].sort_values("signal_date")
        indices = stationary_block_bootstrap_indices(
            observation_count=len(pred),
            path_length=len(pred),
            path_count=spec.bootstrap_paths,
            expected_block_length=spec.bootstrap_block_days,
            seed=17 + sum(ord(char) for char in f"{feature_set}:{target}"),
        )
        actual = pred["actual"].to_numpy(dtype=float)
        forecast = pred["prediction"].to_numpy(dtype=float)
        boot_values = []
        for path, sample in enumerate(indices):
            sampled_actual = actual[sample]
            sampled_forecast = forecast[sample]
            if metric_name == "roc_auc":
                value = roc_auc_score(sampled_actual, sampled_forecast) if len(np.unique(sampled_actual)) == 2 else 0.5
            else:
                value = _safe_ic(sampled_actual, sampled_forecast)
            boot_values.append(float(value))
            bootstrap_rows.append(
                {"feature_set": feature_set, "target": target, "path": path, "metric": value}
            )
        threshold = 0.5 if metric_name == "roc_auc" else 0.0
        rows.append(
            {
                "feature_set": feature_set,
                "target": target,
                "metric_name": metric_name,
                "mean_fold_metric": float(values.mean()),
                "median_fold_metric": float(np.median(values)),
                "positive_years": int((values > threshold).sum()),
                "fold_years": int(len(values)),
                "bootstrap_p05": float(np.quantile(boot_values, 0.05)),
                "bootstrap_p50": float(np.quantile(boot_values, 0.50)),
                "bootstrap_p95": float(np.quantile(boot_values, 0.95)),
                "stable_signal": bool((values > threshold).sum() >= 5 and np.quantile(boot_values, 0.05) > threshold),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def _outer_predictability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    outer = predictions.loc[predictions["segment"].eq("outer_diagnostic")]
    rows: list[dict[str, Any]] = []
    for (feature_set, target, year), group in outer.groupby(["feature_set", "target", "fold_year"]):
        if target == "tail_event_5d":
            value = roc_auc_score(group["actual"], group["prediction"]) if group["actual"].nunique() == 2 else np.nan
            name = "roc_auc"
        else:
            value = _safe_ic(group["actual"].to_numpy(), group["prediction"].to_numpy())
            name = "spearman_ic"
        rows.append(
            {
                "feature_set": feature_set,
                "target": target,
                "outer_year": int(year),
                "metric_name": name,
                "metric": value,
                "rows": len(group),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def _candidate_weights(market: pd.DataFrame) -> dict[str, np.ndarray]:
    rows = len(market)
    zero = np.zeros(rows)
    candidates: dict[str, np.ndarray] = {
        "cash": np.column_stack([zero, zero, zero, np.ones(rows)]),
        "soxx_buyhold": np.column_stack([zero, np.ones(rows), zero, zero]),
    }
    for weight in (0.15, 0.25, 1.0 / 3.0):
        candidates[f"fixed_soxl_cash_{weight:.3f}"] = np.column_stack(
            [np.full(rows, weight), zero, zero, np.full(rows, 1.0 - weight)]
        )
    for weight in (0.20, 0.40):
        candidates[f"fixed_soxl_soxx_{weight:.2f}"] = np.column_stack(
            [np.full(rows, weight), np.full(rows, 1.0 - weight), zero, zero]
        )
    for weight in (0.50, 0.75):
        candidates[f"fixed_soxx_cash_{weight:.2f}"] = np.column_stack(
            [zero, np.full(rows, weight), zero, np.full(rows, 1.0 - weight)]
        )
    for window in (50, 100, 200):
        on = market["soxx_close"].gt(market["soxx_close"].rolling(window).mean()).astype(float).to_numpy()
        candidates[f"trend_soxx_cash_ma{window}"] = np.column_stack([zero, on, zero, 1.0 - on])
        candidates[f"trend_soxl_cash_ma{window}"] = np.column_stack([on / 3.0, zero, zero, 1.0 - on / 3.0])
    soxl_vol = market["soxl_close_return"].rolling(20).std(ddof=1).shift(1) * np.sqrt(252.0)
    for target in (0.10, 0.15, 0.20):
        weight = (target / soxl_vol).clip(0.0, 0.50).fillna(0.0).to_numpy()
        candidates[f"vol_target_soxl_{int(target * 100)}"] = np.column_stack([weight, zero, zero, 1.0 - weight])
    defense = (
        market["soxx_close"].lt(market["soxx_close"].rolling(200).mean()) & market["soxs_trend5"].gt(0.0)
    ).astype(float).to_numpy()
    candidates["rule_soxx_or_10pct_soxs"] = np.column_stack(
        [zero, 1.0 - defense, 0.10 * defense, 0.90 * defense]
    )
    return candidates


def _simulate(market: pd.DataFrame, weights: np.ndarray, *, cost_bps: int, name: str) -> pd.DataFrame:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(market), 4) or not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("weights must align with market and sum to one")
    asset_returns = market.loc[:, ["soxl_forward_open_return", "soxx_forward_open_return", "soxs_forward_open_return"]].to_numpy()
    turnover = np.r_[0.5 * np.abs(weights[0] - np.array([0.0, 0.0, 0.0, 1.0])).sum(), 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)]
    result = pd.DataFrame(
        {
            "signal_date": market.index,
            "trade_date": market["trade_date"].to_numpy(),
            "exit_date": market["exit_date"].to_numpy(),
            "strategy": name,
            "soxl_return": asset_returns[:, 0],
            "soxx_return": asset_returns[:, 1],
            "soxs_return": asset_returns[:, 2],
            "soxl_weight": weights[:, 0],
            "soxx_weight": weights[:, 1],
            "soxs_weight": weights[:, 2],
            "cash_weight": weights[:, 3],
            "turnover": turnover,
        }
    )
    result["strategy_return"] = (weights[:, :3] * asset_returns).sum(axis=1) - turnover * cost_bps / 10_000.0
    return result


def _oracle_weights(market: pd.DataFrame, *, include_soxs: bool, cost_bps: int, penalty: float, dwell: int) -> np.ndarray:
    actions = [
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.4, 0.6, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    ]
    if include_soxs:
        actions.append(np.array([0.0, 0.0, 0.10, 0.90]))
    actions_array = np.vstack(actions)
    daily = market.loc[:, ["soxl_forward_open_return", "soxx_forward_open_return", "soxs_forward_open_return"]].to_numpy()
    blocks = [(start, min(start + dwell, len(market))) for start in range(0, len(market), dwell)]
    value = np.full((len(blocks), len(actions)), -np.inf)
    parent = np.full((len(blocks), len(actions)), -1, dtype=int)
    for block_index, (start, end) in enumerate(blocks):
        block_return = daily[start:end] @ actions_array[:, :3].T
        reward = np.log1p(block_return.clip(min=-0.99)).sum(axis=0) - penalty * np.square(np.minimum(block_return, 0.0)).sum(axis=0)
        for action_index, action in enumerate(actions_array):
            if block_index == 0:
                transition = 0.5 * np.abs(action - np.array([0.0, 0.0, 0.0, 1.0])).sum() * cost_bps / 10_000.0
                value[block_index, action_index] = reward[action_index] - transition
            else:
                transitions = 0.5 * np.abs(actions_array - action).sum(axis=1) * cost_bps / 10_000.0
                choices = value[block_index - 1] - transitions
                parent[block_index, action_index] = int(np.argmax(choices))
                value[block_index, action_index] = reward[action_index] + choices[parent[block_index, action_index]]
    selected = np.empty(len(blocks), dtype=int)
    selected[-1] = int(np.argmax(value[-1]))
    for block_index in range(len(blocks) - 1, 0, -1):
        selected[block_index - 1] = parent[block_index, selected[block_index]]
    weights = np.zeros((len(market), 4))
    for action_index, (start, end) in zip(selected, blocks):
        weights[start:end] = actions_array[action_index]
    return weights


def _return_maxdd(metrics: dict[str, float]) -> float:
    return float(metrics["return_pct"]) / abs(float(metrics["max_drawdown_pct"])) if metrics["max_drawdown_pct"] < 0 else np.nan


def _evaluate_result(result: pd.DataFrame, *, cost_bps: int) -> dict[str, Any]:
    metrics = _risk_metrics(result)
    benchmark = _simulate(
        result.set_index("signal_date").rename(columns={
            "soxl_return": "soxl_forward_open_return",
            "soxx_return": "soxx_forward_open_return",
            "soxs_return": "soxs_forward_open_return",
        }),
        np.column_stack([np.zeros(len(result)), np.ones(len(result)), np.zeros(len(result)), np.zeros(len(result))]),
        cost_bps=cost_bps,
        name="soxx_buyhold",
    )
    benchmark_wealth = float((1.0 + benchmark["strategy_return"]).prod())
    strategy_wealth = float((1.0 + result["strategy_return"]).prod())
    efficiency = _benchmark_efficiency(result, cost_bps=cost_bps)
    metrics.update(efficiency)
    metrics.update(
        {
            "terminal_wealth_ratio_vs_soxx": strategy_wealth / benchmark_wealth,
            "soxx_benchmark_return_pct": 100.0 * (benchmark_wealth - 1.0),
            "avg_soxs_position_pct": 100.0 * float(result["soxs_weight"].mean()),
            "avg_cash_position_pct": 100.0 * float(result["cash_weight"].mean()),
            "avg_total_risk_position_pct": 100.0 * float((result["soxl_weight"] + result["soxx_weight"] + result["soxs_weight"]).mean()),
        }
    )
    risk_gate = all(
        [
            metrics["max_drawdown_pct"] >= -20.0,
            metrics["worst_day_pct"] >= -12.0,
            metrics["cvar5_pct"] >= -8.0,
            metrics["worst_rolling_5d_pct"] >= -12.0,
        ]
    )
    metrics["risk_gate_pass"] = risk_gate
    metrics["minimum_position_gate_pass"] = bool(metrics["avg_total_risk_position_pct"] >= 15.0)
    metrics["upside_capture_gate_pass"] = bool(metrics["upside_capture_pct"] >= 60.0)
    metrics["relative_soxx_gate_pass"] = bool(metrics["terminal_wealth_ratio_vs_soxx"] > 1.0)
    metrics["efficiency_gate_pass"] = bool(
        metrics["return_maxdd"] > metrics["same_avg_risk_soxl_return_maxdd"]
        and metrics["return_maxdd"] > metrics["matched_beta_return_maxdd"]
    )
    metrics["all_primary_gates_pass"] = bool(
        risk_gate
        and metrics["minimum_position_gate_pass"]
        and metrics["upside_capture_gate_pass"]
        and metrics["relative_soxx_gate_pass"]
        and metrics["efficiency_gate_pass"]
    )
    return metrics


def _feasibility_frontier(market: pd.DataFrame, spec: ObjectiveFeasibilityAuditSpec):
    rows: list[dict[str, Any]] = []
    stored: list[pd.DataFrame] = []
    positions: list[pd.DataFrame] = []
    for segment, bounds in (("development", spec.development), ("outer", spec.outer)):
        sample = market.loc[bounds[0] : bounds[1]].copy()
        candidates = _candidate_weights(sample)
        for penalty in (0.0, 10.0, 50.0, 200.0, 1000.0):
            for include_soxs in (False, True):
                label = "with_soxs" if include_soxs else "long_only"
                name = f"oracle_{label}_lambda_{penalty:g}"
                candidates[name] = _oracle_weights(
                    sample, include_soxs=include_soxs, cost_bps=10, penalty=penalty, dwell=spec.oracle_dwell_days
                )
        for name, weights in candidates.items():
            kind = "oracle_nontradable" if name.startswith("oracle_") else "mechanical_tradable"
            for cost_bps in spec.costs_bps:
                result = _simulate(sample, weights, cost_bps=cost_bps, name=name)
                rows.append(
                    {
                        "segment": segment,
                        "strategy": name,
                        "strategy_kind": kind,
                        "cost_bps": cost_bps,
                        "selection_uses_outer_test": False,
                        **_evaluate_result(result, cost_bps=cost_bps),
                    }
                )
                stored.append(result.assign(segment=segment, strategy_kind=kind, cost_bps=cost_bps))
                positions.append(
                    result.loc[:, ["signal_date", "trade_date", "strategy", "soxl_weight", "soxx_weight", "soxs_weight", "cash_weight"]].assign(
                        segment=segment, strategy_kind=kind, cost_bps=cost_bps
                    )
                )
    frontier = pd.DataFrame(rows)
    gate_summary = (
        frontier.groupby(["segment", "strategy_kind", "cost_bps"], as_index=False)
        .agg(
            strategies=("strategy", "nunique"),
            primary_gate_passes=("all_primary_gates_pass", "sum"),
            risk_gate_passes=("risk_gate_pass", "sum"),
            position_gate_passes=("minimum_position_gate_pass", "sum"),
            capture_gate_passes=("upside_capture_gate_pass", "sum"),
            relative_soxx_gate_passes=("relative_soxx_gate_pass", "sum"),
            efficiency_gate_passes=("efficiency_gate_pass", "sum"),
        )
    )
    strategy_returns = pd.concat(stored, ignore_index=True)
    annual_rows: list[dict[str, Any]] = []
    outer_returns = strategy_returns.loc[strategy_returns["segment"].eq("outer")].copy()
    outer_returns["outer_year"] = pd.to_datetime(outer_returns["trade_date"]).dt.year
    for (strategy, kind, cost_bps, year), group in outer_returns.groupby(
        ["strategy", "strategy_kind", "cost_bps", "outer_year"], sort=True
    ):
        metrics = _risk_metrics(group.reset_index(drop=True))
        annual_rows.append(
            {
                "strategy": strategy,
                "strategy_kind": kind,
                "cost_bps": int(cost_bps),
                "outer_year": int(year),
                "risk_gate_pass": bool(
                    metrics["max_drawdown_pct"] >= -20.0
                    and metrics["worst_day_pct"] >= -12.0
                    and metrics["cvar5_pct"] >= -8.0
                    and metrics["worst_rolling_5d_pct"] >= -12.0
                ),
                "selection_uses_outer_test": False,
                **metrics,
            }
        )
    return (
        frontier,
        gate_summary,
        strategy_returns,
        pd.concat(positions, ignore_index=True),
        pd.DataFrame(annual_rows),
    )


def _run_card(spec: ObjectiveFeasibilityAuditSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, data_audit = _load_feature_frame(SoxsDefensiveHurdleSpec(root=spec.root))
    market = _add_targets(market)
    dataset, dataset_audit = _build_dataset(market, spec)
    folds, predictability, outer_predictability, predictions, bootstrap, models = _walk_forward_predictability(dataset, market, spec)
    frontier, gate_summary, strategy_returns, positions, outer_year_metrics = _feasibility_frontier(market, spec)

    dev10 = frontier.query("segment == 'development' and cost_bps == 10")
    mechanical_pass = int(dev10.query("strategy_kind == 'mechanical_tradable'")["all_primary_gates_pass"].sum())
    oracle_pass = int(dev10.query("strategy_kind == 'oracle_nontradable'")["all_primary_gates_pass"].sum())
    oracle_long_only_pass = int(
        dev10.loc[
            dev10["strategy"].str.startswith("oracle_long_only")
            & dev10["all_primary_gates_pass"]
        ].shape[0]
    )
    oracle_with_soxs_pass = int(
        dev10.loc[
            dev10["strategy"].str.startswith("oracle_with_soxs")
            & dev10["all_primary_gates_pass"]
        ].shape[0]
    )
    mechanical = dev10.query("strategy_kind == 'mechanical_tradable'")
    highest_capture_mechanical = mechanical.sort_values("upside_capture_pct", ascending=False).iloc[0]
    stable_base = set(predictability.query("feature_set == 'base' and stable_signal")["target"])
    stable_soxs = set(predictability.query("feature_set == 'base_plus_soxs' and stable_signal")["target"])
    incremental_soxs = sorted(stable_soxs - stable_base)
    if oracle_pass == 0:
        decision = "blocked_joint_constraints_not_feasible_under_tested_oracle"
    elif mechanical_pass == 0 and not stable_soxs:
        decision = "blocked_feasible_only_with_hindsight_and_targets_not_stably_predictable"
    elif mechanical_pass == 0:
        decision = "needs_iteration_feasible_oracle_and_some_stable_targets_but_no_tradable_rule_pass"
    else:
        decision = "needs_sealed_forward_mechanical_rule_pass_is_not_model_promotion"

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
        predictions.to_csv(temp / "predictions.csv", index=False)
        strategy_returns.to_csv(temp / "strategy_returns.csv", index=False)
        positions.to_csv(temp / "positions.csv", index=False)
        bootstrap.to_csv(temp / "bootstrap_metrics.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                development=f"{spec.development[0]}:{spec.development[1]}",
                outer=f"{spec.outer[0]}:{dataset_audit['segments']['outer'][1]}",
                fold_years=",".join(map(str, spec.fold_years)),
                selection_uses_outer_test="false",
                oracle_dwell_days=spec.oracle_dwell_days,
            )
            R.log_metrics(
                development_mechanical_primary_gate_passes=mechanical_pass,
                development_oracle_primary_gate_passes=oracle_pass,
                stable_base_targets=len(stable_base),
                stable_base_plus_soxs_targets=len(stable_soxs),
            )
            for artifact in [model_dir, *temp.glob("*.csv")]:
                recorder.log_artifact(artifact.as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            summary = {
                "experiment": spec.experiment,
                "card_type": "objective_feasibility_and_predictability_audit",
                "primary_notebook": f"{spec.experiment}.ipynb",
                "promotion_decision": decision,
                "selection_uses_outer_test": False,
                "outer_test_use": "diagnostic_only_previously_observed",
                "data_audit": data_audit,
                "qlib_dataset": dataset_audit,
                "qlib_docs": [
                    "https://qlib.readthedocs.io/en/latest/component/data.html",
                    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                    "https://qlib.readthedocs.io/en/latest/component/strategy.html",
                ],
                "gates": {
                    "max_drawdown_pct_min": -20.0,
                    "worst_day_pct_min": -12.0,
                    "cvar5_pct_min": -8.0,
                    "worst_rolling_5d_pct_min": -12.0,
                    "avg_total_risk_position_pct_min": 15.0,
                    "upside_capture_pct_min": 60.0,
                    "terminal_wealth_ratio_vs_soxx_min_exclusive": 1.0,
                    "return_maxdd_exceeds_same_avg_risk_soxl": True,
                    "return_maxdd_exceeds_matched_beta": True,
                    "costs_bps": list(spec.costs_bps),
                },
                "audit_result": {
                    "development_mechanical_primary_gate_passes_10bps": mechanical_pass,
                    "development_oracle_primary_gate_passes_10bps": oracle_pass,
                    "development_oracle_long_only_passes_10bps": oracle_long_only_pass,
                    "development_oracle_with_soxs_passes_10bps": oracle_with_soxs_pass,
                    "soxs_expands_oracle_pass_count": bool(oracle_with_soxs_pass > oracle_long_only_pass),
                    "highest_capture_mechanical_strategy": str(highest_capture_mechanical["strategy"]),
                    "highest_capture_mechanical_pct": float(highest_capture_mechanical["upside_capture_pct"]),
                    "highest_capture_mechanical_max_drawdown_pct": float(highest_capture_mechanical["max_drawdown_pct"]),
                    "highest_capture_mechanical_worst_rolling_5d_pct": float(
                        highest_capture_mechanical["worst_rolling_5d_pct"]
                    ),
                    "stable_base_targets": sorted(stable_base),
                    "stable_base_plus_soxs_targets": sorted(stable_soxs),
                    "incremental_stable_soxs_targets": incremental_soxs,
                    "oracle_is_nontradable": True,
                },
                "predictability_protocol": {
                    "folds": list(spec.fold_years),
                    "purge_rows": 20,
                    "models": ["fixed Ridge(alpha=10)", "fixed balanced LogisticRegression(C=0.1)"],
                    "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
                    "targets": sorted(predictability["target"].unique()),
                    "stable_definition": "at least 5 of 7 positive years and stationary-bootstrap p05 above zero or AUC 0.5",
                },
                "feasibility_protocol": {
                    "mechanical_rules_are_fixed": True,
                    "oracle_is_nontradable": True,
                    "oracle_dwell_days": spec.oracle_dwell_days,
                    "oracle_soxs_max_weight": 0.10,
                    "oracle_penalties": [0, 10, 50, 200, 1000],
                    "cash_return_assumption": 0.0,
                },
                "recorder_id": recorder_id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "sidecars": {
                    "feasibility_frontier": "feasibility_frontier.csv",
                    "feasibility_gate_summary": "feasibility_gate_summary.csv",
                    "predictability_fold_metrics": "predictability_fold_metrics.csv",
                    "predictability_summary": "predictability_summary.csv",
                    "outer_predictability_diagnostic": "outer_predictability_diagnostic.csv",
                    "outer_year_risk_metrics": "outer_year_risk_metrics.csv",
                    "risk_analysis": "risk_analysis.csv",
                    "analysis_summary": "analysis_summary.json",
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
                frontier,
                gate_summary,
                folds,
                predictability,
                outer_predictability,
                outer_year_metrics,
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
    frontier: pd.DataFrame,
    gate_summary: pd.DataFrame,
    folds: pd.DataFrame,
    predictability: pd.DataFrame,
    outer_predictability: pd.DataFrame,
    outer_year_metrics: pd.DataFrame,
) -> None:
    frontier.to_csv(card / "feasibility_frontier.csv", index=False)
    gate_summary.to_csv(card / "feasibility_gate_summary.csv", index=False)
    folds.to_csv(card / "predictability_fold_metrics.csv", index=False)
    predictability.to_csv(card / "predictability_summary.csv", index=False)
    outer_predictability.to_csv(card / "outer_predictability_diagnostic.csv", index=False)
    outer_year_metrics.to_csv(card / "outer_year_risk_metrics.csv", index=False)
    frontier.loc[:, [
        "segment", "strategy", "strategy_kind", "cost_bps", "max_drawdown_pct", "worst_day_pct",
        "cvar5_pct", "worst_rolling_5d_pct", "annualized_vol_pct", "risk_gate_pass",
    ]].to_csv(card / "risk_analysis.csv", index=False)
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (card / "analysis_summary.json").write_text(
        json.dumps({"promotion_decision": summary["promotion_decision"], **summary["audit_result"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    result = summary["audit_result"]
    (card / "conclusion.md").write_text(
        f"""# 中文结论

## 跑了什么

用 qlib SOXL/SOXX/SOXS 日线 provider 做两层审计。第一层在 2011–2022 开发期比较固定机械规则与持有至少 3 日、最多 10% SOXS 的事后 oracle；第二层对相对收益、现金优势、未来波动与五日尾部事件做 2016–2022 逐年 walk-forward。2023–2026 只做诊断，不参与任何选择。

## 关键结果

10bp 开发期通过全部主门槛的机械规则数为 {result['development_mechanical_primary_gate_passes_10bps']}；不可交易 oracle 为 {result['development_oracle_primary_gate_passes_10bps']}（不含 SOXS {result['development_oracle_long_only_passes_10bps']}，允许最多 10% SOXS {result['development_oracle_with_soxs_passes_10bps']}，因此 SOXS 没有扩大通过数量）。机械规则中上行捕获最高的是 `{result['highest_capture_mechanical_strategy']}`，为 {result['highest_capture_mechanical_pct']:.2f}%，但 MaxDD {result['highest_capture_mechanical_max_drawdown_pct']:.2f}%、最差五日 {result['highest_capture_mechanical_worst_rolling_5d_pct']:.2f}%。基础特征稳定目标为 `{result['stable_base_targets']}`；加入 SOXS 后为 `{result['stable_base_plus_soxs_targets']}`；SOXS 独有稳定目标为 `{result['incremental_stable_soxs_targets']}`。

## 决定、风险与下一步

`{summary['promotion_decision']}`。oracle 使用未来收益，只回答约束是否存在可行域，不能当作策略；机械规则通过也不等于可晋升。现金按 0 收益处理，SOXS 是每日重置产品，且 2023–2026 已被观察。只有开发期存在可行域且标签稳定，下一步才值得设计单一、冻结的 allocation model，并进入新的 sealed-forward paper window。

不构成买卖建议。
""",
        encoding="utf-8",
    )
    (card / "workflow.yaml").write_text(
        f"""experiment: {summary['experiment']}
card_type: {summary['card_type']}
provider_uri: {summary['data_audit']['provider_uri']}
recorder_uri: {summary['recorder_uri']}
recorder_id: {summary['recorder_id']}
qlib:
  dataset: [DatasetH, DataHandlerLP, StaticDataLoader]
  recorder: qlib.workflow.R
selection:
  development: 2011-2022
  walk_forward_folds: 2016-2022
  outer: 2023-2026_diagnostic_only
  selection_uses_outer_test: false
feasibility:
  oracle_nontradable: true
  oracle_min_dwell_days: 3
  oracle_soxs_max_weight: 0.10
promotion:
  decision: {summary['promotion_decision']}
artifact_policy:
  policy: slim_card_recorder_first
  workflow_config: workflow.yaml
  machine_entrypoint: summary.json
  recorder_only_artifacts:
    - artifacts/{summary['experiment']}/models/
    - artifacts/{summary['experiment']}/predictions.csv
    - artifacts/{summary['experiment']}/strategy_returns.csv
    - artifacts/{summary['experiment']}/positions.csv
    - artifacts/{summary['experiment']}/bootstrap_metrics.csv
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(json.dumps(run_objective_feasibility_audit_card(), indent=2))
