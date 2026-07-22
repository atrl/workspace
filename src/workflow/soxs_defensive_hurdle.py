from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.objective.relative_wealth import stationary_block_bootstrap_indices, terminal_relative_wealth_metrics
from src.workflow.leveraged_etf_paper_replay import _risk_metrics
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "soxs_defensive_hurdle_v1"
BASE_FEATURES = (
    "soxx_trend20",
    "soxx_trend60",
    "soxx_vol20",
    "soxx_drawdown60",
    "relative_momentum20",
)
SOXS_FEATURES = (
    "soxs_trend5",
    "soxs_trend20",
    "inverse_tracking_gap1",
    "inverse_tracking_gap20",
    "soxl_soxs_path20",
)
ALL_FEATURES = (*BASE_FEATURES, *SOXS_FEATURES)
VARIANTS = ("base_no_soxs", "soxs_features_no_trade", "soxs_defense_only", "soxs_features_and_defense")
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/probabilities.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/bootstrap_terminal_wealth.csv",
)


@dataclass(frozen=True)
class SoxsDefensiveHurdleSpec:
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
class PolicyConfig:
    variant: str
    attack_threshold: float = 1.0
    attack_tilt: float = 0.0
    defense_threshold: float = 1.0
    soxs_weight: float = 0.0
    max_soxs_days: int = 5


def run_soxs_defensive_hurdle_card(*, root: Path | str | None = None) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
    return _run_card(SoxsDefensiveHurdleSpec(root=repo_root))


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


def _load_feature_frame(spec: SoxsDefensiveHurdleSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    import qlib
    from qlib.data import D

    manifest_path = spec.provider_uri / "qlib_store_manifest.json"
    source_manifest_path = spec.root / "data" / "raw" / "soxs_defensive_v1" / "source_manifest.json"
    if not manifest_path.exists() or not source_manifest_path.exists():
        raise FileNotFoundError("blocked: qlib unavailable: SOXS defensive provider or source manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if set(manifest["instruments"]) != {"SOXL", "SOXX", "SOXS"}:
        raise ValueError("SOXS defensive provider must contain exactly SOXL/SOXX/SOXS")

    qlib.init(provider_uri=spec.provider_uri.resolve().as_posix(), region="us", kernels=1)
    calendar = D.calendar(start_time=manifest["calendar_start"], end_time=manifest["calendar_end"], freq="day")
    raw = D.features(
        ["SOXL", "SOXX", "SOXS"],
        ["$open", "$close"],
        start_time=manifest["calendar_start"],
        end_time=manifest["calendar_end"],
        freq="day",
    ).reset_index()
    open_panel = raw.pivot(index="datetime", columns="instrument", values="$open").sort_index()
    close_panel = raw.pivot(index="datetime", columns="instrument", values="$close").sort_index()
    frame = pd.DataFrame(index=open_panel.index)
    for symbol in ("SOXL", "SOXX", "SOXS"):
        prefix = symbol.lower()
        frame[f"{prefix}_open"] = open_panel[symbol]
        frame[f"{prefix}_close"] = close_panel[symbol]
        frame[f"{prefix}_close_return"] = frame[f"{prefix}_close"].pct_change()
        frame[f"{prefix}_forward_open_return"] = frame[f"{prefix}_open"].shift(-2) / frame[
            f"{prefix}_open"
        ].shift(-1) - 1.0
    frame["trade_date"] = frame.index.to_series().shift(-1)
    frame["exit_date"] = frame.index.to_series().shift(-2)
    frame["soxx_trend20"] = frame["soxx_close"] / frame["soxx_close"].rolling(20).mean() - 1.0
    frame["soxx_trend60"] = frame["soxx_close"] / frame["soxx_close"].rolling(60).mean() - 1.0
    frame["soxx_vol20"] = frame["soxx_close_return"].rolling(20).std(ddof=1)
    frame["soxx_drawdown60"] = frame["soxx_close"] / frame["soxx_close"].rolling(60).max() - 1.0
    frame["relative_momentum20"] = frame["soxl_close"].pct_change(20) - frame["soxx_close"].pct_change(20)
    frame["soxs_trend5"] = frame["soxs_close"] / frame["soxs_close"].rolling(5).mean() - 1.0
    frame["soxs_trend20"] = frame["soxs_close"] / frame["soxs_close"].rolling(20).mean() - 1.0
    frame["inverse_tracking_gap1"] = frame["soxs_close_return"] + 3.0 * frame["soxx_close_return"]
    frame["inverse_tracking_gap20"] = frame["inverse_tracking_gap1"].rolling(20).std(ddof=1)
    frame["soxl_soxs_path20"] = (
        np.log1p(frame["soxl_close_return"].clip(lower=-0.99))
        + np.log1p(frame["soxs_close_return"].clip(lower=-0.99))
    ).rolling(20).sum()
    attack_relative = np.log(frame["soxl_open"].shift(-22) / frame["soxl_open"].shift(-1)) - np.log(
        frame["soxx_open"].shift(-22) / frame["soxx_open"].shift(-1)
    )
    defense_return = frame["soxx_open"].shift(-6) / frame["soxx_open"].shift(-1) - 1.0
    frame["attack_label"] = np.where(attack_relative.notna(), (attack_relative > 0.01).astype(float), np.nan)
    frame["defense_label"] = np.where(defense_return.notna(), (defense_return <= -0.05).astype(float), np.nan)
    frame = frame.dropna(subset=[*ALL_FEATURES, "soxl_forward_open_return", "soxx_forward_open_return", "soxs_forward_open_return", "trade_date", "exit_date"])
    frame.index.name = "signal_date"
    audit = {
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "provider_manifest": manifest_path.resolve().as_posix(),
        "source_manifest": source_manifest_path.resolve().as_posix(),
        "as_of": manifest["as_of"],
        "instruments": manifest["instruments"],
        "fields": manifest["fields"],
        "calendar_rows": int(len(calendar)),
        "usable_rows": int(len(frame)),
        "usable_start": frame.index.min().date().isoformat(),
        "usable_end": frame.index.max().date().isoformat(),
        "relationship_audit": source_manifest["relationship_audit"],
        "source_api": ["qlib.init", "qlib.data.D.calendar", "qlib.data.D.features"],
    }
    return frame, audit


def _build_dataset(frame: pd.DataFrame, spec: SoxsDefensiveHurdleSpec):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    static = pd.concat(
        {
            "feature": frame.loc[:, ALL_FEATURES],
            "label": frame.loc[:, ["attack_label", "defense_label"]],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [frame.index, np.full(len(frame), "SOXL_SOXX_SOXS")], names=["datetime", "instrument"]
    )
    outer_end = min(spec.outer[1], frame.index.max().date().isoformat())
    segments = {
        "initial_train": spec.initial_train,
        "tune": spec.tune,
        "refit": spec.refit,
        "validation": spec.validation,
        "outer": (spec.outer[0], outer_end),
    }
    handler = DataHandlerLP(
        instruments=None,
        start_time=frame.index.min(),
        end_time=frame.index.max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    audit = {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {key: list(value) for key, value in segments.items()},
        "segment_rows": {key: int(len(dataset.prepare(key))) for key in segments},
        "purge_embargo": "initial/refit label ends are at least 22 calendar days before next evaluation segment",
    }
    return dataset, audit


def _prepare(dataset, segment: str) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"]).droplevel("instrument").sort_index()
    result.columns = [str(column[-1]) for column in result.columns]
    return result


def _fit_classifier(frame: pd.DataFrame, features: tuple[str, ...], label: str) -> Pipeline:
    usable = frame.dropna(subset=[*features, label])
    if usable[label].nunique() != 2:
        raise ValueError(f"training label {label} must contain both classes")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=1_000, random_state=17),
            ),
        ]
    )
    model.fit(usable.loc[:, features], usable[label].astype(int))
    return model


def _candidate_configs() -> list[PolicyConfig]:
    configs: list[PolicyConfig] = []
    for variant in ("base_no_soxs", "soxs_features_no_trade"):
        for attack_threshold in (0.55, 0.60, 0.65, 0.70):
            for attack_tilt in (0.10, 0.20):
                configs.append(PolicyConfig(variant, attack_threshold, attack_tilt))
    for defense_threshold in (0.55, 0.60, 0.65, 0.70):
        for soxs_weight in (0.10, 0.20):
            configs.append(
                PolicyConfig(
                    "soxs_defense_only",
                    defense_threshold=defense_threshold,
                    soxs_weight=soxs_weight,
                )
            )
    for attack_threshold in (0.60, 0.65, 0.70):
        for attack_tilt in (0.10, 0.20):
            for defense_threshold in (0.55, 0.60, 0.65):
                for soxs_weight in (0.10, 0.20):
                    configs.append(
                        PolicyConfig(
                            "soxs_features_and_defense",
                            attack_threshold,
                            attack_tilt,
                            defense_threshold,
                            soxs_weight,
                        )
                    )
    return configs


def _predict_probabilities(
    frame: pd.DataFrame,
    *,
    attack_base: Pipeline,
    attack_soxs: Pipeline,
    defense_soxs: Pipeline,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "attack_base_probability": attack_base.predict_proba(frame.loc[:, BASE_FEATURES])[:, 1],
            "attack_soxs_probability": attack_soxs.predict_proba(frame.loc[:, ALL_FEATURES])[:, 1],
            "defense_soxs_probability": defense_soxs.predict_proba(frame.loc[:, ALL_FEATURES])[:, 1],
        },
        index=frame.index,
    )


def _simulate_policy(
    market: pd.DataFrame,
    probabilities: pd.DataFrame,
    config: PolicyConfig,
    *,
    cost_bps: int,
) -> pd.DataFrame:
    weights = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    defense_streak = 0
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(market.itertuples()):
        probability = probabilities.loc[row.Index]
        use_defense = config.variant in {"soxs_defense_only", "soxs_features_and_defense"}
        use_attack = config.variant in {"base_no_soxs", "soxs_features_no_trade", "soxs_features_and_defense"}
        if use_defense and probability["defense_soxs_probability"] >= config.defense_threshold:
            defense_streak += 1
            if defense_streak <= config.max_soxs_days:
                target = np.array([0.0, 0.0, config.soxs_weight, 1.0 - config.soxs_weight])
                state = "soxs_defense"
            else:
                target = np.array([0.0, 0.0, 0.0, 1.0])
                state = "cash_after_soxs_limit"
        else:
            defense_streak = 0
            attack_probability = probability[
                "attack_base_probability" if config.variant == "base_no_soxs" else "attack_soxs_probability"
            ]
            if use_attack and attack_probability >= config.attack_threshold:
                target = np.array([config.attack_tilt, 1.0 - config.attack_tilt, 0.0, 0.0])
                state = "soxl_tilt"
            else:
                target = np.array([0.0, 1.0, 0.0, 0.0])
                state = "soxx_anchor"
        turnover = float(0.5 * np.abs(target - weights).sum())
        cost = turnover * float(cost_bps) / 10_000.0
        asset_gross = np.array(
            [
                1.0 + row.soxl_forward_open_return,
                1.0 + row.soxx_forward_open_return,
                1.0 + row.soxs_forward_open_return,
                1.0,
            ]
        )
        portfolio_gross = float(target @ asset_gross)
        strategy_return = (1.0 - cost) * portfolio_gross - 1.0
        weights = target * asset_gross / max(portfolio_gross, 1e-12)
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
                "attack_probability": probability[
                    "attack_base_probability" if config.variant == "base_no_soxs" else "attack_soxs_probability"
                ],
                "defense_probability": probability["defense_soxs_probability"],
            }
        )
    return pd.DataFrame(rows)


def _metrics(result: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    metrics = _risk_metrics(result)
    strategy_wealth = float((1.0 + result["strategy_return"]).prod())
    benchmark_wealth = float((1.0 + benchmark["strategy_return"]).prod())
    active = result["strategy_return"].to_numpy() - benchmark["strategy_return"].to_numpy()
    active_std = float(active.std(ddof=1))
    metrics.update(
        {
            "terminal_wealth_ratio_vs_soxx": strategy_wealth / benchmark_wealth,
            "soxx_benchmark_return_pct": 100.0 * (benchmark_wealth - 1.0),
            "active_information_ratio": float(active.mean() / active_std * np.sqrt(252.0))
            if active_std > 0.0
            else 0.0,
            "avg_soxs_position_pct": 100.0 * float(result["soxs_weight"].mean()),
            "avg_cash_position_pct": 100.0 * float(result["cash_weight"].mean()),
            "avg_total_risk_position_pct": 100.0
            * float((result["soxl_weight"] + result["soxx_weight"] + result["soxs_weight"]).mean()),
            "soxs_days": int(result["soxs_weight"].gt(0.0).sum()),
        }
    )
    return metrics


def _annual_metrics(result: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    years = pd.to_datetime(result["signal_date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year in sorted(years.unique()):
        mask = years.eq(year)
        strategy_return = float((1.0 + result.loc[mask, "strategy_return"]).prod() - 1.0)
        benchmark_return = float((1.0 + benchmark.loc[mask, "strategy_return"]).prod() - 1.0)
        rows.append(
            {
                "year": int(year),
                "return_pct": 100.0 * strategy_return,
                "soxx_return_pct": 100.0 * benchmark_return,
                "wealth_ratio": (1.0 + strategy_return) / (1.0 + benchmark_return),
                "max_drawdown_pct": _risk_metrics(result.loc[mask].reset_index(drop=True))["max_drawdown_pct"],
            }
        )
    return pd.DataFrame(rows)


def _benchmark(market: pd.DataFrame, *, cost_bps: int) -> pd.DataFrame:
    probabilities = pd.DataFrame(
        {"attack_base_probability": 0.0, "attack_soxs_probability": 0.0, "defense_soxs_probability": 0.0},
        index=market.index,
    )
    return _simulate_policy(market, probabilities, PolicyConfig("base_no_soxs"), cost_bps=cost_bps)


def _select_on_tune(
    tune_market: pd.DataFrame,
    tune_probabilities: pd.DataFrame,
    spec: SoxsDefensiveHurdleSpec,
) -> tuple[pd.DataFrame, dict[str, PolicyConfig]]:
    rows: list[dict[str, Any]] = []
    for config in _candidate_configs():
        cost_metrics: dict[int, dict[str, Any]] = {}
        for cost_bps in spec.costs_bps[:2]:
            benchmark = _benchmark(tune_market, cost_bps=cost_bps)
            result = _simulate_policy(tune_market, tune_probabilities, config, cost_bps=cost_bps)
            cost_metrics[cost_bps] = _metrics(result, benchmark)
        ten = cost_metrics[10]
        stress = cost_metrics[25]
        risk_improvement = ten["max_drawdown_pct"] - _risk_metrics(_benchmark(tune_market, cost_bps=10))[
            "max_drawdown_pct"
        ]
        score = float(
            np.log(max(ten["terminal_wealth_ratio_vs_soxx"], 1e-8))
            + np.log(max(stress["terminal_wealth_ratio_vs_soxx"], 1e-8))
            + 0.02 * (ten["active_information_ratio"] + stress["active_information_ratio"])
            + (0.002 * risk_improvement if "defense" in config.variant else 0.0)
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
    selected: dict[str, PolicyConfig] = {}
    for variant in VARIANTS:
        row = candidates.loc[candidates["variant"].eq(variant)].iloc[0]
        selected[variant] = PolicyConfig(
            variant=variant,
            attack_threshold=float(row["attack_threshold"]),
            attack_tilt=float(row["attack_tilt"]),
            defense_threshold=float(row["defense_threshold"]),
            soxs_weight=float(row["soxs_weight"]),
            max_soxs_days=int(row["max_soxs_days"]),
        )
    return candidates, selected


def _bootstrap_validation(result: pd.DataFrame, benchmark: pd.DataFrame, spec: SoxsDefensiveHurdleSpec) -> tuple[dict[str, Any], pd.DataFrame]:
    indices = stationary_block_bootstrap_indices(
        observation_count=len(result),
        path_length=len(result),
        path_count=spec.bootstrap_paths,
        expected_block_length=spec.bootstrap_block_days,
        seed=91_733,
    )
    strategy = np.prod(1.0 + result["strategy_return"].to_numpy()[indices], axis=1)
    soxx = np.prod(1.0 + benchmark["strategy_return"].to_numpy()[indices], axis=1)
    metrics = terminal_relative_wealth_metrics(strategy, soxx)
    distribution = pd.DataFrame(
        {"strategy_terminal_wealth": strategy, "soxx_terminal_wealth": soxx, "terminal_wealth_ratio": strategy / soxx}
    )
    return metrics, distribution


def _feature_importance(models: dict[str, Pipeline]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_map = {"attack_base": BASE_FEATURES, "attack_soxs": ALL_FEATURES, "defense_soxs": ALL_FEATURES}
    for name, model in models.items():
        coefficients = model.named_steps["logistic"].coef_[0]
        for feature, coefficient in zip(feature_map[name], coefficients, strict=True):
            rows.append({"model": name, "feature": feature, "coefficient": coefficient, "abs_coefficient": abs(coefficient)})
    return pd.DataFrame(rows).sort_values(["model", "abs_coefficient"], ascending=[True, False])


def _run_card(spec: SoxsDefensiveHurdleSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    market, data_audit = _load_feature_frame(spec)
    dataset, dataset_audit = _build_dataset(market, spec)
    initial_train = _prepare(dataset, "initial_train")
    tune = _prepare(dataset, "tune")
    refit = _prepare(dataset, "refit")
    validation = market.loc[spec.validation[0] : spec.validation[1]].copy()
    outer_end = dataset_audit["segments"]["outer"][1]
    outer = market.loc[spec.outer[0] : outer_end].copy()

    initial_models = {
        "attack_base": _fit_classifier(initial_train, BASE_FEATURES, "attack_label"),
        "attack_soxs": _fit_classifier(initial_train, ALL_FEATURES, "attack_label"),
        "defense_soxs": _fit_classifier(initial_train, ALL_FEATURES, "defense_label"),
    }
    tune_market = market.loc[spec.tune[0] : spec.tune[1]].copy()
    tune_probabilities = _predict_probabilities(tune_market, **initial_models)
    candidate_selection, selected_configs = _select_on_tune(tune_market, tune_probabilities, spec)

    refit_models = {
        "attack_base": _fit_classifier(refit, BASE_FEATURES, "attack_label"),
        "attack_soxs": _fit_classifier(refit, ALL_FEATURES, "attack_label"),
        "defense_soxs": _fit_classifier(refit, ALL_FEATURES, "defense_label"),
    }
    validation_probabilities = _predict_probabilities(validation, **refit_models)
    outer_probabilities = _predict_probabilities(outer, **refit_models)
    comparison_rows: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    stored_returns: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_distributions: list[pd.DataFrame] = []
    validation_results: dict[tuple[str, int], pd.DataFrame] = {}
    for segment_name, segment_market, probabilities in (
        ("validation", validation, validation_probabilities),
        ("outer", outer, outer_probabilities),
    ):
        for cost_bps in spec.costs_bps:
            benchmark = _benchmark(segment_market, cost_bps=cost_bps)
            benchmark_metrics = _metrics(benchmark, benchmark)
            comparison_rows.append(
                {"segment": segment_name, "variant": "soxx_buyhold", "cost_bps": cost_bps, **benchmark_metrics}
            )
            for variant, config in selected_configs.items():
                result = _simulate_policy(segment_market, probabilities, config, cost_bps=cost_bps)
                metrics = _metrics(result, benchmark)
                comparison_rows.append(
                    {"segment": segment_name, "variant": variant, "cost_bps": cost_bps, **metrics}
                )
                result["segment"] = segment_name
                result["variant"] = variant
                result["cost_bps"] = cost_bps
                stored_returns.append(result)
                annual = _annual_metrics(result, benchmark)
                annual["segment"] = segment_name
                annual["variant"] = variant
                annual["cost_bps"] = cost_bps
                annual_frames.append(annual)
                if segment_name == "validation":
                    validation_results[(variant, cost_bps)] = result
                    if cost_bps in (10, 25):
                        bootstrap_metrics, distribution = _bootstrap_validation(result, benchmark, spec)
                        bootstrap_rows.append({"variant": variant, "cost_bps": cost_bps, **bootstrap_metrics})
                        distribution["variant"] = variant
                        distribution["cost_bps"] = cost_bps
                        bootstrap_distributions.append(distribution)

    comparison = pd.DataFrame(comparison_rows)
    annual_metrics = pd.concat(annual_frames, ignore_index=True)
    bootstrap_metrics = pd.DataFrame(bootstrap_rows)
    model_rows: list[dict[str, Any]] = []
    for variant, config in selected_configs.items():
        ten = comparison.loc[
            comparison["segment"].eq("validation") & comparison["variant"].eq(variant) & comparison["cost_bps"].eq(10)
        ].iloc[0]
        stress = comparison.loc[
            comparison["segment"].eq("validation") & comparison["variant"].eq(variant) & comparison["cost_bps"].eq(25)
        ].iloc[0]
        bootstrap10 = bootstrap_metrics.loc[
            bootstrap_metrics["variant"].eq(variant) & bootstrap_metrics["cost_bps"].eq(10)
        ].iloc[0]
        years = annual_metrics.loc[
            annual_metrics["segment"].eq("validation")
            & annual_metrics["variant"].eq(variant)
            & annual_metrics["cost_bps"].eq(10)
        ]
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
        gate = bool(relative_gate and (risk_gate if "defense" in variant else True))
        validation_score = float(
            np.log(max(ten["terminal_wealth_ratio_vs_soxx"], 1e-8))
            + 0.02 * ten["active_information_ratio"]
            + 0.002 * (ten["max_drawdown_pct"] - comparison.loc[
                comparison["segment"].eq("validation")
                & comparison["variant"].eq("soxx_buyhold")
                & comparison["cost_bps"].eq(10),
                "max_drawdown_pct",
            ].iloc[0])
        )
        model_rows.append(
            {
                **asdict(config),
                "validation_gate_pass": gate,
                "relative_gate_pass": relative_gate,
                "risk_gate_pass": risk_gate,
                "validation_score": validation_score,
                "validation_ratio_10bps": ten["terminal_wealth_ratio_vs_soxx"],
                "validation_ratio_25bps": stress["terminal_wealth_ratio_vs_soxx"],
                "validation_active_ir_10bps": ten["active_information_ratio"],
                "validation_max_drawdown_10bps": ten["max_drawdown_pct"],
                "validation_bootstrap_outperformance_probability_10bps": bootstrap10[
                    "outperformance_probability"
                ],
                "positive_validation_years": positive_years,
                "selection_uses_outer_test": False,
            }
        )
    model_selection = pd.DataFrame(model_rows).sort_values(
        ["validation_gate_pass", "validation_score"], ascending=[False, False]
    )
    selected = model_selection.iloc[0].to_dict()
    selected_variant = str(selected["variant"])
    validation_pass = bool(selected["validation_gate_pass"])
    promotion_decision = (
        "research_diagnostic_validation_pass_outer_previously_observed_requires_sealed_forward"
        if validation_pass
        else "research_reject_soxs_ablation_failed_validation_gates"
    )
    feature_importance = _feature_importance(refit_models)
    probabilities = pd.concat(
        [
            validation_probabilities.assign(segment="validation"),
            outer_probabilities.assign(segment="outer"),
        ]
    ).reset_index(names="signal_date")
    strategy_returns = pd.concat(stored_returns, ignore_index=True)
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

    sidecars = {
        "model_selection": "model_selection.csv",
        "candidate_selection": "candidate_selection.csv",
        "bootstrap_metrics": "bootstrap_metrics.csv",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "annual_metrics": "annual_metrics.csv",
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
            summary = {
                "experiment": spec.experiment,
                "card_type": "soxs_feature_and_constrained_defense_ablation",
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
                "hypotheses": {
                    "feature_only": "SOXS path and tracking-gap features may improve conditional SOXL-vs-SOXX timing",
                    "defense_sleeve": "a mutually exclusive, five-day-capped SOXS sleeve may improve tail risk",
                    "unrestricted_four_asset_allocator": "not implemented because simultaneous SOXL/SOXS softmax exposure is structurally unstable",
                },
                "labels": {
                    "attack": "future 21-day open-to-open log SOXL minus SOXX greater than 1 percent",
                    "defense": "future 5-day SOXX open-to-open return at or below -5 percent",
                },
                "constraints": {
                    "default": "100 percent SOXX",
                    "soxl_tilt": [0.10, 0.20],
                    "soxs_weight": [0.10, 0.20],
                    "soxl_soxs_mutually_exclusive": True,
                    "max_consecutive_soxs_days": 5,
                    "after_max_soxs_days": "cash until defense probability resets",
                },
                "selected_model": _json_ready(selected),
                "selected_config": asdict(selected_configs[selected_variant]),
                "model_selection_uses_segment": "validation; candidate hyperparameters fixed on tune",
                "selected_validation_metrics": _json_ready(
                    comparison.loc[
                        comparison["segment"].eq("validation") & comparison["variant"].eq(selected_variant)
                    ].to_dict(orient="records")
                ),
                "selected_outer_metrics": _json_ready(
                    comparison.loc[
                        comparison["segment"].eq("outer") & comparison["variant"].eq(selected_variant)
                    ].to_dict(orient="records")
                ),
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
                candidate_selection=candidate_selection,
                bootstrap_metrics=bootstrap_metrics,
                comparison=comparison,
                annual_metrics=annual_metrics,
                feature_importance=feature_importance,
            )
            for rel_path in sidecars.values():
                path = card / rel_path
                if path.exists() and path.suffix != ".md":
                    recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


def _write_card(
    card: Path,
    summary: dict[str, Any],
    *,
    model_selection: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    annual_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    model_selection.to_csv(card / "model_selection.csv", index=False)
    candidate_selection.to_csv(card / "candidate_selection.csv", index=False)
    bootstrap_metrics.to_csv(card / "bootstrap_metrics.csv", index=False)
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    annual_metrics.to_csv(card / "annual_metrics.csv", index=False)
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
    analysis = {
        "promotion_decision": summary["promotion_decision"],
        "selected_model": selected,
        "selected_validation_metrics": summary["selected_validation_metrics"],
        "selected_outer_metrics": summary["selected_outer_metrics"],
    }
    (card / "analysis_summary.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    validation10 = next(row for row in summary["selected_validation_metrics"] if row["cost_bps"] == 10)
    outer10 = next(row for row in summary["selected_outer_metrics"] if row["cost_bps"] == 10)
    conclusion = f"""# 中文结论

## 跑了什么

从独立 qlib SOXL/SOXX/SOXS provider 读取复权后的 open/close，比较基础特征、仅增加 SOXS 特征、仅允许受约束 SOXS 防守、SOXS 特征加防守四组消融。候选参数只在 2017–2019 tune 选择，2011–2019-11 refit，2020–2022 validation，2023–2026-07 outer 只诊断。SOXL 与 SOXS 互斥，SOXS 上限 20%，连续最多五日，之后转现金直到风险信号复位。

## Validation

选中 `{selected['variant']}`。10bp：相对 SOXX 财富比 {validation10['terminal_wealth_ratio_vs_soxx']:.4f}，Active IR {validation10['active_information_ratio']:.4f}，收益 {validation10['return_pct']:.2f}%，MaxDD {validation10['max_drawdown_pct']:.2f}%，SOXS 平均仓位 {validation10['avg_soxs_position_pct']:.2f}%。validation gate={selected['validation_gate_pass']}。

## Outer诊断

10bp：收益 {outer10['return_pct']:.2f}%，SOXX {outer10['soxx_benchmark_return_pct']:.2f}%，财富比 {outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD {outer10['max_drawdown_pct']:.2f}%。outer 不参与选择。

## 决定

`{summary['promotion_decision']}`。SOXS 是否提供增量必须以 feature-only 与 tradable-sleeve 的差异判断；已观察 outer 不能用于返调阈值。下一步只能进入 sealed-forward paper shadow，或在 validation 失败时停止本架构。

不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    (card / "failure_modes.md").write_text(
        """# Failure modes

- SOXS is a daily-reset inverse leveraged ETF; even five-day exposure is path dependent and can decay in chop.
- Yahoo corporate-action history was inconsistent in the recent SOXS window; the provider uses Nasdaq as authoritative after the stitch date, records the scale audit, and applies Yahoo adjusted-close ratios for distributions.
- The defense label is rare and regime-dependent, so class balancing does not create new crash regimes.
- Cash has zero return because a PIT Treasury series is not part of this provider.
- The 2023-2026 outer interval was already observed before this experiment and is diagnostic only.
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
    print(json.dumps(run_soxs_defensive_hurdle_card(), indent=2))
