from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _build_teacher_training_frame,
    _load_source_frame,
    _passes_risk,
    _risk_metrics,
    _simulate_policy,
)
from src.workflow.stage1_futu_sim import (
    SOURCE_EXPERIMENT,
    SOURCE_RECORDER_ID,
    _with_intraday,
    build_matched_comparison,
    load_frozen_inference_frame,
)


EXPERIMENT = "stage1_generalization_validation_v1"
CARD_TYPE = "stage1_frozen_generalization_validation_card"
QLIB_DOCS = {
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "strategy": "https://qlib.readthedocs.io/en/latest/component/strategy.html",
    "analysis": "https://qlib.readthedocs.io/en/latest/component/report.html",
}


@dataclass(frozen=True)
class Stage1GeneralizationSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    training_end: str = "2022-12-30"
    required_cost_bps: tuple[float, ...] = (10.0, 25.0)
    stress_cost_bps: tuple[float, ...] = (50.0,)
    rolling_windows: tuple[int, ...] = (126, 252)
    rolling_step: int = 21
    parameter_thresholds: tuple[float, ...] = (0.50, 0.55, 0.60)
    parameter_action_betas: tuple[float, ...] = (0.50, 0.60, 0.70)
    bootstrap_block_lengths: tuple[int, ...] = (20, 60)
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20_260_725
    minimum_annual_relative_passes: int = 3
    minimum_annual_risk_passes: int = 4
    minimum_parameter_wealth_ratio: float = 0.90
    minimum_bootstrap_outperformance_probability: float = 0.70
    minimum_rolling_outperformance_rate: float = 0.60

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _benchmark_wealth(returns: pd.Series | np.ndarray) -> float:
    return float(np.prod(1.0 + np.asarray(returns, dtype="float64")))


def build_annual_metrics(stage1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    replay = stage1.copy()
    replay["signal_date"] = pd.to_datetime(replay["signal_date"])
    for year, section in replay.groupby(replay["signal_date"].dt.year):
        metrics = _risk_metrics(section["strategy_return"].to_numpy())
        soxx_wealth = _benchmark_wealth(section["soxx_return"])
        rows.append(
            {
                "year": int(year),
                "rows": int(len(section)),
                **metrics,
                "soxx_terminal_wealth": soxx_wealth,
                "terminal_wealth_ratio_vs_soxx": metrics["terminal_wealth"]
                / soxx_wealth,
                "beats_soxx": bool(metrics["terminal_wealth"] > soxx_wealth),
                "passes_risk": _passes_risk(metrics),
                "avg_soxl_weight_pct": 100.0
                * float(section["soxl_weight"].mean()),
                "avg_soxx_weight_pct": 100.0
                * float(section["soxx_weight"].mean()),
                "avg_cash_weight_pct": 100.0
                * float(section["cash_weight"].mean()),
                "recovery_action_count": int(section["recovery_action"].sum()),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def build_cost_stress(stage1: pd.DataFrame, costs: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost_bps in costs:
        returns = (
            stage1["gross_return"].to_numpy(dtype="float64")
            - stage1["turnover"].to_numpy(dtype="float64") * cost_bps / 10_000.0
        )
        metrics = _risk_metrics(returns)
        rows.append(
            {
                "cost_bps": float(cost_bps),
                **metrics,
                "passes_risk": _passes_risk(metrics),
                "selection_uses_outer_test": False,
            }
        )
    return pd.DataFrame(rows)


def build_matched_exposure_metrics(
    stage1: pd.DataFrame,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    mean_soxl = float(stage1["open_soxl_weight"].mean())
    mean_soxx = float(stage1["open_soxx_weight"].mean())
    mean_cash = float(stage1["open_cash_weight"].mean())
    returns = (
        mean_soxl * stage1["soxl_return"].to_numpy(dtype="float64")
        + mean_soxx * stage1["soxx_return"].to_numpy(dtype="float64")
    )
    returns[0] -= (mean_soxl + mean_soxx) * cost_bps / 10_000.0
    matched = _risk_metrics(returns)
    selected = _risk_metrics(stage1["strategy_return"].to_numpy())
    return pd.DataFrame(
        [
            {
                "benchmark": "constant_mean_weight_matched_exposure",
                "cost_bps": float(cost_bps),
                "mean_soxl_weight": mean_soxl,
                "mean_soxx_weight": mean_soxx,
                "mean_cash_weight": mean_cash,
                **{
                    f"matched_{key}": value
                    for key, value in matched.items()
                },
                "stage1_terminal_wealth": selected["terminal_wealth"],
                "stage1_terminal_wealth_ratio_vs_matched": selected[
                    "terminal_wealth"
                ]
                / matched["terminal_wealth"],
                "stage1_beats_matched_exposure": bool(
                    selected["terminal_wealth"] > matched["terminal_wealth"]
                ),
                "benchmark_cost_convention": (
                    "initial_entry_cost_only_frictionless_daily_rebalanced"
                ),
                "selection_uses_outer_test": False,
            }
        ]
    )


def build_rolling_metrics(
    stage1: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    step: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    replay = stage1.reset_index(drop=True).copy()
    for window in windows:
        if len(replay) < window:
            continue
        starts = list(range(0, len(replay) - window + 1, step))
        final_start = len(replay) - window
        if final_start not in starts:
            starts.append(final_start)
        for start in sorted(set(starts)):
            section = replay.iloc[start : start + window]
            metrics = _risk_metrics(section["strategy_return"].to_numpy())
            benchmark = _benchmark_wealth(section["soxx_return"])
            rows.append(
                {
                    "window_days": int(window),
                    "start": pd.Timestamp(section["trade_date"].iloc[0])
                    .date()
                    .isoformat(),
                    "end": pd.Timestamp(section["exit_date"].iloc[-1])
                    .date()
                    .isoformat(),
                    **metrics,
                    "soxx_terminal_wealth": benchmark,
                    "terminal_wealth_ratio_vs_soxx": metrics["terminal_wealth"]
                    / benchmark,
                    "beats_soxx": bool(metrics["terminal_wealth"] > benchmark),
                    "passes_risk": _passes_risk(metrics),
                    "selection_uses_outer_test": False,
                }
            )
    return pd.DataFrame(rows)


def build_regime_metrics(
    stage1: pd.DataFrame,
    inference: pd.DataFrame,
) -> pd.DataFrame:
    features = inference[
        [
            "vix",
            "soxx_trend60",
            "high_chop_flag",
            "tail_probability",
        ]
    ].copy()
    features.index = pd.to_datetime(features.index)
    features.index.name = "signal_date"
    replay = stage1.copy()
    replay["signal_date"] = pd.to_datetime(replay["signal_date"])
    replay = replay.merge(
        features.reset_index(),
        on="signal_date",
        how="left",
        validate="one_to_one",
    )
    replay["vix_regime"] = pd.cut(
        replay["vix"],
        bins=[-np.inf, 20.0, 30.0, np.inf],
        labels=["vix_below_20", "vix_20_to_30", "vix_above_30"],
    ).astype(str)
    replay["trend_regime"] = np.where(
        replay["soxx_trend60"].ge(0.0),
        "soxx_trend_up",
        "soxx_trend_down",
    )
    replay["chop_regime"] = np.where(
        replay["high_chop_flag"].ge(0.5),
        "high_chop",
        "normal_chop",
    )
    rows: list[dict[str, Any]] = []
    for regime_type in ("vix_regime", "trend_regime", "chop_regime"):
        for regime, section in replay.groupby(regime_type, observed=True):
            metrics = _risk_metrics(section["strategy_return"].to_numpy())
            benchmark = _benchmark_wealth(section["soxx_return"])
            rows.append(
                {
                    "regime_type": regime_type,
                    "regime": str(regime),
                    "rows": int(len(section)),
                    **metrics,
                    "soxx_terminal_wealth": benchmark,
                    "terminal_wealth_ratio_vs_soxx": metrics["terminal_wealth"]
                    / benchmark,
                    "avg_soxl_weight_pct": 100.0
                    * float(section["soxl_weight"].mean()),
                    "avg_soxx_weight_pct": 100.0
                    * float(section["soxx_weight"].mean()),
                    "avg_cash_weight_pct": 100.0
                    * float(section["cash_weight"].mean()),
                    "selection_uses_outer_test": False,
                }
            )
    return pd.DataFrame(rows)


def build_parameter_stress(
    *,
    spec: Stage1GeneralizationSpec,
    recovery_spec: IntradayRecoveryDirectAllocationSpec,
    complete: pd.DataFrame,
    teacher: pd.DataFrame,
    selected_model: dict[str, Any],
    selected_stage1: pd.DataFrame,
) -> pd.DataFrame:
    selected_wealth = _risk_metrics(
        selected_stage1["strategy_return"].to_numpy()
    )["terminal_wealth"]
    soxx_wealth = _benchmark_wealth(selected_stage1["soxx_return"])
    rows: list[dict[str, Any]] = []
    for threshold in spec.parameter_thresholds:
        for action_beta in spec.parameter_action_betas:
            replay = _simulate_policy(
                complete.reset_index(),
                spec=recovery_spec,
                teacher_frame=teacher,
                c_value=float(selected_model["logistic_c"]),
                threshold=float(threshold),
                action_beta=float(action_beta),
                first_action_year=min(recovery_spec.outer_years),
                model_cache={},
                cost_bps=recovery_spec.transaction_cost_bps,
            )
            metrics = _risk_metrics(replay["strategy_return"].to_numpy())
            rows.append(
                {
                    "probability_threshold": float(threshold),
                    "action_beta": float(action_beta),
                    "is_selected_parameter": bool(
                        np.isclose(
                            threshold,
                            float(selected_model["probability_threshold"]),
                        )
                        and np.isclose(
                            action_beta,
                            float(selected_model["action_beta"]),
                        )
                    ),
                    **metrics,
                    "terminal_wealth_ratio_vs_selected": metrics[
                        "terminal_wealth"
                    ]
                    / selected_wealth,
                    "terminal_wealth_ratio_vs_soxx": metrics["terminal_wealth"]
                    / soxx_wealth,
                    "passes_risk": _passes_risk(metrics),
                    "recovery_action_count": int(replay["recovery_action"].sum()),
                    "selection_uses_outer_test": False,
                    "outer_role": "frozen_parameter_sensitivity_only",
                }
            )
    result = pd.DataFrame(rows)
    selected = result.loc[result["is_selected_parameter"]]
    if len(selected) != 1:
        raise ValueError("blocked: selected Stage1 parameter is missing from stress grid")
    if (
        abs(float(selected.iloc[0]["terminal_wealth"]) - selected_wealth)
        > 1e-10
    ):
        raise ValueError("blocked: selected parameter stress does not match Stage1 replay")
    return result


def circular_block_bootstrap(
    active_log_returns: np.ndarray,
    *,
    block_length: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    values = np.asarray(active_log_returns, dtype="float64")
    if values.ndim != 1 or len(values) < block_length:
        raise ValueError("active returns must be one-dimensional and longer than the block")
    if block_length <= 1 or samples <= 0:
        raise ValueError("block_length must exceed one and samples must be positive")
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    result = np.empty(samples, dtype="float64")
    for sample in range(samples):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        sampled = values[indices.reshape(-1)[: len(values)]]
        result[sample] = float(np.exp(sampled.sum()))
    return result


def build_bootstrap_evidence(
    stage1: pd.DataFrame,
    *,
    block_lengths: tuple[int, ...],
    samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_log = np.log1p(
        stage1["strategy_return"].to_numpy(dtype="float64")
    ) - np.log1p(stage1["soxx_return"].to_numpy(dtype="float64"))
    summaries: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    for offset, block_length in enumerate(block_lengths):
        ratios = circular_block_bootstrap(
            active_log,
            block_length=block_length,
            samples=samples,
            seed=seed + offset,
        )
        summaries.append(
            {
                "block_length": int(block_length),
                "samples": int(samples),
                "terminal_wealth_ratio_p05": float(np.quantile(ratios, 0.05)),
                "terminal_wealth_ratio_median": float(np.median(ratios)),
                "terminal_wealth_ratio_p95": float(np.quantile(ratios, 0.95)),
                "outperformance_probability": float(np.mean(ratios > 1.0)),
                "selection_uses_outer_test": False,
            }
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "block_length": int(block_length),
                    "sample": np.arange(samples),
                    "terminal_wealth_ratio_vs_soxx": ratios,
                }
            )
        )
    return pd.DataFrame(summaries), pd.concat(sample_frames, ignore_index=True)


def build_cross_asset_proxy(
    *,
    spec: Stage1GeneralizationSpec,
    stage1: pd.DataFrame,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    pairs = (
        ("TQQQ_QQQ", "TQQQ", "QQQ"),
        ("UPRO_SPY", "UPRO", "SPY"),
        ("SPXL_SPY", "SPXL", "SPY"),
        ("TECL_XLK", "TECL", "XLK"),
    )
    replay = stage1.copy()
    trade_dates = pd.to_datetime(replay["trade_date"])
    exit_dates = pd.to_datetime(replay["exit_date"])
    weights = replay[
        ["open_soxl_weight", "open_soxx_weight", "open_cash_weight"]
    ].to_numpy(dtype="float64")
    previous = np.vstack([np.zeros(3), weights[:-1]])
    turnover = np.abs(weights[:, :2] - previous[:, :2]).sum(axis=1)
    metrics_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for pair_name, levered, underlying in pairs:
        provider = spec.root / "data" / "qlib" / levered.lower()
        config = QlibProviderConfig(
            provider_uri=provider,
            region="us",
            market="all",
            freq="day",
            kernels=1,
        )
        features = load_qlib_feature_frame(
            config,
            instruments=[levered, underlying],
            fields=["$open"],
            start=trade_dates.min().date().isoformat(),
            end=exit_dates.max().date().isoformat(),
        ).reset_index()
        features["datetime"] = pd.to_datetime(features["datetime"])
        opens = features.pivot(
            index="datetime",
            columns="instrument",
            values="$open",
        ).sort_index()
        levered_entry = opens[levered].reindex(trade_dates).to_numpy()
        levered_exit = opens[levered].reindex(exit_dates).to_numpy()
        base_entry = opens[underlying].reindex(trade_dates).to_numpy()
        base_exit = opens[underlying].reindex(exit_dates).to_numpy()
        levered_return = levered_exit / levered_entry - 1.0
        base_return = base_exit / base_entry - 1.0
        usable = np.isfinite(levered_return) & np.isfinite(base_return)
        if int(usable.sum()) < int(0.95 * len(replay)):
            raise ValueError(
                f"blocked: insufficient qlib cross-asset coverage for {pair_name}"
            )
        proxy_return = (
            weights[:, 0] * levered_return
            + weights[:, 1] * base_return
            - turnover * cost_bps / 10_000.0
        )
        mean_weights = weights.mean(axis=0)
        matched_return = (
            mean_weights[0] * levered_return
            + mean_weights[1] * base_return
        )
        first_usable = int(np.flatnonzero(usable)[0])
        matched_return[first_usable] -= (
            mean_weights[0] + mean_weights[1]
        ) * cost_bps / 10_000.0
        section = pd.DataFrame(
            {
                "pair": pair_name,
                "signal_date": pd.to_datetime(replay["signal_date"]),
                "trade_date": trade_dates,
                "exit_date": exit_dates,
                "levered_return": levered_return,
                "underlying_return": base_return,
                "proxy_return": proxy_return,
                "matched_exposure_return": matched_return,
                "usable": usable,
            }
        ).loc[usable]
        metrics = _risk_metrics(section["proxy_return"].to_numpy())
        benchmark = _benchmark_wealth(section["underlying_return"])
        levered_buyhold = _benchmark_wealth(section["levered_return"])
        matched_exposure = _benchmark_wealth(
            section["matched_exposure_return"]
        )
        metrics_rows.append(
            {
                "pair": pair_name,
                "levered_asset": levered,
                "underlying_asset": underlying,
                "rows": int(len(section)),
                **metrics,
                "underlying_terminal_wealth": benchmark,
                "levered_terminal_wealth": levered_buyhold,
                "matched_exposure_terminal_wealth": matched_exposure,
                "terminal_wealth_ratio_vs_underlying": metrics[
                    "terminal_wealth"
                ]
                / benchmark,
                "terminal_wealth_ratio_vs_matched_exposure": metrics[
                    "terminal_wealth"
                ]
                / matched_exposure,
                "beats_underlying": bool(metrics["terminal_wealth"] > benchmark),
                "beats_matched_exposure": bool(
                    metrics["terminal_wealth"] > matched_exposure
                ),
                "passes_risk": _passes_risk(metrics),
                "proxy_role": "negative_control_not_model_selection",
                "execution_approximation": "frozen_open_weight_schedule_open_to_open",
                "matched_exposure_convention": (
                    "constant_mean_weight_initial_cost_only_"
                    "frictionless_daily_rebalanced"
                ),
                "selection_uses_outer_test": False,
            }
        )
        daily_frames.append(section)
        audits.append(
            {
                "pair": pair_name,
                "provider_uri": provider.resolve().as_posix(),
                "qlib_api": [
                    "qlib.init",
                    "qlib.data.D.calendar",
                    "qlib.data.D.instruments",
                    "qlib.data.D.features",
                ],
                "start": section["trade_date"].min().date().isoformat(),
                "end": section["exit_date"].max().date().isoformat(),
                "rows": int(len(section)),
            }
        )
    return (
        pd.DataFrame(metrics_rows),
        pd.concat(daily_frames, ignore_index=True),
        audits,
    )


def _record_artifacts(
    spec: Stage1GeneralizationSpec,
    *,
    stage1: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    cross_asset_daily: pd.DataFrame,
    summary_metrics: dict[str, float],
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        artifacts = {
            "stage1_daily.csv": stage1,
            "bootstrap_samples.csv": bootstrap_samples,
            "cross_asset_proxy_daily.csv": cross_asset_daily,
        }
        for filename, frame in artifacts.items():
            frame.to_csv(tmp_dir / filename, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=spec.source_recorder_id,
                source_training_end=spec.training_end,
                selection_uses_outer_test="false",
                outer_role="diagnostic_only",
                bootstrap_samples=str(spec.bootstrap_samples),
                bootstrap_seed=str(spec.bootstrap_seed),
            )
            R.log_metrics(**summary_metrics)
            for filename in artifacts:
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(),
                    artifact_path=spec.experiment,
                )
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(
                    recorder.get_local_dir()
                ).resolve().as_posix(),
            }
    recorder_artifacts = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (recorder_artifacts / filename).unlink(missing_ok=True)
    return result


def run_stage1_generalization_validation_card(
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = _resolve_root(root)
    spec = Stage1GeneralizationSpec(root=repo_root)
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(
        (
            repo_root
            / "research"
            / spec.source_experiment
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    if source_summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: Stage1 source recorder drifted")
    if source_summary.get("selection_uses_outer_test") is not False:
        raise ValueError("blocked: Stage1 source used outer test for selection")
    selected_model = dict(source_summary["selected_model"])
    if bool(selected_model.get("stage2_enabled")):
        raise ValueError("blocked: generalization validation only permits Stage1")

    comparison, stage1, _ = build_matched_comparison(root=repo_root)
    inference, inference_audit = load_frozen_inference_frame(root=repo_root)
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=repo_root)
    complete = _with_intraday(
        inference,
        recovery_spec=recovery_spec,
    ).dropna(subset=["soxl_forward_open_return", "soxx_forward_open_return"])
    source_frame, source_frame_audit = _load_source_frame(recovery_spec)
    teacher = _build_teacher_training_frame(
        source_frame,
        spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta,
    )

    annual = build_annual_metrics(stage1)
    cost = build_cost_stress(
        stage1,
        spec.required_cost_bps + spec.stress_cost_bps,
    )
    rolling = build_rolling_metrics(
        stage1,
        windows=spec.rolling_windows,
        step=spec.rolling_step,
    )
    regimes = build_regime_metrics(stage1, inference)
    parameters = build_parameter_stress(
        spec=spec,
        recovery_spec=recovery_spec,
        complete=complete,
        teacher=teacher,
        selected_model=selected_model,
        selected_stage1=stage1,
    )
    bootstrap, bootstrap_samples = build_bootstrap_evidence(
        stage1,
        block_lengths=spec.bootstrap_block_lengths,
        samples=spec.bootstrap_samples,
        seed=spec.bootstrap_seed,
    )
    cross_asset, cross_asset_daily, cross_asset_audit = build_cross_asset_proxy(
        spec=spec,
        stage1=stage1,
        cost_bps=recovery_spec.transaction_cost_bps,
    )
    matched_exposure = build_matched_exposure_metrics(
        stage1,
        cost_bps=recovery_spec.transaction_cost_bps,
    )

    annual_relative_passes = int(annual["beats_soxx"].sum())
    annual_risk_passes = int(annual["passes_risk"].sum())
    required_cost = cost.loc[cost["cost_bps"].isin(spec.required_cost_bps)]
    cost_gate = bool(required_cost["passes_risk"].all())
    parameter_min_ratio = float(
        parameters["terminal_wealth_ratio_vs_selected"].min()
    )
    parameter_gate = bool(
        parameters["passes_risk"].all()
        and parameter_min_ratio >= spec.minimum_parameter_wealth_ratio
    )
    bootstrap_min_probability = float(
        bootstrap["outperformance_probability"].min()
    )
    bootstrap_gate = bool(
        bootstrap_min_probability
        >= spec.minimum_bootstrap_outperformance_probability
    )
    matched_exposure_gate = bool(
        matched_exposure["stage1_beats_matched_exposure"].iloc[0]
    )
    rolling_summary = (
        rolling.groupby("window_days")
        .agg(
            window_count=("beats_soxx", "size"),
            beats_soxx_count=("beats_soxx", "sum"),
            beats_soxx_rate=("beats_soxx", "mean"),
            risk_pass_rate=("passes_risk", "mean"),
            ratio_min=("terminal_wealth_ratio_vs_soxx", "min"),
            ratio_p25=(
                "terminal_wealth_ratio_vs_soxx",
                lambda value: float(np.quantile(value, 0.25)),
            ),
            ratio_median=("terminal_wealth_ratio_vs_soxx", "median"),
        )
        .reset_index()
    )
    rolling_gate = bool(
        rolling_summary["beats_soxx_rate"]
        .ge(spec.minimum_rolling_outperformance_rate)
        .all()
        and rolling_summary["risk_pass_rate"].eq(1.0).all()
    )
    temporal_gate = bool(
        annual_relative_passes >= spec.minimum_annual_relative_passes
        and annual_risk_passes >= spec.minimum_annual_risk_passes
    )
    core_generalization_gate = bool(
        temporal_gate
        and cost_gate
        and parameter_gate
        and bootstrap_gate
        and rolling_gate
        and matched_exposure_gate
    )
    clean_unseen_holdout_available = False
    status = (
        "conditional_generalization_supported_paper_only"
        if core_generalization_gate
        else "generalization_not_proven_paper_validation_required"
    )
    decision = "continue_futu_paper_sim_collect_clean_forward_holdout"
    summary_metrics = {
        "stage1_return_pct": float(
            comparison["stage1_10bps"]["return_pct"]
        ),
        "stage1_max_drawdown_pct": float(
            comparison["stage1_10bps"]["max_drawdown_pct"]
        ),
        "annual_relative_pass_count": float(annual_relative_passes),
        "annual_risk_pass_count": float(annual_risk_passes),
        "parameter_min_wealth_ratio": parameter_min_ratio,
        "bootstrap_min_outperformance_probability": bootstrap_min_probability,
        "stage1_terminal_wealth_ratio_vs_matched_exposure": float(
            matched_exposure[
                "stage1_terminal_wealth_ratio_vs_matched"
            ].iloc[0]
        ),
    }
    recorder = _record_artifacts(
        spec,
        stage1=stage1,
        bootstrap_samples=bootstrap_samples,
        cross_asset_daily=cross_asset_daily,
        summary_metrics=summary_metrics,
    )
    sidecars = {
        "annual_metrics": "annual_metrics.csv",
        "cost_stress": "cost_stress.csv",
        "rolling_window_metrics": "rolling_window_metrics.csv",
        "rolling_summary": "rolling_summary.csv",
        "regime_metrics": "regime_metrics.csv",
        "parameter_stress": "parameter_stress.csv",
        "bootstrap_summary": "bootstrap_summary.csv",
        "cross_asset_proxy_metrics": "cross_asset_proxy_metrics.csv",
        "matched_exposure_metrics": "matched_exposure_metrics.csv",
        "selected_model": "selected_model.json",
    }
    summary = {
        "experiment": spec.experiment,
        "card_type": CARD_TYPE,
        "status": status,
        "decision": decision,
        "real_money_promotion": "blocked",
        "clean_unseen_holdout_available": clean_unseen_holdout_available,
        "clean_holdout_limitation": (
            "The model was frozen after 2022, but 2023-2026 outer results have "
            "been repeatedly inspected during human-in-the-loop research. They "
            "remain diagnostic and are not a pristine final holdout."
        ),
        "selection_uses_outer_test": False,
        "outer_role": "frozen_diagnostic_only_not_model_selection",
        "system_of_record": "qlib",
        "model_selection": {
            "source_model_frozen": True,
            "source_stage2_enabled": False,
            "selection_uses_outer_test": False,
            "validation_does_not_replace_selected_parameters": True,
        },
        "source_experiment": spec.source_experiment,
        "source_recorder_id": spec.source_recorder_id,
        "source_training_end": spec.training_end,
        "provider_uri": inference_audit["market_audit"][
            "execution_provider_uri"
        ],
        "market": "all",
        "region": "us",
        "as_of": comparison["window"]["end"],
        "window": comparison["window"],
        "qlib_docs": QLIB_DOCS,
        "frozen_stage1_metrics_10bps": comparison["stage1_10bps"],
        "soxx_metrics": comparison["soxx"],
        "generalization_gates": {
            "core_generalization_gate": core_generalization_gate,
            "temporal_gate": temporal_gate,
            "annual_relative_pass_count": annual_relative_passes,
            "minimum_annual_relative_passes": spec.minimum_annual_relative_passes,
            "annual_risk_pass_count": annual_risk_passes,
            "minimum_annual_risk_passes": spec.minimum_annual_risk_passes,
            "cost_gate": cost_gate,
            "required_cost_bps": list(spec.required_cost_bps),
            "parameter_gate": parameter_gate,
            "parameter_min_wealth_ratio": parameter_min_ratio,
            "minimum_parameter_wealth_ratio": spec.minimum_parameter_wealth_ratio,
            "bootstrap_gate": bootstrap_gate,
            "bootstrap_min_outperformance_probability": bootstrap_min_probability,
            "minimum_bootstrap_outperformance_probability": (
                spec.minimum_bootstrap_outperformance_probability
            ),
            "rolling_gate": rolling_gate,
            "minimum_rolling_outperformance_rate": (
                spec.minimum_rolling_outperformance_rate
            ),
            "matched_exposure_gate": matched_exposure_gate,
            "stage1_terminal_wealth_ratio_vs_matched_exposure": float(
                matched_exposure[
                    "stage1_terminal_wealth_ratio_vs_matched"
                ].iloc[0]
            ),
        },
        "rolling_summary": rolling_summary.to_dict(orient="records"),
        "cross_asset_negative_control": {
            "role": "diagnostic_only_not_a_transferable_model_test",
            "warning": (
                "The Stage1 features and labels are semiconductor-specific. "
                "Applying its frozen open-weight schedule to other 3x/1x pairs "
                "only tests generic beta timing and cannot establish direct "
                "cross-asset model portability."
            ),
            "beats_underlying_count": int(cross_asset["beats_underlying"].sum()),
            "beats_matched_exposure_count": int(
                cross_asset["beats_matched_exposure"].sum()
            ),
            "pair_count": int(len(cross_asset)),
        },
        "source_audit": {
            "inference": inference_audit,
            "source_frame": source_frame_audit,
            "cross_asset": cross_asset_audit,
        },
        "sidecars": sidecars,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/stage1_daily.csv",
                f"artifacts/{spec.experiment}/bootstrap_samples.csv",
                f"artifacts/{spec.experiment}/cross_asset_proxy_daily.csv",
            ],
        },
        **recorder,
    }
    annual.to_csv(spec.card_dir / sidecars["annual_metrics"], index=False)
    cost.to_csv(spec.card_dir / sidecars["cost_stress"], index=False)
    rolling.to_csv(
        spec.card_dir / sidecars["rolling_window_metrics"],
        index=False,
    )
    rolling_summary.to_csv(
        spec.card_dir / sidecars["rolling_summary"],
        index=False,
    )
    regimes.to_csv(spec.card_dir / sidecars["regime_metrics"], index=False)
    parameters.to_csv(
        spec.card_dir / sidecars["parameter_stress"],
        index=False,
    )
    bootstrap.to_csv(
        spec.card_dir / sidecars["bootstrap_summary"],
        index=False,
    )
    cross_asset.to_csv(
        spec.card_dir / sidecars["cross_asset_proxy_metrics"],
        index=False,
    )
    matched_exposure.to_csv(
        spec.card_dir / sidecars["matched_exposure_metrics"],
        index=False,
    )
    (spec.card_dir / sidecars["selected_model"]).write_text(
        json.dumps(
            {
                "source_experiment": spec.source_experiment,
                "source_recorder_id": spec.source_recorder_id,
                "source_training_end": spec.training_end,
                "source_model_frozen": True,
                "selection_uses_outer_test": False,
                "selected_model": selected_model,
                "stage2_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "workflow.yaml").write_text(
        "\n".join(
            [
                f"experiment: {spec.experiment}",
                f"provider_uri: {summary['provider_uri']}",
                "market: all",
                "region: us",
                f"as_of: {summary['as_of']}",
                f"recorder_uri: {recorder['recorder_uri']}",
                f"source_experiment: {spec.source_experiment}",
                f"source_recorder_id: {spec.source_recorder_id}",
                f"source_training_end: {spec.training_end}",
                "model_selection:",
                "  source_model_frozen: true",
                "  selection_uses_outer_test: false",
                "  outer_role: frozen_diagnostic_only",
                "validation:",
                "  annual_years: [2023, 2024, 2025, 2026]",
                "  costs_bps: [10, 25, 50]",
                "  rolling_windows_days: [126, 252]",
                "  circular_block_bootstrap_days: [20, 60]",
                "  cross_asset_role: negative_control_only",
                "artifact_policy:",
                "  policy: slim_card_recorder_first",
                "  workflow_config: workflow.yaml",
                "  recorder_only_artifacts:",
                f"    - artifacts/{spec.experiment}/stage1_daily.csv",
                f"    - artifacts/{spec.experiment}/bootstrap_samples.csv",
                f"    - artifacts/{spec.experiment}/cross_asset_proxy_daily.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    conclusion = _build_conclusion(summary, annual, cost, cross_asset)
    (spec.card_dir / "conclusion.md").write_text(
        conclusion,
        encoding="utf-8",
    )
    return summary


def _build_conclusion(
    summary: dict[str, Any],
    annual: pd.DataFrame,
    cost: pd.DataFrame,
    cross_asset: pd.DataFrame,
) -> str:
    gates = summary["generalization_gates"]
    failed_years = annual.loc[~annual["beats_soxx"], "year"].astype(int).tolist()
    stress_50 = cost.loc[cost["cost_bps"].eq(50.0)].iloc[0]
    return "\n".join(
        [
            "# 中文结论",
            "",
            (
                f"- 结论：{summary['status']}；当前动作是 "
                f"`{summary['decision']}`，真实资金晋升仍为 blocked。"
            ),
            (
                "- 时间外推：冻结模型训练截止 "
                f"{summary['source_training_end']}；2023–2026 中 "
                f"{gates['annual_relative_pass_count']}/4 年跑赢 SOXX，"
                f"{gates['annual_risk_pass_count']}/4 年通过风险门。"
            ),
            (
                f"- 年度不稳定点：未跑赢 SOXX 的年份为 {failed_years}；"
                "说明超额不是逐年稳定存在。"
            ),
            (
                f"- 参数稳定：邻域最差财富相对当前参数为 "
                f"{gates['parameter_min_wealth_ratio']:.4f}，"
                f"gate={gates['parameter_gate']}。"
            ),
            (
                f"- 路径重采样：20/60 日块 bootstrap 的最低跑赢概率为 "
                f"{100.0 * gates['bootstrap_min_outperformance_probability']:.2f}%，"
                f"gate={gates['bootstrap_gate']}。"
            ),
            (
                "- 滚动稳定：126 日和 252 日窗口的跑赢率分别为 "
                f"{100.0 * summary['rolling_summary'][0]['beats_soxx_rate']:.2f}%/"
                f"{100.0 * summary['rolling_summary'][1]['beats_soxx_rate']:.2f}%，"
                f"gate={gates['rolling_gate']}。"
            ),
            (
                f"- 成本压力：10/25bp gate={gates['cost_gate']}；50bp "
                f"return={stress_50['return_pct']:.2f}%，"
                f"MaxDD={stress_50['max_drawdown_pct']:.2f}%。"
            ),
            (
                "- 跨标的负对照：冻结开盘权重时序在 "
                f"{int(cross_asset['beats_underlying'].sum())}/{len(cross_asset)} "
                "个 3x/1x 代理组合中跑赢对应底层，在 "
                f"{int(cross_asset['beats_matched_exposure'].sum())}/{len(cross_asset)} "
                "个组合中跑赢同平均权重基准；这只判断通用 beta 择时，"
                "不能证明半导体模型可直接迁移。"
            ),
            (
                "- 同暴露基准：Stage1 全区间财富相对同平均权重基准为 "
                f"{gates['stage1_terminal_wealth_ratio_vs_matched_exposure']:.4f}，"
                f"gate={gates['matched_exposure_gate']}。"
            ),
            (
                "- 核心限制：2023–2026 虽未参与代码选模，但已经被人工反复"
                "查看，不再是干净最终 holdout；真正泛化只能由后续富途模拟盘"
                "的连续前向样本确认。"
            ),
            f"- qlib recorder：{summary['recorder_id']}",
            "- 下一步：保持 Stage1 参数冻结，累计至少 6 个月模拟盘或一个完整压力阶段，再做一次不可回看晋升评估。",
            "- 不构成买卖建议。",
            "",
        ]
    )
