from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data import QlibProviderConfig, init_qlib_provider, load_qlib_feature_frame
from src.workflow.intraday_recovery_direct_allocation import _risk_metrics


EXPERIMENT = "soxl_trend_clock_core_v1"
PRIMARY_PAIR = ("SOXL", "SOXX", "soxl_soxx_soxs")
PROXY_PAIRS = (
    ("TQQQ", "QQQ", "tqqq"),
    ("UPRO", "SPY", "upro"),
    ("SPXL", "SPY", "spxl"),
    ("TECL", "XLK", "tecl"),
)
DEVELOPMENT = ("2016-01-01", "2022-12-31")
REUSED_OUTER = ("2023-01-01", "2026-12-31")
TREND_CLOCKS = (140, 150, 160)


@dataclass(frozen=True)
class TrendClockProfile:
    name: str
    periods: tuple[int, ...]
    entry_buffer: float
    exit_buffer: float = 0.0


PROFILES = (
    TrendClockProfile("clock_140", (140,), 0.08),
    TrendClockProfile("clock_150", (150,), 0.08),
    TrendClockProfile("clock_160", (160,), 0.08),
    TrendClockProfile("clock_140_150_160", TREND_CLOCKS, 0.08),
    TrendClockProfile("clock_140_150_160_buffer09", TREND_CLOCKS, 0.09),
    TrendClockProfile("clock_140_150_160_buffer10", TREND_CLOCKS, 0.10),
)


@dataclass(frozen=True)
class TrendClockSpec:
    root: Path
    experiment: str = EXPERIMENT
    transaction_cost_bps: float = 10.0
    stress_costs_bps: tuple[float, ...] = (25.0, 50.0)
    selected_profile: str = "clock_140_150_160"
    warmup_start: str = "2013-01-01"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / PRIMARY_PAIR[2]


class FixedTargetWeightModel:
    """Qlib model adapter that exposes causal precomputed target weights."""

    def predict(self, dataset, segment: str = "test") -> pd.Series:
        prepared = dataset.prepare(segment, col_set="feature")
        if isinstance(prepared, pd.DataFrame):
            if isinstance(prepared.columns, pd.MultiIndex):
                prepared.columns = [str(column[-1]) for column in prepared.columns]
            values = prepared["target_weight"]
        else:
            values = prepared
        return pd.to_numeric(values, errors="coerce").rename("score")


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _load_pair(
    root: Path, leveraged: str, underlying: str, provider_name: str, *, start: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    provider_uri = root / "data" / "qlib" / provider_name
    manifest_path = provider_uri / "qlib_store_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"blocked: qlib manifest missing for {leveraged}/{underlying}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured = set(manifest.get("instruments", []))
    if not {leveraged, underlying}.issubset(configured):
        raise ValueError(f"blocked: qlib instruments drifted for {leveraged}/{underlying}")
    config = QlibProviderConfig(
        provider_uri=provider_uri,
        market="all",
        region="us",
        freq="day",
        kernels=1,
    )
    provider_audit = init_qlib_provider(config)
    data = load_qlib_feature_frame(
        config,
        instruments=[leveraged, underlying],
        fields=["$open", "$close"],
        start=start,
        end=str(manifest["calendar_end"]),
    ).reset_index()
    opens = data.pivot(index="datetime", columns="instrument", values="$open").dropna()
    closes = data.pivot(index="datetime", columns="instrument", values="$close").reindex(opens.index)
    frame = pd.DataFrame(index=opens.index)
    frame.index.name = "signal_date"
    frame["underlying_close"] = closes[underlying]
    frame["trade_date"] = frame.index.to_series().shift(-1)
    frame["exit_date"] = frame.index.to_series().shift(-2)
    frame["leveraged_return"] = opens[leveraged].shift(-2) / opens[leveraged].shift(-1) - 1.0
    frame["underlying_return"] = opens[underlying].shift(-2) / opens[underlying].shift(-1) - 1.0
    frame = frame.iloc[:-2].copy()
    for period in sorted({period for profile in PROFILES for period in profile.periods}):
        frame[f"ma_{period}"] = frame["underlying_close"].rolling(
            period, min_periods=period
        ).mean()
    audit = {
        "pair": f"{leveraged}/{underlying}",
        "provider_uri": provider_uri.resolve().as_posix(),
        "provider_as_of": manifest.get("as_of"),
        "calendar_start": manifest.get("calendar_start"),
        "calendar_end": manifest.get("calendar_end"),
        "rows": int(len(frame)),
        "qlib_api": provider_audit["qlib_api"] + ["qlib.data.D.features"],
        "signal_timing": (
            "signal_date close uses only underlying close and trailing moving averages; "
            "target executes from next open to following open"
        ),
    }
    return frame, audit


def replay_trend_clock(
    frame: pd.DataFrame,
    profile: TrendClockProfile,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    """Run one continuous state path; callers may slice metrics afterwards."""

    states = np.zeros(len(profile.periods), dtype="float64")
    previous = np.zeros(2, dtype="float64")
    rows: list[dict[str, Any]] = []
    for signal_date, row in frame.iterrows():
        previous_states = states.copy()
        for index, period in enumerate(profile.periods):
            moving_average = float(row[f"ma_{period}"])
            if not np.isfinite(moving_average):
                continue
            if (
                states[index] == 0.0
                and float(row["underlying_close"])
                > moving_average * (1.0 + profile.entry_buffer)
            ):
                states[index] = 1.0
            elif (
                states[index] == 1.0
                and float(row["underlying_close"])
                < moving_average * (1.0 - profile.exit_buffer)
            ):
                states[index] = 0.0
        soxl_weight = float(states.mean())
        target = np.array([soxl_weight, 1.0 - soxl_weight])
        turnover = float(np.abs(target - previous).sum())
        gross_return = float(
            target[0] * row["leveraged_return"]
            + target[1] * row["underlying_return"]
        )
        rows.append(
            {
                "signal_date": signal_date,
                "trade_date": row["trade_date"],
                "exit_date": row["exit_date"],
                "profile": profile.name,
                "leveraged_weight": target[0],
                "underlying_weight": target[1],
                "turnover": turnover,
                "action": bool(np.any(previous_states != states)),
                "gross_return": gross_return,
                "strategy_return": gross_return - turnover * cost_bps / 10_000.0,
                "cost_bps": float(cost_bps),
            }
        )
        previous = target
    return pd.DataFrame(rows)


def _slice(frame: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    dates = pd.to_datetime(frame["signal_date"])
    return frame[dates.between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))]


def _metrics(
    replay: pd.DataFrame,
    market: pd.DataFrame,
    *,
    pair: str,
    profile: str,
    segment: str,
    cost_bps: float,
) -> dict[str, Any]:
    leveraged = _risk_metrics(market["leveraged_return"].to_numpy(dtype="float64"))
    underlying = _risk_metrics(market["underlying_return"].to_numpy(dtype="float64"))
    candidate = _risk_metrics(replay["strategy_return"].to_numpy(dtype="float64"))
    return {
        "pair": pair,
        "profile": profile,
        "segment": segment,
        "cost_bps": float(cost_bps),
        **candidate,
        "leveraged_buy_hold_return_pct": leveraged["return_pct"],
        "leveraged_buy_hold_max_drawdown_pct": leveraged["max_drawdown_pct"],
        "underlying_buy_hold_return_pct": underlying["return_pct"],
        "return_delta_vs_leveraged_pct_points": (
            candidate["return_pct"] - leveraged["return_pct"]
        ),
        "terminal_wealth_ratio_vs_leveraged": (
            candidate["terminal_wealth"] / leveraged["terminal_wealth"]
        ),
        "max_drawdown_delta_vs_leveraged_pct_points": (
            candidate["max_drawdown_pct"] - leveraged["max_drawdown_pct"]
        ),
        "action_days": int(replay["action"].sum()),
        "total_turnover": float(replay["turnover"].sum()),
        "average_leveraged_weight": float(replay["leveraged_weight"].mean()),
    }


def _build_dataset(targets: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    feature = targets.pivot(
        index="signal_date", columns="instrument", values="target_weight"
    ).stack(future_stack=True).rename("target_weight").to_frame()
    feature.index.names = ["datetime", "instrument"]
    label = targets.pivot(
        index="signal_date", columns="instrument", values="next_open_return"
    ).stack(future_stack=True).rename("next_open_return").to_frame()
    label.index.names = ["datetime", "instrument"]
    static = pd.concat({"feature": feature, "label": label}, axis=1).sort_index()
    handler = DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static),
        infer_processors=[],
        learn_processors=[],
    )
    segments = {
        "train": DEVELOPMENT,
        "test": (
            static.index.get_level_values("datetime").min().date().isoformat(),
            static.index.get_level_values("datetime").max().date().isoformat(),
        ),
    }
    dataset = DatasetH(handler=handler, segments=segments)
    return dataset, {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "segments": {key: list(value) for key, value in segments.items()},
        "segment_rows": {key: int(len(dataset.prepare(key))) for key in segments},
        "features": ["target_weight"],
        "label": "next_open_to_open_return",
    }


def _target_frame(market: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    merged = replay.merge(
        market.reset_index()[
            ["signal_date", "leveraged_return", "underlying_return"]
        ],
        on="signal_date",
        validate="one_to_one",
    )
    levered = merged[
        ["signal_date", "leveraged_weight", "leveraged_return"]
    ].rename(
        columns={
            "leveraged_weight": "target_weight",
            "leveraged_return": "next_open_return",
        }
    )
    levered["instrument"] = PRIMARY_PAIR[0]
    underlying = merged[
        ["signal_date", "underlying_weight", "underlying_return"]
    ].rename(
        columns={
            "underlying_weight": "target_weight",
            "underlying_return": "next_open_return",
        }
    )
    underlying["instrument"] = PRIMARY_PAIR[1]
    return pd.concat([levered, underlying], ignore_index=True).sort_values(
        ["signal_date", "instrument"]
    )


def _qlib_port_config(spec: TrendClockSpec, targets: pd.DataFrame) -> dict[str, Any]:
    return {
        "strategy": {
            "class": "DirectTargetWeightStrategy",
            "module_path": "src.backtest.strategy",
            "kwargs": {
                "signal": "<PRED>",
                "max_weight": 1.0,
                "risk_degree": 1.0,
            },
        },
        "backtest": {
            "start_time": pd.to_datetime(targets["trade_date"]).min().date().isoformat(),
            # Qlib's daily executor needs a following calendar boundary while
            # finishing the last step.  Stop at the final executable trade day
            # instead of the provider's terminal exit-price day.
            "end_time": pd.to_datetime(targets["trade_date"]).max().date().isoformat(),
            "account": 100_000.0,
            "benchmark": PRIMARY_PAIR[0],
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": None,
                "deal_price": "open",
                "open_cost": spec.transaction_cost_bps / 10_000.0,
                "close_cost": spec.transaction_cost_bps / 10_000.0,
                "min_cost": 0.0,
            },
        },
    }


def _record(
    spec: TrendClockSpec,
    *,
    dataset,
    targets: pd.DataFrame,
    selected_replay: pd.DataFrame,
    model_selection: pd.DataFrame,
    comparisons: pd.DataFrame,
    cross_asset: pd.DataFrame,
    dataset_audit: dict[str, Any],
) -> dict[str, Any]:
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

    # Proxy negative controls reinitialize qlib to their own providers.  Restore
    # the primary store explicitly before native Signal/PortAna records.
    init_qlib_provider(
        QlibProviderConfig(
            provider_uri=spec.provider_uri,
            market="all",
            region="us",
            freq="day",
            kernels=1,
        )
    )
    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        temp = Path(tmp)
        selected_replay.to_csv(temp / "portfolio_report.csv", index=False)
        targets.to_csv(temp / "target_weights.csv", index=False)
        model_selection.to_csv(temp / "model_selection_full.csv", index=False)
        comparisons.to_csv(temp / "benchmark_comparison_full.csv", index=False)
        cross_asset.to_csv(temp / "cross_asset_negative_control_full.csv", index=False)
        (temp / "dataset_audit.json").write_text(
            json.dumps(dataset_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                architecture="three_clock_state_ensemble_soxl_soxx",
                periods=",".join(map(str, TREND_CLOCKS)),
                selected_profile=spec.selected_profile,
                signal_time="signal_date_close",
                execution_time="next_market_open",
                selection_uses_outer_test="true_retrospective_search_card",
            )
            selected_dev = comparisons[
                comparisons["profile"].eq(spec.selected_profile)
                & comparisons["pair"].eq("SOXL/SOXX")
                & comparisons["segment"].eq("development_2016_2022")
                & comparisons["cost_bps"].eq(spec.transaction_cost_bps)
            ].iloc[0]
            R.log_metrics(
                development_return_pct=float(selected_dev["return_pct"]),
                development_excess_vs_soxl_pct_points=float(
                    selected_dev["return_delta_vs_leveraged_pct_points"]
                ),
                development_max_drawdown_pct=float(selected_dev["max_drawdown_pct"]),
            )
            SignalRecord(FixedTargetWeightModel(), dataset, recorder).generate()
            SigAnaRecord(recorder).generate()
            PortAnaRecord(
                recorder,
                _qlib_port_config(spec, selected_replay),
                risk_analysis_freq="day",
            ).generate()
            for path in temp.iterdir():
                recorder.log_artifact(path.as_posix(), artifact_path=spec.experiment)
            info = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
            }
    artifact_root = Path(info["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (artifact_root / filename).unlink(missing_ok=True)
    info["recorder_artifacts"] = sorted(
        path.relative_to(Path(info["recorder_local_dir"])).as_posix()
        for path in Path(info["recorder_local_dir"]).rglob("*")
        if path.is_file()
    )
    native_report = pd.read_pickle(
        artifact_root / "portfolio_analysis" / "report_normal_1day.pkl"
    )
    reconciliation: list[dict[str, Any]] = []
    for segment, bounds in (
        ("development_2016_2022", DEVELOPMENT),
        ("outer_reused_2023_2026", REUSED_OUTER),
    ):
        native = native_report.loc[bounds[0] : bounds[1]].copy()
        native_wealth = (1.0 + native["return"].fillna(0.0)).cumprod()
        benchmark_wealth = (1.0 + native["bench"].fillna(0.0)).cumprod()
        manual = comparisons[
            comparisons["profile"].eq(spec.selected_profile)
            & comparisons["pair"].eq("SOXL/SOXX")
            & comparisons["segment"].eq(segment)
            & comparisons["cost_bps"].eq(spec.transaction_cost_bps)
        ].iloc[0]
        native_return = float((native_wealth.iloc[-1] - 1.0) * 100.0)
        native_benchmark_return = float((benchmark_wealth.iloc[-1] - 1.0) * 100.0)
        reconciliation.append(
            {
                "segment": segment,
                "qlib_native_return_pct": native_return,
                "manual_open_to_open_return_pct": float(manual["return_pct"]),
                "qlib_minus_manual_return_pct_points": native_return
                - float(manual["return_pct"]),
                "qlib_native_max_drawdown_pct": float(
                    (native_wealth / native_wealth.cummax() - 1.0).min() * 100.0
                ),
                "manual_open_to_open_max_drawdown_pct": float(
                    manual["max_drawdown_pct"]
                ),
                "qlib_native_benchmark_return_pct": native_benchmark_return,
                "manual_soxl_buy_hold_return_pct": float(
                    manual["leveraged_buy_hold_return_pct"]
                ),
                "qlib_native_excess_vs_benchmark_pct_points": native_return
                - native_benchmark_return,
                "qlib_native_cost_sum_pct_points": float(native["cost"].sum() * 100.0),
                "qlib_native_turnover_sum": float(native["turnover"].sum()),
                "qlib_native_rows": int(len(native)),
            }
        )
    info["qlib_native_reconciliation"] = reconciliation
    return info


def _stage1_overlap(spec: TrendClockSpec, replay: pd.DataFrame) -> dict[str, Any]:
    source = (
        spec.root
        / "research/intraday_on_policy_staged_validation_v1/qlib_recorder"
        / "300782104043818423/f6b09330175e41719886d58ee2d11bec/artifacts"
        / "intraday_on_policy_staged_validation_v1/strategy_returns.csv"
    )
    stage1 = pd.read_csv(source, parse_dates=["signal_date"])
    candidate = stage1[["signal_date", "strategy_return"]].merge(
        replay[["signal_date", "strategy_return"]].rename(
            columns={"strategy_return": "candidate_return"}
        ),
        on="signal_date",
        how="inner",
        validate="one_to_one",
    )
    return {
        "source_recorder_id": "f6b09330175e41719886d58ee2d11bec",
        "start": candidate["signal_date"].min().date().isoformat(),
        "end": candidate["signal_date"].max().date().isoformat(),
        "rows": int(len(candidate)),
        "stage1": _risk_metrics(candidate["strategy_return"].to_numpy()),
        "trend_clock": _risk_metrics(candidate["candidate_return"].to_numpy()),
    }


def run_soxl_trend_clock_core_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    spec = TrendClockSpec(root=_resolve_root(root))
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    market, primary_audit = _load_pair(spec.root, *PRIMARY_PAIR, start=spec.warmup_start)
    profiles = {profile.name: profile for profile in PROFILES}
    replays = {
        (profile.name, cost): replay_trend_clock(market, profile, cost_bps=cost)
        for profile in PROFILES
        for cost in (spec.transaction_cost_bps, *spec.stress_costs_bps)
    }
    comparison_rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        for cost in (spec.transaction_cost_bps, *spec.stress_costs_bps):
            replay = replays[(profile.name, cost)]
            for segment, bounds in (
                ("development_2016_2022", DEVELOPMENT),
                ("outer_reused_2023_2026", REUSED_OUTER),
            ):
                market_section = _slice(market.reset_index(), bounds)
                replay_section = _slice(replay, bounds)
                comparison_rows.append(
                    _metrics(
                        replay_section,
                        market_section,
                        pair="SOXL/SOXX",
                        profile=profile.name,
                        segment=segment,
                        cost_bps=cost,
                    )
                )
    comparisons = pd.DataFrame(comparison_rows)
    development_10 = comparisons[
        comparisons["segment"].eq("development_2016_2022")
        & comparisons["cost_bps"].eq(spec.transaction_cost_bps)
    ].copy()
    development_50 = comparisons[
        comparisons["segment"].eq("development_2016_2022")
        & comparisons["cost_bps"].eq(50.0)
    ].set_index("profile")
    model_selection = development_10.assign(
        stress_50bps_return_pct=lambda table: table["profile"].map(
            development_50["return_pct"]
        ),
        development_return_gate=lambda table: table[
            "return_delta_vs_leveraged_pct_points"
        ].gt(0.0),
        development_risk_gate=lambda table: table[
            "max_drawdown_delta_vs_leveraged_pct_points"
        ].gt(0.0),
        stress_50bps_gate=lambda table: table["profile"].map(
            development_50["return_delta_vs_leveraged_pct_points"]
        ).gt(0.0),
        selection_uses_outer_test=True,
    )
    model_selection["development_gates_pass"] = model_selection[
        ["development_return_gate", "development_risk_gate", "stress_50bps_gate"]
    ].all(axis=1)
    selected = profiles[spec.selected_profile]
    selected_replay = replays[(selected.name, spec.transaction_cost_bps)]

    cross_rows: list[dict[str, Any]] = []
    proxy_audits: list[dict[str, Any]] = []
    for leveraged, underlying, provider_name in PROXY_PAIRS:
        proxy, audit = _load_pair(
            spec.root, leveraged, underlying, provider_name, start=spec.warmup_start
        )
        proxy_audits.append(audit)
        replay = replay_trend_clock(proxy, selected, cost_bps=spec.transaction_cost_bps)
        for segment, bounds in (
            ("development_2016_2022", DEVELOPMENT),
            ("outer_reused_2023_2026", REUSED_OUTER),
        ):
            cross_rows.append(
                _metrics(
                    _slice(replay, bounds),
                    _slice(proxy.reset_index(), bounds),
                    pair=f"{leveraged}/{underlying}",
                    profile=selected.name,
                    segment=segment,
                    cost_bps=spec.transaction_cost_bps,
                )
            )
    cross_asset = pd.DataFrame(cross_rows)
    targets = _target_frame(market, selected_replay)
    dataset, dataset_audit = _build_dataset(targets)
    recorder = _record(
        spec,
        dataset=dataset,
        targets=targets,
        selected_replay=selected_replay,
        model_selection=model_selection,
        comparisons=comparisons,
        cross_asset=cross_asset,
        dataset_audit=dataset_audit,
    )
    selected_metrics = comparisons[comparisons["profile"].eq(selected.name)]
    outer_10 = selected_metrics[
        selected_metrics["segment"].eq("outer_reused_2023_2026")
        & selected_metrics["cost_bps"].eq(spec.transaction_cost_bps)
    ].iloc[0].to_dict()
    proxy_outer = cross_asset[cross_asset["segment"].eq("outer_reused_2023_2026")]
    cross_asset_gate = bool(
        proxy_outer["terminal_wealth_ratio_vs_leveraged"].ge(1.0).all()
    )
    stage1_overlap = _stage1_overlap(spec, selected_replay)
    summary = {
        "experiment": spec.experiment,
        "card_type": "retrospective_qlib_search_card",
        "decision": "not_promoted_outer_contaminated_and_cross_asset_gate_failed",
        "production_change": "blocked",
        "selection_uses_outer_test": True,
        "outer_role": "reused_retrospective_diagnostic_not_clean_holdout",
        "hypothesis": (
            "A slow SOXX trend state should control leverage level: each independent "
            "clock votes SOXL or SOXX, reducing all-or-nothing timing and daily churn."
        ),
        "selected_model": {
            "name": selected.name,
            "periods": list(selected.periods),
            "entry_buffer": selected.entry_buffer,
            "exit_buffer": selected.exit_buffer,
            "signal_time": "signal_date_close",
            "execution_time": "next_market_open",
            "state_continuity": "warmup_2013_then_single_path_without_segment_reset",
            "selection_status": "retrospective_outer_reviewed_not_promotable",
        },
        "development_gates_pass": bool(
            model_selection.loc[
                model_selection["profile"].eq(selected.name), "development_gates_pass"
            ].iloc[0]
        ),
        "outer_soxl_comparison": outer_10,
        "stage1_overlap_comparison": stage1_overlap,
        "cross_asset_outer_gate_pass": cross_asset_gate,
        "cross_asset_outer_pass_count": int(
            proxy_outer["terminal_wealth_ratio_vs_leveraged"].ge(1.0).sum()
        ),
        "cross_asset_outer_pair_count": int(len(proxy_outer)),
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "market": "all",
        "region": "us",
        "as_of": primary_audit["provider_as_of"],
        "data_range": [
            market.index.min().date().isoformat(),
            pd.to_datetime(market["exit_date"]).max().date().isoformat(),
        ],
        "segments": {
            "development": list(DEVELOPMENT),
            "outer_reused": list(REUSED_OUTER),
        },
        "provider_audit": primary_audit,
        "proxy_provider_audits": proxy_audits,
        "dataset_audit": dataset_audit,
        "qlib_native_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R.start",
            "SignalRecord",
            "SigAnaRecord",
            "PortAnaRecord",
            "DirectTargetWeightStrategy",
        ],
        "qlib_docs": {
            "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
            "workflow": "https://qlib.readthedocs.io/en/stable/component/workflow.html",
            "recorder": "https://qlib.readthedocs.io/en/stable/component/recorder.html",
        },
        "sidecars": {
            "selected_model": "selected_model.json",
            "model_selection": "model_selection.csv",
            "benchmark_comparison": "backtest_benchmark_comparison.csv",
            "risk_analysis": "risk_analysis.csv",
            "qlib_native_reconciliation": "qlib_native_reconciliation.csv",
            "cross_asset_negative_control": "cross_asset_negative_control.csv",
            "analysis_summary": "analysis_summary.json",
            "failure_modes": "failure_modes.md",
            "conclusion": "conclusion.md",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": {
                "predictions": "artifacts/pred.pkl",
                "labels": "artifacts/label.pkl",
                "portfolio_report": "artifacts/portfolio_analysis/report_normal_1day.pkl",
                "positions": "artifacts/portfolio_analysis/positions_normal_1day.pkl",
                "risk_analysis": "artifacts/portfolio_analysis/port_analysis_1day.pkl",
                "manual_replay": f"artifacts/{spec.experiment}/portfolio_report.csv",
                "target_weights": f"artifacts/{spec.experiment}/target_weights.csv",
            },
        },
        **recorder,
    }
    _write_card(
        spec,
        summary=summary,
        model_selection=model_selection,
        comparisons=comparisons,
        cross_asset=cross_asset,
    )
    return summary


def _write_card(
    spec: TrendClockSpec,
    *,
    summary: dict[str, Any],
    model_selection: pd.DataFrame,
    comparisons: pd.DataFrame,
    cross_asset: pd.DataFrame,
) -> None:
    card = spec.card_dir
    selected = summary["selected_model"]
    (card / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (card / "selected_model.json").write_text(
        json.dumps(
            {**selected, "production_enabled": False},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model_selection.to_csv(card / "model_selection.csv", index=False)
    comparisons.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    comparisons[
        [
            "pair",
            "profile",
            "segment",
            "cost_bps",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "leveraged_buy_hold_max_drawdown_pct",
            "max_drawdown_delta_vs_leveraged_pct_points",
        ]
    ].to_csv(card / "risk_analysis.csv", index=False)
    pd.DataFrame(summary["qlib_native_reconciliation"]).to_csv(
        card / "qlib_native_reconciliation.csv", index=False
    )
    cross_asset.to_csv(card / "cross_asset_negative_control.csv", index=False)
    analysis = {
        "decision": summary["decision"],
        "primary_result": summary["outer_soxl_comparison"],
        "qlib_native_reconciliation": summary["qlib_native_reconciliation"],
        "stage1_overlap": summary["stage1_overlap_comparison"],
        "cross_asset_outer_gate_pass": summary["cross_asset_outer_gate_pass"],
        "interpretation": (
            "The candidate is a materially stronger low-frequency SOXL retrospective, "
            "but it is not validated because the reused outer influenced the final profile "
            "and the same parameters fail every proxy outer benchmark."
        ),
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        "- `outer contamination`: 2023-2026 SOXL evidence was viewed during architecture refinement.\n"
        "- `cross-asset failure`: the frozen 140/150/160 and 8% rule lagged TQQQ, UPRO, SPXL, and TECL buy-and-hold in their reused outer windows.\n"
        "- `tail remains material`: SOXL drawdown is reduced, not eliminated; a roughly 65% drawdown remains possible.\n"
        "- `regime dependence`: most relative value comes from avoiding leveraged decay in selected semiconductor drawdowns.\n"
        "- `execution`: qlib reports and the explicit open-to-open replay can differ because qlib models cash, exchange fills, and portfolio accounting. Both are retained in the recorder.\n",
        encoding="utf-8",
    )
    workflow = {
        "experiment": spec.experiment,
        "system_of_record": "qlib",
        "provider_uri": summary["provider_uri"],
        "market": "all",
        "region": "us",
        "segments": summary["segments"],
        "model_selection": {
            "selected_profile": selected["name"],
            "selection_uses_outer_test": True,
            "outer_role": summary["outer_role"],
        },
        "execution": {
            "signal_time": selected["signal_time"],
            "execution_time": selected["execution_time"],
            "transaction_cost_bps": spec.transaction_cost_bps,
            "stress_costs_bps": list(spec.stress_costs_bps),
        },
        "recorder_uri": summary["recorder_uri"],
        "recorder_id": summary["recorder_id"],
        "artifact_policy": summary["artifact_policy"],
    }
    (card / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    dev = comparisons[
        comparisons["profile"].eq(selected["name"])
        & comparisons["segment"].eq("development_2016_2022")
        & comparisons["cost_bps"].eq(spec.transaction_cost_bps)
    ].iloc[0]
    outer = summary["outer_soxl_comparison"]
    overlap = summary["stage1_overlap_comparison"]
    native_outer = next(
        row
        for row in summary["qlib_native_reconciliation"]
        if row["segment"] == "outer_reused_2023_2026"
    )
    conclusion = f"""# 中文结论

- 做了什么：停止围绕 Stage1 30% SOXL 上限做小幅日内调参，改成低频核心结构。SOXX 收盘价相对 140/150/160 日均线的三个独立状态时钟各投一票；每个时钟上穿均线 8% 后持 SOXL，跌回均线后降为 SOXX，三票均值形成 0/33%/67%/100% SOXL 权重。信号只用当日收盘已知数据，下一交易日开盘执行。
- 状态和成本修复：2013 年开始预热并连续运行一次状态机，开发/outer 边界不重置；显式回放按真实目标权重变化计双边换手，另由 qlib SignalRecord/PortAnaRecord 生成独立投组证据。
- 开发期 2016–2022：候选 10bp 累计收益 {dev['return_pct']:.2f}%，SOXL 买入持有 {dev['leveraged_buy_hold_return_pct']:.2f}%，超额 {dev['return_delta_vs_leveraged_pct_points']:.2f} 个百分点；MaxDD {dev['max_drawdown_pct']:.2f}% 对 SOXL {dev['leveraged_buy_hold_max_drawdown_pct']:.2f}%；调仓日 {int(dev['action_days'])}。
- 复用 outer 2023–截至 {summary['data_range'][1]}：候选 10bp 累计收益 {outer['return_pct']:.2f}%，SOXL {outer['leveraged_buy_hold_return_pct']:.2f}%，超额 {outer['return_delta_vs_leveraged_pct_points']:.2f} 个百分点；MaxDD {outer['max_drawdown_pct']:.2f}% 对 SOXL {outer['leveraged_buy_hold_max_drawdown_pct']:.2f}%；调仓日 {int(outer['action_days'])}。50bp 压力仍保存在 benchmark sidecar 与 recorder。
- qlib 原生投组对账：同一 outer 的账户收益 {native_outer['qlib_native_return_pct']:.2f}%，原生 SOXL benchmark {native_outer['qlib_native_benchmark_return_pct']:.2f}%，超额 {native_outer['qlib_native_excess_vs_benchmark_pct_points']:.2f} 个百分点，MaxDD {native_outer['qlib_native_max_drawdown_pct']:.2f}%。相对显式 open-to-open 回放的收益差 {native_outer['qlib_minus_manual_return_pct_points']:.2f} 个百分点，来自 qlib 的逐日账户、持仓取整和交易估值口径；两个口径均保留，不混算。
- 与生产 Stage1 的同窗比较（{overlap['start']}–{overlap['end']}）：趋势核心 {overlap['trend_clock']['return_pct']:.2f}%，Stage1 {overlap['stage1']['return_pct']:.2f}%。这回答了此前“只超过一点”的问题：Stage1 的 30% SOXL 上限在强牛市结构上先天限制收益，低频杠杆档位模型才是量级更大的方向。
- 反证：同一冻结参数在 TQQQ/UPRO/SPXL/TECL 的复用 outer 全部未跑赢各自 3x 买入持有，cross-asset gate={summary['cross_asset_outer_gate_pass']}。说明 SOXL outer 超额带半导体周期特异性，不能视为普适稳健性。
- 是否晋升：{summary['decision']}。不改生产模拟盘；原因不是收益不足，而是 outer 已在研发中被查看且跨资产外层失败。该卡是强 retrospective challenger，不是干净可晋升模型。
- 风险：即便改善，历史 MaxDD 仍约 65%；趋势反转、跳空和长时间震荡会产生滞后损失。下一步必须预注册参数并积累新的 forward holdout，或引入截至信号日可得的半导体周期 PIT 信息后重新训练。
- qlib recorder：{summary['recorder_id']}。完整预测、标签、positions、portfolio report、手工逐日回放和目标权重只在 recorder，卡根保持 slim。
- 不构成买卖建议。
"""
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")


def clean_recorder_uploads(card_dir: Path) -> None:
    """Compatibility helper for tests and interrupted notebook runs."""

    uploads = card_dir / ".recorder_uploads"
    if uploads.exists():
        shutil.rmtree(uploads)
