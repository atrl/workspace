from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.intraday_bar_model import _recorder_files, _remove_transient_code_snapshots


EXPERIMENT = "safe_haven_tail_hedge_v1"
QLIB_DOCS = {
    "data": "https://qlib.readthedocs.io/en/latest/component/data.html",
    "workflow": "https://qlib.readthedocs.io/en/latest/component/workflow.html",
    "recorder": "https://qlib.readthedocs.io/en/latest/component/recorder.html",
}
QLIB_NATIVE_API = [
    "qlib.init",
    "qlib.data.D.features",
    "qlib.workflow.R.start",
    "qlib.workflow.R.log_params",
    "qlib.workflow.R.log_metrics",
    "qlib.workflow.Recorder.log_artifact",
]
RAW_SNAPSHOT_ARTIFACTS = {
    "option_daily_bars": "data/raw/options/polygon_soxl_safe_haven_daily/option_daily_bars.parquet",
    "selected_contracts": "data/raw/options/polygon_soxl_safe_haven_daily/selected_contracts.csv",
    "contracts": "data/raw/options/polygon_soxl_safe_haven_daily/contracts.parquet",
}
RECORDER_ONLY_ARTIFACTS = {
    "same_day_option_trace": "artifacts/safe_haven_tail_hedge_v1/same_day_option_trace.csv",
}


@dataclass(frozen=True)
class SafeHavenTailHedgeSpec:
    root: Path
    experiment: str = EXPERIMENT
    underlying: str = "SOXL"
    provider_uri_name: str = "soxl"
    option_snapshot_name: str = "polygon_soxl_safe_haven_daily"
    start: str = "2026-01-02"
    end: str = "2026-07-02"
    as_of: str = "2026-07-02"
    event_date: str = "2026-07-02"
    pre_event_entry_start: str = "2026-06-17"
    region: str = "us"
    market: str = "all"
    freq: str = "day"
    targets: tuple[str, ...] = ("put_80", "put_70", "put_60")
    funding_budgets: tuple[float, ...] = (0.005, 0.01, 0.02)
    option_roundtrip_spread_pct: float = 0.03
    safe_haven_min_net_multiple: float = 10.0

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / self.provider_uri_name

    @property
    def recorder_uri(self) -> Path:
        return self.card_dir / "qlib_recorder"

    @property
    def option_snapshot_dir(self) -> Path:
        return self.root / "data" / "raw" / "options" / self.option_snapshot_name


def run_safe_haven_tail_hedge_card(*, root: str | Path | None = None) -> dict[str, Any]:
    spec = SafeHavenTailHedgeSpec(root=_resolve_root(root))
    return _run_card(spec)


def _resolve_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2]


def _run_card(spec: SafeHavenTailHedgeSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    workflow_yaml = _write_workflow_yaml(spec)
    spot = _load_spot_frame(spec)
    selections, bars, option_manifest = _load_option_snapshot(spec)
    diagnostics = build_safe_haven_diagnostics(spec, spot=spot, selections=selections, bars=bars)
    sidecars = _write_sidecars(spec, diagnostics=diagnostics)

    import qlib
    from qlib.workflow import R

    _ = qlib
    spec.recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(spec.recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        same_day_path = tmp_dir / "same_day_option_trace.csv"
        diagnostics["same_day_trace"].to_csv(same_day_path, index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                experiment=spec.experiment,
                provider_uri=spec.provider_uri.resolve().as_posix(),
                option_snapshot_dir=spec.option_snapshot_dir.resolve().as_posix(),
                execution_basis="SOXL open_to_close plus option open_to_close daily aggregate",
                targets=",".join(spec.targets),
                funding_budgets=",".join(str(value) for value in spec.funding_budgets),
                option_roundtrip_spread_pct=float(spec.option_roundtrip_spread_pct),
                start=spec.start,
                end=spec.end,
                as_of=spec.as_of,
                event_date=spec.event_date,
                pre_event_entry_start=spec.pre_event_entry_start,
            )
            R.log_metrics(**_recorder_metrics(diagnostics))
            recorder.log_artifact(same_day_path.as_posix(), artifact_path=spec.experiment)
            for rel_path in sidecars.values():
                recorder.log_artifact((spec.card_dir / rel_path).as_posix(), artifact_path=spec.experiment)
            recorder_id = recorder.id
            recorder_local_dir = Path(recorder.get_local_dir()).resolve()
            _remove_transient_code_snapshots(recorder_local_dir)

    summary = _summary_payload(
        spec,
        workflow_yaml=workflow_yaml,
        option_manifest=option_manifest,
        diagnostics=diagnostics,
        sidecars=sidecars,
        recorder_id=recorder_id,
        recorder_local_dir=recorder_local_dir,
        recorder_artifacts=_recorder_files(recorder_local_dir),
    )
    (spec.card_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    (spec.card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")
    return summary


def build_safe_haven_diagnostics(
    spec: SafeHavenTailHedgeSpec,
    *,
    spot: pd.DataFrame,
    selections: pd.DataFrame,
    bars: pd.DataFrame,
) -> dict[str, Any]:
    spot_frame = _normalize_spot_frame(spot, spec)
    selections_frame = _filter_selections(selections, spec)
    bars_frame, future_bar_rows_excluded = _filter_bars(bars, spec)
    same_day = _same_day_trace(spot_frame, selections_frame, bars_frame, spec)
    quote_coverage = _quote_coverage(same_day, spec)
    benchmark = _return_metrics(spot_frame["soxl_open_close_return"].to_numpy(dtype="float64"))
    benchmark.update(
        {
            "model": "soxl_open_close_benchmark",
            "target": "",
            "funding_budget": 0.0,
            "role": "benchmark",
        }
    )
    model_comparison = _model_comparison(same_day, benchmark=benchmark, spec=spec)
    backtest = _backtest_benchmark_comparison(model_comparison, benchmark=benchmark)
    event_trace = _event_trace(same_day, spec)
    tail_payoff_cases = _tail_payoff_cases(selections_frame, bars_frame, spec)
    tail_payoff_summary = _tail_payoff_summary(tail_payoff_cases)
    safe_haven_screen = _safe_haven_screen(
        same_day,
        model_comparison,
        tail_payoff_cases,
        spec=spec,
        benchmark=benchmark,
    )
    risk_analysis = _risk_analysis(
        quote_coverage,
        tail_payoff_summary,
        safe_haven_screen=safe_haven_screen,
        future_bar_rows_excluded=future_bar_rows_excluded,
    )
    analysis_summary = _analysis_summary(
        spec,
        spot_frame=spot_frame,
        selections_frame=selections_frame,
        bars_frame=bars_frame,
        future_bar_rows_excluded=future_bar_rows_excluded,
        quote_coverage=quote_coverage,
        model_comparison=model_comparison,
        event_trace=event_trace,
        tail_payoff_cases=tail_payoff_cases,
        tail_payoff_summary=tail_payoff_summary,
        safe_haven_screen=safe_haven_screen,
        benchmark=benchmark,
    )
    return {
        "analysis_summary": analysis_summary,
        "backtest_benchmark_comparison": backtest,
        "event_trace": event_trace,
        "model_comparison": model_comparison,
        "quote_coverage": quote_coverage,
        "risk_analysis": risk_analysis,
        "safe_haven_screen": safe_haven_screen,
        "same_day_trace": same_day,
        "selected_model": _selected_model_payload(spec, safe_haven_screen=safe_haven_screen),
        "tail_payoff_cases": tail_payoff_cases,
        "tail_payoff_summary": tail_payoff_summary,
    }


def _load_spot_frame(spec: SafeHavenTailHedgeSpec) -> pd.DataFrame:
    qframe = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.provider_uri,
            region=spec.region,
            market=spec.market,
            freq=spec.freq,
            kernels=1,
        ),
        instruments=[spec.underlying],
        fields=["$open", "$close"],
        start=spec.start,
        end=spec.end,
    )
    frame = qframe.reset_index().rename(columns={"datetime": "date", "$open": "open", "$close": "close"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    return frame[["date", "instrument", "open", "close"]].copy()


def _load_option_snapshot(spec: SafeHavenTailHedgeSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_path = spec.option_snapshot_dir / "selected_contracts.csv"
    bars_path = spec.option_snapshot_dir / "option_daily_bars.parquet"
    manifest_path = spec.option_snapshot_dir / "manifest.json"
    missing = [path.as_posix() for path in [selected_path, bars_path, manifest_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing option snapshot artifacts: {missing}")
    selections = pd.read_csv(selected_path)
    bars = pd.read_parquet(bars_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return selections, bars, manifest


def _normalize_spot_frame(spot: pd.DataFrame, spec: SafeHavenTailHedgeSpec) -> pd.DataFrame:
    required = {"date", "open", "close"}
    missing = sorted(required.difference(spot.columns))
    if missing:
        raise ValueError(f"spot missing columns: {missing}")
    frame = spot.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame = frame.loc[(frame["date"] >= spec.start) & (frame["date"] <= spec.end)].copy()
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["open", "close"]).sort_values("date").reset_index(drop=True)
    frame["soxl_open_close_return"] = frame["close"] / frame["open"] - 1.0
    return frame


def _filter_selections(selections: pd.DataFrame, spec: SafeHavenTailHedgeSpec) -> pd.DataFrame:
    required = {"date", "target", "option_ticker", "strike_to_open", "dte", "strike_price"}
    missing = sorted(required.difference(selections.columns))
    if missing:
        raise ValueError(f"selected_contracts missing columns: {missing}")
    frame = selections.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame = frame.loc[
        (frame["date"] >= spec.start) & (frame["date"] <= spec.end) & frame["target"].isin(spec.targets)
    ].copy()
    frame["dte"] = pd.to_numeric(frame["dte"], errors="coerce")
    frame["strike_to_open"] = pd.to_numeric(frame["strike_to_open"], errors="coerce")
    frame["strike_price"] = pd.to_numeric(frame["strike_price"], errors="coerce")
    return frame.sort_values(["date", "target"]).reset_index(drop=True)


def _filter_bars(bars: pd.DataFrame, spec: SafeHavenTailHedgeSpec) -> tuple[pd.DataFrame, int]:
    required = {"date", "option_ticker", "open", "close", "high", "low", "volume", "transactions"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"option_daily_bars missing columns: {missing}")
    frame = bars.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    future_rows = int((frame["date"] > spec.as_of).sum())
    frame = frame.loc[(frame["date"] >= spec.start) & (frame["date"] <= spec.as_of)].copy()
    for column in ["open", "close", "high", "low", "volume", "transactions"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["option_ticker", "date"]).reset_index(drop=True), future_rows


def _same_day_trace(
    spot: pd.DataFrame,
    selections: pd.DataFrame,
    bars: pd.DataFrame,
    spec: SafeHavenTailHedgeSpec,
) -> pd.DataFrame:
    joined = selections.merge(
        bars[
            [
                "date",
                "option_ticker",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "transactions",
            ]
        ].rename(
            columns={
                "open": "option_open",
                "close": "option_close",
                "high": "option_high",
                "low": "option_low",
                "volume": "option_volume",
                "transactions": "option_transactions",
            }
        ),
        on=["date", "option_ticker"],
        how="left",
    )
    joined = joined.merge(
        spot[["date", "open", "close", "soxl_open_close_return"]].rename(
            columns={"open": "soxl_open", "close": "soxl_close"}
        ),
        on="date",
        how="left",
    )
    executable = joined["option_open"].notna() & (joined["option_open"] > 0) & joined["option_close"].notna()
    joined["option_raw_return"] = np.where(executable, joined["option_close"] / joined["option_open"] - 1.0, np.nan)
    joined["option_cost_return"] = np.where(
        executable,
        ((1.0 - spec.option_roundtrip_spread_pct) * joined["option_close"])
        / ((1.0 + spec.option_roundtrip_spread_pct) * joined["option_open"])
        - 1.0,
        np.nan,
    )
    joined["has_same_day_quote"] = executable.astype(bool)
    return joined.sort_values(["date", "target"]).reset_index(drop=True)


def _quote_coverage(same_day: pd.DataFrame, spec: SafeHavenTailHedgeSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in spec.targets:
        frame = same_day.loc[same_day["target"].eq(target)].copy()
        quoted = frame.loc[frame["has_same_day_quote"]]
        rows.append(
            {
                "target": target,
                "selection_days": int(len(frame)),
                "quote_days": int(len(quoted)),
                "quote_rate": float(len(quoted) / len(frame)) if len(frame) else 0.0,
                "median_volume": float(quoted["option_volume"].median()) if not quoted.empty else np.nan,
                "median_transactions": float(quoted["option_transactions"].median()) if not quoted.empty else np.nan,
                "avg_option_open": float(quoted["option_open"].mean()) if not quoted.empty else np.nan,
                "avg_strike_to_open": float(frame["strike_to_open"].mean()) if not frame.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _model_comparison(
    same_day: pd.DataFrame,
    *,
    benchmark: dict[str, Any],
    spec: SafeHavenTailHedgeSpec,
) -> pd.DataFrame:
    rows = [dict(benchmark)]
    for target in spec.targets:
        frame = same_day.loc[same_day["target"].eq(target)].sort_values("date").copy()
        base_returns = frame["soxl_open_close_return"].fillna(0.0).to_numpy(dtype="float64")
        option_returns = frame["option_cost_return"].to_numpy(dtype="float64")
        has_quote = frame["has_same_day_quote"].to_numpy(dtype=bool)
        for budget in spec.funding_budgets:
            funded_returns = base_returns.copy()
            funded_returns[has_quote] = (1.0 - budget) * base_returns[has_quote] + budget * option_returns[has_quote]
            metrics = _return_metrics(funded_returns)
            rows.append(
                {
                    **metrics,
                    "model": f"{budget:.1%}_funded_{target}_daily_cost{spec.option_roundtrip_spread_pct:.0%}",
                    "target": target,
                    "funding_budget": float(budget),
                    "role": "daily_funded_otm_put_sleeve",
                    "quote_days_used": int(has_quote.sum()),
                    "selection_days": int(len(frame)),
                    "quote_rate": float(has_quote.mean()) if len(frame) else 0.0,
                    "excess_return_vs_soxl_pct": float(metrics["total_return_pct"] - benchmark["total_return_pct"]),
                    "drawdown_improvement_vs_soxl_pct": float(metrics["max_drawdown_pct"] - benchmark["max_drawdown_pct"]),
                }
            )
    return pd.DataFrame(rows)


def _backtest_benchmark_comparison(model_comparison: pd.DataFrame, *, benchmark: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in model_comparison.to_dict("records"):
        rows.append(
            {
                "model": row["model"],
                "benchmark": "soxl_open_close_benchmark",
                "model_return_pct": float(row["total_return_pct"]),
                "benchmark_return_pct": float(benchmark["total_return_pct"]),
                "return_gap_pct": float(row["total_return_pct"] - benchmark["total_return_pct"]),
                "model_max_drawdown_pct": float(row["max_drawdown_pct"]),
                "benchmark_max_drawdown_pct": float(benchmark["max_drawdown_pct"]),
                "drawdown_gap_pct": float(row["max_drawdown_pct"] - benchmark["max_drawdown_pct"]),
                "beats_return": bool(row["total_return_pct"] > benchmark["total_return_pct"]),
                "beats_drawdown": bool(row["max_drawdown_pct"] > benchmark["max_drawdown_pct"]),
                "role": row["role"],
            }
        )
    return pd.DataFrame(rows)


def _event_trace(same_day: pd.DataFrame, spec: SafeHavenTailHedgeSpec) -> pd.DataFrame:
    frame = same_day.loc[same_day["date"].eq(spec.event_date)].copy()
    if frame.empty:
        return pd.DataFrame()
    for budget in spec.funding_budgets:
        improvement = budget * (frame["option_cost_return"] - frame["soxl_open_close_return"])
        frame[f"{budget:.1%}_portfolio_improvement_pct"] = improvement * 100.0
    return frame[
        [
            "date",
            "target",
            "option_ticker",
            "strike_price",
            "strike_to_open",
            "dte",
            "soxl_open",
            "soxl_close",
            "soxl_open_close_return",
            "option_open",
            "option_close",
            "option_volume",
            "option_transactions",
            "option_raw_return",
            "option_cost_return",
            "has_same_day_quote",
            *[f"{budget:.1%}_portfolio_improvement_pct" for budget in spec.funding_budgets],
        ]
    ].reset_index(drop=True)


def _tail_payoff_cases(selections: pd.DataFrame, bars: pd.DataFrame, spec: SafeHavenTailHedgeSpec) -> pd.DataFrame:
    entry = selections.loc[
        (selections["date"] >= spec.pre_event_entry_start) & (selections["date"] < spec.event_date)
    ].copy()
    if entry.empty:
        return pd.DataFrame()
    entry_bars = bars.rename(
        columns={
            "date": "entry_date",
            "open": "entry_open",
            "volume": "entry_volume",
            "transactions": "entry_transactions",
        }
    )[["entry_date", "option_ticker", "entry_open", "entry_volume", "entry_transactions"]]
    exit_bars = bars.loc[bars["date"].eq(spec.event_date)].rename(
        columns={
            "close": "exit_close",
            "volume": "exit_volume",
            "transactions": "exit_transactions",
        }
    )[["option_ticker", "exit_close", "exit_volume", "exit_transactions"]]
    cases = entry.rename(columns={"date": "entry_date"}).merge(
        entry_bars,
        on=["entry_date", "option_ticker"],
        how="inner",
    )
    cases = cases.merge(exit_bars, on="option_ticker", how="inner")
    if cases.empty:
        return cases
    executable = (cases["entry_open"] > 0) & cases["exit_close"].notna()
    cases = cases.loc[executable].copy()
    cases["gross_multiple"] = cases["exit_close"] / cases["entry_open"]
    cases["net_cost_multiple"] = (
        (1.0 - spec.option_roundtrip_spread_pct) * cases["exit_close"]
    ) / ((1.0 + spec.option_roundtrip_spread_pct) * cases["entry_open"])
    return cases.sort_values("net_cost_multiple", ascending=False).reset_index(drop=True)


def _tail_payoff_summary(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(
            columns=[
                "target",
                "case_count",
                "median_net_multiple",
                "max_net_multiple",
                "median_entry_volume",
                "median_exit_volume",
            ]
        )
    rows: list[dict[str, Any]] = []
    for target, group in cases.groupby("target", sort=True):
        rows.append(
            {
                "target": target,
                "case_count": int(len(group)),
                "median_net_multiple": float(group["net_cost_multiple"].median()),
                "max_net_multiple": float(group["net_cost_multiple"].max()),
                "median_entry_volume": float(group["entry_volume"].median()),
                "median_exit_volume": float(group["exit_volume"].median()),
            }
        )
    return pd.DataFrame(rows)


def _safe_haven_screen(
    same_day: pd.DataFrame,
    model_comparison: pd.DataFrame,
    tail_payoff_cases: pd.DataFrame,
    *,
    spec: SafeHavenTailHedgeSpec,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    event = same_day.loc[same_day["date"].eq(spec.event_date)].copy()
    event_put80 = event.loc[event["target"].eq("put_80")]
    event_put80_cost_return = (
        float(event_put80["option_cost_return"].iloc[0]) if not event_put80.empty and event_put80["option_cost_return"].notna().any() else np.nan
    )
    event_soxl_return = (
        float(event["soxl_open_close_return"].dropna().iloc[0]) if event["soxl_open_close_return"].notna().any() else np.nan
    )
    max_net_multiple = float(tail_payoff_cases["net_cost_multiple"].max()) if not tail_payoff_cases.empty else 0.0
    event_targets_with_quotes = sorted(event.loc[event["has_same_day_quote"], "target"].astype(str).tolist())
    best_daily = model_comparison.loc[model_comparison["role"].eq("daily_funded_otm_put_sleeve")].copy()
    best_dd_improvement = float(best_daily["drawdown_improvement_vs_soxl_pct"].max()) if not best_daily.empty else 0.0
    best_return_gap = float(best_daily["excess_return_vs_soxl_pct"].max()) if not best_daily.empty else 0.0
    required_event_option_return_for_5pp_1pct = (
        float(event_soxl_return + 0.05 / 0.01) if np.isfinite(event_soxl_return) else np.nan
    )
    return {
        "promotion_decision": "not_promoted_convexity_insufficient",
        "benchmark_total_return_pct": float(benchmark["total_return_pct"]),
        "benchmark_max_drawdown_pct": float(benchmark["max_drawdown_pct"]),
        "event_date": spec.event_date,
        "event_soxl_open_close_return_pct": float(event_soxl_return * 100.0) if np.isfinite(event_soxl_return) else np.nan,
        "event_put80_cost_return_pct": float(event_put80_cost_return * 100.0)
        if np.isfinite(event_put80_cost_return)
        else np.nan,
        "event_targets_with_same_day_quotes": event_targets_with_quotes,
        "max_pre_event_net_multiple_to_event": max_net_multiple,
        "safe_haven_min_net_multiple": float(spec.safe_haven_min_net_multiple),
        "passes_safe_haven_convexity": bool(max_net_multiple >= spec.safe_haven_min_net_multiple),
        "best_daily_drawdown_improvement_pct": best_dd_improvement,
        "best_daily_return_gap_pct": best_return_gap,
        "required_event_option_return_for_5pp_improvement_with_1pct_budget_pct": float(
            required_event_option_return_for_5pp_1pct * 100.0
        )
        if np.isfinite(required_event_option_return_for_5pp_1pct)
        else np.nan,
        "diagnosis": "SOXL OTM puts were expensive and not deep-convex enough into the 2026-07-02 selloff; daily aggregate liquidity is sparse for deeper strikes.",
    }


def _risk_analysis(
    quote_coverage: pd.DataFrame,
    tail_payoff_summary: pd.DataFrame,
    *,
    safe_haven_screen: dict[str, Any],
    future_bar_rows_excluded: int,
) -> pd.DataFrame:
    rows = [
        {
            "risk": "convexity_shortfall",
            "status": "failed",
            "evidence": f"max net multiple {safe_haven_screen['max_pre_event_net_multiple_to_event']:.2f}x below required {safe_haven_screen['safe_haven_min_net_multiple']:.2f}x",
        },
        {
            "risk": "same_day_liquidity_sparse",
            "status": "active",
            "evidence": quote_coverage.to_json(orient="records", force_ascii=True),
        },
        {
            "risk": "future_leakage_control",
            "status": "controlled",
            "evidence": f"excluded {future_bar_rows_excluded} raw option bar rows after as_of before analysis",
        },
    ]
    if not tail_payoff_summary.empty:
        rows.append(
            {
                "risk": "tail_payoff_liquidity",
                "status": "active",
                "evidence": tail_payoff_summary.to_json(orient="records", force_ascii=True),
            }
        )
    return pd.DataFrame(rows)


def _analysis_summary(
    spec: SafeHavenTailHedgeSpec,
    *,
    spot_frame: pd.DataFrame,
    selections_frame: pd.DataFrame,
    bars_frame: pd.DataFrame,
    future_bar_rows_excluded: int,
    quote_coverage: pd.DataFrame,
    model_comparison: pd.DataFrame,
    event_trace: pd.DataFrame,
    tail_payoff_cases: pd.DataFrame,
    tail_payoff_summary: pd.DataFrame,
    safe_haven_screen: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": spec.experiment,
        "source": "qlib_spot_plus_polygon_option_daily_snapshot",
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "option_snapshot_dir": spec.option_snapshot_dir.resolve().as_posix(),
        "as_of": spec.as_of,
        "event_date": spec.event_date,
        "execution_basis": "SOXL open_to_close with option daily open_to_close aggregate",
        "option_roundtrip_spread_pct": float(spec.option_roundtrip_spread_pct),
        "rows": {
            "spot_days": int(len(spot_frame)),
            "selection_rows": int(len(selections_frame)),
            "option_bar_rows_used": int(len(bars_frame)),
            "future_option_bar_rows_excluded": int(future_bar_rows_excluded),
            "tail_payoff_cases": int(len(tail_payoff_cases)),
        },
        "benchmark": benchmark,
        "quote_coverage": quote_coverage.to_dict("records"),
        "event_trace": event_trace.to_dict("records"),
        "tail_payoff_summary": tail_payoff_summary.to_dict("records"),
        "best_daily_sleeve": _best_sleeve(model_comparison),
        "safe_haven_screen": safe_haven_screen,
        "qlib_native_api": QLIB_NATIVE_API,
        "qlib_docs": QLIB_DOCS,
    }


def _selected_model_payload(spec: SafeHavenTailHedgeSpec, *, safe_haven_screen: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_model": "none",
        "strategy_class": "diagnostic_safe_haven_otm_put_sleeve",
        "promotion_decision": safe_haven_screen["promotion_decision"],
        "selection_reason": "No candidate passed the pre-declared convexity screen.",
        "targets": list(spec.targets),
        "funding_budgets": list(spec.funding_budgets),
        "safe_haven_min_net_multiple": float(spec.safe_haven_min_net_multiple),
    }


def _best_sleeve(model_comparison: pd.DataFrame) -> dict[str, Any]:
    sleeves = model_comparison.loc[model_comparison["role"].eq("daily_funded_otm_put_sleeve")].copy()
    if sleeves.empty:
        return {}
    row = sleeves.sort_values(["max_drawdown_pct", "total_return_pct"], ascending=[False, False]).iloc[0]
    return row.to_dict()


def _return_metrics(returns: np.ndarray) -> dict[str, float]:
    clean = np.nan_to_num(np.asarray(returns, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    equity = np.cumprod(1.0 + clean)
    if len(equity) == 0:
        return {
            "days": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "worst_day_pct": 0.0,
            "best_day_pct": 0.0,
            "avg_day_pct": 0.0,
            "annual_vol_pct": 0.0,
            "sharpe": 0.0,
        }
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    vol = float(np.std(clean, ddof=1) * np.sqrt(252.0)) if len(clean) > 1 else 0.0
    avg = float(np.mean(clean))
    sharpe = float(avg / np.std(clean, ddof=1) * np.sqrt(252.0)) if len(clean) > 1 and np.std(clean, ddof=1) > 0 else 0.0
    return {
        "days": float(len(clean)),
        "total_return_pct": float((equity[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(np.min(drawdown) * 100.0),
        "worst_day_pct": float(np.min(clean) * 100.0),
        "best_day_pct": float(np.max(clean) * 100.0),
        "avg_day_pct": float(avg * 100.0),
        "annual_vol_pct": float(vol * 100.0),
        "sharpe": sharpe,
    }


def _recorder_metrics(diagnostics: dict[str, Any]) -> dict[str, float]:
    screen = diagnostics["safe_haven_screen"]
    return {
        "benchmark_total_return_pct": float(screen["benchmark_total_return_pct"]),
        "benchmark_max_drawdown_pct": float(screen["benchmark_max_drawdown_pct"]),
        "event_soxl_open_close_return_pct": float(screen["event_soxl_open_close_return_pct"]),
        "event_put80_cost_return_pct": float(screen["event_put80_cost_return_pct"]),
        "max_pre_event_net_multiple_to_event": float(screen["max_pre_event_net_multiple_to_event"]),
        "best_daily_drawdown_improvement_pct": float(screen["best_daily_drawdown_improvement_pct"]),
        "best_daily_return_gap_pct": float(screen["best_daily_return_gap_pct"]),
    }


def _write_sidecars(spec: SafeHavenTailHedgeSpec, *, diagnostics: dict[str, Any]) -> dict[str, str]:
    sidecars = {
        "analysis_summary": "analysis_summary.json",
        "backtest_benchmark_comparison": "backtest_benchmark_comparison.csv",
        "event_trace": "event_trace.csv",
        "failure_modes": "failure_modes.md",
        "model_comparison": "model_comparison.csv",
        "quote_coverage": "quote_coverage.csv",
        "risk_analysis": "risk_analysis.csv",
        "selected_model": "selected_model.json",
        "tail_payoff_cases": "tail_payoff_cases.csv",
        "tail_payoff_summary": "tail_payoff_summary.csv",
    }
    (spec.card_dir / sidecars["analysis_summary"]).write_text(
        json.dumps(_json_ready(diagnostics["analysis_summary"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnostics["backtest_benchmark_comparison"].to_csv(spec.card_dir / sidecars["backtest_benchmark_comparison"], index=False)
    diagnostics["event_trace"].to_csv(spec.card_dir / sidecars["event_trace"], index=False)
    (spec.card_dir / sidecars["failure_modes"]).write_text(_failure_modes(diagnostics), encoding="utf-8")
    diagnostics["model_comparison"].to_csv(spec.card_dir / sidecars["model_comparison"], index=False)
    diagnostics["quote_coverage"].to_csv(spec.card_dir / sidecars["quote_coverage"], index=False)
    diagnostics["risk_analysis"].to_csv(spec.card_dir / sidecars["risk_analysis"], index=False)
    (spec.card_dir / sidecars["selected_model"]).write_text(
        json.dumps(_json_ready(diagnostics["selected_model"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    diagnostics["tail_payoff_cases"].to_csv(spec.card_dir / sidecars["tail_payoff_cases"], index=False)
    diagnostics["tail_payoff_summary"].to_csv(spec.card_dir / sidecars["tail_payoff_summary"], index=False)
    return sidecars


def _write_workflow_yaml(spec: SafeHavenTailHedgeSpec) -> str:
    text = "\n".join(
        [
            f"experiment: {spec.experiment}",
            "workflow_type: safe_haven_tail_hedge_diagnostic",
            f"provider_uri: {spec.provider_uri.resolve().as_posix()}",
            f"option_snapshot_dir: {spec.option_snapshot_dir.resolve().as_posix()}",
            f"recorder_uri: {spec.recorder_uri.resolve().as_posix()}",
            f"market: {spec.market}",
            f"region: {spec.region}",
            f"freq: {spec.freq}",
            f"as_of: {spec.as_of}",
            f"event_date: {spec.event_date}",
            "execution_basis: soxl_open_to_close_plus_option_daily_open_to_close",
            "targets:",
            *[f"  - {target}" for target in spec.targets],
            "funding_budgets:",
            *[f"  - {budget}" for budget in spec.funding_budgets],
            "artifact_policy:",
            "  policy: slim_card_recorder_first",
            f"  workflow_config: workflow.yaml",
            "  recorder_only_artifacts:",
            *[f"    {key}: {value}" for key, value in RECORDER_ONLY_ARTIFACTS.items()],
            "  raw_snapshot_artifacts:",
            *[f"    {key}: {value}" for key, value in RAW_SNAPSHOT_ARTIFACTS.items()],
            "",
        ]
    )
    (spec.card_dir / "workflow.yaml").write_text(text, encoding="utf-8")
    return "workflow.yaml"


def _summary_payload(
    spec: SafeHavenTailHedgeSpec,
    *,
    workflow_yaml: str,
    option_manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    sidecars: dict[str, str],
    recorder_id: str,
    recorder_local_dir: Path,
    recorder_artifacts: list[str],
) -> dict[str, Any]:
    return {
        "experiment": spec.experiment,
        "card_type": "safe_haven_tail_hedge_diagnostic",
        "promotion_decision": diagnostics["safe_haven_screen"]["promotion_decision"],
        "provider_uri": spec.provider_uri.resolve().as_posix(),
        "option_snapshot_dir": spec.option_snapshot_dir.resolve().as_posix(),
        "option_snapshot_manifest_path": (spec.option_snapshot_dir / "manifest.json").resolve().as_posix(),
        "option_snapshot_manifest": _compact_option_manifest(option_manifest),
        "recorder_uri": spec.recorder_uri.resolve().as_posix(),
        "recorder_id": recorder_id,
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "recorder_artifacts": recorder_artifacts,
        "workflow_config": workflow_yaml,
        "sidecars": sidecars,
        "analysis_summary": diagnostics["analysis_summary"],
        "selected_model": diagnostics["selected_model"],
        "safe_haven_screen": diagnostics["safe_haven_screen"],
        "qlib_docs": QLIB_DOCS,
        "qlib_native_api": QLIB_NATIVE_API,
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "primary_artifact": f"{spec.experiment}.ipynb",
            "machine_entrypoint": "summary.json",
            "workflow_config": workflow_yaml,
            "kept_sidecars": list(sidecars.values()),
            "recorder_only_artifacts": RECORDER_ONLY_ARTIFACTS,
            "raw_snapshot_artifacts": RAW_SNAPSHOT_ARTIFACTS,
            "reproducible_from_recorder": True,
        },
    }


def _compact_option_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    audits = manifest.get("audits", {})
    bar_audits = audits.get("bar_requests", []) or []
    contract_audits = audits.get("contract_requests", []) or []
    bar_status_counts = pd.Series([str(row.get("status", "")) for row in bar_audits]).value_counts().to_dict()
    contract_status_counts = pd.Series([str(row.get("status", "")) for row in contract_audits]).value_counts().to_dict()
    return {
        "schema_version": manifest.get("schema_version"),
        "provider": manifest.get("provider"),
        "underlying": manifest.get("underlying"),
        "requested_start": manifest.get("requested_start"),
        "requested_end": manifest.get("requested_end"),
        "fetch_until": manifest.get("fetch_until"),
        "target_specs": manifest.get("target_specs", []),
        "min_dte": manifest.get("min_dte"),
        "max_dte": manifest.get("max_dte"),
        "created_at": manifest.get("created_at"),
        "contracts": manifest.get("contracts", {}),
        "selections": manifest.get("selections", {}),
        "bars": manifest.get("bars", {}),
        "bar_request_status_counts": {str(key): int(value) for key, value in bar_status_counts.items()},
        "contract_request_status_counts": {str(key): int(value) for key, value in contract_status_counts.items()},
        "limitations": manifest.get("limitations", []),
    }


def _failure_modes(diagnostics: dict[str, Any]) -> str:
    screen = diagnostics["safe_haven_screen"]
    return "\n".join(
        [
            "# Failure Modes",
            "",
            f"- convexity_shortfall: max pre-event net multiple was {screen['max_pre_event_net_multiple_to_event']:.2f}x, below {screen['safe_haven_min_net_multiple']:.2f}x.",
            "- liquidity_gap: deeper OTM strikes did not always have same-day aggregate bars on the stress date.",
            "- premium_drag: daily funded sleeves reduced total return versus SOXL open-close benchmark in this sample.",
            "- evidence_limit: daily aggregates do not include NBBO, implied volatility, Greeks, or executable size at bid/ask.",
            "",
        ]
    )


def _conclusion(summary: dict[str, Any]) -> str:
    screen = summary["safe_haven_screen"]
    best = summary["analysis_summary"].get("best_daily_sleeve", {})
    return "\n".join(
        [
            "# Safe Haven Tail Hedge V1 Conclusion",
            "",
            f"- 做了什么：用 qlib SOXL 日线 open/close 和 Polygon/Massive 30DTE OTM put 日线聚合数据，测试 put_80/put_70/put_60 的 Safe Haven 式尾部对冲诊断，recorder id={summary['recorder_id']}。",
            f"- 核心指标：SOXL open-close 样本收益 {screen['benchmark_total_return_pct']:.2f}%，最大回撤 {screen['benchmark_max_drawdown_pct']:.2f}%；7/2 SOXL open-close {screen['event_soxl_open_close_return_pct']:.2f}%，put_80 扣 3% 往返成本后 {screen['event_put80_cost_return_pct']:.2f}%。",
            f"- 凸性结论：预先持有到 7/2 的最高净倍数只有 {screen['max_pre_event_net_multiple_to_event']:.2f}x，低于 {screen['safe_haven_min_net_multiple']:.2f}x 的避风港筛选线，不是 Spitznagel/Safe Haven 意义上的极端凸性。",
            f"- 回测差异：最佳日度 funded sleeve 为 {best.get('model', 'n/a')}，收益 {best.get('total_return_pct', 0.0):.2f}%，最大回撤 {best.get('max_drawdown_pct', 0.0):.2f}%，相比 SOXL 收益差 {best.get('excess_return_vs_soxl_pct', 0.0):.2f}pp。",
            "- 晋升决定：not_promoted_convexity_insufficient。该方案可以作为风险诊断，不适合作为当前 SOXL 主策略对冲模块。",
            "- 风险：日线期权聚合没有 bid/ask、IV、Greeks 和可成交深度；深 OTM 档在 7/2 的成交覆盖不足；后续必须用分钟级 NBBO/quote 或更长历史期权链验证。",
            "- 下一步：不要继续优化普通保护性 put；应寻找更便宜的远 OTM/更长周期/跨标的 crash proxy，目标是小预算在真实尾部事件中达到 10x+ 净倍数，并用 quote 级数据做成交约束。",
            "",
        ]
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
