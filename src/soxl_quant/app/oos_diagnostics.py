from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExternalReference:
    label: str
    url: str
    note: str


def _date_str(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _coerce_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "strategy_nav", "buyhold_nav", "position"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily frame missing columns: {missing}")
    out = daily[["datetime", "strategy_nav", "buyhold_nav", "position"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ("strategy_nav", "buyhold_nav", "position"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["datetime", "strategy_nav", "buyhold_nav", "position"])
    out = out.sort_values("datetime").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("daily frame needs at least two rows")
    return out


def _pct_return(values: pd.Series) -> float:
    first = float(values.iloc[0])
    last = float(values.iloc[-1])
    if first == 0.0:
        return 0.0
    return (last / first - 1.0) * 100.0


def _max_drawdown(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdown = arr / peak - 1.0
    return abs(float(np.min(drawdown)) * 100.0)


def find_poor_windows(
    daily: pd.DataFrame,
    *,
    window_size: int = 63,
    top_n: int = 12,
    min_excess_gap: float = 0.0,
) -> pd.DataFrame:
    """Find rolling windows where strategy excess is poor versus buy-and-hold.

    `window_size` is expressed in rows/trading days. The score intentionally
    combines negative excess with local drawdown so the output is useful for
    overfit triage rather than only for return ranking.
    """

    frame = _coerce_daily_frame(daily)
    window_size = max(int(window_size), 2)
    rows: list[dict[str, float | str | int]] = []
    for start_idx in range(0, max(len(frame) - window_size + 1, 0)):
        sub = frame.iloc[start_idx : start_idx + window_size]
        strategy_return = _pct_return(sub["strategy_nav"])
        buyhold_return = _pct_return(sub["buyhold_nav"])
        excess_return = strategy_return - buyhold_return
        max_dd = _max_drawdown(sub["strategy_nav"])
        avg_pos = float(sub["position"].mean())
        bad_score = max(0.0, -excess_return - float(min_excess_gap)) + max_dd
        if bad_score <= 0.0:
            continue
        rows.append(
            {
                "start": _date_str(sub["datetime"].iloc[0]),
                "end": _date_str(sub["datetime"].iloc[-1]),
                "rows": int(len(sub)),
                "strategy_return": float(strategy_return),
                "buyhold_return": float(buyhold_return),
                "excess_return": float(excess_return),
                "max_dd": float(max_dd),
                "avg_pos": avg_pos,
                "bad_score": float(bad_score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "rows",
                "strategy_return",
                "buyhold_return",
                "excess_return",
                "max_dd",
                "avg_pos",
                "bad_score",
            ]
        )
    return out.sort_values(["bad_score", "start"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def _normalise_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"])
        out = out.set_index("datetime")
    if isinstance(out.index, pd.MultiIndex):
        if "datetime" in out.index.names:
            out = out.reset_index("instrument", drop=True)
        else:
            out.index = pd.to_datetime(out.index.get_level_values(-1))
    else:
        out.index = pd.to_datetime(out.index)
    out = out.select_dtypes(include=[np.number]).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def _window_mask(index: pd.DatetimeIndex, windows: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index)
    for row in windows.itertuples(index=False):
        start = pd.Timestamp(getattr(row, "start"))
        end = pd.Timestamp(getattr(row, "end"))
        mask |= (index >= start) & (index <= end)
    return mask


def _spearman_to_excess(feature: pd.Series, daily: pd.DataFrame | None) -> float:
    if daily is None or "strategy_nav" not in daily.columns or "buyhold_nav" not in daily.columns:
        return float("nan")
    frame = _coerce_daily_frame(daily).set_index("datetime")
    strategy_return = frame["strategy_nav"].pct_change()
    buyhold_return = frame["buyhold_nav"].pct_change()
    excess = (strategy_return - buyhold_return).rename("daily_excess")
    joined = pd.concat([feature.rename("feature"), excess], axis=1).dropna()
    if len(joined) < 5 or joined["feature"].nunique() <= 1 or joined["daily_excess"].nunique() <= 1:
        return float("nan")
    return float(joined["feature"].corr(joined["daily_excess"], method="spearman"))


def compute_feature_differences(
    features: pd.DataFrame,
    poor_windows: pd.DataFrame,
    *,
    daily: pd.DataFrame | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare feature distributions in poor windows against the rest of OOS."""

    feature_frame = _normalise_feature_frame(features)
    if feature_frame.empty or poor_windows.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "bad_mean",
                "reference_mean",
                "delta",
                "delta_z",
                "bad_coverage",
                "reference_coverage",
                "spearman_to_excess",
            ]
        )
    bad_mask = _window_mask(pd.DatetimeIndex(feature_frame.index), poor_windows)
    bad = feature_frame.loc[bad_mask]
    reference = feature_frame.loc[~bad_mask]
    if bad.empty or reference.empty:
        reference = feature_frame
    rows: list[dict[str, float | str]] = []
    for col in feature_frame.columns:
        bad_series = bad[col].dropna()
        ref_series = reference[col].dropna()
        if bad_series.empty or ref_series.empty:
            continue
        ref_std = float(ref_series.std(ddof=0))
        if ref_std == 0.0 or np.isnan(ref_std):
            ref_std = float(feature_frame[col].std(ddof=0))
        if ref_std == 0.0 or np.isnan(ref_std):
            continue
        bad_mean = float(bad_series.mean())
        ref_mean = float(ref_series.mean())
        delta = bad_mean - ref_mean
        rows.append(
            {
                "feature": str(col),
                "bad_mean": bad_mean,
                "reference_mean": ref_mean,
                "delta": float(delta),
                "delta_z": float(delta / ref_std),
                "bad_coverage": float(len(bad_series) / max(len(bad), 1)),
                "reference_coverage": float(len(ref_series) / max(len(reference), 1)),
                "spearman_to_excess": _spearman_to_excess(feature_frame[col], daily),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_delta_z"] = out["delta_z"].abs()
    return out.sort_values(["abs_delta_z", "feature"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def _fmt_pct(value: float) -> str:
    return f"{float(value):.2f}%"


def _reference_rows(external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference]) -> list[str]:
    rows = []
    for ref in external_references:
        if isinstance(ref, ExternalReference):
            label, url, note = ref.label, ref.url, ref.note
        else:
            label = str(ref.get("label", "reference"))
            url = str(ref.get("url", ""))
            note = str(ref.get("note", ""))
        rows.append(f"- [{label}]({url}): {note}")
    return rows


def render_oos_diagnostic_review(
    *,
    profile: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    target_excess: float,
    target_max_dd: float,
    windows: pd.DataFrame,
    feature_differences: pd.DataFrame,
    external_references: Sequence[Mapping[str, str]] | Sequence[ExternalReference] = (),
) -> str:
    lines = [
        f"# OOS Poor-Window Diagnostic: {profile}",
        "",
        "## Scope",
        "",
        f"- train_end: `{train_end}`",
        f"- OOS window: `{oos_start}` .. `{oos_end}`",
        f"- target_excess={target_excess:.2f}%",
        f"- target_max_dd={target_max_dd:.2f}%",
        "",
    ]
    if external_references:
        lines.extend(["## External References", "", *_reference_rows(external_references), ""])

    lines.extend(
        [
            "## Poor Windows",
            "",
            "| start | end | strategy | buy-hold | excess | max_dd | avg_pos | bad_score |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if windows.empty:
        lines.append("| none | none | 0.00% | 0.00% | 0.00pp | 0.00% | 0.000 | 0.00 |")
    else:
        for row in windows.itertuples(index=False):
            lines.append(
                "| "
                f"{row.start} | {row.end} | {_fmt_pct(row.strategy_return)} | "
                f"{_fmt_pct(row.buyhold_return)} | {row.excess_return:.2f}pp | "
                f"{_fmt_pct(row.max_dd)} | {float(row.avg_pos):.3f} | {float(row.bad_score):.2f} |"
            )

    lines.extend(
        [
            "",
            "## Feature Differences",
            "",
            "| feature | bad_mean | reference_mean | delta_z | spearman_to_excess |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if feature_differences.empty:
        lines.append("| none | 0.0000 | 0.0000 | 0.000 | nan |")
    else:
        for row in feature_differences.itertuples(index=False):
            corr = getattr(row, "spearman_to_excess", float("nan"))
            corr_text = "nan" if pd.isna(corr) else f"{float(corr):.3f}"
            lines.append(
                "| "
                f"{row.feature} | {float(row.bad_mean):.4f} | {float(row.reference_mean):.4f} | "
                f"{float(row.delta_z):.3f} | {corr_text} |"
            )

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "- Treat repeated negative-excess windows as overfit candidates before retraining.",
            "- Prefer fixed-actor execution/risk counterfactuals when bad windows are driven by position mapping.",
            "- Promote new features only after the feature-difference signal survives factor IC and walk-forward checks.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_diagnostic_artifacts(
    *,
    output_dir: Path,
    profile: str,
    train_end: str,
    oos_start: str,
    oos_end: str,
    target_excess: float,
    target_max_dd: float,
    windows: pd.DataFrame,
    feature_differences: pd.DataFrame,
    external_references: Sequence[Mapping[str, str]] = (),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_path = output_dir / "poor_windows.csv"
    features_path = output_dir / "feature_differences.csv"
    review_path = output_dir / "oos_diagnostic_review.md"
    windows.to_csv(windows_path, index=False)
    feature_differences.to_csv(features_path, index=False)
    review_path.write_text(
        render_oos_diagnostic_review(
            profile=profile,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            target_excess=target_excess,
            target_max_dd=target_max_dd,
            windows=windows,
            feature_differences=feature_differences,
            external_references=external_references,
        ),
        encoding="utf-8",
    )
    return {"windows": windows_path, "features": features_path, "review": review_path}


def load_qlib_feature_frame(start: str, end: str) -> pd.DataFrame:
    """Load the repo qlib feature frame for a date range."""

    from src.soxl_quant.data import build_dataset, init_qlib
    from src.soxl_quant.settings import QLIB_DATA_PATH

    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset()
    frame = dataset.handler.feature_frame.droplevel("instrument").sort_index()
    return frame.loc[start:end]


def run_oos_diagnostic(
    *,
    daily_csv: Path,
    profile: str,
    output_dir: Path,
    oos_start: str = "2020-01-02",
    oos_end: str = "",
    window_size: int = 63,
    top_windows: int = 12,
    top_features: int = 30,
    target_excess: float = 30.0,
    target_max_dd: float = 15.0,
    train_end: str | None = None,
    external_references: Sequence[Mapping[str, str]] = (),
) -> dict[str, object]:
    """Run the full poor-window and feature-difference diagnostic."""

    from src.soxl_quant.settings import TRAIN_SEG

    daily = pd.read_csv(daily_csv)
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    resolved_oos_end = oos_end or str(daily["datetime"].max().date())
    daily = daily[(daily["datetime"] >= oos_start) & (daily["datetime"] <= resolved_oos_end)].copy()
    if len(daily) < window_size:
        raise ValueError(f"not enough rows for window_size={window_size}: {len(daily)}")

    windows = find_poor_windows(daily, window_size=window_size, top_n=top_windows)
    features = load_qlib_feature_frame(oos_start, resolved_oos_end)
    feature_diffs = compute_feature_differences(features, windows, daily=daily, top_n=top_features)
    artifacts = write_diagnostic_artifacts(
        output_dir=output_dir,
        profile=profile,
        train_end=train_end or TRAIN_SEG[1],
        oos_start=oos_start,
        oos_end=resolved_oos_end,
        target_excess=target_excess,
        target_max_dd=target_max_dd,
        windows=windows,
        feature_differences=feature_diffs,
        external_references=external_references,
    )
    manifest: dict[str, object] = {
        "profile": profile,
        "daily_csv": str(daily_csv),
        "oos_start": oos_start,
        "oos_end": resolved_oos_end,
        "window_size": window_size,
        "target_excess": target_excess,
        "target_max_dd": target_max_dd,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "external_references": list(external_references),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    manifest["windows"] = int(len(windows))
    manifest["features"] = int(len(feature_diffs))
    manifest["manifest"] = str(manifest_path)
    return manifest


__all__ = [
    "ExternalReference",
    "compute_feature_differences",
    "find_poor_windows",
    "load_qlib_feature_frame",
    "render_oos_diagnostic_review",
    "run_oos_diagnostic",
    "write_diagnostic_artifacts",
]
