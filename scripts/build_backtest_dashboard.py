#!/usr/bin/env python3
"""Build a standalone interactive HTML dashboard with Plotly for backtest inspection."""
from __future__ import annotations

import csv
from datetime import datetime
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.paper_review import public_paper_review_from_summary

GENERALIZATION_DIR = ROOT / "runtime" / "execution" / "generalization"
GENERALIZATION_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = GENERALIZATION_DIR
DETAIL_DIR = GENERALIZATION_DIR / "detailed"
FEATURES_PATH = ROOT / "data" / "features_v2.pkl"
MODELS_DIR = ROOT / "runtime" / "model" / "card"
MODEL_SELECTION_DIR = ROOT / "runtime" / "model_selection" / "card"
OUTPUT_PATH = ROOT / "dashboard.html"
DASHBOARD_DATA_DIR = ROOT / "dashboard_data"
MAX_RESEARCH_BACKTESTS = 30

C_STRATEGY = "#2563eb"
C_BUYHOLD = "#0f766e"
C_BUY = "#0f766e"
C_SELL = "#b91c1c"
C_INK = "#111827"
C_MUTED = "#607080"
C_LINE = "#d7dee6"
C_PANEL = "#ffffff"
C_GOOD = "#0f766e"
C_BAD = "#b91c1c"

FEATURE_COLS = [
    "vix", "rsi", "volatility", "spread", "vix_zscore",
    "spread_trend", "ma5_ma20", "vol_strength",
    "mom_20", "intraday_range", "vix_mom", "spread_vix",
]


def load_features() -> dict[str, dict[str, float]] | None:
    if not FEATURES_PATH.exists():
        return None
    with open(FEATURES_PATH, "rb") as f:
        df = pickle.load(f)
    df.index = pd.to_datetime(df.index)
    records: dict[str, dict[str, float]] = {}
    available = [c for c in FEATURE_COLS if c in df.columns]
    for dt, row in df.iterrows():
        key = dt.strftime("%Y-%m-%d")
        records[key] = {c: (float(row[c]) if pd.notna(row[c]) else None) for c in available}
    return records


def load_ohlc_and_indicators(etf: str) -> dict[str, dict]:
    """Load OHLC from data/processed/panels/ (preferred) or data/raw/symbols/ and compute soxl_backtest indicators."""
    # Prefer processed panel (kept up-to-date by update_market_data.py)
    processed_path = ROOT / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv"
    if etf.upper() == "SOXL" and processed_path.exists():
        df = pd.read_csv(processed_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.rename(columns={
            "soxl_open": "open",
            "soxl_high": "high",
            "soxl_low": "low",
            "soxl_close": "close",
            "soxl_volume": "volume",
        })
    else:
        csv_path = ROOT / "data" / "raw" / "symbols" / f"{etf.upper()}.csv"
        if not csv_path.exists():
            return {}
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # RSI6
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=6, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=6, adjust=False).mean()
    df["rsi6"] = 100 - (100 / (1 + gain / (loss + 1e-5)))

    # MA180 (soxl_backtest calls it MA20 but uses window=180)
    df["ma180"] = df["close"].rolling(window=180).mean()

    # HV_20 (annualized)
    returns = np.log(df["close"] / df["close"].shift(1))
    df["hv_20"] = returns.rolling(window=20).std() * np.sqrt(252)

    # VOL_COEF
    hv_median = df["hv_20"].median()
    if hv_median and hv_median > 0:
        df["vol_coef"] = (df["hv_20"] / hv_median).clip(0.5, 2.0)
    else:
        df["vol_coef"] = 1.0

    # VIX: prefer processed panel, then features_v2.pkl, then fallback 20
    vix_series = pd.Series(20.0, index=df.index)
    if etf.upper() == "SOXL" and processed_path.exists() and "vix" in df.columns:
        vix_series = df["vix"].fillna(20.0)
    elif FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, "rb") as f:
                feat = pickle.load(f)
            if "vix" in feat.columns:
                feat_idx = pd.to_datetime(feat.index)
                for i, row in df.iterrows():
                    dt = row["date"]
                    mask = feat_idx == pd.Timestamp(dt)
                    if mask.any():
                        vix_val = feat.loc[mask, "vix"].values
                        if len(vix_val) > 0 and pd.notna(vix_val[0]):
                            vix_series.iloc[i] = float(vix_val[0])
        except Exception:
            pass
    df["vix"] = vix_series

    # Panic Score (Sigmoid on VIX + RSI6 adjustment)
    vix_low, vix_high, fear_min, fear_max, steepness = 10, 30, -100, 100, 0.5
    vix_clipped = df["vix"].clip(vix_low, vix_high)
    vix_mid = (vix_high + vix_low) / 2
    sigmoid = 1 / (1 + np.exp(-(vix_mid - vix_clipped) * steepness))
    df["base_panic"] = sigmoid * (fear_max - fear_min) + fear_min

    def _adjust_panic(row):
        rsi6 = row["rsi6"]
        base = row["base_panic"]
        if pd.isna(rsi6):
            return base
        if rsi6 <= 30:
            return max(-150, base - (20 + (30 - rsi6) * 2.5))
        elif rsi6 >= 70:
            return min(150, base + (20 + (rsi6 - 70)))
        return base

    df["panic_score"] = df.apply(_adjust_panic, axis=1)

    # Market state: overvalued (price > MA180)
    df["market_state"] = df["close"] > df["ma180"]

    result = {}
    for _, row in df.iterrows():
        key = row["date"].strftime("%Y-%m-%d")
        result[key] = {
            "open": float(row["open"]) if pd.notna(row["open"]) else None,
            "high": float(row["high"]) if pd.notna(row["high"]) else None,
            "low": float(row["low"]) if pd.notna(row["low"]) else None,
            "close": float(row["close"]) if pd.notna(row["close"]) else None,
            "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            "rsi6": float(row["rsi6"]) if pd.notna(row["rsi6"]) else None,
            "ma180": float(row["ma180"]) if pd.notna(row["ma180"]) else None,
            "hv_20": float(row["hv_20"]) if pd.notna(row["hv_20"]) else None,
            "vol_coef": float(row["vol_coef"]) if pd.notna(row["vol_coef"]) else None,
            "panic_score": float(row["panic_score"]) if pd.notna(row["panic_score"]) else None,
            "market_state": bool(row["market_state"]) if pd.notna(row["market_state"]) else None,
        }
    return result


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_or(path: Path, default):
    return load_json(path) if path.exists() else default


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key == "datetime" or key == "action":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def compact_payload_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep only the trading-series fields needed to rehydrate historical views."""
    compact: list[dict[str, object]] = []
    optional_fields = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi6",
        "ma180",
        "hv_20",
        "vol_coef",
        "panic_score",
        "market_state",
        "profile",
        "execution_status",
        "dry_run",
        "target_position",
        "entry_status",
        "entry_side",
        "entry_target_position",
        "summary_path",
    )
    for row in rows:
        compact_row = {
            "datetime": row["datetime"],
            "strategy_nav": row["strategy_nav"],
            "price": row.get("price"),
            "buyhold_nav": row.get("buyhold_nav"),
            "position": row.get("position"),
            "excess": row.get("excess"),
            "excess_pct": row.get("excess_pct"),
            "action": row.get("action", "HOLD"),
        }
        for field in optional_fields:
            if field in row and row[field] is not None:
                compact_row[field] = row[field]
        compact.append(compact_row)
    return compact


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _summary_execution_status(payload: dict[str, object]) -> str:
    execution_result = payload.get("execution_result")
    if isinstance(execution_result, dict) and execution_result.get("status"):
        return str(execution_result["status"])
    if payload.get("dry_run") is True:
        return "dry_run"
    if payload.get("executed") is True:
        return "executed"
    return "unknown"


def _summary_target_position(payload: dict[str, object]) -> float | None:
    signal = payload.get("signal")
    if isinstance(signal, dict) and signal.get("target_position") is not None:
        return float(signal["target_position"])
    execution_result = payload.get("execution_result")
    if isinstance(execution_result, dict) and execution_result.get("target_position") is not None:
        return float(execution_result["target_position"])
    return None


def _summary_capped_target_position(payload: dict[str, object]) -> float | None:
    risk_gate = payload.get("risk_gate")
    if isinstance(risk_gate, dict) and risk_gate.get("capped_target_position") is not None:
        return float(risk_gate["capped_target_position"])
    return None


def _summary_priority(path: Path, payload: dict[str, object]) -> int:
    if payload.get("dry_run") is True:
        return 10
    if "duplicate_skip" in path.parts:
        return 40
    status = _summary_execution_status(payload)
    if status in {"skipped", "executed"}:
        return 35
    if payload.get("executed") is True:
        return 30
    return 20


def _public_review_for_summary(path: Path, payload: dict[str, object]) -> dict[str, object]:
    return public_paper_review_from_summary(payload, summary_path=_relative_to_root(path))


def _latest_paper_signal_metadata(paper_dir: Path) -> dict[str, object]:
    candidates: list[tuple[tuple[str, int, float], dict[str, object]]] = []
    for summary_path in sorted(paper_dir.glob("20??-??-??/**/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        date = payload.get("date")
        if not date:
            continue
        risk_gate = payload.get("risk_gate")
        composition = payload.get("composition")
        public_review = _public_review_for_summary(summary_path, payload)
        metadata: dict[str, object] = {
            "latest_signal_date": str(date),
            "latest_signal_profile": str(payload.get("profile") or ""),
            "latest_signal_execution_status": _summary_execution_status(payload),
            "latest_signal_target_position": _summary_target_position(payload),
            "latest_signal_capped_target_position": _summary_capped_target_position(payload),
            "latest_signal_summary_path": _relative_to_root(summary_path),
            "latest_signal_broker": str(payload.get("broker") or ""),
            "latest_signal_dry_run": payload.get("dry_run") is True,
            "latest_public_review": public_review,
            "latest_entry_status": public_review.get("entry_status"),
            "latest_entry_side": public_review.get("entry_side"),
            "latest_entry_target_position": public_review.get("entry_target_position"),
        }
        if isinstance(risk_gate, dict) and risk_gate.get("approved") is not None:
            metadata["latest_signal_risk_gate_approved"] = risk_gate.get("approved") is True
        if isinstance(composition, dict):
            for source, target in (
                ("slow_profile", "latest_signal_slow_profile"),
                ("fast_profile", "latest_signal_fast_profile"),
                ("mode", "latest_signal_mode"),
            ):
                if composition.get(source) is not None:
                    metadata[target] = str(composition[source])
        key = (str(date), _summary_priority(summary_path, payload), float(summary_path.stat().st_mtime))
        candidates.append((key, {key: value for key, value in metadata.items() if value is not None}))
    if not candidates:
        return {}
    return max(candidates, key=lambda item: item[0])[1]


def _paper_summary_metadata(path: Path, payload: dict[str, object]) -> dict[str, object] | None:
    date = payload.get("date")
    if not date or payload.get("dry_run") is True:
        return None
    public_review = _public_review_for_summary(path, payload)
    entry_target = public_review.get("entry_target_position")
    return {
        "date": str(date),
        "profile": str(payload.get("profile") or ""),
        "execution_status": _summary_execution_status(payload),
        "dry_run": payload.get("dry_run") is True,
        "target_position": entry_target,
        "entry_status": public_review.get("entry_status"),
        "entry_side": public_review.get("entry_side"),
        "entry_target_position": entry_target,
        "summary_path": _relative_to_root(path),
        "_priority": _summary_priority(path, payload),
        "_mtime": path.stat().st_mtime,
    }


def _load_paper_trading_summary_metadata(paper_dir: Path) -> dict[str, dict[str, object]]:
    metadata_by_date: dict[str, dict[str, object]] = {}
    for summary_path in sorted(paper_dir.glob("20??-??-??/**/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = _paper_summary_metadata(summary_path, payload)
        if not metadata:
            continue
        date = str(metadata["date"])
        current = metadata_by_date.get(date)
        candidate_key = (int(metadata["_priority"]), float(metadata["_mtime"]))
        current_key = (
            int(current.get("_priority", -1)),
            float(current.get("_mtime", -1.0)),
        ) if current else (-1, -1.0)
        if candidate_key >= current_key:
            metadata_by_date[date] = metadata

    for metadata in metadata_by_date.values():
        metadata.pop("_priority", None)
        metadata.pop("_mtime", None)
    return metadata_by_date


def _attach_paper_trading_summary_metadata(rows: list[dict[str, object]], paper_dir: Path) -> None:
    metadata_by_date = _load_paper_trading_summary_metadata(paper_dir)
    for row in rows:
        metadata = metadata_by_date.get(str(row.get("datetime")))
        if not metadata:
            continue
        for key, value in metadata.items():
            if key != "date":
                row[key] = value


def _paper_row_has_summary_metadata(row: dict[str, object]) -> bool:
    return any(row.get(key) is not None for key in ("summary_path", "profile", "execution_status"))


def _prepare_paper_trading_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    annotated_rows = [row for row in rows if _paper_row_has_summary_metadata(row)]
    prepared = annotated_rows if annotated_rows else rows
    if not prepared:
        return prepared

    for row in prepared:
        entry_target = row.get("entry_target_position")
        if entry_target is not None:
            row["position"] = entry_target
            row["target_position"] = entry_target
        entry_side = row.get("entry_side")
        if isinstance(entry_side, str) and entry_side in {"BUY", "SELL", "HOLD"}:
            row["action"] = entry_side

    first_strategy = float(prepared[0].get("strategy_nav") or 0.0)
    first_buyhold = float(prepared[0].get("buyhold_nav") or 0.0)
    if first_strategy <= 0 or first_buyhold <= 0:
        return prepared

    scale = first_strategy / first_buyhold
    for row in prepared:
        strategy_nav = float(row.get("strategy_nav") or 0.0)
        buyhold_nav = float(row.get("buyhold_nav") or 0.0) * scale
        strategy_index = strategy_nav / first_strategy if first_strategy else 0.0
        buyhold_index = buyhold_nav / first_strategy if first_strategy else 0.0
        row["strategy_nav"] = strategy_index
        row["buyhold_nav"] = buyhold_index
        row["excess"] = strategy_index - buyhold_index
        row["excess_pct"] = (strategy_index / buyhold_index - 1) * 100 if buyhold_index else 0.0
    return prepared


def _paper_trading_detail_meta(
    rows: list[dict[str, object]],
    *,
    csv_path: Path,
    source_kind: str,
) -> dict[str, object]:
    meta: dict[str, object] = {
        "source_kind": source_kind,
        "source": str(csv_path.relative_to(ROOT)),
    }
    if source_kind != "paper_trading" or not rows:
        return meta
    paper_dir = csv_path.parent

    latest = rows[-1]
    meta.update(
        {
            "latest_data_date": latest.get("datetime"),
            "latest_profile": latest.get("profile"),
            "latest_execution_status": latest.get("execution_status"),
            "latest_target_position": latest.get("target_position"),
            "latest_entry_status": latest.get("entry_status"),
            "latest_entry_side": latest.get("entry_side"),
            "latest_entry_target_position": latest.get("entry_target_position"),
            "latest_summary_path": latest.get("summary_path"),
        }
    )
    signal_meta = _latest_paper_signal_metadata(paper_dir)
    meta.update(signal_meta)
    signal_date = str(signal_meta.get("latest_signal_date") or "")
    row_date = str(meta.get("latest_data_date") or "")
    if signal_date and signal_date >= row_date:
        meta.update(
            {
                "latest_data_date": signal_meta.get("latest_signal_date"),
                "latest_profile": signal_meta.get("latest_signal_profile"),
                "latest_execution_status": signal_meta.get("latest_signal_execution_status"),
                "latest_target_position": (
                    signal_meta.get("latest_signal_capped_target_position")
                    if signal_meta.get("latest_signal_capped_target_position") is not None
                    else signal_meta.get("latest_signal_target_position")
                ),
                "latest_reconcile_action": signal_meta.get("latest_signal_reconcile_action"),
                "latest_summary_path": signal_meta.get("latest_signal_summary_path"),
            }
        )
    return {key: value for key, value in meta.items() if value is not None}


def load_model_versions() -> list[dict]:
    """Load all historical model versions from runtime/model/card/*/*_summary.json."""
    versions = []
    if not MODELS_DIR.exists():
        return versions
    for f in sorted(MODELS_DIR.rglob("*_summary.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            raw_name = f.stem.replace("_summary", "")
            # Files are named soxl_<name>_summary.json; strip soxl_ prefix if present
            name = raw_name[5:] if raw_name.startswith("soxl_") else raw_name
            tp = data.get("test_proxy", {})
            valid = data.get("valid", {})
            versions.append(
                {
                    "name": name,
                    "test_return": tp.get("return"),
                    "test_sharpe": tp.get("sharpe"),
                    "test_max_dd": tp.get("max_dd"),
                    "test_avg_pos": tp.get("avg_pos"),
                    "valid_return": valid.get("return"),
                    "valid_sharpe": valid.get("sharpe"),
                    "valid_max_dd": valid.get("max_dd"),
                    "promoted": data.get("promotion", False),
                    "selection_history": data.get("selection_history", []),
                }
            )
        except Exception:
            continue
    return versions


def _load_dashboard_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _first_number(*values: object) -> float | int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            return value
    return None


def _first_text(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
    return None


def _summary_conclusion(summary: dict[str, object], selection: dict[str, object]) -> str:
    raw = summary.get("conclusion") or selection.get("conclusion")
    if isinstance(raw, list):
        parts = [str(item) for item in raw if item is not None]
        return " ".join(parts[:2])
    if isinstance(raw, str):
        return raw
    return ""


def _scenario_base(summary: dict[str, object]) -> dict[str, object]:
    scenarios = summary.get("scenario_summary")
    if not isinstance(scenarios, list):
        return {}
    for item in scenarios:
        if isinstance(item, dict) and item.get("scenario") == "base_replay":
            return item
    for item in scenarios:
        if isinstance(item, dict):
            return item
    return {}


def _research_backtest_record(card_dir: Path) -> dict[str, object] | None:
    selection_path = card_dir / "selection.json"
    summary_path = card_dir / "summary.json"
    if not selection_path.exists() and not summary_path.exists():
        return None

    selection = _load_dashboard_json(selection_path) if selection_path.exists() else {}
    summary = _load_dashboard_json(summary_path) if summary_path.exists() else {}
    selected_score = selection.get("selected_base_score")
    if not isinstance(selected_score, dict):
        selected_score = {}
    scenario_base = _scenario_base(summary)

    file_mtimes = [
        path.stat().st_mtime
        for path in (selection_path, summary_path, *card_dir.glob("*.ipynb"))
        if path.exists()
    ]
    updated_at = max(file_mtimes) if file_mtimes else card_dir.stat().st_mtime
    notebook_path = next(iter(sorted(card_dir.glob("*.ipynb"))), None)
    created_at = _first_text(selection.get("created_at"), summary.get("created_at"))

    stress_pass = _first_number(selection.get("stress_pass_count"), summary.get("stress_pass_count"))
    stress_total = _first_number(selection.get("stress_scenario_count"), summary.get("stress_scenario_count"))
    selected_stress_profile = summary.get("selected_stress_profile")
    if stress_total is None and isinstance(selected_stress_profile, list):
        stress_total = len(selected_stress_profile)

    record = {
        "experiment": _first_text(selection.get("experiment"), summary.get("experiment"), card_dir.name),
        "decision": _first_text(selection.get("decision"), summary.get("decision"), "unknown"),
        "promotion_allowed": bool(selection.get("promotion_allowed", summary.get("promotion_allowed", False))),
        "role": _first_text(selection.get("role")),
        "changed_layer": selection.get("changed_layer") if isinstance(selection.get("changed_layer"), list) else [],
        "candidate": _first_text(
            selection.get("selected_candidate"),
            selected_score.get("candidate_id"),
            scenario_base.get("best_candidate_id"),
        ),
        "min_relative_return": _first_number(
            selected_score.get("min_relative_return"),
            scenario_base.get("best_min_relative_return"),
            summary.get("best_min_relative_return"),
        ),
        "max_drawdown": _first_number(
            selected_score.get("max_max_dd"),
            scenario_base.get("best_max_dd"),
            summary.get("best_max_dd"),
        ),
        "full_gate": selected_score.get("full_gate"),
        "near_miss": selected_score.get("near_miss"),
        "stress_pass_count": stress_pass,
        "stress_scenario_count": stress_total,
        "next_mechanism": _first_text(selection.get("next_mechanism"), summary.get("next_mechanism")),
        "conclusion": _summary_conclusion(summary, selection),
        "created_at": created_at,
        "updated_at": datetime.fromtimestamp(updated_at).strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at_ts": updated_at,
        "notebook_path": _relpath(notebook_path) if notebook_path else None,
        "selection_path": _relpath(selection_path) if selection_path.exists() else None,
        "summary_path": (
            _relpath(summary_path)
            if summary_path.exists()
            else (_relpath(selection_path) if selection_path.exists() else None)
        ),
        "external_reference": selection.get("external_reference") or summary.get("external_sources") or [],
    }
    return {key: value for key, value in record.items() if value not in (None, "")}


def load_research_backtests(limit: int = MAX_RESEARCH_BACKTESTS) -> list[dict[str, object]]:
    """Load recent runtime/model_selection notebook backtest diagnostics for the static dashboard."""
    if not MODEL_SELECTION_DIR.exists():
        return []
    records = []
    for card_dir in sorted(MODEL_SELECTION_DIR.iterdir()):
        if not card_dir.is_dir():
            continue
        record = _research_backtest_record(card_dir)
        if record:
            records.append(record)
    records.sort(key=lambda item: (float(item.get("updated_at_ts") or 0.0), str(item.get("experiment") or "")), reverse=True)
    return records[:limit]


def select_best_version(model_versions: list[dict], versioned: dict[str, dict]) -> str:
    best_version = ""
    best_sharpe = -float("inf")
    for mv in model_versions:
        name = mv.get("name", "")
        sharpe = mv.get("test_sharpe")
        if sharpe is None:
            continue
        tag_candidates = [name]
        if name.startswith("soxl_"):
            tag_candidates.append(name[5:])
        for tag in tag_candidates:
            if tag in versioned and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_version = tag
    return best_version


def _latest_source_priority(source_kind: str) -> int:
    priorities = {
        "paper_trading": 2,
        "post_train_review": 1,
    }
    return priorities.get(source_kind, 0)


def select_latest_version(versioned: dict[str, dict]) -> str:
    latest_version = ""
    latest_priority = -1
    latest_updated_at = -float("inf")
    for tag, payload in versioned.items():
        meta = payload.get("_meta", {})
        updated_at = meta.get("updated_at", float("-inf"))
        source_kind = meta.get("source_kind", "generalization_version")
        priority = _latest_source_priority(source_kind)
        if priority > latest_priority or (priority == latest_priority and updated_at > latest_updated_at):
            latest_priority = priority
            latest_updated_at = updated_at
            latest_version = tag
    return latest_version


def compute_regime_shapes(rows: list[dict]) -> list[dict]:
    shapes = []
    if not rows:
        return shapes
    current_regime = None
    current_start = None
    prev_dt = None
    for row in rows:
        dt = str(row["datetime"])
        f = row.get("_features", {})
        vixz = f.get("vix_zscore")
        regime = None
        if vixz is not None:
            if vixz > 2:
                regime = "stress"
            elif vixz < -1:
                regime = "calm"
        if regime != current_regime:
            if current_regime and current_start and prev_dt:
                fillcolor = "rgba(185,28,28,0.07)" if current_regime == "stress" else "rgba(15,118,110,0.07)"
                shapes.append(
                    {
                        "type": "rect",
                        "xref": "x",
                        "yref": "paper",
                        "x0": current_start,
                        "x1": prev_dt,
                        "y0": 0,
                        "y1": 1,
                        "fillcolor": fillcolor,
                        "line": {"width": 0},
                        "layer": "below",
                    }
                )
            current_regime = regime
            current_start = dt if regime else None
        prev_dt = dt
    if current_regime and current_start and prev_dt:
        fillcolor = "rgba(185,28,28,0.07)" if current_regime == "stress" else "rgba(15,118,110,0.07)"
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": current_start,
                "x1": prev_dt,
                "y0": 0,
                "y1": 1,
                "fillcolor": fillcolor,
                "line": {"width": 0},
                "layer": "below",
            }
        )
    return shapes


def compute_trades(rows: list[dict]) -> list[dict]:
    trades = []
    entry_idx = -1
    cumulative = 0.0
    for i, row in enumerate(rows):
        if row["action"] == "BUY" and entry_idx < 0:
            entry_idx = i
        elif row["action"] == "SELL" and entry_idx >= 0:
            entry_row = rows[entry_idx]
            entry_price = entry_row["price"] or entry_row["strategy_nav"]
            exit_price = row["price"] or row["strategy_nav"]
            entry_nav = entry_row["strategy_nav"]
            exit_nav = row["strategy_nav"]
            days = i - entry_idx
            ret_pct = entry_price > 0 and ((exit_price / entry_price - 1) * 100) or 0
            pnl = exit_nav - entry_nav
            cumulative += pnl

            vix_vals = []
            rsi_vals = []
            for j in range(entry_idx, i + 1):
                f = rows[j].get("_features", {})
                if f.get("vix") is not None:
                    vix_vals.append(f["vix"])
                if f.get("rsi") is not None:
                    rsi_vals.append(f["rsi"])

            trades.append(
                {
                    "entry_date": entry_row["datetime"],
                    "exit_date": row["datetime"],
                    "days": days,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "pnl": pnl,
                    "cumulative": cumulative,
                    "entry_nav": entry_nav,
                    "exit_nav": exit_nav,
                    "avg_vix": sum(vix_vals) / len(vix_vals) if vix_vals else None,
                    "avg_rsi": sum(rsi_vals) / len(rsi_vals) if rsi_vals else None,
                }
            )
            entry_idx = -1
    return trades


def compute_kelly_stats(trades: list[dict]) -> dict:
    if not trades:
        return {}
    wins = [t["return_pct"] for t in trades if t["return_pct"] > 0]
    losses = [t["return_pct"] for t in trades if t["return_pct"] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    r = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    kelly = win_rate - (1 - win_rate) / r if r != float("inf") and r > 0 else win_rate
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for t in trades:
        if t["return_pct"] > 0:
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)
    return {
        "total": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "kelly": kelly,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def infer_action_from_position(prev_pos: float | None, pos: float, index: int) -> str:
    if index == 0:
        return "BUY" if pos > 0 else "HOLD"
    if prev_pos is None:
        return "HOLD"
    eps = 1e-9
    if pos > prev_pos + eps:
        return "BUY"
    if pos < prev_pos - eps:
        return "SELL"
    return "HOLD"


def sanitize_explicit_action(action: str, prev_pos: float | None, pos: float) -> str:
    eps = 1e-6
    if action == "BUY":
        if pos <= eps:
            return "HOLD"
        if prev_pos is not None and pos <= prev_pos + eps:
            return "HOLD"
    elif action == "SELL":
        if prev_pos is None:
            return "HOLD"
        if prev_pos <= eps and pos <= eps:
            return "HOLD"
        if pos >= prev_pos - eps and pos > eps:
            return "HOLD"
    return action


def normalize_post_train_review_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prev_pos: float | None = None
    for i, r in enumerate(raw_rows):
        pos = float(r.get("position", 0.0))
        strat_nav = float(r["strategy_nav"])
        bh_nav = float(r["buyhold_nav"])
        raw_action = str(r.get("action", "") or "").upper()
        action = raw_action if raw_action in {"BUY", "SELL", "HOLD"} else infer_action_from_position(prev_pos, pos, i)
        if raw_action in {"BUY", "SELL"}:
            action = sanitize_explicit_action(action, prev_pos, pos)

        row: dict[str, object] = {
            "datetime": r["datetime"],
            "strategy_nav": strat_nav,
            "price": float(r["price"]),
            "buyhold_nav": bh_nav,
            "position": pos,
            "excess": strat_nav - bh_nav,
            "excess_pct": (strat_nav / bh_nav - 1) * 100 if bh_nav != 0 else 0.0,
            "action": action,
        }
        rows.append(row)
        prev_pos = pos

    return rows


def _load_post_train_review_versions(features: dict[str, dict[str, float]] | None) -> dict[str, dict]:
    """Load SOXL daily CSVs from runtime/model/card/ and runtime/soxl/paper_trading/ as versioned data."""
    versioned: dict[str, dict] = {}

    def version_source_score(tag: str, source_path: Path, end_date: str, rows_count: int, updated_at: float) -> tuple:
        # If two files produce the same dashboard tag, prefer the canonical
        # profile directory (`runtime/model/card/<tag>/<tag>_soxl_daily.csv`)
        # over alias/helper directories that happen to use the same stem.
        return (
            1 if source_path.parent.name == tag else 0,
            end_date,
            rows_count,
            updated_at,
        )

    dirs_to_scan = [
        (ROOT / "runtime" / "model" / "card", True),   # (path, recursive)
        (ROOT / "runtime" / "soxl" / "paper_trading", False),
    ]

    for ptr_dir, recursive in dirs_to_scan:
        if not ptr_dir.exists():
            continue
        globber = ptr_dir.rglob if recursive else ptr_dir.glob
        for csv_path in sorted(globber("*_soxl_daily.csv")):
            tag = csv_path.stem.replace("_soxl_daily", "")
            source_kind = "paper_trading" if "paper_trading" in str(csv_path) else "post_train_review"
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    raw_rows = list(reader)
            except Exception:
                continue
            if not raw_rows:
                continue
            if len(raw_rows) < 10 and source_kind != "paper_trading":
                continue

            rows = normalize_post_train_review_rows(raw_rows)
            if source_kind == "paper_trading":
                _attach_paper_trading_summary_metadata(rows, ptr_dir)
                rows = _prepare_paper_trading_rows(rows)
            ohlc = load_ohlc_and_indicators("SOXL")

            # Merge features
            if features:
                for r in rows:
                    dt = str(r["datetime"])
                    r["_features"] = features.get(dt, {})
            else:
                for r in rows:
                    r["_features"] = {}
            for r in rows:
                dt = str(r["datetime"])
                if dt in ohlc:
                    r.update(ohlc[dt])

            # Compute summary from rows
            strategy_return = rows[-1]["strategy_nav"] / rows[0]["strategy_nav"] - 1
            buyhold_return = rows[-1]["buyhold_nav"] / rows[0]["buyhold_nav"] - 1
            excess_return = strategy_return - buyhold_return
            peak_s = rows[0]["strategy_nav"]
            max_dd_strategy = 0.0
            buy_count = sell_count = 0
            max_pos = min_pos = rows[0]["position"]
            for r in rows:
                peak_s = max(peak_s, r["strategy_nav"])
                max_dd_strategy = max(max_dd_strategy, (peak_s - r["strategy_nav"]) / peak_s)
                if r["action"] == "BUY":
                    buy_count += 1
                elif r["action"] == "SELL":
                    sell_count += 1
                max_pos = max(max_pos, r["position"])
                min_pos = min(min_pos, r["position"])

            summary = {
                "etf": "SOXL",
                "strategy_return": strategy_return,
                "buyhold_return": buyhold_return,
                "excess_return": excess_return,
                "max_dd_strategy": max_dd_strategy,
                "max_dd_buyhold": 0.0,  # not computed here
                "trades": buy_count + sell_count,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "max_position": max_pos,
                "min_position": min_pos,
            }

            regime_shapes = compute_regime_shapes(rows)
            trades = compute_trades(rows)
            kelly_stats = compute_kelly_stats(trades)

            updated_at = csv_path.stat().st_mtime
            payload = {
                "_meta": {
                    "updated_at": updated_at,
                    "source": str(csv_path.relative_to(ROOT)),
                    "source_kind": source_kind,
                    "shared_market_data": True,
                },
                "detailed": {
                    "SOXL": {
                        "summary": summary,
                        "csv_path": str(csv_path.relative_to(ROOT)),
                        "meta": _paper_trading_detail_meta(
                            rows,
                            csv_path=csv_path,
                            source_kind=source_kind,
                        ),
                        "rows": compact_payload_rows(rows),
                        "regime_shapes": regime_shapes,
                        "trades": trades,
                        "kelly": kelly_stats,
                    }
                },
                "windows": {
                    "SOXL": {
                        "start": rows[0]["datetime"] if rows else "",
                        "end": rows[-1]["datetime"] if rows else "",
                        "rows": len(rows),
                    }
                },
            }
            existing = versioned.get(tag)
            if existing is not None:
                existing_meta = existing.get("_meta", {})
                existing_source = ROOT / str(existing_meta.get("source") or "")
                existing_window = existing.get("windows", {}).get("SOXL", {})
                existing_score = version_source_score(
                    tag,
                    existing_source,
                    str(existing_window.get("end") or ""),
                    int(existing_window.get("rows") or 0),
                    float(existing_meta.get("updated_at") or 0.0),
                )
                new_score = version_source_score(tag, csv_path, str(rows[-1]["datetime"]), len(rows), updated_at)
                if new_score < existing_score:
                    continue
            versioned[tag] = payload
    return versioned


def _load_retained_dashboard_versions() -> dict[str, dict]:
    """Load existing static dashboard payloads when source CSVs were intentionally retired."""
    versioned: dict[str, dict] = {}
    versioned_dir = DASHBOARD_DATA_DIR / "versioned"
    if not versioned_dir.exists():
        return versioned

    for tag_dir in sorted(versioned_dir.iterdir()):
        if not tag_dir.is_dir():
            continue
        details: dict[str, dict] = {}
        windows: dict[str, dict] = {}
        updated_at = 0.0
        for json_path in sorted(tag_dir.glob("*.json")):
            try:
                detail = load_json(json_path)
            except Exception:
                continue
            rows = detail.get("rows") or []
            if not isinstance(rows, list) or not rows:
                continue
            etf = json_path.stem.upper()
            if tag_dir.name == "paper_trading":
                detail = _sanitize_retained_paper_detail(detail)
                rows = detail.get("rows") or []
            details[etf] = detail
            windows[etf] = {
                "start": rows[0].get("datetime", ""),
                "end": rows[-1].get("datetime", ""),
                "rows": len(rows),
            }
            updated_at = max(updated_at, json_path.stat().st_mtime)
        if details:
            source_kind = "paper_trading" if tag_dir.name == "paper_trading" else "retained_dashboard_export"
            versioned[tag_dir.name] = {
                "_meta": {
                    "updated_at": updated_at,
                    "source": str(tag_dir.relative_to(ROOT)),
                    "source_kind": source_kind,
                    "shared_market_data": True,
                },
                "detailed": details,
                "windows": windows,
            }
    return versioned


def _retained_paper_public_review(detail: dict[str, object]) -> dict[str, object]:
    rows = detail.get("rows")
    latest = rows[-1] if isinstance(rows, list) and rows else {}
    if not isinstance(latest, dict):
        latest = {}
    meta = detail.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    target = meta.get("latest_signal_capped_target_position")
    if target is None:
        target = meta.get("latest_signal_target_position")
    pseudo_summary = {
        "date": meta.get("latest_signal_date") or latest.get("datetime"),
        "profile": meta.get("latest_signal_profile") or meta.get("latest_profile") or latest.get("profile"),
        "broker": meta.get("latest_signal_broker") or "",
        "dry_run": meta.get("latest_signal_dry_run") is True,
        "executed": meta.get("latest_signal_execution_status") == "executed",
        "signal": {
            "symbol": "SOXL",
            "target_position": meta.get("latest_signal_target_position") or target,
            "current_position": 0.0,
            "price": latest.get("price"),
        },
        "risk_gate": {
            "approved": meta.get("latest_signal_risk_gate_approved"),
            "capped_target_position": target,
        },
        "execution_result": {
            "status": meta.get("latest_signal_execution_status"),
            "target_position": target,
        },
    }
    return public_paper_review_from_summary(
        pseudo_summary,
        summary_path=meta.get("latest_signal_summary_path"),
    )


def _sanitize_retained_paper_detail(detail: dict[str, object]) -> dict[str, object]:
    sanitized = dict(detail)
    rows = [dict(row) for row in sanitized.get("rows", []) if isinstance(row, dict)]
    review = _retained_paper_public_review(sanitized)
    if rows:
        rows[-1]["entry_status"] = review.get("entry_status")
        rows[-1]["entry_side"] = review.get("entry_side")
        rows[-1]["entry_target_position"] = review.get("entry_target_position")
        rows[-1]["target_position"] = review.get("entry_target_position")
    sanitized["rows"] = _prepare_paper_trading_rows(rows)

    old_meta = sanitized.get("meta")
    old_meta = old_meta if isinstance(old_meta, dict) else {}
    meta = {
        "source_kind": "paper_trading",
        "source": old_meta.get("source") or sanitized.get("csv_path") or "runtime/soxl/paper_trading/paper_trading_soxl_daily.csv",
        "latest_data_date": old_meta.get("latest_data_date") or review.get("date"),
        "latest_signal_date": old_meta.get("latest_signal_date") or review.get("date"),
        "latest_signal_profile": old_meta.get("latest_signal_profile") or review.get("profile"),
        "latest_signal_execution_status": old_meta.get("latest_signal_execution_status") or review.get("entry_status"),
        "latest_signal_target_position": old_meta.get("latest_signal_target_position"),
        "latest_signal_capped_target_position": old_meta.get("latest_signal_capped_target_position") or review.get("entry_target_position"),
        "latest_signal_summary_path": old_meta.get("latest_signal_summary_path") or review.get("summary_path"),
        "latest_signal_broker": old_meta.get("latest_signal_broker") or review.get("broker"),
        "latest_signal_dry_run": old_meta.get("latest_signal_dry_run") if "latest_signal_dry_run" in old_meta else review.get("dry_run"),
        "latest_signal_risk_gate_approved": old_meta.get("latest_signal_risk_gate_approved") if "latest_signal_risk_gate_approved" in old_meta else review.get("risk_gate_approved"),
        "latest_public_review": review,
        "latest_entry_status": review.get("entry_status"),
        "latest_entry_side": review.get("entry_side"),
        "latest_entry_target_position": review.get("entry_target_position"),
    }
    for source, target in (
        ("latest_signal_slow_profile", "latest_signal_slow_profile"),
        ("latest_signal_fast_profile", "latest_signal_fast_profile"),
        ("latest_signal_mode", "latest_signal_mode"),
    ):
        if old_meta.get(source) is not None:
            meta[target] = old_meta[source]
    sanitized["meta"] = {key: value for key, value in meta.items() if value is not None}
    return sanitized


def _load_versioned_data(features: dict[str, dict[str, float]] | None) -> dict[str, dict]:
    """Load all versioned backtest results from runtime/execution/generalization/versions/."""
    versioned: dict[str, dict] = {}
    versions_dir = RUNTIME_DIR / "versions"
    if not versions_dir.exists():
        return versioned
    for version_dir in sorted(versions_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        detail_dir = version_dir / "detailed"
        summary_path = detail_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = load_json(summary_path)
        except Exception:
            continue
        version_rows: dict[str, dict] = {}
        version_windows: dict[str, dict] = {}
        for item in summary:
            etf = item.get("etf")
            if not etf:
                continue
            csv_path = detail_dir / f"{etf.lower()}_detailed_trades.csv"
            if not csv_path.exists():
                continue
            try:
                rows = load_csv_rows(csv_path)
            except Exception:
                continue
            if features:
                for r in rows:
                    dt = str(r["datetime"])
                    r["_features"] = features.get(dt, {})
            else:
                for r in rows:
                    r["_features"] = {}
            ohlc = load_ohlc_and_indicators(etf)
            for r in rows:
                dt = str(r["datetime"])
                if dt in ohlc:
                    r.update(ohlc[dt])
            regime_shapes = compute_regime_shapes(rows)
            trades = compute_trades(rows)
            kelly_stats = compute_kelly_stats(trades)
            version_rows[etf] = {
                "summary": item,
                "csv_path": str(csv_path.relative_to(ROOT)),
                "rows": compact_payload_rows(rows),
                "regime_shapes": regime_shapes,
                "trades": trades,
                "kelly": kelly_stats,
            }
            version_windows[etf] = {
                "start": rows[0]["datetime"] if rows else "",
                "end": rows[-1]["datetime"] if rows else "",
                "rows": len(rows),
            }
        if version_rows:
            versioned[version_dir.name] = {
                "_meta": {
                    "updated_at": summary_path.stat().st_mtime,
                    "source": str(summary_path.relative_to(ROOT)),
                    "source_kind": "generalization_version",
                    "shared_market_data": True,
                },
                "detailed": version_rows,
                "windows": version_windows,
            }
    return versioned


def build_payload() -> dict[str, object]:
    detailed_summary_path = DETAIL_DIR / "summary.json"
    detailed_summary = load_json(detailed_summary_path) if detailed_summary_path.exists() else []
    detailed_rows = {}
    windows = {}
    features = load_features()
    for item in detailed_summary:
        etf = item["etf"]
        rows = load_csv_rows(DETAIL_DIR / f"{etf.lower()}_detailed_trades.csv")
        ohlc = load_ohlc_and_indicators(etf)
        if features:
            for r in rows:
                dt = str(r["datetime"])
                r["_features"] = features.get(dt, {})
        else:
            for r in rows:
                r["_features"] = {}
        # Merge OHLC + indicators into rows
        for r in rows:
            dt = str(r["datetime"])
            if dt in ohlc:
                r.update(ohlc[dt])
        regime_shapes = compute_regime_shapes(rows)
        trades = compute_trades(rows)
        kelly_stats = compute_kelly_stats(trades)
        detailed_rows[etf] = {
            "summary": item,
            "csv_path": f"detailed/{etf.lower()}_detailed_trades.csv",
            "rows": rows,
            "regime_shapes": regime_shapes,
            "trades": trades,
            "kelly": kelly_stats,
        }
        windows[etf] = {
            "start": rows[0]["datetime"] if rows else "",
            "end": rows[-1]["datetime"] if rows else "",
            "rows": len(rows),
        }
    versioned = _load_versioned_data(features)
    ptr_versioned = _load_post_train_review_versions(features)
    versioned.update(ptr_versioned)
    retained_versioned = _load_retained_dashboard_versions()
    for tag, payload in retained_versioned.items():
        versioned.setdefault(tag, payload)

    # If no detailed data exists but versioned data does, populate from latest version
    # so that the dashboard has a default ETF to display.
    if not detailed_rows and versioned:
        latest_tag = select_latest_version(versioned)
        if latest_tag and latest_tag in versioned:
            latest_payload = versioned[latest_tag]
            for etf, detail in (latest_payload.get("detailed") or {}).items():
                detailed_rows[etf] = detail
                windows[etf] = latest_payload.get("windows", {}).get(etf, {
                    "start": detail["rows"][0]["datetime"] if detail.get("rows") else "",
                    "end": detail["rows"][-1]["datetime"] if detail.get("rows") else "",
                    "rows": len(detail.get("rows", [])),
                })
                if detailed_summary is not None and isinstance(detailed_summary, list):
                    detailed_summary.append(detail.get("summary", {}))

    model_versions = load_model_versions()
    research_backtests = load_research_backtests()
    best_version = select_best_version(model_versions, versioned)
    latest_version = select_latest_version(versioned)

    return {
        "generated_from": {
            "summary": "runtime/execution/generalization/detailed/summary.json",
            "detailed_dir": "runtime/execution/generalization/detailed",
            "results": "runtime/execution/generalization/results.json",
            "rule_based": "runtime/execution/generalization/rule_based_results.json",
            "execution_overlay": "runtime/execution/generalization/execution_overlay_results.json",
            "execution_review": "runtime/execution/generalization/execution_overlay_review.md",
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": detailed_summary,
        "windows": windows,
        "detailed": detailed_rows,
        "versioned": versioned,
        "best_version": best_version,
        "latest_version": latest_version,
        "results": load_json_or(RUNTIME_DIR / "results.json", {}),
        "rule_based": load_json_or(RUNTIME_DIR / "rule_based_results.json", {}),
        "execution_overlay": load_json_or(RUNTIME_DIR / "execution_overlay_results.json", {}),
        "execution_review": (RUNTIME_DIR / "execution_overlay_review.md").read_text(encoding="utf-8")
        if (RUNTIME_DIR / "execution_overlay_review.md").exists()
        else "",
        "model_versions": model_versions,
        "research_backtests": research_backtests,
    }


def _build_layout_all() -> dict:
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.24, 0.24, 0.16, 0.18, 0.18],
        subplot_titles=("NAV", "Price + Signals", "Position", "Excess Return %", "Drawdown"),
        specs=[[{}], [{}], [{}], [{}], [{}]],
    )
    fig.update_layout(
        height=920,
        margin=dict(l=60, r=60, t=90, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C_PANEL,
        font=dict(family='ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', color=C_INK, size=12),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,255,255,0.92)"),
        dragmode="zoom",
    )
    for i in range(1, 6):
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor=C_LINE,
            showticklabels=(i == 5), tickfont=dict(size=11),
            showspikes=True, spikemode="across", spikecolor=C_MUTED, spikethickness=1, spikedash="dot",
            rangeslider_visible=False,
            row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor=C_LINE,
            tickfont=dict(size=11),
            showspikes=True, spikemode="across", spikecolor=C_MUTED, spikethickness=1, spikedash="dot",
            fixedrange=True,
            row=i, col=1,
        )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Position %", range=[0, 1], tickformat=".0%", row=3, col=1)
    fig.update_yaxes(title_text="Excess %", row=4, col=1)
    fig.update_yaxes(title_text="DD %", row=5, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig.to_plotly_json()["layout"]


def build_plotly_layouts() -> dict[str, dict]:
    return {"all": _build_layout_all()}


def sanitize_json_floats(raw: str) -> str:
    """Replace JSON-invalid NaN / Infinity / -Infinity with null."""
    import re
    # Match bare NaN / Infinity / -Infinity that appear as JSON values
    # (not inside strings). Use negative lookbehind/lookahead for quotes.
    raw = re.sub(r'(?<!["\w])NaN(?!["\w])', 'null', raw)
    raw = re.sub(r'(?<!["\w])-Infinity(?!["\w])', 'null', raw)
    raw = re.sub(r'(?<!["\w])Infinity(?!["\w])', 'null', raw)
    return raw


def write_versioned_json(versioned: dict[str, dict]) -> None:
    """Write each version's per-ETF data as individual JSON files."""
    versioned_dir = DASHBOARD_DATA_DIR / "versioned"
    versioned_dir.mkdir(parents=True, exist_ok=True)
    for tag, payload in versioned.items():
        tag_dir = versioned_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        for etf, detail in (payload.get("detailed") or {}).items():
            out_path = tag_dir / f"{etf.upper()}.json"
            out_path.write_text(
                sanitize_json_floats(json.dumps(detail, ensure_ascii=False)),
                encoding="utf-8",
            )


def write_model_versions_json(model_versions: list[dict]) -> None:
    out_path = DASHBOARD_DATA_DIR / "model_versions.json"
    out_path.write_text(
        sanitize_json_floats(json.dumps(model_versions, ensure_ascii=False)),
        encoding="utf-8",
    )


def write_research_backtests_json(research_backtests: list[dict]) -> None:
    out_path = DASHBOARD_DATA_DIR / "research_backtests.json"
    export_rows = []
    for row in research_backtests:
        export_row = dict(row)
        export_row.pop("updated_at_ts", None)
        export_rows.append(export_row)
    out_path.write_text(
        sanitize_json_floats(json.dumps(export_rows, ensure_ascii=False)),
        encoding="utf-8",
    )


def write_index_json(
    detailed: dict,
    versioned: dict[str, dict],
    model_versions: list[dict],
    best_version: str,
    latest_version: str,
    research_backtests: list[dict] | None = None,
) -> None:
    """Write a lightweight index with ETF lists, version metadata, and summaries."""
    index = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "etfs": list(detailed.keys()),
        "versions": list(versioned.keys()),
        "best_version": best_version,
        "latest_version": latest_version,
        "version_summaries": {},
    }
    for tag, payload in versioned.items():
        index["version_summaries"][tag] = {
            "etfs": list((payload.get("detailed") or {}).keys()),
            "windows": payload.get("windows", {}),
            "meta": payload.get("_meta", {}),
        }
    # Also include current/detailed summaries
    index["detailed_summaries"] = {
        etf: detail.get("summary", {}) for etf, detail in detailed.items()
    }
    index["model_version_count"] = len(model_versions)
    index["research_backtest_count"] = len(research_backtests or [])
    out_path = DASHBOARD_DATA_DIR / "index.json"
    out_path.write_text(
        sanitize_json_floats(json.dumps(index, ensure_ascii=False)),
        encoding="utf-8",
    )


def build_html(payload: dict[str, object], layouts: dict[str, dict]) -> str:
    payload_json = sanitize_json_floats(json.dumps(payload, ensure_ascii=False))
    layouts_json = sanitize_json_floats(json.dumps(layouts, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #eef2f5;
      --panel: #ffffff;
      --panel-2: #f7f9fb;
      --ink: #111827;
      --muted: #607080;
      --line: #d7dee6;
      --accent: #2563eb;
      --accent-2: #0f766e;
      --good: #0f766e;
      --bad: #b91c1c;
      --shadow: 0 10px 28px rgba(17, 24, 39, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    .shell {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .hero h1 {{
      margin: 0 0 12px;
      font-size: 34px;
      line-height: 1;
      letter-spacing: 0;
    }}
    .sub {{
      color: var(--muted);
      line-height: 1.55;
      margin: 0;
    }}
    .chart-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: flex-end;
      margin-bottom: 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }}
    .toolbar-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .toolbar-group label {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .help-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 14px;
      height: 14px;
      border-radius: 999px;
      background: var(--line);
      color: var(--muted);
      font-size: 10px;
      cursor: help;
      position: relative;
    }}
    .help-icon:hover::after {{
      content: attr(data-tip);
      position: absolute;
      bottom: 120%;
      left: 50%;
      transform: translateX(-50%);
      background: var(--ink);
      color: white;
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 11px;
      white-space: nowrap;
      z-index: 100;
      letter-spacing: 0;
      text-transform: none;
    }}
    select, .date-input {{
      min-width: 140px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #ffffff;
      font-size: 14px;
      color: var(--ink);
      font-family: inherit;
    }}
    .date-input {{ min-width: 130px; }}
    .pill-row {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #ffffff;
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
      font-family: inherit;
    }}
    .pill.active {{
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }}
    .zoom-row {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .btn {{
      padding: 9px 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      cursor: pointer;
      color: var(--ink);
      font-size: 13px;
      font-family: inherit;
    }}
    .btn.primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 18px;
    }}
    .metric {{
      background: var(--panel-2);
      border-radius: 8px;
      padding: 18px;
      border: 1px solid var(--line);
    }}
    .metric .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .metric .v {{
      font-size: 30px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .metric .s {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .jump-links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 8px;
      background: #ffffff;
      border: 1px solid var(--line);
      color: var(--ink);
      font-size: 13px;
      text-decoration: none;
    }}
    .badge.good {{
      background: rgba(15, 118, 110, 0.10);
      border-color: rgba(15, 118, 110, 0.25);
    }}
    .badge.bad {{
      background: rgba(185, 28, 28, 0.08);
      border-color: rgba(185, 28, 28, 0.20);
    }}
    .production-status {{
      display: grid;
      grid-template-columns: minmax(220px, 0.9fr) minmax(150px, 0.65fr) minmax(170px, 0.75fr) minmax(320px, 1.4fr) minmax(130px, 0.6fr);
      gap: 10px;
      align-items: stretch;
      margin-bottom: 18px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .production-status[hidden] {{
      display: none;
    }}
    .production-main, .production-detail {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      padding: 10px 12px;
    }}
    .production-main {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      justify-content: center;
    }}
    .production-label, .production-k {{
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .production-value {{
      overflow-wrap: normal;
      word-break: keep-all;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-size: 14px;
      font-weight: 700;
      color: var(--ink);
    }}
    .production-detail {{
      display: flex;
      flex-direction: column;
      gap: 5px;
      justify-content: center;
      color: var(--ink);
      font-size: 13px;
    }}
    .production-v {{
      overflow-wrap: anywhere;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-weight: 700;
    }}
    .production-paper-model {{
      grid-column: span 2;
    }}
    .production-paper-model .production-v {{
      font-size: 12px;
      line-height: 1.25;
    }}
    .production-paper-source {{
      grid-column: span 3;
    }}
    .production-blockers {{
      grid-column: span 2;
    }}
    .production-model-replay {{
      grid-column: span 2;
    }}
    .production-link {{
      color: var(--accent-2);
      text-decoration: none;
    }}
    .production-link:hover {{
      text-decoration: underline;
    }}
    .production-tone-good {{
      color: var(--good);
    }}
    .production-tone-bad {{
      color: var(--bad);
    }}
    .production-tone-warn {{
      color: var(--accent);
    }}
    .research-panel {{
      margin-bottom: 18px;
    }}
    .research-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .research-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .research-metric {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px 12px;
    }}
    .research-metric .rk {{
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .research-metric .rv {{
      overflow-wrap: anywhere;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
      font-size: 13px;
      font-weight: 700;
    }}
    .research-table td, .research-table th {{
      vertical-align: top;
      white-space: normal;
    }}
    .research-table td:nth-child(1), .research-table th:nth-child(1) {{
      min-width: 240px;
    }}
    .research-table td:nth-child(2), .research-table th:nth-child(2) {{
      min-width: 180px;
    }}
    .research-table td:nth-child(7), .research-table th:nth-child(7) {{
      min-width: 260px;
      text-align: left;
    }}
    .research-path {{
      color: var(--accent-2);
      text-decoration: none;
      overflow-wrap: anywhere;
    }}
    .research-path:hover {{
      text-decoration: underline;
    }}
    .chart-panel {{ min-height: 420px; }}
    .chart-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .chart-wrapper {{
      display: grid;
      grid-template-columns: 1fr 240px;
      gap: 14px;
      align-items: start;
    }}
    .regime-strip {{
      display: flex;
      height: 10px;
      border-radius: 5px;
      overflow: hidden;
      margin: 0 0 8px 0;
      border: 1px solid var(--line);
    }}
    .regime-bar {{
      flex: 1;
      min-width: 1px;
      cursor: crosshair;
    }}
    .info-panel {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font-size: 12px;
      line-height: 1.6;
      min-height: 400px;
    }}
    .info-panel .info-title {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .info-row {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(215,222,230,0.7);
    }}
    .info-row .info-k {{
      color: var(--muted);
    }}
    .info-row .info-v {{
      font-weight: 600;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    }}
    .info-row.highlight {{
      background: rgba(37, 99, 235, 0.08);
      margin: 0 -14px;
      padding: 4px 14px;
      border-bottom: 1px solid var(--line);
    }}
    .info-row.highlight .info-k {{
      color: var(--accent);
      font-weight: 600;
    }}
    .info-empty {{
      color: var(--muted);
      font-style: italic;
      text-align: center;
      padding: 40px 0;
    }}
    .info-section {{
      margin-top: 12px;
      padding-top: 8px;
      border-top: 1px solid var(--line);
    }}
    .info-section-title {{
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .pos-bar-wrap {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin: 3px 0;
    }}
    .pos-bar-label {{
      width: 40px;
      font-size: 10px;
      color: var(--muted);
      text-align: right;
    }}
    .pos-bar-track {{
      flex: 1;
      height: 10px;
      background: rgba(215,222,230,0.55);
      border-radius: 5px;
      overflow: hidden;
    }}
    .pos-bar-fill {{
      height: 100%;
      border-radius: 5px;
      background: var(--accent-2);
      opacity: 0.7;
    }}
    .pos-bar-pct {{
      width: 32px;
      font-size: 10px;
      color: var(--muted);
      text-align: right;
      font-family: ui-monospace, monospace;
    }}
    .trade-list {{
      max-height: 120px;
      overflow-y: auto;
    }}
    .trade-item {{
      padding: 4px 0;
      border-bottom: 1px solid rgba(215,222,230,0.55);
      font-size: 11px;
      line-height: 1.4;
    }}
    .trade-item:last-child {{ border-bottom: none; }}
    .trade-entry {{ color: var(--good); }}
    .trade-exit {{ color: var(--bad); }}

    .trade-table-wrap {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}
    .trade-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .trade-table th, .trade-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    .trade-table th:first-child, .trade-table td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      background: #ffffff;
      z-index: 1;
    }}
    .trade-table th {{
      background: var(--panel-2);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      position: sticky;
      top: 0;
      z-index: 2;
      cursor: pointer;
      user-select: none;
    }}
    .trade-table th:hover {{
      background: var(--line);
    }}
    .trade-table th.sort-asc::after {{ content: ' ▲'; font-size: 9px; }}
    .trade-table th.sort-desc::after {{ content: ' ▼'; font-size: 9px; }}
    .trade-table tr.clickable {{ cursor: pointer; }}
    .trade-table tr.clickable:hover {{ background: rgba(37, 99, 235, 0.06); }}
    .trade-table td.good {{ color: var(--good); font-weight: 700; }}
    .trade-table td.bad {{ color: var(--bad); font-weight: 700; }}

    .panel-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .panel-card h2 {{
      margin: 0 0 14px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    .stat-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 14px;
    }}
    .stat-cell {{
      background: var(--panel-2);
      border-radius: 8px;
      padding: 12px;
      border: 1px solid var(--line);
      text-align: center;
    }}
    .stat-cell .sk {{
      font-size: 10px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .stat-cell .sv {{
      font-size: 20px;
      font-weight: 700;
      font-family: ui-monospace, monospace;
    }}
    .stat-cell .ss {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 4px;
    }}
    .kelly-highlight {{
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(37, 99, 235, 0.06));
    }}

    /* Model version comparison */
    .version-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .version-table th, .version-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    .version-table th:first-child, .version-table td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      background: #ffffff;
      z-index: 1;
    }}
    .version-table th {{
      background: var(--panel-2);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .version-table td.promoted {{
      background: rgba(15, 118, 110, 0.06);
      font-weight: 600;
    }}
    .version-table td.best {{
      background: rgba(37, 99, 235, 0.08);
      font-weight: 700;
    }}
    .version-table tr.clickable-version {{
      cursor: pointer;
    }}
    .version-table tr.clickable-version:hover {{
      background: rgba(37, 99, 235, 0.06);
    }}
    .version-table tr.clickable-version.selected {{
      background: rgba(37, 99, 235, 0.10);
      outline: 2px solid var(--accent);
      outline-offset: -2px;
    }}
    .version-wrap {{
      max-height: 400px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
    }}

    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .mono {{ font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }}
    .raw-tools {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      align-items: center;
    }}
    .button-link {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      text-decoration: none;
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .production-status {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .production-paper-model, .production-paper-source {{ grid-column: auto; }}
      .production-blockers {{ grid-column: auto; }}
      .research-summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-wrapper {{ grid-template-columns: 1fr; }}
      .info-panel {{ display: none; }}
      .panel-grid {{ grid-template-columns: 1fr; }}
      .stat-row {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 14px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
      .production-status {{ grid-template-columns: 1fr; }}
      .production-value {{ overflow-wrap: anywhere; font-size: 13px; }}
      .research-head {{ flex-direction: column; align-items: flex-start; }}
      .research-summary {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 28px; }}
      .chart-toolbar {{ flex-direction: column; align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="jump-links" style="margin-bottom:18px;">
      <a class="badge" href="#researchBacktests">Research Backtests</a>
      <a class="badge" href="#mainChart">Main Chart</a>
      <a class="badge" href="#tradeStats">Trade Quality</a>
      <a class="badge" href="#tradeTable">Trade Performance</a>
    </div>

    <section class="production-status" id="productionStatusCard" hidden aria-live="polite">
      <div class="production-main">
        <span class="production-label">Production</span>
        <span class="production-value" id="productionStatusValue">loading</span>
      </div>
      <div class="production-detail">
        <span class="production-k">Auth</span>
        <span class="production-v" id="productionAuthStatus">n/a</span>
      </div>
      <div class="production-detail">
        <span class="production-k">Project</span>
        <span class="production-v" id="productionProjectStatus">n/a</span>
      </div>
      <div class="production-detail production-paper-model">
        <span class="production-k">Paper Model</span>
        <span class="production-v" id="productionPaperModel">n/a</span>
      </div>
      <div class="production-detail">
        <span class="production-k">Paper Exec</span>
        <span class="production-v" id="productionPaperExecution">n/a</span>
      </div>
      <div class="production-detail production-paper-source">
        <span class="production-k">Paper Source</span>
        <span class="production-v" id="productionPaperSource">n/a</span>
      </div>
      <div class="production-detail">
        <span class="production-k">Paper End</span>
        <span class="production-v" id="productionPaperEnd">n/a</span>
      </div>
      <div class="production-detail">
        <span class="production-k">Deploy</span>
        <span class="production-v" id="productionDeploymentStatus">n/a</span>
      </div>
      <div class="production-detail">
        <span class="production-k">Gates</span>
        <span class="production-v" id="productionGateOverall">n/a</span>
      </div>
      <div class="production-detail production-blockers">
        <span class="production-k">Blockers</span>
        <span class="production-v" id="productionGateIssues">n/a</span>
      </div>
      <div class="production-detail production-model-replay">
        <span class="production-k">Model Replays</span>
        <span class="production-v" id="productionModelReplayStatus">n/a</span>
      </div>
    </section>

    <section class="card research-panel" id="researchBacktests">
      <div class="research-head">
        <h2 class="section-title" style="margin:0;">Research Backtests</h2>
        <div class="sub">Latest runtime/model_selection notebook diagnostics exported for Vercel.</div>
      </div>
      <div class="research-summary" id="researchBacktestSummary">
        <div class="research-metric">
          <div class="rk">Status</div>
          <div class="rv">loading</div>
        </div>
      </div>
      <div class="version-wrap">
        <table class="version-table research-table" id="researchBacktestTable"></table>
      </div>
    </section>

    <div class="grid summary-grid" id="metricGrid"></div>

    <!-- Main Chart -->
    <div class="card chart-panel" id="mainChart">
      <h2 class="section-title">Main Chart</h2>
      <div class="chart-toolbar">
        <div class="toolbar-group">
          <label>ETF <span class="help-icon" data-tip="选择要查看的 ETF 标的">?</span></label>
          <select id="etfSelect"></select>
        </div>
        <div class="toolbar-group">
          <label>Model <span class="help-icon" data-tip="选择回测模型版本">?</span></label>
          <select id="versionSelect"></select>
        </div>
        <div class="toolbar-group">
          <label>Range <span class="help-icon" data-tip="按年份快速筛选数据">?</span></label>
          <div class="pill-row">
            <button class="pill active" data-range="all">All</button>
            <button class="pill" data-range="2020">2020</button>
            <button class="pill" data-range="2022">2022</button>
            <button class="pill" data-range="2023_2024">23-24</button>
          </div>
        </div>
        <div class="toolbar-group">
          <label>Zoom <span class="help-icon" data-tip="快速缩放到最近 N 个月 [+/-]">?</span></label>
          <div class="zoom-row">
            <button class="btn" data-zoom="6m">6M</button>
            <button class="btn" data-zoom="1y">1Y</button>
            <button class="btn" data-zoom="2y">2Y</button>
            <button class="btn" data-zoom="3y">3Y</button>
          </div>
        </div>
        <div class="toolbar-group">
          <label>Chart Start</label>
          <input id="startDateInput" class="date-input" type="date">
        </div>
        <div class="toolbar-group">
          <label>Chart End</label>
          <input id="endDateInput" class="date-input" type="date">
        </div>
        <div class="toolbar-group">
          <label>Action</label>
          <div class="zoom-row">
            <button class="btn primary" id="applyZoomBtn" type="button">Apply</button>
            <button class="btn" id="resetZoomBtn" type="button">Reset <span style="font-size:10px;color:var(--muted);margin-left:4px;opacity:0.7;">[R]</span></button>
          </div>
        </div>
        <div class="toolbar-group">
          <label>Slice & Recompute <span class="help-icon" data-tip="以当前选中的日期区间为起点，重新计算收益率、回撤、交易统计（纯前端，无需服务器）">?</span></label>
          <div class="zoom-row">
            <button class="btn primary" id="sliceRecomputeBtn" type="button">Recompute</button>
          </div>
        </div>
        <div class="toolbar-group">
          <label>Run Model Backtest <span class="help-icon" data-tip="选择模型并在当前日期区间运行回测（需要本地启动 dashboard_server.py）">?</span></label>
          <div class="zoom-row">
            <select id="runProfileSelect" style="min-width:120px;padding:9px 10px;border-radius:8px;border:1px solid var(--line);background:var(--panel);font-size:13px;">
              <option value="">Select profile...</option>
            </select>
            <input id="runStartDateInput" class="date-input" type="date">
            <input id="runEndDateInput" class="date-input" type="date">
            <button class="btn primary" id="runBacktestBtn" type="button">Run</button>
          </div>
        </div>
      </div>
      <div id="segmentBanner" style="display:none;background:rgba(15,118,110,0.08);border:1px solid rgba(15,118,110,0.3);border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:13px;color:var(--accent-2);align-items:center;gap:8px;">
        <span>Segment View</span>
        <span id="segmentRange" style="font-family:ui-monospace,monospace;font-weight:600;"></span>
        <span style="color:var(--muted);">— 所有指标已按该区间重新计算</span>
        <button class="btn" id="clearSegmentBtn" type="button" style="margin-left:auto;padding:6px 12px;font-size:12px;">Restore Full Range</button>
      </div>
      <div class="chart-meta" id="chartMeta"></div>
      <div id="regimeStrip" class="regime-strip"></div>
      <div class="chart-wrapper">
        <div id="plotlyChart" style="width:100%;height:920px;border-radius:8px;background:var(--panel);border:1px solid var(--line);"></div>
        <div class="info-panel" id="infoPanel">
          <div class="info-title">数据面板</div>
          <div class="info-empty" id="infoEmpty">鼠标移到图表上查看当日数据</div>
          <div id="infoContent" style="display:none;">
            <div class="info-row highlight">
              <span class="info-k">日期</span>
              <span class="info-v" id="infoDate">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Action</span>
              <span class="info-v" id="infoAction">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Profile</span>
              <span class="info-v" id="infoProfile">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Execution</span>
              <span class="info-v" id="infoExecution">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Target</span>
              <span class="info-v" id="infoTarget">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Open</span>
              <span class="info-v" id="infoOpen">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">High</span>
              <span class="info-v" id="infoHigh">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Low</span>
              <span class="info-v" id="infoLow">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Close</span>
              <span class="info-v" id="infoClose">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Strategy NAV</span>
              <span class="info-v" id="infoStrat">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Buy&Hold NAV</span>
              <span class="info-v" id="infoBH">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Position</span>
              <span class="info-v" id="infoPos">-</span>
            </div>
            <div class="info-row">
              <span class="info-k">Excess %</span>
              <span class="info-v" id="infoExcess">-</span>
            </div>
            <div class="info-section">
              <div class="info-section-title">市场特征</div>
              <div class="info-row"><span class="info-k">State</span><span class="info-v" id="infoState">-</span></div>
              <div class="info-row"><span class="info-k">Panic</span><span class="info-v" id="infoPanic">-</span></div>
              <div class="info-row"><span class="info-k">RSI6</span><span class="info-v" id="infoRsi6">-</span></div>
              <div class="info-row"><span class="info-k">HV20</span><span class="info-v" id="infoHv20">-</span></div>
              <div class="info-row"><span class="info-k">VolCoef</span><span class="info-v" id="infoVolCoef">-</span></div>
              <div class="info-row"><span class="info-k">VIX</span><span class="info-v" id="infoVix">-</span></div>
              <div class="info-row"><span class="info-k">RSI</span><span class="info-v" id="infoRsi">-</span></div>
              <div class="info-row"><span class="info-k">Volatility</span><span class="info-v" id="infoVol">-</span></div>
              <div class="info-row"><span class="info-k">Spread</span><span class="info-v" id="infoSpread">-</span></div>
              <div class="info-row"><span class="info-k">VIX z-score</span><span class="info-v" id="infoVixZ">-</span></div>
              <div class="info-row"><span class="info-k">Spread Trend</span><span class="info-v" id="infoSpTr">-</span></div>
              <div class="info-row"><span class="info-k">MA5/20</span><span class="info-v" id="infoMA">-</span></div>
              <div class="info-row"><span class="info-k">Vol Strength</span><span class="info-v" id="infoVolStr">-</span></div>
            </div>
            <div class="info-section" id="tradeSection" style="display:none;">
              <div class="info-section-title">交易详情</div>
              <div class="info-row"><span class="info-k">入场日期</span><span class="info-v" id="tradeEntry">-</span></div>
              <div class="info-row"><span class="info-k">出场日期</span><span class="info-v" id="tradeExit">-</span></div>
              <div class="info-row"><span class="info-k">持仓天数</span><span class="info-v" id="tradeDays">-</span></div>
              <div class="info-row"><span class="info-k">价格收益</span><span class="info-v" id="tradeRet">-</span></div>
            </div>
            <div class="info-section">
              <div class="info-section-title">仓位分布</div>
              <div id="posDist"></div>
            </div>
            <div class="info-section">
              <div class="info-section-title">最近交易 <span style="font-weight:normal;color:var(--muted);">(最近5笔)</span></div>
              <div class="trade-list" id="tradeList"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Trade Quality Stats -->
    <div class="card" id="tradeStats">
      <h2 class="section-title">Trade Quality <span style="font-size:12px;color:var(--muted);font-weight:normal;">— Kelly / Expectancy / Profit Factor</span></h2>
      <div class="stat-row" id="tradeStatRow"></div>
    </div>

    <!-- Trade Performance Table -->
    <div class="card" id="tradeTable">
      <h2 class="section-title">Trade Performance — 每笔 T 收益明细</h2>
      <div class="raw-tools">
        <a id="csvLink" class="button-link" href="#" target="_blank" rel="noreferrer">Open Raw CSV</a>
        <div class="sub">点击表头可排序，点击交易行聚焦到图表对应区间。BUY → SELL 配对。</div>
      </div>
      <div class="trade-table-wrap">
        <table class="trade-table" id="tradePerfTable"></table>
      </div>
    </div>

  </div>

  <script>
    const DATA = {payload_json};
    const PLOTLY_LAYOUTS = {layouts_json};
    const ETFS = (() => {{
      const fromDetailed = Object.keys(DATA.detailed || {{}});
      if (fromDetailed.length) return fromDetailed;
      // Fallback: collect ETFs from versioned data
      const etfs = new Set();
      Object.values(DATA.versioned || {{}}).forEach(v => {{
        Object.keys(v.detailed || {{}}).forEach(e => etfs.add(e));
      }});
      return Array.from(etfs);
    }})();
    const VERSIONS = Object.keys(DATA.versioned || {{}});
    const state = {{
      etf: ETFS[0] || 'SOXL',
      version: DATA.latest_version || DATA.best_version || '',
      versionSource: 'auto_latest',
      range: 'all',
      zoomMode: 'all',
      zoomStart: null,
      zoomEnd: null,
      sortCol: 'return_pct',
      sortAsc: false,
      segmentMode: false,
      segmentKey: '',
      segmentCache: {{}},
    }};

    const etfSelect = document.getElementById('etfSelect');
    const versionSelect = document.getElementById('versionSelect');
    const metricGrid = document.getElementById('metricGrid');
    const tradeStatRow = document.getElementById('tradeStatRow');
    const tradePerfTable = document.getElementById('tradePerfTable');
    const csvLink = document.getElementById('csvLink');
    const chartMeta = document.getElementById('chartMeta');
    const startDateInput = document.getElementById('startDateInput');
    const endDateInput = document.getElementById('endDateInput');
    const runProfileSelect = document.getElementById('runProfileSelect');
    const runStartDateInput = document.getElementById('runStartDateInput');
    const runEndDateInput = document.getElementById('runEndDateInput');
    const applyZoomBtn = document.getElementById('applyZoomBtn');
    const resetZoomBtn = document.getElementById('resetZoomBtn');
    const infoEmpty = document.getElementById('infoEmpty');
    const infoContent = document.getElementById('infoContent');
    const regimeStrip = document.getElementById('regimeStrip');
    const productionStatusCard = document.getElementById('productionStatusCard');
    const productionStatusValue = document.getElementById('productionStatusValue');
    const productionAuthStatus = document.getElementById('productionAuthStatus');
    const productionProjectStatus = document.getElementById('productionProjectStatus');
    const productionPaperModel = document.getElementById('productionPaperModel');
    const productionPaperExecution = document.getElementById('productionPaperExecution');
    const productionPaperSource = document.getElementById('productionPaperSource');
    const productionPaperEnd = document.getElementById('productionPaperEnd');
    const productionDeploymentStatus = document.getElementById('productionDeploymentStatus');
    const productionGateOverall = document.getElementById('productionGateOverall');
    const productionGateIssues = document.getElementById('productionGateIssues');
    const researchBacktestSummary = document.getElementById('researchBacktestSummary');
    const researchBacktestTable = document.getElementById('researchBacktestTable');
    const STORAGE_KEY = 'soxl-dashboard-state-v10';
    const hydratedVersionCache = {{}};
    let indexVersionSummaries = {{}};
    let modelReplayProfiles = [];
    const baseRowLookup = Object.fromEntries(
      ETFS.map(etf => [
        etf,
        Object.fromEntries(
          (DATA.detailed[etf].rows || []).map(row => [
            row.datetime,
            {{
              open: row.open,
              high: row.high,
              low: row.low,
              close: row.close,
              volume: row.volume,
              rsi6: row.rsi6,
              ma180: row.ma180,
              hv_20: row.hv_20,
              vol_coef: row.vol_coef,
              panic_score: row.panic_score,
              market_state: row.market_state,
              _features: row._features || {{}},
            }},
          ])
        ),
      ])
    );

    function fmtPct(v, digits = 2) {{
      if (v == null || !Number.isFinite(v)) return 'n/a';
      return `${{(v * 100).toFixed(digits)}}%`;
    }}
    function fmtNum(v, digits = 2) {{
      if (v == null || !Number.isFinite(v)) return 'n/a';
      return Number(v).toFixed(digits);
    }}
    function fmtDelta(v, digits = 2) {{
      const n = Number(v);
      if (!Number.isFinite(n)) return 'n/a';
      const sign = n > 0 ? '+' : '';
      return `${{sign}}${{n.toFixed(digits)}}`;
    }}
    function fmtMoney(v) {{
      if (v == null || !Number.isFinite(v)) return 'n/a';
      const n = Number(v);
      const sign = n >= 0 ? '+' : '';
      return `${{sign}}${{n.toLocaleString('en-US', {{minimumFractionDigits:0, maximumFractionDigits:0}})}}`;
    }}

    function loadSavedState() {{
      try {{
        const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
        if (!parsed) return;
        Object.assign(state, parsed);
        if (parsed.versionSource !== 'manual' && parsed.versionSource !== 'current') {{
          state.version = DATA.latest_version || DATA.best_version || '';
          state.versionSource = 'auto_latest';
        }}
        if (state.version && (!DATA.versioned || !DATA.versioned[state.version])) {{
          state.version = DATA.latest_version || DATA.best_version || '';
          state.versionSource = 'auto_latest';
        }}
        // Segment cache is not persisted; always start clean
        state.segmentMode = false;
        state.segmentKey = '';
        state.segmentCache = {{}};
      }} catch (_err) {{}}
    }}
    function saveState() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }}

    function hydrateVersionDetail(etf, version, detail) {{
      if (!detail) return null;
      const cacheKey = `${{version}}::${{etf}}`;
      if (hydratedVersionCache[cacheKey]) return hydratedVersionCache[cacheKey];
      const lookup = baseRowLookup[etf] || {{}};
      const hydrated = {{
        ...detail,
        rows: (detail.rows || []).map(row => {{
          const market = lookup[row.datetime] || {{}};
          return {{
            ...market,
            ...row,
            _features: market._features || {{}},
          }};
        }}),
      }};
      hydratedVersionCache[cacheKey] = hydrated;
      return hydrated;
    }}

    function getSegmentKey(etf, version, start, end) {{
      return `${{etf}}::${{version || 'current'}}::${{start}}::${{end}}`;
    }}

    function getDetailed(etf) {{
      if (!etf) return {{ rows: [], summary: {{}}, trades: [], kelly: {{}}, regime_shapes: [] }};
      if (state.segmentMode && state.segmentKey && state.segmentCache[state.segmentKey]) {{
        return state.segmentCache[state.segmentKey];
      }}
      if (state.version && DATA.versioned && DATA.versioned[state.version]) {{
        const versionDetail = (DATA.versioned[state.version].detailed || {{}})[etf];
        if (versionDetail) {{
          return hydrateVersionDetail(etf, state.version, versionDetail);
        }}
      }}
      return DATA.detailed[etf] || {{ rows: [], summary: {{}}, trades: [], kelly: {{}}, regime_shapes: [] }};
    }}
    function computeTradesFromRows(rows) {{
      const trades = [];
      let entryIdx = -1;
      let cumulative = 0.0;
      for (let i = 0; i < rows.length; i++) {{
        const row = rows[i];
        if (row.action === 'BUY' && entryIdx < 0) {{
          entryIdx = i;
        }} else if (row.action === 'SELL' && entryIdx >= 0) {{
          const entryRow = rows[entryIdx];
          const entryPrice = entryRow.price || entryRow.close || entryRow.strategy_nav;
          const exitPrice = row.price || row.close || row.strategy_nav;
          const days = i - entryIdx;
          const retPct = entryPrice > 0 ? ((exitPrice / entryPrice - 1) * 100) : 0;
          const pnl = row.strategy_nav - entryRow.strategy_nav;
          cumulative += pnl;
          const vixVals = [];
          const rsiVals = [];
          for (let j = entryIdx; j <= i; j++) {{
            const f = rows[j]._features || {{}};
            if (f.vix != null) vixVals.push(f.vix);
            if (f.rsi != null) rsiVals.push(f.rsi);
          }}
          trades.push({{
            entry_date: entryRow.datetime,
            exit_date: row.datetime,
            days: days,
            entry_price: entryPrice,
            exit_price: exitPrice,
            return_pct: retPct,
            pnl: pnl,
            cumulative: cumulative,
            entry_nav: entryRow.strategy_nav,
            exit_nav: row.strategy_nav,
            avg_vix: vixVals.length ? (vixVals.reduce((a,b)=>a+b,0) / vixVals.length) : null,
            avg_rsi: rsiVals.length ? (rsiVals.reduce((a,b)=>a+b,0) / rsiVals.length) : null,
          }});
          entryIdx = -1;
        }}
      }}
      return trades;
    }}

    function computeKellyStatsFromTrades(trades) {{
      if (!trades || !trades.length) return {{}};
      const wins = trades.filter(t => t.return_pct > 0).map(t => t.return_pct);
      const losses = trades.filter(t => t.return_pct <= 0).map(t => t.return_pct);
      const winRate = trades.length ? (wins.length / trades.length) : 0;
      const avgWin = wins.length ? (wins.reduce((a,b)=>a+b,0) / wins.length) : 0;
      const avgLoss = losses.length ? (losses.reduce((a,b)=>a+b,0) / losses.length) : 0;
      const r = avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : Infinity;
      const kelly = r !== Infinity && r > 0 ? (winRate - (1 - winRate) / r) : winRate;
      const expectancy = (winRate * avgWin) - ((1 - winRate) * Math.abs(avgLoss));
      const grossProfit = wins.reduce((a,b)=>a+b,0);
      const grossLoss = Math.abs(losses.reduce((a,b)=>a+b,0));
      const profitFactor = grossLoss > 0 ? (grossProfit / grossLoss) : Infinity;
      let maxWinStreak = 0, maxLossStreak = 0, curWin = 0, curLoss = 0;
      for (const t of trades) {{
        if (t.return_pct > 0) {{ curWin++; curLoss = 0; maxWinStreak = Math.max(maxWinStreak, curWin); }}
        else {{ curLoss++; curWin = 0; maxLossStreak = Math.max(maxLossStreak, curLoss); }}
      }}
      return {{
        total: trades.length,
        win_rate: winRate,
        avg_win: avgWin,
        avg_loss: avgLoss,
        kelly: kelly,
        expectancy: expectancy,
        profit_factor: profitFactor,
        gross_profit: grossProfit,
        gross_loss: grossLoss,
        max_win_streak: maxWinStreak,
        max_loss_streak: maxLossStreak,
      }};
    }}

    function sliceAndRecompute(rows, start, end) {{
      if (!rows || !rows.length) return {{ rows: [], summary: {{}}, trades: [], kelly: {{}}, regime_shapes: [] }};
      const sliced = rows.filter(r => String(r.datetime) >= start && String(r.datetime) <= end);
      if (!sliced.length) return {{ rows: [], summary: {{}}, trades: [], kelly: {{}}, regime_shapes: [] }};
      const firstRow = sliced[0];
      const lastRow = sliced[sliced.length - 1];
      const initialNav = 1000000.0;
      const stratNav0 = Number(firstRow.strategy_nav) || 1;
      const bhNav0 = Number(firstRow.buyhold_nav) || 1;
      const stratScale = stratNav0 > 0 ? (initialNav / stratNav0) : 1;
      const bhScale = bhNav0 > 0 ? (initialNav / bhNav0) : 1;

      const newRows = [];
      let peakS = -Infinity;
      let maxDdStrategy = 0.0;
      let buyCount = 0, sellCount = 0;
      let maxPos = Number(firstRow.position) || 0;
      let minPos = Number(firstRow.position) || 0;

      for (const r of sliced) {{
        const rawNav = Number(r.strategy_nav) || 0;
        const rawBh = Number(r.buyhold_nav) || 0;
        const nav = rawNav * stratScale;
        const bhNav = rawBh * bhScale;
        const excess = nav - bhNav;
        const excessPct = bhNav !== 0 ? (excess / bhNav * 100) : 0.0;
        peakS = Math.max(peakS, nav);
        maxDdStrategy = Math.max(maxDdStrategy, (peakS - nav) / peakS);
        if (r.action === 'BUY') buyCount++;
        else if (r.action === 'SELL') sellCount++;
        const pos = Number(r.position) || 0;
        maxPos = Math.max(maxPos, pos);
        minPos = Math.min(minPos, pos);
        newRows.push({{
          ...r,
          strategy_nav: nav,
          buyhold_nav: bhNav,
          excess: excess,
          excess_pct: excessPct,
        }});
      }}

      const strategyReturn = stratNav0 > 0 ? ((Number(lastRow.strategy_nav) || 0) / stratNav0 - 1) : 0;
      const buyholdReturn = bhNav0 > 0 ? ((Number(lastRow.buyhold_nav) || 0) / bhNav0 - 1) : 0;
      const excessReturn = strategyReturn - buyholdReturn;

      const trades = computeTradesFromRows(newRows);
      const kelly = computeKellyStatsFromTrades(trades);

      const summary = {{
        etf: state.etf,
        strategy_return: strategyReturn,
        buyhold_return: buyholdReturn,
        excess_return: excessReturn,
        max_dd_strategy: maxDdStrategy,
        max_dd_buyhold: 0,
        trades: buyCount + sellCount,
        buy_count: buyCount,
        sell_count: sellCount,
        max_position: maxPos,
        min_position: minPos,
      }};

      return {{
        rows: newRows,
        summary: summary,
        trades: trades,
        kelly: kelly,
        regime_shapes: [],
        csv_path: null,
      }};
    }}

    function recomputeSegment() {{
      try {{
        const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
        const start = state.zoomStart || bounds.start;
        const end = state.zoomEnd || bounds.end;
        if (!start || !end || start >= end) {{
          chartMeta.innerHTML = '<span style="color:var(--bad);">✗ 请选择一个有效的日期区间</span>';
          return;
        }}
        const sourceRows = (() => {{
          if (state.version && DATA.versioned && DATA.versioned[state.version]) {{
            const vd = (DATA.versioned[state.version].detailed || {{}})[state.etf];
            if (vd) return hydrateVersionDetail(state.etf, state.version, vd).rows;
          }}
          return (DATA.detailed[state.etf] || {{}}).rows || [];
        }})();
        if (!sourceRows || !sourceRows.length) {{
          chartMeta.innerHTML = '<span style="color:var(--bad);">✗ 当前版本没有可用数据</span>';
          return;
        }}
        const key = getSegmentKey(state.etf, state.version, start, end);
        if (!state.segmentCache[key]) {{
          state.segmentCache[key] = sliceAndRecompute(sourceRows, start, end);
        }}
        state.segmentMode = true;
        state.segmentKey = key;
        saveState();
        render();
        chartMeta.innerHTML = `<span style="color:var(--good);">✓ Segment recomputed: ${{start}} → ${{end}}</span>`;
      }} catch (err) {{
        chartMeta.innerHTML = `<span style="color:var(--bad);">✗ Recompute error: ${{err.message}}</span>`;
        console.error('recomputeSegment error:', err);
      }}
    }}

    function clearSegmentMode() {{
      state.segmentMode = false;
      state.segmentKey = '';
      saveState();
    }}

    function getWindows(etf) {{
      if (state.version && DATA.versioned && DATA.versioned[state.version]) {{
        const versionWindows = (DATA.versioned[state.version].windows || {{}})[etf];
        if (versionWindows) {{
          return versionWindows;
        }}
      }}
      return DATA.windows[etf] || {{ start: '', end: '', rows: 0 }};
    }}
    function getRangeRows(rows) {{
      if (state.range === 'all') return rows;
      if (state.range === '2020') return rows.filter(r => String(r.datetime).startsWith('2020-'));
      if (state.range === '2022') return rows.filter(r => String(r.datetime).startsWith('2022-'));
      if (state.range === '2023_2024') return rows.filter(r => String(r.datetime).startsWith('2023-') || String(r.datetime).startsWith('2024-'));
      return rows;
    }}
    function getDateBounds(rows) {{
      if (!rows.length) return {{ start: '', end: '' }};
      return {{ start: rows[0].datetime, end: rows[rows.length - 1].datetime }};
    }}
    function getVersionWindow(version, etf) {{
      if (!version) return null;
      const payload = DATA.versioned && DATA.versioned[version];
      const payloadWindows = payload && payload.windows;
      if (payloadWindows) {{
        const direct = payloadWindows[etf] || payloadWindows.SOXL || Object.values(payloadWindows)[0];
        if (direct && direct.start && direct.end) return direct;
      }}
      const summary = indexVersionSummaries[version] || {{}};
      const summaryWindows = summary.windows || {{}};
      const indexed = summaryWindows[etf] || summaryWindows.SOXL || Object.values(summaryWindows)[0];
      if (indexed && indexed.start && indexed.end) return indexed;
      const detail = payload && payload.detailed && (payload.detailed[etf] || payload.detailed.SOXL);
      const rows = detail && detail.rows;
      if (rows && rows.length) return {{ start: rows[0].datetime, end: rows[rows.length - 1].datetime, rows: rows.length }};
      return null;
    }}
    function getBacktestBoundsForProfile(profile) {{
      const window = getVersionWindow(profile, state.etf);
      if (window) return {{ start: window.start, end: window.end }};
      const profileRecord = modelReplayProfiles.find(item => item && item.profile === profile);
      if (profileRecord && profileRecord.before_end) {{
        return {{ start: '', end: profileRecord.after_end || profileRecord.before_end }};
      }}
      return null;
    }}
    function getBacktestBounds() {{
      const selectedProfile = runProfileSelect ? runProfileSelect.value : '';
      const selectedBounds = getBacktestBoundsForProfile(selectedProfile);
      if (selectedBounds && selectedBounds.start && selectedBounds.end) return selectedBounds;
      const windows = [];
      Object.keys(DATA.versioned || {{}}).forEach(version => {{
        if (version === 'paper_trading' || String(version).startsWith('rerun:')) return;
        const window = getVersionWindow(version, state.etf);
        if (window && window.start && window.end) windows.push(window);
      }});
      Object.keys(indexVersionSummaries || {{}}).forEach(version => {{
        if (version === 'paper_trading' || String(version).startsWith('rerun:')) return;
        const window = getVersionWindow(version, state.etf);
        if (window && window.start && window.end) windows.push(window);
      }});
      if (windows.length) {{
        return {{
          start: windows.map(w => w.start).sort()[0],
          end: windows.map(w => w.end).sort().slice(-1)[0],
        }};
      }}
      return getDateBounds(getRangeRows(getDetailed(state.etf).rows));
    }}
    function syncBacktestInputs(reset = false) {{
      if (!runStartDateInput || !runEndDateInput) return;
      const bounds = getBacktestBounds();
      runStartDateInput.min = bounds.start || '';
      runStartDateInput.max = bounds.end || '';
      runEndDateInput.min = bounds.start || '';
      runEndDateInput.max = bounds.end || '';
      const startOutOfBounds = runStartDateInput.value && bounds.start && runStartDateInput.value < bounds.start;
      const endOutOfBounds = runEndDateInput.value && bounds.end && runEndDateInput.value > bounds.end;
      if (reset || !runStartDateInput.value || startOutOfBounds) runStartDateInput.value = bounds.start || '';
      if (reset || !runEndDateInput.value || endOutOfBounds) runEndDateInput.value = bounds.end || '';
    }}
    function getFilteredRows(rows) {{
      let filtered = getRangeRows(rows);
      if (state.zoomStart) {{
        filtered = filtered.filter(r => String(r.datetime) >= state.zoomStart);
      }}
      if (state.zoomEnd) {{
        filtered = filtered.filter(r => String(r.datetime) <= state.zoomEnd);
      }}
      return filtered;
    }}
    function shiftIsoDate(iso, days) {{
      const dt = new Date(`${{iso}}T00:00:00Z`);
      dt.setUTCDate(dt.getUTCDate() + days);
      return dt.toISOString().slice(0, 10);
    }}
    function syncZoomInputs() {{
      const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
      startDateInput.min = bounds.start;
      startDateInput.max = bounds.end;
      endDateInput.min = bounds.start;
      endDateInput.max = bounds.end;
      startDateInput.value = state.zoomStart || bounds.start;
      endDateInput.value = state.zoomEnd || bounds.end;
      syncBacktestInputs();
    }}
    function applyZoomPreset(mode) {{
      const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
      if (!bounds.start) return;
      if (mode === 'all') {{
        state.zoomMode = 'all';
        state.zoomStart = null;
        state.zoomEnd = null;
        clearSegmentMode();
      }} else {{
        const monthsMap = {{ '6m': -180, '1y': -365, '2y': -730, '3y': -1095 }};
        const shifted = shiftIsoDate(bounds.end, monthsMap[mode]);
        state.zoomMode = mode;
        state.zoomStart = shifted < bounds.start ? bounds.start : shifted;
        state.zoomEnd = bounds.end;
      }}
      saveState();
      syncZoomInputs();
      renderChart();
    }}
    function applyCustomZoom() {{
      const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
      let start = startDateInput.value || bounds.start;
      let end = endDateInput.value || bounds.end;
      if (start > end) {{ [start, end] = [end, start]; }}
      state.zoomMode = 'custom';
      state.zoomStart = start;
      state.zoomEnd = end;
      saveState();
      syncZoomInputs();
      renderChart();
    }}

    function runProfileLabel(profile) {{
      const window = getVersionWindow(profile, state.etf);
      if (window) return `${{profile}} · ${{window.start}} → ${{window.end}}`;
      const record = modelReplayProfiles.find(item => item && item.profile === profile);
      if (record && (record.after_end || record.before_end)) {{
        return `${{profile}} · end ${{record.after_end || record.before_end}}`;
      }}
      return profile;
    }}
    function runnableProfilesFromVersions(versions) {{
      const names = [];
      (versions || []).forEach(version => {{
        if (!version || version === 'paper_trading' || String(version).startsWith('rerun:')) return;
        if (getVersionWindow(version, state.etf)) names.push(version);
      }});
      return names;
    }}
    function populateRunProfileSelect(profileSource, resetWindow = false) {{
      if (!runProfileSelect) return;
      const previous = runProfileSelect.value;
      const names = [];
      const seen = new Set();
      (profileSource || []).forEach(item => {{
        const name = typeof item === 'string' ? item : (item && item.profile);
        const result = typeof item === 'string' ? 'current' : (item && item.result);
        if (!name || seen.has(name) || name === 'paper_trading' || String(name).startsWith('rerun:')) return;
        if (result === 'failed' || result === 'skipped') return;
        seen.add(name);
        names.push(name);
      }});
      const fallback = runnableProfilesFromVersions(VERSIONS);
      fallback.forEach(name => {{
        if (!seen.has(name)) {{
          seen.add(name);
          names.push(name);
        }}
      }});
      runProfileSelect.innerHTML = '<option value="">Select profile...</option>';
      names.forEach(name => {{
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = runProfileLabel(name);
        runProfileSelect.appendChild(opt);
      }});
      if (previous && seen.has(previous)) runProfileSelect.value = previous;
      syncBacktestInputs(resetWindow || !runStartDateInput.value || !runEndDateInput.value);
    }}
    function upsertVersionOption(version, label) {{
      let opt = Array.from(versionSelect.options).find(item => item.value === version);
      if (!opt) {{
        opt = document.createElement('option');
        opt.value = version;
        versionSelect.appendChild(opt);
      }}
      opt.textContent = label || version;
    }}
    function registerBacktestResult(data, profile, start, end) {{
      const rows = Array.isArray(data.rows) ? data.rows : [];
      const trades = Array.isArray(data.trades) ? data.trades : computeTradesFromRows(rows);
      const strategyReturn = rows.length > 1 && rows[0].strategy_nav
        ? (rows[rows.length - 1].strategy_nav / rows[0].strategy_nav - 1)
        : 0;
      const buyholdReturn = rows.length > 1 && rows[0].buyhold_nav
        ? (rows[rows.length - 1].buyhold_nav / rows[0].buyhold_nav - 1)
        : 0;
      const summary = {{
        etf: state.etf,
        strategy_return: strategyReturn,
        buyhold_return: buyholdReturn,
        excess_return: strategyReturn - buyholdReturn,
        max_dd_strategy: Math.abs(Number(data.summary && data.summary.max_dd) || 0) / 100,
        max_dd_buyhold: 0,
        trades: Number(data.summary && data.summary.trades) || trades.length,
        buy_count: rows.filter(r => r.action === 'BUY').length,
        sell_count: rows.filter(r => r.action === 'SELL').length,
        max_position: rows.length ? Math.max(...rows.map(r => Number(r.position) || 0)) : 0,
        min_position: rows.length ? Math.min(...rows.map(r => Number(r.position) || 0)) : 0,
      }};
      const resultVersion = `rerun:${{profile}}:${{start}}:${{end}}`;
      if (!DATA.versioned) DATA.versioned = {{}};
      DATA.versioned[resultVersion] = {{
        detailed: {{
          [state.etf]: {{
            summary,
            rows,
            trades,
            csv_path: null,
            kelly: computeKellyStatsFromTrades(trades),
            regime_shapes: [],
          }},
        }},
        windows: data.windows || {{
          [state.etf]: {{ start, end, rows: rows.length }},
        }},
        _meta: {{
          source_kind: 'interactive_backtest',
          profile,
          start_date: start,
          end_date: end,
          updated_at: Date.now() / 1000,
        }},
      }};
      if (!DATA.versioned[resultVersion].windows[state.etf]) {{
        DATA.versioned[resultVersion].windows[state.etf] = {{ start, end, rows: rows.length }};
      }}
      if (!VERSIONS.includes(resultVersion)) VERSIONS.unshift(resultVersion);
      indexVersionSummaries[resultVersion] = {{
        etfs: [state.etf],
        windows: DATA.versioned[resultVersion].windows,
        meta: DATA.versioned[resultVersion]._meta,
      }};
      upsertVersionOption(resultVersion, `${{resultVersion}} · rows ${{rows.length}} · interactive`);
      state.version = resultVersion;
      state.versionSource = 'manual';
      state.range = 'all';
      state.zoomMode = 'all';
      state.zoomStart = null;
      state.zoomEnd = null;
      clearSegmentMode();
      saveState();
      return resultVersion;
    }}

    function initControls() {{
      ETFS.forEach(etf => {{
        const opt = document.createElement('option');
        opt.value = etf;
        opt.textContent = etf;
        etfSelect.appendChild(opt);
      }});
      // version selector
      const defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.textContent = 'Current (default)';
      versionSelect.appendChild(defaultOpt);
      VERSIONS.forEach(v => {{
        const versionPayload = DATA.versioned[v] || {{}};
        const etfCount = Object.keys(versionPayload.detailed || {{}}).length;
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = etfCount <= 1 ? `${{v}} (SOXL)` : v;
        versionSelect.appendChild(opt);
      }});
      etfSelect.addEventListener('change', e => {{
        state.etf = e.target.value;
        state.zoomMode = 'all';
        state.zoomStart = null;
        state.zoomEnd = null;
        clearSegmentMode();
        saveState();
        render();
      }});
      versionSelect.addEventListener('change', async e => {{
        state.version = e.target.value;
        state.versionSource = state.version ? 'manual' : 'current';
        state.zoomMode = 'all';
        state.zoomStart = null;
        state.zoomEnd = null;
        clearSegmentMode();
        saveState();
        if (state.version && !DATA.versioned[state.version]) {{
          await loadVersionData(state.version, state.etf);
        }}
        render();
      }});
      document.querySelectorAll('.pill').forEach(btn => {{
        btn.addEventListener('click', () => {{
          document.querySelectorAll('.pill').forEach(x => x.classList.remove('active'));
          btn.classList.add('active');
          state.range = btn.dataset.range;
          state.zoomMode = 'all';
          state.zoomStart = null;
          state.zoomEnd = null;
          clearSegmentMode();
          saveState();
          syncZoomInputs();
          renderChart();
        }});
      }});
      document.querySelectorAll('[data-zoom]').forEach(btn => {{
        btn.addEventListener('click', () => applyZoomPreset(btn.dataset.zoom));
      }});
      applyZoomBtn.addEventListener('click', applyCustomZoom);
      resetZoomBtn.addEventListener('click', () => applyZoomPreset('all'));

      populateRunProfileSelect(VERSIONS, true);
      runProfileSelect.addEventListener('change', () => syncBacktestInputs(true));

      // Slice & Recompute button
      document.getElementById('sliceRecomputeBtn').addEventListener('click', recomputeSegment);

      // Clear segment mode button
      document.getElementById('clearSegmentBtn').addEventListener('click', () => {{
        clearSegmentMode();
        render();
      }});

      // Run backtest button
      document.getElementById('runBacktestBtn').addEventListener('click', runSegmentBacktest);

      document.addEventListener('keydown', e => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === '+' || e.key === '=') {{
          e.preventDefault();
          const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
          const s = state.zoomStart || bounds.start;
          const e_ = state.zoomEnd || bounds.end;
          const mid = new Date(`${{s}}T00:00:00Z`).getTime() + (new Date(`${{e_}}T00:00:00Z`).getTime() - new Date(`${{s}}T00:00:00Z`).getTime()) / 2;
          const half = (new Date(`${{e_}}T00:00:00Z`).getTime() - new Date(`${{s}}T00:00:00Z`).getTime()) / 4;
          const ns = new Date(mid - half).toISOString().slice(0,10);
          const ne = new Date(mid + half).toISOString().slice(0,10);
          state.zoomStart = ns < bounds.start ? bounds.start : ns;
          state.zoomEnd = ne > bounds.end ? bounds.end : ne;
          state.zoomMode = 'custom';
          saveState();
          syncZoomInputs();
          renderChart();
          return;
        }}
        if (e.key === '-' || e.key === '_') {{
          e.preventDefault();
          const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
          const s = state.zoomStart || bounds.start;
          const e_ = state.zoomEnd || bounds.end;
          const mid = new Date(`${{s}}T00:00:00Z`).getTime() + (new Date(`${{e_}}T00:00:00Z`).getTime() - new Date(`${{s}}T00:00:00Z`).getTime()) / 2;
          const double = new Date(`${{e_}}T00:00:00Z`).getTime() - new Date(`${{s}}T00:00:00Z`).getTime();
          const ns = new Date(mid - double).toISOString().slice(0,10);
          const ne = new Date(mid + double).toISOString().slice(0,10);
          state.zoomStart = ns < bounds.start ? bounds.start : ns;
          state.zoomEnd = ne > bounds.end ? bounds.end : ne;
          state.zoomMode = 'custom';
          saveState();
          syncZoomInputs();
          renderChart();
          return;
        }}
        if (e.key === 'r' || e.key === 'R') {{
          e.preventDefault();
          applyZoomPreset('all');
          return;
        }}
        if (e.key === 'ArrowLeft') {{
          e.preventDefault();
          const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
          const s = state.zoomStart || bounds.start;
          const e_ = state.zoomEnd || bounds.end;
          const span = Math.max(1, Math.floor((new Date(`${{e_}}T00:00:00Z`).getTime() - new Date(`${{s}}T00:00:00Z`).getTime()) / (86400000 * 10)));
          const ns = shiftIsoDate(s, -span);
          const ne = shiftIsoDate(e_, -span);
          state.zoomStart = ns < bounds.start ? bounds.start : ns;
          state.zoomEnd = ne < bounds.start ? bounds.end : ne;
          if (state.zoomStart === bounds.start && state.zoomEnd === bounds.end) {{ state.zoomMode = 'all'; state.zoomStart = null; state.zoomEnd = null; }}
          else state.zoomMode = 'custom';
          saveState();
          syncZoomInputs();
          renderChart();
          return;
        }}
        if (e.key === 'ArrowRight') {{
          e.preventDefault();
          const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
          const s = state.zoomStart || bounds.start;
          const e_ = state.zoomEnd || bounds.end;
          const span = Math.max(1, Math.floor((new Date(`${{e_}}T00:00:00Z`).getTime() - new Date(`${{s}}T00:00:00Z`).getTime()) / (86400000 * 10)));
          const ns = shiftIsoDate(s, span);
          const ne = shiftIsoDate(e_, span);
          state.zoomStart = ns > bounds.end ? bounds.start : ns;
          state.zoomEnd = ne > bounds.end ? bounds.end : ne;
          if (state.zoomStart === bounds.start && state.zoomEnd === bounds.end) {{ state.zoomMode = 'all'; state.zoomStart = null; state.zoomEnd = null; }}
          else state.zoomMode = 'custom';
          saveState();
          syncZoomInputs();
          renderChart();
          return;
        }}
      }});
    }}

    async function runSegmentBacktest() {{
      const profile = runProfileSelect.value;
      if (!profile) {{
        alert('Please select a profile first');
        return;
      }}
      let start = runStartDateInput.value;
      let end = runEndDateInput.value;
      if (!start || !end) {{
        alert('Please set start and end dates');
        return;
      }}
      if (start > end) {{ [start, end] = [end, start]; }}

      showLoading(`Running backtest: ${{profile}} [${{start}} → ${{end}}]...`);
      try {{
        const resp = await fetch('/api/run-backtest', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ profile, start_date: start, end_date: end, tag: '' }}),
        }});
        if (!resp.ok) {{
          const err = await resp.json().catch(() => ({{ error: `HTTP ${{resp.status}}` }}));
          throw new Error(err.error || `HTTP ${{resp.status}}`);
        }}
        const data = await resp.json();
        if (data.error) {{
          throw new Error(data.error);
        }}

        const resultVersion = registerBacktestResult(data, profile, start, end);

        hideLoading();
        render();
        chartMeta.innerHTML = `<span style="color:var(--good);">✓ Backtest complete: ${{resultVersion}} | Return ${{data.summary.return}}% | Sharpe ${{data.summary.sharpe}} | Trades ${{data.summary.trades}}</span>`;
      }} catch (err) {{
        hideLoading();
        const serverHint = window.location.protocol === 'file:'
          ? '请用 qlib_venv/bin/python scripts/dashboard_server.py 启动后访问 http://127.0.0.1:8765/dashboard.html'
          : '请确认当前页面由 scripts/dashboard_server.py 提供，而不是普通静态 http.server。';
        chartMeta.innerHTML = `<span style="color:var(--bad);">✗ ${{err.message}} · ${{serverHint}}</span>`;
        console.error(err);
      }}
    }}

    const DATA_BASE = 'dashboard_data';
    let _indexLoaded = false;

    function showLoading(msg) {{
      chartMeta.innerHTML = `<span style="color:var(--accent);">⏳ ${{msg}}</span>`;
    }}
    function hideLoading() {{
      chartMeta.innerHTML = '';
    }}

    async function loadJson(path) {{
      try {{
        const sep = path.includes('?') ? '&' : '?';
        const cacheKey = encodeURIComponent(DATA.generated_at || Date.now());
        const res = await fetch(`${{path}}${{sep}}v=${{cacheKey}}`, {{ cache: 'no-store' }});
        if (!res.ok) throw new Error(`${{res.status}} ${{res.statusText}}`);
        return await res.json();
      }} catch (err) {{
        console.warn('Failed to load', path, err);
        return null;
      }}
    }}

    function productionStatusTone(status) {{
      const text = String(status || '').toLowerCase();
      if (['deployed', 'ready_for_deploy', 'linked', 'env_configured', 'token', 'ok'].includes(text)) return 'good';
      if (text.includes('blocked') || text.includes('missing') || text.includes('failed') || text.includes('unavailable')) return 'bad';
      if (text.includes('pending') || text.includes('waiting')) return 'warn';
      return '';
    }}

    function setProductionText(el, value) {{
      const text = value == null || value === '' ? 'n/a' : String(value);
      el.textContent = text;
      el.classList.remove('production-tone-good', 'production-tone-bad', 'production-tone-warn');
      const tone = productionStatusTone(text);
      if (tone) el.classList.add(`production-tone-${{tone}}`);
    }}

    function fmtPctValue(value) {{
      const n = Number(value);
      if (!Number.isFinite(n)) return 'n/a';
      return `${{(n * 100).toFixed(1)}}%`;
    }}

    function renderPaperStatus(paperPayload) {{
      const rows = paperPayload && Array.isArray(paperPayload.rows) ? paperPayload.rows : [];
      const latest = rows.length ? rows[rows.length - 1] : {{}};
      const meta = (paperPayload && paperPayload.meta) || {{}};
      const review = meta.latest_public_review || {{}};
      const signalBits = [];
      if (meta.latest_signal_slow_profile) signalBits.push(`slow=${{meta.latest_signal_slow_profile}}`);
      if (meta.latest_signal_fast_profile) signalBits.push(`fast=${{meta.latest_signal_fast_profile}}`);
      const signalProfile = meta.latest_signal_profile
        ? (signalBits.length ? `${{meta.latest_signal_profile}} (${{signalBits.join(', ')}})` : meta.latest_signal_profile)
        : null;
      const profile = review.profile || meta.latest_profile || latest.profile || signalProfile || 'unknown';
      const execution = review.entry_status || meta.latest_entry_status || meta.latest_execution_status || latest.execution_status || meta.latest_signal_execution_status || 'unknown';
      const dataDate = review.date || meta.latest_data_date || meta.latest_signal_date || latest.datetime || 'unknown';
      const target = review.entry_target_position != null
        ? review.entry_target_position
        : (meta.latest_entry_target_position != null
          ? meta.latest_entry_target_position
          : (latest.entry_target_position != null
            ? latest.entry_target_position
            : latest.target_position));
      const side = review.entry_side || meta.latest_entry_side || latest.entry_side || latest.action || 'entry';
      const source = review.summary_path || meta.latest_summary_path || latest.summary_path || meta.latest_signal_summary_path || paperPayload?.csv_path || 'paper_trading_soxl_daily.csv';
      setProductionText(productionPaperModel, profile);
      setProductionText(productionPaperExecution, execution);
      setProductionText(productionPaperSource, `${{dataDate}} · ${{side}} ${{fmtPctValue(target)}} · ${{source}}`);
    }}

    function renderModelReplayStatus(modelReplay) {{
      if (typeof productionModelReplayStatus === 'undefined' || !productionModelReplayStatus) return;
      const data = modelReplay || {{}};
      const counts = data.counts || {{}};
      const profiles = Array.isArray(data.profiles) ? data.profiles : [];
      const ready = Number(counts.current || 0) + Number(counts.refreshed || 0);
      const failed = Number(counts.failed || 0);
      const skipped = Number(counts.skipped || 0);
      const total = profiles.length;
      const safeEnd = data.safe_backtest_end || 'unknown';
      const latestDate = data.latest_data_date || 'unknown';
      const issueText = failed || skipped ? ` · failed ${{failed}} skipped ${{skipped}}` : '';
      const text = total ? `${{ready}}/${{total}} @ ${{safeEnd}} · data ${{latestDate}}${{issueText}}` : 'n/a';
      setProductionText(productionModelReplayStatus, text);
    }}

    function escapeHtml(value) {{
      const map = {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }};
      return String(value == null ? '' : value).replace(/[&<>"']/g, ch => map[ch]);
    }}

    function fmtBacktestMetric(value, suffix, digits = 2) {{
      const n = Number(value);
      if (!Number.isFinite(n)) return 'n/a';
      return `${{n.toFixed(digits)}}${{suffix}}`;
    }}

    function gateLabel(record) {{
      if (record.promotion_allowed) return 'promoted';
      if (record.full_gate === true) return 'gate pass';
      if (record.near_miss === true) return 'near miss';
      if (record.full_gate === false) return 'failed';
      return 'diagnostic';
    }}

    function gateClass(record) {{
      const label = gateLabel(record);
      if (label === 'promoted' || label === 'gate pass') return 'good';
      if (label === 'failed') return 'bad';
      return '';
    }}

    function renderResearchBacktests(records) {{
      if (!researchBacktestSummary || !researchBacktestTable) return;
      const rows = Array.isArray(records) ? records : [];
      if (!rows.length) {{
        researchBacktestSummary.innerHTML = `
          <div class="research-metric" style="grid-column:1/-1;">
            <div class="rk">Status</div>
            <div class="rv">No model-selection backtests exported</div>
          </div>
        `;
        researchBacktestTable.innerHTML = '<tr><th>No research backtests</th></tr>';
        return;
      }}
      const latest = rows[0] || {{}};
      const stressText = latest.stress_scenario_count != null
        ? `${{latest.stress_pass_count || 0}}/${{latest.stress_scenario_count}}`
        : 'n/a';
      researchBacktestSummary.innerHTML = [
        ['Latest', latest.experiment || 'n/a'],
        ['Decision', latest.decision || 'n/a'],
        ['Min Excess', fmtBacktestMetric(latest.min_relative_return, 'pp')],
        ['Max DD', fmtBacktestMetric(latest.max_drawdown, '%')],
      ].map(item => `
        <div class="research-metric">
          <div class="rk">${{escapeHtml(item[0])}}</div>
          <div class="rv">${{escapeHtml(item[1])}}</div>
        </div>
      `).join('');

      const header = `
        <tr>
          <th>Experiment</th>
          <th>Decision</th>
          <th>Gate</th>
          <th>Min Excess</th>
          <th>Max DD</th>
          <th>Stress</th>
          <th>Conclusion / Notebook</th>
        </tr>
      `;
      const body = rows.slice(0, 12).map(record => {{
        const gate = gateLabel(record);
        const gateTone = gateClass(record);
        const notebook = record.notebook_path
          ? `<a class="research-path" href="${{escapeHtml(record.notebook_path)}}" target="_blank" rel="noreferrer">${{escapeHtml(record.notebook_path)}}</a>`
          : '';
        const conclusion = record.conclusion || record.next_mechanism || '';
        const stress = record.stress_scenario_count != null
          ? `${{record.stress_pass_count || 0}}/${{record.stress_scenario_count}}`
          : 'n/a';
        return `
          <tr>
            <td><span class="mono">${{escapeHtml(record.experiment || '')}}</span><br><span class="sub">${{escapeHtml(record.updated_at || '')}}</span></td>
            <td>${{escapeHtml(record.decision || '')}}</td>
            <td class="${{gateTone}}">${{escapeHtml(gate)}}</td>
            <td>${{escapeHtml(fmtBacktestMetric(record.min_relative_return, 'pp'))}}</td>
            <td>${{escapeHtml(fmtBacktestMetric(record.max_drawdown, '%'))}}</td>
            <td>${{escapeHtml(stress)}}</td>
            <td>${{escapeHtml(conclusion)}}${{notebook ? `<br>${{notebook}}` : ''}}</td>
          </tr>
        `;
      }}).join('');
      researchBacktestTable.innerHTML = header + body;
    }}

    async function loadResearchBacktests() {{
      renderResearchBacktests(DATA.research_backtests || []);
      if (window.location.protocol === 'file:') return;
      const rows = await loadJson('dashboard_data/research_backtests.json');
      if (rows) renderResearchBacktests(rows);
    }}

    function embeddedPaperPayload() {{
      if (DATA.latest_version === 'paper_trading' && DATA.detailed && DATA.detailed[state.etf]) {{
        return DATA.detailed[state.etf];
      }}
      const fallback = DATA.detailed && DATA.detailed.SOXL;
      if (DATA.latest_version === 'paper_trading' && fallback) {{
        return fallback;
      }}
      const versioned = DATA.versioned && DATA.versioned.paper_trading;
      if (versioned && versioned.detailed && versioned.detailed.SOXL) {{
        return versioned.detailed.SOXL;
      }}
      return null;
    }}

    function paperPayloadFreshness(payload) {{
      const rows = payload && Array.isArray(payload.rows) ? payload.rows : [];
      const latest = rows.length ? rows[rows.length - 1] : {{}};
      const meta = (payload && payload.meta) || {{}};
      return String(meta.latest_data_date || meta.latest_signal_date || latest.datetime || '');
    }}

    function chooseFreshestPaperPayload(primary, fallback) {{
      if (!primary) return fallback || null;
      if (!fallback) return primary;
      return paperPayloadFreshness(fallback) > paperPayloadFreshness(primary) ? fallback : primary;
    }}

    function renderProductionStatus(payload, goalGates, paperPayload, modelReplay) {{
      if (!productionStatusCard) return;
      const data = payload || {{}};
      const staticExport = data.static_export || {{}};
      const paperRows = paperPayload && Array.isArray(paperPayload.rows) ? paperPayload.rows : [];
      const paperMeta = (paperPayload && paperPayload.meta) || {{}};
      const latestPaperRow = paperRows.length ? paperRows[paperRows.length - 1] : {{}};
      const paperEnd = paperMeta.latest_data_date || paperMeta.latest_signal_date || staticExport.paper_trading_end || data.paper_trading_end || latestPaperRow.datetime;
      const gates = goalGates || data.goal_gates || {{}};
      const overall = gates.overall || {{}};
      const issues = Array.isArray(overall.issues) ? overall.issues : [];
      productionStatusCard.hidden = false;
      setProductionText(productionStatusValue, data.status || 'unavailable');
      setProductionText(productionAuthStatus, data.auth_status || (data.auth && data.auth.status));
      setProductionText(productionProjectStatus, data.project_status || (data.project && data.project.status));
      setProductionText(productionPaperEnd, paperEnd);
      renderPaperStatus(paperPayload);
      setProductionText(productionGateOverall, overall.status || 'unknown');
      setProductionText(productionGateIssues, issues.length ? issues.join(', ') : (overall.detail || 'none'));
      renderModelReplayStatus(modelReplay);
      productionDeploymentStatus.classList.remove('production-tone-good', 'production-tone-bad', 'production-tone-warn');
      productionDeploymentStatus.textContent = '';
      if (data.deployment_url) {{
        const link = document.createElement('a');
        link.href = data.deployment_url;
        link.target = '_blank';
        link.rel = 'noreferrer';
        link.className = 'production-link';
        link.textContent = data.deployment_url.replace(/^https?:\\/\\//, '');
        productionDeploymentStatus.appendChild(link);
      }} else {{
        setProductionText(productionDeploymentStatus, data.deployment_url || 'none');
      }}
    }}

    async function loadProductionStatus() {{
      const [status, goalGates, paperPayload, modelReplay] = await Promise.all([
        loadJson('dashboard_data/production_status.json'),
        loadJson('dashboard_data/production_goal_gates.json'),
        loadJson('dashboard_data/versioned/paper_trading/SOXL.json'),
        loadJson('dashboard_data/model_replay_freshness.json'),
      ]);
      if (modelReplay && Array.isArray(modelReplay.profiles)) {{
        modelReplayProfiles = modelReplay.profiles;
        populateRunProfileSelect(modelReplayProfiles, true);
      }}
      renderProductionStatus(
        status || {{ status: 'unavailable' }},
        goalGates,
        chooseFreshestPaperPayload(paperPayload, embeddedPaperPayload()),
        modelReplay
      );
    }}

    function versionOptionLabel(version, summary) {{
      const windows = (summary && summary.windows) || {{}};
      const firstWindow = windows[state.etf] || windows.SOXL || Object.values(windows)[0] || {{}};
      const end = firstWindow.end || 'no-date';
      const rows = firstWindow.rows != null ? firstWindow.rows : 'n/a';
      const source = (summary && summary.meta && summary.meta.source_kind) || 'unknown';
      return `${{version}} · end ${{end}} · rows ${{rows}} · ${{source}}`;
    }}

    async function loadIndex() {{
      if (_indexLoaded) return;
      const idx = await loadJson(`${{DATA_BASE}}/index.json`);
      if (!idx) return;
      _indexLoaded = true;
      indexVersionSummaries = idx.version_summaries || {{}};
      // Populate version selector from index
      versionSelect.innerHTML = '';
      const defaultOpt = document.createElement('option');
      defaultOpt.value = '';
      defaultOpt.textContent = 'Current (default)';
      versionSelect.appendChild(defaultOpt);
      (idx.versions || []).forEach(v => {{
        const vs = idx.version_summaries[v] || {{}};
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = versionOptionLabel(v, vs);
        versionSelect.appendChild(opt);
      }});
      // Restore selection if valid
      if (state.version && idx.versions && idx.versions.includes(state.version)) {{
        versionSelect.value = state.version;
        // Auto-load version data if not already cached
        if (!DATA.versioned[state.version]) {{
          loadVersionData(state.version, state.etf).then(() => render());
        }}
      }}
      populateRunProfileSelect(modelReplayProfiles.length ? modelReplayProfiles : (idx.versions || []), true);
    }}

    async function loadVersionData(version, etf) {{
      if (!version || DATA.versioned[version]) return;
      showLoading(`正在加载 ${{version}} / ${{etf}} ...`);
      const data = await loadJson(`${{DATA_BASE}}/versioned/${{version}}/${{etf}}.json`);
      hideLoading();
      if (!data) {{
        alert(`无法加载版本 ${{version}} 的数据。请确认使用了 HTTP server 运行。`);
        return;
      }}
      if (!DATA.versioned[version]) DATA.versioned[version] = {{}};
      if (!DATA.versioned[version].detailed) DATA.versioned[version].detailed = {{}};
      DATA.versioned[version].detailed[etf] = data;
      // Also need windows; reconstruct from data.rows if not present
      const rows = data.rows || [];
      DATA.versioned[version].windows = {{
        [etf]: {{
          start: rows[0] ? rows[0].datetime : '',
          end: rows[rows.length - 1] ? rows[rows.length - 1].datetime : '',
          rows: rows.length,
        }}
      }};
    }}

    function initAsyncData() {{
      loadProductionStatus();
      loadResearchBacktests();
      // Detect file:// protocol and show banner
      if (window.location.protocol === 'file:') {{
        const banner = document.createElement('div');
        banner.style.cssText = 'background:rgba(37,99,235,0.08);border:1px solid var(--accent);border-radius:8px;padding:12px 16px;margin-bottom:18px;font-size:13px;color:var(--accent);';
        banner.innerHTML = 'Warning: 当前通过 file:// 协议打开，版本切换功能不可用。请运行 <code>python -m http.server</code> 后通过 http://localhost:8000 访问以获得完整功能。';
        document.querySelector('.shell').insertBefore(banner, document.querySelector('.shell').firstChild);
        return; // Skip async loads on file protocol
      }}
      loadIndex();
    }}

    function renderMetrics() {{
      const detail = getDetailed(state.etf);
      const summary = detail.summary;
      let window = getWindows(state.etf);
      if (state.segmentMode && state.segmentKey && state.segmentCache[state.segmentKey]) {{
        const seg = state.segmentCache[state.segmentKey];
        if (seg.rows && seg.rows.length) {{
          window = {{ start: seg.rows[0].datetime, end: seg.rows[seg.rows.length-1].datetime, rows: seg.rows.length }};
        }}
      }}
      const rl = DATA.results[state.etf] || {{}};
      const rule = DATA.rule_based[state.etf] || {{}};
      const cards = [
        {{
          key: 'Strategy Return',
          value: fmtPct(summary.strategy_return),
          sub: `Buy&Hold ${{fmtPct(summary.buyhold_return)}} | Excess ${{fmtPct(summary.excess_return)}}`,
        }},
        {{
          key: 'QLib Proxy',
          value: `${{fmtNum(rl.sharpe, 3)}}`,
          sub: `Return ${{fmtNum(rl.return)}}% | MaxDD ${{fmtNum(rl.max_dd)}}%`,
        }},
        {{
          key: 'Trades / Window',
          value: String(summary.trades),
          sub: `BUY ${{summary.buy_count}} | SELL ${{summary.sell_count}} | ${{window.start}} → ${{window.end}}`,
        }},
        {{
          key: 'Rule-Based',
          value: `${{fmtNum(rule.sharpe, 3)}}`,
          sub: `Return ${{fmtNum(rule.return)}}% | MaxDD ${{fmtNum(rule.max_dd)}}%`,
        }},
      ];
      metricGrid.innerHTML = cards.map(card => `
        <div class="metric">
          <div class="k">${{card.key}}</div>
          <div class="v">${{card.value}}</div>
          <div class="s">${{card.sub}}</div>
        </div>
      `).join('');
    }}

    function renderTradeStats(k) {{
      if (!k || !k.total) {{
        tradeStatRow.innerHTML = '<div class="stat-cell"><div class="sk">无交易数据</div></div>';
        return;
      }}
      // For data sources without discrete trade signals (e.g. post_train_review daily CSVs
      // that only record position, not entry/exit actions), trade pairing may yield only
      // 1 trade. Show a warning instead of misleading Kelly stats.
      if (k.total <= 1) {{
        tradeStatRow.innerHTML = `
          <div class="stat-cell" style="grid-column:1/-1;text-align:left;">
            <div class="sk" style="color:var(--accent);">连续持仓策略</div>
            <div style="font-size:13px;color:var(--muted);margin-top:6px;">
              该模型版本的数据源只记录了每日仓位，没有离散买卖信号。<br>
              整个回测期间仅 ${'{k.total}'} 笔完整持仓交易，无法计算有意义的 Kelly / Expectancy。<br>
              切换到 <b>Current</b> 或其他有 detailed_trades.csv 的版本可查看逐笔交易质量。
            </div>
          </div>
        `;
        return;
      }}
      const kellyPct = k.kelly * 100;
      const kellyColor = kellyPct > 0 ? 'var(--good)' : kellyPct < 0 ? 'var(--bad)' : 'var(--muted)';
      tradeStatRow.innerHTML = [
        {{ k: '总交易', v: String(k.total), s: `${{fmtNum(k.win_rate*100,1)}}% 盈 | ${{fmtNum((1-k.win_rate)*100,1)}}% 亏` }},
        {{ k: '凯利比例', v: fmtNum(kellyPct, 1)+'%', s: `R=${{fmtNum(k.avg_win/Math.abs(k.avg_loss),2)}}`, color: kellyColor, highlight: true }},
        {{ k: '期望收益', v: fmtNum(k.expectancy, 2)+'%', s: '每笔期望', color: k.expectancy >= 0 ? 'var(--good)' : 'var(--bad)' }},
        {{ k: '收益因子', v: k.profit_factor > 900 ? '∞' : fmtNum(k.profit_factor, 2), s: '总盈/总亏' }},
      ].map((s, idx) => `
        <div class="stat-cell ${{s.highlight ? 'kelly-highlight' : ''}}">
          <div class="sk">${{s.k}}</div>
          <div class="sv" style="color:${{s.color || (String(s.v).includes('-') ? 'var(--bad)' : 'var(--good)')}}">${{s.v}}</div>
          <div class="ss">${{s.s}}</div>
        </div>
      `).join('');
    }}

    function renderTradeTable() {{
      const csvPath = getDetailed(state.etf).csv_path;
      if (csvPath) {{
        csvLink.style.display = '';
        csvLink.href = csvPath;
      }} else {{
        csvLink.style.display = 'none';
      }}
      const allRows = getDetailed(state.etf).rows;
      let trades = [...(getDetailed(state.etf).trades || [])];
      if (!trades.length) {{
        tradePerfTable.innerHTML = '<tr><th>无交易数据</th></tr>';
        return;
      }}

      trades.sort((a, b) => {{
        const av = a[state.sortCol];
        const bv = b[state.sortCol];
        if (av == null && bv == null) return 0;
        if (av == null) return state.sortAsc ? -1 : 1;
        if (bv == null) return state.sortAsc ? 1 : -1;
        return state.sortAsc ? (av > bv ? 1 : -1) : (av > bv ? -1 : 1);
      }});

      const header = ['#', '入场日期', '出场日期', '持仓', '入场价', '出场价', '收益率', 'PnL', '累计', 'Avg VIX', 'Avg RSI'];
      const colKeys = ['', 'entry_date', 'exit_date', 'days', 'entry_price', 'exit_price', 'return_pct', 'pnl', 'cumulative', 'avg_vix', 'avg_rsi'];

      let headerHtml = '<tr>' + header.map((h, idx) => {{
        const key = colKeys[idx];
        const isSort = key !== '';
        const cls = isSort && state.sortCol === key ? (state.sortAsc ? 'sort-asc' : 'sort-desc') : '';
        return `<th class="${{cls}}" data-col="${{key}}" ${{isSort ? '' : 'style="cursor:default"'}}>${{h}}</th>`;
      }}).join('') + '</tr>';

      const bodyHtml = trades.map((t, idx) => `
        <tr class="clickable" data-entry="${{t.entry_date}}" data-exit="${{t.exit_date}}">
          <td>${{idx + 1}}</td>
          <td>${{t.entry_date}}</td>
          <td>${{t.exit_date}}</td>
          <td>${{t.days}}天</td>
          <td>${{fmtNum(t.entry_price, 2)}}</td>
          <td>${{fmtNum(t.exit_price, 2)}}</td>
          <td class="${{t.return_pct > 0 ? 'good' : 'bad'}}">${{fmtNum(t.return_pct, 2)}}%</td>
          <td class="${{t.pnl > 0 ? 'good' : 'bad'}}">${{fmtMoney(t.pnl)}}</td>
          <td>${{fmtMoney(t.cumulative)}}</td>
          <td>${{t.avg_vix != null ? fmtNum(t.avg_vix, 2) : 'n/a'}}</td>
          <td>${{t.avg_rsi != null ? fmtNum(t.avg_rsi, 2) : 'n/a'}}</td>
        </tr>
      `).join('');

      tradePerfTable.innerHTML = headerHtml + bodyHtml;

      tradePerfTable.querySelectorAll('th[data-col]').forEach(th => {{
        th.addEventListener('click', () => {{
          const col = th.dataset.col;
          if (!col) return;
          if (state.sortCol === col) state.sortAsc = !state.sortAsc;
          else {{ state.sortCol = col; state.sortAsc = false; }}
          saveState();
          renderTradeTable();
        }});
      }});

      tradePerfTable.querySelectorAll('tr.clickable').forEach(tr => {{
        tr.addEventListener('click', () => {{
          const entry = tr.dataset.entry;
          const exit = tr.dataset.exit;
          if (!entry || !exit) return;
          const sIdx = allRows.findIndex(r => r.datetime === entry);
          const eIdx = allRows.findIndex(r => r.datetime === exit);
          if (sIdx < 0 || eIdx < 0) return;
          const start = Math.max(0, sIdx - 5);
          const end = Math.min(allRows.length - 1, eIdx + 5);
          state.zoomStart = allRows[start].datetime;
          state.zoomEnd = allRows[end].datetime;
          state.zoomMode = 'custom';
          saveState();
          syncZoomInputs();
          renderChart();
          document.getElementById('mainChart').scrollIntoView({{ behavior: 'smooth' }});
        }});
      }});
    }}

    function renderRegimeStrip(rows) {{
      if (!rows.length) {{ regimeStrip.innerHTML = ''; return; }}
      regimeStrip.innerHTML = rows.map(r => {{
        const f = r._features || {{}};
        const vz = f.vix_zscore;
        let color = 'rgba(215,222,230,0.6)';
        let title = `${{r.datetime}} 正常`;
        if (vz != null) {{
          if (vz > 2) {{ color = 'rgba(185,28,28,0.55)'; title = `${{r.datetime}} 高压力 (VIXz ${{fmtNum(vz,2)}})`; }}
          else if (vz < -1) {{ color = 'rgba(15,118,110,0.4)'; title = `${{r.datetime}} 低波动 (VIXz ${{fmtNum(vz,2)}})`; }}
        }}
        return `<div class="regime-bar" style="background:${{color}}" title="${{title}}"></div>`;
      }}).join('');
    }}

    function renderPositionDist(rows) {{
      const bins = {{ '0': 0, '低(0-0.3)': 0, '中(0.3-0.7)': 0, '高(0.7-1)': 0 }};
      rows.forEach(r => {{
        const p = r.position;
        if (p === 0) bins['0']++;
        else if (p < 0.3) bins['低(0-0.3)']++;
        else if (p < 0.7) bins['中(0.3-0.7)']++;
        else bins['高(0.7-1)']++;
      }});
      const total = rows.length || 1;
      const maxCount = Math.max(...Object.values(bins));
      const container = document.getElementById('posDist');
      container.innerHTML = Object.entries(bins).map(([label, count]) => {{
        const pct = (count / total * 100).toFixed(1);
        const width = maxCount > 0 ? (count / maxCount * 100).toFixed(1) : 0;
        return `
          <div class="pos-bar-wrap">
            <div class="pos-bar-label">${{label}}</div>
            <div class="pos-bar-track"><div class="pos-bar-fill" style="width:${{width}}%"></div></div>
            <div class="pos-bar-pct">${{pct}}%</div>
          </div>
        `;
      }}).join('');
    }}

    function renderTradeList(trades) {{
      const container = document.getElementById('tradeList');
      const recent = trades.slice(-5).reverse();
      container.innerHTML = recent.map(t => `
        <div class="trade-item">
          <span class="trade-entry">BUY</span> ${{t.entry_date}} → <span class="trade-exit">SELL</span> ${{t.exit_date}}<br>
          持有 ${{t.days}} 天 | 收益 <span class="${{t.return_pct >= 0 ? 'good' : 'bad'}}">${{fmtNum(t.return_pct, 2)}}%</span> | PnL ${{fmtMoney(t.pnl)}}
        </div>
      `).join('');
    }}

    function updateInfoPanel(row, tradeMap) {{
      if (!row) {{
        infoEmpty.style.display = 'block';
        infoContent.style.display = 'none';
        return;
      }}
      infoEmpty.style.display = 'none';
      infoContent.style.display = 'block';
      const f = row._features || {{}};
      document.getElementById('infoDate').textContent = row.datetime;
      const actionEl = document.getElementById('infoAction');
      if (actionEl) {{
        actionEl.textContent = row.action || 'HOLD';
        actionEl.style.color = row.action === 'BUY' ? 'var(--good)' : row.action === 'SELL' ? 'var(--bad)' : 'var(--ink)';
      }}
      const profileEl = document.getElementById('infoProfile');
      if (profileEl) profileEl.textContent = row.profile || 'n/a';
      const executionEl = document.getElementById('infoExecution');
      if (executionEl) executionEl.textContent = row.execution_status || 'n/a';
      const targetEl = document.getElementById('infoTarget');
      if (targetEl) targetEl.textContent = row.target_position != null ? fmtNum(row.target_position, 4) : 'n/a';
      document.getElementById('infoOpen').textContent = row.open != null ? '$' + fmtNum(row.open, 2) : 'n/a';
      document.getElementById('infoHigh').textContent = row.high != null ? '$' + fmtNum(row.high, 2) : 'n/a';
      document.getElementById('infoLow').textContent = row.low != null ? '$' + fmtNum(row.low, 2) : 'n/a';
      document.getElementById('infoClose').textContent = row.close != null ? '$' + fmtNum(row.close, 2) : 'n/a';
      document.getElementById('infoStrat').textContent = fmtNum(row.strategy_nav);
      document.getElementById('infoBH').textContent = fmtNum(row.buyhold_nav);
      document.getElementById('infoPos').textContent = fmtNum(row.position, 4);
      document.getElementById('infoExcess').textContent = fmtNum(row.excess_pct, 2) + '%';
      const stateEl = document.getElementById('infoState');
      if (row.market_state != null) {{
        stateEl.textContent = row.market_state ? 'Overvalued 🔴' : 'Undervalued 🟢';
        stateEl.style.color = row.market_state ? 'var(--bad)' : 'var(--good)';
      }} else {{ stateEl.textContent = 'n/a'; stateEl.style.color = 'var(--ink)'; }}
      document.getElementById('infoPanic').textContent = row.panic_score != null ? fmtNum(row.panic_score, 1) : 'n/a';
      document.getElementById('infoRsi6').textContent = row.rsi6 != null ? fmtNum(row.rsi6, 1) : 'n/a';
      document.getElementById('infoHv20').textContent = row.hv_20 != null ? fmtNum(row.hv_20, 3) : 'n/a';
      document.getElementById('infoVolCoef').textContent = row.vol_coef != null ? fmtNum(row.vol_coef, 2) : 'n/a';
      document.getElementById('infoVix').textContent = f.vix != null ? fmtNum(f.vix, 2) : 'n/a';
      document.getElementById('infoRsi').textContent = f.rsi != null ? fmtNum(f.rsi, 2) : 'n/a';
      document.getElementById('infoVol').textContent = f.volatility != null ? fmtNum(f.volatility, 4) : 'n/a';
      document.getElementById('infoSpread').textContent = f.spread != null ? fmtNum(f.spread, 4) : 'n/a';
      document.getElementById('infoVixZ').textContent = f.vix_zscore != null ? fmtNum(f.vix_zscore, 2) : 'n/a';
      document.getElementById('infoSpTr').textContent = f.spread_trend != null ? fmtNum(f.spread_trend, 4) : 'n/a';
      document.getElementById('infoMA').textContent = f.ma5_ma20 != null ? fmtNum(f.ma5_ma20, 4) : 'n/a';
      document.getElementById('infoVolStr').textContent = f.vol_strength != null ? fmtNum(f.vol_strength, 4) : 'n/a';

      const trade = tradeMap[row.datetime];
      const tradeSection = document.getElementById('tradeSection');
      if (trade) {{
        tradeSection.style.display = 'block';
        document.getElementById('tradeEntry').textContent = trade.entry_date;
        document.getElementById('tradeExit').textContent = trade.exit_date;
        document.getElementById('tradeDays').textContent = trade.days + ' 天';
        const retEl = document.getElementById('tradeRet');
        retEl.textContent = fmtNum(trade.return_pct, 2) + '%';
        retEl.style.color = trade.return_pct >= 0 ? 'var(--good)' : 'var(--bad)';
      }} else {{
        tradeSection.style.display = 'none';
      }}
    }}

    function renderChart() {{
      const allRows = getDetailed(state.etf).rows;
      const rows = getFilteredRows(allRows);
      const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUTS[state.view || 'all']));
      const shapes = getDetailed(state.etf).regime_shapes || [];
      // Overvalued / Undervalued background shapes
      const ovlShapes = [];
      let curState = null, stateStart = null;
      for (let i = 0; i < rows.length; i++) {{
        const ms = rows[i].market_state;
        if (ms !== curState) {{
          if (curState !== null && stateStart !== null) {{
            ovlShapes.push({{
              type: 'rect',
              x0: stateStart, x1: rows[i-1].datetime,
              y0: 0, y1: 1, yref: 'paper',
              fillcolor: curState ? 'rgba(185,28,28,0.05)' : 'rgba(15,118,110,0.05)',
              line: {{width: 0}}, layer: 'below',
            }});
          }}
          curState = ms;
          stateStart = rows[i].datetime;
        }}
      }}
      if (curState !== null && stateStart !== null) {{
        ovlShapes.push({{
          type: 'rect',
          x0: stateStart, x1: rows[rows.length-1].datetime,
          y0: 0, y1: 1, yref: 'paper',
          fillcolor: curState ? 'rgba(185,28,28,0.05)' : 'rgba(15,118,110,0.05)',
          line: {{width: 0}}, layer: 'below',
        }});
      }}
      layout.shapes = shapes.concat(ovlShapes);

      const dates = rows.map(r => r.datetime);
      const strategyNav = rows.map(r => r.strategy_nav);
      const buyholdNav = rows.map(r => r.buyhold_nav);
      const positions = rows.map(r => r.position);
      const excessPct = rows.map(r => r.excess_pct);

      let peakS = -Infinity, peakB = -Infinity;
      const strategyDD = [], buyholdDD = [];
      for (const r of rows) {{
        peakS = Math.max(peakS, r.strategy_nav);
        peakB = Math.max(peakB, r.buyhold_nav);
        strategyDD.push(r.strategy_nav / peakS - 1);
        buyholdDD.push(r.buyhold_nav / peakB - 1);
      }}

      const buyRows = rows.filter(r => r.action === 'BUY');
      const sellRows = rows.filter(r => r.action === 'SELL');
      const trades = getDetailed(state.etf).trades || [];
      const tradeMap = {{}};
      trades.forEach(t => {{ tradeMap[t.entry_date] = t; }});

      // Build date maps for BUY <-> SELL click pairing
      const buyToSell = {{}};
      const sellToBuy = {{}};
      trades.forEach(t => {{ buyToSell[t.entry_date] = t.exit_date; sellToBuy[t.exit_date] = t.entry_date; }});

      renderRegimeStrip(rows);
      renderPositionDist(rows);
      renderTradeList(trades);

      const traces = [];
      const xa = 'x', ya = 'y';
      const xaPrice = 'x2', yaPrice = 'y2';
      const xa2 = 'x3', ya2 = 'y3';
      const xa3 = 'x4', ya3 = 'y4';
      const xa4 = 'x5', ya4 = 'y5';

      traces.push({{
        x: dates, y: strategyNav,
        name: 'Strategy NAV', type: 'scatter', mode: 'lines',
        line: {{ color: '{C_STRATEGY}', width: 2 }},
        hovertemplate: '%{{x}}<br>Strategy: %{{y:,.2f}}<extra></extra>',
        xaxis: xa, yaxis: ya,
      }});
      traces.push({{
        x: dates, y: buyholdNav,
        name: 'Buy&Hold NAV', type: 'scatter', mode: 'lines',
        line: {{ color: '{C_BUYHOLD}', width: 2 }},
        hovertemplate: '%{{x}}<br>Buy&Hold: %{{y:,.2f}}<extra></extra>',
        xaxis: xa, yaxis: ya,
      }});
      // BUY/SELL markers on NAV line
      if (buyRows.length) {{
        traces.push({{
          x: buyRows.map(r => r.datetime),
          y: buyRows.map(r => r.strategy_nav),
          name: 'BUY (NAV)', type: 'scatter', mode: 'markers',
          marker: {{ symbol: 'triangle-up', size: 10, color: '#0f766e', line: {{ color: 'white', width: 1 }} }},
          hovertemplate: '<b>BUY</b> %{{x}}<br>NAV: %{{y:,.2f}}<extra></extra>',
          xaxis: xa, yaxis: ya,
          showlegend: false,
        }});
      }}
      if (sellRows.length) {{
        traces.push({{
          x: sellRows.map(r => r.datetime),
          y: sellRows.map(r => r.strategy_nav),
          name: 'SELL (NAV)', type: 'scatter', mode: 'markers',
          marker: {{ symbol: 'triangle-down', size: 10, color: '#b91c1c', line: {{ color: 'white', width: 1 }} }},
          hovertemplate: '<b>SELL</b> %{{x}}<br>NAV: %{{y:,.2f}}<extra></extra>',
          xaxis: xa, yaxis: ya,
          showlegend: false,
        }});
      }}
      // Price row: prefer true OHLC candles, fall back to the daily close/price series.
      const ohlcRows = rows.filter(r => r.open != null && r.high != null && r.low != null && r.close != null);
      const ma180Rows = rows.filter(r => r.ma180 != null);
      if (ohlcRows.length) {{
        traces.push({{
          x: ohlcRows.map(r => r.datetime),
          open: ohlcRows.map(r => r.open),
          high: ohlcRows.map(r => r.high),
          low: ohlcRows.map(r => r.low),
          close: ohlcRows.map(r => r.close),
          name: 'SOXL', type: 'candlestick',
          increasing: {{line: {{color: '#0f766e'}}, fillcolor: 'rgba(15,118,110,0.12)'}},
          decreasing: {{line: {{color: '#b91c1c'}}, fillcolor: 'rgba(185,28,28,0.12)'}},
          xaxis: xaPrice, yaxis: yaPrice,
        }});
      }} else {{
        const priceRows = rows.filter(r => r.close != null || r.price != null);
        if (priceRows.length) {{
          traces.push({{
            x: priceRows.map(r => r.datetime),
            y: priceRows.map(r => r.close != null ? r.close : r.price),
            name: 'SOXL Close', type: 'scatter', mode: 'lines',
            line: {{ color: '{C_BUYHOLD}', width: 1.8 }},
            hovertemplate: '%{{x}}<br>SOXL: $%{{y:,.2f}}<extra></extra>',
            xaxis: xaPrice, yaxis: yaPrice,
          }});
        }}
      }}
      if (ma180Rows.length) {{
        traces.push({{
          x: ma180Rows.map(r => r.datetime),
          y: ma180Rows.map(r => r.ma180),
          name: 'MA180', type: 'scatter', mode: 'lines',
          line: {{ color: '#2563eb', width: 1.5, dash: 'dash' }},
          hovertemplate: '%{{x}}<br>MA180: $%{{y:,.2f}}<extra></extra>',
          xaxis: xaPrice, yaxis: yaPrice,
        }});
      }}
      if (buyRows.length) {{
        traces.push({{
          x: buyRows.map(r => r.datetime),
          y: buyRows.map(r => r.close != null ? r.close : (r.price != null ? r.price : r.strategy_nav)),
          name: 'BUY', type: 'scatter', mode: 'markers',
          marker: {{ symbol: 'triangle-up', size: 12, color: '{C_BUY}', line: {{ color: 'white', width: 1.5 }} }},
          hovertemplate: '<b>BUY</b> %{{x}}<br>Price: $%{{y:.2f}}<br>NAV: %{{text}}<extra></extra>',
          text: buyRows.map(r => fmtNum(r.strategy_nav, 2)),
          customdata: buyRows.map(r => buyToSell[r.datetime] || ''),
          xaxis: xaPrice, yaxis: yaPrice,
        }});
      }}
      if (sellRows.length) {{
        traces.push({{
          x: sellRows.map(r => r.datetime),
          y: sellRows.map(r => r.close != null ? r.close : (r.price != null ? r.price : r.strategy_nav)),
          name: 'SELL', type: 'scatter', mode: 'markers',
          marker: {{ symbol: 'triangle-down', size: 12, color: '{C_SELL}', line: {{ color: 'white', width: 1.5 }} }},
          hovertemplate: '<b>SELL</b> %{{x}}<br>Price: $%{{y:.2f}}<br>NAV: %{{text}}<extra></extra>',
          text: sellRows.map(r => fmtNum(r.strategy_nav, 2)),
          customdata: sellRows.map(r => sellToBuy[r.datetime] || ''),
          xaxis: xaPrice, yaxis: yaPrice,
        }});
      }}

      traces.push({{
        x: dates, y: positions,
        name: 'Position %', type: 'bar',
        marker: {{ color: positions.map(p => p > 0 ? 'rgba(15,118,110,0.7)' : 'rgba(109,100,88,0.4)') }},
        hovertemplate: '%{{x}}<br>Position: %{{y:.1%}}<extra></extra>',
        xaxis: xa2, yaxis: ya2,
      }});

      traces.push({{
        x: dates, y: excessPct,
        name: 'Excess %', type: 'scatter', mode: 'lines',
        line: {{ color: '{C_INK}', width: 1.5 }},
        fill: 'tozeroy',
        fillcolor: 'rgba(27,26,23,0.08)',
        hovertemplate: '%{{x}}<br>Excess: %{{y:.2f}}%<extra></extra>',
        xaxis: xa3, yaxis: ya3,
        showlegend: false,
      }});

      traces.push({{
        x: dates, y: strategyDD,
        name: 'Strategy DD', type: 'scatter', mode: 'lines',
        line: {{ color: '{C_STRATEGY}', width: 1.5 }},
        fill: 'tozeroy',
        fillcolor: 'rgba(37,99,235,0.12)',
        hovertemplate: '%{{x}}<br>Strat DD: %{{y:.2%}}<extra></extra>',
        xaxis: xa4, yaxis: ya4,
        showlegend: false,
      }});
      traces.push({{
        x: dates, y: buyholdDD,
        name: 'Buy&Hold DD', type: 'scatter', mode: 'lines',
        line: {{ color: '{C_BUYHOLD}', width: 1.5 }},
        fill: 'tozeroy',
        fillcolor: 'rgba(15,118,110,0.10)',
        hovertemplate: '%{{x}}<br>BH DD: %{{y:.2%}}<extra></extra>',
        xaxis: xa4, yaxis: ya4,
        showlegend: false,
      }});

      Plotly.react('plotlyChart', traces, layout, {{
        responsive: true,
        scrollZoom: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
      }}).then(gd => {{
        gd.on('plotly_hover', function(data) {{
          const pt = data.points[0];
          let dateStr = pt.x;
          if (dateStr instanceof Date) {{
            dateStr = dateStr.toISOString().slice(0, 10);
          }}
          const row = rows.find(r => r.datetime === dateStr);
          updateInfoPanel(row, tradeMap);
        }});
        gd.on('plotly_click', function(data) {{
          const pt = data.points[0];
          const trace = pt.data;
          const idx = pt.pointNumber;
          const pairedDate = trace.customdata ? trace.customdata[idx] : null;
          if (pairedDate) {{
            const dateStr = pt.x;
            const start = dateStr < pairedDate ? dateStr : pairedDate;
            const end = dateStr > pairedDate ? dateStr : pairedDate;
            state.zoomStart = start;
            state.zoomEnd = end;
            state.zoomMode = 'custom';
            saveState();
            syncZoomInputs();
            renderChart();
          }}
        }});
        gd.on('plotly_doubleclick', function() {{
          state.zoomMode = 'all';
          state.zoomStart = null;
          state.zoomEnd = null;
          clearSegmentMode();
          saveState();
          syncZoomInputs();
          renderChart();
        }});
      }});

      if (rows.length) {{
        const versionPayload = state.version && DATA.versioned ? DATA.versioned[state.version] : null;
        const versionDetail = versionPayload ? ((versionPayload.detailed || {{}})[state.etf]) : null;
        const isInlineLatest = Boolean(state.version && state.version === DATA.latest_version && (DATA.detailed || {{}})[state.etf]);
        const isFallbackView = Boolean(versionPayload && !versionDetail && !isInlineLatest);
        chartMeta.innerHTML = `
          <span>Built: <span class="mono">${{DATA.generated_at}}</span></span>
          <span>Version: <span class="mono">${{state.version || 'current'}}</span></span>
          <span>Zoom: <span class="mono">${{state.zoomMode}}</span></span>
          <span>Window: <span class="mono">${{rows[0].datetime}}</span> → <span class="mono">${{rows[rows.length-1].datetime}}</span></span>
          <span>Rows: <span class="mono">${{rows.length}}</span></span>
          <span>BUY: <span class="mono">${{buyRows.length}}</span></span>
          <span>SELL: <span class="mono">${{sellRows.length}}</span></span>
          ${{isFallbackView ? '<span>Market Data: <span class="mono">current ETF fallback</span></span>' : ''}}
        `;
      }} else {{
        chartMeta.innerHTML = `<span>当前日期筛选下没有数据，请放宽缩放范围。</span>`;
      }}
    }}

    function render() {{
      try {{
        etfSelect.value = state.etf;
        versionSelect.value = state.version || '';
        document.querySelectorAll('.pill').forEach(p => {{
          p.classList.toggle('active', p.dataset.range === state.range);
        }});
        syncZoomInputs();
        // Segment banner
        const banner = document.getElementById('segmentBanner');
        const segmentRange = document.getElementById('segmentRange');
        if (state.segmentMode && state.segmentKey && state.segmentCache[state.segmentKey]) {{
          const seg = state.segmentCache[state.segmentKey];
          if (seg.rows && seg.rows.length) {{
            banner.style.display = 'flex';
            segmentRange.textContent = `${{seg.rows[0].datetime}} → ${{seg.rows[seg.rows.length-1].datetime}}`;
          }} else {{
            banner.style.display = 'none';
          }}
        }} else {{
          banner.style.display = 'none';
        }}
        renderMetrics();
        const detail = getDetailed(state.etf);
        const allRows = detail.rows || [];
        const rows = getFilteredRows(allRows);
        const k = detail.kelly || {{}};
        renderChart();
        renderTradeStats(k);
        renderTradeTable();
      }} catch (err) {{
        console.error('Dashboard render error:', err);
        document.getElementById('plotlyChart').innerHTML = `
          <div style="display:flex;align-items:center;justify-content:center;height:100%;flex-direction:column;gap:12px;color:var(--muted);">
            <div style="font-size:18px;font-weight:600;color:var(--bad);">数据加载异常</div>
            <div style="font-size:13px;max-width:500px;text-align:center;line-height:1.6;">
              ${{err.message}}<br>
              请检查浏览器控制台获取详细信息，或尝试清除 localStorage 后刷新页面。
            </div>
            <button class="btn primary" onclick="localStorage.removeItem('${{STORAGE_KEY}}');location.reload();">清除缓存并刷新</button>
          </div>
        `;
      }}
    }}

    loadSavedState();
    initControls();
    render();
    initAsyncData();
  </script>
</body>
</html>
"""


def main() -> None:
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    layouts = build_plotly_layouts()

    # Write split data files
    write_versioned_json(payload.get("versioned", {}))
    write_model_versions_json(payload.get("model_versions", []))
    write_research_backtests_json(payload.get("research_backtests", []))
    write_index_json(
        payload.get("detailed", {}),
        payload.get("versioned", {}),
        payload.get("model_versions", []),
        payload.get("best_version", ""),
        payload.get("latest_version", ""),
        payload.get("research_backtests", []),
    )

    # Build HTML with reduced inline payload (default view only)
    inline_payload = dict(payload)
    # Keep version keys for immediate dropdown population, strip heavy detailed data
    for v in list(inline_payload.get("versioned", {}).keys()):
        inline_payload["versioned"][v] = {}
    inline_payload["model_versions"] = []
    inline_payload["research_backtests"] = [
        {key: value for key, value in row.items() if key != "updated_at_ts"}
        for row in inline_payload.get("research_backtests", [])
    ]
    OUTPUT_PATH.write_text(build_html(inline_payload, layouts), encoding="utf-8")
    print(f"[done] wrote {OUTPUT_PATH}")
    print(f"[done] wrote {DASHBOARD_DATA_DIR}/index.json")
    print(f"[done] wrote {DASHBOARD_DATA_DIR}/model_versions.json")
    print(f"[done] wrote {DASHBOARD_DATA_DIR}/research_backtests.json")
    vc = len(payload.get('versioned', {}))
    print(f"[done] wrote {DASHBOARD_DATA_DIR}/versioned/ ({vc} versions)")


if __name__ == "__main__":
    main()
