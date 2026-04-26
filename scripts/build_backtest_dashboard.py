#!/usr/bin/env python3
"""Build a standalone interactive HTML dashboard with Plotly for backtest inspection."""
from __future__ import annotations

import csv
from datetime import datetime
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
GENERALIZATION_DIR = ROOT / "runtime" / "execution" / "generalization"
RUNTIME_DIR = GENERALIZATION_DIR
DETAIL_DIR = GENERALIZATION_DIR / "detailed"
FEATURES_PATH = ROOT / "data" / "features_v2.pkl"
MODELS_DIR = ROOT / "artifacts" / "models"
OUTPUT_PATH = GENERALIZATION_DIR / "dashboard.html"

C_STRATEGY = "#9a3412"
C_BUYHOLD = "#0f766e"
C_BUY = "#0f766e"
C_SELL = "#b91c1c"
C_INK = "#1b1a17"
C_MUTED = "#6d6458"
C_LINE = "#d8ccbb"
C_PANEL = "#fffaf2"
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
    """Load OHLC from data/raw/symbols/{etf}.csv and compute soxl_backtest indicators."""
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

    # VIX: try features_v2.pkl first, else fallback 20
    vix_series = pd.Series(20.0, index=df.index)
    if FEATURES_PATH.exists():
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
    for row in rows:
        compact.append(
            {
                "datetime": row["datetime"],
                "strategy_nav": row["strategy_nav"],
                "price": row.get("price"),
                "buyhold_nav": row.get("buyhold_nav"),
                "position": row.get("position"),
                "excess": row.get("excess"),
                "excess_pct": row.get("excess_pct"),
                "action": row.get("action", "HOLD"),
            }
        )
    return compact


def load_model_versions() -> list[dict]:
    """Load all historical model versions from artifacts/models/*_summary.json."""
    versions = []
    if not MODELS_DIR.exists():
        return versions
    for f in sorted(MODELS_DIR.glob("*_summary.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            name = f.stem.replace("_summary", "")
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


def select_latest_version(versioned: dict[str, dict]) -> str:
    latest_version = ""
    latest_priority = -1
    latest_updated_at = -float("inf")
    for tag, payload in versioned.items():
        meta = payload.get("_meta", {})
        updated_at = meta.get("updated_at", float("-inf"))
        source_kind = meta.get("source_kind", "generalization_version")
        priority = 1 if source_kind == "post_train_review" else 0
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


def normalize_post_train_review_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prev_pos: float | None = None
    for i, r in enumerate(raw_rows):
        pos = float(r.get("position", 0.0))
        strat_nav = float(r["strategy_nav"])
        bh_nav = float(r["buyhold_nav"])
        raw_action = str(r.get("action", "") or "").upper()
        action = raw_action if raw_action in {"BUY", "SELL", "HOLD"} else infer_action_from_position(prev_pos, pos, i)

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

    if rows and rows[-1]["position"] > 0 and rows[-1]["action"] == "HOLD":
        rows[-1]["action"] = "SELL"
    return rows


def _load_post_train_review_versions(features: dict[str, dict[str, float]] | None) -> dict[str, dict]:
    """Load SOXL daily CSVs from runtime/model/cards/ and runtime/soxl/paper_trading/ as versioned data."""
    versioned: dict[str, dict] = {}

    dirs_to_scan = [
        (ROOT / "runtime" / "model" / "cards", True),   # (path, recursive)
        (ROOT / "runtime" / "soxl" / "paper_trading", False),
    ]

    for ptr_dir, recursive in dirs_to_scan:
        if not ptr_dir.exists():
            continue
        globber = ptr_dir.rglob if recursive else ptr_dir.glob
        for csv_path in sorted(globber("*_soxl_daily.csv")):
            tag = csv_path.stem.replace("_soxl_daily", "")
            try:
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    raw_rows = list(reader)
            except Exception:
                continue
            if not raw_rows:
                continue

            rows = normalize_post_train_review_rows(raw_rows)
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

            source_kind = "paper_trading" if "paper_trading" in str(csv_path) else "post_train_review"
            versioned[tag] = {
                "_meta": {
                    "updated_at": csv_path.stat().st_mtime,
                    "source": str(csv_path.relative_to(ROOT)),
                    "source_kind": source_kind,
                    "shared_market_data": True,
                },
                "detailed": {
                    "SOXL": {
                        "summary": summary,
                        "csv_path": str(csv_path.relative_to(ROOT)),
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
    return versioned


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

    model_versions = load_model_versions()
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,250,242,0.8)"),
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


def build_html(payload: dict[str, object], layouts: dict[str, dict]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    layouts_json = json.dumps(layouts, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --panel-2: #f7f0e4;
      --ink: #1b1a17;
      --muted: #6d6458;
      --line: #d8ccbb;
      --accent: #9a3412;
      --accent-2: #0f766e;
      --good: #0f766e;
      --bad: #b91c1c;
      --shadow: 0 18px 40px rgba(63, 44, 26, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(154, 52, 18, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.08), transparent 28%),
        linear-gradient(180deg, #f9f4ec 0%, var(--bg) 100%);
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
      background: rgba(255, 250, 242, 0.94);
      border: 1px solid rgba(216, 204, 187, 0.9);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .hero h1 {{
      margin: 0 0 12px;
      font-size: 34px;
      line-height: 1;
      letter-spacing: -0.04em;
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
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #fffdf8;
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
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fffdf8;
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
      border-radius: 12px;
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
      border-radius: 18px;
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
      letter-spacing: -0.04em;
    }}
    .metric .s {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: -0.02em;
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
      border-radius: 999px;
      background: #fffdf8;
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
      border-radius: 16px;
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
      border-bottom: 1px solid rgba(216,204,187,0.4);
    }}
    .info-row .info-k {{
      color: var(--muted);
    }}
    .info-row .info-v {{
      font-weight: 600;
      font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
    }}
    .info-row.highlight {{
      background: rgba(154, 52, 18, 0.06);
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
      background: rgba(216,204,187,0.3);
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
      border-bottom: 1px solid rgba(216,204,187,0.3);
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
      border-radius: 16px;
      background: #fffdf8;
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
      background: #fffdf8;
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
    .trade-table tr.clickable:hover {{ background: rgba(154, 52, 18, 0.04); }}
    .trade-table td.good {{ color: var(--good); font-weight: 700; }}
    .trade-table td.bad {{ color: var(--bad); font-weight: 700; }}

    .panel-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .panel-card {{
      background: rgba(255, 250, 242, 0.94);
      border: 1px solid rgba(216, 204, 187, 0.9);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .panel-card h2 {{
      margin: 0 0 14px;
      font-size: 16px;
      letter-spacing: -0.02em;
    }}
    .stat-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 14px;
    }}
    .stat-cell {{
      background: var(--panel-2);
      border-radius: 12px;
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
      background: linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(154, 52, 18, 0.06));
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
      background: #fffdf8;
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
      background: rgba(154, 52, 18, 0.08);
      font-weight: 700;
    }}
    .version-table tr.clickable-version {{
      cursor: pointer;
    }}
    .version-table tr.clickable-version:hover {{
      background: rgba(154, 52, 18, 0.04);
    }}
    .version-table tr.clickable-version.selected {{
      background: rgba(154, 52, 18, 0.10);
      outline: 2px solid var(--accent);
      outline-offset: -2px;
    }}
    .version-wrap {{
      max-height: 400px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf8;
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
      border-radius: 999px;
      background: var(--accent);
      color: white;
      text-decoration: none;
      font-size: 13px;
    }}
    @media (max-width: 1100px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-wrapper {{ grid-template-columns: 1fr; }}
      .info-panel {{ display: none; }}
      .panel-grid {{ grid-template-columns: 1fr; }}
      .stat-row {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 14px; }}
      .summary-grid {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 28px; }}
      .chart-toolbar {{ flex-direction: column; align-items: stretch; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="jump-links" style="margin-bottom:18px;">
      <a class="badge" href="#mainChart">Main Chart</a>
      <a class="badge" href="#tradeStats">Trade Quality</a>
      <a class="badge" href="#tradeTable">Trade Performance</a>
      <a class="badge" href="#versionCompare">Model Versions</a>
    </div>

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
          <label>Start</label>
          <input id="startDateInput" class="date-input" type="date">
        </div>
        <div class="toolbar-group">
          <label>End</label>
          <input id="endDateInput" class="date-input" type="date">
        </div>
        <div class="toolbar-group">
          <label>Action</label>
          <div class="zoom-row">
            <button class="btn primary" id="applyZoomBtn" type="button">Apply</button>
            <button class="btn" id="resetZoomBtn" type="button">Reset <span style="font-size:10px;color:var(--muted);margin-left:4px;opacity:0.7;">[R]</span></button>
          </div>
        </div>
      </div>
      <div class="chart-meta" id="chartMeta"></div>
      <div id="regimeStrip" class="regime-strip"></div>
      <div class="chart-wrapper">
        <div id="plotlyChart" style="width:100%;height:760px;border-radius:16px;background:var(--panel);border:1px solid var(--line);"></div>
        <div class="info-panel" id="infoPanel">
          <div class="info-title">📊 数据面板</div>
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
      <h2 class="section-title">🎯 Trade Quality <span style="font-size:12px;color:var(--muted);font-weight:normal;">— Kelly / Expectancy / Profit Factor</span></h2>
      <div class="stat-row" id="tradeStatRow"></div>
    </div>

    <!-- Trade Performance Table -->
    <div class="card" id="tradeTable">
      <h2 class="section-title">📝 Trade Performance — 每笔 T 收益明细</h2>
      <div class="raw-tools">
        <a id="csvLink" class="button-link" href="#" target="_blank" rel="noreferrer">Open Raw CSV</a>
        <div class="sub">点击表头可排序，点击交易行聚焦到图表对应区间。BUY → SELL 配对。</div>
      </div>
      <div class="trade-table-wrap">
        <table class="trade-table" id="tradePerfTable"></table>
      </div>
    </div>

    <!-- Model Version Comparison -->
    <div class="card" id="versionCompare">
      <h2 class="section-title">📚 Model Version Comparison — 历史策略对比</h2>
      <div class="raw-tools">
        <div class="sub">点击表格中的版本名查看训练曲线。绿色高亮 = 该列最优值。</div>
      </div>
      <div class="version-wrap">
        <table class="version-table" id="versionTable"></table>
      </div>
    </div>

    <!-- Training Curve -->
    <div class="card" id="trainingCurve">
      <h2 class="section-title">📈 Training Curve <span style="font-size:12px;color:var(--muted);font-weight:normal;">— 点击上方表格查看训练过程</span></h2>
      <div id="trainingCurveChart" style="width:100%;height:360px;border-radius:16px;background:var(--panel);border:1px solid var(--line);"></div>
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    const PLOTLY_LAYOUTS = {layouts_json};
    const ETFS = Object.keys(DATA.detailed);
    const VERSIONS = Object.keys(DATA.versioned || {{}});
    const state = {{
      etf: ETFS[0],
      version: DATA.latest_version || DATA.best_version || '',
      versionSource: 'auto_latest',
      range: 'all',
      zoomMode: 'all',
      zoomStart: null,
      zoomEnd: null,
      sortCol: 'return_pct',
      sortAsc: false,
    }};

    const etfSelect = document.getElementById('etfSelect');
    const versionSelect = document.getElementById('versionSelect');
    const metricGrid = document.getElementById('metricGrid');
    const tradeStatRow = document.getElementById('tradeStatRow');
    const tradePerfTable = document.getElementById('tradePerfTable');
    const versionTable = document.getElementById('versionTable');
    const csvLink = document.getElementById('csvLink');
    const chartMeta = document.getElementById('chartMeta');
    const startDateInput = document.getElementById('startDateInput');
    const endDateInput = document.getElementById('endDateInput');
    const applyZoomBtn = document.getElementById('applyZoomBtn');
    const resetZoomBtn = document.getElementById('resetZoomBtn');
    const infoEmpty = document.getElementById('infoEmpty');
    const infoContent = document.getElementById('infoContent');
    const regimeStrip = document.getElementById('regimeStrip');
    const STORAGE_KEY = 'soxl-dashboard-state-v10';
    const hydratedVersionCache = {{}};
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
      return v == null ? 'n/a' : `${{(v * 100).toFixed(digits)}}%`;
    }}
    function fmtNum(v, digits = 2) {{
      return v == null ? 'n/a' : Number(v).toFixed(digits);
    }}
    function fmtDelta(v, digits = 2) {{
      const n = Number(v);
      const sign = n > 0 ? '+' : '';
      return `${{sign}}${{n.toFixed(digits)}}`;
    }}
    function fmtMoney(v) {{
      if (v == null) return 'n/a';
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

    function getDetailed(etf) {{
      if (state.version && DATA.versioned && DATA.versioned[state.version]) {{
        const versionDetail = DATA.versioned[state.version].detailed[etf];
        if (versionDetail) {{
          return hydrateVersionDetail(etf, state.version, versionDetail);
        }}
      }}
      return DATA.detailed[etf];
    }}
    function getWindows(etf) {{
      if (state.version && DATA.versioned && DATA.versioned[state.version]) {{
        const versionWindows = DATA.versioned[state.version].windows[etf];
        if (versionWindows) {{
          return versionWindows;
        }}
      }}
      return DATA.windows[etf];
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
    }}
    function applyZoomPreset(mode) {{
      const bounds = getDateBounds(getRangeRows(getDetailed(state.etf).rows));
      if (!bounds.start) return;
      if (mode === 'all') {{
        state.zoomMode = 'all';
        state.zoomStart = null;
        state.zoomEnd = null;
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
        saveState();
        render();
      }});
      versionSelect.addEventListener('change', e => {{
        state.version = e.target.value;
        state.versionSource = state.version ? 'manual' : 'current';
        state.zoomMode = 'all';
        state.zoomStart = null;
        state.zoomEnd = null;
        saveState();
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

    function renderMetrics() {{
      const detail = getDetailed(state.etf);
      const summary = detail.summary;
      const window = getWindows(state.etf);
      const rl = DATA.results[state.etf];
      const rule = DATA.rule_based[state.etf];
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
            <div class="sk" style="color:var(--accent);">⚠ 连续持仓策略</div>
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
      csvLink.href = getDetailed(state.etf).csv_path;
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

    function renderTrainingCurve(idx) {{
      const versions = DATA.model_versions || [];
      if (idx == null || !versions[idx] || !versions[idx].selection_history || !versions[idx].selection_history.length) {{
        Plotly.purge('trainingCurveChart');
        document.getElementById('trainingCurveChart').innerHTML = `
          <div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted);font-size:14px;">
            点击上方表格中的版本名查看训练曲线
          </div>
        `;
        return;
      }}
      const v = versions[idx];
      const sh = v.selection_history;
      const iters = sh.map(h => h.iter);
      const returns = sh.map(h => h.return);
      const sharpes = sh.map(h => h.sharpe);
      const maxDds = sh.map(h => h.max_dd);

      const traces = [
        {{
          x: iters, y: returns, name: 'Return',
          type: 'scatter', mode: 'lines+markers',
          line: {{ color: '{C_STRATEGY}', width: 2 }},
          marker: {{ size: 6 }},
          yaxis: 'y1',
        }},
        {{
          x: iters, y: sharpes, name: 'Sharpe',
          type: 'scatter', mode: 'lines+markers',
          line: {{ color: '{C_BUYHOLD}', width: 2 }},
          marker: {{ size: 6 }},
          yaxis: 'y2',
        }},
        {{
          x: iters, y: maxDds, name: 'MaxDD',
          type: 'scatter', mode: 'lines+markers',
          line: {{ color: '{C_BAD}', width: 2, dash: 'dot' }},
          marker: {{ size: 6 }},
          yaxis: 'y3',
        }},
      ];

      const layout = {{
        title: {{
          text: `${{v.name}}${{v.promoted ? ' ★' : ''}} — 训练过程 (valid)`,
          font: {{ size: 14, color: '{C_INK}' }},
        }},
        height: 340,
        margin: {{ l: 60, r: 60, t: 50, b: 40 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '{C_PANEL}',
        font: {{ family: 'ui-sans-serif, -apple-system, sans-serif', color: '{C_INK}', size: 11 }},
        hovermode: 'x unified',
        legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 }},
        xaxis: {{ title: 'Iteration', gridcolor: '{C_LINE}', showgrid: true }},
        yaxis: {{ title: 'Return', side: 'left', gridcolor: '{C_LINE}', showgrid: true, titlefont: {{ color: '{C_STRATEGY}' }}, tickfont: {{ color: '{C_STRATEGY}' }} }},
        yaxis2: {{ title: 'Sharpe', overlaying: 'y', side: 'right', showgrid: false, titlefont: {{ color: '{C_BUYHOLD}' }}, tickfont: {{ color: '{C_BUYHOLD}' }} }},
        yaxis3: {{ title: 'MaxDD', overlaying: 'y', side: 'right', position: 0.98, showgrid: false, titlefont: {{ color: '{C_BAD}' }}, tickfont: {{ color: '{C_BAD}' }} }},
      }};

      Plotly.newPlot('trainingCurveChart', traces, layout, {{
        responsive: true,
        displayModeBar: false,
      }});
    }}

    function renderVersionTable() {{
      const versions = DATA.model_versions || [];
      if (!versions.length) {{
        versionTable.innerHTML = '<tr><th>无历史模型版本数据</th></tr>';
        return;
      }}

      const bestTestSharpe = Math.max(...versions.map(v => v.test_sharpe != null ? v.test_sharpe : -Infinity));
      const bestTestReturn = Math.max(...versions.map(v => v.test_return != null ? v.test_return : -Infinity));
      const bestTestMDD = Math.min(...versions.map(v => v.test_max_dd != null ? v.test_max_dd : Infinity));
      const bestValidSharpe = Math.max(...versions.map(v => v.valid_sharpe != null ? v.valid_sharpe : -Infinity));

      const header = ['Model', 'Test Return', 'Test Sharpe', 'Test MaxDD', 'Test AvgPos', 'Valid Return', 'Valid Sharpe', 'Valid MaxDD'];
      const body = versions.map(v => [
        v.name + (v.promoted ? ' ★' : ''),
        v.test_return != null ? fmtNum(v.test_return, 1) + '%' : 'n/a',
        v.test_sharpe != null ? fmtNum(v.test_sharpe, 3) : 'n/a',
        v.test_max_dd != null ? fmtNum(v.test_max_dd, 1) + '%' : 'n/a',
        v.test_avg_pos != null ? fmtNum(v.test_avg_pos, 3) : 'n/a',
        v.valid_return != null ? fmtNum(v.valid_return, 1) + '%' : 'n/a',
        v.valid_sharpe != null ? fmtNum(v.valid_sharpe, 3) : 'n/a',
        v.valid_max_dd != null ? fmtNum(v.valid_max_dd, 1) + '%' : 'n/a',
      ]);

      versionTable.innerHTML = [header, ...body].map((row, ridx) => `
        <tr data-idx="${{ridx > 0 ? ridx - 1 : ''}}" class="${{ridx > 0 ? 'clickable-version' : ''}}">
          ${{row.map((cell, col) => {{
            if (ridx === 0) return `<th>${{cell}}</th>`;
            const v = versions[ridx - 1];
            let cls = v.promoted ? 'promoted' : '';
            if (col === 1 && v.test_return === bestTestReturn) cls += ' best';
            if (col === 2 && v.test_sharpe === bestTestSharpe) cls += ' best';
            if (col === 3 && v.test_max_dd === bestTestMDD) cls += ' best';
            if (col === 6 && v.valid_sharpe === bestValidSharpe) cls += ' best';
            return `<td class="${{cls.trim()}}">${{cell}}</td>`;
          }}).join('')}}
        </tr>
      `).join('');

      versionTable.querySelectorAll('tr.clickable-version').forEach(tr => {{
        tr.addEventListener('click', () => {{
          versionTable.querySelectorAll('tr.clickable-version').forEach(r => r.classList.remove('selected'));
          tr.classList.add('selected');
          const idx = parseInt(tr.dataset.idx, 10);
          renderTrainingCurve(idx);
          document.getElementById('trainingCurve').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }});
      }});
    }}

    function renderRegimeStrip(rows) {{
      if (!rows.length) {{ regimeStrip.innerHTML = ''; return; }}
      regimeStrip.innerHTML = rows.map(r => {{
        const f = r._features || {{}};
        const vz = f.vix_zscore;
        let color = 'rgba(216,204,187,0.5)';
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
      // K-Line on dedicated price row
      const opens = [], highs = [], lows = [], closes = [], ma180Dates = [], ma180Vals = [];
      for (const r of rows) {{
        if (r.open != null) opens.push(r.open);
        if (r.high != null) highs.push(r.high);
        if (r.low != null) lows.push(r.low);
        if (r.close != null) closes.push(r.close);
        if (r.ma180 != null) {{ ma180Dates.push(r.datetime); ma180Vals.push(r.ma180); }}
      }}
      if (opens.length) {{
        traces.push({{
          x: dates, open: opens, high: highs, low: lows, close: closes,
          name: 'SOXL', type: 'candlestick',
          increasing: {{line: {{color: '#0f766e'}}, fillcolor: 'rgba(15,118,110,0.12)'}},
          decreasing: {{line: {{color: '#b91c1c'}}, fillcolor: 'rgba(185,28,28,0.12)'}},
          xaxis: xaPrice, yaxis: yaPrice,
        }});
      }}
      if (ma180Vals.length) {{
        traces.push({{
          x: ma180Dates, y: ma180Vals,
          name: 'MA180', type: 'scatter', mode: 'lines',
          line: {{ color: '#9a3412', width: 1.5, dash: 'dash' }},
          hovertemplate: '%{{x}}<br>MA180: $%{{y:,.2f}}<extra></extra>',
          xaxis: xaPrice, yaxis: yaPrice,
        }});
      }}
      if (buyRows.length) {{
        traces.push({{
          x: buyRows.map(r => r.datetime),
          y: buyRows.map(r => r.price || r.close || r.strategy_nav),
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
          y: sellRows.map(r => r.price || r.close || r.strategy_nav),
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
        fillcolor: 'rgba(154,52,18,0.12)',
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
          saveState();
          syncZoomInputs();
          renderChart();
        }});
      }});

      if (rows.length) {{
        const versionPayload = state.version && DATA.versioned ? DATA.versioned[state.version] : null;
        const isFallbackView = Boolean(versionPayload && !((versionPayload.detailed || {{}})[state.etf]));
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
      etfSelect.value = state.etf;
      versionSelect.value = state.version || '';
      document.querySelectorAll('.pill').forEach(p => {{
        p.classList.toggle('active', p.dataset.range === state.range);
      }});
      syncZoomInputs();
      renderMetrics();
      const allRows = getDetailed(state.etf).rows;
      const rows = getFilteredRows(allRows);
      const k = getDetailed(state.etf).kelly;
      renderChart();
      renderTradeStats(k);
      renderTradeTable();
      renderVersionTable();
      renderTrainingCurve(null);
    }}

    loadSavedState();
    initControls();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    layouts = build_plotly_layouts()
    OUTPUT_PATH.write_text(build_html(payload, layouts), encoding="utf-8")
    print(f"[done] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
