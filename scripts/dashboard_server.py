#!/usr/bin/env python3
"""Lightweight HTTP server for dashboard backtest API.

Serves static files from workspace root and provides /api/run-backtest endpoint.

Usage:
    qlib_venv/bin/python scripts/dashboard_server.py [--port 8765]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DASHBOARD_DATA_DIR = ROOT / "dashboard_data"

try:
    from run_segment_backtest import (
        _compute_metrics_from_daily,
        _pair_trades,
    )
except ModuleNotFoundError:
    from scripts.run_segment_backtest import (
        _compute_metrics_from_daily,
        _pair_trades,
    )
from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import (
    build_dataset as _build_dataset,
    feature_columns_for_packs as _feature_columns_for_packs,
    init_qlib as _init_qlib,
    segment_frame as _segment_frame,
)
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import (
    ACCOUNT,
    QLIB_DATA_PATH,
    TEST_SEG,
    load_profile,
    model_artifact_path,
)

import torch
import numpy as np
import pandas as pd


# Pre-init qlib once at module load to avoid repeated init
_QLIB_INITIALIZED = False
_DATASETS: dict[tuple[str, tuple[str, ...]], object] = {}

MARKET_ROW_FIELDS = (
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
)


def _ensure_qlib(*, feature_packs: tuple[str, ...] = (), profile_name: str = ""):
    global _QLIB_INITIALIZED
    if not _QLIB_INITIALIZED:
        _init_qlib(QLIB_DATA_PATH)
        _QLIB_INITIALIZED = True
    key = (profile_name, tuple(feature_packs))
    if key not in _DATASETS:
        _DATASETS[key] = _build_dataset(feature_packs=feature_packs, profile_name=profile_name or None)
    return _DATASETS[key]


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _build_daily_rows(
    *,
    account: pd.Series,
    price: pd.Series,
    position_history: list[float],
    trade_signals: list[tuple],
    market_lookup: dict[str, dict] | None = None,
    feature_lookup: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build JSON-ready daily rows for the interactive dashboard API."""
    account = account.astype(float)
    aligned_price = price.reindex(account.index).ffill().bfill()
    if aligned_price.empty:
        return []

    initial_price = float(aligned_price.iloc[0]) if float(aligned_price.iloc[0]) > 0 else 1.0
    buyhold_nav = ACCOUNT * (aligned_price.astype(float) / initial_price)

    action_map: dict[str, str] = {}
    for signal in trade_signals:
        if len(signal) < 4:
            continue
        sig_date, _sig_price, _sig_pos, sig_type = signal[:4]
        action_map[pd.Timestamp(sig_date).strftime("%Y-%m-%d")] = str(sig_type).upper()

    market_lookup = market_lookup or {}
    feature_lookup = feature_lookup or {}
    rows: list[dict[str, object]] = []
    for i, (ts, acc) in enumerate(account.items()):
        dt_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        p = float(aligned_price.iloc[i])
        bh_nav = float(buyhold_nav.iloc[i])
        pos = float(position_history[i]) if i < len(position_history) else 0.0
        excess = float(acc) - bh_nav
        excess_pct = (excess / bh_nav * 100) if bh_nav != 0 else 0.0

        row: dict[str, object] = {
            "datetime": dt_str,
            "strategy_nav": round(float(acc), 6),
            "price": round(p, 6),
            "buyhold_nav": round(bh_nav, 6),
            "position": round(pos, 6),
            "excess": round(excess, 6),
            "excess_pct": round(excess_pct, 6),
            "action": action_map.get(dt_str, "HOLD"),
        }
        market_row = market_lookup.get(dt_str) or {}
        for field in MARKET_ROW_FIELDS:
            if field not in market_row:
                continue
            value = _json_scalar(market_row.get(field))
            if value is not None:
                row[field] = value
        feature_snapshot = {
            str(key): _json_scalar(value)
            for key, value in (feature_lookup.get(dt_str) or {}).items()
            if _json_scalar(value) is not None
        }
        if feature_snapshot:
            row["feature_snapshot"] = feature_snapshot
        rows.append(row)
    return rows


def _feature_lookup_from_frame(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, object]]:
    if feature_frame.empty:
        return {}
    frame = feature_frame.copy()
    if isinstance(frame.index, pd.MultiIndex):
        if "instrument" in frame.index.names:
            frame = frame.droplevel("instrument")
        else:
            frame = frame.reset_index(level=list(range(1, frame.index.nlevels)), drop=True)
    frame.index = pd.to_datetime(frame.index)
    columns = [column for column in feature_columns if column in frame.columns]
    if not columns:
        columns = [str(column) for column in frame.columns]
    lookup: dict[str, dict[str, object]] = {}
    for dt, source in frame[columns].sort_index().iterrows():
        snapshot: dict[str, object] = {}
        for column in columns:
            value = _json_scalar(source[column])
            if value is not None:
                snapshot[str(column)] = value
        if snapshot:
            lookup[pd.Timestamp(dt).strftime("%Y-%m-%d")] = snapshot
    return lookup


def _response_window_from_rows(
    rows: list[dict[str, object]],
    requested_start: str,
    requested_end: str,
) -> dict[str, object]:
    if rows:
        return {
            "start": str(rows[0]["datetime"]),
            "end": str(rows[-1]["datetime"]),
            "rows": len(rows),
        }
    return {"start": requested_start, "end": requested_end, "rows": 0}


def _first_csv_date(csv_path: Path) -> str:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                dt = str(row.get("datetime") or row.get("date") or "").strip()
                if dt:
                    return dt
    except Exception:
        return ""
    return ""


def _profile_replay_start(profile_name: str) -> str:
    manifest_path = DASHBOARD_DATA_DIR / "model_replay_freshness.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for record in manifest.get("profiles") or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("profile") or "").strip() != profile_name:
            continue
        output = str(record.get("output") or "").strip()
        if not output:
            return ""
        return _first_csv_date(ROOT / output)
    return ""


def _warmup_start_date(profile_name: str, requested_start: str) -> str:
    replay_start = _profile_replay_start(profile_name)
    if replay_start and replay_start < requested_start:
        return replay_start
    return requested_start


def _backtest_request_scope(
    profile_name: str,
    requested_start: str,
    requested_end: str,
    *,
    default_test_seg: tuple[str, str] = TEST_SEG,
) -> dict[str, object]:
    warmup_start = _warmup_start_date(profile_name, requested_start)
    replay_aligned = warmup_start < requested_start
    if replay_aligned:
        return {
            "warmup_start": warmup_start,
            "feature_start": warmup_start,
            "feature_end": "",
            "test_seg": default_test_seg,
            "replay_aligned": True,
        }
    return {
        "warmup_start": requested_start,
        "feature_start": requested_start,
        "feature_end": requested_end,
        "test_seg": (requested_start, requested_end),
        "replay_aligned": False,
    }


def _slice_daily_rows(
    rows: list[dict[str, object]],
    requested_start: str,
    requested_end: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if requested_start <= str(row.get("datetime") or "") <= requested_end
    ]


def _health_payload() -> dict[str, object]:
    return {
        "ok": True,
        "service": "dashboard_server",
        "run_backtest_api": "/api/run-backtest",
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler: static files + /api/run-backtest endpoint."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(_health_payload())
            return
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run-backtest":
            self._handle_run_backtest()
        else:
            self.send_error(404, "Not Found")

    def _handle_run_backtest(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        profile_name = req.get("profile", "")
        start_date = req.get("start_date", "")
        end_date = req.get("end_date", "")
        tag = req.get("tag", "")

        if not profile_name or not start_date or not end_date:
            self._send_json({"error": "Missing required fields: profile, start_date, end_date"}, 400)
            return

        try:
            profile = load_profile(profile_name)
            feature_packs = tuple(str(pack) for pack in profile.feature.packs)
            feature_columns = _feature_columns_for_packs(feature_packs, profile.name)
            dataset = _ensure_qlib(feature_packs=feature_packs, profile_name=profile.name)
            scope = _backtest_request_scope(profile_name, start_date, end_date)
            warmup_start_date = str(scope["warmup_start"])
            feature_start = str(scope["feature_start"])
            feature_end = str(scope["feature_end"])
            full_df = _segment_frame(dataset, "test")
            if feature_end:
                seg_df = full_df.loc[feature_start:feature_end]
            else:
                seg_df = full_df.loc[feature_start:]

            if seg_df.empty:
                self._send_json({"error": f"No data for {start_date} to {end_date}"}, 400)
                return

            price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[seg_df.index]

            mpath = model_artifact_path(profile_name, tag)
            if not mpath.exists():
                self._send_json({"error": f"Model not found: {mpath}"}, 404)
                return

            ckpt = torch.load(mpath, map_location="cpu", weights_only=False)
            _, actor = build_policy(
                int(ckpt["obs_dim"]),
                profile,
                use_moe=ckpt.get("use_moe", False),
                hard_regime_routing=ckpt.get("use_moe", False),
            )
            actor.load_state_dict(ckpt["state_dict"])
            actor.eval()

            result = backtest_profile(
                actor=actor,
                feature_df=seg_df,
                price_series=price,
                position_cap_config=profile.position_cap,
                position_budget_config=profile.position_budget,
                execution_config=profile.execution,
                symbol="SOXL",
                test_seg=scope["test_seg"],
                model_config=profile.model,
            )

            # Build daily rows
            pdf = result["portfolio"]
            account = pdf["account"].astype(float)
            from scripts.build_backtest_dashboard import load_ohlc_and_indicators

            all_rows = _build_daily_rows(
                account=account,
                price=price,
                position_history=result.get("position_history", []),
                trade_signals=result.get("trade_signals", []),
                market_lookup=load_ohlc_and_indicators("SOXL"),
                feature_lookup=_feature_lookup_from_frame(seg_df, feature_columns),
            )
            rows = _slice_daily_rows(all_rows, start_date, end_date)

            # Metrics
            metrics = _compute_metrics_from_daily(rows)
            metrics["trades"] = result["trades"]

            # Pair trades
            paired = _pair_trades(rows)

            window = _response_window_from_rows(rows, start_date, end_date)
            response = {
                "profile": profile_name,
                "tag": tag,
                "start_date": start_date,
                "end_date": end_date,
                "warmup_start_date": warmup_start_date,
                "replay_aligned": bool(scope["replay_aligned"]),
                "source_kind": "interactive_rerun",
                "feature_packs": list(feature_packs),
                "feature_columns": feature_columns,
                "effective_start_date": window["start"],
                "effective_end_date": window["end"],
                "summary": metrics,
                "rows": rows,
                "trades": paired,
                "windows": {
                    "SOXL": window,
                },
            }
            self._send_json(response)

        except Exception as exc:
            import traceback
            self._send_json({"error": str(exc), "traceback": traceback.format_exc()}, 500)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

    # Override translate_path to serve from workspace root
    def translate_path(self, path):
        # Clean up the path
        path = path.split("?", 1)[0]
        path = path.split("#", 1)[0]
        path = urllib.parse.unquote(path)
        # Serve from workspace root
        return str(ROOT / path.lstrip("/"))


def main():
    parser = argparse.ArgumentParser(description="Dashboard HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    server = HTTPServer((args.bind, args.port), DashboardHandler)
    print(f"[dashboard_server] Serving at http://{args.bind}:{args.port}")
    print(f"[dashboard_server] API endpoint: POST http://{args.bind}:{args.port}/api/run-backtest")
    print(f"[dashboard_server] Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard_server] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
