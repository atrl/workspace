#!/usr/bin/env python3
"""Build a standalone interactive HTML dashboard with Plotly for backtest inspection."""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT / "runtime" / "generalization"
DETAIL_DIR = RUNTIME_DIR / "detailed"
FEATURES_PATH = ROOT / "data" / "features_v2.pkl"
OUTPUT_PATH = RUNTIME_DIR / "dashboard.html"

C_STRATEGY = "#9a3412"
C_BUYHOLD = "#0f766e"
C_BUY = "#0f766e"
C_SELL = "#b91c1c"
C_INK = "#1b1a17"
C_MUTED = "#6d6458"
C_LINE = "#d8ccbb"
C_BG = "#f4efe6"
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def compute_tear_sheet(rows: list[dict]) -> dict:
    """Compute monthly returns heatmap data, yearly returns, daily return distribution."""
    monthly: dict[int, dict[int, dict]] = {}
    yearly: dict[int, dict] = {}
    daily_rets_strategy = []
    daily_rets_buyhold = []

    prev = None
    for row in rows:
        if prev:
            daily_rets_strategy.append(row["strategy_nav"] / prev["strategy_nav"] - 1)
            daily_rets_buyhold.append(row["buyhold_nav"] / prev["buyhold_nav"] - 1)

        dt = str(row["datetime"])
        year = int(dt[:4])
        month = int(dt[5:7])

        if year not in monthly:
            monthly[year] = {}
        if month not in monthly[year]:
            monthly[year][month] = {"first": row, "last": row}
        else:
            monthly[year][month]["last"] = row

        if year not in yearly:
            yearly[year] = {"first": row, "last": row}
        else:
            yearly[year]["last"] = row
        prev = row

    monthly_rets = []
    years = sorted(monthly.keys())
    for y in years:
        for m in range(1, 13):
            if m in monthly.get(y, {}):
                d = monthly[y][m]
                ret = d["last"]["strategy_nav"] / d["first"]["strategy_nav"] - 1
                monthly_rets.append({"year": y, "month": m, "return": ret})
            else:
                monthly_rets.append({"year": y, "month": m, "return": None})

    yearly_rets = []
    for y in sorted(yearly.keys()):
        d = yearly[y]
        strat_ret = d["last"]["strategy_nav"] / d["first"]["strategy_nav"] - 1
        bh_ret = d["last"]["buyhold_nav"] / d["first"]["buyhold_nav"] - 1
        yearly_rets.append({"year": y, "strategy": strat_ret, "buyhold": bh_ret})

    return {
        "monthly_rets": monthly_rets,
        "yearly_rets": yearly_rets,
        "daily_rets_strategy": daily_rets_strategy,
        "daily_rets_buyhold": daily_rets_buyhold,
    }


def compute_trades(rows: list[dict]) -> list[dict]:
    trades = []
    entry_idx = -1
    for i, row in enumerate(rows):
        if row["action"] == "BUY" and entry_idx < 0:
            entry_idx = i
        elif row["action"] == "SELL" and entry_idx >= 0:
            entry_row = rows[entry_idx]
            entry_price = entry_row["price"] or entry_row["strategy_nav"]
            exit_price = row["price"] or row["strategy_nav"]
            days = i - entry_idx
            ret = entry_price > 0 and ((exit_price / entry_price - 1) * 100) or 0
            trades.append(
                {
                    "entry_date": entry_row["datetime"],
                    "exit_date": row["datetime"],
                    "days": days,
                    "return_pct": ret,
                }
            )
            entry_idx = -1
    return trades


def build_payload() -> dict[str, object]:
    detailed_summary = load_json(DETAIL_DIR / "summary.json")
    detailed_rows = {}
    windows = {}
    features = load_features()
    for item in detailed_summary:
        etf = item["etf"]
        rows = load_csv_rows(DETAIL_DIR / f"{etf.lower()}_detailed_trades.csv")
        if features:
            for r in rows:
                dt = str(r["datetime"])
                r["_features"] = features.get(dt, {})
        else:
            for r in rows:
                r["_features"] = {}
        regime_shapes = compute_regime_shapes(rows)
        tear_sheet = compute_tear_sheet(rows)
        trades = compute_trades(rows)
        detailed_rows[etf] = {
            "summary": item,
            "csv_path": f"detailed/{etf.lower()}_detailed_trades.csv",
            "rows": rows,
            "regime_shapes": regime_shapes,
            "tear_sheet": tear_sheet,
            "trades": trades,
        }
        windows[etf] = {
            "start": rows[0]["datetime"] if rows else "",
            "end": rows[-1]["datetime"] if rows else "",
            "rows": len(rows),
        }
    return {
        "generated_from": {
            "summary": "runtime/generalization/detailed/summary.json",
            "detailed_dir": "runtime/generalization/detailed",
            "results": "runtime/generalization/results.json",
            "rule_based": "runtime/generalization/rule_based_results.json",
            "execution_overlay": "runtime/generalization/execution_overlay_results.json",
            "execution_review": "runtime/generalization/execution_overlay_review.md",
        },
        "summary": detailed_summary,
        "windows": windows,
        "detailed": detailed_rows,
        "results": load_json(RUNTIME_DIR / "results.json"),
        "rule_based": load_json(RUNTIME_DIR / "rule_based_results.json"),
        "execution_overlay": load_json(RUNTIME_DIR / "execution_overlay_results.json"),
        "execution_review": (RUNTIME_DIR / "execution_overlay_review.md").read_text(encoding="utf-8"),
    }


def _build_layout_all() -> dict:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        subplot_titles=("NAV + Trades", "Position", "Excess Return %", "Drawdown"),
    )
    fig.update_layout(
        height=760,
        margin=dict(l=60, r=30, t=90, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C_PANEL,
        font=dict(family='ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', color=C_INK, size=12),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,250,242,0.8)"),
        dragmode="zoom",
    )
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor=C_LINE,
            showticklabels=(i == 4), tickfont=dict(size=11),
            showspikes=True, spikemode="across", spikecolor=C_MUTED, spikethickness=1, spikedash="dot",
            row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor=C_LINE,
            tickfont=dict(size=11),
            showspikes=True, spikemode="across", spikecolor=C_MUTED, spikethickness=1, spikedash="dot",
            row=i, col=1,
        )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Pos", row=2, col=1)
    fig.update_yaxes(title_text="Excess %", row=3, col=1)
    fig.update_yaxes(title_text="DD %", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return fig.to_plotly_json()["layout"]


def _build_layout_single(title: str, y_title: str) -> dict:
    fig = go.Figure()
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=C_PANEL,
        font=dict(family='ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', color=C_INK, size=12),
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255,250,242,0.8)"),
        dragmode="zoom",
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(showgrid=True, gridcolor=C_LINE, tickfont=dict(size=11), showspikes=True, spikemode="across", spikecolor=C_MUTED, spikethickness=1, spikedash="dot"),
        yaxis=dict(showgrid=True, gridcolor=C_LINE, tickfont=dict(size=11), title=y_title, showspikes=True, spikemode="across", spikecolor=C_MUTED, spikethickness=1, spikedash="dot"),
    )
    return fig.to_plotly_json()["layout"]


def build_plotly_layouts() -> dict[str, dict]:
    return {
        "all": _build_layout_all(),
        "nav": _build_layout_single("NAV + Trades", "NAV"),
        "position": _build_layout_single("Position", "Position"),
        "excess": _build_layout_single("Excess Return %", "Excess %"),
        "drawdown": _build_layout_single("Drawdown", "DD %"),
    }


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
    .kb-hint {{
      font-size: 10px;
      color: var(--muted);
      margin-left: 4px;
      opacity: 0.7;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
      margin-bottom: 18px;
    }}
    .window-bar {{
      display: grid;
      grid-template-columns: 1.3fr 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .status-strip {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .notice {{
      background: linear-gradient(135deg, rgba(154, 52, 18, 0.08), rgba(15, 118, 110, 0.06));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .notice .title {{
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .notice .body {{
      font-size: 15px;
      line-height: 1.5;
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
    .chart-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      margin-bottom: 18px;
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
      display: flex;
      justify-content: space-between;
      align-items: center;
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

    /* Tear sheet & analysis panels */
    .analysis-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .analysis-grid .card {{
      min-height: 380px;
    }}
    .mini-chart {{
      width: 100%;
      height: 300px;
      border-radius: 12px;
      background: var(--panel);
      border: 1px solid var(--line);
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

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr.clickable {{ cursor: pointer; }}
    tr.clickable:hover {{ background: rgba(154, 52, 18, 0.04); }}
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
    .table-wrap {{
      max-height: 400px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf8;
    }}
    @media (max-width: 1100px) {{
      .hero, .window-bar, .status-strip {{ grid-template-columns: 1fr; }}
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-wrapper {{ grid-template-columns: 1fr; }}
      .info-panel {{ display: none; }}
      .analysis-grid {{ grid-template-columns: 1fr; }}
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
    <div class="hero">
      <div class="card">
        <h1>Backtest Dashboard</h1>
        <p class="sub">
          交互式回测查看器。4 子图联动，鼠标悬停查看当日特征，快捷键 1-5 切换视图，+/- 缩放，R 重置，←→ 平移。
        </p>
        <div class="jump-links">
          <a class="badge" href="#mainChart">Main Chart</a>
          <a class="badge" href="#tearSheet">Tear Sheet</a>
          <a class="badge" href="#tradeAnalysis">Trade Analysis</a>
          <a class="badge" href="#tradeLog">Trade Log</a>
        </div>
      </div>
    </div>

    <div class="grid summary-grid" id="metricGrid"></div>

    <div class="status-strip">
      <div class="notice" id="statusStrip"></div>
      <div class="notice">
        <div class="title">Evidence Navigator</div>
        <div class="source-list" id="sourceList"></div>
      </div>
    </div>

    <div class="window-bar">
      <div class="notice">
        <div class="title">Evaluation Window</div>
        <div class="body" id="windowBox"></div>
      </div>
      <div class="notice">
        <div class="title">Chart Range</div>
        <div class="body" id="rangeBox"></div>
      </div>
      <div class="notice">
        <div class="title">Page Notes</div>
        <div class="body">页面默认只展示 <span class="mono">test</span> 窗口，固定区间是 <span class="mono">2020-01-02</span> → <span class="mono">2026-03-31</span>。<span class="mono">2010-2018</span> 是 train、<span class="mono">2019</span> 是 valid，因此图上从 2020 年开始是设计行为，不是数据丢失。</div>
      </div>
    </div>

    <!-- Main Chart -->
    <div class="chart-grid">
      <div class="card chart-panel" id="mainChart">
        <h2 class="section-title">Main Chart</h2>
        <div class="chart-toolbar">
          <div class="toolbar-group">
            <label>ETF <span class="help-icon" data-tip="选择要查看的 ETF 标的">?</span></label>
            <select id="etfSelect"></select>
          </div>
          <div class="toolbar-group">
            <label>View <span class="help-icon" data-tip="图表布局：All=4子图，其他=单图聚焦">?</span></label>
            <select id="viewSelect">
              <option value="all">All Panels <span class="kb-hint">[1]</span></option>
              <option value="nav">NAV + Trades <span class="kb-hint">[2]</span></option>
              <option value="position">Position <span class="kb-hint">[3]</span></option>
              <option value="excess">Excess Return <span class="kb-hint">[4]</span></option>
              <option value="drawdown">Drawdown <span class="kb-hint">[5]</span></option>
            </select>
          </div>
          <div class="toolbar-group">
            <label>Rows <span class="help-icon" data-tip="Trade Log 表格显示行数">?</span></label>
            <select id="rowsSelect">
              <option value="50">50</option>
              <option value="100" selected>100</option>
              <option value="250">250</option>
            </select>
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
              <button class="btn" id="resetZoomBtn" type="button">Reset <span class="kb-hint">[R]</span></button>
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
                <span class="info-k">Price</span>
                <span class="info-v" id="infoPrice">-</span>
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
    </div>

    <!-- Analysis panels -->
    <div class="analysis-grid">
      <div class="card" id="tearSheet">
        <h2 class="section-title">📈 Tear Sheet</h2>
        <div class="stat-row" id="tearStats"></div>
        <div id="monthlyHeatmap" class="mini-chart"></div>
        <div id="yearlyBar" class="mini-chart" style="margin-top:14px;height:240px;"></div>
      </div>
      <div class="card" id="tradeAnalysis">
        <h2 class="section-title">🎯 Trade Analysis</h2>
        <div class="stat-row" id="tradeStats"></div>
        <div id="tradeScatter" class="mini-chart"></div>
        <div id="retDist" class="mini-chart" style="margin-top:14px;height:240px;"></div>
      </div>
    </div>

    <!-- Trade Log -->
    <div class="card" id="tradeLog">
      <h2 class="section-title">📝 Trade Log</h2>
      <div class="raw-tools">
        <a id="csvLink" class="button-link" href="#" target="_blank" rel="noreferrer">Open Raw CSV</a>
        <div class="sub">点击表格行聚焦到图表对应日期。仅展示有信号（BUY/SELL/HOLD 变动）的交易日。</div>
      </div>
      <div class="table-wrap">
        <table id="rawTable"></table>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    const PLOTLY_LAYOUTS = {layouts_json};
    const ETFS = Object.keys(DATA.detailed);
    const state = {{
      etf: ETFS[0],
      view: 'all',
      range: 'all',
      zoomMode: 'all',
      zoomStart: null,
      zoomEnd: null,
      rows: 100,
    }};

    const etfSelect = document.getElementById('etfSelect');
    const viewSelect = document.getElementById('viewSelect');
    const rowsSelect = document.getElementById('rowsSelect');
    const metricGrid = document.getElementById('metricGrid');
    const tearStats = document.getElementById('tearStats');
    const tradeStats = document.getElementById('tradeStats');
    const rawTable = document.getElementById('rawTable');
    const csvLink = document.getElementById('csvLink');
    const windowBox = document.getElementById('windowBox');
    const rangeBox = document.getElementById('rangeBox');
    const chartMeta = document.getElementById('chartMeta');
    const statusStrip = document.getElementById('statusStrip');
    const sourceList = document.getElementById('sourceList');
    const startDateInput = document.getElementById('startDateInput');
    const endDateInput = document.getElementById('endDateInput');
    const applyZoomBtn = document.getElementById('applyZoomBtn');
    const resetZoomBtn = document.getElementById('resetZoomBtn');
    const infoEmpty = document.getElementById('infoEmpty');
    const infoContent = document.getElementById('infoContent');
    const regimeStrip = document.getElementById('regimeStrip');
    const STORAGE_KEY = 'soxl-dashboard-state-v5';

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

    function loadSavedState() {{
      try {{
        const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
        if (!parsed) return;
        Object.assign(state, parsed);
      }} catch (_err) {{}}
    }}
    function saveState() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }}

    function getDetailed(etf) {{
      return DATA.detailed[etf];
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
      renderRawTable();
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
      renderRawTable();
    }}

    function initControls() {{
      ETFS.forEach(etf => {{
        const opt = document.createElement('option');
        opt.value = etf;
        opt.textContent = etf;
        etfSelect.appendChild(opt);
      }});
      etfSelect.addEventListener('change', e => {{
        state.etf = e.target.value;
        state.zoomMode = 'all';
        state.zoomStart = null;
        state.zoomEnd = null;
        saveState();
        render();
      }});
      viewSelect.addEventListener('change', e => {{
        state.view = e.target.value;
        saveState();
        renderChart();
      }});
      rowsSelect.addEventListener('change', e => {{
        state.rows = Number(e.target.value);
        saveState();
        renderRawTable();
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
          renderRawTable();
        }});
      }});
      document.querySelectorAll('[data-zoom]').forEach(btn => {{
        btn.addEventListener('click', () => applyZoomPreset(btn.dataset.zoom));
      }});
      applyZoomBtn.addEventListener('click', applyCustomZoom);
      resetZoomBtn.addEventListener('click', () => applyZoomPreset('all'));

      document.addEventListener('keydown', e => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        const views = {{ '1': 'all', '2': 'nav', '3': 'position', '4': 'excess', '5': 'drawdown' }};
        if (views[e.key]) {{
          e.preventDefault();
          viewSelect.value = views[e.key];
          state.view = views[e.key];
          saveState();
          renderChart();
          return;
        }}
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
      const window = DATA.windows[state.etf];
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
      windowBox.innerHTML = `
        当前 ETF: <span class="mono">${{state.etf}}</span><br>
        Test window: <span class="mono">${{window.start}}</span> → <span class="mono">${{window.end}}</span><br>
        Daily rows: <span class="mono">${{window.rows}}</span><br>
        Split: train <span class="mono">2010-2018</span> | valid <span class="mono">2019</span> | test <span class="mono">2020-2026</span>
      `;
    }}

    function renderStatusStrip() {{
      const rl = DATA.results[state.etf];
      const rule = DATA.rule_based[state.etf];
      const betterSharpe = rl.sharpe >= rule.sharpe;
      const betterReturn = rl.return >= rule.return;
      const safer = rl.max_dd <= rule.max_dd;
      statusStrip.innerHTML = `
        <div class="title">Decision Snapshot</div>
        <div class="body">
          当前 ETF: <span class="mono">${{state.etf}}</span><br>
          <span class="badge ${{betterReturn ? 'good' : 'bad'}}">Return ${{fmtDelta(rl.return - rule.return)}}pp</span>
          <span class="badge ${{betterSharpe ? 'good' : 'bad'}}">Sharpe ${{fmtDelta(rl.sharpe - rule.sharpe, 3)}}</span>
          <span class="badge ${{safer ? 'good' : 'bad'}}">MaxDD ${{fmtDelta(rule.max_dd - rl.max_dd)}}pp saved</span>
        </div>
      `;
      sourceList.innerHTML = Object.entries(DATA.generated_from).map(([label, path]) => `
        <div class="source-item">
          <span>${{label}}</span>
          <a href="${{path}}" target="_blank" rel="noreferrer">${{path}}</a>
        </div>
      `).join('');
    }}

    function renderTearSheet(rows) {{
      const ts = getDetailed(state.etf).tear_sheet;
      const trades = getDetailed(state.etf).trades || [];

      // Stats
      const totalRet = rows.length > 0 ? (rows[rows.length-1].strategy_nav / rows[0].strategy_nav - 1) * 100 : 0;
      const cagr = rows.length > 252 ? (Math.pow(rows[rows.length-1].strategy_nav / rows[0].strategy_nav, 252/rows.length) - 1) * 100 : totalRet;
      const avgDailyRet = ts.daily_rets_strategy.length > 0 ? ts.daily_rets_strategy.reduce((a,b)=>a+b,0)/ts.daily_rets_strategy.length * 100 : 0;
      const vol = ts.daily_rets_strategy.length > 1 ? Math.sqrt(ts.daily_rets_strategy.length/(ts.daily_rets_strategy.length-1)) * Math.sqrt(252) * 100 * Math.sqrt(ts.daily_rets_strategy.map(r=>r*r).reduce((a,b)=>a+b,0)/ts.daily_rets_strategy.length - Math.pow(ts.daily_rets_strategy.reduce((a,b)=>a+b,0)/ts.daily_rets_strategy.length, 2)) : 0;
      const winRate = trades.length > 0 ? (trades.filter(t=>t.return_pct>0).length / trades.length * 100).toFixed(1) : 0;

      tearStats.innerHTML = [
        {{ k: '总收益', v: fmtNum(totalRet, 1)+'%', s: fmtNum((rows[rows.length-1].buyhold_nav/rows[0].buyhold_nav-1)*100,1)+'% BH' }},
        {{ k: '年化收益', v: fmtNum(cagr, 1)+'%', s: '' }},
        {{ k: '日收益', v: fmtNum(avgDailyRet, 3)+'%', s: '均值' }},
        {{ k: '胜率', v: winRate+'%', s: trades.length+' 笔' }},
      ].map(s => `
        <div class="stat-cell">
          <div class="sk">${{s.k}}</div>
          <div class="sv" style="color:${{s.v.includes('-')?'var(--bad)':'var(--good)'}}">${{s.v}}</div>
          <div class="ss">${{s.s}}</div>
        </div>
      `).join('');

      // Monthly heatmap
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      const years = [...new Set(ts.monthly_rets.map(r=>r.year))].sort();
      const z = years.map(y => months.map((_,m) => {{
        const item = ts.monthly_rets.find(r=>r.year===y&&r.month===m+1);
        return item ? item.return : null;
      }}));
      const heatmapTrace = {{
        x: months,
        y: years,
        z: z,
        type: 'heatmap',
        colorscale: [[0, '#b91c1c'], [0.35, '#fffaf2'], [0.5, '#fffaf2'], [0.65, '#fffaf2'], [1, '#0f766e']],
        zmid: 0,
        showscale: true,
        colorbar: {{ title: '月度收益', titleside: 'right', thickness: 12, len: 0.8 }},
        hovertemplate: '%{{y}}-%{{x}}<br>收益: %{{z:.2%}}<extra></extra>',
      }};
      const heatmapLayout = {{
        height: 260,
        margin: {{ l: 50, r: 60, t: 20, b: 40 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: C_PANEL,
        font: {{ family: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11 }},
        xaxis: {{ tickfont: {{ size: 10 }} }},
        yaxis: {{ tickfont: {{ size: 10 }} }},
      }};
      Plotly.react('monthlyHeatmap', [heatmapTrace], heatmapLayout, {{ displayModeBar: false, displaylogo: false }});

      // Yearly bar
      const yStrat = ts.yearly_rets.map(r => r.strategy * 100);
      const yBH = ts.yearly_rets.map(r => r.buyhold * 100);
      const yearlyTrace1 = {{
        x: ts.yearly_rets.map(r => r.year),
        y: yStrat,
        name: 'Strategy',
        type: 'bar',
        marker: {{ color: yStrat.map(v => v >= 0 ? C_STRATEGY : C_BAD) }},
        hovertemplate: '%{{x}}<br>Strategy: %{{y:.2f}}%<extra></extra>',
      }};
      const yearlyTrace2 = {{
        x: ts.yearly_rets.map(r => r.year),
        y: yBH,
        name: 'Buy&Hold',
        type: 'bar',
        marker: {{ color: yBH.map(v => v >= 0 ? C_BUYHOLD : '#6d6458') }},
        hovertemplate: '%{{x}}<br>Buy&Hold: %{{y:.2f}}%<extra></extra>',
      }};
      const yearlyLayout = {{
        height: 240,
        margin: {{ l: 50, r: 20, t: 30, b: 40 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: C_PANEL,
        font: {{ family: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11 }},
        barmode: 'group',
        legend: {{ orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' }},
        xaxis: {{ tickfont: {{ size: 10 }} }},
        yaxis: {{ title: '年度收益 %', tickfont: {{ size: 10 }} }},
      }};
      Plotly.react('yearlyBar', [yearlyTrace1, yearlyTrace2], yearlyLayout, {{ displayModeBar: false, displaylogo: false }});
    }}

    function renderTradeAnalysis(rows) {{
      const trades = getDetailed(state.etf).trades || [];
      const ts = getDetailed(state.etf).tear_sheet;

      // Stats
      const wins = trades.filter(t => t.return_pct > 0);
      const losses = trades.filter(t => t.return_pct <= 0);
      const avgWin = wins.length > 0 ? wins.reduce((a,t)=>a+t.return_pct,0)/wins.length : 0;
      const avgLoss = losses.length > 0 ? losses.reduce((a,t)=>a+t.return_pct,0)/losses.length : 0;
      const avgDays = trades.length > 0 ? trades.reduce((a,t)=>a+t.days,0)/trades.length : 0;
      const profitFactor = losses.length > 0 && avgLoss !== 0 ? Math.abs((wins.length*avgWin)/(losses.length*avgLoss)) : (wins.length > 0 ? 999 : 0);

      tradeStats.innerHTML = [
        {{ k: '盈利笔数', v: String(wins.length), s: fmtNum(avgWin,2)+'% 均盈' }},
        {{ k: '亏损笔数', v: String(losses.length), s: fmtNum(avgLoss,2)+'% 均亏' }},
        {{ k: '平均持仓', v: fmtNum(avgDays,1)+'天', s: '' }},
        {{ k: '盈亏比', v: fmtNum(profitFactor,2), s: '' }},
      ].map(s => `
        <div class="stat-cell">
          <div class="sk">${{s.k}}</div>
          <div class="sv">${{s.v}}</div>
          <div class="ss">${{s.s}}</div>
        </div>
      `).join('');

      // Trade scatter
      const scatterTrace = {{
        x: trades.map((t,i) => i+1),
        y: trades.map(t => t.return_pct),
        mode: 'markers',
        type: 'scatter',
        marker: {{
          color: trades.map(t => t.return_pct >= 0 ? C_GOOD : C_BAD),
          size: trades.map(t => Math.max(6, Math.min(18, Math.abs(t.return_pct) * 2))),
          opacity: 0.8,
          line: {{ width: 1, color: 'white' }},
        }},
        text: trades.map(t => `入场: ${{t.entryDate}}<br>出场: ${{t.exitDate}}<br>持有: ${{t.days}}天<br>收益: ${{fmtNum(t.return_pct,2)}}%`),
        hovertemplate: '%{{text}}<extra></extra>',
      }};
      const scatterLayout = {{
        height: 260,
        margin: {{ l: 50, r: 20, t: 20, b: 40 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: C_PANEL,
        font: {{ family: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11 }},
        xaxis: {{ title: '交易序号', tickfont: {{ size: 10 }} }},
        yaxis: {{ title: '收益 %', tickfont: {{ size: 10 }} }},
        shapes: [{{ type: 'line', x0: 0, x1: trades.length+1, y0: 0, y1: 0, line: {{ color: C_LINE, width: 1, dash: 'dot' }} }}],
      }};
      Plotly.react('tradeScatter', [scatterTrace], scatterLayout, {{ displayModeBar: false, displaylogo: false }});

      // Return distribution
      const binSize = 2;
      const allRets = ts.daily_rets_strategy.map(r => r * 100);
      const minR = Math.floor((Math.min(...allRets, 0) - binSize) / binSize) * binSize;
      const maxR = Math.ceil((Math.max(...allRets, 0) + binSize) / binSize) * binSize;
      const bins = [];
      for (let b = minR; b <= maxR; b += binSize) bins.push(b);
      const stratCounts = new Array(bins.length).fill(0);
      allRets.forEach(r => {{
        const idx = Math.min(Math.floor((r - minR) / binSize), bins.length - 1);
        stratCounts[idx]++;
      }});
      const bhRets = ts.daily_rets_buyhold.map(r => r * 100);
      const bhCounts = new Array(bins.length).fill(0);
      bhRets.forEach(r => {{
        const idx = Math.min(Math.floor((r - minR) / binSize), bins.length - 1);
        bhCounts[idx]++;
      }});
      const binLabels = bins.map(b => `${{b}}~${{b+binSize}}`);
      const distTrace1 = {{
        x: binLabels,
        y: stratCounts,
        name: 'Strategy',
        type: 'bar',
        marker: {{ color: C_STRATEGY, opacity: 0.7 }},
        hovertemplate: 'Strategy<br>%{{x}}%: %{{y}} 天<extra></extra>',
      }};
      const distTrace2 = {{
        x: binLabels,
        y: bhCounts,
        name: 'Buy&Hold',
        type: 'bar',
        marker: {{ color: C_BUYHOLD, opacity: 0.7 }},
        hovertemplate: 'Buy&Hold<br>%{{x}}%: %{{y}} 天<extra></extra>',
      }};
      const distLayout = {{
        height: 240,
        margin: {{ l: 50, r: 20, t: 30, b: 60 }},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: C_PANEL,
        font: {{ family: 'ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 11 }},
        barmode: 'overlay',
        legend: {{ orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' }},
        xaxis: {{ title: '日收益区间 %', tickangle: -45, tickfont: {{ size: 9 }} }},
        yaxis: {{ title: '天数', tickfont: {{ size: 10 }} }},
      }};
      Plotly.react('retDist', [distTrace1, distTrace2], distLayout, {{ displayModeBar: false, displaylogo: false }});
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
          <span class="trade-entry">BUY</span> ${{t.entryDate}} → <span class="trade-exit">SELL</span> ${{t.exitDate}}<br>
          持有 ${{t.days}} 天 | 收益 <span class="${{t.returnPct >= 0 ? 'good' : 'bad'}}">${{fmtNum(t.returnPct, 2)}}%</span>
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
      actionEl.textContent = row.action || 'HOLD';
      actionEl.style.color = row.action === 'BUY' ? 'var(--good)' : row.action === 'SELL' ? 'var(--bad)' : 'var(--ink)';
      document.getElementById('infoPrice').textContent = fmtNum(row.price || row.strategy_nav);
      document.getElementById('infoStrat').textContent = fmtNum(row.strategy_nav);
      document.getElementById('infoBH').textContent = fmtNum(row.buyhold_nav);
      document.getElementById('infoPos').textContent = fmtNum(row.position, 4);
      document.getElementById('infoExcess').textContent = fmtNum(row.excess_pct, 2) + '%';
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
        document.getElementById('tradeEntry').textContent = trade.entryDate;
        document.getElementById('tradeExit').textContent = trade.exitDate;
        document.getElementById('tradeDays').textContent = trade.days + ' 天';
        const retEl = document.getElementById('tradeRet');
        retEl.textContent = fmtNum(trade.returnPct, 2) + '%';
        retEl.style.color = trade.returnPct >= 0 ? 'var(--good)' : 'var(--bad)';
      }} else {{
        tradeSection.style.display = 'none';
      }}
    }}

    function renderChart() {{
      const allRows = getDetailed(state.etf).rows;
      const rows = getFilteredRows(allRows);
      const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUTS[state.view || 'all']));
      const shapes = getDetailed(state.etf).regime_shapes || [];
      layout.shapes = shapes;

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
      trades.forEach(t => {{ tradeMap[t.entryDate] = t; }});

      renderRegimeStrip(rows);
      renderPositionDist(rows);
      renderTradeList(trades);

      const traces = [];
      const isAll = state.view === 'all';
      const xa = 'x', ya = 'y';
      const xa2 = isAll ? 'x2' : 'x', ya2 = isAll ? 'y2' : 'y';
      const xa3 = isAll ? 'x3' : 'x', ya3 = isAll ? 'y3' : 'y';
      const xa4 = isAll ? 'x4' : 'x', ya4 = isAll ? 'y4' : 'y';

      if (state.view === 'nav' || isAll) {{
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
        if (buyRows.length) {{
          traces.push({{
            x: buyRows.map(r => r.datetime),
            y: buyRows.map(r => r.price || r.strategy_nav),
            name: 'BUY', type: 'scatter', mode: 'markers',
            marker: {{ symbol: 'triangle-up', size: 11, color: '{C_BUY}', line: {{ color: 'white', width: 1 }} }},
            hovertemplate: '<b>BUY</b> %{{x}}<extra></extra>',
            xaxis: xa, yaxis: ya,
          }});
        }}
        if (sellRows.length) {{
          traces.push({{
            x: sellRows.map(r => r.datetime),
            y: sellRows.map(r => r.price || r.strategy_nav),
            name: 'SELL', type: 'scatter', mode: 'markers',
            marker: {{ symbol: 'triangle-down', size: 11, color: '{C_SELL}', line: {{ color: 'white', width: 1 }} }},
            hovertemplate: '<b>SELL</b> %{{x}}<extra></extra>',
            xaxis: xa, yaxis: ya,
          }});
        }}
      }}

      if (state.view === 'position' || isAll) {{
        traces.push({{
          x: dates, y: positions,
          name: 'Position', type: 'bar',
          marker: {{ color: positions.map(p => p > 0 ? 'rgba(15,118,110,0.7)' : 'rgba(109,100,88,0.4)') }},
          hovertemplate: '%{{x}}<br>Pos: %{{y:.4f}}<extra></extra>',
          xaxis: xa2, yaxis: ya2,
        }});
      }}

      if (state.view === 'excess' || isAll) {{
        traces.push({{
          x: dates, y: excessPct,
          name: 'Excess %', type: 'scatter', mode: 'lines',
          line: {{ color: '{C_INK}', width: 1.5 }},
          fill: 'tozeroy',
          fillcolor: 'rgba(27,26,23,0.08)',
          hovertemplate: '%{{x}}<br>Excess: %{{y:.2f}}%<extra></extra>',
          xaxis: xa3, yaxis: ya3,
        }});
      }}

      if (state.view === 'drawdown' || isAll) {{
        traces.push({{
          x: dates, y: strategyDD,
          name: 'Strategy DD', type: 'scatter', mode: 'lines',
          line: {{ color: '{C_STRATEGY}', width: 1.5 }},
          fill: 'tozeroy',
          fillcolor: 'rgba(154,52,18,0.12)',
          hovertemplate: '%{{x}}<br>Strat DD: %{{y:.2%}}<extra></extra>',
          xaxis: xa4, yaxis: ya4,
        }});
        traces.push({{
          x: dates, y: buyholdDD,
          name: 'Buy&Hold DD', type: 'scatter', mode: 'lines',
          line: {{ color: '{C_BUYHOLD}', width: 1.5 }},
          fill: 'tozeroy',
          fillcolor: 'rgba(15,118,110,0.10)',
          hovertemplate: '%{{x}}<br>BH DD: %{{y:.2%}}<extra></extra>',
          xaxis: xa4, yaxis: ya4,
        }});
      }}

      Plotly.react('plotlyChart', traces, layout, {{
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
      }}).then(gd => {{
        gd.on('plotly_hover', function(data) {{
          const pt = data.points[0];
          const dateStr = pt.x;
          const row = rows.find(r => r.datetime === dateStr);
          updateInfoPanel(row, tradeMap);
        }});
      }});

      if (rows.length) {{
        chartMeta.innerHTML = `
          <span>Zoom: <span class="mono">${{state.zoomMode}}</span></span>
          <span>Window: <span class="mono">${{rows[0].datetime}}</span> → <span class="mono">${{rows[rows.length-1].datetime}}</span></span>
          <span>Rows: <span class="mono">${{rows.length}}</span></span>
          <span>View: <span class="mono">${{state.view}}</span></span>
          <span>BUY: <span class="mono">${{buyRows.length}}</span></span>
          <span>SELL: <span class="mono">${{sellRows.length}}</span></span>
        `;
        rangeBox.innerHTML = `
          当前筛选: <span class="mono">${{state.range}}</span><br>
          缩放模式: <span class="mono">${{state.zoomMode}}</span><br>
          实际显示: <span class="mono">${{rows[0].datetime}}</span> → <span class="mono">${{rows[rows.length-1].datetime}}</span><br>
          说明: 当前页面不显示 <span class="mono">2019 valid</span>，所以不会看到 2019 年的数据点。
        `;
      }} else {{
        chartMeta.innerHTML = `<span>当前日期筛选下没有数据，请放宽缩放范围。</span>`;
        rangeBox.innerHTML = `
          当前筛选: <span class="mono">${{state.range}}</span><br>
          缩放模式: <span class="mono">${{state.zoomMode}}</span><br>
          说明: 当前日期区间没有命中任何样本。
        `;
      }}
    }}

    function renderRawTable() {{
      const allRows = getDetailed(state.etf).rows;
      const rows = getFilteredRows(allRows).slice(-state.rows);
      csvLink.href = getDetailed(state.etf).csv_path;
      const header = ['datetime', 'strategy_nav', 'buyhold_nav', 'position', 'excess', 'excess_pct', 'action'];
      const body = rows.map(r => [
        r.datetime,
        fmtNum(r.strategy_nav),
        fmtNum(r.buyhold_nav),
        fmtNum(r.position, 4),
        fmtNum(r.excess),
        fmtNum(r.excess_pct, 2),
        r.action,
      ]);
      rawTable.innerHTML = [header, ...body].map((row, idx) => `
        <tr class="clickable" data-date="${{row[0]}}">${{row.map((cell, col) => idx === 0 ? `<th>${{cell}}</th>` : `<td class="${{col === 6 && cell === 'BUY' ? 'good' : col === 6 && cell === 'SELL' ? 'bad' : ''}}">${{cell}}</td>`).join('')}}</tr>
      `).join('');

      rawTable.querySelectorAll('tr.clickable').forEach(tr => {{
        tr.addEventListener('click', () => {{
          const dateStr = tr.dataset.date;
          const idx = allRows.findIndex(r => r.datetime === dateStr);
          if (idx < 0) return;
          const start = Math.max(0, idx - 15);
          const end = Math.min(allRows.length - 1, idx + 15);
          state.zoomStart = allRows[start].datetime;
          state.zoomEnd = allRows[end].datetime;
          state.zoomMode = 'custom';
          saveState();
          syncZoomInputs();
          renderChart();
          renderRawTable();
          document.getElementById('mainChart').scrollIntoView({{ behavior: 'smooth' }});
        }});
      }});
    }}

    function render() {{
      etfSelect.value = state.etf;
      viewSelect.value = state.view;
      rowsSelect.value = String(state.rows);
      document.querySelectorAll('.pill').forEach(p => {{
        p.classList.toggle('active', p.dataset.range === state.range);
      }});
      syncZoomInputs();
      renderMetrics();
      renderStatusStrip();
      const allRows = getDetailed(state.etf).rows;
      const rows = getFilteredRows(allRows);
      renderChart();
      renderTearSheet(rows);
      renderTradeAnalysis(rows);
      renderRawTable();
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
