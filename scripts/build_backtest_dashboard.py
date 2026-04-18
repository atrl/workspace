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

# Design-system colours
C_STRATEGY = "#9a3412"
C_BUYHOLD = "#0f766e"
C_BUY = "#0f766e"
C_SELL = "#b91c1c"
C_INK = "#1b1a17"
C_MUTED = "#6d6458"
C_LINE = "#d8ccbb"
C_BG = "#f4efe6"
C_PANEL = "#fffaf2"

FEATURE_COLS = [
    "vix", "rsi", "volatility", "spread", "vix_zscore",
    "spread_trend", "ma5_ma20", "vol_strength",
    "mom_20", "intraday_range", "vix_mom", "spread_vix",
]


def load_features() -> dict[str, dict[str, float]] | None:
    """Load features_v2.pkl and return {date_str: {feature: value}}."""
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
        detailed_rows[etf] = {
            "summary": item,
            "csv_path": f"detailed/{etf.lower()}_detailed_trades.csv",
            "rows": rows,
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
    """4-subplot layout (default)."""
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
    """Single-subplot layout for focused view."""
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
      grid-template-columns: 3fr 1fr;
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
      grid-template-columns: 1fr 220px;
      gap: 14px;
      align-items: start;
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
    .split {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
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
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .mono {{ font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }}
    .raw-panel {{ margin-top: 18px; }}
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
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf8;
    }}
    .review {{
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.6;
      color: var(--muted);
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .source-list {{ display: grid; gap: 10px; }}
    .source-item {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 10px 12px;
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 14px;
      font-size: 13px;
    }}
    .source-item a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    td.delta-good {{ color: var(--good); font-weight: 700; }}
    td.delta-bad {{ color: var(--bad); font-weight: 700; }}
    @media (max-width: 1100px) {{
      .hero, .chart-grid, .split, .status-strip {{ grid-template-columns: 1fr; }}
      .summary-grid, .window-bar {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .chart-wrapper {{ grid-template-columns: 1fr; }}
      .info-panel {{ display: none; }}
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 14px; }}
      .summary-grid, .window-bar {{ grid-template-columns: 1fr; }}
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
          交互式回测查看器。4 子图联动（NAV + 买卖点、仓位、超额收益、回撤），鼠标悬停可查看当日特征（VIX、RSI、波动率等）。
          直接打开 <span class="mono">runtime/generalization/dashboard.html</span> 即可查看。
        </p>
        <div class="jump-links">
          <a class="badge" href="#mainChart">Main Chart</a>
          <a class="badge" href="#ruleComparison">Rule Comparison</a>
          <a class="badge" href="#executionOverlay">Overlay Review</a>
          <a class="badge" href="#rawDaily">Raw Daily Data</a>
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
              <option value="all">All Panels</option>
              <option value="nav">NAV + Trades</option>
              <option value="position">Position</option>
              <option value="excess">Excess Return</option>
              <option value="drawdown">Drawdown</option>
            </select>
          </div>
          <div class="toolbar-group">
            <label>Rows <span class="help-icon" data-tip="Raw Daily Data 表格显示行数">?</span></label>
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
            <label>Zoom <span class="help-icon" data-tip="快速缩放到最近 N 个月">?</span></label>
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
              <button class="btn" id="resetZoomBtn" type="button">Reset</button>
            </div>
          </div>
        </div>
        <div class="chart-meta" id="chartMeta"></div>
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
            </div>
          </div>
        </div>
      </div>
      <div class="card" id="ruleComparison">
        <h2 class="section-title">Rule-Based Comparison</h2>
        <table id="comparisonTable"></table>
      </div>
    </div>

    <div class="split">
      <div class="card" id="executionOverlay">
        <h2 class="section-title">Execution Overlay</h2>
        <table id="overlayTable"></table>
      </div>
      <div class="card">
        <h2 class="section-title">Execution Review</h2>
        <div class="review" id="reviewBox"></div>
      </div>
    </div>

    <div class="card raw-panel" id="rawDaily">
      <h2 class="section-title">Raw Daily Data</h2>
      <div class="raw-tools">
        <a id="csvLink" class="button-link" href="#" target="_blank" rel="noreferrer">Open Raw CSV</a>
        <div class="sub">展示字段：datetime / strategy_nav / buyhold_nav / position / excess / excess_pct / action / features</div>
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
    const comparisonTable = document.getElementById('comparisonTable');
    const overlayTable = document.getElementById('overlayTable');
    const reviewBox = document.getElementById('reviewBox');
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
    const STORAGE_KEY = 'soxl-dashboard-state-v3';

    function fmtPct(v, digits = 2) {{
      return `${{(v * 100).toFixed(digits)}}%`;
    }}
    function fmtNum(v, digits = 2) {{
      return Number(v).toFixed(digits);
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
    function shiftIsoDate(iso, months) {{
      const dt = new Date(`${{iso}}T00:00:00Z`);
      dt.setUTCMonth(dt.getUTCMonth() + months);
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
        const monthsMap = {{ '6m': -6, '1y': -12, '2y': -24, '3y': -36 }};
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

    function renderComparison() {{
      const summary = getDetailed(state.etf).summary;
      const rl = DATA.results[state.etf];
      const rule = DATA.rule_based[state.etf];
      const rows = [
        ['Model', 'Return', 'Sharpe', 'MaxDD', 'Trades', 'Avg Pos'],
        ['RL', `${{fmtNum(rl.return)}}%`, fmtNum(rl.sharpe, 3), `${{fmtNum(rl.max_dd)}}%`, rl.trades, 'n/a'],
        ['Rule', `${{fmtNum(rule.return)}}%`, fmtNum(rule.sharpe, 3), `${{fmtNum(rule.max_dd)}}%`, rule.trades, fmtNum(rule.avg_pos, 3)],
        ['Detailed Excess', `${{fmtPct(summary.excess_return)}}`, 'n/a', `${{fmtPct(summary.max_dd_strategy)}}`, summary.trades, fmtNum(summary.max_position, 3)],
        ['Delta RL-Rule', `${{fmtDelta(rl.return - rule.return)}}pp`, fmtDelta(rl.sharpe - rule.sharpe, 3), `${{fmtDelta(rule.max_dd - rl.max_dd)}}pp`, rl.trades - rule.trades, 'n/a'],
      ];
      comparisonTable.innerHTML = rows.map((row, idx) => `
        <tr>${{row.map((cell, col) => {{
          if (idx === 0) return `<th>${{cell}}</th>`;
          const deltaClass = idx === 4 && col > 0
            ? (String(cell).startsWith('-') ? 'delta-bad' : 'delta-good')
            : '';
          return `<td class="${{deltaClass}}">${{cell}}</td>`;
        }}).join('')}}</tr>
      `).join('');
    }}

    function renderOverlay() {{
      const payload = DATA.execution_overlay.results[state.etf];
      const header = ['Variant', 'Proxy Return', 'Sharpe', 'MaxDD', 'Trades', '2022 Return', '2022 MaxDD', 'Vs RL Return'];
      const body = Object.entries(payload).map(([name, item]) => {{
        const bear = item.scenarios.bear_2022;
        return [
          name,
          `${{fmtNum(item.test_proxy.return)}}%`,
          fmtNum(item.test_proxy.sharpe, 3),
          `${{fmtNum(item.test_proxy.max_dd)}}%`,
          item.test_proxy.trades,
          `${{fmtNum(bear.return)}}%`,
          `${{fmtNum(bear.max_dd)}}%`,
          `${{fmtDelta(item.test_proxy.return - DATA.results[state.etf].return)}}pp`,
        ];
      }});
      overlayTable.innerHTML = [header, ...body].map((row, idx) => `
        <tr>${{row.map((cell, col) => {{
          if (idx === 0) return `<th>${{cell}}</th>`;
          const deltaClass = col === 7 ? (String(cell).startsWith('-') ? 'delta-bad' : 'delta-good') : '';
          return `<td class="${{deltaClass}}">${{cell}}</td>`;
        }}).join('')}}</tr>
      `).join('');
      reviewBox.textContent = DATA.execution_review;
    }}

    function updateInfoPanel(row) {{
      if (!row) {{
        infoEmpty.style.display = 'block';
        infoContent.style.display = 'none';
        return;
      }}
      infoEmpty.style.display = 'none';
      infoContent.style.display = 'block';
      const f = row._features || {{}};
      document.getElementById('infoDate').textContent = row.datetime;
      document.getElementById('infoAction').textContent = row.action || 'HOLD';
      document.getElementById('infoAction').style.color = row.action === 'BUY' ? 'var(--good)' : row.action === 'SELL' ? 'var(--bad)' : 'var(--ink)';
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
    }}

    function renderChart() {{
      const rows = getFilteredRows(getDetailed(state.etf).rows);
      const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUTS[state.view || 'all']));
      const dates = rows.map(r => r.datetime);
      const strategyNav = rows.map(r => r.strategy_nav);
      const buyholdNav = rows.map(r => r.buyhold_nav);
      const prices = rows.map(r => r.price || r.strategy_nav);
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
          updateInfoPanel(row);
        }});
        gd.on('plotly_unhover', function() {{
          // keep last hovered row visible instead of clearing
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
      const rows = getFilteredRows(getDetailed(state.etf).rows).slice(-state.rows);
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
        <tr>${{row.map((cell, col) => idx === 0 ? `<th>${{cell}}</th>` : `<td class="${{col === 6 && cell === 'BUY' ? 'good' : col === 6 && cell === 'SELL' ? 'bad' : ''}}">${{cell}}</td>`).join('')}}</tr>
      `).join('');
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
      renderComparison();
      renderOverlay();
      renderChart();
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
