#!/usr/bin/env python3
"""Build a standalone local HTML dashboard for backtest inspection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT / "runtime" / "generalization"
DETAIL_DIR = RUNTIME_DIR / "detailed"
OUTPUT_PATH = RUNTIME_DIR / "dashboard.html"


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
    for item in detailed_summary:
        etf = item["etf"]
        rows = load_csv_rows(DETAIL_DIR / f"{etf.lower()}_detailed_trades.csv")
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


def build_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Backtest Dashboard</title>
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
      grid-template-columns: 1.5fr 1fr;
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
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    select {{
      min-width: 180px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #fffdf8;
      font-size: 14px;
      color: var(--ink);
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
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .chart-panel {{
      min-height: 420px;
    }}
    .chart-tools {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fffdf8;
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill.active {{
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }}
    svg {{
      width: 100%;
      height: 320px;
      display: block;
      border-radius: 16px;
      background: linear-gradient(180deg, #fffdf9 0%, #f7f0e4 100%);
      border: 1px solid var(--line);
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
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
    th:first-child, td:first-child {{
      text-align: left;
    }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .mono {{ font-family: ui-monospace, "SFMono-Regular", Menlo, monospace; }}
    .raw-panel {{
      margin-top: 18px;
    }}
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
    .chart-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .zoom-tools {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 12px 0 14px;
      align-items: end;
    }}
    .zoom-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .zoom-field label {{
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .date-input {{
      width: 100%;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--text);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
    }}
    .button-link.ghost {{
      background: transparent;
      color: var(--text);
    }}
    .source-list {{
      display: grid;
      gap: 10px;
    }}
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
    td.delta-good {{
      color: var(--good);
      font-weight: 700;
    }}
    td.delta-bad {{
      color: var(--bad);
      font-weight: 700;
    }}
    @media (max-width: 1100px) {{
      .hero, .chart-grid, .split, .status-strip {{ grid-template-columns: 1fr; }}
      .summary-grid, .window-bar {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .zoom-tools {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 14px; }}
      .summary-grid, .window-bar {{ grid-template-columns: 1fr; }}
      .zoom-tools {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div class="card">
        <h1>Backtest Dashboard</h1>
        <p class="sub">
          本地查看器，直接展示详细回测原始数据、策略净值、Buy&Hold、超额收益、仓位、买卖点，
          以及 execution overlay 与 rule-based 的对比。数据已内嵌进页面，直接打开
          <span class="mono">runtime/generalization/dashboard.html</span> 即可查看。
        </p>
        <div class="jump-links">
          <a class="badge" href="#mainChart">Main Chart</a>
          <a class="badge" href="#ruleComparison">Rule Comparison</a>
          <a class="badge" href="#executionOverlay">Overlay Review</a>
          <a class="badge" href="#rawDaily">Raw Daily Data</a>
        </div>
      </div>
      <div class="card">
        <div class="controls">
          <div>
            <label>ETF</label>
            <select id="etfSelect"></select>
          </div>
          <div>
            <label>Chart</label>
            <select id="chartSelect">
              <option value="nav">Strategy vs Buy&Hold</option>
              <option value="excess">Excess Return %</option>
              <option value="position">Position + Trades</option>
              <option value="drawdown">Drawdown</option>
            </select>
          </div>
          <div>
            <label>Rows</label>
            <select id="rowsSelect">
              <option value="50">50</option>
              <option value="100" selected>100</option>
              <option value="250">250</option>
            </select>
          </div>
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
        <div class="chart-tools">
          <button class="pill active" data-range="all">All</button>
          <button class="pill" data-range="2020">2020</button>
          <button class="pill" data-range="2022">2022</button>
          <button class="pill" data-range="2023_2024">2023-2024</button>
        </div>
        <div class="zoom-tools">
          <div>
            <div class="zoom-field">
              <label>Zoom Preset</label>
              <div class="zoom-row">
                <button class="button-link ghost" data-zoom="6m" type="button">6M</button>
                <button class="button-link ghost" data-zoom="1y" type="button">1Y</button>
                <button class="button-link ghost" data-zoom="2y" type="button">2Y</button>
                <button class="button-link ghost" data-zoom="3y" type="button">3Y</button>
              </div>
            </div>
          </div>
          <div class="zoom-field">
            <label>Start Date</label>
            <input id="startDateInput" class="date-input" type="date">
          </div>
          <div class="zoom-field">
            <label>End Date</label>
            <input id="endDateInput" class="date-input" type="date">
          </div>
          <div>
            <div class="zoom-field">
              <label>Zoom Action</label>
              <div class="zoom-row">
                <button class="button-link" id="applyZoomBtn" type="button">Apply</button>
                <button class="button-link ghost" id="resetZoomBtn" type="button">Reset</button>
              </div>
            </div>
          </div>
        </div>
        <div class="chart-meta" id="chartMeta"></div>
        <svg id="chartSvg" viewBox="0 0 980 320" preserveAspectRatio="none"></svg>
        <div class="legend" id="legend"></div>
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
        <div class="sub">展示字段：datetime / strategy_nav / buyhold_nav / position / excess / excess_pct / action</div>
      </div>
      <div class="table-wrap">
        <table id="rawTable"></table>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    const ETFS = Object.keys(DATA.detailed);
    const state = {{
      etf: ETFS[0],
      chart: 'nav',
      range: 'all',
      zoomMode: 'all',
      zoomStart: null,
      zoomEnd: null,
      rows: 100,
    }};

    const etfSelect = document.getElementById('etfSelect');
    const chartSelect = document.getElementById('chartSelect');
    const rowsSelect = document.getElementById('rowsSelect');
    const metricGrid = document.getElementById('metricGrid');
    const comparisonTable = document.getElementById('comparisonTable');
    const overlayTable = document.getElementById('overlayTable');
    const reviewBox = document.getElementById('reviewBox');
    const chartSvg = document.getElementById('chartSvg');
    const legend = document.getElementById('legend');
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
    const STORAGE_KEY = 'soxl-dashboard-state-v2';

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
      }} catch (_err) {{
      }}
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
      if (start > end) {{
        const tmp = start;
        start = end;
        end = tmp;
      }}
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
      chartSelect.addEventListener('change', e => {{
        state.chart = e.target.value;
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

    function scale(values, minPx, maxPx) {{
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;
      return values.map(v => minPx + ((v - min) / span) * (maxPx - minPx));
    }}

    function polyline(points) {{
      return points.map(p => `${{p[0].toFixed(2)}},${{p[1].toFixed(2)}}`).join(' ');
    }}

    function renderChart() {{
      const rows = getFilteredRows(getDetailed(state.etf).rows);
      const width = 980;
      const height = 320;
      const left = 54;
      const right = 22;
      const top = 18;
      const bottom = 28;
      const x = rows.map((_, idx) => left + (idx / Math.max(rows.length - 1, 1)) * (width - left - right));
      const dates = rows.map(r => new Date(r.datetime));

      let series = [];
      let tradeDots = [];
      if (state.chart === 'nav') {{
        const strategy = rows.map(r => r.strategy_nav);
        const buyhold = rows.map(r => r.buyhold_nav);
        const ys = scale([...strategy, ...buyhold], height - bottom, top);
        series = [
          {{ name: 'Strategy NAV', color: '#9a3412', values: strategy, y: ys.slice(0, strategy.length) }},
          {{ name: 'Buy&Hold NAV', color: '#0f766e', values: buyhold, y: ys.slice(strategy.length) }},
        ];
      }} else if (state.chart === 'excess') {{
        const excess = rows.map(r => r.excess_pct);
        series = [
          {{ name: 'Excess %', color: '#1b1a17', values: excess, y: scale(excess, height - bottom, top) }},
        ];
      }} else if (state.chart === 'position') {{
        const position = rows.map(r => r.position);
        series = [
          {{ name: 'Position', color: '#1b1a17', values: position, y: scale(position, height - bottom, top) }},
        ];
        tradeDots = rows
          .map((r, idx) => (r.action === 'BUY' || r.action === 'SELL') ? {{
            x: x[idx],
            y: scale(position, height - bottom, top)[idx],
            color: r.action === 'BUY' ? '#0f766e' : '#b91c1c',
            action: r.action,
          }} : null)
          .filter(Boolean);
      }} else {{
        const strategyDD = [];
        const bhDD = [];
        let peakA = -Infinity;
        let peakB = -Infinity;
        rows.forEach(r => {{
          peakA = Math.max(peakA, r.strategy_nav);
          peakB = Math.max(peakB, r.buyhold_nav);
          strategyDD.push(r.strategy_nav / peakA - 1);
          bhDD.push(r.buyhold_nav / peakB - 1);
        }});
        const ys = scale([...strategyDD, ...bhDD], top, height - bottom).map(v => height - v);
        series = [
          {{ name: 'Strategy DD', color: '#9a3412', values: strategyDD, y: ys.slice(0, strategyDD.length) }},
          {{ name: 'Buy&Hold DD', color: '#0f766e', values: bhDD, y: ys.slice(strategyDD.length) }},
        ];
      }}

      const gridYs = [top, top + (height - top - bottom) / 2, height - bottom];
      const grid = gridYs.map(y => `<line x1="${{left}}" y1="${{y}}" x2="${{width-right}}" y2="${{y}}" stroke="#d8ccbb" stroke-dasharray="4 6"/>`).join('');
      const yearMap = new Map();
      dates.forEach((d, idx) => {{
        const key = d.getUTCFullYear();
        if (!yearMap.has(key)) yearMap.set(key, idx);
      }});
      const tickEntries = Array.from(yearMap.entries());
      const axis = tickEntries.map(([year, idx]) => {{
        const xpos = x[idx];
        return `
          <line x1="${{xpos}}" y1="${{height-bottom}}" x2="${{xpos}}" y2="${{height-bottom+6}}" stroke="#9f907e"/>
          <text x="${{xpos}}" y="${{height-bottom+18}}" text-anchor="middle" font-size="12" fill="#6d6458">${{year}}</text>
        `;
      }}).join('');
      const paths = series.map(s => {{
        const pts = s.values.map((_, i) => [x[i], s.y[i]]);
        return `<polyline fill="none" stroke="${{s.color}}" stroke-width="2.5" points="${{polyline(pts)}}"/>`;
      }}).join('');
      const dots = tradeDots.map(dot => `<circle cx="${{dot.x}}" cy="${{dot.y}}" r="4.5" fill="${{dot.color}}"><title>${{dot.action}}</title></circle>`).join('');
      chartSvg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="transparent"/>
        ${{grid}}
        <line x1="${{left}}" y1="${{height-bottom}}" x2="${{width-right}}" y2="${{height-bottom}}" stroke="#9f907e"/>
        ${{axis}}
        ${{paths}}
        ${{dots}}
      `;
      legend.innerHTML = series.map(s => `<span><span class="dot" style="background:${{s.color}}"></span>${{s.name}}</span>`).join('');
      if (tradeDots.length) {{
        legend.innerHTML += `<span><span class="dot" style="background:#0f766e"></span>BUY</span><span><span class="dot" style="background:#b91c1c"></span>SELL</span>`;
      }}
      if (rows.length) {{
        chartMeta.innerHTML = `
          <span>Zoom: <span class="mono">${{state.zoomMode}}</span></span>
          <span>Window: <span class="mono">${{rows[0].datetime}}</span> → <span class="mono">${{rows[rows.length-1].datetime}}</span></span>
          <span>Rows: <span class="mono">${{rows.length}}</span></span>
          <span>Chart: <span class="mono">${{state.chart}}</span></span>
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
      chartSelect.value = state.chart;
      rowsSelect.value = String(state.rows);
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
    OUTPUT_PATH.write_text(build_html(payload), encoding="utf-8")
    print(f"[done] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
