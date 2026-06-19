from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def _options(values: list[str]) -> str:
    return "\n".join(f'<option value="{escape(value)}">{escape(value)}</option>' for value in values)


def _metric(value: Any, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    return f"{number:.{digits}f}"


def _pill(text: str, *, kind: str = "") -> str:
    classes = "pill" + (f" {kind}" if kind else "")
    return f'<span class="{classes}">{escape(text)}</span>'


def _attribution_list(features: list[dict[str, Any]]) -> str:
    if not features:
        return '<li class="muted">暂无归因特征</li>'
    rows = []
    for row in features[:5]:
        rows.append(
            "<li>"
            f"<span>{escape(str(row.get('feature', '')))}</span>"
            f"<strong>{_metric(row.get('normalized_importance', 0.0), digits=3)}</strong>"
            "</li>"
        )
    return "\n".join(rows)


def _segment_rows(rows: list[dict[str, Any]], *, include_window: bool) -> str:
    if not rows:
        return '<tr><td colspan="5" class="muted">暂无分段数据</td></tr>'
    html_rows = []
    for row in rows[:8]:
        segment = f"{row.get('window', '')} / {row.get('regime', '')}" if include_window else row.get("regime", "")
        html_rows.append(
            "<tr "
            f'data-regime="{escape(str(row.get("regime", "")))}" '
            f'data-window="{escape(str(row.get("window", "")))}">'
            f"<td>{escape(str(segment))}</td>"
            f"<td>{_metric(row.get('return_pct', 0.0))}</td>"
            f"<td>{_metric(row.get('relative_return_pct', 0.0))}</td>"
            f"<td>{_metric(row.get('max_drawdown_pct', 0.0))}</td>"
            f"<td>{_metric(row.get('stability_score', 0.0))}</td>"
            "</tr>"
        )
    return "\n".join(html_rows)


def _diagnostics(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<li class="muted">暂无诊断</li>'
    return "\n".join(
        f"<li>{escape(str(row.get('segment_type', '')))} / {escape(str(row.get('segment', '')))}: "
        f"{escape(str(row.get('diagnosis', '')))}</li>"
        for row in rows[:4]
    )


def _failure_modes(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<li class="muted">暂无失败模式</li>'
    return "\n".join(
        "<li "
        f'data-failure-mode="{escape(str(row.get("failure_mode", "")))}" '
        f'data-segment="{escape(str(row.get("segment", "")))}">'
        f"<strong>{escape(str(row.get('failure_mode', '')))}</strong>"
        f"<span>{escape(str(row.get('segment', '')))} / severity {_metric(row.get('severity_score', 0.0))}</span>"
        f"<p>{escape(str(row.get('review_note', '')))}</p>"
        "</li>"
        for row in rows[:4]
    )


def _review_briefing(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<li class="muted">暂无 Review Briefing</li>'
    return "\n".join(
        "<li "
        f'data-review-signal="{escape(str(row.get("review_signal", "")))}" '
        f'data-decision-hint="{escape(str(row.get("decision_hint", "")))}" '
        f'data-segment="{escape(str(row.get("segment", "")))}">'
        f"<strong>{escape(str(row.get('review_signal', '')))} / {escape(str(row.get('decision_hint', '')))}</strong>"
        f"<span>{escape(str(row.get('segment', '')))} / priority {escape(str(row.get('priority_rank', '')))}</span>"
        f"<p>{escape(str(row.get('review_note', '')))}</p>"
        "</li>"
        for row in rows[:4]
    )


def _model_difference_explanations(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<li class="muted">暂无模型差异解释</li>'
    return "\n".join(
        "<li "
        f'data-explanation-signal="{escape(str(row.get("explanation_signal", "")))}" '
        f'data-difference-decision="{escape(str(row.get("decision_hint", "")))}" '
        f'data-segment="{escape(str(row.get("segment", "")))}">'
        f"<strong>{escape(str(row.get('explanation_signal', '')))} / {escape(str(row.get('decision_hint', '')))}</strong>"
        f"<span>{escape(str(row.get('segment', '')))} / "
        f"{escape(str(row.get('loser_model', '')))} -> {escape(str(row.get('winner_model', '')))} / "
        f"spread {_metric(row.get('relative_return_pct_spread', 0.0))}</span>"
        f"<p>{escape(str(row.get('review_note', '')))}</p>"
        "</li>"
        for row in rows[:6]
    )


def _model_difference_panel(view_model: dict[str, Any]) -> str:
    rows = (
        view_model.get("model_difference_explanations", [])
        if isinstance(view_model.get("model_difference_explanations"), list)
        else []
    )
    summary = (
        view_model.get("model_difference_summary", {})
        if isinstance(view_model.get("model_difference_summary"), dict)
        else {}
    )
    if not rows and not summary:
        return ""
    return f"""
    <section class="model-difference-overview" aria-label="model difference explanations">
      <header>
        <h2>Model Difference Explanations</h2>
        <p>winner / loser、regime-window 差异和 review action</p>
      </header>
      <div class="difference-grid">
        <div><span>解释数</span><strong>{escape(str(summary.get('count', len(rows))))}</strong></div>
        <div><span>Top Signal</span><strong>{escape(str(summary.get('top_explanation_signal', '')))}</strong></div>
        <div><span>Top Decision</span><strong>{escape(str(summary.get('top_decision_hint', '')))}</strong></div>
        <div><span>Top Segment</span><strong>{escape(str(summary.get('top_segment', '')))}</strong><p>{escape(str(summary.get('top_loser_model', '')))} -> {escape(str(summary.get('top_winner_model', '')))}</p></div>
      </div>
      <ul class="model-difference-list">{_model_difference_explanations(rows)}</ul>
    </section>
    """


def _analysis_overview_panel(view_model: dict[str, Any]) -> str:
    overview = view_model.get("analysis_overview", {}) if isinstance(view_model.get("analysis_overview"), dict) else {}
    if not overview:
        return ""
    gaps = overview.get("largest_relative_return_spread_segments")
    top_gap = gaps[0] if isinstance(gaps, list) and gaps and isinstance(gaps[0], dict) else {}
    weakness = overview.get("top_weakness")
    top_weakness = weakness if isinstance(weakness, dict) else {}
    gap_flow = ""
    if top_gap:
        gap_flow = (
            f"{escape(str(top_gap.get('worst_relative_return_model', '')))} -> "
            f"{escape(str(top_gap.get('best_relative_return_model', '')))}"
        )
    return f"""
    <section class="analysis-overview" aria-label="analysis overview">
      <header>
        <h2>Analysis Overview</h2>
        <p>{escape(str(overview.get('dashboard_note', '')))}</p>
      </header>
      <div class="overview-grid">
        <div><span>基准模型</span><strong>{escape(str(overview.get('baseline_model', '')))}</strong></div>
        <div><span>Regime / Window</span><strong>{escape(str(overview.get('regime_count', 0)))} / {escape(str(overview.get('window_count', 0)))}</strong></div>
        <div><span>最大差异</span><strong>{escape(str(top_gap.get('segment', '')))}</strong><p>{gap_flow}</p></div>
        <div><span>Spread</span><strong>{_metric(top_gap.get('relative_return_pct_spread', 0.0))}</strong></div>
        <div><span>Top Weakness</span><strong>{escape(str(top_weakness.get('review_signal', '')))} / {escape(str(top_weakness.get('decision_hint', '')))}</strong><p>{escape(str(top_weakness.get('model', '')))} @ {escape(str(top_weakness.get('segment', '')))}</p></div>
      </div>
    </section>
    """


def _production_health_panel(view_model: dict[str, Any]) -> str:
    health = view_model.get("production_health", {}) if isinstance(view_model.get("production_health"), dict) else {}
    if not health:
        return ""
    missing = health.get("missing_required_paths")
    missing_count = len(missing) if isinstance(missing, list) else 0
    live_enabled = bool(health.get("live_broker_enabled", False))
    live_text = "live broker on" if live_enabled else "live broker off"
    health_kind = "healthy" if bool(health.get("is_healthy", False)) else "warn"
    paper_execution_schema = str(health.get("paper_execution_pipeline_schema_version", ""))
    paper_execution_status = str(health.get("paper_execution_job_status", ""))
    return f"""
    <section class="production-health" aria-label="production health">
      <header>
        <h2>Production Health</h2>
        <p>{escape(str(health.get('strategy_id', '')))}</p>
      </header>
      <div class="health-grid">
        <div><span>Health</span><strong>{_pill(str(health.get('health_status', 'unknown')), kind=health_kind)}</strong></div>
        <div><span>Mode</span><strong>{escape(str(health.get('mode', '')))}</strong><p>{escape(live_text)}</p></div>
        <div><span>Required Paths</span><strong>{escape(str(health.get('checked_path_count', 0)))}</strong><p>missing {missing_count}</p></div>
        <div><span>Paper Execution</span><strong>{escape(paper_execution_status or 'unknown')}</strong><p>{escape(paper_execution_schema)}</p></div>
        <div><span>Manifest</span><strong>{escape(str(health.get('manifest_hash', ''))[:10])}</strong><p>{escape(str(health.get('published_at', '')))}</p></div>
      </div>
    </section>
    """


def _readiness_overview_panel(view_model: dict[str, Any]) -> str:
    overview = (
        view_model.get("readiness_overview", {})
        if isinstance(view_model.get("readiness_overview"), dict)
        else {}
    )
    if not overview:
        return ""
    gates = overview.get("gates")
    gate_rows = gates if isinstance(gates, list) else []
    rows = "\n".join(
        "<li>"
        f"<strong>{escape(str(gate.get('label', '')))}</strong>"
        f"{_pill(str(gate.get('status', 'unknown')), kind=str(gate.get('status', '')))}"
        f"<span>{escape(str(gate.get('detail', '')))}</span>"
        "</li>"
        for gate in gate_rows
        if isinstance(gate, dict)
    )
    if not rows:
        rows = '<li class="muted">暂无 readiness gate</li>'
    return f"""
    <section class="readiness-overview" aria-label="readiness overview">
      <header>
        <h2>Readiness Overview</h2>
        <p>{escape(str(overview.get('schema_version', '')))}</p>
      </header>
      <div class="readiness-grid">
        <div><span>Overall</span><strong>{_pill(str(overview.get('status', 'unknown')), kind=str(overview.get('status', '')))}</strong></div>
        <div><span>Pass</span><strong>{escape(str(overview.get('pass_count', 0)))}</strong></div>
        <div><span>Review</span><strong>{escape(str(overview.get('review_count', 0)))}</strong></div>
        <div><span>Block</span><strong>{escape(str(overview.get('block_count', 0)))}</strong></div>
      </div>
      <ul class="readiness-gates">{rows}</ul>
    </section>
    """


def _notebook_card_readiness_panel(view_model: dict[str, Any]) -> str:
    readiness = (
        view_model.get("notebook_card_readiness", {})
        if isinstance(view_model.get("notebook_card_readiness"), dict)
        else {}
    )
    if not readiness:
        return ""
    summary = readiness.get("summary")
    summary_payload = summary if isinstance(summary, dict) else {}
    production_cards = readiness.get("production_cards")
    production_card_rows = production_cards if isinstance(production_cards, list) else []
    if not summary_payload and not production_card_rows:
        return ""
    card_rows = "\n".join(
        "<li>"
        f"<span>{escape(str(card.get('card_id', '')))}</span>"
        f"{_pill(str(card.get('production_status', 'unknown')), kind='native' if str(card.get('production_status', '')) == 'published' else 'warn')}"
        "</li>"
        for card in production_card_rows[:6]
        if isinstance(card, dict)
    )
    if not card_rows:
        card_rows = '<li class="muted">暂无 production notebook card</li>'
    return f"""
    <section class="notebook-readiness" aria-label="notebook card readiness">
      <header>
        <h2>Notebook Readiness</h2>
        <p>{escape(str(readiness.get('schema_version', '')))}</p>
      </header>
      <div class="notebook-grid">
        <div><span>Notebook Cards</span><strong>{escape(str(summary_payload.get('card_count', 0)))}</strong><p>production {escape(str(summary_payload.get('production_card_count', 0)))}</p></div>
        <div><span>Production Status</span><strong>published {escape(str(summary_payload.get('production_published_count', 0)))}</strong><p>template {escape(str(summary_payload.get('production_template_count', 0)))}</p></div>
        <div><span>Production Gaps</span><strong>missing {escape(str(summary_payload.get('production_manifest_missing_count', 0)))}</strong><p>registry mismatch {escape(str(summary_payload.get('production_registry_mismatch_count', 0)))}</p></div>
        <div><span>Manifest</span><strong>{escape(str(readiness.get('manifest_hash', ''))[:10])}</strong><p>{escape(str(readiness.get('source_schema_version', '')))}</p></div>
      </div>
      <ul class="notebook-card-list">{card_rows}</ul>
    </section>
    """


def _schedule_health_panel(view_model: dict[str, Any]) -> str:
    health = view_model.get("schedule_health", {}) if isinstance(view_model.get("schedule_health"), dict) else {}
    if not health:
        return ""
    jobs = health.get("jobs")
    job_rows = jobs if isinstance(jobs, list) else []
    rendered_rows = []
    for job in job_rows[:8]:
        if not isinstance(job, dict):
            continue
        last_error_type = str(job.get("last_error_type", ""))
        last_error_message = str(job.get("last_error_message", ""))
        error_text = f"{last_error_type}:{last_error_message}" if last_error_type or last_error_message else "no retry error"
        rendered_rows.append(
            "<li>"
            f"<strong>{escape(str(job.get('name', '')))}</strong>"
            f"{_pill(str(job.get('status', 'unknown')), kind=str(job.get('status', '')))}"
            f"<span>{escape(str(job.get('status', '')))} / attempts {escape(str(job.get('attempts', 0)))} / retry_errors {escape(str(job.get('retry_error_count', 0)))}</span>"
            f"<p>{escape(error_text)}</p>"
            "</li>"
        )
    rows = "\n".join(rendered_rows)
    if not rows:
        rows = '<li class="muted">暂无 schedule job</li>'
    return f"""
    <section class="schedule-health" aria-label="schedule health">
      <header>
        <h2>Schedule Health</h2>
        <p>{escape(str(health.get('schema_version', '')))}</p>
      </header>
      <div class="schedule-grid">
        <div><span>Jobs</span><strong>{escape(str(health.get('job_count', 0)))}</strong></div>
        <div><span>Failed</span><strong>{escape(str(health.get('failed_job_count', 0)))}</strong></div>
        <div><span>Attempts</span><strong>{escape(str(health.get('total_attempts', 0)))}</strong></div>
        <div><span>Retry Errors</span><strong>{escape(str(health.get('total_retry_error_count', 0)))}</strong><p>{escape(str(health.get('top_failure_last_error', '')))}</p></div>
      </div>
      <ul class="schedule-jobs">{rows}</ul>
    </section>
    """


def _objective_loss_panel(view_model: dict[str, Any]) -> str:
    summary = (
        view_model.get("objective_loss_summary", {})
        if isinstance(view_model.get("objective_loss_summary"), dict)
        else {}
    )
    if not summary:
        return ""
    components = summary.get("loss_components")
    component_rows = components if isinstance(components, list) else []
    rows = "\n".join(
        "<li>"
        f"<span>{escape(str(component.get('name', '')))}</span>"
        f"<strong>{_metric(component.get('value', 0.0), digits=4)}</strong>"
        "</li>"
        for component in component_rows
        if isinstance(component, dict)
    )
    if not rows:
        rows = '<li class="muted">暂无 objective loss component</li>'
    top = summary.get("top_loss_component") if isinstance(summary.get("top_loss_component"), dict) else {}
    return f"""
    <section class="objective-loss" aria-label="objective loss summary">
      <header>
        <h2>Objective Loss</h2>
        <p>{escape(str(summary.get('objective_loss_schema_version', '')))}</p>
      </header>
      <div class="objective-loss-grid">
        <div><span>Direction</span><strong>{escape(str(summary.get('optimization_direction', '')))}</strong></div>
        <div><span>Reward Surface</span><strong>{escape(str(summary.get('reward_surface', '')))}</strong></div>
        <div><span>Rows</span><strong>{escape(str(summary.get('row_count', 0)))}</strong><p>rank groups {escape(str(summary.get('rank_group_count', 0)))}</p></div>
        <div><span>Top Loss</span><strong>{escape(str(top.get('name', '')))}</strong><p>{_metric(top.get('value', 0.0), digits=4)}</p></div>
      </div>
      <ul class="objective-loss-components">{rows}</ul>
    </section>
    """


def _model_card(card: dict[str, Any]) -> str:
    model = str(card.get("model", ""))
    fidelity = str(card.get("paper_fidelity", ""))
    proxy_class = " proxy" if card.get("proxy_model") else ""
    attribution_features = sorted(
        {
            str(row.get("feature", ""))
            for row in card.get("top_attribution_features", [])
            if isinstance(row, dict) and row.get("feature")
        }
    )
    failure_modes = sorted(
        {
            str(row.get("failure_mode", ""))
            for row in card.get("failure_modes", [])
            if isinstance(row, dict) and row.get("failure_mode")
        }
    )
    review_signals = sorted(
        {
            str(row.get("review_signal", ""))
            for row in card.get("model_review_briefing", [])
            if isinstance(row, dict) and row.get("review_signal")
        }
    )
    decision_hints = sorted(
        {
            str(row.get("decision_hint", ""))
            for row in card.get("model_review_briefing", [])
            if isinstance(row, dict) and row.get("decision_hint")
        }
    )
    explanation_signals = sorted(
        {
            str(row.get("explanation_signal", ""))
            for row in card.get("model_difference_explanations", [])
            if isinstance(row, dict) and row.get("explanation_signal")
        }
    )
    difference_decisions = sorted(
        {
            str(row.get("decision_hint", ""))
            for row in card.get("model_difference_explanations", [])
            if isinstance(row, dict) and row.get("decision_hint")
        }
    )
    return f"""
      <article class="model-card{proxy_class}" data-model="{escape(model)}" data-base-model="{escape(str(card.get('base_model', '')))}" data-fidelity="{escape(fidelity)}" data-attribution-features="{escape(','.join(attribution_features))}" data-failure-modes="{escape(','.join(failure_modes))}" data-review-signals="{escape(','.join(review_signals))}" data-decision-hints="{escape(','.join(decision_hints))}" data-explanation-signals="{escape(','.join(explanation_signals))}" data-difference-decisions="{escape(','.join(difference_decisions))}">
        <header>
          <div>
            <h2>{escape(model)}</h2>
            <p>{escape(str(card.get('base_model', '')))} / {escape(str(card.get('feature_set', '')))}</p>
          </div>
          <div class="badges">
            {_pill(fidelity or 'unknown_fidelity', kind='fidelity')}
            {_pill('proxy' if card.get('proxy_model') else 'native', kind='proxy' if card.get('proxy_model') else 'native')}
          </div>
        </header>
        <section class="metrics">
          <div><span>收益</span><strong>{_metric(card.get('return_pct', 0.0))}%</strong></div>
          <div><span>相对收益</span><strong>{_metric(card.get('relative_return_pct', 0.0))}%</strong></div>
          <div><span>回撤</span><strong>{_metric(card.get('max_drawdown_pct', 0.0))}%</strong></div>
          <div><span>稳定性</span><strong>{_metric(card.get('stability_score', 0.0))}</strong></div>
        </section>
        <section class="split">
          <div>
            <h3>归因特征</h3>
            <ul class="feature-list">{_attribution_list(card.get('top_attribution_features', []))}</ul>
          </div>
          <div>
            <h3>诊断</h3>
            <ul class="diagnostics">{_diagnostics(card.get('diagnostics', []))}</ul>
          </div>
        </section>
        <section>
          <h3>Failure Modes</h3>
          <ul class="failure-modes">{_failure_modes(card.get('failure_modes', []))}</ul>
        </section>
        <section>
          <h3>Model Review Briefing</h3>
          <ul class="review-briefing">{_review_briefing(card.get('model_review_briefing', []))}</ul>
        </section>
        <section>
          <h3>Model Difference Explanations</h3>
          <ul class="model-difference-list">{_model_difference_explanations(card.get('model_difference_explanations', []))}</ul>
        </section>
        <section>
          <h3>Regime</h3>
          <table>
            <thead><tr><th>状态</th><th>收益%</th><th>相对%</th><th>回撤%</th><th>稳定性</th></tr></thead>
            <tbody>{_segment_rows(card.get('regime_metrics', []), include_window=False)}</tbody>
          </table>
        </section>
        <section>
          <h3>Window-Regime</h3>
          <table class="window-regime">
            <thead><tr><th>窗口 / 状态</th><th>收益%</th><th>相对%</th><th>回撤%</th><th>稳定性</th></tr></thead>
            <tbody>{_segment_rows(card.get('window_regime_metrics', []), include_window=True)}</tbody>
          </table>
        </section>
      </article>
    """


def _style() -> str:
    return """
    :root { color-scheme: light; --bg:#f7f8fa; --panel:#ffffff; --ink:#1f2933; --muted:#667085; --line:#d8dee8; --accent:#1b6b93; --warn:#8a4b0f; --good:#126b45; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
    main { max-width: 1440px; margin: 0 auto; padding: 24px; }
    header.page { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 18px; }
    h1 { margin: 0 0 6px; font-size: 28px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 14px; letter-spacing: 0; }
    p { margin: 0; color: var(--muted); }
    .toolbar { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; padding: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin-bottom: 16px; }
    label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
    select { width: 100%; min-height: 34px; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 6px 8px; }
    .analysis-overview { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .analysis-overview header { margin-bottom: 10px; }
    .readiness-overview { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .readiness-overview header { margin-bottom: 10px; }
    .production-health { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .production-health header { margin-bottom: 10px; }
    .notebook-readiness { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .notebook-readiness header { margin-bottom: 10px; }
    .schedule-health { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .schedule-health header { margin-bottom: 10px; }
    .objective-loss { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .objective-loss header { margin-bottom: 10px; }
    .model-difference-overview { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    .model-difference-overview header { margin-bottom: 10px; }
    .health-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; }
    .health-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .health-grid span { color: var(--muted); font-size: 12px; }
    .health-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .health-grid p { margin-top: 3px; font-size: 12px; overflow-wrap: anywhere; }
    .notebook-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; }
    .notebook-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .notebook-grid span { color: var(--muted); font-size: 12px; }
    .notebook-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .notebook-grid p { margin-top: 3px; font-size: 12px; overflow-wrap: anywhere; }
    .notebook-card-list { display: flex; flex-wrap: wrap; gap: 8px; list-style: none; padding-left: 0; margin-top: 10px; }
    .notebook-card-list li { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; font-size: 12px; }
    .schedule-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 10px; }
    .schedule-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .schedule-grid span { color: var(--muted); font-size: 12px; }
    .schedule-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .schedule-grid p { margin-top: 3px; font-size: 12px; overflow-wrap: anywhere; }
    .schedule-jobs { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; list-style: none; padding-left: 0; margin: 0; }
    .schedule-jobs li { display: grid; grid-template-columns: 1fr auto; gap: 4px 8px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .schedule-jobs span, .schedule-jobs p { grid-column: 1 / -1; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; margin: 0; }
    .objective-loss-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 10px; }
    .objective-loss-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .objective-loss-grid span { color: var(--muted); font-size: 12px; }
    .objective-loss-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .objective-loss-grid p { margin-top: 3px; font-size: 12px; overflow-wrap: anywhere; }
    .objective-loss-components { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 8px; list-style: none; padding-left: 0; margin: 0; }
    .objective-loss-components li { display: flex; justify-content: space-between; gap: 10px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .objective-loss-components span { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .objective-loss-components strong { font-size: 13px; }
    .overview-grid { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; }
    .overview-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .overview-grid span { color: var(--muted); font-size: 12px; }
    .overview-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .overview-grid p { margin-top: 3px; font-size: 12px; overflow-wrap: anywhere; }
    .difference-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 10px; }
    .difference-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .difference-grid span { color: var(--muted); font-size: 12px; }
    .difference-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .difference-grid p { margin-top: 3px; font-size: 12px; overflow-wrap: anywhere; }
    .readiness-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 10px; }
    .readiness-grid div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .readiness-grid span { color: var(--muted); font-size: 12px; }
    .readiness-grid strong { display: block; margin-top: 3px; font-size: 15px; overflow-wrap: anywhere; }
    .readiness-gates { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; list-style: none; padding-left: 0; margin: 0; }
    .readiness-gates li { display: grid; grid-template-columns: 1fr auto; gap: 4px 8px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .readiness-gates span { grid-column: 1 / -1; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .summary div, .model-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .summary div { padding: 12px; }
    .summary span, .metrics span { color: var(--muted); font-size: 12px; }
    .summary strong, .metrics strong { display: block; margin-top: 3px; font-size: 18px; }
    .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }
    .model-card { padding: 14px; }
    .model-card > header { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
    .badges { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
    .pill { display: inline-flex; align-items: center; min-height: 24px; border-radius: 999px; padding: 3px 8px; font-size: 12px; background: #eef2f7; color: var(--ink); }
    .pill.proxy { background: #fff4df; color: var(--warn); }
    .pill.native { background: #e8f5ee; color: var(--good); }
    .pill.fidelity { background: #e7f2f8; color: var(--accent); }
    .pill.healthy { background: #e8f5ee; color: var(--good); }
    .pill.warn { background: #fff4df; color: var(--warn); }
    .pill.pass, .pill.ready { background: #e8f5ee; color: var(--good); }
    .pill.review { background: #fff4df; color: var(--warn); }
    .pill.block, .pill.blocked { background: #fdecec; color: #9f1f1f; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; }
    .metrics div { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
    .split { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    ul { margin: 0; padding-left: 18px; }
    .feature-list li { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 5px; }
    .failure-modes { display: grid; gap: 8px; padding-left: 0; list-style: none; margin-bottom: 12px; }
    .review-briefing { display: grid; gap: 8px; padding-left: 0; list-style: none; margin-bottom: 12px; }
    .model-difference-list { display: grid; gap: 8px; padding-left: 0; list-style: none; margin-bottom: 12px; }
    .failure-modes li, .review-briefing li, .model-difference-list li { border: 1px solid var(--line); border-left: 3px solid var(--warn); border-radius: 6px; padding: 8px; }
    .review-briefing li { border-left-color: var(--accent); }
    .model-difference-list li { border-left-color: var(--accent); }
    .failure-modes strong, .review-briefing strong, .model-difference-list strong { display: block; font-size: 13px; }
    .failure-modes span, .review-briefing span, .model-difference-list span { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
    .failure-modes p, .review-briefing p, .model-difference-list p { margin-top: 5px; font-size: 12px; }
    .muted { color: var(--muted); }
    table { width: 100%; border-collapse: collapse; margin-bottom: 12px; table-layout: fixed; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: right; font-size: 12px; }
    th:first-child, td:first-child { text-align: left; }
    @media (max-width: 840px) { main { padding: 14px; } .toolbar, .analysis-overview .overview-grid, .readiness-overview .readiness-grid, .readiness-overview .readiness-gates, .production-health .health-grid, .notebook-readiness .notebook-grid, .schedule-health .schedule-grid, .schedule-health .schedule-jobs, .objective-loss .objective-loss-grid, .objective-loss .objective-loss-components, .model-difference-overview .difference-grid, .summary, .model-grid, .split { grid-template-columns: 1fr; } .metrics { grid-template-columns: repeat(2, 1fr); } }
    """


def _script() -> str:
    return """
    const filters = ["model", "regime", "window", "fidelity", "attribution", "failure-mode", "review-signal", "decision-hint", "explanation-signal", "difference-decision"];
    function selected(id) {
      const node = document.getElementById(id + "-filter");
      return node ? node.value : "";
    }
    function containsValue(csv, value) {
      if (!value) return true;
      return (csv || "").split(",").filter(Boolean).includes(value);
    }
    function applyFilters() {
      const model = selected("model");
      const fidelity = selected("fidelity");
      const regime = selected("regime");
      const windowName = selected("window");
      const attribution = selected("attribution");
      const failureMode = selected("failure-mode");
      const reviewSignal = selected("review-signal");
      const decisionHint = selected("decision-hint");
      const explanationSignal = selected("explanation-signal");
      const differenceDecision = selected("difference-decision");
      document.querySelectorAll(".model-card").forEach(card => {
        const modelOk = !model || card.dataset.model === model;
        const fidelityOk = !fidelity || card.dataset.fidelity === fidelity;
        const regimeOk = !regime || card.querySelector(`[data-regime="${regime}"]`);
        const windowOk = !windowName || card.querySelector(`[data-window="${windowName}"]`);
        const attributionOk = containsValue(card.dataset.attributionFeatures, attribution);
        const failureModeOk = containsValue(card.dataset.failureModes, failureMode);
        const reviewSignalOk = containsValue(card.dataset.reviewSignals, reviewSignal);
        const decisionHintOk = containsValue(card.dataset.decisionHints, decisionHint);
        const explanationSignalOk = containsValue(card.dataset.explanationSignals, explanationSignal);
        const differenceDecisionOk = containsValue(card.dataset.differenceDecisions, differenceDecision);
        card.hidden = !(modelOk && fidelityOk && regimeOk && windowOk && attributionOk && failureModeOk && reviewSignalOk && decisionHintOk && explanationSignalOk && differenceDecisionOk);
      });
    }
    filters.forEach(id => {
      const node = document.getElementById(id + "-filter");
      if (node) node.addEventListener("change", applyFilters);
    });
    applyFilters();
    """


def render_dashboard_html(view_model: dict[str, Any], *, title: str = "SOXL Dashboard") -> str:
    controls = view_model.get("controls", {}) if isinstance(view_model.get("controls"), dict) else {}
    model_cards = view_model.get("model_cards", []) if isinstance(view_model.get("model_cards"), list) else []
    summary = view_model.get("summary", {}) if isinstance(view_model.get("summary"), dict) else {}
    fidelity = view_model.get("model_fidelity", {}) if isinstance(view_model.get("model_fidelity"), dict) else {}
    requested_contract = (
        view_model.get("requested_model_contract", {})
        if isinstance(view_model.get("requested_model_contract"), dict)
        else {}
    )
    missing_requested = requested_contract.get("missing_from_ledger", [])
    missing_requested_count = len(missing_requested) if isinstance(missing_requested, list) else 0
    requested_contract_card = (
        "<div><span>Requested 模型</span>"
        f"<strong>{escape(str(requested_contract.get('completed_requested_model_count', 0)))} / "
        f"{escape(str(requested_contract.get('requested_model_count', 0)))}</strong>"
        f"<p>missing {escape(str(missing_requested_count))}</p></div>"
        if requested_contract
        else ""
    )
    embedded = json.dumps(view_model, ensure_ascii=False, indent=2).replace("</", "<\\/")
    overview_panel = _analysis_overview_panel(view_model)
    model_difference_panel = _model_difference_panel(view_model)
    readiness_overview_panel = _readiness_overview_panel(view_model)
    production_health_panel = _production_health_panel(view_model)
    notebook_readiness_panel = _notebook_card_readiness_panel(view_model)
    schedule_health_panel = _schedule_health_panel(view_model)
    objective_loss_panel = _objective_loss_panel(view_model)
    cards = "\n".join(_model_card(card) for card in model_cards)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{_style()}</style>
</head>
<body>
  <main>
    <header class="page">
      <div>
        <h1>{escape(title)}</h1>
        <p>模型回测、regime 差异、fidelity 和 attribution 对比</p>
      </div>
      <p>{escape(str(view_model.get('source_artifact_id', '')))}</p>
    </header>
    <section class="toolbar" aria-label="dashboard filters">
      <label>模型<select id="model-filter"><option value="">全部</option>{_options(controls.get('models', []))}</select></label>
      <label>Regime<select id="regime-filter"><option value="">全部</option>{_options(controls.get('regimes', []))}</select></label>
      <label>Window<select id="window-filter"><option value="">全部</option>{_options(controls.get('windows', []))}</select></label>
      <label>Fidelity<select id="fidelity-filter"><option value="">全部</option>{_options(controls.get('paper_fidelities', []))}</select></label>
      <label>Attribution<select id="attribution-filter"><option value="">全部</option>{_options(controls.get('attribution_features', []))}</select></label>
      <label>Failure Mode<select id="failure-mode-filter"><option value="">全部</option>{_options(controls.get('failure_modes', []))}</select></label>
      <label>Review Signal<select id="review-signal-filter"><option value="">全部</option>{_options(controls.get('review_signals', []))}</select></label>
      <label>Decision<select id="decision-hint-filter"><option value="">全部</option>{_options(controls.get('decision_hints', []))}</select></label>
      <label>Difference Signal<select id="explanation-signal-filter"><option value="">全部</option>{_options(controls.get('explanation_signals', []))}</select></label>
      <label>Difference Decision<select id="difference-decision-filter"><option value="">全部</option>{_options(controls.get('difference_decisions', []))}</select></label>
    </section>
    {readiness_overview_panel}
    {overview_panel}
    {model_difference_panel}
    {schedule_health_panel}
    {objective_loss_panel}
    {production_health_panel}
    {notebook_readiness_panel}
    <section class="summary">
      <div><span>模型卡片</span><strong>{len(model_cards)}</strong></div>
      <div><span>基准模型</span><strong>{escape(str(summary.get('analysis_summary', {}).get('baseline_model', '')) if isinstance(summary.get('analysis_summary'), dict) else '')}</strong></div>
      <div><span>Proxy 模型</span><strong>{len(fidelity.get('proxy_models', [])) if isinstance(fidelity.get('proxy_models'), list) else 0}</strong></div>
      <div><span>Promotion Blocker</span><strong>{escape(str(fidelity.get('promotion_blocker_count', 0)))}</strong></div>
      {requested_contract_card}
    </section>
    <section class="model-grid">
      {cards}
    </section>
  </main>
  <script type="application/json" id="dashboard-view-model">{embedded}</script>
  <script>{_script()}</script>
</body>
</html>
"""
    return "\n".join(line.rstrip() for line in html.splitlines()) + "\n"


def write_dashboard_html(
    view_model: dict[str, Any],
    *,
    output_path: str | Path,
    title: str = "SOXL Dashboard",
) -> str:
    html = render_dashboard_html(view_model, title=title)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return html
