#!/usr/bin/env python3
"""Validate the static dashboard bundle used by Vercel deployments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.paper_review import collect_sensitive_public_keys


NO_CACHE_HEADER = {
    "key": "Cache-Control",
    "value": "public, max-age=0, must-revalidate",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _headers_by_source(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    headers: dict[str, list[dict[str, str]]] = {}
    for entry in config.get("headers", []):
        source = entry.get("source")
        if isinstance(source, str):
            headers[source] = entry.get("headers", [])
    return headers


def check_static_export(root: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    info: dict[str, Any] = {
        "generated_at": "unknown",
        "versions": 0,
        "missing": 0,
        "model_versions": 0,
        "latest": "unknown",
        "paper_trading_rows": "n/a",
        "paper_trading_end": "n/a",
        "paper_latest_signal": "n/a",
        "production_status": "n/a",
        "goal_gates": "n/a",
        "model_replays": "n/a",
        "model_latest_data": "n/a",
        "dashboard_v2": "n/a",
        "research_backtests": "n/a",
    }

    vercel_path = root / "vercel.json"
    dashboard_path = root / "dashboard.html"
    data_dir = root / "dashboard_data"
    index_path = data_dir / "index.json"
    model_versions_path = data_dir / "model_versions.json"
    production_status_path = data_dir / "production_status.json"
    production_goal_gates_path = data_dir / "production_goal_gates.json"
    model_replay_freshness_path = data_dir / "model_replay_freshness.json"
    research_backtests_path = data_dir / "research_backtests.json"
    dashboard_v2_manifest_path = data_dir / "v2" / "manifest.json"
    dashboard_v2_comparison_path = data_dir / "v2" / "comparison" / "default.json"

    for path in (
        vercel_path,
        dashboard_path,
        index_path,
        model_versions_path,
        model_replay_freshness_path,
        production_status_path,
        research_backtests_path,
    ):
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(root)}")

    if not vercel_path.exists() or not index_path.exists() or not model_versions_path.exists():
        return errors, info

    vercel_config = _load_json(vercel_path)
    if {"source": "/", "destination": "/dashboard.html"} not in vercel_config.get("rewrites", []):
        errors.append("vercel.json must rewrite / to /dashboard.html")

    headers = _headers_by_source(vercel_config)
    for source in ("/", "/dashboard.html", "/dashboard_data/:path*"):
        if NO_CACHE_HEADER not in headers.get(source, []):
            errors.append(f"missing no-cache header for {source}")

    index = _load_json(index_path)
    model_versions = _load_json(model_versions_path)
    versions = index.get("versions", [])
    version_summaries = index.get("version_summaries", {})
    etfs = index.get("etfs", [])
    info.update(
        {
            "generated_at": index.get("generated_at", "unknown"),
            "versions": len(versions),
            "model_versions": len(model_versions) if isinstance(model_versions, list) else 0,
            "latest": index.get("latest_version", "unknown"),
        }
    )

    if not versions:
        errors.append("dashboard_data/index.json has no versions")
    if "paper_trading" not in versions:
        errors.append("dashboard_data/index.json does not include paper_trading")
    elif index.get("latest_version") != "paper_trading":
        errors.append("latest_version must be paper_trading when paper_trading is exported")
    if index.get("latest_version") not in versions:
        errors.append("latest_version is not listed in versions")
    if index.get("best_version") and index.get("best_version") not in versions:
        errors.append("best_version is not listed in versions")
    if not isinstance(model_versions, list):
        errors.append("dashboard_data/model_versions.json must contain a list")

    if production_status_path.exists():
        production_status = _load_json(production_status_path)
        if not isinstance(production_status, dict):
            errors.append("dashboard_data/production_status.json must contain an object")
        else:
            status = production_status.get("status")
            if not isinstance(status, str) or not status:
                errors.append("dashboard_data/production_status.json missing status")
            else:
                info["production_status"] = status
            for key in ("auth_status", "project_status"):
                if not isinstance(production_status.get(key), str) or not production_status.get(key):
                    errors.append(f"dashboard_data/production_status.json missing {key}")
            status_static = production_status.get("static_export", {})
            if not isinstance(status_static, dict):
                errors.append("dashboard_data/production_status.json static_export must be an object")
            elif status_static.get("latest") != index.get("latest_version"):
                errors.append("dashboard_data/production_status.json latest does not match dashboard_data/index.json")

    if production_goal_gates_path.exists():
        goal_gates = _load_json(production_goal_gates_path)
        if not isinstance(goal_gates, dict):
            errors.append("dashboard_data/production_goal_gates.json must contain an object")
        else:
            overall = goal_gates.get("overall")
            gates = goal_gates.get("gates")
            if not isinstance(overall, dict):
                errors.append("dashboard_data/production_goal_gates.json overall must be an object")
            else:
                status = overall.get("status")
                if not isinstance(status, str) or not status:
                    errors.append("dashboard_data/production_goal_gates.json missing overall status")
                else:
                    info["goal_gates"] = status
            if not isinstance(gates, dict):
                errors.append("dashboard_data/production_goal_gates.json gates must be an object")

    if model_replay_freshness_path.exists():
        model_replays = _load_json(model_replay_freshness_path)
        if not isinstance(model_replays, dict):
            errors.append("dashboard_data/model_replay_freshness.json must contain an object")
        else:
            counts = model_replays.get("counts")
            profiles = model_replays.get("profiles")
            safe_end = model_replays.get("safe_backtest_end")
            latest_date = model_replays.get("latest_data_date")
            if not isinstance(counts, dict):
                errors.append("dashboard_data/model_replay_freshness.json missing counts")
            if not isinstance(profiles, list):
                errors.append("dashboard_data/model_replay_freshness.json profiles must be a list")
            if not isinstance(safe_end, str) or not safe_end:
                errors.append("dashboard_data/model_replay_freshness.json missing safe_backtest_end")
            if not isinstance(latest_date, str) or not latest_date:
                errors.append("dashboard_data/model_replay_freshness.json missing latest_data_date")
            else:
                info["model_latest_data"] = latest_date
            if isinstance(counts, dict) and isinstance(profiles, list):
                ready = int(counts.get("current", 0) or 0) + int(counts.get("refreshed", 0) or 0)
                failed = int(counts.get("failed", 0) or 0)
                skipped = int(counts.get("skipped", 0) or 0)
                info["model_replays"] = f"{ready}/{len(profiles)}@{safe_end or 'unknown'}"
                if failed or skipped:
                    errors.append(
                        "dashboard_data/model_replay_freshness.json has "
                        f"failed={failed} skipped={skipped}"
                    )

    if research_backtests_path.exists():
        research_backtests = _load_json(research_backtests_path)
        if not isinstance(research_backtests, list):
            errors.append("dashboard_data/research_backtests.json must contain a list")
        else:
            info["research_backtests"] = len(research_backtests)
            if not research_backtests:
                errors.append("dashboard_data/research_backtests.json has no rows")
            for idx, row in enumerate(research_backtests[:10]):
                if not isinstance(row, dict):
                    errors.append(f"dashboard_data/research_backtests.json row {idx} must be an object")
                    continue
                for key in ("experiment", "decision", "summary_path"):
                    if not row.get(key):
                        errors.append(f"dashboard_data/research_backtests.json row {idx} missing {key}")

    if dashboard_v2_manifest_path.exists() or dashboard_v2_comparison_path.exists():
        if not dashboard_v2_manifest_path.exists():
            errors.append("missing dashboard v2 manifest: dashboard_data/v2/manifest.json")
        elif not dashboard_v2_comparison_path.exists():
            errors.append("missing dashboard v2 comparison: dashboard_data/v2/comparison/default.json")
        else:
            manifest_v2 = _load_json(dashboard_v2_manifest_path)
            comparison_v2 = _load_json(dashboard_v2_comparison_path)
            latest = manifest_v2.get("latest_market_date")
            safe_end = manifest_v2.get("safe_backtest_end")
            common_end = comparison_v2.get("common_end")
            profiles = manifest_v2.get("default_comparison_profiles")
            if not isinstance(latest, str) or not latest:
                errors.append("dashboard_data/v2/manifest.json missing latest_market_date")
            if not isinstance(safe_end, str) or not safe_end:
                errors.append("dashboard_data/v2/manifest.json missing safe_backtest_end")
            if not isinstance(common_end, str) or not common_end:
                errors.append("dashboard_data/v2/comparison/default.json missing common_end")
            if not isinstance(profiles, list) or not profiles:
                errors.append("dashboard_data/v2/manifest.json missing default_comparison_profiles")
            info["dashboard_v2"] = (
                f"{len(profiles) if isinstance(profiles, list) else 0}"
                f"@{common_end or 'unknown'}/latest={latest or 'unknown'}"
            )

    missing = 0
    for version in versions:
        summary = version_summaries.get(version, {})
        version_etfs = summary.get("etfs") or etfs
        for etf in version_etfs:
            payload_path = data_dir / "versioned" / version / f"{etf}.json"
            if not payload_path.exists():
                errors.append(f"missing version payload: {payload_path.relative_to(root)}")
                missing += 1
                continue
            payload = _load_json(payload_path)
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                errors.append(f"empty rows in {payload_path.relative_to(root)}")
                continue
            expected_rows = (
                summary.get("windows", {})
                .get(etf, {})
                .get("rows")
            )
            if expected_rows is not None and len(rows) != expected_rows:
                errors.append(
                    f"row mismatch in {payload_path.relative_to(root)}: "
                    f"index={expected_rows} payload={len(rows)}"
                )
            if version == "paper_trading":
                meta = payload.get("meta", {})
                if etf == "SOXL" and isinstance(meta, dict):
                    latest_signal = meta.get("latest_signal_date")
                    if isinstance(latest_signal, str) and latest_signal:
                        info["paper_latest_signal"] = latest_signal
                findings = collect_sensitive_public_keys(
                    {
                        "meta": meta,
                        "rows": rows,
                    }
                )
                for finding in findings[:10]:
                    errors.append(f"sensitive paper_trading field in {payload_path.relative_to(root)}: {finding}")
                for idx, row in enumerate(rows[:10]):
                    if not isinstance(row, dict):
                        continue
                    for nav_key in ("strategy_nav", "buyhold_nav"):
                        value = row.get(nav_key)
                        if isinstance(value, (int, float)) and abs(float(value)) > 1000.0:
                            errors.append(
                                "paper_trading rows must use normalized nav "
                                f"in {payload_path.relative_to(root)} row={idx} field={nav_key}"
                            )
                            break

    info["missing"] = missing
    paper = version_summaries.get("paper_trading", {})
    if paper:
        paper_window = paper.get("windows", {}).get("SOXL", {})
        paper_meta = paper.get("meta", {})
        if paper_meta.get("source_kind") != "paper_trading":
            errors.append("paper_trading source_kind is not paper_trading")
        info["paper_trading_rows"] = paper_window.get("rows", "n/a")
        info["paper_trading_end"] = paper_window.get("end", "n/a")
        paper_latest = info["paper_latest_signal"]
        if paper_latest == "n/a":
            paper_latest = info["paper_trading_end"]
        model_latest = info["model_latest_data"]
        if isinstance(paper_latest, str) and isinstance(model_latest, str):
            if paper_latest not in ("", "n/a") and model_latest not in ("", "n/a") and paper_latest < model_latest:
                errors.append(
                    "paper_trading latest signal is stale: "
                    f"paper_latest={paper_latest} model_latest_data={model_latest}"
                )

    return errors, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    errors, info = check_static_export(root)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(
            "failed dashboard_static_export "
            f"generated_at={info['generated_at']} versions={info['versions']} "
            f"missing={info['missing']} model_versions={info['model_versions']} "
            f"latest={info['latest']} paper_trading_rows={info['paper_trading_rows']} "
            f"paper_trading_end={info['paper_trading_end']} "
            f"paper_latest_signal={info['paper_latest_signal']} "
            f"production_status={info['production_status']} "
            f"goal_gates={info['goal_gates']} "
            f"model_replays={info['model_replays']} "
            f"dashboard_v2={info['dashboard_v2']} "
            f"research_backtests={info['research_backtests']}"
        )
        return 1

    print(
        "ok dashboard_static_export "
        f"generated_at={info['generated_at']} versions={info['versions']} "
        f"missing={info['missing']} model_versions={info['model_versions']} "
        f"latest={info['latest']} paper_trading_rows={info['paper_trading_rows']} "
        f"paper_trading_end={info['paper_trading_end']} "
        f"paper_latest_signal={info['paper_latest_signal']} "
        f"production_status={info['production_status']} "
        f"goal_gates={info['goal_gates']} "
        f"model_replays={info['model_replays']} "
        f"dashboard_v2={info['dashboard_v2']} paper_trading"
        f" research_backtests={info['research_backtests']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
