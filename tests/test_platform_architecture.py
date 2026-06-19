from __future__ import annotations

import json
from pathlib import Path


def test_full_platform_surfaces_are_first_class():
    root = Path(__file__).resolve().parents[1]

    required_dirs = [
        "src/core",
        "src/data",
        "src/factor",
        "src/model",
        "src/objective",
        "src/backtest",
        "src/analysis",
        "src/dashboard",
        "src/schedule",
        "src/api",
        "research",
        "production",
    ]
    for rel_path in required_dirs:
        assert (root / rel_path).is_dir(), f"{rel_path} should be a tracked platform surface"


def test_reward_is_retired_in_favor_of_objective():
    root = Path(__file__).resolve().parents[1]

    assert not (root / "src" / "reward").exists()
    assert (root / "src" / "objective").is_dir()


def test_research_and_production_use_notebooks_as_primary_artifacts():
    root = Path(__file__).resolve().parents[1]

    notebooks = [
        root / "research" / "platform_architecture_reference" / "platform_architecture_reference.ipynb",
        root / "research" / "quantbench_model_matrix_smoke_v1" / "quantbench_model_matrix_smoke_v1.ipynb",
        root / "production" / "paper_reference_strategy" / "paper_reference_strategy.ipynb",
        root / "production" / "paper_sim_replay_v1" / "paper_sim_replay_v1.ipynb",
    ]
    for notebook in notebooks:
        assert notebook.exists(), f"missing notebook artifact: {notebook}"
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        assert "中文结论" in markdown
        assert "是否晋升" in markdown


def test_all_research_and_production_notebooks_end_with_chinese_conclusion():
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted([*root.glob("research/**/*.ipynb"), *root.glob("production/**/*.ipynb")])

    assert notebooks, "expected tracked research/production notebooks"

    offenders: list[str] = []
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        non_empty_cells = [
            cell
            for cell in payload.get("cells", [])
            if "".join(cell.get("source", [])).strip()
        ]
        last = non_empty_cells[-1] if non_empty_cells else {}
        conclusion = "".join(last.get("source", []))
        if (
            last.get("cell_type") != "markdown"
            or "中文结论" not in conclusion
            or "是否晋升" not in conclusion
        ):
            offenders.append(str(notebook.relative_to(root)))

    assert offenders == []


def test_production_paper_sim_replay_records_no_live_broker_guardrail():
    root = Path(__file__).resolve().parents[1]
    card = root / "production" / "paper_sim_replay_v1"

    summary = json.loads((card / "production_summary.json").read_text(encoding="utf-8"))
    ledger = json.loads((card / "intent_ledger.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == "production.paper_sim_replay.v1"
    assert summary["status"] == "paper_replay_completed"
    assert summary["live_broker_enabled"] is False
    assert summary["mode"] == "paper"
    assert ledger["intent"]["mode"] == "paper"
    assert ledger["execution_result"]["status"] == "paper_executed"


def test_production_paper_sim_replay_records_market_data_quote_source():
    root = Path(__file__).resolve().parents[1]
    card = root / "production" / "paper_sim_replay_v1"

    summary = json.loads((card / "production_summary.json").read_text(encoding="utf-8"))
    quote = json.loads((card / "market_quote.json").read_text(encoding="utf-8"))
    execution = json.loads((card / "paper_execution_result.json").read_text(encoding="utf-8"))

    assert summary["market_data_adapter"] == "paper_market_data"
    assert summary["market_quote"]["schema_version"] == "api.market_quote.v1"
    assert summary["market_quote"] == quote
    assert summary["market_quote"]["instrument"] == summary["instrument"]
    assert summary["market_quote"]["data_mode"] == "paper"
    assert summary["market_quote"]["source_path"]
    assert summary["market_data_dataset_key"] == "daily.soxl_paper_replay"
    assert summary["data_update_status"] == "success"
    assert "data/snapshots/" in summary["market_quote"]["source_path"]
    assert summary["market_quote"]["source_path"].endswith("canonical_frame.csv")
    assert not (card / "paper_market_data_frame.csv").exists()
    assert execution["snapshot"]["price"] == summary["market_quote"]["price"]


def test_production_paper_sim_replay_publishes_strategy_manifest_and_registry():
    root = Path(__file__).resolve().parents[1]
    card = root / "production" / "paper_sim_replay_v1"

    manifest = json.loads((card / "production_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((root / "production" / "production_registry.json").read_text(encoding="utf-8"))
    current = registry["strategies"]["paper_sim_replay_v1"]["current"]

    assert manifest["schema_version"] == "production.manifest.v1"
    assert manifest["artifact_type"] == "production_strategy"
    assert manifest["producer"] == "src.core.production"
    assert manifest["artifact_id"] == "paper_sim_replay_v1"
    assert manifest["paths"]["notebook"] == "paper_sim_replay_v1.ipynb"
    assert manifest["paths"]["summary"] == "production_summary.json"
    assert manifest["paths"]["dashboard_export"] == "dashboard_export.json"
    assert manifest["paths"]["intent_ledger"] == "intent_ledger.json"
    assert manifest["paths"]["paper_execution_result"] == "paper_execution_result.json"
    assert manifest["paths"]["data_registry"] == "data/data_registry.json"
    assert manifest["paths"]["analysis_manifest"] == "matrix_card/analysis_manifest.json"
    assert manifest["metadata"]["mode"] == "paper"
    assert manifest["metadata"]["live_broker_enabled"] is False
    assert len(manifest["manifest_hash"]) == 64
    assert registry["schema_version"] == "production.registry.v1"
    assert current["manifest_hash"] == manifest["manifest_hash"]
    assert current["manifest_path"].endswith("production/paper_sim_replay_v1/production_manifest.json")


def test_project_docs_describe_full_platform_not_cleaned_workspace():
    root = Path(__file__).resolve().parents[1]

    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    spec = (root / "SPEC.md").read_text(encoding="utf-8")

    assert "全平台" in agents
    assert "src/analysis" in agents
    assert "不再考虑强化学习" in agents
    assert "Production execution, dashboards, scheduler code" not in agents
    assert "objective" in spec
    assert "reward" not in spec.lower()
