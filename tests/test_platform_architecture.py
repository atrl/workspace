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
        root / "production" / "paper_reference_strategy" / "paper_reference_strategy.ipynb",
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
