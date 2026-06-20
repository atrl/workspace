from __future__ import annotations

from pathlib import Path


def test_rebuild_uses_qlib_native_active_surfaces():
    root = Path(__file__).resolve().parents[1]

    active_paths = ["data", "src", "research", "production", "tests"]
    for rel_path in active_paths:
        assert (root / rel_path).exists(), f"{rel_path} should be active for the qlib-native rebuild"

    retired_paths = ["dashboard.html", "dashboard_data", "web", "scripts", "codex_exec_plans"]
    for rel_path in retired_paths:
        assert not (root / rel_path).exists(), f"{rel_path} should not be a top-level active surface"


def test_remaining_skills_are_minimal_and_qlib_or_data_focused():
    root = Path(__file__).resolve().parents[1]
    skills = sorted(path.parent.name for path in (root / ".codex" / "skills").glob("*/SKILL.md"))

    assert skills == ["futu-api", "soxl-qlib-experiment"]


def test_runtime_keeps_only_cleanup_and_current_experiment_cards():
    root = Path(__file__).resolve().parents[1]
    runtime_cards = sorted(path.relative_to(root).as_posix() for path in (root / "runtime").glob("*/*/*"))

    assert runtime_cards == [
        "runtime/harness/card/project_cleanup_architecture_v1",
        "runtime/model_selection/card/soxlbacktest_qlib_reference_v1",
    ]


def test_research_and_production_notebooks_are_qlib_native():
    root = Path(__file__).resolve().parents[1]
    notebooks = [
        root / "research" / "qlib_native_reference_v1" / "qlib_native_reference_v1.ipynb",
        root / "production" / "qlib_paper_sim_v1" / "qlib_paper_sim_v1.ipynb",
    ]
    required_tokens = [
        "import qlib",
        "qlib.init",
        "from qlib.data import D",
        "from qlib.workflow import R",
        "DatasetH",
        "DataHandlerLP",
        "SignalRecord",
        "SigAnaRecord",
    ]

    for notebook_path in notebooks:
        source = notebook_path.read_text(encoding="utf-8")
        for token in required_tokens:
            assert token in source, f"{notebook_path} must include qlib-native token {token}"
