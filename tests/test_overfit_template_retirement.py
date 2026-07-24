from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_overfit_regime_template_chain_is_retired_from_repo_artifacts_and_workflows():
    banned_tokens = [
        "regime" + "_template",
        "template" + "_target_position",
        "template" + "_strategy_return",
        "template" + "_position_source",
        "selected" + "_template_id",
        "SOURCE" + "_TEMPLATE_ARTIFACT",
        "dynamic" + "_soxx_soxl_cash_allocation_wf_v10",
        "dynamic" + "_soxx_soxl_cash_kelly_quarter",
    ]
    allowed = {Path("tests/test_overfit_template_retirement.py")}
    scanned_suffixes = {".py", ".json", ".md", ".yaml", ".csv", ".ipynb", ".txt"}
    offenders: list[str] = []

    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if rel in allowed or not path.is_file() or path.suffix not in scanned_suffixes:
            continue
        if any(
            part in rel.parts
            for part in {
                ".git",
                ".worktrees",
                "qlib_venv",
                "__pycache__",
            }
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in banned_tokens:
            if token in text or token in rel.as_posix():
                offenders.append(f"{rel}:{token}")

    assert offenders == []
