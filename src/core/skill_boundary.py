from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SKILL_BOUNDARY_AUDIT_SCHEMA_VERSION = "core.skill_boundary_audit.v1"


_ISSUE_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "issue_type": "concrete_project_path",
        "severity": "block",
        "pattern": re.compile(
            r"(?P<match>/Users/[^\s`'\")]+|(?:runtime|research|production|data)/(?!<)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)"
        ),
        "recommendation": "Move concrete project paths into a research/production notebook or a src/core manifest.",
    },
    {
        "issue_type": "executable_command",
        "severity": "block",
        "pattern": re.compile(r"(?P<match>PYTHONPATH=|qlib_venv/bin/python|python -m pytest|nbconvert|pytest tests/)"),
        "recommendation": "Keep runnable commands in notebook evidence, AGENTS.md, or stable repo docs, not in a method skill.",
    },
    {
        "issue_type": "business_logic_entrypoint",
        "severity": "block",
        "pattern": re.compile(r"(?P<match>soxl_backtest\.(?:framework|reference)|from src\.[A-Za-z0-9_.]+ import|import src\.[A-Za-z0-9_.]+)"),
        "recommendation": "Move reusable business entrypoints into src/ and let the skill describe only the research method.",
    },
    {
        "issue_type": "project_state_result",
        "severity": "block",
        "pattern": re.compile(
            r"(?P<match>(?:current best model|best model|sharpe|max_drawdown|drawdown|return_pct|accuracy|score)\s*[=:]?\s*-?\d+(?:\.\d+)?%?)",
            re.IGNORECASE,
        ),
        "recommendation": "Move project state, model results, and fixed performance numbers into notebook conclusions or manifests.",
    },
    {
        "issue_type": "fixed_parameter",
        "severity": "review",
        "pattern": re.compile(r"(?P<match>(?:lookback|threshold|top_n|train_end|start|end)\s*[=:]\s*['\"]?[A-Za-z0-9_.:-]+)"),
        "recommendation": "Keep fixed experiment parameters in notebooks or production manifests unless the skill is only describing a placeholder.",
    },
)


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for issue in issues:
        key = (str(issue["issue_type"]), str(issue["matched_text"]), int(issue["line"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def build_skill_boundary_audit(
    skill_name: str,
    skill_text: str,
    *,
    source_path: str | Path = "",
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    text = str(skill_text or "")
    for pattern_config in _ISSUE_PATTERNS:
        pattern = pattern_config["pattern"]
        for match in pattern.finditer(text):
            matched_text = str(match.group("match") if "match" in match.groupdict() else match.group(0))
            issues.append(
                {
                    "schema_version": "core.skill_boundary_issue.v1",
                    "issue_type": str(pattern_config["issue_type"]),
                    "severity": str(pattern_config["severity"]),
                    "line": _line_for_offset(text, match.start()),
                    "matched_text": matched_text[:160],
                    "recommendation": str(pattern_config["recommendation"]),
                }
            )

    issues = _dedupe_issues(issues)
    block_count = sum(1 for issue in issues if issue["severity"] == "block")
    review_count = sum(1 for issue in issues if issue["severity"] == "review")
    status = "block" if block_count else "review" if review_count else "pass"
    migration_required = bool(block_count)
    return {
        "schema_version": SKILL_BOUNDARY_AUDIT_SCHEMA_VERSION,
        "skill_name": str(skill_name),
        "source_path": str(source_path),
        "status": status,
        "method_only": not issues,
        "allowed_skill_role": "method_prompt",
        "issue_count": int(len(issues)),
        "blocked_issue_count": int(block_count),
        "review_issue_count": int(review_count),
        "migration_required": migration_required,
        "recommended_destination": "research_notebook_or_src_core_manifest" if migration_required else "keep_in_skill",
        "blocked_surfaces_before_migration": ["skill"] if migration_required else [],
        "issues": issues,
    }


def audit_skill_file(path: str | Path) -> dict[str, Any]:
    skill_path = Path(path)
    return build_skill_boundary_audit(
        skill_path.parent.name,
        skill_path.read_text(encoding="utf-8"),
        source_path=skill_path,
    )
