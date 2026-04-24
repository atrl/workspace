#!/usr/bin/env python3
from __future__ import annotations

"""Shared helpers for memory normalization and retrieval."""

import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "memory"
INDEX_PATH = MEMORY_DIR / "index.json"
ACTIVE_WORKPAD = ROOT / "codex_exec_plans" / "active" / "current.md"

KIND_ORDER = ("experiment", "decision", "failure", "review", "ops", "skill", "other")
STATUS_ORDER = ("kept", "dropped", "superseded", "active", "reference")


def memory_markdown_paths() -> list[Path]:
    return sorted(MEMORY_DIR.glob("*.md"))


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    _, rest = text.split("---\n", 1)
    front, marker, body = rest.partition("\n---\n")
    if not marker:
        return {}, text
    payload = yaml.safe_load(front) or {}
    return payload, body


def dump_front_matter(payload: dict[str, Any]) -> str:
    normalized = {
        "date": payload.get("date", ""),
        "title": payload.get("title", ""),
        "kind": payload.get("kind", "other"),
        "profiles": payload.get("profiles", []),
        "status": payload.get("status", "reference"),
        "tags": payload.get("tags", []),
        "artifacts": payload.get("artifacts", []),
        "workpad": payload.get("workpad", ""),
        "summary": payload.get("summary", ""),
    }
    return "---\n" + yaml.safe_dump(
        normalized,
        allow_unicode=False,
        sort_keys=False,
        default_flow_style=False,
    ) + "---\n\n"


def extract_title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def infer_date(path: Path, body: str) -> str:
    match = re.match(r"(\d{4}-\d{2}-\d{2})", path.stem)
    if match:
        return match.group(1)
    heading = extract_title(body, "")
    heading_match = re.search(r"(\d{4}-\d{2}-\d{2})", heading)
    return heading_match.group(1) if heading_match else ""


def infer_kind(title: str, body: str) -> str:
    text = f"{title}\n{body}".lower()
    if "skill" in text:
        return "skill"
    if any(token in text for token in ("failed", "failure", "not promotable", "degraded")):
        return "failure"
    if any(token in text for token in ("review", "comparison", "gap", "analysis")):
        return "review"
    if any(token in text for token in ("decision", "roadmap", "next optimization order")):
        return "decision"
    if any(token in text for token in ("workflow", "ops", "dashboard", "runner")):
        return "ops"
    if any(token in text for token in ("experiment", "trial", "attempt", "retrain", "ablation")):
        return "experiment"
    return "other"


def infer_status(title: str, body: str) -> str:
    text = f"{title}\n{body}".lower()
    if any(token in text for token in ("dropped", "failed", "failure", "not promotable", "lose", "loses")):
        return "dropped"
    if any(token in text for token in ("superseded", "replaced")):
        return "superseded"
    if any(token in text for token in ("breakthrough", "beats", "best", "keep canonical", "keep the idea")):
        return "kept"
    if "active" in text or "next action" in text:
        return "active"
    return "reference"


def infer_profiles(title: str, body: str) -> list[str]:
    tokens = sorted(set(re.findall(r"`?([A-Za-z0-9_]+(?:_[A-Za-z0-9_]+)*)`?", f"{title}\n{body}")))
    profiles: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower.startswith("balanced") or lower.startswith("smoke") or lower in {"moe", "soxl"}:
            profiles.append(lower)
    return sorted(set(profiles))


def infer_artifacts(body: str) -> list[str]:
    matches = re.findall(
        r"(artifacts/models/[A-Za-z0-9._/-]+|runtime/[A-Za-z0-9._/-]+|evidence/[A-Za-z0-9._/-]+)",
        body,
    )
    normalized: list[str] = []
    for match in matches:
        if match.startswith("runtime/post_train_review"):
            normalized.append(match.replace("runtime/post_train_review", "evidence/post_train_review", 1))
        elif match.startswith("runtime/generalization"):
            normalized.append(match.replace("runtime/generalization", "evidence/generalization", 1))
        elif match.startswith("runtime/diagnosis"):
            normalized.append(match.replace("runtime/diagnosis", "evidence/diagnosis", 1))
        elif match.startswith("runtime/feature_analysis"):
            normalized.append(match.replace("runtime/feature_analysis", "evidence/feature_analysis", 1))
        else:
            normalized.append(match)
    return sorted(set(normalized))


def infer_tags(title: str, body: str, kind: str, status: str, profiles: list[str]) -> list[str]:
    text = f"{title}\n{body}".lower()
    tags = {kind, status}
    if "2022" in text:
        tags.add("eval-2022")
    if "walk-forward" in text or "wf_" in text:
        tags.add("walk-forward")
    if "dashboard" in text:
        tags.add("dashboard")
    if "rule-based" in text:
        tags.add("rule-based")
    if "transition" in text:
        tags.add("transition-features")
    if "stress" in text:
        tags.add("stress-regime")
    if "overlay" in text:
        tags.add("execution-overlay")
    if "checkpoint" in text or "iter " in text:
        tags.add("checkpoint-selection")
    tags.update(profile.lower() for profile in profiles)
    return sorted(tags)


def infer_summary(body: str) -> str:
    lines = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        if line.startswith("- ") or line.startswith("* "):
            line = line[2:].strip()
        lines.append(line)
        if len(" ".join(lines)) >= 240:
            break
    return " ".join(lines)[:240].strip()


def normalize_body(body: str) -> str:
    body = body.strip() + "\n"
    if "## Context" in body:
        return body
    title = extract_title(body, "")
    content_lines = [line for line in body.splitlines() if not line.startswith("# ")]
    content = "\n".join(content_lines).strip()
    if not title:
        title = "Memory Note"
    return (
        f"# {title}\n\n"
        "## Context\n\n"
        "Legacy memory note normalized into the repo-wide schema.\n\n"
        "## Evidence\n\n"
        f"{content}\n\n"
        "## Decision\n\n"
        "See the summary and status fields in front matter.\n\n"
        "## Next Use\n\n"
        "Use `scripts/memory_search.py` before opening additional memory files.\n"
    )


def normalize_memory_file(path: Path) -> dict[str, Any]:
    original = path.read_text(encoding="utf-8")
    front_matter, raw_body = split_front_matter(original)
    body = normalize_body(raw_body)
    title = front_matter.get("title") or extract_title(body, path.stem)
    date = front_matter.get("date") or infer_date(path, body)
    kind = front_matter.get("kind") or infer_kind(title, body)
    status = front_matter.get("status") or infer_status(title, body)
    profiles = infer_profiles(title, body)
    artifacts = infer_artifacts(body)
    summary = front_matter.get("summary") or infer_summary(body)
    tags = infer_tags(title, body, kind, status, profiles)
    payload = {
        "date": date,
        "title": title,
        "kind": kind if kind in KIND_ORDER else "other",
        "profiles": profiles,
        "status": status if status in STATUS_ORDER else "reference",
        "tags": tags,
        "artifacts": artifacts,
        "workpad": front_matter.get("workpad") or str(ACTIVE_WORKPAD.relative_to(ROOT)),
        "summary": summary,
    }
    normalized = dump_front_matter(payload) + body
    path.write_text(normalized, encoding="utf-8")
    excerpt = re.sub(r"\s+", " ", body.replace("\n", " ")).strip()[:280]
    return {
        "path": str(path.relative_to(ROOT)),
        **payload,
        "excerpt": excerpt,
    }


def build_index() -> list[dict[str, Any]]:
    records = [normalize_memory_file(path) for path in memory_markdown_paths()]
    INDEX_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return records


def load_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return build_index()
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
