from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import stable_hash


ARCHITECTURE_EXTENSION_DECISION_SCHEMA_VERSION = "core.architecture_extension_decision.v1"
ARCHITECTURE_EXTENSION_REGISTRY_SCHEMA_VERSION = "core.architecture_extension_registry.v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _clean_items(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _require_text(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def architecture_extension_content_hash(payload: Mapping[str, Any]) -> str:
    content = {str(key): value for key, value in dict(payload).items() if key != "decision_hash"}
    return stable_hash(_json_ready(content))


def build_architecture_extension_decision(
    *,
    decision_id: str,
    target_surface: str,
    change_type: str,
    reason: str,
    impact_scope: Iterable[str],
    evidence_notebook: str,
    verification_commands: Iterable[str],
    alternatives_considered: Iterable[str] = (),
    constraints: Iterable[str] = (),
    status: str = "accepted",
    created_at: str = "",
) -> dict[str, Any]:
    impact = _clean_items(impact_scope)
    commands = _clean_items(verification_commands)
    if not impact:
        raise ValueError("impact_scope is required")
    if not commands:
        raise ValueError("verification_commands is required")

    payload: dict[str, Any] = {
        "schema_version": ARCHITECTURE_EXTENSION_DECISION_SCHEMA_VERSION,
        "decision_id": _require_text(decision_id, "decision_id"),
        "target_surface": _require_text(target_surface, "target_surface"),
        "change_type": _require_text(change_type, "change_type"),
        "reason": _require_text(reason, "reason"),
        "impact_scope": impact,
        "evidence_notebook": _require_text(evidence_notebook, "evidence_notebook"),
        "verification_commands": commands,
        "alternatives_considered": _clean_items(alternatives_considered),
        "constraints": _clean_items(constraints),
        "status": _require_text(status, "status"),
        "created_at": str(created_at).strip(),
        "extension_requires_reason": True,
        "blocked_without_evidence": True,
        "skill_boundary": "method_only",
        "review_gate": "src extension requires reason, evidence notebook, focused verification, and Chinese review.",
    }
    payload["decision_hash"] = architecture_extension_content_hash(payload)
    return _json_ready(payload)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": ARCHITECTURE_EXTENSION_REGISTRY_SCHEMA_VERSION, "decisions": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("schema_version", ARCHITECTURE_EXTENSION_REGISTRY_SCHEMA_VERSION)
    payload.setdefault("decisions", {})
    return payload


def _registry_record(*, decision: Mapping[str, Any], decision_path: Path) -> dict[str, Any]:
    return {
        "decision_id": str(decision.get("decision_id", "")),
        "target_surface": str(decision.get("target_surface", "")),
        "change_type": str(decision.get("change_type", "")),
        "status": str(decision.get("status", "")),
        "decision_hash": str(decision.get("decision_hash", "")),
        "decision_path": str(decision_path),
        "evidence_notebook": str(decision.get("evidence_notebook", "")),
        "created_at": str(decision.get("created_at", "")),
    }


def _validate_decision_payload(decision: Mapping[str, Any]) -> None:
    _require_text(str(decision.get("decision_id", "")), "decision_id")
    _require_text(str(decision.get("target_surface", "")), "target_surface")
    _require_text(str(decision.get("change_type", "")), "change_type")
    _require_text(str(decision.get("reason", "")), "reason")
    _require_text(str(decision.get("evidence_notebook", "")), "evidence_notebook")
    if not _clean_items(decision.get("impact_scope", [])):
        raise ValueError("impact_scope is required")
    if not _clean_items(decision.get("verification_commands", [])):
        raise ValueError("verification_commands is required")


def _write_review(path: Path, *, decision: Mapping[str, Any], registry_path: Path) -> None:
    lines = [
        "# 架构扩展决策 Review",
        "",
        "## 中文结论",
        "",
        f"- 决策 ID：{decision.get('decision_id', '')}。",
        f"- 目标 surface：{decision.get('target_surface', '')}。",
        f"- 扩展原因：{decision.get('reason', '')}",
        f"- 影响范围：{', '.join(str(item) for item in decision.get('impact_scope', []))}。",
        f"- 证据 notebook：{decision.get('evidence_notebook', '')}。",
        f"- 验证命令：{'; '.join(str(item) for item in decision.get('verification_commands', []))}。",
        "- 是否晋升：否。本记录只说明稳定 src surface 的扩展理由，不代表策略或模型晋升。",
        "- 风险：没有 evidence notebook、聚焦验证和中文结论时，不应继续扩展稳定 src 契约。",
        f"- 下一步：在 registry `{registry_path}` 中追踪后续架构扩展记录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_architecture_extension_decision(
    *,
    output_dir: str | Path,
    decision: Mapping[str, Any],
    registry_path: str | Path | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    payload = _json_ready(dict(decision))
    _validate_decision_payload(payload)
    payload["decision_hash"] = architecture_extension_content_hash(payload)

    decision_id = _require_text(str(payload.get("decision_id", "")), "decision_id")
    decision_path = output / f"{decision_id}.json"
    decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    registry_file = Path(registry_path) if registry_path is not None else output / "architecture_extension_registry.json"
    registry = _load_registry(registry_file)
    decisions = registry.setdefault("decisions", {})
    decisions[decision_id] = _registry_record(decision=payload, decision_path=decision_path)
    registry["registry_hash"] = stable_hash(
        {
            "schema_version": registry["schema_version"],
            "decisions": decisions,
        }
    )
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(_json_ready(registry), ensure_ascii=False, indent=2), encoding="utf-8")

    review_path = output / "architecture_extension_review.md"
    _write_review(review_path, decision=payload, registry_path=registry_file)

    return {
        "schema_version": "core.architecture_extension_write_result.v1",
        "decision_path": str(decision_path),
        "registry_path": str(registry_file),
        "review_path": str(review_path),
        "decision_hash": str(payload["decision_hash"]),
        "registry_hash": str(registry["registry_hash"]),
    }
