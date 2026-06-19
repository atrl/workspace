from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


DASHBOARD_MANIFEST_ALLOWED_KINDS = ("analysis_manifest", "artifact_manifest")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetSnapshot:
    name: str
    universe: tuple[str, ...]
    start: str
    end: str
    calendar_path: str
    data_paths: tuple[str, ...]
    schema_version: str = "dataset.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def schema_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    artifact_type: str
    producer: str
    schema_version: str
    dataset: DatasetSnapshot
    paths: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def manifest_hash(self) -> str:
        return stable_hash(self.to_dict())


def build_dashboard_manifest_source_contract(
    manifest_source: Any,
    *,
    required: bool = True,
    allowed_kinds: tuple[str, ...] = DASHBOARD_MANIFEST_ALLOWED_KINDS,
) -> dict[str, Any]:
    allowed = tuple(str(kind) for kind in allowed_kinds)
    base = {
        "schema_version": "core.dashboard_manifest_source_contract.v1",
        "required": bool(required),
        "status": "not_checked" if not required else "missing",
        "is_complete": False,
        "kind": "",
        "path": "",
        "manifest_hash": "",
        "producer": "",
        "source_schema_version": "",
        "allowed_kinds": list(allowed),
        "missing_fields": [],
        "error": "",
        "readiness_detail": "missing / missing hash ",
    }
    if not isinstance(manifest_source, dict):
        return base

    kind = str(manifest_source.get("kind", ""))
    manifest_path = str(manifest_source.get("path", ""))
    manifest_hash = str(manifest_source.get("manifest_hash", ""))
    producer = str(manifest_source.get("producer", ""))
    contract = {
        **base,
        "status": "missing",
        "kind": kind,
        "path": manifest_path,
        "manifest_hash": manifest_hash,
        "producer": producer,
        "source_schema_version": str(manifest_source.get("schema_version", "")),
    }

    if kind not in allowed:
        contract["status"] = "weak_kind"
        contract["error"] = f"kind {kind} not allowed"
        contract["readiness_detail"] = f"{kind} / weak_kind hash {manifest_hash[:10]}"
        return contract

    missing_fields = [
        field
        for field, value in (
            ("path", manifest_path),
            ("manifest_hash", manifest_hash),
            ("producer", producer),
        )
        if not value
    ]
    if missing_fields:
        contract["status"] = "incomplete"
        contract["missing_fields"] = missing_fields
        contract["error"] = "missing " + ",".join(missing_fields)
        contract["readiness_detail"] = f"{kind} / incomplete hash {manifest_hash[:10]}"
        return contract

    contract["status"] = "complete"
    contract["is_complete"] = True
    contract["readiness_detail"] = f"{kind} / complete hash {manifest_hash[:10]}"
    return contract


def _int_field(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def build_dashboard_readiness_overview(payload: dict[str, Any]) -> dict[str, Any]:
    gates: list[dict[str, Any]] = []

    manifest_contract = build_dashboard_manifest_source_contract(payload.get("manifest_source"))
    gates.append(
        {
            "gate": "dashboard_manifest",
            "label": "Dashboard Manifest",
            "status": "pass" if manifest_contract["status"] == "complete" else "block",
            "detail": str(manifest_contract.get("readiness_detail", "")),
            "source_schema_version": str(manifest_contract.get("source_schema_version", "")),
        }
    )

    production_health = payload.get("production_health")
    if isinstance(production_health, dict) and production_health:
        missing_count = _list_count(production_health.get("missing_required_paths"))
        healthy = bool(production_health.get("is_healthy", False)) and missing_count == 0
        gates.append(
            {
                "gate": "production_health",
                "label": "Production Health",
                "status": "pass" if healthy else "block",
                "detail": f"{production_health.get('health_status', 'unknown')} / missing {missing_count}",
                "source_schema_version": str(production_health.get("schema_version", "")),
            }
        )

    schedule_health = payload.get("schedule_health")
    if isinstance(schedule_health, dict) and schedule_health:
        job_count = _int_field(schedule_health, "job_count")
        failed_count = _int_field(schedule_health, "failed_job_count")
        retry_error_count = _int_field(schedule_health, "total_retry_error_count")
        top_failure_job = str(schedule_health.get("top_failure_job", ""))
        top_failure_error = str(schedule_health.get("top_failure_last_error", ""))
        if failed_count > 0:
            status = "block"
        elif job_count > 0:
            status = "pass"
        else:
            status = "review"
        detail = f"failed {failed_count} / jobs {job_count} / retry_errors {retry_error_count}"
        if top_failure_job or top_failure_error:
            detail += f" / top {top_failure_job} {top_failure_error}".rstrip()
        gates.append(
            {
                "gate": "schedule_health",
                "label": "Schedule Health",
                "status": status,
                "detail": detail,
                "source_schema_version": str(schedule_health.get("schema_version", "")),
            }
        )

    notebook_readiness = payload.get("notebook_card_readiness")
    if isinstance(notebook_readiness, dict) and notebook_readiness:
        summary = notebook_readiness.get("summary")
        notebook_summary = summary if isinstance(summary, dict) else {}
        published_count = _int_field(notebook_summary, "production_published_count")
        template_count = _int_field(notebook_summary, "production_template_count")
        missing_count = _int_field(notebook_summary, "production_manifest_missing_count")
        mismatch_count = _int_field(notebook_summary, "production_registry_mismatch_count")
        if missing_count > 0 or mismatch_count > 0 or published_count == 0:
            status = "block"
        elif template_count > 0:
            status = "review"
        else:
            status = "pass"
        gates.append(
            {
                "gate": "notebook_cards",
                "label": "Notebook Cards",
                "status": status,
                "detail": (
                    f"published {published_count} / template {template_count} / "
                    f"missing {missing_count} / mismatch {mismatch_count}"
                ),
                "source_schema_version": str(notebook_readiness.get("schema_version", "")),
            }
        )

    requested_contract = payload.get("requested_model_contract")
    if isinstance(requested_contract, dict) and requested_contract:
        requested_count = _int_field(requested_contract, "requested_model_count")
        completed_count = _int_field(requested_contract, "completed_requested_model_count")
        missing_count = _list_count(requested_contract.get("missing_from_ledger"))
        incomplete_count = _list_count(requested_contract.get("incomplete_requested_models"))
        all_completed = bool(requested_contract.get("all_requested_models_completed", False))
        status = "pass" if all_completed else "review" if completed_count > 0 else "block"
        gates.append(
            {
                "gate": "requested_models",
                "label": "Requested Models",
                "status": status,
                "detail": (
                    f"{completed_count} / {requested_count} complete / "
                    f"missing {missing_count} / incomplete {incomplete_count}"
                ),
                "source_schema_version": str(requested_contract.get("schema_version", "")),
            }
        )

    model_run_coverage = payload.get("model_run_coverage")
    if isinstance(model_run_coverage, dict) and model_run_coverage:
        expected_count = _int_field(model_run_coverage, "expected_model_count")
        trained_count = _int_field(model_run_coverage, "trained_model_count")
        backtested_count = _int_field(model_run_coverage, "backtested_model_count")
        all_completed = bool(model_run_coverage.get("all_models_completed", False))
        gates.append(
            {
                "gate": "model_runs",
                "label": "Model Runs",
                "status": "pass" if all_completed else "review",
                "detail": f"trained {trained_count} / backtested {backtested_count} / expected {expected_count}",
                "source_schema_version": str(model_run_coverage.get("schema_version", "")),
            }
        )

    pass_count = sum(1 for gate in gates if gate["status"] == "pass")
    review_count = sum(1 for gate in gates if gate["status"] == "review")
    block_count = sum(1 for gate in gates if gate["status"] == "block")
    return {
        "schema_version": "dashboard.readiness_overview.v1",
        "status": "blocked" if block_count else "review" if review_count else "ready",
        "gate_count": int(len(gates)),
        "pass_count": int(pass_count),
        "review_count": int(review_count),
        "block_count": int(block_count),
        "gates": gates,
    }
