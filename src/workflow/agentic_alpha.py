from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


AGENTIC_ALPHA_STAGE_ORDER = ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
AGENTIC_ALPHA_DECISION_LABELS = ["Promote", "Iterate", "Drop"]
AGENTIC_ALPHA_SOURCE_URL = "https://github.com/JacobDu/agentic-alpha"


@dataclass(frozen=True)
class AgenticAlphaWorkflowSpec:
    name: str
    route: str
    market: str
    provider_uri: Path
    recorder_uri: Path
    source_url: str = AGENTIC_ALPHA_SOURCE_URL


def build_agentic_alpha_workflow_manifest(spec: AgenticAlphaWorkflowSpec) -> dict[str, Any]:
    route = spec.route.lower()
    if route not in {"research", "sfa", "mfa"}:
        raise ValueError("route must be one of: research, sfa, mfa")

    return {
        "workflow": spec.name,
        "route": route,
        "source_url": spec.source_url,
        "market": spec.market,
        "provider_uri": Path(spec.provider_uri).as_posix(),
        "recorder_uri": Path(spec.recorder_uri).as_posix(),
        "stage_order": AGENTIC_ALPHA_STAGE_ORDER,
        "decision_labels": AGENTIC_ALPHA_DECISION_LABELS,
        "qlib_native_required": True,
        "qlib_contract": {
            "data": ["qlib.init", "qlib.data.D.features", "DatasetH", "DataHandlerLP"],
            "workflow": ["qlib.workflow.R.start", "SignalRecord", "SigAnaRecord", "PortAnaRecord"],
        },
        "src_boundary": {
            "business_logic_surface": "src",
            "skill_scripts_imported": False,
            "notebook_surface": "research_or_production_card",
            "external_repo_role": "workflow_reference",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "machine_entrypoint": "summary.json",
            "recorder_only_artifacts": [
                "predictions",
                "signals",
                "portfolio_report",
                "positions",
                "all_model_predictions",
            ],
        },
    }
