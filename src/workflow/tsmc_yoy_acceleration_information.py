"""Workflow runner for the fixed TSMC YoY-acceleration 5-session information screen."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis import tsmc_yoy_acceleration_information as A

EXPERIMENT = "tsmc_yoy_acceleration_5session_information_20261010"
SCHEMA_VERSION = "tsmc_yoy_acceleration_5session_information_v1"
QLIB_DOCS = ["https://qlib.readthedocs.io/en/latest/component/data.html",
             "https://qlib.readthedocs.io/en/latest/component/recorder.html"]


def _resolve_root(root) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def run_tsmc_yoy_acceleration_information(*, root=None, card_dir=None) -> dict[str, Any]:
    spec_root = _resolve_root(root)
    card = Path(card_dir) if card_dir is not None else (spec_root / "research" / EXPERIMENT)
    card.mkdir(parents=True, exist_ok=True)
    plan_path = spec_root / "research" / EXPERIMENT / "preregister_plan.json"
    plan = json.loads(plan_path.read_text())
    outcome = A.evaluate(spec_root, plan)
    results, obs = outcome["results"], outcome["observations"]

    summary = {
        "experiment": EXPERIMENT, "schema_version": SCHEMA_VERSION, "card_status": "active_research",
        "models_trained": 0, "strategy": False, "orders": False, "alpha": False,
        "decision": results["decision"], "promotion_eligible": False,
        "plan_sha256": A._sha(plan_path),
        "single_mechanism": plan["single_mechanism"], "horizon_sessions": A.HORIZON,
        "primary_stat": plan["primary_stat"], "windows": results["windows"],
        "frozen_sources": {"source_manifest_sha256": results["source_manifest_sha256"],
                           "price_manifest_sha256": results["price_manifest_sha256"],
                           "source_recorder": results["source_meta"], "frozen_verified": results["frozen_verified"]},
        "data_lineage": {"source_provider_uri": A.SOURCE_PROVIDER, "price_provider_uri": A.PRICE_PROVIDER,
                         "source_as_of": "2026-10-10",
                         "source_manifest_sha256": results["source_manifest_sha256"],
                         "price_manifest_sha256": results["price_manifest_sha256"],
                         "source_events_object_sha256": results["source_meta"]["events_object_sha256"]},
        "per_window": results["per_window"], "primary_gate": results["primary_gate"],
        "caveats": results["caveats"],
        "qlib_docs_reference": QLIB_DOCS,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": ["observations", "bootstrap_stats", "source_snapshot"]},
    }
    summary.update(_record(spec_root, card, obs, results, plan_path))
    _write_card(summary, card)
    return summary


def _record(spec_root, card, obs, results, plan_path):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    import qlib
    from qlib.workflow import R

    uri = (Path(card) / "qlib_recorder").resolve()
    uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec_root / A.PRICE_PROVIDER).resolve()), region="us")
    boot = pd.DataFrame([{"window": w, **blk["bootstrap"]} for w, blk in results["per_window"].items()])
    source_snapshot = pd.DataFrame([{"source_manifest_sha256": results["source_manifest_sha256"],
                                     "price_manifest_sha256": results["price_manifest_sha256"],
                                     "source_recorder_id": results["source_meta"]["recorder_id"],
                                     "source_events_object_sha256": results["source_meta"]["events_object_sha256"],
                                     "source_card": results["source_meta"]["card"],
                                     "source_provider_uri": A.SOURCE_PROVIDER, "price_provider_uri": A.PRICE_PROVIDER}])
    saved = {"observations": obs, "bootstrap_stats": boot, "source_snapshot": source_snapshot}
    with R.start(experiment_name=EXPERIMENT, recorder_name="tsmc_yoy_accel_info", uri=str(uri)):
        rec = R.get_recorder()
        rec.log_params(schema_version=SCHEMA_VERSION, models_trained=0, horizon_sessions=A.HORIZON,
                       decision=results["decision"])
        rec.save_objects(**saved)
        rec.log_artifact(str(plan_path), artifact_path="plan")
        rid, eid = rec.id, str(rec.experiment_id)
    return {"recorder_id": rid, "experiment_id": eid, "recorder_uri": str(uri),
            "recorder_local_dir": str(uri / eid / rid), "saved_objects": sorted(saved)}


def _write_card(summary, card: Path) -> None:
    (card / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    rows = []
    for window, blk in summary["per_window"].items():
        rows.append({"window": window, "valid_n": blk["valid_n"], "primary_ic": blk["primary_ic"],
                     "pearson": blk["pearson"], "ci_low": blk["bootstrap"]["ci_low"],
                     "ci_high": blk["bootstrap"]["ci_high"], "valid_replicates": blk["bootstrap"]["valid_replicates"]})
    pd.DataFrame(rows).to_csv(card / "window_ic.csv", index=False)
    year_rows = []
    for window, blk in summary["per_window"].items():
        for year, y in blk["by_year"].items():
            year_rows.append({"window": window, "year": year, **y})
    pd.DataFrame(year_rows).to_csv(card / "year_ic.csv", index=False)
    coverage = [{"window": w, "events_in_window": b["events_in_window"], "valid_n": b["valid_n"],
                 "unknown": b["unknown"]} for w, b in summary["per_window"].items()]
    pd.DataFrame(coverage).to_csv(card / "coverage.csv", index=False)
    workflow = {"schema_version": SCHEMA_VERSION, "experiment": summary["experiment"], "models_trained": 0,
                "horizon_sessions": summary["horizon_sessions"], "primary_stat": summary["primary_stat"],
                "windows": summary["windows"], "primary_gate": summary["primary_gate"],
                "frozen_sources": summary["frozen_sources"], "qlib_docs_reference": QLIB_DOCS}
    (card / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary) -> str:
    g = summary["primary_gate"]
    lines = [
        "# TSMC YoY 加速度 5-session 信息初筛（0 model，中文结论）\n",
        f"- 固定机制：`{summary['single_mechanism']}`；固定 horizon=5 session；固定正方向；无 threshold/year/horizon/sign 搜索。\n",
        f"- primary stat=`{summary['primary_stat']['name']}`（时序 Spearman(score, spread5d)，非两资产横截面 IC）；secondary 仅 SOXL/SOXX label IC/Pearson。\n",
        f"- 决定：`{summary['decision']}`；gate：{g}。\n",
        "### 分窗口\n",
    ]
    for w, b in summary["per_window"].items():
        lines.append(f"- {w}：valid_n={b['valid_n']}；primaryIC={b['primary_ic']}；"
                     f"CI=[{b['bootstrap']['ci_low']},{b['bootstrap']['ci_high']}]；"
                     f"secondary SOXL/SOXX IC={b['secondary_soxl_ic']}/{b['secondary_soxx_ic']}；unknown={b['unknown']}。\n")
    lines += [
        f"- 源/价格 SHA：source {summary['frozen_sources']['source_manifest_sha256'][:12]} / "
        f"price {summary['frozen_sources']['price_manifest_sha256'][:12]}（frozen_verified={summary['frozen_sources']['frozen_verified']['verified']}）。\n",
        "- 边界：raw association 初筛；未证明超越 price/vol、无经济 alpha、不可上线；"
        "CI 跨 0 表示证据不足，而非证明不存在信息；后续仍需 priced-vol control 与 10/25bp economic/native 及独立前向。\n",
        "- 本 fixed route 因证据不足而停止；next action=end_fixed_monthly_information_route_no_parameter_search（不调 sign/horizon/threshold）。\n",
        "- outer=reused_diagnostic，不作资格/选择；unknown 全保留不填0；每 period 仅 1 独立观察。\n",
    ]
    if summary.get("recorder_id"):
        lines.append(f"- 子 Recorder：{summary['recorder_id']}。\n")
    return "".join(lines)
