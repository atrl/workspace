"""One frozen Qlib account: static SOXL/cash at saved XGB mean exposure."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import xgboost  # noqa: F401  # Import before Qlib/PyTorch on this macOS runtime.
import numpy as np
import pandas as pd
import qlib
from qlib.workflow import R

from src.analysis.integrated_allocation import (
    annual_paired_metrics, paired_metrics, stationary_bootstrap_excess,
)
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.core.recorder_sources import load_verified_recorder_objects
from src.core.recorder_run import publish_evidence_manifest


ROOT = Path.cwd()
CARD = ROOT / "research/independent_tree_exposure_matched_reference_20261112"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())
REPAIR = json.loads((CARD / "repair_plan.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["code_sha256"]
assert REPAIR["plan_sha256"] == FREEZE["plan_sha256"]
assert REPAIR["scientific_weight_calendar_tariff_source_or_gate_changes"] is False
old_card = ROOT / PLAN["source_card"]
full_card = ROOT / PLAN["full_path_source_card"]
provider = ROOT / PLAN["provider_uri"]
assert sha(old_card / "summary.json") == PLAN["source_summary_sha256"]
assert sha(old_card / "verification_manifest.json") == PLAN["source_manifest_sha256"]
assert sha(old_card / "plan.json") == PLAN["source_plan_sha256"]
assert sha(provider / "qlib_store_manifest.json") == PLAN["provider_manifest_sha256"]
for path, digest in PLAN["mechanical_code_sha256"].items():
    assert sha(ROOT / path) == digest
old_summary = json.loads((old_card / "summary.json").read_text())
old_plan = json.loads((old_card / "plan.json").read_text())
assert old_plan["expected_test_rows"] == PLAN["sessions"] == 944
assert old_summary["new_model_fits"] == 2
xgb_metrics = next(
    row for row in old_summary["result"] if row["case"] == PLAN["source_account_case"]
)
assert abs(PLAN["fixed_soxl_weight"] - xgb_metrics["average_soxl_pct"] / 100) < 1e-14
assert abs(
    PLAN["fixed_soxl_weight"] + PLAN["fixed_cash_weight"] - 1
) < 1e-14
assert PLAN["fixed_soxx_weight"] == 0
assert sha(full_card / "plan.json") == old_plan["source"]["plan_sha256"]
assert sha(full_card / "verification_manifest.json") == old_plan["source"]["inventory_sha256"]

qlib.init(provider_uri=str(provider), region="us", kernels=1)
source, source_receipt = load_verified_recorder_objects(
    full_card,
    expected_recorder_id=PLAN["full_path_source_recorder_id"],
    artifacts=["inputs/frame", "analysis/native/reused_outer_soxx_ensemble_0"],
    expected_manifest_sha256=old_plan["source"]["inventory_sha256"],
)
previous, previous_receipt = load_verified_recorder_objects(
    old_card,
    expected_recorder_id=PLAN["source_xgb_recorder_id"],
    artifacts=["analysis/native/xgb_unrestricted_0"],
    expected_manifest_sha256=PLAN["source_manifest_sha256"],
)
frame = source["inputs/frame"]
benchmark = source["analysis/native/reused_outer_soxx_ensemble_0"]
xgb_book = previous["analysis/native/xgb_unrestricted_0"]
test = frame.loc[frame.signal_date.between(PLAN["test_start"], PLAN["test_end"])].copy()
assert len(test) == PLAN["sessions"]
for book in (benchmark, xgb_book):
    assert test.trade_date.reset_index(drop=True).equals(
        book["authority"].trade_date.reset_index(drop=True)
    )
if os.environ.get("STAGE1_SOURCE_ONLY") == "1":
    print(json.dumps({
        "source_rows": len(frame),
        "paired_account_days": len(test),
        "provider_manifest_sha256": PLAN["provider_manifest_sha256"],
        "saved_xgb_recorder_id": PLAN["source_xgb_recorder_id"],
        "saved_benchmark_recorder_id": PLAN["full_path_source_recorder_id"],
        "no_new_account": True,
    }, indent=2))
    raise SystemExit(0)


class ConstantTarget:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def target(self, row, q, cash, known_prices, peak):
        return self.weights.copy()


weights = (
    PLAN["fixed_soxl_weight"],
    PLAN["fixed_soxx_weight"],
    PLAN["fixed_cash_weight"],
)
progress = CARD / "run_state.json"
assert not (CARD / "summary.json").exists()
if progress.exists():
    previous = json.loads(progress.read_text())
    failed = json.loads((CARD / "execution_receipt.json").read_text())
    assert previous["plan_sha256"] == FREEZE["plan_sha256"]
    assert previous["account_attempts"] == 1
    assert failed["status"] == "failed"
    assert previous["recorder_id"] == REPAIR["first_attempt_recorder_id"]
    first_artifacts = (
        CARD / "qlib_recorder" / previous["experiment_id"]
        / previous["recorder_id"] / "artifacts"
    )
    assert not (first_artifacts / "reference_native_account").exists()
    account_attempts = 2
else:
    account_attempts = 1
progress.write_text(json.dumps({
    "plan_sha256": FREEZE["plan_sha256"],
    "status": "native_reference_account_started",
    "account_attempts": account_attempts,
    "model_fit_attempts": 0,
}, indent=2) + "\n")
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.save_objects(
        frozen_plan=PLAN,
        source_readbacks={
            "full_path": source_receipt,
            "independent_tree": previous_receipt,
        },
    )
    recorder = R.get_recorder()
    progress.write_text(json.dumps({
        "plan_sha256": FREEZE["plan_sha256"],
        "status": "native_reference_account_ready",
        "account_attempts": account_attempts,
        "model_fit_attempts": 0,
        "recorder_id": recorder.id,
        "experiment_id": str(recorder.experiment_id),
    }, indent=2) + "\n")
progress.write_text(json.dumps({
    "plan_sha256": FREEZE["plan_sha256"],
    "status": "native_reference_account_running",
    "account_attempts": account_attempts,
    "model_fit_attempts": 0,
    "recorder_id": recorder.id,
    "experiment_id": str(recorder.experiment_id),
}, indent=2) + "\n")
replay = replay_full_path(
    test, models=[ConstantTarget(weights)], plan=old_plan, slippage_bps=0
)
book = native_tiered_stage1(
    test, replay, provider_uri=provider, tariff=old_plan["tariff"],
    slippage_bps=0, initial_cash=old_plan["initial_cash"], freq="1min",
    opening_offset=pd.Timedelta(hours=9, minutes=47),
    position_type=QlibContinuousPosition,
)
book["targets"] = replay["targets"]
path, base, xgb = (
    book["authority"], benchmark["authority"], xgb_book["authority"]
)
assert len(path) == len(base) == len(xgb) == PLAN["sessions"]
assert path.trade_date.reset_index(drop=True).equals(
    base.trade_date.reset_index(drop=True)
)
static_vs_soxx = paired_metrics(path, base)
xgb_vs_static = paired_metrics(xgb, path)
xgb_vs_soxx = paired_metrics(xgb, base)
assert abs(xgb_vs_soxx["relative_wealth"] - xgb_metrics["relative_wealth"]) < 1e-10
annual_static = annual_paired_metrics(path, base)
annual_xgb_static = annual_paired_metrics(xgb, path)
log_static_soxx = (
    np.log1p(path.strategy_return.to_numpy(float))
    - np.log1p(base.strategy_return.to_numpy(float))
)
log_xgb_static = (
    np.log1p(xgb.strategy_return.to_numpy(float))
    - np.log1p(path.strategy_return.to_numpy(float))
)
rolling_static = pd.Series(log_static_soxx).rolling(126).sum().dropna()
rolling_xgb = pd.Series(log_xgb_static).rolling(126).sum().dropna()
assert len(rolling_static) == len(rolling_xgb) == 819
bootstrap = stationary_bootstrap_excess(
    xgb, path, mean_block=63, resamples=10000, alpha=.025, seed=20261112
)
result = {
    "status": "exposure_matched_posthoc_reference_account_complete",
    "source_model_case": PLAN["source_account_case"],
    "frozen_soxl_weight": PLAN["fixed_soxl_weight"],
    "frozen_soxx_weight": 0.0,
    "frozen_cash_weight": PLAN["fixed_cash_weight"],
    "sessions": len(path),
    "new_model_fits": 0,
    "new_native_reference_accounts": 1,
    "implementation_account_invocations": account_attempts,
    "first_invocation_completed_accounts": 0 if account_attempts == 2 else None,
    "static_vs_soxx": static_vs_soxx,
    "xgb_vs_static": xgb_vs_static,
    "xgb_vs_soxx": xgb_vs_soxx,
    "static_average_soxl_weight_pct": float(100 * path.ending_soxl_weight.mean()),
    "xgb_average_soxl_weight_pct": xgb_metrics["average_soxl_pct"],
    "static_rolling126_positive_fraction": float(rolling_static.gt(1e-8).mean()),
    "xgb_vs_static_rolling126_positive_fraction": float(rolling_xgb.gt(1e-8).mean()),
    "static_monitored_drawdown_pct": float(-100 * path.monitored_drawdown.max()),
    "xgb_monitored_drawdown_pct": xgb_metrics["monitored_drawdown_pct"],
    "static_orders": len(book["orders"]),
    "xgb_orders": len(xgb_book["orders"]),
    "bootstrap_xgb_vs_static": bootstrap,
    "historical_research_passed": False,
    "candidate_promoted": False,
    "claim_limit": (
        "Posthoc exposure-matched diagnostic chosen from XGB ending holdings. "
        "Neither this static reference nor XGB passed the full SOXX 0bp risk "
        "and stability gates; no 2/5bp or forward result."
    ),
}
recorder.save_objects(
    reference_native_account=book,
    comparison=result,
    annual_static_vs_soxx=annual_static,
    annual_xgb_vs_static=annual_xgb_static,
    rolling126_static_vs_soxx=rolling_static,
    rolling126_xgb_vs_static=rolling_xgb,
    bootstrap_xgb_vs_static=bootstrap,
)
recorder.log_metrics(
    new_model_fits=0,
    new_native_reference_accounts=1,
    static_vs_soxx_relative_wealth=static_vs_soxx["relative_wealth"],
    xgb_vs_static_relative_wealth=xgb_vs_static["relative_wealth"],
)
recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)

summary = {
    "experiment": CARD.name,
    "plan_sha256": FREEZE["plan_sha256"],
    "result": result,
    "recorder_id": recorder_id,
    "experiment_id": experiment_id,
    "recorder_uri": str(CARD / "qlib_recorder"),
    "recorder_local_dir": str(CARD / "qlib_recorder" / experiment_id / recorder_id),
    "artifact_policy": PLAN["artifact_policy"],
    "implementation_repair": "repair_plan.json" if account_attempts == 2 else None,
}
publish_evidence_manifest(CARD, summary)
progress.write_text(json.dumps({
    "plan_sha256": FREEZE["plan_sha256"],
    "status": "native_reference_account_complete",
    "account_attempts": account_attempts,
    "completed_native_reference_accounts": 1,
    "model_fit_attempts": 0,
    "recorder_id": recorder_id,
    "experiment_id": experiment_id,
}, indent=2) + "\n")
print(json.dumps({
    "static_vs_soxx": static_vs_soxx["relative_wealth"],
    "xgb_vs_static": xgb_vs_static["relative_wealth"],
    "static_drawdown": result["static_monitored_drawdown_pct"],
    "xgb_drawdown": result["xgb_monitored_drawdown_pct"],
}, indent=2))
