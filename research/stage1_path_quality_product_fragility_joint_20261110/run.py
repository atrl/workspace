"""Frozen two-factor interaction screen on saved same-state Tiered labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.workflow import R
from sklearn.linear_model import Ridge

from src.core.recorder_sources import load_verified_recorder_objects
from src.core.recorder_run import publish_evidence_manifest


ROOT = Path.cwd()
CARD = ROOT / "research/stage1_path_quality_product_fragility_joint_20261110"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["replay_code_sha256"]
assert sha(ROOT / PLAN["factor_source_reference"]) == PLAN["factor_source_sha256"]
provider = ROOT / PLAN["source_provider"]
assert sha(provider / "qlib_store_manifest.json") == PLAN["provider_manifest_sha256"]
label_card = ROOT / PLAN["label_card"]
assert sha(label_card / "summary.json") == PLAN["label_summary_sha256"]
assert sha(label_card / "verification_manifest.json") == PLAN["label_manifest_sha256"]

# Source identity and causal clock checks precede access to any value label.
qlib.init(provider_uri=str(provider), region="us", kernels=1)
calendar = pd.DatetimeIndex(
    D.calendar(start_time=PLAN["source_start"], end_time=PLAN["source_end"])
)
prices = (
    D.features(
        ["SOXL", "SOXX"], ["$close"],
        start_time=calendar.min(), end_time=calendar.max(),
    )
    .reset_index()
    .pivot(index="datetime", columns="instrument", values="$close")
    .reindex(calendar)
    .astype(float)
)
assert prices[["SOXL", "SOXX"]].notna().all().all()
assert prices[["SOXL", "SOXX"]].gt(0).all().all()
soxx_log = np.log(prices["SOXX"]).diff()
soxx_pct = prices["SOXX"].pct_change(fill_method=None)
soxl_log = np.log(prices["SOXL"]).diff()
signed = soxx_log.rolling(21).sum() / soxx_log.abs().rolling(21).sum()
tracking = (
    soxl_log - np.log1p(3.0 * soxx_pct)
).rolling(21).std(ddof=1)
source = pd.DataFrame(
    {
        "signed_path_efficiency_21": signed,
        "soxl_tracking_error_vol_21": tracking,
        "soxx_vol20": soxx_pct.rolling(20).std(ddof=1),
        "prior_return": soxx_pct,
    },
    index=calendar,
).shift(1)
source["signed_positive"] = source["signed_path_efficiency_21"].clip(lower=0.0)
source["joint_fragility"] = (
    source["signed_positive"] * source["soxl_tracking_error_vol_21"]
)
source["source_date"] = pd.Series(calendar, index=calendar).shift(1)
source = source.loc["2018-10-15":"2020-12-31"].copy()
source.index.name = "trade_date"
assert len(source) == 558
feature_cols = PLAN["control_features"] + ["joint_fragility"]
assert source[feature_cols].notna().all().all()
assert (source["source_date"] < source.index).all()
assert source.groupby(source.index.year).size().to_dict() == {
    2018: 53, 2019: 252, 2020: 253
}
base = source[PLAN["control_features"]].to_numpy(dtype=float)
interaction = source["joint_fragility"].to_numpy(dtype=float)
residual = interaction - np.column_stack(
    [np.ones(len(base)), base]
) @ np.linalg.lstsq(
    np.column_stack([np.ones(len(base)), base]), interaction, rcond=None
)[0]
assert np.isfinite(residual).all() and np.std(residual) > 1e-10
source_gate = {
    "source_days": len(source),
    "source_years": source.groupby(source.index.year).size().to_dict(),
    "prior_completed_session_clock": True,
    "provider_manifest_sha256": PLAN["provider_manifest_sha256"],
    "source_price_sha256": {
        str(p.relative_to(provider)): sha(p)
        for p in provider.glob("features/*/close.day.bin")
    },
    "source_spearman": float(
        source["signed_path_efficiency_21"].corr(
            source["soxl_tracking_error_vol_21"], method="spearman"
        )
    ),
    "interaction_residual_std": float(np.std(residual)),
    "target_values_loaded_before_gate": False,
    "source_limit": "adjusted historical daily prices; first live receipt unverified",
}
(CARD / "source_gate.json").write_text(
    json.dumps(source_gate, indent=2, allow_nan=False) + "\n"
)

# Read the verified native Recorder, then slice years before any target operation.
objects, readback = load_verified_recorder_objects(
    label_card,
    expected_recorder_id=PLAN["label_recorder_id"],
    artifacts=["analysis/labels", "analysis/masks"],
    expected_manifest_sha256=PLAN["label_manifest_sha256"],
)
labels = objects["analysis/labels"]
labels = labels.loc[labels.trade_date.dt.year.le(2020)].copy()
assert labels.trade_date.max() <= pd.Timestamp("2020-12-31")
first = labels.sort_values(["trade_date", "decision_at"]).groupby(
    "trade_date", sort=True
).head(1).copy()
cutoff = pd.to_datetime((labels.trade_date.dt.year + 1).astype(str) + "-01-01")
mature = labels.label_available_at.lt(cutoff)
paired = labels.soxl_funded & labels.soxx_funded
known = paired & mature & np.isfinite(labels.relative_value_soxl_minus_soxx)
source["value"] = labels.loc[known].groupby(
    "trade_date"
).relative_value_soxl_minus_soxx.mean()
support = source.groupby(source.index.year)["value"].count().to_dict()
assert support.get(2019, 0) >= 200 and support.get(2020, 0) >= 200, support
assert (labels.loc[paired, "label_available_at"] > labels.loc[paired, "decision_at"]).all()
source_audit = {
    "source_gate": source_gate,
    "source_intent_rows": len(labels),
    "paired_mature_rows": int(known.sum()),
    "paired_mature_days": support,
    "year_end_purged_rows": int((paired & ~mature).sum()),
    "first_intent_days": len(first),
    "first_intent_paired_funded_days": int(
        (first.soxl_funded & first.soxx_funded).sum()
    ),
    "later_outcomes_accessed": False,
    "label_recorder_readback": readback,
}

# DatasetH is the train/test authority; day rows retain Qlib source lineage.
static = source[feature_cols + ["value"]].copy()
static["instrument"] = "SOXL_SOXX"
static = static.set_index("instrument", append=True)
static.index.names = ["datetime", "instrument"]
frame = pd.concat(
    {"feature": static[feature_cols], "label": static[["value"]]}, axis=1
)
dataset = DatasetH(
    handler=DataHandlerLP(
        instruments=None,
        start_time="2018-10-15",
        end_time="2020-12-31",
        data_loader=StaticDataLoader(config=frame),
        infer_processors=[],
        learn_processors=[],
    ),
    segments={
        "train": ("2018-10-15", "2019-12-31"),
        "test": ("2020-01-01", "2020-12-31"),
    },
)


def prepared(segment: str) -> pd.DataFrame:
    data = dataset.prepare(segment).copy()
    data.columns = [c[-1] if isinstance(c, tuple) else c for c in data.columns]
    data = data.droplevel("instrument")
    data.index.name = "trade_date"
    return data.dropna(subset=feature_cols + ["value"]).sort_index()


train, test = prepared("train"), prepared("test")
assert len(train) >= 250 and len(test) >= 200


def fit_predict(columns: list[str]) -> tuple[Ridge, dict, np.ndarray]:
    mean = train[columns].mean()
    std = train[columns].std(ddof=0)
    assert std.gt(0).all()
    model = Ridge(alpha=10.0, fit_intercept=True)
    model.fit(((train[columns] - mean) / std).to_numpy(), train["value"].to_numpy())
    prediction = model.predict(((test[columns] - mean) / std).to_numpy())
    return model, {"mean": mean.to_dict(), "std": std.to_dict()}, prediction


control_model, control_scaler, control_pred = fit_predict(PLAN["control_features"])
candidate_model, candidate_scaler, candidate_pred = fit_predict(feature_cols)
predictions = test[["value"]].copy()
predictions["control"] = control_pred
predictions["candidate"] = candidate_pred
control_mse = float(np.mean((predictions.value - predictions.control) ** 2))
candidate_mse = float(np.mean((predictions.value - predictions.candidate) ** 2))
interaction_coef = float(candidate_model.coef_[-1])

# A single fixed first source intent per day. Unknown/no-fill remains unknown.
first = first.loc[first.trade_date.dt.year.eq(2020)].copy()
first["trade_date"] = pd.to_datetime(first["trade_date"])
actions = first.merge(
    predictions[["control", "candidate"]].reset_index(),
    on="trade_date", how="left", validate="one_to_one",
)
first_cutoff = pd.to_datetime((actions.trade_date.dt.year + 1).astype(str) + "-01-01")
actions["known"] = (
    actions.soxl_funded & actions.soxx_funded
    & actions.label_available_at.lt(first_cutoff)
    & np.isfinite(actions.soxl_value)
    & np.isfinite(actions.soxx_value)
    & actions.control.notna() & actions.candidate.notna()
)
actions["control_arm"] = np.where(actions.control.gt(0), "SOXL", "SOXX")
actions["candidate_arm"] = np.where(actions.candidate.gt(0), "SOXL", "SOXX")
actions["control_value"] = np.where(
    actions.control_arm.eq("SOXL"), actions.soxl_value, actions.soxx_value
)
actions["candidate_value"] = np.where(
    actions.candidate_arm.eq("SOXL"), actions.soxl_value, actions.soxx_value
)
actions["increment"] = np.where(
    actions.known, actions.candidate_value - actions.control_value, np.nan
)
actions = actions.sort_values("trade_date")
delta = actions.increment.to_numpy(dtype=float)
rng = np.random.default_rng(PLAN["bootstrap"]["seed"])
n = len(delta)
draws = []
for _ in range(PLAN["bootstrap"]["draws"]):
    starts = rng.integers(0, n, size=(n + 9) // 10)
    ix = ((starts[:, None] + np.arange(10)) % n).ravel()[:n]
    selected = delta[ix]
    draws.append(float(np.nanmean(selected)) if np.isfinite(selected).any() else np.nan)
finite = np.asarray(draws)[np.isfinite(draws)]
lower = float(np.quantile(finite, PLAN["bootstrap"]["lower_quantile"])) if len(finite) else None
increment_mean = float(np.nanmean(delta)) if np.isfinite(delta).any() else None
action_changes = int(
    (actions.loc[actions.known, "control_arm"] != actions.loc[actions.known, "candidate_arm"]).sum()
)
information_passed = candidate_mse < control_mse and interaction_coef < 0
action_passed = bool(
    increment_mean is not None and increment_mean > 0 and lower is not None and lower > 0
)
result = {
    "status": "qualified_local_information_only" if information_passed and action_passed else "closed_fixed_joint_interaction_failed",
    "train_days": len(train),
    "test_days": len(test),
    "control_mse": control_mse,
    "candidate_mse": candidate_mse,
    "interaction_standardized_coefficient": interaction_coef,
    "information_gate_passed": information_passed,
    "first_intent_days": len(actions),
    "first_intent_known_days": int(actions.known.sum()),
    "first_intent_unknown_days": int((~actions.known).sum()),
    "funded_action_changes": action_changes,
    "incremental_local_value_mean": increment_mean,
    "incremental_local_value_block_lower": lower,
    "undefined_bootstrap_draws": int(PLAN["bootstrap"]["draws"] - len(finite)),
    "action_gate_passed": action_passed,
    "model_fits": 2,
    "candidate_accounts": 0,
    "2021plus_target_outcomes_accessed": False,
    "historical_research_passed": False,
}

with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"], source_provider=str(provider))
    R.log_metrics(
        model_fits=2,
        control_mse=control_mse,
        candidate_mse=candidate_mse,
        first_intent_known_days=int(actions.known.sum()),
        funded_action_changes=action_changes,
    )
    R.save_objects(
        frozen_plan=PLAN,
        source_audit=source_audit,
        daily_source=source,
        daily_labels=source[["value"]],
        control_model=control_model,
        candidate_model=candidate_model,
        control_scaler=control_scaler,
        candidate_scaler=candidate_scaler,
        predictions=predictions,
        action_rows=actions,
        result=result,
    )
    recorder = R.get_recorder()
    result["recorder_id"] = recorder.id
    result["experiment_id"] = str(recorder.experiment_id)

summary = {
    "experiment": CARD.name,
    "plan_sha256": FREEZE["plan_sha256"],
    "source_audit": {
        k: v for k, v in source_audit.items() if k != "label_recorder_readback"
    },
    "result": result,
    "recorder_id": result["recorder_id"],
    "experiment_id": result["experiment_id"],
    "recorder_uri": str(CARD / "qlib_recorder"),
    "recorder_local_dir": str(
        CARD / "qlib_recorder" / result["experiment_id"] / result["recorder_id"]
    ),
    "artifact_policy": PLAN["artifact_policy"],
}
(CARD / "summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False) + "\n"
)
publish_evidence_manifest(CARD, summary)
print(json.dumps(result, indent=2, allow_nan=False))
