"""Fixed existing-data bucket sell-pressure x trend cash-value screen."""
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
from sklearn.preprocessing import StandardScaler

from src.analysis.stage1_impact_asymmetry_information import CONTROL_NUMERIC
from src.core.recorder_sources import load_verified_recorder_objects
from src.core.recorder_run import publish_evidence_manifest


ROOT = Path.cwd()
CARD = ROOT / "research/stage1_bucket_trend_cash_joint_20261110"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["code_sha256"]
bucket_card = ROOT / PLAN["bucket_source_card"]
impact_card = ROOT / PLAN["impact_source_card"]
for card, prefix in ((bucket_card, "bucket"), (impact_card, "impact")):
    assert sha(card / "summary.json") == PLAN[prefix + "_summary_sha256"]
    assert sha(card / "verification_manifest.json") == PLAN[prefix + "_manifest_sha256"]
minute_provider = ROOT / PLAN["minute_provider"]
daily_provider = ROOT / PLAN["daily_provider"]
assert sha(minute_provider / "qlib_store_manifest.json") == PLAN["minute_provider_manifest_sha256"]
assert sha(daily_provider / "qlib_store_manifest.json") == PLAN["daily_provider_manifest_sha256"]

# Read only Qlib source coordinates and source-intent keys before any label object.
qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
calendar = pd.DatetimeIndex(D.calendar(start_time="2017-01-01", end_time="2020-12-31"))
price = (
    D.features(["SOXX"], ["$close"], start_time=calendar.min(), end_time=calendar.max())
    .xs("SOXX", level="instrument")["$close"]
    .reindex(calendar)
    .astype(float)
)
assert price.notna().all() and price.gt(0).all()
prior_mom20 = price.pct_change(20, fill_method=None).shift(1)
trend = prior_mom20.rename("prior_mom20")
qlib.init(provider_uri=str(minute_provider), region="us", kernels=1)
bucket_objects, bucket_receipt = load_verified_recorder_objects(
    bucket_card,
    expected_recorder_id=PLAN["bucket_recorder_id"],
    artifacts=["full_market_bucket_panel", "source_intentions"],
    expected_manifest_sha256=PLAN["bucket_manifest_sha256"],
)
bucket = bucket_objects["full_market_bucket_panel"]
bucket = bucket.loc[bucket.trade_date.dt.year.le(2020)].copy()
intents = bucket_objects["source_intentions"]
intents = intents.loc[intents.trade_date.dt.year.le(2020)].copy()
keys = ["trade_date", "bar_end_at", "decision_at"]
assert not intents.duplicated(keys).any()
assert not bucket.duplicated(["trade_date", "bar_end_at"]).any()
source = intents[keys].merge(
    bucket[
        ["trade_date", "bar_end_at", "feature_available_at", "available", "bucket_signed_net"]
    ],
    on=["trade_date", "bar_end_at"], how="left", validate="one_to_one",
)
assert len(source) == len(intents)
assert source.feature_available_at.le(source.decision_at).all()
source["prior_mom20"] = source.trade_date.map(trend)
source["positive_prior_mom20"] = source.prior_mom20.clip(lower=0.0)
source["bucket_available"] = source.available.fillna(False).astype(bool)
source["sell_pressure"] = np.where(
    source.bucket_available,
    (-source.bucket_signed_net).clip(lower=0.0),
    0.0,
)
source["interaction"] = source.sell_pressure * source.positive_prior_mom20
assert source[["prior_mom20", "positive_prior_mom20", "interaction"]].notna().all().all()
available_days = source.loc[source.bucket_available].groupby(
    source.trade_date.dt.year
).trade_date.nunique().to_dict()
assert available_days.get(2019, 0) >= 200 and available_days.get(2020, 0) >= 200
available = source.loc[source.bucket_available]
matrix = np.column_stack([
    np.ones(len(available)),
    available["bucket_signed_net"],
    available["sell_pressure"],
    available["prior_mom20"],
    available["positive_prior_mom20"],
])
residual = available.interaction.to_numpy() - matrix @ np.linalg.lstsq(
    matrix, available.interaction.to_numpy(), rcond=None
)[0]
assert np.isfinite(residual).all() and np.std(residual) > 1e-10
source_audit = {
    "source_intents": len(source),
    "source_days": int(source.trade_date.nunique()),
    "bucket_available_days_by_year": available_days,
    "bucket_available_rows": int(source.bucket_available.sum()),
    "clock_and_key_identity": True,
    "source_spearman_signed_net_bar_return": float(
        bucket.loc[bucket.available, "bucket_signed_net"].corr(
            bucket.loc[bucket.available, "bar_return"], method="spearman"
        )
    ),
    "interaction_residual_std": float(np.std(residual)),
    "daily_provider_manifest_sha256": PLAN["daily_provider_manifest_sha256"],
    "minute_provider_manifest_sha256": PLAN["minute_provider_manifest_sha256"],
    "target_values_loaded_before_source_gate": False,
    "source_receipt": bucket_receipt,
}
(CARD / "source_gate.json").write_text(json.dumps(
    {k: v for k, v in source_audit.items() if k != "source_receipt"},
    indent=2, allow_nan=False,
) + "\n")

# The sealed impact card contributes matched state/minute controls and the
# previously saved H5 fixed-share label. Slice years before target operations.
impact_objects, impact_receipt = load_verified_recorder_objects(
    impact_card,
    expected_recorder_id=PLAN["impact_recorder_id"],
    artifacts=["source_feature_frame"],
    expected_manifest_sha256=PLAN["impact_manifest_sha256"],
)
frame = impact_objects["source_feature_frame"]
frame = frame.loc[frame.trade_date.dt.year.le(2020)].copy()
assert not frame.duplicated(keys).any()
frame = frame.merge(
    source[keys + [
        "feature_available_at", "bucket_available", "bucket_signed_net",
        "sell_pressure", "prior_mom20", "positive_prior_mom20", "interaction",
    ]],
    on=keys, how="left", validate="one_to_one",
)
assert len(frame) == len(source)
assert frame.feature_available_at.le(frame.decision_at).all()
assert frame.trade_date.max() <= pd.Timestamp("2020-12-31")
cutoff = pd.to_datetime((frame.trade_date.dt.year + 1).astype(str) + "-01-01")
frame["label_known"] = (
    frame.status.isin(["paired_funded", "attempt_no_fill"])
    & frame.label_available_at.lt(cutoff)
    & np.isfinite(frame.soxx_value)
)
frame["bucket_signed_net"] = frame.bucket_signed_net.where(frame.bucket_available)
frame["bucket_available"] = frame.bucket_available.astype(float)
controls = [
    *CONTROL_NUMERIC, "bucket_signed_net", "sell_pressure",
    "prior_mom20", "positive_prior_mom20", "bucket_available",
]
candidate_features = controls + ["interaction"]
assert all(c in frame for c in candidate_features)
training = frame.loc[frame.trade_date.dt.year.le(2019) & frame.label_known].copy()
evaluation = frame.loc[frame.trade_date.dt.year.eq(2020)].copy()
assert training.trade_date.nunique() >= 250
assert evaluation.trade_date.nunique() >= 200
weights = 1.0 / training.groupby("trade_date").trade_date.transform("size").to_numpy()
fill = {}
for column in candidate_features:
    values = training[column].to_numpy(dtype=float)
    good = np.isfinite(values)
    if not good.any():
        raise ValueError("all training values missing: " + column)
    fill[column] = float(np.average(values[good], weights=weights[good]))
for column in candidate_features:
    frame[column] = frame[column].astype(float).fillna(fill[column])
training = frame.loc[frame.trade_date.dt.year.le(2019) & frame.label_known].copy()
evaluation = frame.loc[frame.trade_date.dt.year.eq(2020)].copy()

# Native DatasetH carries the saved Qlib source and aligned H5 label through fit.
static = frame.set_index(["decision_at", "trade_date"]).sort_index()
static["instrument"] = "SOXX"
static = static.reset_index().set_index(["decision_at", "instrument"])
static.index.names = ["datetime", "instrument"]
static_frame = pd.concat({
    "feature": static[candidate_features],
    "label": static[["soxx_value"]],
}, axis=1)
dataset = DatasetH(
    handler=DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static_frame),
        infer_processors=[], learn_processors=[],
    ),
    segments={
        "train": ("2018-10-15", "2019-12-31 23:59"),
        "test": ("2020-01-01", "2020-12-31 23:59"),
    },
)
assert len(dataset.prepare("train")) and len(dataset.prepare("test"))


def fit(columns: list[str]) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler().fit(training[columns], sample_weight=weights)
    model = Ridge(alpha=100.0, fit_intercept=True)
    model.fit(scaler.transform(training[columns]), training.soxx_value, sample_weight=weights)
    return model, scaler


control_model, control_scaler = fit(controls)
candidate_model, candidate_scaler = fit(candidate_features)
evaluation["control_prediction"] = control_model.predict(
    control_scaler.transform(evaluation[controls])
)
evaluation["candidate_prediction"] = candidate_model.predict(
    candidate_scaler.transform(evaluation[candidate_features])
)
same_model_zero = evaluation[candidate_features].copy()
same_model_zero["interaction"] = 0.0
evaluation["candidate_interaction_zero"] = candidate_model.predict(
    candidate_scaler.transform(same_model_zero)
)
missing = ~evaluation.bucket_available.astype(bool)
evaluation.loc[missing, "candidate_prediction"] = evaluation.loc[missing, "control_prediction"]
evaluation.loc[missing, "candidate_interaction_zero"] = evaluation.loc[missing, "control_prediction"]
known = evaluation.label_known
train_mean = float(np.average(training.soxx_value, weights=weights))
errors = evaluation.loc[known, ["trade_date", "soxx_value", "control_prediction", "candidate_prediction"]].copy()
errors["control_sq"] = (errors.soxx_value - errors.control_prediction) ** 2
errors["candidate_sq"] = (errors.soxx_value - errors.candidate_prediction) ** 2
errors["trainmean_sq"] = (errors.soxx_value - train_mean) ** 2
daily_error = errors.groupby("trade_date")[["control_sq", "candidate_sq", "trainmean_sq"]].mean()
control_mse = float(daily_error.control_sq.mean())
candidate_mse = float(daily_error.candidate_sq.mean())
trainmean_mse = float(daily_error.trainmean_sq.mean())

dates = sorted(evaluation.trade_date.unique())
calendar_improvement = daily_error.control_sq.sub(daily_error.candidate_sq).reindex(dates).to_numpy(float)
rng = np.random.default_rng(PLAN["bootstrap"]["seed"])


def block_lower(values: np.ndarray) -> tuple[float | None, int]:
    n = len(values)
    draws = []
    for _ in range(PLAN["bootstrap"]["draws"]):
        starts = rng.integers(0, n, size=(n + 9) // 10)
        ix = ((starts[:, None] + np.arange(10)) % n).ravel()[:n]
        sample = values[ix]
        draws.append(float(np.nanmean(sample)) if np.isfinite(sample).any() else np.nan)
    finite = np.asarray(draws)[np.isfinite(draws)]
    return (
        float(np.quantile(finite, PLAN["bootstrap"]["lower_quantile"])) if len(finite) else None,
        int(PLAN["bootstrap"]["draws"] - len(finite)),
    )


mse_lower, mse_undefined = block_lower(calendar_improvement)

# Each policy scores every intention before its future fill. The first
# positive score is its only local SOXX action; no positive score means HOLD.
actions = []
for day, rows in evaluation.sort_values(["trade_date", "decision_at"]).groupby("trade_date"):
    row = {"trade_date": day}
    for arm, pred in (
        ("control", "control_prediction"), ("candidate", "candidate_prediction"),
        ("same_model_zero", "candidate_interaction_zero"),
    ):
        selected = rows.loc[rows[pred].gt(0)].head(1)
        if selected.empty:
            row[arm + "_decision_at"] = pd.NaT
            row[arm + "_known"] = True
            row[arm + "_value"] = 0.0
        else:
            selected = selected.iloc[0]
            row[arm + "_decision_at"] = selected.decision_at
            row[arm + "_known"] = bool(selected.label_known)
            row[arm + "_value"] = float(selected.soxx_value) if selected.label_known else np.nan
    actions.append(row)
actions = pd.DataFrame(actions).sort_values("trade_date")
hold_stamp = pd.Timestamp("1900-01-01")
actions["changed"] = actions.control_decision_at.fillna(hold_stamp).ne(
    actions.candidate_decision_at.fillna(hold_stamp)
)
actions["same_model_changed"] = actions.candidate_decision_at.fillna(hold_stamp).ne(
    actions.same_model_zero_decision_at.fillna(hold_stamp)
)
actions["paired_known"] = actions.control_known & actions.candidate_known
actions["increment"] = np.where(
    actions.paired_known,
    actions.candidate_value - actions.control_value,
    np.nan,
)
action_delta = actions.set_index("trade_date").increment.reindex(dates).to_numpy(float)
action_lower, action_undefined = block_lower(action_delta)
candidate_values = actions.set_index("trade_date").candidate_value.reindex(dates).to_numpy(float)
hold_lower, hold_undefined = block_lower(candidate_values)
increment_mean = float(np.nanmean(action_delta)) if np.isfinite(action_delta).any() else None
candidate_mean = float(np.nanmean(candidate_values)) if np.isfinite(candidate_values).any() else None
coefficient = float(candidate_model.coef_[-1])
information_pass = bool(
    candidate_mse < control_mse
    and candidate_mse < trainmean_mse
    and coefficient > 0
    and mse_lower is not None and mse_lower > 0
)
action_pass = bool(
    int(actions.changed.sum()) >= 20
    and increment_mean is not None and increment_mean > 0
    and action_lower is not None and action_lower > 0
    and candidate_mean is not None and candidate_mean > 0
    and hold_lower is not None and hold_lower > 0
    and bool(actions.candidate_known.all())
)
result = {
    "status": "qualified_local_information_only" if information_pass and action_pass else "closed_fixed_bucket_trend_cash_interaction_failed",
    "training_days": int(training.trade_date.nunique()),
    "training_rows": len(training),
    "test_source_days": len(dates),
    "test_known_days": len(daily_error),
    "control_dayweighted_mse": control_mse,
    "candidate_dayweighted_mse": candidate_mse,
    "trainmean_dayweighted_mse": trainmean_mse,
    "candidate_minus_control_mse_lower": mse_lower,
    "mse_undefined_bootstrap": mse_undefined,
    "interaction_standardized_coefficient": coefficient,
    "information_gate_passed": information_pass,
    "changed_action_days": int(actions.changed.sum()),
    "same_model_zero_changed_action_days": int(actions.same_model_changed.sum()),
    "candidate_selected_unknown_days": int((~actions.candidate_known).sum()),
    "paired_action_known_days": int(actions.paired_known.sum()),
    "incremental_local_value_mean": increment_mean,
    "incremental_local_value_lower": action_lower,
    "candidate_vs_hold_mean": candidate_mean,
    "candidate_vs_hold_lower": hold_lower,
    "action_undefined_bootstrap": action_undefined,
    "hold_undefined_bootstrap": hold_undefined,
    "action_gate_passed": action_pass,
    "model_fits": 2,
    "candidate_accounts": 0,
    "2021plus_target_outcomes_accessed": False,
    "historical_research_passed": False,
}
audit = {
    "source": source_audit,
    "impact_source_readback": impact_receipt,
    "label_known_training_rows": len(training),
    "label_known_test_rows": int(evaluation.label_known.sum()),
    "year_boundary_maturity_used": True,
    "source_intents_scored_before_fill": len(evaluation),
    "unknown_evaluation_rows": int((~evaluation.label_known).sum()),
    "train_only_missing_fills": fill,
}
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"], bucket_recorder_id=PLAN["bucket_recorder_id"])
    R.log_metrics(
        model_fits=2, control_dayweighted_mse=control_mse,
        candidate_dayweighted_mse=candidate_mse,
        changed_action_days=int(actions.changed.sum()),
        candidate_selected_unknown_days=int((~actions.candidate_known).sum()),
    )
    R.save_objects(
        frozen_plan=PLAN, audit=audit, result=result,
        control_model=control_model, candidate_model=candidate_model,
        control_scaler=control_scaler, candidate_scaler=candidate_scaler,
        daily_error=daily_error,
        predictions=evaluation[
            keys + ["bucket_available", "control_prediction", "candidate_prediction",
                    "candidate_interaction_zero", "label_known"]
        ],
        action_rows=actions,
    )
    recorder = R.get_recorder()
    result["recorder_id"] = recorder.id
    result["experiment_id"] = str(recorder.experiment_id)
summary = {
    "experiment": CARD.name,
    "plan_sha256": FREEZE["plan_sha256"],
    "source_gate": {
        k: v for k, v in source_audit.items() if k != "source_receipt"
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
publish_evidence_manifest(CARD, summary)
print(json.dumps(result, indent=2, allow_nan=False))
