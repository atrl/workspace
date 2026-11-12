"""Frozen action-coupled shock risk and shared fee-net value local screen."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
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

from src.core.recorder_sources import load_verified_recorder_objects
from src.core.recorder_run import publish_evidence_manifest


ROOT = Path.cwd()
CARD = ROOT / "research/stage1_shock_action_marginal_value_risk_20261112"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["code_sha256"]
provider = ROOT / PLAN["source_provider"]
label_card = ROOT / PLAN["label_card"]
assert sha(provider / "qlib_store_manifest.json") == PLAN["source_manifest_sha256"]
assert sha(label_card / "summary.json") == PLAN["label_summary_sha256"]
assert sha(label_card / "verification_manifest.json") == PLAN["label_manifest_sha256"]
assert sha(ROOT / PLAN["source_preflight"]) == PLAN["source_preflight_sha256"]

qlib.init(provider_uri=str(provider), region="us", kernels=1)
calendar = pd.DatetimeIndex(D.calendar(start_time="2017-01-01", end_time="2020-12-31"))
price = (
    D.features(["SOXX"], ["$close"], start_time=calendar.min(), end_time=calendar.max())
    .xs("SOXX", level="instrument")["$close"]
    .reindex(calendar)
    .astype(float)
)
assert price.notna().all() and price.gt(0).all()
daily_log = np.log(price).diff()
shock = (
    daily_log.abs().rolling(21).max() / daily_log.abs().rolling(21).sum()
).shift(1)
assert shock.loc["2018-10-15":"2020-12-31"].notna().all()
objects, source_receipt = load_verified_recorder_objects(
    label_card,
    expected_recorder_id=PLAN["label_recorder_id"],
    artifacts=["full_label_panel"],
    expected_manifest_sha256=PLAN["label_manifest_sha256"],
)
raw = objects["full_label_panel"]
raw = raw.loc[pd.to_datetime(raw.day).dt.year.le(2020)].copy()
assert not raw.duplicated(["day", "anchor_id", "action_id"]).any()
raw["day"] = pd.to_datetime(raw.day).dt.normalize()
raw["shock"] = raw.day.map(shock)
z = np.vstack(raw.z.to_numpy())
u = np.vstack(raw.u.to_numpy())
assert z.shape == (len(raw), 10) and u.shape == (len(raw), 2)
assert np.isfinite(np.r_[z.ravel(), u.ravel(), raw.shock.to_numpy(float)]).all()
assert raw.decision_at.dt.normalize().eq(raw.day).all()
for i in range(10):
    raw[f"z{i}"] = z[:, i]
for j in range(2):
    raw[f"u{j}"] = u[:, j]
for j in range(2):
    for i in range(10):
        raw[f"z{i}_u{j}"] = z[:, i] * u[:, j]
raw["u0_sq"] = u[:, 0] ** 2
raw["u0_u1"] = u[:, 0] * u[:, 1]
raw["u1_sq"] = u[:, 1] ** 2
raw["shock_z5"] = raw.shock * raw.z5
raw["shock_z5_beta_u"] = raw.shock_z5 * (3.0 * raw.u0 + raw.u1)
controls = [
    *(f"z{i}" for i in range(10)),
    "u0", "u1",
    *(f"z{i}_u{j}" for j in range(2) for i in range(10)),
    "u0_sq", "u0_u1", "u1_sq",
    "quoted_fee_fraction", "shock", "shock_z5",
]
candidate_features = controls + ["shock_z5_beta_u"]
assert np.isfinite(raw[candidate_features].to_numpy(float)).all()
source_train = raw.loc[raw.day.dt.year.le(2019) & raw.role.eq("fit")]
source_test = raw.loc[raw.day.dt.year.eq(2020) & raw.role.eq("held")]
assert len(source_train) == 15250 and len(source_test) == 3720
assert source_test.day.nunique() == 248
source_matrix = source_train[controls].to_numpy(float)
source_new = source_train.shock_z5_beta_u.to_numpy(float)
scale = source_matrix.std(axis=0)
scale[scale == 0] = 1.0
source_residual = source_new - source_matrix / scale @ np.linalg.lstsq(
    source_matrix / scale, source_new, rcond=None
)[0]
assert np.std(source_residual) > 1e-10
range_by_anchor = source_test.groupby(["day", "anchor_id"]).shock_z5_beta_u.agg(
    lambda values: float(values.max() - values.min())
)
assert len(range_by_anchor) == 1240
assert (range_by_anchor > 1e-12).sum() == 1230
source_gate = {
    "train_source_rows": len(source_train),
    "test_source_rows": len(source_test),
    "test_source_days": int(source_test.day.nunique()),
    "held_anchors": len(range_by_anchor),
    "held_anchors_with_action_variation": int((range_by_anchor > 1e-12).sum()),
    "residual_std_beyond_full_control_basis": float(np.std(source_residual)),
    "source_manifest_sha256": PLAN["source_manifest_sha256"],
    "label_manifest_sha256": PLAN["label_manifest_sha256"],
    "decision_time_clock_pass": True,
    "label_outcomes_analyzed_before_source_gate": False,
    "source_limit": "historical daily first receipt remains unverified",
}
(CARD / "source_gate.json").write_text(json.dumps(source_gate, indent=2) + "\n")
if os.environ.get("STAGE1_SOURCE_ONLY") == "1":
    print(json.dumps(source_gate, indent=2))
    raise SystemExit(0)

# Training labels are strictly mature before the next calendar-year start.
cutoff = pd.to_datetime((raw.day.dt.year + 1).astype(str) + "-01-01")
raw["known"] = (
    raw.risk_known & raw.value_known
    & raw.label_available_at.lt(cutoff)
    & np.isfinite(raw.risk) & np.isfinite(raw.value)
)
train = raw.loc[raw.day.dt.year.le(2019) & raw.role.eq("fit") & raw.known].copy()
test = raw.loc[raw.day.dt.year.eq(2020) & raw.role.eq("held")].copy()
assert train.day.nunique() >= 250 and test.day.nunique() == 248
train["weight"] = 1.0 / train.groupby("day").day.transform("size")
test["weight"] = 1.0 / test.groupby("day").day.transform("size")
panel = pd.concat([train, test], ignore_index=True)
panel["row_id"] = panel.anchor_id.astype(str) + "::" + panel.action_id.astype(str)
assert not panel.duplicated(["decision_at", "row_id"]).any()
static = panel.set_index(["decision_at", "row_id"]).sort_index()
static.index.names = ["datetime", "instrument"]
static_frame = pd.concat(
    {
        "feature": static[candidate_features],
        "label": static[["risk", "value"]],
        "weight": static[["weight"]],
    },
    axis=1,
)
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
train_native = dataset.prepare(
    "train", col_set=["feature", "label", "weight"], data_key=DataHandlerLP.DK_L
)
test_native = dataset.prepare(
    "test", col_set=["feature", "label", "weight"], data_key=DataHandlerLP.DK_L
)
assert len(train_native) == len(train) and len(test_native) == len(test)
x_train = train_native["feature"]
x_test = test_native["feature"]
w_train = train_native[("weight", "weight")].to_numpy(float)
y_value = train_native[("label", "value")].to_numpy(float)
y_risk = train_native[("label", "risk")].to_numpy(float)
assert np.isfinite(np.r_[x_train.to_numpy().ravel(), w_train, y_value, y_risk]).all()


def fit_or_load(name: str, columns: list[str], target: np.ndarray):
    checkpoint = CARD / (name + "_model.pkl")
    receipt_path = CARD / (name + "_model_receipt.json")
    if checkpoint.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["plan_sha256"] == FREEZE["plan_sha256"]
        assert receipt["sha256"] == sha(checkpoint)
        with checkpoint.open("rb") as stream:
            model, scaler = pickle.load(stream)
        return model, scaler, 0
    scaler = StandardScaler().fit(x_train[columns], sample_weight=w_train)
    model = Ridge(alpha=100.0, fit_intercept=True)
    model.fit(scaler.transform(x_train[columns]), target, sample_weight=w_train)
    with checkpoint.open("wb") as stream:
        pickle.dump((model, scaler), stream, protocol=pickle.HIGHEST_PROTOCOL)
    receipt_path.write_text(json.dumps({
        "plan_sha256": FREEZE["plan_sha256"],
        "sha256": sha(checkpoint),
        "solver_invocations": 1,
    }, indent=2) + "\n")
    return model, scaler, 1


value_model, value_scaler, value_fits = fit_or_load("value", controls, y_value)
control_model, control_scaler, control_fits = fit_or_load(
    "risk_control", controls, y_risk
)
candidate_model, candidate_scaler, candidate_fits = fit_or_load(
    "risk_candidate", candidate_features, y_risk
)
actual_fit_calls = value_fits + control_fits + candidate_fits
assert actual_fit_calls in (0, 3)
pred = pd.DataFrame(index=x_test.index)
pred.index.names = ["decision_at", "row_id"]
pred["value_pred"] = value_model.predict(value_scaler.transform(x_test[controls]))
pred["risk_control_raw"] = control_model.predict(
    control_scaler.transform(x_test[controls])
)
pred["risk_candidate_raw"] = candidate_model.predict(
    candidate_scaler.transform(x_test[candidate_features])
)
zero_features = x_test[candidate_features].copy()
zero_features["shock_z5_beta_u"] = 0.0
pred["risk_same_model_zero_raw"] = candidate_model.predict(
    candidate_scaler.transform(zero_features)
)
test["row_id"] = test.anchor_id.astype(str) + "::" + test.action_id.astype(str)
held = test.set_index(["decision_at", "row_id"]).join(pred, validate="one_to_one")
assert len(held) == len(test)
for arm in ("control", "candidate", "same_model_zero"):
    held["risk_" + arm] = np.clip(
        np.maximum(held["risk_" + arm + "_raw"], held.z5), 0.0, 1.0
    )
known = held.known
daily_risk = held.loc[known, ["day", "risk", "risk_control", "risk_candidate"]].copy()
daily_risk["control_sq"] = (daily_risk.risk - daily_risk.risk_control) ** 2
daily_risk["candidate_sq"] = (daily_risk.risk - daily_risk.risk_candidate) ** 2
train_risk_mean = float(np.average(y_risk, weights=w_train))
# The constant comparator receives the same decision-known drawdown floor.
daily_risk["trainmean_sq"] = (
    daily_risk.risk
    - np.maximum(train_risk_mean, held.loc[known, "z5"].to_numpy(float))
) ** 2
risk_day_error = daily_risk.groupby("day")[
    ["control_sq", "candidate_sq", "trainmean_sq"]
].mean()
# Preserve actual labels under an unambiguous name before scoring.
held["actual_value"] = held["value"]
daily_value = held.loc[known, ["day", "actual_value", "value_pred"]].copy()
train_value_mean = float(np.average(y_value, weights=w_train))
daily_value["model_sq"] = (daily_value.actual_value - daily_value.value_pred) ** 2
daily_value["trainmean_sq"] = (daily_value.actual_value - train_value_mean) ** 2
value_day_error = daily_value.groupby("day")[["model_sq", "trainmean_sq"]].mean()
days = sorted(held.day.unique())


def block_interval(values: np.ndarray, *, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = []
    for _ in range(PLAN["bootstrap"]["draws"]):
        starts = rng.integers(0, n, size=(n + 9) // 10)
        ix = ((starts[:, None] + np.arange(10)) % n).ravel()[:n]
        part = values[ix]
        samples.append(float(np.nanmean(part)) if np.isfinite(part).any() else np.nan)
    finite = np.asarray(samples)[np.isfinite(samples)]
    if not len(finite):
        return None, None, PLAN["bootstrap"]["draws"]
    return (
        float(np.quantile(finite, PLAN["bootstrap"]["lower"])),
        float(np.quantile(finite, PLAN["bootstrap"]["upper"])),
        int(PLAN["bootstrap"]["draws"] - len(finite)),
    )


risk_control_mse = float(risk_day_error.control_sq.mean())
risk_candidate_mse = float(risk_day_error.candidate_sq.mean())
risk_null_mse = float(risk_day_error.trainmean_sq.mean())
value_model_mse = float(value_day_error.model_sq.mean())
value_null_mse = float(value_day_error.trainmean_sq.mean())
risk_lower, _, _ = block_interval(
    risk_day_error.control_sq.sub(risk_day_error.candidate_sq).reindex(days).to_numpy(float),
    seed=20261112,
)
value_lower, _, _ = block_interval(
    value_day_error.trainmean_sq.sub(value_day_error.model_sq).reindex(days).to_numpy(float),
    seed=20261113,
)
value_calibration_pass = bool(
    value_model_mse < value_null_mse and value_lower is not None and value_lower > 0
)
risk_information_pass = bool(
    risk_candidate_mse < risk_control_mse
    and risk_candidate_mse < risk_null_mse
    and candidate_model.coef_[-1] > 0
    and risk_lower is not None and risk_lower > 0
)

# Score HOLD at u=(0,0) with each model; actual HOLD risk is consulted only
# after the fixed policy has selected the action.
representative = held.reset_index().sort_values(
    ["day", "anchor_id", "action_id"]
).groupby(["day", "anchor_id"]).head(1).copy()
hold_x = representative[candidate_features].copy()
for col in candidate_features:
    if col.startswith("u") or "_u" in col or col == "quoted_fee_fraction" or col == "shock_z5_beta_u":
        hold_x[col] = 0.0
hold = representative.copy()
hold["action_id"] = "HOLD"
hold["value_pred"] = 0.0
hold["actual_value"] = 0.0
hold["risk_control"] = np.clip(np.maximum(
    control_model.predict(control_scaler.transform(hold_x[controls])), hold.z5
), 0.0, 1.0)
hold["risk_candidate"] = np.clip(np.maximum(
    candidate_model.predict(candidate_scaler.transform(hold_x[candidate_features])), hold.z5
), 0.0, 1.0)
hold["risk_same_model_zero"] = hold["risk_candidate"]
hold["actual_risk"] = hold.hold_segment_risk
hold["known"] = np.isfinite(hold.actual_risk) & hold.label_available_at.lt(
    pd.to_datetime((hold.day.dt.year + 1).astype(str) + "-01-01")
)
held["actual_risk"] = held.risk
candidate_set = pd.concat([
    held.reset_index(), hold,
], ignore_index=True, sort=False)
candidate_set["tie_priority"] = candidate_set.action_id.ne("HOLD").astype(int)
choices = {}
for arm in ("control", "candidate", "same_model_zero"):
    rows = []
    score = "risk_" + arm
    for (day, anchor_id), group in candidate_set.groupby(["day", "anchor_id"], sort=True):
        feasible = group.loc[group[score].le(0.30 + 1e-12)]
        if feasible.empty:
            selected = group.loc[group.action_id.eq("HOLD")].iloc[0]
        else:
            selected = feasible.sort_values(
                ["value_pred", "tie_priority", "action_id"],
                ascending=[False, True, True], kind="mergesort",
            ).iloc[0]
        rows.append({
            "day": day, "anchor_id": anchor_id,
            arm + "_action_id": selected.action_id,
            arm + "_actual_value": float(selected.actual_value) if selected.known else np.nan,
            arm + "_actual_risk": float(selected.actual_risk) if selected.known else np.nan,
            arm + "_known": bool(selected.known),
        })
    choices[arm] = pd.DataFrame(rows).set_index(["day", "anchor_id"])
actions = choices["control"].join(choices["candidate"], validate="one_to_one")
actions = actions.join(choices["same_model_zero"], validate="one_to_one")
actions["changed"] = actions.control_action_id.ne(actions.candidate_action_id)
actions["direct_changed"] = actions.same_model_zero_action_id.ne(
    actions.candidate_action_id
)
actions["paired_known"] = actions.control_known & actions.candidate_known
actions["value_improvement"] = np.where(
    actions.paired_known,
    actions.candidate_actual_value - actions.control_actual_value, np.nan,
)
actions["risk_damage"] = np.where(
    actions.paired_known,
    actions.candidate_actual_risk - actions.control_actual_risk, np.nan,
)
actions["candidate_value"] = np.where(
    actions.candidate_known, actions.candidate_actual_value, np.nan
)
daily_action = actions.groupby(level="day")[
    ["value_improvement", "risk_damage", "candidate_value"]
].mean().reindex(days)
value_gain_lower, _, _ = block_interval(
    daily_action.value_improvement.to_numpy(float), seed=20261114
)
hold_gain_lower, _, _ = block_interval(
    daily_action.candidate_value.to_numpy(float), seed=20261115
)
_, risk_damage_upper, _ = block_interval(
    daily_action.risk_damage.to_numpy(float), seed=20261116
)
changed_days = int(actions.groupby(level="day").changed.any().sum())
direct_changed_days = int(actions.groupby(level="day").direct_changed.any().sum())
action_pass = bool(
    value_calibration_pass and direct_changed_days >= 20 and changed_days >= 20
    and bool(actions.candidate_known.all())
    and daily_action.value_improvement.mean() > 0
    and value_gain_lower is not None and value_gain_lower > 0
    and hold_gain_lower is not None and hold_gain_lower > 0
    and risk_damage_upper is not None and risk_damage_upper <= 0
)
result = {
    "status": "qualified_local_value_risk_action_only"
    if risk_information_pass and action_pass else "closed_fixed_action_marginal_joint_failed",
    "train_days": int(train.day.nunique()),
    "train_rows": len(train),
    "test_source_days": len(days),
    "test_known_days": len(risk_day_error),
    "risk_control_day_mse": risk_control_mse,
    "risk_candidate_day_mse": risk_candidate_mse,
    "risk_trainmean_day_mse": risk_null_mse,
    "risk_loss_improvement_block_lower": risk_lower,
    "risk_candidate_interaction_coef": float(candidate_model.coef_[-1]),
    "risk_information_pass": risk_information_pass,
    "value_model_day_mse": value_model_mse,
    "value_trainmean_day_mse": value_null_mse,
    "value_loss_improvement_block_lower": value_lower,
    "value_calibration_pass": value_calibration_pass,
    "matched_changed_action_days": changed_days,
    "same_model_direct_changed_days": direct_changed_days,
    "candidate_selected_unknown_anchors": int((~actions.candidate_known).sum()),
    "local_value_improvement_mean": float(daily_action.value_improvement.mean()),
    "local_value_improvement_block_lower": value_gain_lower,
    "candidate_vs_hold_block_lower": hold_gain_lower,
    "risk_damage_block_upper": risk_damage_upper,
    "action_gate_pass": action_pass,
    "model_fit_calls_this_attempt": actual_fit_calls,
    "candidate_accounts": 0,
    "historical_research_passed": False,
}
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"])
    R.log_metrics(
        model_fit_calls=actual_fit_calls,
        risk_control_day_mse=risk_control_mse,
        risk_candidate_day_mse=risk_candidate_mse,
        matched_changed_action_days=changed_days,
        same_model_direct_changed_days=direct_changed_days,
    )
    R.save_objects(
        frozen_plan=PLAN, source_gate=source_gate, source_readback=source_receipt,
        result=result, value_model=value_model, value_scaler=value_scaler,
        risk_control_model=control_model, risk_control_scaler=control_scaler,
        risk_candidate_model=candidate_model, risk_candidate_scaler=candidate_scaler,
        predictions=pred, action_rows=actions,
        risk_daily_error=risk_day_error, value_daily_error=value_day_error,
    )
    rec = R.get_recorder()
    result["recorder_id"] = rec.id
    result["experiment_id"] = str(rec.experiment_id)
summary = {
    "experiment": CARD.name,
    "plan_sha256": FREEZE["plan_sha256"],
    "source_gate": source_gate,
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
