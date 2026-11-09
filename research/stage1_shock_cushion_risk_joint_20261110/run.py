"""Fixed prior-shock by own-drawdown interaction on pi2 physical risk paths."""
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
CARD = ROOT / "research/stage1_shock_cushion_risk_joint_20261110"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())
REPAIR = json.loads((CARD / "repair_plan.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["code_sha256"]
assert REPAIR["plan_sha256"] == FREEZE["plan_sha256"]
assert REPAIR["previous_postfit_lost_model_calls"] == 2
provider = ROOT / PLAN["source_provider"]
label_card = ROOT / PLAN["label_card"]
assert sha(provider / "qlib_store_manifest.json") == PLAN["source_manifest_sha256"]
assert sha(label_card / "summary.json") == PLAN["label_summary_sha256"]
assert sha(label_card / "verification_manifest.json") == PLAN["label_manifest_sha256"]

qlib.init(provider_uri=str(provider), region="us", kernels=1)
calendar = pd.DatetimeIndex(D.calendar(start_time="2017-01-01", end_time="2020-12-31"))
price = (
    D.features(["SOXX"], ["$close"], start_time=calendar.min(), end_time=calendar.max())
    .xs("SOXX", level="instrument")["$close"]
    .reindex(calendar)
    .astype(float)
)
assert price.notna().all() and price.gt(0).all()
log_return = np.log(price).diff()
shock = (
    log_return.abs().rolling(21).max()
    / log_return.abs().rolling(21).sum()
).shift(1)
assert shock.loc["2018-10-15":"2020-12-31"].notna().all()

# A sealed native Qlib label object is loaded, then restricted to its
# 2018-2020 calendar before any risk or value calculation.
objects, label_readback = load_verified_recorder_objects(
    label_card,
    expected_recorder_id=PLAN["label_recorder_id"],
    artifacts=["full_label_panel"],
    expected_manifest_sha256=PLAN["label_manifest_sha256"],
)
raw = objects["full_label_panel"]
raw = raw.loc[pd.to_datetime(raw.day).dt.year.le(2020)].copy()
assert raw.role.isin(["fit", "held"]).all()
assert not raw.duplicated(["day", "anchor_id", "action_id"]).any()
raw["day"] = pd.to_datetime(raw.day).dt.normalize()
raw["shock"] = raw.day.map(shock)
raw["z5"] = raw.z.map(lambda z: float(z[5]))
raw["shock_x_drawdown"] = raw.shock * raw.z5
assert raw[["shock", "z5", "shock_x_drawdown"]].notna().all().all()
has_future_label = raw.value_known | raw.risk_known
assert raw.loc[has_future_label, "label_available_at"].gt(
    raw.loc[has_future_label, "decision_at"]
).all()
assert raw.decision_at.dt.normalize().eq(raw.day).all()
assert raw.decision_at.dt.hour.ge(9).all()
risk_known_2020 = raw.loc[raw.day.dt.year.eq(2020) & raw.role.eq("held") & raw.risk_known]
assert risk_known_2020.day.nunique() >= 200
assert raw.groupby(raw.day.dt.year).day.nunique().to_dict() == {
    2018: 53, 2019: 252, 2020: 248
}
source_matrix = np.column_stack(
    [np.ones(len(raw)), raw.shock.to_numpy(), raw.z5.to_numpy()]
)
interaction = raw.shock_x_drawdown.to_numpy()
residual = interaction - source_matrix @ np.linalg.lstsq(
    source_matrix, interaction, rcond=None
)[0]
assert np.isfinite(residual).all() and np.std(residual) > 1e-10
source_gate = {
    "source_days": 553,
    "shock_source_calendar_days": 558,
    "risk_known_2020_held_days": int(risk_known_2020.day.nunique()),
    "history_risk_known_is_not_fit_filter": True,
    "source_manifest_sha256": PLAN["source_manifest_sha256"],
    "label_manifest_sha256": PLAN["label_manifest_sha256"],
    "clock_and_key_identity": True,
    "interaction_residual_std": float(np.std(residual)),
    "target_value_analyzed_before_source_gate": False,
    "source_limit": "adjusted historical daily price; first live receipt unverified",
}
(CARD / "source_gate.json").write_text(
    json.dumps(source_gate, indent=2, allow_nan=False) + "\n"
)

# Keep only mature train-role 2018/19 and test-role 2020 rows. Held source
# rows with an unknown future label remain in scoring and action accounting.
year_cutoff = pd.to_datetime((raw.day.dt.year + 1).astype(str) + "-01-01")
raw["known"] = (
    raw.risk_known & raw.value_known
    & raw.label_available_at.lt(year_cutoff)
    & np.isfinite(raw.risk)
    & np.isfinite(raw.value)
)
training = raw.loc[
    raw.day.dt.year.le(2019) & raw.role.eq("fit") & raw.known
].copy()
evaluation = raw.loc[
    raw.day.dt.year.eq(2020) & raw.role.eq("held")
].copy()
assert training.day.nunique() >= 250
assert evaluation.day.nunique() == 248
z_train = np.vstack(training.z.to_numpy())
u_train = np.vstack(training.u.to_numpy())
z_eval = np.vstack(evaluation.z.to_numpy())
u_eval = np.vstack(evaluation.u.to_numpy())
assert z_train.shape[1] == z_eval.shape[1] == 10
assert u_train.shape[1] == u_eval.shape[1] == 2
features = [*(f"z{i}" for i in range(10)), "u0", "u1", "quoted_fee_fraction", "shock"]
candidate_features = features + ["shock_x_drawdown"]
for frame, z, u in ((training, z_train, u_train), (evaluation, z_eval, u_eval)):
    for i in range(10):
        frame[f"z{i}"] = z[:, i]
    for i in range(2):
        frame[f"u{i}"] = u[:, i]
    assert np.isfinite(frame[candidate_features].to_numpy(float)).all()
training["weight"] = 1.0 / training.groupby("day").day.transform("size")
evaluation["weight"] = 1.0 / evaluation.groupby("day").day.transform("size")

panel = pd.concat([training, evaluation], ignore_index=True)
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
    "train", col_set=["feature", "label", "weight"],
    data_key=DataHandlerLP.DK_L,
)
test_native = dataset.prepare(
    "test", col_set=["feature", "label", "weight"],
    data_key=DataHandlerLP.DK_L,
)
assert len(train_native) == len(training) and len(test_native) == len(evaluation)
train_x = train_native["feature"]
test_x = test_native["feature"]
train_y = train_native[("label", "risk")].to_numpy(float)
train_w = train_native[("weight", "weight")].to_numpy(float)
assert np.isfinite(np.r_[train_x.to_numpy().ravel(), train_y, train_w]).all()


def fit(columns: list[str]) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler().fit(train_x[columns], sample_weight=train_w)
    model = Ridge(alpha=100.0, fit_intercept=True)
    model.fit(scaler.transform(train_x[columns]), train_y, sample_weight=train_w)
    return model, scaler


checkpoint = CARD / "model_checkpoint.pkl"
checkpoint_receipt = CARD / "model_checkpoint_receipt.json"
if checkpoint.exists():
    receipt = json.loads(checkpoint_receipt.read_text())
    assert receipt["plan_sha256"] == FREEZE["plan_sha256"]
    assert receipt["sha256"] == sha(checkpoint)
    with checkpoint.open("rb") as stream:
        control_model, control_scaler, candidate_model, candidate_scaler = pickle.load(stream)
    this_attempt_fits = 0
else:
    control_model, control_scaler = fit(features)
    candidate_model, candidate_scaler = fit(candidate_features)
    this_attempt_fits = 2
    with checkpoint.open("wb") as stream:
        pickle.dump(
            (control_model, control_scaler, candidate_model, candidate_scaler),
            stream, protocol=pickle.HIGHEST_PROTOCOL,
        )
    checkpoint_receipt.write_text(json.dumps({
        "plan_sha256": FREEZE["plan_sha256"],
        "sha256": sha(checkpoint),
        "model_fits_this_attempt": 2,
        "scientific_parameters_changed": False,
    }, indent=2) + "\n")
pred = pd.DataFrame(index=test_x.index)
pred.index.names = ["decision_at", "row_id"]
pred["control"] = control_model.predict(control_scaler.transform(test_x[features]))
pred["candidate"] = candidate_model.predict(candidate_scaler.transform(test_x[candidate_features]))
zero = test_x[candidate_features].copy()
zero["shock_x_drawdown"] = 0.0
pred["candidate_interaction_zero"] = candidate_model.predict(
    candidate_scaler.transform(zero)
)
evaluation["row_id"] = evaluation.anchor_id.astype(str) + "::" + evaluation.action_id.astype(str)
joined = evaluation.set_index(["decision_at", "row_id"]).join(pred, validate="one_to_one")
assert len(joined) == len(evaluation)
known = joined.known
errors = joined.loc[known, ["day", "risk", "control", "candidate"]].copy()
train_mean = float(np.average(train_y, weights=train_w))
errors["control_sq"] = (errors.risk - errors.control) ** 2
errors["candidate_sq"] = (errors.risk - errors.candidate) ** 2
errors["trainmean_sq"] = (errors.risk - train_mean) ** 2
daily_error = errors.groupby("day")[["control_sq", "candidate_sq", "trainmean_sq"]].mean()
control_mse = float(daily_error.control_sq.mean())
candidate_mse = float(daily_error.candidate_sq.mean())
trainmean_mse = float(daily_error.trainmean_sq.mean())
days = sorted(joined.day.unique())
rng = np.random.default_rng(PLAN["bootstrap"]["seed"])


def block_interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    n = len(values)
    draws = []
    for _ in range(PLAN["bootstrap"]["draws"]):
        starts = rng.integers(0, n, size=(n + 9) // 10)
        ix = ((starts[:, None] + np.arange(10)) % n).ravel()[:n]
        sample = values[ix]
        draws.append(float(np.nanmean(sample)) if np.isfinite(sample).any() else np.nan)
    finite = np.asarray(draws)[np.isfinite(draws)]
    if not len(finite):
        return None, None, PLAN["bootstrap"]["draws"]
    return float(np.quantile(finite, .025)), float(np.quantile(finite, .975)), int(
        PLAN["bootstrap"]["draws"] - len(finite)
    )


mse_improvement = daily_error.control_sq.sub(daily_error.candidate_sq).reindex(days)
mse_lower, _, mse_undefined = block_interval(mse_improvement.to_numpy(float))

# The three frozen held physical actions are only a local same-anchor screen.
choices = {}
for arm in ("control", "candidate", "candidate_interaction_zero"):
    choices[arm] = (
        joined.reset_index()
        .sort_values(["day", "anchor_id", arm, "action_id"])
        .groupby(["day", "anchor_id"], sort=True)
        .head(1)
        .set_index(["day", "anchor_id"])
        [["action_id", "risk", "value", "known"]]
        .rename(columns={
            "action_id": arm + "_action_id",
            "risk": arm + "_risk",
            "value": arm + "_value",
            "known": arm + "_known",
        })
    )
actions = choices["control"].join(choices["candidate"], validate="one_to_one")
actions = actions.join(choices["candidate_interaction_zero"], validate="one_to_one")
actions["changed"] = actions.control_action_id.ne(actions.candidate_action_id)
actions["same_model_changed"] = actions.candidate_action_id.ne(
    actions.candidate_interaction_zero_action_id
)
actions["paired_known"] = actions.control_known & actions.candidate_known
actions["risk_improvement"] = np.where(
    actions.paired_known, actions.control_risk - actions.candidate_risk, np.nan
)
actions["value_damage"] = np.where(
    actions.paired_known, actions.control_value - actions.candidate_value, np.nan
)
daily_action = actions.groupby(level="day")[["risk_improvement", "value_damage"]].mean()
risk_lower, _, risk_undefined = block_interval(
    daily_action.risk_improvement.reindex(days).to_numpy(float)
)
_, value_damage_upper, value_undefined = block_interval(
    daily_action.value_damage.reindex(days).to_numpy(float)
)
changed_days = int(actions.groupby(level="day").changed.any().sum())
coefficient = float(candidate_model.coef_[-1])
information_passed = bool(
    candidate_mse < control_mse and candidate_mse < trainmean_mse
    and coefficient > 0 and mse_lower is not None and mse_lower > 0
)
action_passed = bool(
    changed_days >= 20 and bool(actions.candidate_known.all())
    and risk_lower is not None and risk_lower > 0
    and value_damage_upper is not None and value_damage_upper <= 0
)
result = {
    "status": "qualified_local_risk_information_only" if information_passed and action_passed else "closed_fixed_shock_cushion_risk_interaction_failed",
    "train_fit_days": int(training.day.nunique()),
    "train_fit_rows": len(training),
    "test_held_source_days": len(days),
    "test_held_known_days": len(daily_error),
    "control_dayweighted_risk_mse": control_mse,
    "candidate_dayweighted_risk_mse": candidate_mse,
    "trainmean_dayweighted_risk_mse": trainmean_mse,
    "mse_improvement_block_lower": mse_lower,
    "mse_undefined_bootstrap": mse_undefined,
    "interaction_standardized_coefficient": coefficient,
    "information_gate_passed": information_passed,
    "changed_action_days": changed_days,
    "same_model_zero_changed_action_days": int(
        actions.groupby(level="day").same_model_changed.any().sum()
    ),
    "candidate_selected_unknown_anchors": int((~actions.candidate_known).sum()),
    "risk_improvement_mean": float(daily_action.risk_improvement.mean()),
    "risk_improvement_block_lower": risk_lower,
    "value_damage_mean": float(daily_action.value_damage.mean()),
    "value_damage_block_upper": value_damage_upper,
    "action_undefined_bootstrap": risk_undefined,
    "value_undefined_bootstrap": value_undefined,
    "action_gate_passed": action_passed,
    "model_fits": 2,
    "fit_calls_in_this_attempt": this_attempt_fits,
    "previous_postfit_lost_model_calls": 2,
    "actual_solver_invocations_total": REPAIR["previous_postfit_lost_model_calls"] + 2,
    "candidate_accounts": 0,
    "historical_research_passed": False,
}
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"], source_provider=str(provider))
    R.log_metrics(
        model_fits=2, control_dayweighted_risk_mse=control_mse,
        candidate_dayweighted_risk_mse=candidate_mse,
        changed_action_days=changed_days,
        actual_solver_invocations_total=REPAIR["previous_postfit_lost_model_calls"] + 2,
    )
    R.save_objects(
        frozen_plan=PLAN, source_gate=source_gate, label_readback=label_readback,
        result=result, control_model=control_model, candidate_model=candidate_model,
        control_scaler=control_scaler, candidate_scaler=candidate_scaler,
        daily_error=daily_error, predictions=pred, action_rows=actions,
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
