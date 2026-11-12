"""Frozen prior-session SOXX path proxy on same-anchor SOXL value differences."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import warnings
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path.cwd()
CARD = ROOT / "research/stage1_intraday_jump_relative_value_20261112"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["code_sha256"]
source_card = ROOT / PLAN["source_card"]
label_card = ROOT / PLAN["label_card"]
for card, prefix in ((source_card, "source"), (label_card, "label")):
    assert sha(card / "summary.json") == PLAN[prefix + "_summary_sha256"]
    assert sha(card / "verification_manifest.json") == PLAN[prefix + "_manifest_sha256"]
assert sha(source_card / "source_family_overlap_audit.json") == PLAN["source_overlap_sha256"]
source_plan = json.loads((source_card / "plan.json").read_text())
minute_provider = ROOT / source_plan["minute_provider"]
assert sha(minute_provider / "qlib_store_manifest.json") == source_plan["minute_manifest_sha256"]

# Existing Qlib minute bars supply ordinary within-session controls. This is
# source-only preparation, with no future return target or label access.
qlib.init(provider_uri=str(minute_provider), region="us", kernels=1)
raw_bars = D.features(
    ["SOXX"], ["$raw_open", "$raw_close", "$price_valid"],
    start_time="2018-10-12 09:31:00", end_time="2020-12-31 16:00:00", freq="1min",
).xs("SOXX", level="instrument").sort_index()
intraday_rows = []
for day, bars in raw_bars.groupby(raw_bars.index.normalize(), sort=True):
    valid = bool(
        bars["$price_valid"].eq(1).all()
        and bars[["$raw_open", "$raw_close"]].notna().all().all()
        and bars["$raw_open"].gt(0).all()
        and bars["$raw_close"].gt(0).all()
    )
    record = {
        "prior_source_day": day,
        "intraday_down_share": np.nan,
        "intraday_max_bar_share": np.nan,
        "intraday_rv_control": np.nan,
        "intraday_signed_eff": np.nan,
    }
    if valid:
        close = bars["$raw_close"].to_numpy(float)
        opening = bars["$raw_open"].to_numpy(float)
        ret = np.r_[
            np.log(close[0] / opening[0]),
            np.log(close[1:] / close[:-1]),
        ]
        rv = float(np.square(ret).sum())
        absolute = float(np.abs(ret).sum())
        if rv > 0 and absolute > 0:
            record.update(
                intraday_down_share=float(np.square(np.minimum(ret, 0)).sum() / rv),
                intraday_max_bar_share=float(np.square(ret).max() / rv),
                intraday_rv_control=rv,
                intraday_signed_eff=float(ret.sum() / absolute),
            )
    intraday_rows.append(record)
intraday = pd.DataFrame(intraday_rows).set_index("prior_source_day")

# The source card has a NAS-readback-verified next-session decision frame.
qlib.init(provider_uri=str(ROOT / source_plan["daily_provider"]), region="us", kernels=1)
sources, source_readback = load_verified_recorder_objects(
    source_card,
    expected_recorder_id=PLAN["source_recorder_id"],
    artifacts=["decision_source"],
    expected_manifest_sha256=PLAN["source_manifest_sha256"],
)
decision = sources["decision_source"]
decision = decision.loc[decision.index.year <= 2020].copy()
decision = decision.join(intraday, on="prior_source_day")
decision.index.name = "day"
source_cols = [
    "intraday_jump_proxy", "daily_jump_ratio20", "daily_rv20",
    "daily_down_share20", "prior_return", "mom20",
    "intraday_down_share", "intraday_max_bar_share",
    "intraday_rv_control", "intraday_signed_eff",
]
assert decision.index.is_unique
assert len(decision) == 558
valid_source = decision[source_cols].notna().all(axis=1)
assert valid_source.groupby(valid_source.index.year).sum().to_dict() == {
    2018: 53, 2019: 252, 2020: 252
}
assert decision.loc[valid_source, "available_at"].lt(
    decision.loc[valid_source].index + pd.Timedelta(hours=9, minutes=29)
).all()

# The frozen physical action panel is deserialized, but only source fields,
# action masks and clocks are used until the source gate has been written.
labels, label_readback = load_verified_recorder_objects(
    label_card,
    expected_recorder_id=PLAN["label_recorder_id"],
    artifacts=["full_label_panel"],
    expected_manifest_sha256=PLAN["label_manifest_sha256"],
)
panel = labels["full_label_panel"]
panel = panel.loc[panel.day.dt.year <= 2020].copy()
assert not panel.duplicated(["day", "anchor_id", "action_id"]).any()
assert panel.day.max() <= pd.Timestamp("2020-12-31")
assert panel.decision_at.dt.normalize().eq(panel.day).all()

def pair_row(first, second):
    if abs(float(first.u[0]) - float(second.u[0])) <= 1e-12:
        return None
    high, low = (first, second) if first.u[0] > second.u[0] else (second, first)
    z_high = np.asarray(high.z, float)
    z_low = np.asarray(low.z, float)
    assert np.max(np.abs(z_high - z_low)) <= 1e-12
    u_high = np.asarray(high.u, float)
    u_low = np.asarray(low.u, float)
    du = u_high - u_low
    row = {
        "day": high.day,
        "anchor_id": high.anchor_id,
        "decision_at": high.decision_at,
        "high_action_id": high.action_id,
        "low_action_id": low.action_id,
        "du0": float(du[0]),
        "du1": float(du[1]),
        "d_beta": float(3 * du[0] + du[1]),
        "d_u0_sq": float(u_high[0] ** 2 - u_low[0] ** 2),
        "d_u0_u1": float(u_high[0] * u_high[1] - u_low[0] * u_low[1]),
        "d_u1_sq": float(u_high[1] ** 2 - u_low[1] ** 2),
        "d_fee": float(high.quoted_fee_fraction - low.quoted_fee_fraction),
        "high_value_known": bool(high.value_known),
        "low_value_known": bool(low.value_known),
        "high_risk_known": bool(high.risk_known),
        "low_risk_known": bool(low.risk_known),
        "high_available_at": high.label_available_at,
        "low_available_at": low.label_available_at,
    }
    for i in range(10):
        row[f"z{i}_du0"] = float(z_high[i] * du[0])
        row[f"z{i}_du1"] = float(z_high[i] * du[1])
    return row


pairs = []
high_name = "held_SOXL60_SOXX20_cash20"
low_name = "held_SOXL25_SOXX50_cash25"
for (_, _), group in panel.groupby(["day", "anchor_id"], sort=True):
    if (group.role == "fit").sum() == 10 and (group.role == "held").sum() == 3:
        if group.day.iloc[0].year <= 2019:
            fit_rows = list(group.loc[group.role.eq("fit")].itertuples(index=False))
            for left, right in itertools.combinations(fit_rows, 2):
                candidate = pair_row(left, right)
                if candidate is not None:
                    candidate["role"] = "train_source"
                    pairs.append(candidate)
        else:
            held = group.loc[group.role.eq("held")]
            assert high_name in set(held.action_id) and low_name in set(held.action_id)
            candidate = pair_row(
                next(held.loc[held.action_id.eq(high_name)].itertuples(index=False)),
                next(held.loc[held.action_id.eq(low_name)].itertuples(index=False)),
            )
            assert candidate is not None and candidate["du0"] > 0
            candidate["role"] = "test_source"
            pairs.append(candidate)
    else:
        raise ValueError("frozen 10 fit / 3 held physical actions differ")
pair = pd.DataFrame(pairs)
pair = pair.merge(
    decision[source_cols].reset_index(), on="day", how="left", validate="many_to_one"
)
pair["source_valid"] = pair[source_cols].notna().all(axis=1)
for name in (
    "daily_jump_ratio20", "daily_rv20", "daily_down_share20",
    "prior_return", "mom20", "intraday_down_share",
    "intraday_max_bar_share", "intraday_rv_control", "intraday_signed_eff",
):
    pair[name + "_du0"] = pair[name] * pair.du0
pair["intraday_jump_beta"] = pair.intraday_jump_proxy * pair.d_beta
pair["intraday_jump_du0"] = pair.intraday_jump_proxy * pair.du0
controls = [
    "du0", "du1", *(f"z{i}_du0" for i in range(10)),
    *(f"z{i}_du1" for i in range(10)),
    "d_u0_sq", "d_u0_u1", "d_u1_sq", "d_fee",
    *(name + "_du0" for name in (
        "daily_jump_ratio20", "daily_rv20", "daily_down_share20",
        "prior_return", "mom20", "intraday_down_share",
        "intraday_max_bar_share", "intraday_rv_control", "intraday_signed_eff",
    )),
    "intraday_jump_beta",
]
candidate_features = controls + ["intraday_jump_du0"]
train_source = pair.loc[pair.role.eq("train_source") & pair.source_valid]
test_source = pair.loc[pair.role.eq("test_source")]
assert train_source.day.nunique() == 305
assert test_source.day.nunique() == 248
assert test_source.source_valid.groupby(test_source.day).first().sum() == 247
matrix = train_source[controls].to_numpy(float)
feature = train_source.intraday_jump_du0.to_numpy(float)
scale = matrix.std(axis=0)
scale[scale == 0] = 1.0
residual = feature - matrix / scale @ np.linalg.lstsq(
    matrix / scale, feature, rcond=None
)[0]
assert np.std(residual) > 1e-10
source_gate = {
    "source_days": len(decision),
    "valid_source_days_by_year": valid_source.groupby(valid_source.index.year).sum().to_dict(),
    "train_source_pair_rows": len(train_source),
    "train_source_days": int(train_source.day.nunique()),
    "test_source_anchors": len(test_source),
    "test_valid_source_days": int(test_source.loc[test_source.source_valid, "day"].nunique()),
    "candidate_residual_std_beyond_controls": float(np.std(residual)),
    "source_clock_pass": True,
    "target_value_analyzed_before_source_gate": False,
    "source_summary_sha256": PLAN["source_summary_sha256"],
    "label_summary_sha256": PLAN["label_summary_sha256"],
    "claim_limit": "Action-space and prior-session source support only; no predictor or account pass.",
}
(CARD / "source_gate.json").write_text(
    json.dumps(source_gate, indent=2, allow_nan=False) + "\n"
)
if os.environ.get("STAGE1_SOURCE_ONLY") == "1":
    print(json.dumps(source_gate, indent=2, allow_nan=False))
    raise SystemExit(0)

# Outcome use starts here. All 2020 unknown source/label anchors stay in the
# evaluation calendar; no future-funded filtering is used for scoring.
outcomes = panel[["day", "anchor_id", "action_id", "value", "risk"]].copy()
for side in ("high", "low"):
    mapped = outcomes.rename(columns={
        "action_id": side + "_action_id",
        "value": side + "_value",
        "risk": side + "_risk",
    })
    pair = pair.merge(
        mapped,
        on=["day", "anchor_id", side + "_action_id"],
        how="left", validate="many_to_one",
    )
assert len(pair) == len(pairs)
cutoff = pd.to_datetime((pair.day.dt.year + 1).astype(str) + "-01-01")
pair["label_known"] = (
    pair.high_value_known & pair.low_value_known
    & pair.high_risk_known & pair.low_risk_known
    & pair.high_available_at.lt(cutoff) & pair.low_available_at.lt(cutoff)
    & np.isfinite(pair.high_value) & np.isfinite(pair.low_value)
    & np.isfinite(pair.high_risk) & np.isfinite(pair.low_risk)
)
pair["target_value"] = pair.high_value - pair.low_value
pair["target_risk_damage"] = pair.high_risk - pair.low_risk
train = pair.loc[pair.role.eq("train_source") & pair.source_valid & pair.label_known].copy()
test = pair.loc[pair.role.eq("test_source")].copy()
assert train.day.nunique() >= 250 and len(test) == 1240
train["weight"] = 1.0 / train.groupby("day").day.transform("size")
test["weight"] = 1.0 / test.groupby("day").day.transform("size")
fill = train[candidate_features].mean()
for frame in (train, test):
    frame[candidate_features] = frame[candidate_features].fillna(fill)
assert np.isfinite(train[candidate_features].to_numpy(float)).all()
assert np.isfinite(test[candidate_features].to_numpy(float)).all()
combined = pd.concat([train, test], ignore_index=True)
combined["pair_id"] = (
    combined.anchor_id.astype(str) + "::" + combined.high_action_id.astype(str)
    + "::" + combined.low_action_id.astype(str)
)
assert not combined.duplicated(["decision_at", "pair_id"]).any()
static = combined.set_index(["decision_at", "pair_id"]).sort_index()
static.index.names = ["datetime", "instrument"]
native = pd.concat(
    {"feature": static[candidate_features], "label": static[["target_value"]],
     "weight": static[["weight"]]},
    axis=1,
)
dataset = DatasetH(
    handler=DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=native),
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
y_train = train_native[("label", "target_value")].to_numpy(float)
w_train = train_native[("weight", "weight")].to_numpy(float)
assert np.isfinite(np.r_[x_train.to_numpy().ravel(), y_train, w_train]).all()


def fit_or_load(name: str, fields: list[str]):
    checkpoint = CARD / (name + "_model.pkl")
    receipt_file = CARD / (name + "_model_receipt.json")
    if checkpoint.exists():
        receipt = json.loads(receipt_file.read_text())
        assert receipt["plan_sha256"] == FREEZE["plan_sha256"]
        assert receipt["sha256"] == sha(checkpoint)
        with checkpoint.open("rb") as stream:
            model, scaler = pickle.load(stream)
        return model, scaler, 0
    scaler = StandardScaler().fit(x_train[fields], sample_weight=w_train)
    model = Ridge(alpha=100.0, fit_intercept=True)
    model.fit(scaler.transform(x_train[fields]), y_train, sample_weight=w_train)
    with checkpoint.open("wb") as stream:
        pickle.dump((model, scaler), stream, protocol=pickle.HIGHEST_PROTOCOL)
    receipt_file.write_text(json.dumps({
        "plan_sha256": FREEZE["plan_sha256"],
        "sha256": sha(checkpoint), "solver_invocations": 1,
    }, indent=2) + "\n")
    return model, scaler, 1


control_model, control_scaler, control_fits = fit_or_load("control", controls)
candidate_model, candidate_scaler, candidate_fits = fit_or_load(
    "candidate", candidate_features
)
fit_calls = control_fits + candidate_fits
assert fit_calls in (0, 2)
pred = pd.DataFrame(index=x_test.index)
pred.index.names = ["decision_at", "pair_id"]
pred["control"] = control_model.predict(control_scaler.transform(x_test[controls]))
pred["candidate"] = candidate_model.predict(
    candidate_scaler.transform(x_test[candidate_features])
)
zero = x_test[candidate_features].copy()
zero["intraday_jump_du0"] = 0.0
pred["candidate_zero"] = candidate_model.predict(
    candidate_scaler.transform(zero)
)
test["pair_id"] = (
    test.anchor_id.astype(str) + "::" + test.high_action_id.astype(str)
    + "::" + test.low_action_id.astype(str)
)
scored = test.set_index(["decision_at", "pair_id"]).join(
    pred, validate="one_to_one"
)
assert len(scored) == len(test)
missing_source = ~scored.source_valid
for column in ("candidate", "candidate_zero"):
    scored.loc[missing_source, column] = scored.loc[missing_source, "control"]
known = scored.label_known & scored.source_valid
error = scored.loc[known, ["day", "target_value", "control", "candidate"]].copy()
train_mean = float(np.average(y_train, weights=w_train))
error["control_sq"] = np.square(error.target_value - error.control)
error["candidate_sq"] = np.square(error.target_value - error.candidate)
error["trainmean_sq"] = np.square(error.target_value - train_mean)
daily_error = error.groupby("day")[["control_sq", "candidate_sq", "trainmean_sq"]].mean()
days = sorted(scored.day.unique())


def block_interval(values: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
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
    return (
        float(np.quantile(finite, PLAN["bootstrap"]["lower"])),
        float(np.quantile(finite, PLAN["bootstrap"]["upper"])),
        int(PLAN["bootstrap"]["draws"] - len(finite)),
    )


control_mse = float(daily_error.control_sq.mean())
candidate_mse = float(daily_error.candidate_sq.mean())
constant_mse = float(daily_error.trainmean_sq.mean())
loss_lower, _, _ = block_interval(
    daily_error.control_sq.sub(daily_error.candidate_sq).reindex(days).to_numpy(float),
    seed=PLAN["bootstrap"]["seed"],
)
coefficient = float(candidate_model.coef_[-1])
information_pass = bool(
    candidate_mse < control_mse and candidate_mse < constant_mse
    and coefficient < 0 and loss_lower is not None and loss_lower > 0
)
actions = scored[[
    "day", "anchor_id", "source_valid", "label_known",
    "control", "candidate", "candidate_zero",
    "high_value", "low_value", "high_risk", "low_risk",
]].copy()
for arm in ("control", "candidate", "candidate_zero"):
    choose_high = actions[arm].gt(0)
    actions[arm + "_choice"] = np.where(choose_high, "high_SOXL", "higher_SOXX")
    actions[arm + "_value"] = np.where(choose_high, actions.high_value, actions.low_value)
    actions[arm + "_risk"] = np.where(choose_high, actions.high_risk, actions.low_risk)
    actions.loc[~actions.label_known, [arm + "_value", arm + "_risk"]] = np.nan
actions["changed"] = actions.control_choice.ne(actions.candidate_choice)
actions["direct_changed"] = actions.candidate_zero_choice.ne(actions.candidate_choice)
actions["value_improvement"] = actions.candidate_value - actions.control_value
actions["risk_damage"] = actions.candidate_risk - actions.control_risk
daily_action = actions.groupby("day")[
    ["value_improvement", "risk_damage"]
].mean().reindex(days)
value_lower, _, _ = block_interval(
    daily_action.value_improvement.to_numpy(float), seed=PLAN["bootstrap"]["seed"] + 1
)
_, risk_upper, _ = block_interval(
    daily_action.risk_damage.to_numpy(float), seed=PLAN["bootstrap"]["seed"] + 2
)
changed_days = int(actions.groupby("day").changed.any().sum())
direct_changed_days = int(actions.groupby("day").direct_changed.any().sum())
selected_unknown = int(
    (~actions.label_known).sum()
)
action_pass = bool(
    changed_days >= 20 and direct_changed_days >= 20 and selected_unknown == 0
    and daily_action.value_improvement.mean() > 0
    and value_lower is not None and value_lower > 0
    and risk_upper is not None and risk_upper <= 0
)
result = {
    "status": "qualified_local_pairwise_information_only"
    if information_pass and action_pass else "closed_fixed_intraday_jump_relative_value_failed",
    "train_days": int(train.day.nunique()),
    "train_pair_rows": len(train),
    "test_source_days": len(days),
    "test_known_days": len(daily_error),
    "source_unknown_days": int((~scored.source_valid).groupby(scored.day).any().sum()),
    "control_day_mse": control_mse,
    "candidate_day_mse": candidate_mse,
    "trainmean_day_mse": constant_mse,
    "loss_improvement_block_lower": loss_lower,
    "candidate_standardized_coefficient": coefficient,
    "information_gate_pass": information_pass,
    "matched_changed_days": changed_days,
    "same_model_direct_changed_days": direct_changed_days,
    "changed_selected_unknown_rows": selected_unknown,
    "local_value_improvement_mean": float(daily_action.value_improvement.mean()),
    "local_value_improvement_block_lower": value_lower,
    "risk_damage_block_upper": risk_upper,
    "action_gate_pass": action_pass,
    "model_fit_calls": fit_calls,
    "candidate_accounts": 0,
    "historical_research_passed": False,
}
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"])
    R.log_metrics(
        model_fit_calls=fit_calls,
        control_day_mse=control_mse,
        candidate_day_mse=candidate_mse,
        matched_changed_days=changed_days,
        same_model_direct_changed_days=direct_changed_days,
    )
    R.save_objects(
        frozen_plan=PLAN, source_gate=source_gate,
        source_readback=source_readback, label_readback=label_readback,
        pairwise_dataset=pair,
        result=result, control_model=control_model, control_scaler=control_scaler,
        candidate_model=candidate_model, candidate_scaler=candidate_scaler,
        predictions=pred, action_rows=actions, daily_error=daily_error,
    )
    recorder = R.get_recorder()
    result["recorder_id"] = recorder.id
    result["experiment_id"] = str(recorder.experiment_id)
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
