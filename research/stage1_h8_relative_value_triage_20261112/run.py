"""Zero-fit H.8 source to same-anchor SOXL relative-value diagnostic."""
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
from qlib.workflow import R

from src.core.recorder_sources import load_verified_recorder_objects
from src.core.recorder_run import publish_evidence_manifest


ROOT = Path.cwd()
CARD = ROOT / "research/stage1_h8_relative_value_triage_20261112"
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
provider = ROOT / PLAN["daily_provider"]
assert sha(provider / "qlib_store_manifest.json") == PLAN["daily_manifest_sha256"]

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
market = pd.DataFrame(
    {
        "soxx_prior20_log_return": log_return.rolling(20).sum(),
        "soxx_prior20_rvol": log_return.rolling(20).std(ddof=1),
    },
    index=calendar,
).shift(1)
source_objects, source_receipt = load_verified_recorder_objects(
    source_card,
    expected_recorder_id=PLAN["source_recorder_id"],
    artifacts=["source_panel"],
    expected_manifest_sha256=PLAN["source_manifest_sha256"],
)
source = source_objects["source_panel"].copy()
assert len(source) == 558 and not source.trade_date.duplicated().any()
assert source.trade_date.max() <= pd.Timestamp("2020-12-31")
label_objects, label_receipt = load_verified_recorder_objects(
    label_card,
    expected_recorder_id=PLAN["label_recorder_id"],
    artifacts=["full_label_panel"],
    expected_manifest_sha256=PLAN["label_manifest_sha256"],
)
labels = label_objects["full_label_panel"]
labels = labels.loc[labels.day.dt.year.le(2020)].copy()
high_name = "held_SOXL60_SOXX20_cash20"
low_name = "held_SOXL25_SOXX50_cash25"
held = labels.loc[
    labels.role.eq("held") & labels.action_id.isin([high_name, low_name]),
    ["day", "anchor_id", "action_id", "decision_at", "label_available_at",
     "value_known", "risk_known", "z"],
].copy()
assert held.groupby(["day", "anchor_id"]).size().eq(2).all()
assert not held.duplicated(["day", "anchor_id", "action_id"]).any()
rows = []
for (day, anchor_id), group in held.groupby(["day", "anchor_id"], sort=True):
    high = group.loc[group.action_id.eq(high_name)].iloc[0]
    low = group.loc[group.action_id.eq(low_name)].iloc[0]
    z_high = np.asarray(high.z, float)
    z_low = np.asarray(low.z, float)
    assert np.max(np.abs(z_high - z_low)) <= 1e-12
    assert high.decision_at == low.decision_at
    rows.append({
        "day": day, "anchor_id": anchor_id, "decision_at": high.decision_at,
        "label_available_at": max(high.label_available_at, low.label_available_at),
        "both_value_known": bool(high.value_known and low.value_known),
        "both_risk_known": bool(high.risk_known and low.risk_known),
        "z0": float(z_high[0]), "z1": float(z_high[1]),
        "z5": float(z_high[5]), "z7": float(z_high[7]), "z8": float(z_high[8]),
    })
pair = pd.DataFrame(rows)
pair = pair.merge(
    source[[
        "trade_date", "source_known", "source_available_at_local",
        "selected_release_id", "deposit_growth_large_minus_small",
    ]],
    left_on="day", right_on="trade_date", how="left", validate="many_to_one",
)
pair["soxx_prior20_log_return"] = pair.day.map(market.soxx_prior20_log_return)
pair["soxx_prior20_rvol"] = pair.day.map(market.soxx_prior20_rvol)
pair["clock_ok"] = pair.source_available_at_local.le(pair.decision_at)
pair["year"] = pair.day.dt.year
pair["year_mature"] = pair.label_available_at.lt(
    pd.to_datetime((pair.year + 1).astype(str) + "-01-01")
)
controls = [
    "soxx_prior20_log_return", "soxx_prior20_rvol",
    "z0", "z1", "z5", "z7", "z8",
]
pair["source_eligible"] = (
    pair.source_known & pair.clock_ok
    & pair.both_value_known & pair.both_risk_known & pair.year_mature
    & np.isfinite(pair.deposit_growth_large_minus_small)
    & pair[controls].notna().all(axis=1)
)
support = {}
for year, group in pair.groupby("year"):
    eligible = group.loc[group.source_eligible]
    support[str(year)] = {
        "calendar_days": int(group.day.nunique()),
        "eligible_anchors": len(eligible),
        "eligible_days": int(eligible.day.nunique()),
        "distinct_reports": int(eligible.selected_release_id.nunique()),
        "source_unknown_anchors": int((~group.source_known).sum()),
    }
assert support["2019"]["eligible_days"] >= 200
assert support["2020"]["eligible_days"] >= 200
assert support["2019"]["distinct_reports"] >= 40
assert support["2020"]["distinct_reports"] >= 40
source_gate = {
    "support": support,
    "source_clock_pass": bool(pair.loc[pair.source_eligible, "clock_ok"].all()),
    "same_anchor_state_identity": True,
    "target_values_analyzed_before_gate": False,
    "current_annual_vintage_not_first_network_receipt": True,
    "source_summary_sha256": PLAN["source_summary_sha256"],
    "label_summary_sha256": PLAN["label_summary_sha256"],
}
(CARD / "source_gate.json").write_text(
    json.dumps(source_gate, indent=2, allow_nan=False) + "\n"
)

# Actual future target access starts only after source and sample support pass.
outcomes = labels.loc[
    labels.role.eq("held") & labels.action_id.isin([high_name, low_name]),
    ["day", "anchor_id", "action_id", "value"],
].copy()
for side, name in (("high", high_name), ("low", low_name)):
    frame = outcomes.loc[outcomes.action_id.eq(name)].rename(
        columns={"value": side + "_value"}
    )[["day", "anchor_id", side + "_value"]]
    pair = pair.merge(frame, on=["day", "anchor_id"], how="left", validate="one_to_one")
pair["target_value"] = pair.high_value - pair.low_value
pair.loc[~pair.source_eligible, "target_value"] = np.nan
daily = pair.groupby("day").agg(
    candidate=("deposit_growth_large_minus_small", "first"),
    target=("target_value", "mean"),
    z0=("z0", "mean"), z1=("z1", "mean"), z5=("z5", "mean"),
    z7=("z7", "mean"), z8=("z8", "mean"),
    soxx_prior20_log_return=("soxx_prior20_log_return", "first"),
    soxx_prior20_rvol=("soxx_prior20_rvol", "first"),
    selected_release_id=("selected_release_id", "first"),
    eligible_anchors=("source_eligible", "sum"),
).reindex(calendar).loc["2018-10-15":"2020-12-31"]
assert len(daily) == 558


def partial_rank(frame: pd.DataFrame) -> tuple[float, float]:
    available = frame.dropna(subset=["candidate", "target", *controls])
    if len(available) < 20:
        return np.nan, np.nan
    raw = float(available.candidate.corr(available.target, method="spearman"))
    ranks = available[["candidate", "target", *controls]].rank().to_numpy(float)
    design = np.column_stack([np.ones(len(ranks)), ranks[:, 2:]])
    candidate_residual = ranks[:, 0] - design @ np.linalg.lstsq(
        design, ranks[:, 0], rcond=None
    )[0]
    target_residual = ranks[:, 1] - design @ np.linalg.lstsq(
        design, ranks[:, 1], rcond=None
    )[0]
    partial = (
        float(np.corrcoef(candidate_residual, target_residual)[0, 1])
        if min(np.std(candidate_residual), np.std(target_residual)) > 1e-10
        else np.nan
    )
    return raw, partial


rng = np.random.default_rng(20261112)
metrics = []
for year in (2018, 2019, 2020):
    year_frame = daily.loc[daily.index.year == year]
    raw_ic, partial_ic = partial_rank(year_frame)
    if len(year_frame) < 63:
        intervals = {"raw": None, "partial": None}
        undefined = {"raw": 0, "partial": 0}
    else:
        draws = []
        n = len(year_frame)
        for _ in range(2000):
            starts = rng.integers(0, n, size=(n + 62) // 63)
            indices = ((starts[:, None] + np.arange(63)) % n).ravel()[:n]
            draws.append(partial_rank(year_frame.iloc[indices]))
        draws = np.asarray(draws)
        intervals, undefined = {}, {}
        for j, name in enumerate(("raw", "partial")):
            finite = draws[np.isfinite(draws[:, j]), j]
            intervals[name] = (
                np.quantile(finite, [.025, .975]).tolist()
                if len(finite) else None
            )
            undefined[name] = int(2000 - len(finite))
    metrics.append({
        "year": year,
        "native_days": len(year_frame),
        "known_days": int(year_frame.target.notna().sum()),
        "distinct_reports": int(year_frame.loc[
            year_frame.target.notna(), "selected_release_id"
        ].nunique()),
        "raw_rank_ic": raw_ic if np.isfinite(raw_ic) else None,
        "partial_rank_ic": partial_ic if np.isfinite(partial_ic) else None,
        "block95_intervals": intervals,
        "undefined_bootstrap_draws": undefined,
    })
information_gate = all(
    row["partial_rank_ic"] is not None
    and row["partial_rank_ic"] < 0
    and row["block95_intervals"]["partial"] is not None
    and row["block95_intervals"]["partial"][1] < 0
    and row["undefined_bootstrap_draws"]["partial"] == 0
    for row in metrics if row["year"] in (2019, 2020)
)
result = {
    "status": (
        "conditional_negative_relative_value_association_only"
        if information_gate else "closed_fixed_H8_relative_value_association_failed"
    ),
    "information_gate_passed": information_gate,
    "metrics": metrics,
    "predictive_model_fits": 0,
    "candidate_accounts": 0,
    "historical_research_passed": False,
    "claim_limit": (
        "Conditional current-vintage weekly H.8 association on reused pi2 "
        "development, not first-publication-qualified factor or SOXX alpha."
    ),
}
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"])
    R.log_metrics(
        source_known_2019_days=support["2019"]["eligible_days"],
        source_known_2020_days=support["2020"]["eligible_days"],
        predictive_model_fits=0,
        candidate_accounts=0,
    )
    R.save_objects(
        frozen_plan=PLAN, paired_source=pair, daily_pairs=daily,
        metrics=metrics, source_receipts={
            "H8": source_receipt, "physical_labels": label_receipt
        }, audit=source_gate, result=result,
    )
    rec = R.get_recorder()
    recorder_id = rec.id
    experiment_id = str(rec.experiment_id)
summary = {
    "experiment": CARD.name,
    "plan_sha256": FREEZE["plan_sha256"],
    "source_gate": source_gate,
    "result": result,
    "recorder_id": recorder_id,
    "experiment_id": experiment_id,
    "recorder_uri": str(CARD / "qlib_recorder"),
    "recorder_local_dir": str(CARD / "qlib_recorder" / experiment_id / recorder_id),
    "artifact_policy": PLAN["artifact_policy"],
}
publish_evidence_manifest(CARD, summary)
print(json.dumps(result, indent=2, allow_nan=False))
