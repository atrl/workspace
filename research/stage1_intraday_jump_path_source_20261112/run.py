"""Existing Qlib one-minute SOXX path source; no target or model access."""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.workflow import R

from src.core.recorder_run import publish_evidence_manifest

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path.cwd()
CARD = ROOT / "research/stage1_intraday_jump_path_source_20261112"
PLAN = json.loads((CARD / "plan.json").read_text())
FREEZE = json.loads((CARD / "freeze_receipt.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(CARD / "plan.json") == FREEZE["plan_sha256"]
assert sha(Path(__file__)) == FREEZE["code_sha256"]
minute_provider = ROOT / PLAN["minute_provider"]
daily_provider = ROOT / PLAN["daily_provider"]
assert sha(minute_provider / "qlib_store_manifest.json") == PLAN["minute_manifest_sha256"]
assert sha(daily_provider / "qlib_store_manifest.json") == PLAN["daily_manifest_sha256"]

qlib.init(provider_uri=str(minute_provider), region="us", kernels=1)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    bars = D.features(
        ["SOXX"], ["$raw_open", "$raw_close", "$price_valid"],
        start_time=PLAN["source_start"], end_time=PLAN["source_end"], freq="1min",
    ).xs("SOXX", level="instrument").sort_index()
assert not bars.index.has_duplicates
rows = []
for day, group in bars.groupby(bars.index.normalize(), sort=True):
    stamps = pd.DatetimeIndex(group.index)
    count = len(group)
    expected_first = day + pd.Timedelta(hours=9, minutes=31)
    expected_last = day + pd.Timedelta(hours=13 if count == 210 else 16)
    consecutive = bool(
        count in (210, 390)
        and stamps[0] == expected_first
        and stamps[-1] == expected_last
        and np.all(np.diff(stamps.asi8) == 60_000_000_000)
    )
    source_valid = bool(
        consecutive
        and group["$price_valid"].eq(1).all()
        and group[["$raw_open", "$raw_close"]].notna().all().all()
        and group["$raw_open"].gt(0).all()
        and group["$raw_close"].gt(0).all()
    )
    record = {
        "source_day": day,
        "bars": count,
        "terminal_bar": stamps[-1],
        "source_valid": source_valid,
        "reason": "valid" if source_valid else "missing_or_invalid_price_or_clock",
        "available_at": stamps[-1] + pd.Timedelta(minutes=1) if source_valid else pd.NaT,
        "intraday_jump_proxy": np.nan,
        "intraday_rv": np.nan,
        "intraday_bv": np.nan,
    }
    if source_valid:
        closing = group["$raw_close"].to_numpy(float)
        opening = group["$raw_open"].to_numpy(float)
        minute_returns = np.r_[
            np.log(closing[0] / opening[0]),
            np.log(closing[1:] / closing[:-1]),
        ]
        rv = float(np.square(minute_returns).sum())
        bv = float(
            (np.pi / 2.0) * count / (count - 1)
            * np.abs(minute_returns[1:] * minute_returns[:-1]).sum()
        )
        if rv > 0 and np.isfinite([rv, bv]).all():
            record.update(
                intraday_jump_proxy=float(np.clip((rv - bv) / rv, 0.0, 1.0)),
                intraday_rv=rv,
                intraday_bv=bv,
            )
        else:
            record.update(source_valid=False, reason="nonpositive_or_nonfinite_rv")
    rows.append(record)
daily_source = pd.DataFrame(rows).set_index("source_day").sort_index()
assert daily_source.index.is_unique

qlib.init(provider_uri=str(daily_provider), region="us", kernels=1)
calendar = pd.DatetimeIndex(D.calendar(start_time="2017-01-01", end_time="2022-12-22"))
price = (
    D.features(["SOXX"], ["$close"], start_time=calendar.min(), end_time=calendar.max())
    .xs("SOXX", level="instrument")["$close"]
    .reindex(calendar)
    .astype(float)
)
assert price.notna().all() and price.gt(0).all()
returns = np.log(price).diff()
squared = returns.pow(2)
rv20 = squared.rolling(20, min_periods=10).sum()
bv20 = (
    (np.pi / 2.0) * returns.abs() * returns.shift(1).abs()
).rolling(20, min_periods=10).sum()
daily_controls = pd.DataFrame(
    {
        "daily_jump_ratio20": ((rv20 - bv20).clip(lower=0) / rv20).clip(0, 1),
        "daily_rv20": rv20,
        "daily_down_share20": (
            returns.clip(upper=0).pow(2).rolling(20, min_periods=10).sum() / rv20
        ),
        "prior_return": returns,
        "mom20": returns.rolling(20).sum(),
    },
    index=calendar,
).shift(1)
prior = daily_source.reindex(calendar).shift(1)
decision = daily_controls.join(
    prior[[
        "source_valid", "intraday_jump_proxy", "intraday_rv",
        "intraday_bv", "available_at", "bars",
    ]]
).loc["2018-10-15":"2022-12-22"].copy()
decision["prior_source_day"] = pd.Series(calendar, index=calendar).shift(1).reindex(
    decision.index
)
decision.index.name = "decision_day"
known = decision.loc[
    decision.source_valid.fillna(False)
    & decision[[
        "intraday_jump_proxy", "daily_jump_ratio20", "daily_rv20",
        "daily_down_share20", "prior_return", "mom20",
    ]].notna().all(axis=1)
].copy()
assert known.available_at.lt(known.index + pd.Timedelta(hours=9, minutes=29)).all()
assert known.prior_source_day.lt(known.index).all()
per_year = known.groupby(known.index.year).size().to_dict()
assert per_year.get(2018, 0) >= 50
assert all(per_year.get(year, 0) >= 200 for year in (2019, 2020, 2021, 2022))
pairwise_spearman = float(
    known.intraday_jump_proxy.corr(known.daily_jump_ratio20, method="spearman")
)
assert abs(pairwise_spearman) < 0.995
training = known.loc[known.index.year <= 2019]
control_columns = [
    "daily_jump_ratio20", "daily_rv20", "daily_down_share20",
    "prior_return", "mom20",
]
matrix = np.column_stack([
    np.ones(len(training)), training[control_columns].to_numpy(float)
])
candidate = training.intraday_jump_proxy.to_numpy(float)
residual = candidate - matrix @ np.linalg.lstsq(matrix, candidate, rcond=None)[0]
residual_fraction = float(np.var(residual) / np.var(candidate))
assert residual_fraction > 0.05
invalid_source_days = daily_source.index[
    ~daily_source.source_valid
].strftime("%Y-%m-%d").tolist()
source_audit = {
    "source_days": len(daily_source),
    "decision_days": len(decision),
    "valid_decision_days": len(known),
    "valid_decision_days_by_year": per_year,
    "invalid_source_days": invalid_source_days,
    "bar_count_distribution": daily_source.bars.value_counts().to_dict(),
    "proxy_quantiles": known.intraday_jump_proxy.quantile(
        [0, .05, .5, .95, 1]
    ).to_dict(),
    "intraday_vs_daily_jump_spearman": pairwise_spearman,
    "train_residual_variance_fraction": residual_fraction,
    "minute_manifest_sha256": PLAN["minute_manifest_sha256"],
    "daily_manifest_sha256": PLAN["daily_manifest_sha256"],
    "decision_clock_pass": True,
    "target_outcomes_read": False,
    "new_fits": 0,
    "candidate_accounts": 0,
    "claim_limit": (
        "Source and nonduplicate coordinate only. One-minute microstructure "
        "and historical first receipt are unqualified; no economic information or SOXX pass."
    ),
}
with R.start(experiment_name=CARD.name, uri=str(CARD / "qlib_recorder")):
    R.log_params(plan_sha256=FREEZE["plan_sha256"])
    R.log_metrics(
        valid_decision_days=len(known),
        intraday_vs_daily_jump_spearman=pairwise_spearman,
        train_residual_variance_fraction=residual_fraction,
        new_fits=0,
        candidate_accounts=0,
    )
    R.save_objects(
        frozen_plan=PLAN, daily_source=daily_source,
        decision_source=decision, source_audit=source_audit,
    )
    recorder = R.get_recorder()
    recorder_id = recorder.id
    experiment_id = str(recorder.experiment_id)
summary = {
    "experiment": CARD.name,
    "decision": "conditional_novel_intraday_path_source_only",
    "plan_sha256": FREEZE["plan_sha256"],
    "source_audit": source_audit,
    "new_fits": 0,
    "candidate_accounts": 0,
    "historical_research_passed": False,
    "recorder_id": recorder_id,
    "experiment_id": experiment_id,
    "recorder_uri": str(CARD / "qlib_recorder"),
    "recorder_local_dir": str(CARD / "qlib_recorder" / experiment_id / recorder_id),
    "artifact_policy": PLAN["artifact_policy"],
}
publish_evidence_manifest(CARD, summary)
print(json.dumps(source_audit, indent=2, allow_nan=False, default=str))
