"""Frozen event-time guard local labels on the dated f696 training account."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as xc
import pandas as pd

from src.analysis.stage1_event_guard_labels import build_labels
from src.data.minute_intake import atomic_json
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.data import D
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: frozen event-label plan changed")
        return old
    if (_sha(root / plan["provider_uri"] / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]
            or _sha(root / plan["provider_source"]["card"] / "summary.json")
               != plan["provider_source"]["summary_sha256"]):
        raise ValueError("blocked: event label Qlib provider source differs")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    source, sa = _source(root, plan["training_source"], ["analysis/native/dated_f696_0"])
    frame_source, fa = _source(root, plan["frame_source"], ["inputs/covered_frame"])
    book = source["analysis/native/dated_f696_0"]
    frame = frame_source["inputs/covered_frame"]
    if len(frame) != plan["expected_parent_sessions"] or not frame.trade_date.equals(book["authority"].trade_date):
        raise ValueError("blocked: dated account and f696 input calendar differ")
    bars = D.features(["SOXL", "SOXX"], ["$open", "$close", "$raw_open", "$price_valid"],
        start_time=plan["training_start"] + " 09:31:00",
        end_time=plan["training_end"] + " 16:00:00", freq="1min")
    wide = []
    for symbol in ("SOXL", "SOXX"):
        part = bars.xs(symbol, level="instrument").copy()
        part.columns = [symbol + "_" + item.removeprefix("$") for item in part]
        wide.append(part)
    quotes = pd.concat(wide, axis=1).sort_index()
    schedule = xc.get_calendar("XNYS", start=frame.trade_date.min(),
                               end=frame.trade_date.max()).schedule
    closes = {pd.Timestamp(day).tz_localize(None):
        pd.Timestamp(row.close).tz_convert("America/New_York").tz_localize(None)
        for day, row in schedule.iterrows()}
    state_path = card / "run_state.json"
    prior_attempts = 0
    prior = {"plan_sha256": digest}
    if state_path.exists():
        prior = json.loads(state_path.read_text())
        prior_attempts = int(prior.get("label_attempts", 0))
        if prior["plan_sha256"] != digest or prior_attempts >= plan["budget"]["label_pass_attempts"]:
            raise ValueError("blocked: frozen event-label attempt already consumed")
    atomic_json({**prior, "label_attempts": prior_attempts + 1,
                 "status": "label_pass_started"}, state_path)
    labels, audit = build_labels(frame, book, quotes, tariff=plan["tariff"],
                                 session_closes=closes, initial_cash=plan["initial_cash"])
    atomic_json({**prior, "label_attempts": prior_attempts + 1,
                 "status": "label_pass_completed", "event_rows": audit["rows"]}, state_path)
    if (audit["calendar_days"] != plan["expected_parent_sessions"]
            or audit["rows"] > plan["budget"]["maximum_event_rows"]
            or audit["parent_nav_max_error_usd"] > 1e-7):
        raise ValueError("blocked: event label source/budget/parent parity differs")
    eligible = labels.loc[labels.funded_change].copy()
    eligible["year"] = pd.DatetimeIndex(eligible.trade_date).year
    perday = eligible.groupby(["trade_date", "year"], observed=True).local_advantage.agg(
        rows="size", any_positive=lambda x: bool(x.gt(0).any()),
        any_negative=lambda x: bool(x.lt(0).any())).reset_index()
    annual = perday.groupby("year").agg(days=("trade_date", "size"),
        event_rows=("rows", "sum"), positive_days=("any_positive", "sum"),
        negative_days=("any_negative", "sum")).reset_index()
    years20 = int(annual.days.ge(20).sum())
    last = int(annual.loc[annual.year.eq(2022), "days"].sum())
    both_sign = int((annual.positive_days.ge(10) & annual.negative_days.ge(10)).sum())
    gate = (len(perday) >= plan["support_gate"]["minimum_days"]
        and years20 >= plan["support_gate"]["years_with_20_days"]
        and last >= plan["support_gate"]["minimum_2022_days"]
        and both_sign >= plan["support_gate"]["years_with_10_each_sign"])
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=card.name, uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_audits={"training": sa, "frame": fa},
            provider_manifest=json.loads((root / plan["provider_uri"] / "qlib_store_manifest.json").read_text()))
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            labels=(root / "src/analysis/stage1_event_guard_labels.py").read_text(),
            clock=(root / "src/backtest/event_guard_clock.py").read_text())
        recorder.save_objects(artifact_path="analysis", labels=labels, annual=annual,
            perday=perday, audit=audit)
        recorder.log_metrics(funded_event_rows=float(len(eligible)),
            funded_event_days=float(len(perday)), label_support_gate_pass=float(gate))
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "event_guard_label_support_pass_needs_information" if gate
                    else "not_promoted_event_guard_label_support_failed",
        "objective_achieved": False, "historical_research_passed": False,
        "source_audit": audit, "support_gate_pass": bool(gate),
        "funded_event_days": len(perday), "mature_funded_rows": len(eligible),
        "year_support": annual.to_dict("records"),
        "new_model_fits": 0, "new_native_accounts": 0,
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(root / plan["provider_uri"]),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/labels", "analysis/perday", "analysis/annual", "analysis/audit"]}}
    annual.to_csv(card / "annual_support.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# Stage1事件减险同状态分钟标签\n\n"
        f"在逐层点时f696训练账户上，保留{audit['calendar_days']}个日期、"
        f"生成{audit['rows']}条分钟候选、其中{len(eligible)}条实际整股卖SOXL并买SOXX。"
        f"可行动日期{len(perday)}，最近完整开发年2022为{last}日；"
        f"原动作到次日09:47净值最大误差{audit['parent_nav_max_error_usd']:.3g}美元。"
        "无SOXL、原恢复动作、无效分钟及末端复位不全保留未知，不以未来日价填补。"
        f"固定来源支持门：{summary['decision']}。本卡0新模型、0新账户；"
        "局部下一开盘复位不代表事件策略的完整资金路径。\n")
    atomic_json(summary, card / "summary.json")
    return summary
