"""Frozen minute-grain, fee-net allocation labels on the dated Stage1 parent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as xc
import pandas as pd

from src.analysis.stage1_intraday_beta_neutral_labels import build_labels
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
            raise ValueError("blocked: fixed minute label plan changed")
        return old
    if _sha(root / plan["provider_uri"] / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]:
        raise ValueError("blocked: minute provider manifest differs")
    for key in ("training_source", "frame_source"):
        source = plan[key]
        if _sha(root / source["card"] / "summary.json") != source["summary_sha256"]:
            raise ValueError("blocked: dated source summary differs")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    source, sa = _source(root, plan["training_source"], ["analysis/native/dated_f696_0"])
    frame_source, fa = _source(root, plan["frame_source"], ["inputs/covered_frame"])
    book, frame = source["analysis/native/dated_f696_0"], frame_source["inputs/covered_frame"]
    if (len(frame) != plan["expected_parent_sessions"]
            or not frame.trade_date.equals(book["authority"].trade_date)
            or book["audit"]["status"] != "passed"
            or book["audit"]["slippage_bps"] != 0):
        raise ValueError("blocked: dated parent native account differs")
    bars = D.features(["SOXL", "SOXX"],
        ["$open", "$close", "$raw_open", "$price_valid"],
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
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "attempts": 0}
    if (progress["plan_sha256"] != digest
            or progress["attempts"] >= plan["budget"]["label_pass_attempts"]):
        raise ValueError("blocked: frozen label attempt consumed")
    progress["attempts"] += 1
    progress["status"] = "label_build_started"
    atomic_json(progress, progress_path)
    labels, masks, audit = build_labels(frame, book, quotes, closes=closes,
        tariff=plan["tariff"], initial_cash=plan["initial_cash"], action=plan["action"])
    progress["status"] = "label_build_completed"
    progress["funded_rows"] = audit["funded_rows"]
    atomic_json(progress, progress_path)
    if (audit["sessions"] != plan["expected_parent_sessions"]
            or audit["event_slots"] != plan["expected_event_slots"]
            or audit["event_slots"] > plan["budget"]["maximum_event_slots"]
            or audit["parent_nav_max_error_usd"] > plan["support_gate"]["parent_nav_max_error_usd"]):
        raise ValueError("blocked: minute label source or parent parity differs")
    eligible = labels.loc[labels.funded_change].copy()
    eligible["year"] = pd.DatetimeIndex(eligible.trade_date).year
    perday = eligible.groupby(["trade_date", "year"], observed=True).local_advantage.agg(
        rows="size", any_positive=lambda x: bool(x.gt(0).any()),
        any_negative=lambda x: bool(x.lt(0).any())).reset_index()
    annual = perday.groupby("year").agg(days=("trade_date", "size"),
        event_rows=("rows", "sum"), positive_days=("any_positive", "sum"),
        negative_days=("any_negative", "sum")).reset_index()
    gate = (len(perday) >= plan["support_gate"]["minimum_funded_days"]
        and int(annual.days.ge(20).sum()) >= plan["support_gate"]["years_with_20_days"]
        and int(annual.loc[annual.year.eq(2022), "days"].sum()) >= plan["support_gate"]["minimum_2022_days"]
        and int((annual.positive_days.ge(10) & annual.negative_days.ge(10)).sum())
            >= plan["support_gate"]["years_with_10_each_sign"])
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=card.name, uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_audits={"parent": sa, "frame": fa},
            provider_manifest=json.loads((root / plan["provider_uri"] / "qlib_store_manifest.json").read_text()))
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            labels=(root / "src/analysis/stage1_intraday_beta_neutral_labels.py").read_text())
        recorder.save_objects(artifact_path="analysis", labels=labels, masks=masks,
            perday=perday, annual=annual, audit=audit)
        recorder.log_metrics(funded_minute_rows=float(audit["funded_rows"]),
            funded_minute_days=float(audit["funded_days"]),
            support_gate_pass=float(gate))
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "minute_beta_neutral_label_support_pass_needs_information" if gate
                    else "not_promoted_minute_beta_neutral_label_support_failed",
        "objective_achieved": False, "historical_research_passed": False,
        "source_audit": audit, "support_gate_pass": bool(gate),
        "year_support": annual.to_dict("records"), "new_model_fits": 0,
        "new_native_accounts": 0, "recorder_id": recorder.id,
        "experiment_id": str(recorder.experiment_id), "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(root / plan["provider_uri"]),
        "historical_first_receipt_qualified": False,
        "artifact_policy": {"policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/*"]}}
    recorder.save_objects(artifact_path="analysis", summary=summary)
    annual.to_csv(card / "annual_support.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 分钟级名义beta守恒配置标签\n\n"
        f"逐层点时父本1056日的{audit['event_slots']}条盘中时点，"
        f"实际双边整股成交并完成下一开盘共同复位的标签{audit['funded_rows']}条，"
        f"覆盖{audit['funded_days']}日；原父本下一开盘NAV最大误差"
        f"{audit['parent_nav_max_error_usd']:.3g}美元。"
        f"决定`{summary['decision']}`；逐年正负支持见annual_support.csv。"
        "这是SOXL加现金转SOXX的名义beta守恒配置动作，不是暴跌保护。"
        "本卡零新拟合、零新账户；局部价值不可相加为策略收益，"
        "历史分钟首收未认证，SOXX硬门仍未通过。\n")
    atomic_json(summary, card / "summary.json")
    return summary
