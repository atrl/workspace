"""Frozen Stage1/SOXX minute-path risk and loss decomposition on Qlib source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

from src.analysis.stage1_raw_minute_risk import point_price_parity, revalue
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_calendar(raw: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    schedule = xc.get_calendar("XNYS", start=start, end=end).schedule
    expected = {pd.Timestamp(day).tz_localize(None):
        int((pd.Timestamp(row.close).tz_convert("America/New_York")
            - pd.Timestamp(row.open).tz_convert("America/New_York")).total_seconds() / 60)
        for day, row in schedule.iterrows()}
    data = raw.reset_index()
    data["day"] = pd.to_datetime(data.datetime).dt.normalize()
    data = data.loc[data.day.between(start, end)]
    grouped = data.groupby(["day", "instrument"], observed=True).agg(
        rows=("datetime", "size"), invalid=("$price_valid", lambda s: int(s.ne(1).sum()))).reset_index()
    grouped["expected"] = grouped.day.map(expected)
    if (grouped.day.nunique() != len(expected) or len(grouped) != 2 * len(expected)
            or grouped.rows.ne(grouped.expected).any()):
        raise ValueError("needs_data: full-minute XNYS calendar differs")
    return grouped


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
            raise ValueError("blocked: fixed minute attribution plan changed")
        return old
    if _sha(root / plan["minute_provider"] / "qlib_store_manifest.json") != plan["minute_provider_manifest_sha256"]:
        raise ValueError("blocked: native minute provider identity differs")
    binding = json.loads((root / plan["minute_nas_binding"]).read_text())
    if not binding["verified"] or binding["source_id"] != plan["minute_source_id"]:
        raise ValueError("blocked: minute source NAS snapshot identity differs")
    qlib.init(provider_uri=str(root / plan["minute_provider"]), region="us", kernels=1)
    stage, sa = _source(root, plan["stage_source"], ["native/A_active_research_replay_0"])
    raw, ra = _source(root, plan["raw_source"], ["inputs/raw_points", "native/raw_SOXX_0"])
    frame, fa = _source(root, plan["price_source"], ["inputs/frame"])
    a = stage["native/A_active_research_replay_0"]
    b = raw["native/raw_SOXX_0"]
    points = raw["inputs/raw_points"]
    if (len(a["authority"]) != plan["held_sessions"]
            or not a["authority"].trade_date.equals(b["authority"].trade_date)):
        raise ValueError("blocked: frozen Stage1/SOXX account dates differ")
    bars = D.features(["SOXL", "SOXX"], ["$open", "$high", "$low", "$close", "$price_valid"],
        start_time=plan["training_start"] + " 09:31:00",
        end_time=plan["held_terminal"] + " 16:00:00", freq="1min")
    training = _source_calendar(bars, plan["training_start"], plan["training_end"])
    held = _source_calendar(bars, plan["held_start"], plan["held_terminal"])
    invalid_training = training.loc[training.invalid.gt(0), ["day", "instrument", "invalid"]]
    if (training.day.nunique() != plan["training_calendar_days"]
            or held.day.nunique() != plan["held_calendar_days"]
            or int(held.invalid.sum()) != 0
            or invalid_training.day.dt.strftime("%Y-%m-%d").tolist() != plan["training_invalid_dates"]):
        raise ValueError("needs_data: minute quality mask differs from frozen plan")
    wide = []
    for symbol in ("SOXL", "SOXX"):
        source = bars.xs(symbol, level="instrument")[["$open", "$high", "$low", "$close"]]
        source.columns = [symbol + "_" + name.removeprefix("$") for name in source]
        wide.append(source)
    quotes = pd.concat(wide, axis=1).sort_index()
    quote_days = quotes.index.normalize()
    quotes = quotes.loc[(quote_days >= pd.Timestamp(plan["held_start"]))
        & (quote_days <= pd.Timestamp(plan["held_terminal"]))]
    parity = point_price_parity(points, quotes)
    stage_path, stage_risk = revalue(a, quotes, initial_cash=plan["initial_cash"])
    soxx_path, soxx_risk = revalue(b, quotes, initial_cash=plan["initial_cash"])
    matched = frame["inputs/frame"]
    matched = matched.loc[matched.trade_date.isin(a["authority"].trade_date)].copy()
    if len(matched) != plan["held_sessions"]:
        raise ValueError("blocked: matched price-frame dates differ")
    day = a["authority"][["trade_date", "strategy_return", "ending_soxl_weight", "ending_cash_weight"]].copy()
    day["soxx_return"] = b["authority"].strategy_return.to_numpy(float)
    day = day.merge(matched[["trade_date", "soxl_open", "soxl_recovery", "close_soxl", "soxl_exit_open"]],
        on="trade_date", validate="one_to_one")
    day["soxl_open_next_return"] = day.soxl_exit_open / day.soxl_open - 1.
    day["soxl_open_recovery_return"] = day.soxl_recovery / day.soxl_open - 1.
    day["soxl_recovery_close_return"] = day.close_soxl / day.soxl_recovery - 1.
    day["soxl_close_next_return"] = day.soxl_exit_open / day.close_soxl - 1.
    day["relative_log_return"] = np.log1p(day.strategy_return) - np.log1p(day.soxx_return)
    worst = day.nsmallest(12, "strategy_return").copy()
    parts = worst[["soxl_open_recovery_return", "soxl_recovery_close_return", "soxl_close_next_return"]].abs()
    dominant = parts.idxmax(axis=1).value_counts().to_dict()
    categories = {}
    for name, mask in (("soxl_down_10pct", day.soxl_open_next_return.le(-.10)),
                       ("soxl_up_10pct", day.soxl_open_next_return.ge(.10))):
        categories[name] = {"days": int(mask.sum()),
            "relative_log_sum": float(day.loc[mask, "relative_log_return"].sum()),
            "relative_win_days": int(day.loc[mask, "relative_log_return"].gt(0).sum()),
            "mean_ending_cash_weight": float(day.loc[mask, "ending_cash_weight"].mean())}
    result = {"experiment": card.name, "plan_sha256": digest,
        "decision": "read_only_minute_loss_attribution_no_policy_promotion",
        "objective_achieved": False, "historical_research_passed": False,
        "training_source_days": int(training.day.nunique()),
        "training_invalid_symbol_days": int(training.invalid.gt(0).sum()),
        "held_source_days": int(held.day.nunique()), "held_invalid_symbol_days": 0,
        "raw_point_parity": parity, "stage1_risk": stage_risk, "soxx_risk": soxx_risk,
        "worst_12_dominant_soxl_segment": dominant, "descriptive_groups": categories,
        "new_model_fits": 0, "new_native_accounts": 0,
        "provider_uri": str(root / plan["minute_provider"]),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/minute/*", "analysis/daily"]}}
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=card.name, uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_audits={"stage": sa, "raw": ra, "price": fa},
            training_calendar=training, held_calendar=held)
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            valuation=(root / "src/analysis/stage1_raw_minute_risk.py").read_text())
        recorder.save_objects(artifact_path="analysis/minute", stage1=stage_path, soxx=soxx_path)
        recorder.save_objects(artifact_path="analysis", daily=day, worst=worst, result=result)
        recorder.log_metrics(stage1_fullminute_observed_drawdown=float(stage_risk["observed_drawdown"]),
            stage1_fullminute_upper_bound=float(stage_risk["conservative_ohlc_bound"]))
    result.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id),
        recorder_uri=str(tracking),
        recorder_local_dir=str(tracking / str(recorder.experiment_id) / recorder.id))
    worst[["trade_date", "strategy_return", "soxx_return", "soxl_open_next_return",
           "soxl_open_recovery_return", "soxl_recovery_close_return",
           "soxl_close_next_return", "ending_soxl_weight", "ending_cash_weight"]].to_csv(
        card / "worst_days.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": result["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": result["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 原 Stage1 全分钟亏损与风险归因\n\n"
        f"封存NAS原始NONE Qlib 1分钟：开发期{result['training_source_days']}日、"
        f"其中{result['training_invalid_symbol_days']}个标的日无效；"
        f"账户期{result['held_source_days']}日两资产全部有效。"
        f"原账户{parity['point_rows']}个固定原价点与分钟源逐点一致。"
        f"原Stage1分钟收盘/原事件最大回撤{stage_risk['observed_drawdown']:.4%}，"
        f"异步高低价保守界{stage_risk['conservative_ohlc_bound']:.4%}；"
        "界超过30%不等于实际同步组合回撤已违约，应按来源限定风险资格报告。\n\n"
        f"最差12个Stage1持有区间的SOXL最大单段跌幅，"
        f"隔夜{dominant.get('soxl_close_next_return',0)}日、"
        f"10:32后至收盘{dominant.get('soxl_recovery_close_return',0)}日。"
        f"SOXL跌逾10%的{categories['soxl_down_10pct']['days']}日，Stage1合计相对SOXX对数贡献"
        f"{categories['soxl_down_10pct']['relative_log_sum']:+.4f}；"
        f"涨逾10%的{categories['soxl_up_10pct']['days']}日为"
        f"{categories['soxl_up_10pct']['relative_log_sum']:+.4f}。"
        "这些事后组只是归因，不生成分钟触发阈值。"
        "单独减险不能自动解决现金机会成本和反弹参与；没有新策略账户或SOXX历史通过。\n")
    atomic_json(result, card / "summary.json")
    return result
