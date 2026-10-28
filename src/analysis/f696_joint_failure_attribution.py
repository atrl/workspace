"""Read-only annual attribution of failed exact-parent action-value screens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: attribution plan changed")
        return old
    for source in plan["sources"].values():
        if _sha(root / source["card"] / "summary.json") != source["summary_sha256"]:
            raise ValueError("blocked: frozen information source changed")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    labels, la = _source(root, plan["sources"]["labels"], ["analysis/labels"])
    frame, fa = _source(root, plan["sources"]["frame"], ["inputs/covered_frame"])
    daily, da = _source(root, plan["sources"]["daily"], ["analysis/predictions"])
    early, ea = _source(root, plan["sources"]["early"], ["analysis/predictions"])
    value = labels["analysis/labels"]
    features = frame["inputs/covered_frame"][["trade_date", "vix", "soxx_vol20", "soxx_drawdown60"]]
    joined = value.merge(features, on="trade_date", validate="one_to_one")
    eligible = joined.loc[joined.eligible].copy()
    eligible["year"] = pd.DatetimeIndex(eligible.trade_date).year
    support = eligible.groupby(["route", "year"], observed=True).agg(
        rows=("local_advantage", "size"), local_value_mean=("local_advantage", "mean"),
        local_positive_fraction=("local_advantage", lambda x: float(x.gt(0).mean())),
        mean_cash_before=("pretrade_cash_weight", "mean"),
        mean_account_drawdown=("known_drawdown", "mean"),
        mean_beta_cap=("current_beta_cap", "mean"),
        mean_vix=("vix", "mean"), mean_soxx_vol20=("soxx_vol20", "mean"),
        mean_soxx_drawdown60=("soxx_drawdown60", "mean")).reset_index()
    predictions = []
    for name, source in (("prior_market", daily), ("volume_timing", early)):
        for route, result in source["analysis/predictions"].items():
            actual = eligible.loc[eligible.route.eq(route), ["trade_date", "local_advantage"]]
            matched = result.merge(actual, on="trade_date", how="left", validate="one_to_one")
            if not np.allclose(matched.actual.to_numpy(float), matched.local_advantage.to_numpy(float), atol=1e-12):
                raise ValueError("blocked: attribution prediction/label identity differs")
            matched["year"] = pd.DatetimeIndex(matched.trade_date).year
            for year, group in matched.groupby("year"):
                predictions.append({"screen": name, "route": route, "year": int(year),
                    "rows": len(group), "actual_mean": float(group.actual.mean()),
                    "candidate_mean": float(group.candidate.mean()),
                    "control_mean": float(group.control.mean()),
                    "actual_positive_fraction": float(group.actual.gt(0).mean()),
                    "candidate_positive_fraction": float(group.candidate.gt(0).mean()),
                    "control_positive_fraction": float(group.control.gt(0).mean())})
    forecast = pd.DataFrame(predictions)
    if (len(support) != plan["expected_support_rows"] or len(forecast) != plan["expected_forecast_rows"]):
        raise ValueError("blocked: annual attribution row inventory differs")
    bind_2022 = forecast.loc[forecast.screen.eq("prior_market")
        & forecast.route.eq("binding_cash_to_soxl") & forecast.year.eq(2022)].iloc[0]
    scarce = support.loc[support.route.eq("nonbinding_soxl_cash_to_soxx") & support.year.eq(2022)].iloc[0]
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "read_only_failure_attribution_no_admitted_candidate",
        "objective_achieved": False, "historical_research_passed": False,
        "source_recorder_ids": {role: value["recorder_id"] for role, value in plan["sources"].items()},
        "source_audits": {"labels": la, "frame": fa, "daily": da, "early": ea},
        "binding_2022_prior_market_predicted_positive_fraction": float(bind_2022.candidate_positive_fraction),
        "binding_2022_realized_positive_fraction": float(bind_2022.actual_positive_fraction),
        "nonbinding_2022_eligible_rows": int(scarce.rows),
        "new_model_fits": 0, "new_native_accounts": 0, "new_recorders": 0,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [], "reused_native_recorders": [v["card"] for v in plan["sources"].values()]}}
    support.to_csv(card / "annual_label_state.csv", index=False)
    forecast.to_csv(card / "annual_prediction_shift.csv", index=False)
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nreused_sources: "
        + json.dumps(summary["source_recorder_ids"], ensure_ascii=False) + "\n")
    (card / "conclusion.md").write_text("# 精确f696双层动作的信息失败只读归因\n\n"
        f"2022受限层既有日线屏的正局部价值预测比例{bind_2022.candidate_positive_fraction:.1%}，"
        f"实际正标签比例{bind_2022.actual_positive_fraction:.1%}；"
        f"非受限层2022年只有{int(scarce.rows)}个资金可达且成熟标签。"
        "逐年标签、账户状态和两次固定屏的预测均值/正预测比例见两个sidecar。"
        "这是跨年漂移与支持度诊断，不证明某个账户状态或VIX阈值可交易；"
        "不能凭此修改已失败模型的符号、时间切分或开关。"
        "需要实质不同且首次可用时间可核验的决策机制先过匹配信息门，才能再开0bp完整账户。\n")
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
