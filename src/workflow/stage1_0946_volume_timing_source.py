"""Qlib-native provenance and novelty gate for a 09:46 volume timing signal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.early_volume_timing_residual import volume_timing_residual
from src.data.minute_intake import atomic_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.data import D
    from qlib.workflow import R
    from src.workflow.minute_action_value import _source

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: fixed source plan changed")
        return old
    if (_sha(root / plan["minute_provider"] / "qlib_store_manifest.json") != plan["minute_provider_manifest_sha256"]
            or _sha(root / plan["reference"]["card"] / "summary.json") != plan["reference"]["summary_sha256"]):
        raise ValueError("blocked: minute provider or f696 reference differs")
    binding = json.loads((root / plan["minute_nas_binding"]).read_text())
    if not binding["verified"] or binding["evidence"]["sha256"] != plan["minute_nas_archive_sha256"]:
        raise ValueError("blocked: minute source NAS receipt differs")
    qlib.init(provider_uri=str(root / plan["minute_provider"]), region="us", kernels=1)
    sources, source_audit = _source(root, plan["reference"], ["inputs/covered_frame"])
    frame = sources["inputs/covered_frame"]
    if len(frame) != plan["expected_sessions"]:
        raise ValueError("blocked: f696 source calendar differs")
    dates = pd.DatetimeIndex(frame.trade_date)
    bars = D.features(["SOXL", "SOXX"], ["$open", "$close", "$volume"],
        start_time=dates.min() + pd.Timedelta(hours=9, minutes=31),
        end_time=dates.max() + pd.Timedelta(hours=9, minutes=45), freq="1min")
    coordinate = volume_timing_residual(bars, dates)
    known = coordinate.source_known
    if (len(coordinate) != len(frame) or int(known.sum()) != plan["expected_known"]
            or coordinate.loc[~known, "trade_date"].dt.strftime("%Y-%m-%d").tolist() != plan["expected_unknown_dates"]
            or not coordinate.loc[known, "first_available_at"].eq(
                coordinate.loc[known, "trade_date"] + pd.Timedelta(hours=9, minutes=46)).all()):
        raise ValueError("needs_data: minute bar coverage or availability differs")
    if float(coordinate.loc[known, "volume_timing_residual"].std()) <= plan["gate"]["minimum_std"]:
        raise ValueError("blocked: volume timing coordinate is constant")
    joined = frame[["trade_date", *plan["control_features"]]].merge(coordinate,
        on="trade_date", how="left", validate="one_to_one")
    train = joined.loc[joined.trade_date.lt(pd.Timestamp(plan["novelty_cutoff"])) & joined.source_known]
    correlations = {name: float(train["volume_timing_residual"].corr(train[name], method="spearman"))
                    for name in [*plan["control_features"], "soxl_early_log_return", "soxx_early_log_return"]}
    maximum = max(abs(value) for value in correlations.values() if np.isfinite(value))
    source_pass = maximum < plan["gate"]["maximum_abs_train_spearman"]
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=card.name, uri=str(tracking)):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, source_audit=source_audit)
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            coordinate=(root / "src/data/early_volume_timing_residual.py").read_text())
        recorder.save_objects(artifact_path="analysis", coordinate=coordinate, correlations=correlations)
        recorder.log_metrics(known_source_sessions=float(known.sum()),
                             maximum_abs_train_spearman=float(maximum))
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "source_novelty_pass_needs_matched_information" if source_pass
                    else "not_promoted_volume_timing_novelty_failed",
        "objective_achieved": False, "historical_research_passed": False,
        "known": int(known.sum()), "unknown": int((~known).sum()),
        "standard_deviation": float(coordinate.loc[known, "volume_timing_residual"].std()),
        "maximum_abs_train_spearman": maximum, "train_correlations": correlations,
        "historical_first_receipt_qualified": False,
        "new_model_fits": 0, "new_native_accounts": 0,
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking),
        "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(root / plan["minute_provider"]),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "analysis/coordinate", "analysis/correlations"]}}
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 09:46 双ETF成交量时序残差来源门\n\n"
        "用原生Qlib分钟源的09:31至09:45十五根完整bar，计算每只ETF成交额加权分钟涨跌减去等权分钟涨跌，"
        "再取SOXL减三倍SOXX。它是跨ETF压力代理，不是SOXL真实跟踪误差；旧日首次送达未经认证。"
        f"来源已知{summary['known']}/{len(frame)}日，缺失日保持未知；训练期与控制特征最高绝对秩相关{maximum:.4f}。"
        f"决定：`{summary['decision']}`。没有读取未来局部价值、拟合或资金账户。"
        "只有来源与重复性通过才另冻精确f696标签的信息检验。\n")
    atomic_json(summary, card / "summary.json")
    return summary
