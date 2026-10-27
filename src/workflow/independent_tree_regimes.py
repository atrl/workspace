"""Read-only regime expansion of the sealed tree-allocation Qlib accounts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import paired_metrics
from src.data.minute_intake import atomic_json


def finalize(*, card: Path) -> dict:
    import qlib
    from qlib.workflow import R

    card = Path(card).resolve()
    summary_path = card / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("regime_evidence", {}).get("status") == "passed":
        return summary
    qlib.init(provider_uri=summary["provider_uri"], region="us", kernels=1)
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(recorder_id=summary["recorder_id"], experiment_name=summary["experiment"])
        annual = recorder.load_object("analysis/annual")
        # The SOXX authority is the verified source object, kept in the input Recorder.
        from src.workflow.minute_action_value import _source
        root = card.parents[1]
        plan = json.loads((card / "plan.json").read_text())
        source, _ = _source(root, plan["source"], ["analysis/native/reused_outer_soxx_ensemble_0"])
        benchmark = source["analysis/native/reused_outer_soxx_ensemble_0"]["authority"]
        quarters, windows = [], []
        for case in [row["case"] for row in summary["result"]]:
            path = recorder.load_object("analysis/native/" + case)["authority"]
            if not path.trade_date.equals(benchmark.trade_date):
                raise ValueError("blocked: regime account calendar differs")
            dates = pd.DatetimeIndex(path.signal_date)
            for period in sorted(dates.to_period("Q").unique()):
                mask = dates.to_period("Q") == period
                metric = paired_metrics(path.loc[mask].reset_index(drop=True), benchmark.loc[mask].reset_index(drop=True))
                quarters.append({"case": case, "quarter": str(period), "sessions": int(mask.sum()),
                    "start_signal": str(dates[mask][0].date()), "end_signal": str(dates[mask][-1].date()),
                    "relative_wealth": float(metric["relative_wealth"]),
                    "return_pct": float(metric["return_pct"]),
                    "benchmark_return_pct": float(metric["benchmark_return_pct"])})
            excess = np.log1p(path.strategy_return.to_numpy(float)) - np.log1p(benchmark.strategy_return.to_numpy(float))
            rolling = pd.Series(excess).rolling(126).sum()
            for end in range(125, len(path)):
                windows.append({"case": case, "start_signal": str(dates[end - 125].date()),
                    "end_signal": str(dates[end].date()), "sessions": 126,
                    "relative_log_excess": float(rolling.iloc[end]),
                    "relative_wealth": float(np.exp(rolling.iloc[end])),
                    "positive": bool(rolling.iloc[end] > 1e-8)})
        quarter_frame, window_frame = pd.DataFrame(quarters), pd.DataFrame(windows)
        if quarter_frame.groupby("case").size().ne(16).any() or window_frame.groupby("case").size().ne(819).any():
            raise ValueError("blocked: complete quarter/window evidence differs")
        recorder.save_objects(artifact_path="analysis", quarter=quarter_frame, window_detail=window_frame)
        recorder.save_objects(artifact_path="sources", regime_workflow=Path(__file__).read_text())
    annual.to_csv(card / "annual_metrics.csv", index=False)
    quarter_frame.to_csv(card / "quarter_metrics.csv", index=False)
    compact = []
    for case, group in window_frame.groupby("case", sort=False):
        worst, best = group.loc[group.relative_wealth.idxmin()], group.loc[group.relative_wealth.idxmax()]
        compact.append({"case": case, "positive": int(group.positive.sum()), "total": len(group),
            "worst": worst.to_dict(), "best": best.to_dict()})
    atomic_json({"cases": compact}, card / "window_summary.json")
    summary["regime_evidence"] = {"status": "passed", "annual_rows": len(annual),
        "quarter_rows": len(quarter_frame), "window_rows": len(window_frame),
        "windows_per_case": 819, "quarters_per_case": 16}
    atomic_json(summary, summary_path)
    with (card / "conclusion.md").open("a") as output:
        output.write("\n## 区间与风险解读\n\n")
        output.write("完整年度与两端部分年度见 `annual_metrics.csv`；每组 16 个季度见 `quarter_metrics.csv`，全部 819 个连续126日窗口存于 Qlib Recorder `analysis/window_detail`，极值见 `window_summary.json`。\n\n")
        output.write("XGB放开版虽费用后超过SOXX，但平均名义 beta 约1.91、SOXL约63.6%，回撤约79%、CVaR5约-11.75%、窗口胜出61.8%；同一历史的固定50% SOXL+现金参考也超过SOXX，故不能把超额全归于择时。LGB放开版期末相对财富约0.778；加回旧限制后两模型现金均超过92%，显示限制在这组固定动作预测上显著压低参与。\n\n")
        output.write("训练标签1060行中，现金500、SOXL552、SOXX仅8，学习问题对SOXX选择支持很弱；共同现金起点的一步标签也不等于自身持仓路径上的长期净价值。所有候选0bp至少一项硬门失败，固定版本关闭，不进入2/5bp、不改现役release；不能将本卡直接当作精确f696+b472的同钟配对改进或独立前向证据。\n")
    return summary


def constraint_bridge(*, card: Path) -> dict:
    """Attribute the fixed post-model target mapping without another account run."""
    import qlib
    from qlib.workflow import R

    card = Path(card).resolve()
    summary = json.loads((card / "summary.json").read_text())
    if (card / "constraint_bridge.json").exists():
        return json.loads((card / "constraint_bridge.json").read_text())
    qlib.init(provider_uri=summary["provider_uri"], region="us", kernels=1)
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(recorder_id=summary["recorder_id"], experiment_name=summary["experiment"])
        rows = []
        for family in ("xgb", "lgbm"):
            open_book = recorder.load_object(f"analysis/native/{family}_unrestricted_0")
            limited_book = recorder.load_object(f"analysis/native/{family}_constrained_0")
            raw = open_book["targets"].reset_index(drop=True)
            limited = limited_book["targets"].reset_index(drop=True)
            if not raw.trade_date.equals(limited.trade_date):
                raise ValueError("blocked: constraint comparison calendar differs")
            stop = limited.known_drawdown.ge(.04)
            soxl = raw.soxl_weight.gt(.9)
            soxx = raw.soxx_weight.gt(.9)
            rows.append({"family": family, "sessions": len(raw),
                "pure_soxl_predictions": int(soxl.sum()),
                "pure_soxx_predictions": int(soxx.sum()),
                "cash_predictions": int((~soxl & ~soxx).sum()),
                "constrained_stop_days": int(stop.sum()),
                "mean_unrestricted_target_cash_pct": float(100 * raw.cash_weight.mean()),
                "mean_constrained_target_cash_pct": float(100 * limited.cash_weight.mean()),
                "mean_constrained_target_cash_on_stop_pct": float(100 * limited.loc[stop, "cash_weight"].mean()),
                "mean_constrained_target_cash_off_stop_pct": float(100 * limited.loc[~stop, "cash_weight"].mean())})
        recorder.save_objects(artifact_path="analysis", constraint_bridge=pd.DataFrame(rows))
        recorder.save_objects(artifact_path="sources", regime_workflow=Path(__file__).read_text())
    output = {"status": "diagnostic_post_model_mapping", "cases": rows,
        "scope": "same fitted family and decision calendar, separate own-state paths; joint cap-floor-stop effect only, no isolated causal layer claim"}
    atomic_json(output, card / "constraint_bridge.json")
    with (card / "conclusion.md").open("a") as text:
        text.write("\n### 限制传递\n\n")
        for row in rows:
            text.write(f"{row['family'].upper()}：944日原始动作中SOXL {row['pure_soxl_predictions']}日、SOXX {row['pure_soxx_predictions']}日、现金 {row['cash_predictions']}日；受限账户自身回撤触发4%止损 {row['constrained_stop_days']}日，受限目标现金均值 {row['mean_constrained_target_cash_pct']:.2f}%。\n\n")
        text.write("这是仓位上限、现金下限和止损合并后的映射及两条自身账户路径；未单独识别每项限制的因果贡献。\n")
    return output
