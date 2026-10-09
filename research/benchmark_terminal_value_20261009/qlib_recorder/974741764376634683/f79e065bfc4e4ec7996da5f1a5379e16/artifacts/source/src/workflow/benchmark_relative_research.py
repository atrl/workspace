"""Small notebook runner: explicit sources, mature feedback, native evidence.

Experiment parameters live in the card's frozen plan. Old experiment modules
and their recorded identities are never rewritten by this runner.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.analysis.convex_qlib import native_open_reconciliation
from src.analysis.integrated_allocation import paired_metrics, annual_paired_metrics
from src.analysis.benchmark_relative_protocol import evaluate_benchmark_relative
from src.backtest.benchmark_relative_replay import replay_fixed_target, replay_online_allocator
from src.data.benchmark_relative_data import build_relative_dataset
from src.data.integrated_allocation import frame_sha256
from src.data.legacy_execution_view import load_legacy_execution_view
from src.model.online_expert_allocation import FixedShareAllocator, FixedShareConfig


SOURCE_FILES = (
    "src/workflow/benchmark_relative_research.py", "src/model/online_expert_allocation.py",
    "src/data/benchmark_relative_data.py", "src/backtest/benchmark_relative_replay.py",
    "src/analysis/benchmark_relative_protocol.py", "src/analysis/integrated_allocation.py",
    "src/analysis/convex_qlib.py", "src/data/legacy_execution_view.py",
    "src/model/post_fee_settlement.py", "src/backtest/qlib_continuous_position.py",
    "src/backtest/strategy.py", "src/workflow/stage1_economic_action.py",
    "src/data/qlib_native.py", "src/data/price_basis.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _source_hashes(source_root):
    hashes = {}
    for name in SOURCE_FILES:
        module = importlib.import_module(name[:-3].replace("/", "."))
        if Path(module.__file__).resolve() != (source_root / name).resolve():
            raise ValueError("blocked: imported implementation differs from declared source_root")
        hashes[name] = _sha(source_root / name)
    return hashes


def _same_recorder(R, recorder_id):
    recorder = R.get_recorder()
    if recorder.id != recorder_id:
        raise ValueError("blocked: resumed Qlib Recorder identity differs")
    return recorder


def _config(plan):
    return FixedShareConfig(expert_names=tuple(plan["expert_names"]),
        expert_weights=tuple(tuple(x) for x in plan["expert_weights"]),
        anchor_weights=tuple(plan["anchor_weights"]), prior=tuple(plan["prior"]),
        active_fraction=plan["active_fraction"], eta=plan["eta"], share=plan["share"])


def _validate_plan(plan):
    if (plan["schema_version"] != "benchmark_relative_research_v1"
            or plan["evidence_kind"] != "reused_history"
            or plan["candidate"] != "online_fixed_share" or plan["candidate_budget"] != 1
            or plan["expert_names"] != ["soxx", "leveraged", "defensive"]
            or plan["control_names"] != ["soxx", "leveraged", "defensive", "fixed_mixture"]
            or plan["costs_bps"] != [10., 25.]
            or plan["benchmark"] != "SOXX" or plan["anchor_weights"] != [0, 1, 0]
            or plan["expert_weights"][0] != [0, 1, 0]
            or plan["registered_before_replay"] is not True):
        raise ValueError("blocked: unsupported or inconsistent registered protocol")
    stamp = pd.Timestamp(plan["registered_at_utc"])
    if stamp.tzinfo is None or stamp > pd.Timestamp.now(tz="UTC"):
        raise ValueError("blocked: invalid registration timestamp")
    _config(plan)


def _native_path(frame, path, provider, cost, account):
    native = native_open_reconciliation(frame, path, provider_uri=provider, cost_bps=cost, account=account)
    if native["audit"]["status"] != "passed":
        raise ValueError("blocked: native Qlib reconciliation failed")
    result = path.copy()
    # Metrics and expert feedback use native positions marked by native D prices.
    result["strategy_return"] = native["reconciliation"].native_next_open_return.to_numpy()
    result["equity"] = np.cumprod(1 + result.strategy_return.to_numpy())
    return result, native


def _conclusion(comparison, gate, recorder_id, as_of):
    lines = ["# 基准相对架构与在线专家实验结论", "",
        "运行了一个固定分享在线候选、固定混合对照及三个静态专家；全部独立扣费结算并通过原生 Qlib 回放。没有训练新预测网络或搜索参数。", "",
        "每侧25bp，以下按信号年度切片，持仓与学习状态跨年连续：", ""]
    for row in comparison.loc[comparison.cost_bps.eq(25.)].itertuples():
        lines.append(f"- {row.window} / {row.strategy}：收益{row.return_pct:.2f}%，相对SOXX财富{row.relative_wealth:.4f}，MaxDD {row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%，换手{row.turnover:.2f}。")
    lines += ["", f"开发期历史资格门通过：{gate['historical_gate_pass']}。不晋升；所有历史已被查看，本次不能证明独立前向稳定超额。",
        "", "线上反馈只使用决策前已经结算的专家净收益；组合有自己的持仓、费用与收益，不是专家净值的加权平均。固定混合用于隔离在线更新贡献，静态专家不参与事后选优。",
        "", "年度差异见 annual_metrics.csv；2023以后仍为复用历史诊断。没有追加数据驱动的regime阈值，滚动窗口高度重叠，不算独立样本。",
        "", "风险：回溯调整价格并非已独立核验的含息总回报；现金收益为零，历史开盘价不等于真实成交。基准核心与主动比例只是配置约束，不保证回撤或CVaR。",
        "", "下一步：保留本轮失败或通过的全部证据；冻结后取得连续新时点记录。收益预测不确定性惩罚列为独立候选，须先有可信的过去OOS误差，不能根据本轮盈亏修改学习率或扩大候选。",
        "", f"Qlib Recorder：{recorder_id}；as_of={as_of}。"]
    return "\n".join(lines) + "\n"


def run_benchmark_relative_research(*, source_root, evidence_root, card_dir, recorder_uri):
    """All writable destinations are explicit; refuse to overwrite a completed card."""
    from qlib.workflow import R
    import qlib

    source_root, evidence_root, card = map(lambda x: Path(x).resolve(), (source_root, evidence_root, card_dir))
    uri = str(Path(recorder_uri).resolve())
    plan_path = card / "plan.json"
    plan = json.loads(plan_path.read_text())
    _validate_plan(plan)
    if (card / "summary.json").exists():
        raise ValueError("blocked: completed card is immutable; use a new registered card")
    hashes = _source_hashes(source_root)
    plan_hash = _sha(plan_path)
    provider, provider_audit = load_legacy_execution_view(evidence_root)
    if provider != (evidence_root / plan["provider_uri"]).resolve():
        raise ValueError("blocked: plan provider differs from audited native source")
    dataset, frame, calendar = build_relative_dataset(provider_uri=provider,
        signal_start=plan["signal_start"], as_of=plan["as_of"])
    binding = {"plan_sha256": plan_hash, "source_sha256": hashes,
        "provider_audit": provider_audit, "dataset_sha256": frame_sha256(frame),
        "qlib_version": qlib.__version__, "dataset_type": type(dataset).__name__,
        "evidence_kind": plan["evidence_kind"], "code_root": str(source_root),
        "read_only_evidence_root": str(evidence_root), "card_dir": str(card), "recorder_uri": uri}
    _json(card / "run_binding.json", binding)
    with R.start(experiment_name=plan["experiment"], uri=uri):
        recorder = R.get_recorder()
        recorder_id = recorder.id
        recorder.log_params(plan_sha256=plan_hash, evidence_kind=plan["evidence_kind"], candidate_budget=1)
        recorder.save_objects(artifact_path="inputs", **{"binding.pkl": binding, "dataset_rows.pkl": frame, "plan.pkl": plan})
        for name in SOURCE_FILES:
            recorder.log_artifact(str(source_root / name), artifact_path="source/" + str(Path(name).parent))
    paths, native_reports, feedback = {}, {}, {}
    try:
        for cost in plan["costs_bps"]:
            experts = {}
            for name, target in zip(plan["expert_names"], plan["expert_weights"]):
                raw = replay_fixed_target(frame, target=target, cost_bps=cost,
                    rebalance_sessions=plan["rebalance_sessions"], buy_and_hold=name == "soxx")
                path, native = _native_path(frame, raw, provider, cost, plan["initial_cash"])
                experts[name] = path
                paths[(name, cost)], native_reports[(name, cost)] = path, native
                print(f"Native expert passed: {name}, {cost:g}bp", flush=True)
            config = _config(plan)
            raw, audit = replay_online_allocator(frame, experts, allocator=FixedShareAllocator(config),
                cost_bps=cost, rebalance_sessions=plan["rebalance_sessions"])
            mixture = (1 - config.active_fraction) * np.asarray(config.anchor_weights) + config.active_fraction * (np.asarray(config.prior) @ np.asarray(config.expert_weights))
            fixed = replay_fixed_target(frame, target=mixture, cost_bps=cost,
                rebalance_sessions=plan["rebalance_sessions"])
            for name, path in ((plan["candidate"], raw), ("fixed_mixture", fixed)):
                paths[(name, cost)], native_reports[(name, cost)] = _native_path(frame, path, provider, cost, plan["initial_cash"])
                print(f"Native portfolio passed: {name}, {cost:g}bp", flush=True)
            feedback[cost] = audit
        dev_years = plan["development_years"]
        def dev(path):
            return path.loc[path.signal_date.dt.year.isin(dev_years)].copy()
        gate = evaluate_benchmark_relative(
            {cost: dev(paths[(plan["candidate"], cost)]) for cost in plan["costs_bps"]},
            {cost: dev(paths[("soxx", cost)]) for cost in plan["costs_bps"]},
            expected_years=tuple(dev_years), expected_signal_dates=calendar[calendar.year.isin(dev_years)],
            costs_bps=tuple(plan["costs_bps"]), evidence_kind=plan["evidence_kind"], **plan["gates"])
        comparison, annual, ablation = [], [], []
        for (name, cost), path in paths.items():
            benchmark = paths[("soxx", cost)]
            annual.append(annual_paired_metrics(path, benchmark).assign(strategy=name, cost_bps=cost))
            for window, mask in (("development", path.signal_date.dt.year.isin(dev_years)),
                                 ("reused_diagnostic", path.signal_date.dt.year.gt(max(dev_years)))):
                comparison.append({"strategy": name, "cost_bps": cost, "window": window,
                    **paired_metrics(path.loc[mask], benchmark.loc[mask])})
                if name == plan["candidate"]:
                    ablation.append({"cost_bps": cost, "window": window, "reference": "fixed_mixture",
                        **paired_metrics(path.loc[mask], paths[("fixed_mixture", cost)].loc[mask])})
        tables = {"backtest_benchmark_comparison.csv": pd.DataFrame(comparison),
            "annual_metrics.csv": pd.concat(annual, ignore_index=True), "ablation_comparison.csv": pd.DataFrame(ablation),
            "gates.csv": gate["gates"], "rolling_summary.csv": gate["rolling_summary"],
            "bootstrap_analysis.csv": gate["bootstrap"],
            "native_audit.csv": pd.DataFrame([{"strategy": n, **v["audit"]} for (n, c), v in native_reports.items()])}
        # A second readback is cheap and catches mutation during execution.
        _, final_provider_audit = load_legacy_execution_view(evidence_root)
        if (final_provider_audit != provider_audit or _sha(plan_path) != plan_hash
                or any(_sha(source_root / name) != digest for name, digest in hashes.items())):
            raise ValueError("blocked: source/plan changed during experiment")
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        conclusion = _conclusion(tables["backtest_benchmark_comparison.csv"], gate, recorder_id, plan["as_of"])
        (card / "conclusion.md").write_text(conclusion)
        summary = {"experiment": plan["experiment"], "record_type": "benchmark_relative_architecture_diagnostic",
            "as_of": plan["as_of"], "provider_uri": plan["provider_uri"], "region": "us", "market": "all",
            "data_range": [str(frame.signal_date.min().date()), str(frame.exit_date.max().date())],
            "benchmark": "SOXX", "model_selection": "one_registered_candidate_no_search",
            "evidence_kind": plan["evidence_kind"], "historical_gate_pass": gate["historical_gate_pass"],
            "promotion_eligible": False, "decision": "not_promoted_reused_history",
            "recorder_id": recorder_id, "experiment_id": recorder.experiment_id,
            "recorder_uri": uri, "recorder_local_dir": recorder.get_local_dir(),
            "plan_sha256": plan_hash, "run_binding_sha256": _sha(card / "run_binding.json"),
            "source_sha256": hashes, "native_paths": len(native_reports), "native_all_pass": True,
            "qlib_docs": ["https://qlib.readthedocs.io/en/latest/component/data.html",
                          "https://qlib.readthedocs.io/en/latest/component/recorder.html",
                          "https://qlib.readthedocs.io/en/latest/component/strategy.html"],
            "sidecars": list(tables) + ["plan.json", "run_binding.json", "architecture.md", "research_sources.md", "conclusion.md"],
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/dataset_rows.pkl", "analysis/paths.pkl", "analysis/feedback.pkl", "portfolio_analysis/native_reports.pkl"]},
            "limitations": plan["limitations"]}
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            recorder = _same_recorder(R, recorder_id)
            recorder.save_objects(artifact_path="analysis", **{"paths.pkl": paths, "feedback.pkl": feedback, "gates.pkl": gate})
            recorder.save_objects(artifact_path="portfolio_analysis", **{"native_reports.pkl": native_reports})
            recorder.log_metrics(historical_gate_pass=float(gate["historical_gate_pass"]), native_paths=len(native_reports))
            for name in [*tables, "plan.json", "run_binding.json", "conclusion.md", "workflow.yaml"]:
                recorder.log_artifact(str(card / name), artifact_path="card")
            with tempfile.TemporaryDirectory(prefix="soxx-card-summary-") as temporary:
                staged = Path(temporary) / "summary.json"
                _json(staged, summary)
                recorder.log_artifact(str(staged), artifact_path="card")
        _json(card / "summary.json", summary)
    except Exception:
        with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=uri, resume=True):
            _same_recorder(R, recorder_id)
            raise
    # Qlib auto-snapshots the entire dirty checkout; retain only explicit code
    # snapshots above, preventing unrelated data WIP from entering this record.
    for name in ("code_status.txt", "code_diff.txt", "code_cached.txt"):
        (Path(recorder.get_local_dir()) / "artifacts" / name).unlink(missing_ok=True)
    return summary, tables, conclusion
