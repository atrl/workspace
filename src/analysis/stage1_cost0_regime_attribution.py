"""Read-only, full-calendar attribution for the closed Stage1 cost0 account."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _period(frame: pd.DataFrame, name: str, kind: str) -> dict:
    log_b = np.log1p(frame.candidate_return.to_numpy(float)) - np.log1p(frame.B_return.to_numpy(float))
    log_s = np.log1p(frame.candidate_return.to_numpy(float)) - np.log1p(frame.SOXX_return.to_numpy(float))
    result = {"kind": kind, "period": name, "sessions": len(frame),
              "first_trade": str(frame.trade_date.iloc[0].date()),
              "last_exit": str(frame.exit_date.iloc[-1].date()),
              "candidate_minus_B_log": float(log_b.sum()),
              "candidate_over_B_wealth": float(np.exp(log_b.sum())),
              "candidate_minus_SOXX_log": float(log_s.sum()),
              "candidate_over_SOXX_wealth": float(np.exp(log_s.sum()))}
    for layer in ("raw", "smoothed", "controlled", "funded"):
        result[f"candidate_{layer}_cash_mean"] = float(frame[f"candidate_{layer}_cash"].mean())
        result[f"B_{layer}_cash_mean"] = float(frame[f"B_{layer}_cash"].mean())
        result[f"candidate_minus_B_{layer}_cash_mean"] = (
            result[f"candidate_{layer}_cash_mean"] - result[f"B_{layer}_cash_mean"])
    result.update(candidate_fee_usd_sum=float(frame.candidate_fee_usd.sum()),
                  B_fee_usd_sum=float(frame.B_fee_usd.sum()),
                  candidate_risk_state_ge2_fraction=float(frame.candidate_risk_state.ge(2).mean()),
                  B_risk_state_ge2_fraction=float(frame.B_risk_state.ge(2).mean()),
                  candidate_cap_binding_fraction=float(frame.candidate_cap_binding.mean()),
                  B_cap_binding_fraction=float(frame.B_cap_binding.mean()),
                  candidate_execute_fraction=float(frame.candidate_execute.mean()),
                  B_execute_fraction=float(frame.B_execute.mean()))
    return result


def run(root: Path) -> dict:
    import qlib
    from qlib.workflow import R

    from src.data.minute_intake import atomic_json
    from src.data.nas_market_data import _settings, _ssh, archive_nas_recorder

    root = Path(root).resolve()
    card = root / "research/stage1_training_cost0_regime_attribution_20261023"
    plan = json.loads((card / "plan.json").read_text())
    digest = _sha(card / "plan.json")
    assert plan["frozen"] and not (card / "run_state.json").exists()
    qlib.init(provider_uri=str((root / "data/qlib/stage1_training_cost_0bp_ablation_20261023").resolve()),
              region="us", kernels=1)
    source_objects = {}
    loaded = 0
    for role in ("candidate", "control", "benchmark"):
        source = plan["sources"][role]
        source_card = root / source["card"]
        summary = json.loads((source_card / "summary.json").read_text())
        assert _sha(source_card / "summary.json") == source["summary_sha256"]
        assert summary["recorder_id"] == source["recorder_id"]
        assert json.loads((source_card / "nas_binding.json").read_text())["evidence"]["sha256"] == source["nas_archive_sha256"]
        with R.uri_context(str((source_card / "qlib_recorder").resolve())):
            recorder = R.get_recorder(experiment_name=source_card.name, recorder_id=source["recorder_id"])
            names = source["artifacts"] if "artifacts" in source else [source["artifact"]]
            source_objects[role] = {name: recorder.load_object(name) for name in names}
            loaded += len(names)
    assert loaded == 7 and loaded <= plan["budget"]["source_Recorder_object_loads_max"]
    cand = source_objects["candidate"]
    ctrl = source_objects["control"]
    candidate_book = cand["native/candidate_cost0_0"]
    B_book = ctrl["native/B_shared_0"]
    A_book = ctrl["native/A_active_research_replay_0"]
    SOXX_book = source_objects["benchmark"]["native/raw_SOXX_0"]
    books = (candidate_book, B_book, A_book, SOXX_book)
    for book in books:
        assert book["audit"]["status"] == "passed"
    paths = [book["authority"].reset_index(drop=True) for book in books]
    dates = ["signal_date", "trade_date", "exit_date"]
    for path in paths[1:]:
        pd.testing.assert_frame_equal(path[dates], paths[0][dates])
    assert len(paths[0]) == 944
    ensembles = cand["analysis/ensembles"]
    smoothed = cand["analysis/smoothed_targets"]
    signals = pd.DatetimeIndex(paths[0].signal_date)
    frame = paths[0][dates].copy()
    for family in ("candidate_cost0", "B_shared"):
        prefix = "candidate" if family == "candidate_cost0" else "B"
        raw = ensembles[family].reindex(signals)
        smooth = smoothed[family].reindex(signals)
        assert raw.notna().all().all() and smooth.notna().all().all()
        frame[f"{prefix}_raw_cash"] = raw.cash_weight.to_numpy(float)
        frame[f"{prefix}_smoothed_cash"] = smooth.cash_weight.to_numpy(float)
        book = candidate_book if family == "candidate_cost0" else B_book
        target = book["targets"].reset_index(drop=True)
        pd.testing.assert_frame_equal(target[dates[:2]], frame[dates[:2]])
        frame[f"{prefix}_controlled_cash"] = target.cash_weight.to_numpy(float)
        path = paths[0] if family == "candidate_cost0" else paths[1]
        frame[f"{prefix}_funded_cash"] = path.ending_cash_weight.to_numpy(float)
        frame[f"{prefix}_return"] = path.strategy_return.to_numpy(float)
        frame[f"{prefix}_fee_usd"] = path.broker_and_external_fees_usd.to_numpy(float)
        frame[f"{prefix}_risk_state"] = path.risk_state.to_numpy(int)
        frame[f"{prefix}_execute"] = path.execute_open_target.to_numpy(bool)
        beta = 3 * raw.soxl_weight.to_numpy(float) + raw.soxx_weight.to_numpy(float)
        frame[f"{prefix}_cap_binding"] = beta > path.current_beta_cap.to_numpy(float) + 1e-8
    frame["A_return"] = paths[2].strategy_return.to_numpy(float)
    frame["SOXX_return"] = paths[3].strategy_return.to_numpy(float)
    assert np.isfinite(frame.select_dtypes(include=["number"]).to_numpy(float)).all()
    old = json.loads((root / plan["sources"]["candidate"]["card"] / "summary.json").read_text())
    full = _period(frame, "all", "all")
    np.testing.assert_allclose(full["candidate_over_B_wealth"], old["metrics"]["vs_matched_B0"]["relative_wealth"], atol=1e-10)
    np.testing.assert_allclose(full["candidate_over_SOXX_wealth"], old["metrics"]["0bp"]["relative_wealth"], atol=1e-10)
    np.testing.assert_allclose(full["candidate_funded_cash_mean"], old["metrics"]["funded_candidate_cash_mean"], atol=1e-10)
    periods = [full]
    for kind, frequency in (("year", "Y"), ("quarter", "Q")):
        labels = pd.DatetimeIndex(frame.trade_date).to_period(frequency)
        for value in labels.unique():
            periods.append(_period(frame.loc[labels == value], str(value), kind))
    periods = pd.DataFrame(periods)
    assert len(periods.loc[periods.kind.eq("year")]) == 4
    assert len(periods.loc[periods.kind.eq("quarter")]) == 16
    windows = []
    for start in range(len(frame) - 125):
        section = frame.iloc[start:start + 126]
        row = _period(section, str(section.trade_date.iloc[0].date()), "rolling126")
        windows.append(row)
    windows = pd.DataFrame(windows)
    assert len(windows) == 819
    frozen_soxx_windows = cand["analysis/rolling126"]
    np.testing.assert_allclose(windows.candidate_minus_SOXX_log.to_numpy(float),
                               frozen_soxx_windows.relative_log.to_numpy(float), atol=1e-10)
    named = periods.loc[periods.period.isin(["2024Q3", "2025"])].copy()
    for name, start, end in (("original_worst126", "2023-08-01", "2024-01-31"),
                             ("original_best126", "2024-10-03", "2025-04-07")):
        section = frame.loc[frame.trade_date.ge(pd.Timestamp(start)) &
                            frame.exit_date.le(pd.Timestamp(end))]
        if len(section) != 126:
            raise ValueError("frozen named interval is not 126 sessions")
        named = pd.concat([named, pd.DataFrame([_period(section, name, "named")])], ignore_index=True)
    result = {"decision": "descriptive_closed_cost0_failure_chain_no_promotion",
              "plan_sha256": digest, "source_recorder_ids": {k: v["recorder_id"] for k, v in plan["sources"].items()},
              "sessions": len(frame), "years": 4, "quarters": 16, "rolling126_windows": 819,
              "full": full, "named": named.to_dict("records"),
              "new_market_requests": 0, "new_fits": 0, "new_accounts": 0, "new_bootstraps": 0,
              "source_Recorder_object_loads": loaded,
              "interpretation_limit": "Separate own-state account histories; layer weight differences and log returns are descriptive, not local causal interventions."}
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    periods.loc[periods.kind.eq("year")].to_csv(card / "annual_summary.csv", index=False)
    periods.loc[periods.kind.eq("quarter")].to_csv(card / "quarterly_summary.csv", index=False)
    R.set_uri(str((card / "qlib_recorder").resolve()))
    with R.start(experiment_name=card.name):
        recorder = R.get_recorder()
        recorder.save_objects(artifact_path="inputs", plan=plan, source_identities=plan["sources"])
        recorder.save_objects(artifact_path="analysis", daily=frame, periods=periods,
                              rolling126=windows, named=named, result=result)
        R.log_params(source="frozen_Stage1_0bp_raw_corporate_recorders", new_fits=0, new_accounts=0)
        R.log_metrics(candidate_over_B_wealth=full["candidate_over_B_wealth"],
                      candidate_over_SOXX_wealth=full["candidate_over_SOXX_wealth"])
        rid, eid = recorder.id, str(recorder.experiment_id)
    result["recorder_id"] = rid
    result["experiment_id"] = eid
    (card / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    source_binding = json.loads((root / plan["sources"]["candidate"]["card"] / "nas_binding.json").read_text())
    source_binding.pop("evidence", None)
    atomic_json(source_binding, card / "nas_binding.json")
    nas = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
                               card=card, recorder_id=rid, experiment_id=eid)
    remote = _ssh(_settings(root)["ssh"], "sha256sum " + shlex.quote(nas["nas_path"])).split()[0]
    assert remote == nas["sha256"]
    summary = {"experiment": card.name, "decision": result["decision"],
               "objective_achieved": False, "plan_sha256": digest,
               "result": "result.json", "recorder_id": rid, "experiment_id": eid,
               "new_fits": 0, "new_native_accounts": 0,
               "NAS_evidence": nas, "NAS_readback": {"verified": True, "sha256": remote},
               "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                                   "recorder_only_artifacts": ["inputs/*", "analysis/daily", "analysis/periods",
                                                               "analysis/rolling126", "analysis/named"]},
               "limitations": [result["interpretation_limit"],
                               "Reused 2023-2026 history is not independent forward evidence."]}
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text("experiment: " + card.name + "\nplan: plan.json\n"
        "sources: frozen Qlib Recorder objects\nnew_fits: 0\nnew_native_accounts: 0\n")
    worst = named.loc[named.period.eq("original_worst126")].iloc[0]
    conclusion = (
        "# Stage1 训练成本0bp失败链：逐层现金与区间对照\n\n"
        "只读已封存的候选、原 Stage1 A_0、匹配 B_0 和 SOXX 原生账户，覆盖944日、4年、16季及819个连续126日窗。"
        "原始网络现金、平滑目标、投影控制目标和实际期末现金逐日按保存信号/交易时点对齐。"
        "0拟合、0新账户、0新行情请求。\n\n"
        f"全程候选/B_0 净财富比 {full['candidate_over_B_wealth']:.6f}，候选/SOXX {full['candidate_over_SOXX_wealth']:.6f}。"
        f"候选减B_0现金均值：原始 {full['candidate_minus_B_raw_cash_mean']*100:.2f}、"
        f"平滑 {full['candidate_minus_B_smoothed_cash_mean']*100:.2f}、"
        f"投影目标 {full['candidate_minus_B_controlled_cash_mean']*100:.2f}、"
        f"实际期末 {full['candidate_minus_B_funded_cash_mean']*100:.2f} 个百分点。"
        f"原 Stage1 最差126日窗口中候选/B_0 对数差 {worst['candidate_minus_B_log']:.6f}；"
        "所有年度、季度和窗口保存在Recorder。\n\n"
        "这是两条自身状态账户的描述性差异，不能把现金差直接认定为单日因果收益。"
        "训练成本0bp固定版本已经失败，不从季度或窗口反选规则，不再调成本或跑2/5bp。"
        f"Recorder `{rid}`，NAS SHA `{remote}`。\n")
    (card / "conclusion.md").write_text(conclusion)
    return summary
