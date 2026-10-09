"""Stage1 action marginal-information diagnosis workflow (corrected).

Reads the frozen 17f81fbf economic-action recorder and the frozen action card,
read-only, and reports:

1. TRUE out-of-sample conditional information: each year is predicted with that
   year's saved pipeline (fit only on prior data) and compared with the SAME
   rows' baseline-domain ``execute_advantage`` label. Cross-account returns are
   never used as the label.
2. Per-segment, per-cost net contribution of the recorded paths against the true
   no_hold upstream, each segment resetting from its own initialization and never
   multiplied across segments.
3. SOXX buy-hold and a fixed SOXX/cash split, settled through the project's cash
   fee accounting so they share the strategy's entry/exit and cost, and
   reconciled by the OPEN-ONLY native helper. That helper supports open-only
   schedules and rejects recovery; it is NOT a full intraday ledger. A complete
   intraday staged account would need a real 15-minute ledger (2019+ only), and
   2016-2018 lacks that data, so no earlier bars are fabricated.

The child writes its own Recorder (real predictions/label/CI/contribution
objects); parent recorders are read-only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analysis.action_marginal_information import period_metrics
from src.analysis.action_oos_information import (
    conditional_information,
    decidable_mask,
    segment_contribution,
    true_oos_predictions,
    verify_model_identity_and_maturity,
)
# Execution-route thresholds from the frozen action spec. A row is decidable only
# when the model actually chooses (not a forced de-risk, large transition, or
# non-finite probability).
EXECUTE_THRESHOLD = 0.50
MAX_SKIPPABLE_TURNOVER = 0.05
MANDATORY_DERISK_BETA_DELTA = 0.0

EXPERIMENT = "stage1_action_marginal_diagnosis_20261009"
SCHEMA_VERSION = "stage1_action_marginal_diagnosis_v2"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
SOURCE_CARD = "stage1_economic_action_20260929"
SOURCE_RECORDER = "17f81fbf27ee46859816ec1ac52151f6"
ACTION_CARD = "stage1_open_action_value_net_fees_20260928"
ACTION_RECORDER = "b472a458fc9042b4905094fddeaebd33"
PROVIDER = "data/qlib/soxl_soxx_soxs"
FROZEN_OBJECT_SHA256 = {
    "analysis/portfolio_replays.pkl": "ac7281e80aa78d2cc69c9c97dadd51356ac883e85ee87d53ca46eaa976b4b306",
    "analysis/comparison.pkl": "c31ea2c023e27d489400cda8fb715543d732edf30714876b4fd8d7932b0f0356",
    "analysis/annual.pkl": "b8f7423783600612c6b696decb73c15abea604ce9a494242c94a84ff8be733cc",
    "inputs/prepared_inputs.pkl": "8baa4f74449c1da41eb189f51c75074a61dacb8d7c8540710f4f67560d58d4aa",
}
DEVELOPMENT = ("2016-01-04", "2022-12-21")
OUTER = ("2023-01-03", "2026-09-23")
VARIANTS = {
    "active": "active_hold_baseline",
    "no_hold": "no_hold_reference",
    "rules_only": "rules_only",
    "own_state_frozen": "own_state_net_value_frozen_2022",
    "own_state_expanding": "own_state_net_value_expanding_2025",
}
SEGMENTS = {"development": VARIANTS, "outer_reused_diagnostic": VARIANTS}


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class DiagnosisSpec:
    experiment: str = EXPERIMENT
    root: Path = Path(".")


def resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_action_metadata(root: Path, *, action_card: str, action_recorder: str) -> dict:
    """Read the frozen action card's dataset audit and per-year training diagnostics."""
    base = _card_dir(root, action_card) / "qlib_recorder"
    name = Path(action_card).name
    audit_files = list(base.glob(f"*/{action_recorder}/artifacts/{name}/dataset_audit.json"))
    diag_files = list(base.glob(f"*/{action_recorder}/artifacts/{name}/training_diagnostics.csv"))
    _require(audit_files and diag_files,
             "action dataset_audit.json or training_diagnostics.csv is missing")
    dataset_audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
    diagnostics = pd.read_csv(diag_files[0])
    return {"dataset_audit": dataset_audit, "diagnostics": diagnostics,
            "dataset_audit_sha256": _sha256(audit_files[0]),
            "training_diagnostics_sha256": _sha256(diag_files[0])}


def _quiet() -> None:
    """Silence qlib/mlflow INFO+WARNING noise captured into notebooks."""
    import logging
    import warnings

    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*future calendar.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")


def load_plan(root: Path) -> dict:
    """Read and hash-lock the frozen plan; the plan is the config source."""
    import hashlib
    path = root / "research" / EXPERIMENT / "plan.json"
    _require(path.is_file(), "plan.json is missing")
    raw = path.read_bytes()
    return {"plan": json.loads(raw), "plan_sha256": hashlib.sha256(raw).hexdigest()}


def _card_dir(root: Path, card: str) -> Path:
    """Resolve a card path that may be repo-relative or a bare card name."""
    candidate = Path(card)
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return (root / candidate).resolve()
    return (root / "research" / card).resolve()


def load_frozen(root: Path, *, source_card: str, source_recorder: str, object_sha256: dict):
    """Read the frozen replays + prepared inputs through Qlib, hash-bound."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    _quiet()
    import qlib
    from qlib.workflow import R

    base = (_card_dir(root, source_card) / "qlib_recorder").resolve()
    _require(base.is_dir(), "frozen economic-action recorder is unavailable")
    qlib.init(provider_uri=str((root / PROVIDER).resolve()), region="us")
    objects = {}
    with R.uri_context(str(base)):
        recorder = R.get_recorder(recorder_id=source_recorder, experiment_name=Path(source_card).name)
        local = Path(recorder.get_local_dir()) / "artifacts"
        for name, expected in object_sha256.items():
            actual = _sha256(local / name)
            _require(actual == expected, f"frozen object hash mismatch for {name}: {actual} != {expected}")
            objects[name] = recorder.load_object(name)
    replays = objects["analysis/portfolio_replays.pkl"]
    _require(set(replays["variant"]) >= {VARIANTS["active"], VARIANTS["no_hold"]},
             "frozen replays missing active or no_hold variant")
    _require(set(replays["cost_bps"]) == {10.0, 25.0}, "frozen replays must carry 10bp and 25bp")
    return {"replays": replays, "prepared": objects["inputs/prepared_inputs.pkl"]}


def _segment_mask(replays: pd.DataFrame, window: tuple[str, str]) -> pd.Series:
    dates = pd.to_datetime(replays["signal_date"])
    return dates.ge(pd.Timestamp(window[0])) & dates.le(pd.Timestamp(window[1]))


def per_segment_contribution(replays: pd.DataFrame, *, window: tuple[str, str], segment: str,
                             cost_bps: float) -> list[dict]:
    """Contribution within one segment, each variant's own reset account."""
    block = replays[(replays["cost_bps"] == cost_bps) & _segment_mask(replays, window)]
    no_hold = block[block["variant"] == VARIANTS["no_hold"]].set_index("signal_date")["strategy_return"]
    _require(not no_hold.empty, f"no no_hold path in {segment} at {cost_bps}bp")
    paths = {key: block[block["variant"] == name].set_index("signal_date")["strategy_return"]
             for key, name in SEGMENTS[segment].items()}
    return segment_contribution(paths, no_hold, cost_bps=cost_bps, segment=segment)


def reference_paths(root: Path, block: pd.DataFrame, *, cost_bps: float,
                    native: bool = True) -> pd.DataFrame:
    """SOXX buy-hold and a fixed 50/50 SOXX-cash split, settled with cash fees.

    Uses the strategy's own entry/exit sessions and the project's
    ``settle_target_session`` so the reference shares calendar, execution and
    cost. The 50/50 split is rebalanced to target each session and pays fees.
    When ``native`` is set the open-only schedule is additionally replayed and
    reconciled through ``native_open_reconciliation``. That helper supports
    open-only schedules and rejects recovery; it is NOT a substitute for a full
    intraday ledger.
    """
    from src.backtest.position_accounting import settle_target_session
    from src.data import QlibProviderConfig, load_qlib_feature_frame
    from src.workflow.stage1_economic_action import fee_affordable_target

    trade_dates = pd.to_datetime(block["trade_date"]).to_numpy()
    exit_dates = pd.to_datetime(block["exit_date"]).to_numpy()
    config = QlibProviderConfig(provider_uri=root / PROVIDER, region="us", market="all", freq="day")
    start = str((pd.Timestamp(trade_dates.min()) - pd.Timedelta(days=5)).date())
    end = str((pd.Timestamp(exit_dates.max()) + pd.Timedelta(days=5)).date())
    raw = load_qlib_feature_frame(config, instruments=["SOXX"], fields=["$open"], start=start, end=end)
    opening = raw["$open"].unstack(level=0)["SOXX"]
    opening.index = pd.to_datetime(opening.index)
    soxx = opening.reindex(pd.DatetimeIndex(pd.concat([pd.Series(trade_dates), pd.Series(exit_dates)]).drop_duplicates()))
    _require(soxx.isna().sum() == 0, "SOXX open prices missing for reference sessions")

    def settle(target_weights):
        held = np.array([0.0, 0.0, 1.0])
        rets = []
        for t, e in zip(trade_dates, exit_dates):
            soxx_ret = float(soxx.loc[pd.Timestamp(e)] / soxx.loc[pd.Timestamp(t)] - 1.0)
            asset_returns = np.array([0.0, soxx_ret, 0.0])
            # Convert the desired net weights to a cash-funded pre-fee target so
            # fees are affordable and post-fee holdings equal the target.
            funded = fee_affordable_target(target_weights, pretrade_weights=held, cost_bps=cost_bps)
            result = settle_target_session(held, open_target_weights=funded,
                                           asset_returns=asset_returns, cost_bps=cost_bps)
            rets.append(float(result["net_return"]))
            held = np.asarray(result["ending_weights"], dtype=float)
        return np.array(rets)

    rows = []
    for name, target in (("soxx_buy_hold", np.array([0.0, 1.0, 0.0])),
                         ("soxx_cash_50_50", np.array([0.0, 0.5, 0.5]))):
        rets = settle(target)
        row = {"reference": name, "cost_bps": cost_bps, **period_metrics(rets)}
        if native:
            row["native_status"] = _native_check(root, block, target, rets, cost_bps=cost_bps)
        rows.append(row)
    return pd.DataFrame(rows)


def _native_check(root: Path, block: pd.DataFrame, target: np.ndarray, reference_returns: np.ndarray,
                  *, cost_bps: float) -> str:
    """Open-only native reconciliation of a reference target schedule.

    Builds the market frame from Qlib open prices so the retained returns agree
    with the native opens. Open-only: recovery is not representable here.
    """
    _quiet()
    try:
        from src.analysis.stage1_regime_validation import native_open_reconciliation
        from src.data import QlibProviderConfig, load_qlib_feature_frame
    except Exception as error:  # noqa: BLE001
        return f"unavailable: {error}"
    schedule = block[["signal_date", "trade_date", "exit_date"]].copy()
    schedule["open_soxl_weight"] = 0.0
    schedule["open_soxx_weight"] = float(target[1])
    schedule["open_cash_weight"] = float(target[2])
    schedule["recovery_action"] = False
    schedule["execute_open_target"] = True
    # ``native_open_reconciliation`` expects a research strategy_return column to
    # reconcile against; the reference net returns are filled in after the market
    # frame is built (below).
    schedule["strategy_return"] = 0.0
    config = QlibProviderConfig(provider_uri=root / PROVIDER, region="us", market="all", freq="day")
    start = str(pd.Timestamp(schedule["trade_date"].min()).date())
    end = str(pd.Timestamp(schedule["exit_date"].max()).date())
    raw = load_qlib_feature_frame(config, instruments=["SOXL", "SOXX"], fields=["$open"],
                                  start=start, end=end)
    opens = raw["$open"].unstack(level=0)
    opens.index = pd.to_datetime(opens.index)
    market = schedule[["signal_date", "trade_date", "exit_date"]].copy()
    market["soxl_return"] = (opens["SOXL"].reindex(schedule["exit_date"]).to_numpy()
                             / opens["SOXL"].reindex(schedule["trade_date"]).to_numpy() - 1.0)
    market["soxx_return"] = (opens["SOXX"].reindex(schedule["exit_date"]).to_numpy()
                             / opens["SOXX"].reindex(schedule["trade_date"]).to_numpy() - 1.0)
    # Research net return for the reference target, fee-charged.
    _require(len(reference_returns) == len(schedule), "reference returns must align with the schedule")
    schedule["strategy_return"] = reference_returns
    try:
        result = native_open_reconciliation(market, schedule, provider_uri=str(root / PROVIDER),
                                            cost_bps=cost_bps)
        return result.get("audit", {}).get("status", "failed")
    except Exception as error:  # noqa: BLE001 - recorded honestly, never assumed
        return f"needs_data: {error}"


def run_action_marginal_diagnosis(*, root: Path | str | None = None,
                                  card_dir: Path | str | None = None) -> dict[str, Any]:
    _quiet()
    spec_root = resolve_root(root)
    spec = DiagnosisSpec(root=spec_root)
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    target_card.mkdir(parents=True, exist_ok=True)

    plan_bundle = load_plan(spec.root)
    plan = plan_bundle["plan"]
    sources = plan["frozen_sources"]
    windows = plan["windows"]
    thresholds = plan.get("decidable_rule") or {"execute_threshold": 0.50,
                                                    "max_skippable_turnover": 0.05,
                                                    "mandatory_derisk_beta_delta": 0.0}
    stats = plan.get("statistics") or {"block_sessions": 20, "resamples": 2000}
    development = tuple(windows["development"])
    outer = tuple(windows["outer_reused_diagnostic"])

    loaded = load_frozen(spec.root, source_card=sources["primary_evidence_card"],
                         source_recorder=sources["primary_evidence_recorder"],
                         object_sha256=sources["source_object_sha256"])
    replays, prepared = loaded["replays"], loaded["prepared"]

    # 1. True OOS conditional information from each year's own saved pipeline.
    frame = prepared["legacy_training"]
    pipelines = prepared["legacy_models"]
    metadata = load_action_metadata(spec.root, action_card=sources["action_card"],
                                    action_recorder=sources["action_recorder_id"])
    maturity = verify_model_identity_and_maturity(metadata["dataset_audit"], metadata["diagnostics"],
                                                  pipelines=pipelines)
    _require(maturity["status"] == "ok", f"model maturity/identity check did not pass: {maturity}")
    predictions = true_oos_predictions(frame, pipelines)
    # Align the route-rule inputs to the scored rows and mark decidable sessions.
    rule_inputs = frame[["signal_date", "effective_beta_delta", "open_turnover"]].copy()
    rule_inputs["signal_date"] = pd.to_datetime(rule_inputs["signal_date"])
    scored = predictions.merge(rule_inputs, on="signal_date", how="left")
    decidable = decidable_mask(scored, **{"threshold": thresholds["execute_threshold"],
                                          "maximum_skippable_turnover": thresholds["max_skippable_turnover"],
                                          "mandatory_derisk_beta_delta": thresholds["mandatory_derisk_beta_delta"]})
    info = conditional_information(scored, decidable=decidable,
                                   block=stats["block_sessions"], resamples=stats["resamples"])

    # 2. Per-segment, per-cost contribution (own reset, never multiplied).
    contribution = []
    for segment, window in (("development", development), ("outer_reused_diagnostic", outer)):
        for cost in (10.0, 25.0):
            contribution.extend(per_segment_contribution(replays, window=window, segment=segment,
                                                         cost_bps=cost))
    contribution_table = pd.DataFrame(contribution)

    # 3. References over the development segment's own sessions at both costs.
    dev_block = replays[(replays["cost_bps"] == 10.0) & _segment_mask(replays, development)
                        & (replays["variant"] == VARIANTS["active"])]
    references = pd.concat([reference_paths(spec.root, dev_block, cost_bps=c)
                            for c in (10.0, 25.0)], ignore_index=True)

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "diagnostic",
        "decision": ("diagnostic_reported_no_strategy" if info.get("status") == "ok"
                     else "needs_data_no_aligned_active_label"),
        "config_source": "plan.json (hashes/id/windows read from the frozen plan)",
        "promotion_eligible": False,
        "objective_achieved": False,
        "source_card": sources["primary_evidence_card"],
        "source_recorder_id": sources["primary_evidence_recorder"],
        "source_api": ["qlib.workflow.R.get_recorder", "Recorder.load_object"],
        "new_models": 0,
        "plan_sha256": plan_bundle["plan_sha256"],
        "look_ahead_check": "each year predicted by that year's saved pipeline; label matured strictly after decision",
        "model_maturity": maturity,
        "source_metadata_sha256": {"dataset_audit.json": metadata["dataset_audit_sha256"],
                                  "training_diagnostics.csv": metadata["training_diagnostics_sha256"]},
        "decidable_rule": thresholds,
        "statistics": stats,
        "windows": {"development": list(development), "outer_reused_diagnostic": list(outer),
                    "segments_reset_each": True, "concatenation": "none"},
        "conditional_information": info,
        "contribution": contribution_table.to_dict(orient="records"),
        "references": references.to_dict(orient="records"),
        "prior_claim_retracted": [
            "Cross-account active-minus-no_hold returns are NOT the classifier label and were withdrawn.",
            "The earlier AUC 0.321 / 2438-sample result was produced by that wrong proxy and is void.",
            "A near-zero unconditional mean does not imply E[y|X]=0.",
        ],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": ["predictions.pkl", "contribution.pkl"]},
        "limitations": [
            "Diagnostic only; produces no strategy and grants no promotion.",
            "Reused diagnostic window is not a clean holdout.",
            "The 50/50 SOXX-cash reference is rebalanced to target each session and pays fees.",
            "native_open_reconciliation supports open-only schedules and rejects recovery; it is not a full intraday ledger.",
            "A complete intraday staged account needs a real 15-minute ledger; 2016-2018 predates that data, so any intraday verification is layered daily-or-2019+ only and must not fabricate earlier bars.",
            "The parent economic recorder's own-state path is a one-session reset proxy as recorded by its source card.",
        ],
    }
    summary.update(_record(spec=spec, card_dir=target_card, summary=summary,
                           predictions=scored, contribution=contribution_table, info=info))
    write_card_files(summary, card_dir=target_card)
    return summary


def _record(*, spec, card_dir, summary, predictions, contribution, info) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R

    recorder_uri = (card_dir / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / PROVIDER).resolve()), region="us")
    saved = {
        "predictions": predictions,
        "contribution": contribution,
        "conditional_information": pd.DataFrame([{k: v for k, v in info.items()
                                                   if not isinstance(v, (list, dict))}]),
        "calibration": pd.DataFrame(info.get("calibration", [])),
    }
    with R.start(experiment_name=spec.experiment, recorder_name="diagnosis", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, source_card=summary["source_card"],
                            source_recorder_id=summary["source_recorder_id"], new_models=0)
        if info.get("status") == "ok":
            recorder.log_metrics(auc=info["auc"], rank_ic=info["rank_ic"])
        recorder.save_objects(**saved)
        recorder_id = recorder.id
        experiment_id = str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    # Use the exact object names Qlib wrote (no synthetic extension).
    names = sorted(saved)
    summary["artifact_policy"]["recorder_only_artifacts"] = names
    return {"recorder_id": recorder_id, "experiment_id": experiment_id,
            "recorder_uri": str(recorder_uri), "recorder_local_dir": str(local_dir),
            "qlib_docs_reference": QLIB_DOCS, "saved_objects": names}


def write_card_files(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summary["contribution"]).to_csv(card_dir / "contribution_comparison.csv", index=False)
    pd.DataFrame(summary["references"]).to_csv(card_dir / "reference_comparison.csv", index=False)
    pd.DataFrame(summary["conditional_information"].get("calibration", [])).to_csv(
        card_dir / "probability_calibration.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "source_card": summary["source_card"], "source_recorder_id": summary["source_recorder_id"],
        "new_models": 0, "windows": summary["windows"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    info = summary["conditional_information"]
    lines = [
        "# Stage1 action 边际信息诊断（中文结论）\n",
        "- 主对象：**现役 action logistic 模型**；预测为该年**已保存 pipeline**（仅用更早数据拟合）的输出，"
        "标签为**同一基线域**的真实 `execute_advantage`，严格成熟。\n",
        f"- 源证据（只读、对象 sha256 绑定，plan sha `{summary.get('plan_sha256','')[:12]}`）：recorder "
        f"`{summary['source_recorder_id']}`；窗口 dev {summary['windows']['development']}，"
        f"outer {summary['windows']['outer_reused_diagnostic']}（分段 reset，不拼接）。\n",
        "- 撤回：跨账户 active−no_hold 当标签的旧做法作废；旧 AUC 0.321 / 2438 样本结果无效。\n",
        "- 模型成熟度：逐年断言 label_available_at_max < test_decision_at（源 dataset_audit/training_diagnostics）。\n",
        "- native 边界：SOXX/cash 参照仅经 open-only 原生对账；无完整 intraday 原生账本（2016-18 无 15min）。\n",
    ]
    if info.get("status") == "ok":
        ci = info["advantage_bps_ci"]
        ric = info["rank_ic_ci"]
        ac = info["auc_ci"]
        lines.append(
            f"- 真 OOS 条件信息（decidable n={info['decidable_sessions']} / full {info['full_oos_sessions']}）："
            f"AUC {info['auc']:.4f} CI [{ac['lower']:.3f},{ac['upper']:.3f}] null={ac['null']}；"
            f"rank IC {info['rank_ic']:+.4f} CI [{ric['lower']:+.3f},{ric['upper']:+.3f}] null={ric['null']}；"
            f"标签均值 CI [{ci['lower']:.3f},{ci['upper']:.3f}] bp。\n")
        lines.append(
            "- 显著性声明：仅报上述 CI；若 rankIC/AUC/均值 CI 均含各自 null，则不声称存在条件信息，"
            "也不声称反号可用。\n")
    else:
        lines.append(f"- 条件信息：needs_data（{info.get('reason')}）。\n")
    lines.append("- 分段×成本净贡献（各自 reset，不跨段相乘）：\n")
    for row in summary["contribution"]:
        lines.append(
            f"  - {row['segment']} @ {row['cost_bps']:.0f}bp {row['variant']}："
            f"收益 {row['return_pct']:.1f}%，净贡献 {row['net_contribution_pct']:+.1f}pp，"
            f"MaxDD {row['max_drawdown_pct']:.2f}%，gap {row['gaps']}。\n")
    lines.append("- SOXX 参照（同成交、同成本、含费用）：\n")
    for row in summary["references"]:
        lines.append(f"  - {row['reference']} @ {row['cost_bps']:.0f}bp：收益 {row['return_pct']:.1f}%，"
                     f"MaxDD {row['max_drawdown_pct']:.2f}%\n")
    lines.append("- 下一步：只在 OOS 条件信息与分段归因给出证据后才选具体修正实验；本卡不导向叠加风险 cap。\n")
    lines.append(f"- 子 Recorder：{summary.get('recorder_id')}；父只读不改。\n")
    return "".join(lines)
