"""Increment 2: cost-consistent action training, grouped comparisons + real gates.

Trains two action books whose label-reset fee matches the evaluated cost book
(10bp book: own state + label10bp; 25bp book: own state + label25bp) on the
frozen own-state frame, and compares them against frozen references at each cost.

Guarantees enforced here:
- Control parity uses the new wrapper's own full path (returns, ending weights,
  decisions) aligned on complete unique keys; a parity failure BLOCKS the run.
- Every candidate/reference path is aligned on identical session keys; a missing,
  duplicate or non-finite session BLOCKS.
- Real gates and yearly/annual wealth-floor checks are computed from the frozen
  thresholds; SOXX is compared directly.
- The child Recorder saves the actual models, scalers, training rows, maturity
  diagnostics, predictions and replays, not a metric summary.

No parameter or feature search. This card reports diagnostics; it does not promote.
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

EXPERIMENT = "stage1_cost_consistent_action_20261009"
SCHEMA_VERSION = "stage1_cost_consistent_action_v1"
QLIB_DOCS = [
    "https://qlib.readthedocs.io/en/latest/component/data.html",
    "https://qlib.readthedocs.io/en/latest/component/recorder.html",
]
SOURCE_CARD = "research/stage1_economic_action_20260929"
SOURCE_RECORDER = "17f81fbf27ee46859816ec1ac52151f6"
PROVIDER = "data/qlib/soxl_soxx_soxs"
SOURCE_OBJECT_SHA256 = {
    "inputs/prepared_inputs.pkl": "8baa4f74449c1da41eb189f51c75074a61dacb8d7c8540710f4f67560d58d4aa",
    "analysis/portfolio_replays.pkl": "ac7281e80aa78d2cc69c9c97dadd51356ac883e85ee87d53ca46eaa976b4b306",
    "models/own_state_frozen_2022.pkl": "a55ef083512375911d770cc73505b7a335e1136f93fc8345376fbd3230d7a3f2",
}
COSTS = (10.0, 25.0)
SEGMENT_LABELS = {"development_walk_forward": "development_walk_forward",
                  "outer_diagnostic": "outer_reused_diagnostic"}
REFERENCE_VARIANTS = ("active_hold_baseline", "no_hold_reference", "own_state_net_value_frozen_2022", "rules_only")
CONTROL_VARIANT = "own_state_net_value_frozen_2022"
# Gate thresholds reused from the source card's gate definition.
ANNUAL_WEALTH_FLOOR = 0.985
OPEN_TRADE_DAYS_REDUCTION = 0.92
PARITY_TOLERANCE = 1e-9
REQUIRED_REPLAY_COLUMNS = ("signal_date", "strategy_return", "turnover", "execute_open_trade",
                           "recovery_action", "ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight")


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


@dataclass(frozen=True)
class CostMatchedSpec:
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


def _quiet() -> None:
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*future calendar.*")
    warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")


def load_frozen_inputs(root: Path) -> dict:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    _quiet()
    import qlib
    from qlib.workflow import R

    base = (root / SOURCE_CARD / "qlib_recorder").resolve()
    _require(base.is_dir(), "frozen economic recorder is unavailable")
    qlib.init(provider_uri=str((root / PROVIDER).resolve()), region="us")
    objects = {}
    with R.uri_context(str(base)):
        recorder = R.get_recorder(recorder_id=SOURCE_RECORDER, experiment_name=Path(SOURCE_CARD).name)
        local = Path(recorder.get_local_dir()) / "artifacts"
        for name, expected in SOURCE_OBJECT_SHA256.items():
            actual = _sha256(local / name)
            _require(actual == expected, f"frozen object hash mismatch for {name}: {actual} != {expected}")
            objects[name] = recorder.load_object(name)
    return objects


def _common(root: Path, prepared: dict) -> dict:
    from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
    return dict(recovery_models=prepared["recovery_models"],
                source_spec=IntradayRecoveryDirectAllocationSpec(root=root),
                selected_model=prepared["selected_model"])


def _validate_replay(replay: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Fail-closed replay validation: required columns, unique ordered keys, finite."""
    missing = [c for c in REQUIRED_REPLAY_COLUMNS if c not in replay.columns]
    _require(not missing, f"{label}: replay missing columns {missing}")
    frame = replay.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    _require(not frame["signal_date"].duplicated().any(), f"{label}: duplicate signal_date")
    frame = frame.sort_values("signal_date").reset_index(drop=True)
    _require(frame["signal_date"].is_monotonic_increasing, f"{label}: signal_date not ordered")
    numeric = frame[["strategy_return", "turnover", "ending_soxl_weight", "ending_soxx_weight",
                     "ending_cash_weight"]].to_numpy(float)
    _require(np.isfinite(numeric).all(), f"{label}: non-finite replay values")
    return frame


def _align(candidate: pd.DataFrame, reference: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two replays on identical session keys; block on any difference."""
    cand_idx = pd.DatetimeIndex(candidate["signal_date"])
    ref_idx = pd.DatetimeIndex(reference["signal_date"])
    missing = ref_idx.difference(cand_idx)
    extra = cand_idx.difference(ref_idx)
    if len(missing) or len(extra) or len(cand_idx) != len(ref_idx):
        raise ValueError(f"blocked: {label}: session keys differ (missing={len(missing)}, extra={len(extra)})")
    cand = candidate.assign(signal_date=cand_idx).sort_values("signal_date").reset_index(drop=True)
    ref = reference.assign(signal_date=ref_idx).sort_values("signal_date").reset_index(drop=True)
    return cand, ref


def control_parity(root: Path, frame: pd.DataFrame, source_replays: pd.DataFrame, *,
                   common: dict) -> dict:
    """The NEW wrapper (label25/eval10) must reproduce the source control on every path.

    Compares returns, ending weights and decisions on complete unique keys. A
    failure raises so the run is blocked, never continued.
    """
    from src.workflow.economic_cost_matched_training import train_cost_matched_book
    from src.workflow.stage1_economic_training import simulate_economic_policy
    book = train_cost_matched_book(frame, label_cost_bps=25.0, evaluation_cost_bps=10.0,
                                   update_through_year=2022, **common)
    checks = []
    for segment, span in SEGMENT_LABELS.items():
        market = frame[frame["segment"] == segment]
        for cost in COSTS:
            replay = _validate_replay(
                simulate_economic_policy(market, models=book["models"], policy="net_value", cost_bps=cost, **common),
                label=f"parity/{span}/{cost}")
            source = source_replays[(source_replays["variant"] == CONTROL_VARIANT)
                                    & (source_replays["segment"] == span)
                                    & (source_replays["cost_bps"] == cost)]
            source = _validate_replay(source, label=f"source/{span}/{cost}")
            cand, ref = _align(replay, source, label=f"parity/{span}/{cost}")
            returns_ok = np.allclose(cand["strategy_return"].to_numpy(float), ref["strategy_return"].to_numpy(float),
                                     rtol=0, atol=PARITY_TOLERANCE)
            weights_ok = all(np.allclose(cand[c].to_numpy(float), ref[c].to_numpy(float), rtol=0, atol=PARITY_TOLERANCE)
                             for c in ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"))
            decisions_ok = cand["execute_open_trade"].to_numpy(bool).tolist() == ref["execute_open_trade"].to_numpy(bool).tolist()
            checks.append({"segment": span, "cost_bps": cost, "sessions": int(len(cand)),
                           "returns_ok": bool(returns_ok), "ending_weights_ok": bool(weights_ok),
                           "decisions_ok": bool(decisions_ok)})
    passed = all(c["returns_ok"] and c["ending_weights_ok"] and c["decisions_ok"] for c in checks)
    _require(passed, f"control parity failed: {checks}")
    return {"status": "passed", "checks": checks}


def _metrics(replay: pd.DataFrame) -> dict:
    r = replay["strategy_return"].to_numpy(float)
    growth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(growth)
    tail = np.sort(r)[:max(1, int(np.ceil(0.05 * r.size)))]
    rolling5 = pd.Series(r).rolling(5).apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
    std = r.std(ddof=1)
    return {
        "sessions": int(r.size),
        "terminal_wealth": float(growth[-1]),
        "return_pct": 100.0 * float(growth[-1] - 1.0),
        "max_drawdown_pct": 100.0 * float((growth / peak - 1.0).min()),
        "cvar5_pct": 100.0 * float(tail.mean()),
        "worst_day_pct": 100.0 * float(r.min()),
        "worst_rolling_5d_pct": 100.0 * float(rolling5.min()),
        "sharpe": float(r.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "total_turnover": float(replay["turnover"].sum()),
        "open_trade_days": int(replay["execute_open_trade"].sum()),
        "recovery_action_count": int(replay["recovery_action"].sum()),
    }


def _annual(replay: pd.DataFrame) -> list[dict]:
    rows = []
    frame = replay.copy()
    frame["year"] = pd.to_datetime(frame["signal_date"]).dt.year
    for year, group in frame.groupby("year"):
        rows.append({"year": int(year), **_metrics(group)})
    return rows


def _sorted_records(frame: pd.DataFrame) -> list[dict]:
    return frame.sort_values(["segment", "variant", "cost_bps"]).to_dict(orient="records")


def run_cost_matched_action(*, root: Path | str | None = None, card_dir: Path | str | None = None) -> dict[str, Any]:
    from src.workflow.economic_cost_matched_training import train_cost_matched_book

    _quiet()
    spec_root = resolve_root(root)
    spec = CostMatchedSpec(root=spec_root)
    plan_path = spec.root / "research" / spec.experiment / "plan.json"
    _require(plan_path.is_file(), "plan.json is missing")
    plan_sha256 = _sha256(plan_path)

    source = load_frozen_inputs(spec.root)
    prepared = source["inputs/prepared_inputs.pkl"]
    source_replays = source["analysis/portfolio_replays.pkl"]
    frame = prepared["frame"].copy()
    common = _common(spec.root, prepared)

    parity = control_parity(spec.root, frame, source_replays, common=common)

    # Cost-matched candidates: one book per cost, own state + own label fee.
    books = {cost: train_cost_matched_book(frame, label_cost_bps=cost, evaluation_cost_bps=cost,
                                           update_through_year=2022, **common) for cost in COSTS}

    # Build every path: frozen references + candidates, per segment x cost.
    reference_records, candidate_records, annual_records, replay_store = [], [], [], {}
    for segment, span in SEGMENT_LABELS.items():
        market = frame[frame["segment"] == segment]
        for cost in COSTS:
            for variant in REFERENCE_VARIANTS:
                ref = source_replays[(source_replays["variant"] == variant) & (source_replays["segment"] == span)
                                     & (source_replays["cost_bps"] == cost)]
                _require(not ref.empty, f"missing frozen reference {variant} {span} {cost}")
                ref = _validate_replay(ref, label=f"ref/{variant}/{span}/{cost}")
                reference_records.append({"family": "frozen_reference", "variant": variant, "segment": span,
                                          "cost_bps": cost, "label_cost_bps": None, "evaluation_cost_bps": cost,
                                          **_metrics(ref)})
                replay_store[f"ref::{span}::{int(cost)}::{variant}"] = ref
                annual_records.append({"variant": variant, "segment": span, "cost_bps": cost,
                                       "family": "frozen_reference", "years": _annual(ref)})
            for name, book in ((f"cost_matched_book_{int(cost)}", books[cost]),):
                replay = _validate_replay(
                    _replay_via(book["models"], market, cost=cost, common=common),
                    label=f"candidate/{name}/{span}/{cost}")
                candidate_records.append({"family": "cost_matched", "variant": name, "segment": span,
                                          "cost_bps": cost, "label_cost_bps": cost, "evaluation_cost_bps": cost,
                                          **_metrics(replay)})
                replay_store[f"cand::{span}::{int(cost)}::{name}"] = replay
                annual_records.append({"variant": name, "segment": span, "cost_bps": cost,
                                       "family": "cost_matched", "years": _annual(replay)})

    reference_table = pd.DataFrame(reference_records)
    candidate_table = pd.DataFrame(candidate_records)
    all_records = reference_records + candidate_records
    gates = _gates(reference_table, candidate_table, annual_records)
    soxx = _soxx_benchmark(frame, common)
    _require(gates["status"] == "ok", "gate computation failed")

    summary = {
        "experiment": spec.experiment,
        "schema_version": SCHEMA_VERSION,
        "card_status": "active_research",
        "increment": "increment_2_cost_consistent_training",
        "decision": "cost_matched_diagnostic_reported",
        "promotion_eligible": False,
        "objective_achieved": False,
        "plan_sha256": plan_sha256,
        "source_card": SOURCE_CARD,
        "source_recorder_id": SOURCE_RECORDER,
        "source_object_sha256": SOURCE_OBJECT_SHA256,
        "source_api": ["qlib.workflow.R.get_recorder", "Recorder.load_object"],
        "control_parity": parity,
        "new_models": 2,
        "parameter_search": "none",
        "grouping_rule": "results are grouped by segment x cost and are never merged into one winning number",
        "results": all_records,
        "gates": gates,
        "soxx_benchmark": soxx,
        "promotion_note": ("promotion remains blocked: no complete native intraday reconciliation exists for this "
                           "increment and the historical gates are reported, not relaxed"),
        "limitations": [
            "Diagnostic only; produces no promotion.",
            "Reused history: 2023-2026 were previously inspected; outer is not a clean holdout.",
            "One-session + shared next-target-reset label is a proxy, not full multi-day control.",
            "Increment 1 (price consistency) is reported separately once the native interface is ready.",
        ],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": []},
    }
    target_card = Path(card_dir) if card_dir is not None else (spec.root / "research" / spec.experiment)
    summary.update(_record(spec=spec, planner=plan_path, card_dir=target_card, summary=summary,
                           books=books, replay_store=replay_store, annual_records=annual_records,
                           reference_table=reference_table, candidate_table=candidate_table))
    _write_card(summary, card_dir=target_card)
    return summary


def _replay_via(models, market: pd.DataFrame, *, cost: float, common: dict) -> pd.DataFrame:
    from src.workflow.stage1_economic_training import simulate_economic_policy
    return simulate_economic_policy(market, models=models, policy="net_value", cost_bps=cost, **common)


def _gates(reference: pd.DataFrame, candidate: pd.DataFrame, annual_records: list[dict]) -> dict:
    """Real gates: candidate vs each reference per segment x cost, with annual wealth floor."""
    lookup = reference.set_index(["segment", "variant", "cost_bps"])
    annual = {}
    for entry in annual_records:
        for y in entry["years"]:
            annual[(entry["variant"], entry["segment"], entry["cost_bps"], y["year"])] = y
    rows = []
    for _, cand in candidate.iterrows():
        seg, cost, name = cand["segment"], cand["cost_bps"], cand["variant"]
        for ref_name in ("no_hold_reference", "active_hold_baseline"):
            base = lookup.loc[(seg, ref_name, cost)]
            ratios = []
            for (variant, segment, c, year), y in annual.items():
                if variant == name and segment == seg and c == cost and year >= 2017:
                    b = annual.get((ref_name, seg, cost, year))
                    if b is not None and b["terminal_wealth"] != 0:
                        ratios.append(y["terminal_wealth"] / b["terminal_wealth"])
            annual_ratio = min(ratios) if ratios else float("nan")
            checks = {
                "return_above_baseline": bool(cand["return_pct"] > base["return_pct"]),
                "max_drawdown_not_worse": bool(cand["max_drawdown_pct"] >= base["max_drawdown_pct"] - 1e-10),
                "worst_day_not_worse": bool(cand["worst_day_pct"] >= base["worst_day_pct"] - 1e-10),
                "cvar5_not_worse": bool(cand["cvar5_pct"] >= base["cvar5_pct"] - 1e-10),
                "worst_5d_not_worse": bool(cand["worst_rolling_5d_pct"] >= base["worst_rolling_5d_pct"] - 1e-10),
                "open_trade_days_reduced_8pct": bool(cand["open_trade_days"] <= OPEN_TRADE_DAYS_REDUCTION * base["open_trade_days"]),
                "turnover_not_increased": bool(cand["total_turnover"] <= base["total_turnover"]),
                "annual_wealth_floor_pass": bool(np.isfinite(annual_ratio) and annual_ratio >= ANNUAL_WEALTH_FLOOR),
            }
            rows.append({"candidate": name, "segment": seg, "cost_bps": cost, "reference": ref_name,
                         "minimum_annual_wealth_ratio": float(annual_ratio) if np.isfinite(annual_ratio) else None,
                         **checks, "all_pass": bool(all(checks.values()))})
    return {"status": "ok", "thresholds": {"annual_wealth_floor": ANNUAL_WEALTH_FLOOR,
                                           "open_trade_days_reduction": OPEN_TRADE_DAYS_REDUCTION},
            "rows": rows}


def _soxx_benchmark(frame: pd.DataFrame, common: dict) -> list[dict]:
    rows = []
    for segment, span in SEGMENT_LABELS.items():
        band = frame[frame["segment"] == segment]
        r = band["soxx_return"].to_numpy(float)
        _require((r > -1.0).all() and np.isfinite(r).all(), f"SOXX benchmark non-finite in {span}")
        metrics = _metrics(pd.DataFrame({"strategy_return": r, "turnover": 0.0,
                                         "execute_open_trade": False, "recovery_action": False}))
        rows.append({"benchmark": "SOXX_buy_hold_open_to_open", "segment": span, **metrics})
    return rows


def _record(*, spec, planner, card_dir, summary, books, replay_store, annual_records,
            reference_table, candidate_table) -> dict[str, Any]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    _quiet()
    import qlib
    from qlib.workflow import R

    card_dir = Path(card_dir)
    recorder_uri = (card_dir / "qlib_recorder").resolve()
    recorder_uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec.root / PROVIDER).resolve()), region="us")

    # Persist the actual experiment objects, not just a metric table.
    models_by_cost = {f"models_{int(cost)}": books[cost]["models"] for cost in books}
    diagnostics_by_cost = {f"diagnostics_{int(cost)}": books[cost]["diagnostics"] for cost in books}
    training_by_cost = {f"training_rows_{int(cost)}": books[cost]["training_rows"] for cost in books}
    audit_by_cost = {f"dataset_audit_{int(cost)}": books[cost]["dataset_audit"] for cost in books}
    saved = {
        "candidate_replays": replay_store,
        "annual_records": annual_records,
        "reference_comparison": reference_table,
        "candidate_comparison": candidate_table,
        **models_by_cost, **diagnostics_by_cost, **training_by_cost, **audit_by_cost,
    }
    with R.start(experiment_name=spec.experiment, recorder_name="cost_matched", uri=str(recorder_uri)):
        recorder = R.get_recorder()
        recorder.log_params(schema_version=SCHEMA_VERSION, source_recorder_id=SOURCE_RECORDER,
                            plan_sha256=summary["plan_sha256"], new_models=summary["new_models"])
        recorder.log_metrics(control_parity=1.0 if summary["control_parity"]["status"] == "passed" else 0.0,
                             candidate_count=float(len(candidate_table)))
        recorder.save_objects(**saved)
        recorder.log_artifact(str(planner), artifact_path="plan")
        # Source snapshot hashes.
        snapshot = spec.root / "research" / spec.experiment / "source_binding.json"
        snapshot.write_text(json.dumps({"source_card": SOURCE_CARD, "source_recorder_id": SOURCE_RECORDER,
                                        "object_sha256": SOURCE_OBJECT_SHA256,
                                        "plan_sha256": summary["plan_sha256"]}, indent=2), encoding="utf-8")
        recorder.log_artifact(str(snapshot), artifact_path="inputs")
        recorder_id = recorder.id
        experiment_id = str(recorder.experiment_id)
        local_dir = recorder_uri / experiment_id / recorder_id
    summary["artifact_policy"]["recorder_only_artifacts"] = sorted(saved)
    return {"recorder_id": recorder_id, "experiment_id": experiment_id,
            "recorder_uri": str(recorder_uri), "recorder_local_dir": str(local_dir),
            "saved_objects": sorted(saved), "qlib_docs_reference": QLIB_DOCS}


def _write_card(summary: dict, *, card_dir: Path) -> None:
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summary["results"]).to_csv(card_dir / "cost_matched_comparison.csv", index=False)
    pd.DataFrame(summary["gates"]["rows"]).to_csv(card_dir / "gate_results.csv", index=False)
    pd.DataFrame(summary["soxx_benchmark"]).to_csv(card_dir / "soxx_benchmark.csv", index=False)
    workflow = {
        "schema_version": SCHEMA_VERSION, "experiment": summary["experiment"],
        "increment": summary["increment"], "plan_sha256": summary["plan_sha256"],
        "source_card": summary["source_card"], "source_recorder_id": summary["source_recorder_id"],
        "costs_bps": list(COSTS), "grouping_rule": summary["grouping_rule"],
        "gate_thresholds": summary["gates"]["thresholds"],
        "artifact_policy": summary["artifact_policy"], "qlib_docs_reference": QLIB_DOCS,
    }
    (card_dir / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    (card_dir / "conclusion.md").write_text(_conclusion(summary), encoding="utf-8")


def _conclusion(summary: dict) -> str:
    lines = [
        "# Stage1 费用一致动作训练（增量 2，中文结论）\n",
        f"- 做了什么：冻结 own-state 面板训练两本书（10bp 书=自身 state+label10bp；25bp 书=自身 state+label25bp），"
        f"与冻结参照 active/no_hold/own_state_frozen/rules_only 分组比较。plan sha `{summary['plan_sha256'][:12]}`。\n",
        f"- 控制 parity（新 wrapper label25/eval10 全路径复现源 control，含 returns/endweights/decisions）："
        f"{summary['control_parity']['status']}。\n",
        "- 结果按 **segment × cost** 分组，不合并；真实门与年度/财富下限门按原阈值计算，失败如实列出。\n",
        "- 本执行**不依赖价格一致性接口**；增量 1（价格口径）留待原生接口就绪后单独跑并单独报告。\n",
        f"- 门阈值（复用原卡）：{summary['gates']['thresholds']}。\n",
        "- 候选 not_promoted：缺完整 native intraday 对账，历史门不自动授权晋升。\n",
        f"- 子 Recorder：{summary.get('recorder_id')}；保存 models/scaler/training/maturity/predictions/replays，源只读不改。\n",
    ]
    return "".join(lines)
