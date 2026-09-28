"""Replay an explicitly frozen SIM candidate from prior account start dates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.position_accounting import settle_target_session
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec, _load_first_hour_features,
)
from src.workflow.stage1_futu_sim import infer_frozen_allocation
from src.workflow.stage1_open_action_value import Stage1OpenActionValueSpec, _simulate_on_policy_router
from src.workflow.stage1_runtime_release import load_runtime_release
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "stage1_origin_replay_20260928"
SYMBOLS = ("SOXL", "SOXX")


def build_origin_window(inference, prices, intraday, *, start: str, end: str):
    """Keep the last known target and settle the final session at its own close."""
    sessions = prices.index[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    if len(sessions) == 0 or sessions[0] != pd.Timestamp(start) or sessions[-1] != pd.Timestamp(end):
        raise ValueError("blocked: replay endpoints are not fully observed trading sessions")
    source = inference.reset_index().set_index("trade_date", drop=False)
    if source.index.has_duplicates:
        raise ValueError("blocked: duplicate allocation execution dates")
    rows = []
    for index, date in enumerate(sessions):
        row = source.loc[date].copy()
        if pd.Timestamp(row["signal_date"]) >= date or row[["soxl_weight", "soxx_weight", "cash_weight"]].isna().any():
            raise ValueError("blocked: allocation was not known before this execution")
        terminal = index == len(sessions) - 1
        exit_date = date if terminal else sessions[index + 1]
        field = "close" if terminal else "open"
        row["exit_date"] = exit_date
        row["exit_basis"] = "terminal_close" if terminal else "next_session_open"
        for symbol in SYMBOLS:
            row[f"{symbol.lower()}_return"] = prices.loc[exit_date, (field, symbol)] / prices.loc[date, ("open", symbol)] - 1.0
        for name, value in intraday.reindex([date]).iloc[0].items():
            row[name] = value
        row["first_hour_observed"] = date in intraday.index
        rows.append(row)
    result = pd.DataFrame(rows).reset_index(drop=True)
    return result


def _buy_hold(frame, symbol: str):
    asset = SYMBOLS.index(symbol)
    previous = np.array([0.0, 0.0, 1.0])
    equity = 1.0
    rows = []
    for number, (_, row) in enumerate(frame.iterrows()):
        target = None
        if number == 0:
            # Fund the initial 10bp fee from cash without borrowing.
            target = np.zeros(3)
            target[asset] = 1.0 / 1.001
            target[2] = 1.0 - target[asset]
        settled = settle_target_session(
            previous, open_target_weights=target,
            asset_returns=[row["soxl_return"], row["soxx_return"], 0.0], cost_bps=10.0,
        )
        previous = settled["ending_weights"]
        equity *= 1.0 + settled["net_return"]
        rows.append({
            "signal_date": row["signal_date"], "trade_date": row["trade_date"], "exit_date": row["exit_date"],
            "equity": equity, "strategy_return": settled["net_return"], "turnover": settled["turnover"],
            "execute_open_trade": number == 0, "recovery_action": False,
            "ending_soxl_weight": previous[0], "ending_soxx_weight": previous[1], "ending_cash_weight": previous[2],
            "fees": settled["fees"],
        })
    return pd.DataFrame(rows)


def run_origin_replay(*, root: Path | str) -> dict:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D
    from qlib.workflow import R
    from qlib.contrib.evaluate import risk_analysis

    root = Path(root).resolve()
    card = root / "production" / EXPERIMENT
    card.mkdir(parents=True, exist_ok=True)
    release_path = root / "production/stage1_futu_sim_v1/runtime_release.json"
    release_hash = hashlib.sha256(release_path.read_bytes()).hexdigest()
    loaded = load_runtime_release(root, release_path)
    release, action_summary = loaded["release"], loaded["action_summary"]
    inference, inference_audit = infer_frozen_allocation(
        root=root, artifact_dir=loaded["allocation_artifact_dir"],
        source_identity={"release_id": release["release_id"], "source_recorder_id": release["allocation"]["recorder_id"]},
    )
    provider = Path(inference_audit["market_audit"]["execution_provider_uri"])
    qlib.init(provider_uri=str(provider), region=REG_US, kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(start_time="2026-07-24", end_time="2026-09-25"))
    raw = D.features(list(SYMBOLS), ["$open", "$close"], start_time=calendar[0], end_time=calendar[-1])
    raw.columns = [name.removeprefix("$") for name in raw.columns]
    prices = raw.unstack("instrument").sort_index()
    source_spec = IntradayRecoveryDirectAllocationSpec(root=root)
    intraday = _load_first_hour_features(source_spec, end=pd.Timestamp("2026-09-25"))
    selected = action_summary["selected_model"]
    action_spec = Stage1OpenActionValueSpec(
        root=root, execute_probability_threshold=selected["execute_probability_threshold"],
        maximum_skippable_turnover=selected["maximum_skippable_turnover"],
        mandatory_derisk_beta_delta=selected["mandatory_derisk_beta_delta"],
    )
    observation_dir = root / "runtime/soxl/stage1_futu_sim/recovery_evidence/20260928/final_observation"
    position_path = observation_dir / "STOCK_AND_OPTION_positions.json"
    position_rows = json.loads(position_path.read_text())
    marks = {row["code"].removeprefix("US."): float(row["nominal_price"]) for row in position_rows}
    assert marks == {"SOXL": 140.49, "SOXX": 558.62}
    observation = json.loads((observation_dir / "observation_manifest.json").read_text())
    mark_returns = np.array([marks[symbol] / prices.loc[pd.Timestamp("2026-09-25"), ("close", symbol)] - 1.0 for symbol in SYMBOLS] + [0.0])
    observed_path = root / "production/stage1_recovery_20260928/summary.json"
    observed_summary = json.loads(observed_path.read_text())
    observed_reference = {
        "source": str(observed_path.relative_to(root)),
        "sha256": hashlib.sha256(observed_path.read_bytes()).hexdigest(),
        "account_observed_at": observed_summary["account_observed_at"],
        "valuation_basis": observed_summary["valuation_basis"],
        "metrics": observed_summary["metrics"],
    }
    outcomes, replays, risk_tables, windows, missing_intraday_audit = [], [], [], [], []
    for start in ("2026-08-03", "2026-07-27"):
        frame = build_origin_window(inference, prices, intraday, start=start, end="2026-09-25")
        windows.append(frame.assign(replay_start=start))
        for variant in ("active_HOLD", "baseline_without_HOLD", "SOXL_buy_hold", "SOXX_buy_hold"):
            if variant.endswith("buy_hold"):
                replay = _buy_hold(frame, variant.removesuffix("_buy_hold"))
            else:
                replay = _simulate_on_policy_router(
                    frame,
                    models={2026: loaded["models"]["frozen_open_action_value_model"]} if variant == "active_HOLD" else {},
                    recovery_models={2026: loaded["models"]["frozen_recovery_model"]},
                    source_spec=source_spec, spec=action_spec,
                    selected_model=action_summary["source_stage1"]["selected_model"], segment="origin_window_retrospective",
                )
                raw_beta = 3.0 * frame["soxl_weight"] + frame["soxx_weight"]
                recovery_parameters = action_summary["source_stage1"]["selected_model"]
                missing_required = (
                    frame["early_ret_soxx"].isna()
                    & replay["portfolio_drawdown"].ge(source_spec.recovery_eligibility_drawdown)
                    & np.minimum(raw_beta, recovery_parameters["action_beta"]).gt(replay["current_beta_cap"] + 1e-12)
                    & frame["alpha_probability"].ge(recovery_parameters["minimum_source_alpha_probability"])
                )
                if missing_required.any():
                    raise ValueError(f"needs data: missing first hour on recovery-eligible dates {frame.loc[missing_required, 'trade_date'].tolist()}")
                for number in frame.index[frame["early_ret_soxx"].isna()]:
                    missing_intraday_audit.append({
                        "start": start, "variant": variant,
                        "date": frame.loc[number, "trade_date"].date().isoformat(),
                        "portfolio_drawdown": float(replay.loc[number, "portfolio_drawdown"]),
                        "beta_cap": float(replay.loc[number, "current_beta_cap"]),
                        "recovery_target_beta": float(min(raw_beta.loc[number], recovery_parameters["action_beta"])),
                        "source_alpha_probability": float(frame.loc[number, "alpha_probability"]),
                        "recovery_eligible_without_intraday": False,
                        "missing_feature_consumed": False,
                    })
            final = replay.iloc[-1]
            close_equity = 100_000.0 * float(final["equity"])
            weights = final[["ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"]].to_numpy(dtype=float)
            marked_equity = close_equity * (1.0 + weights @ mark_returns)
            outcomes.append({
                "start": start, "variant": variant, "initial_equity": 100_000.0,
                "end": "2026-09-25_close", "sessions": len(frame),
                "close_equity": close_equity, "close_pnl": close_equity - 100_000.0,
                "close_return_pct": 100.0 * (close_equity / 100_000.0 - 1.0),
                "observed_0928_equity": marked_equity, "observed_0928_pnl": marked_equity - 100_000.0,
                "observed_0928_return_pct": 100.0 * (marked_equity / 100_000.0 - 1.0),
                "open_trade_days": int(replay["execute_open_trade"].sum()),
                "recovery_actions": int(replay["recovery_action"].sum()),
                "terminal_soxl_weight": weights[0], "terminal_soxx_weight": weights[1], "terminal_cash_weight": weights[2],
                "missing_intraday_dates_not_recovery_eligible": [date.date().isoformat() for date in frame.loc[frame["early_ret_soxx"].isna(), "trade_date"]],
            })
            pd.DataFrame(outcomes).to_csv(card / "backtest_benchmark_comparison.csv", index=False)
            replays.append(replay.assign(replay_start=start, variant=variant))
            risk = risk_analysis(pd.Series(replay["strategy_return"].to_numpy(), index=pd.to_datetime(replay["trade_date"])), freq="day").reset_index()
            risk_tables.append(risk.assign(replay_start=start, variant=variant))
    comparison = pd.DataFrame(outcomes)
    combined = pd.concat(replays, ignore_index=True)
    risks = pd.concat(risk_tables, ignore_index=True)
    assert hashlib.sha256(release_path.read_bytes()).hexdigest() == release_hash
    # All inference/provider changes finish before starting the evidence recorder.
    R.set_uri(str(card / "qlib_recorder"))
    with R.start(experiment_name=EXPERIMENT):
        recorder = R.get_recorder()
        R.log_params(release_id=release["release_id"], model_refit="false", initial_state="100000_cash", terminal_basis="2026-09-25 close", transaction_cost_bps="10")
        R.save_objects(artifact_path="origin_replay", **{"daily_replays.pkl": combined, "comparison.pkl": comparison,
                         "risk_analysis.pkl": risks, "qlib_prices.pkl": prices,
                         "aligned_inputs.pkl": pd.concat(windows, ignore_index=True),
                         "observed_account_reference.pkl": observed_reference})
        pd.testing.assert_frame_equal(comparison, recorder.load_object("origin_replay/comparison.pkl"))
        recorder_info = {"recorder_id": recorder.id, "recorder_local_dir": recorder.get_local_dir()}
    _remove_qlib_code_snapshot_artifacts(Path(recorder_info["recorder_local_dir"]))
    summary = {
        "experiment": EXPERIMENT, "card_type": "frozen_active_candidate_origin_window_replay", "release_id": release["release_id"],
        "release_sha256": release_hash, "allocation_recorder_id": release["allocation"]["recorder_id"], "action_recorder_id": release["action"]["recorder_id"],
        "model_refit": False, "parameter_selection": False, "orders_submitted": False, "research_promotion_status": "not_promoted",
        "interpretation": "Current frozen model retrospectively replayed from prior paper-account dates; not actual fills or continuously observed historical profit.",
        "price_basis": "Existing model execution provider uses dividend-adjusted OHLC; dividends are embodied in adjusted returns and no extra cash dividends are added. Continuous weights, not actual share quantities.",
        "provider_uri": str(provider), "price_data_source_manifest": "data/raw/soxs_defensive_v1/source_manifest.json",
        "terminal_policy": "2026-09-24 signal executes at actual 2026-09-25 open, then settles at actual 2026-09-25 close; no 2026-09-28 open read.",
        "initial_equity": 100_000.0, "transaction_cost_bps": 10.0,
        "benchmark_fee_policy": "One entry fee funded from initial cash; no terminal liquidation fee; no later rebalancing.",
        "observation_mark": {"query_started_at": observation["query_started_at"], "query_completed_at": observation["query_completed_at"], "prices": marks,
                             "source": str(position_path.relative_to(root)), "sha256": hashlib.sha256(position_path.read_bytes()).hexdigest(),
                             "basis": "Unchanged end-of-9/25 exposures marked with the same archived broker nominal prices; no 9/28 trades. The 9/25 adjusted endpoint has adjustment factor one."},
        "inference_smoothing_audit": inference_audit["smoothing"], "comparisons": outcomes, **recorder_info,
        "missing_intraday_audit": missing_intraday_audit,
        "observed_account_reference": observed_reference,
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["origin_replay/daily_replays.pkl", "origin_replay/aligned_inputs.pkl", "origin_replay/qlib_prices.pkl", "origin_replay/comparison.pkl", "origin_replay/risk_analysis.pkl", "origin_replay/observed_account_reference.pkl"]},
    }
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    risks.to_csv(card / "risk_analysis.csv", index=False)
    (card / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (card / "workflow.yaml").write_text(f"experiment: {EXPERIMENT}\nrelease_id: {release['release_id']}\nprovider_uri: {provider}\nmodel_refit: false\nparameter_selection: false\ninitial_equity: 100000\nstarts: ['2026-08-03', '2026-07-27']\nend: 2026-09-25_close\ntransaction_cost_bps: 10\nrecorder_id: {recorder.id}\n")
    lines = ["# 中文结论", "", "这是当前新HOLD模型按原模拟起点进行的事后重放，不是真实成交收益；两次均从10万美元现金独立启动控制器和持仓，没有重训、选参或提交委托。", "", "沿用原研究的分红复权价格和连续权重，逐笔净费10bp，股息不另加现金。9月25日使用9月24日已知信号，在9月25日开盘执行并结算至当日收盘，未借用9月28日开盘。", ""]
    for row in outcomes:
        lines.append(f"- {row['start']} / {row['variant']}：9/25收盘净值${row['close_equity']:,.2f}，收益{row['close_return_pct']:+.4f}%；按9/28 17:00同一存档标记且不调仓，净值${row['observed_0928_equity']:,.2f}，收益{row['observed_0928_return_pct']:+.4f}%。")
    actual = observed_reference["metrics"]
    lines.extend(["", f"8/3起点的新模型短窗表现弱于已恢复的真实账户：真实账户9/25收盘重估收益{actual['latest_close_revaluation_return_pct']:+.4f}%，9/28 17:00账户快照收益{actual['account_snapshot_return_pct']:+.4f}%。前者使用9/28已观察现金余额重估，并非独立9/25对账单；真实账户在停机后保留原仓位，不能把这种差异解释为两模型持续执行的对决。", "", "7/29首小时缺失，但两策略各自控制器均不满足恢复资格，已记录资格数值且未消费缺失字段。", "", "9/28附加标记采用此前账户快照的SOXL 140.49、SOXX 558.62；不是新的成交或策略执行。调仓日是连续权重模型统计，不保证真实填单次数。研究仍未晋升。", f"QLib recorder：{recorder.id}。"])
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    return summary
