from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
POSITION_CARD = ROOT / "research" / "cognitive_alpha_position_model_v1"
PROVIDER_URI = ROOT / "data" / "qlib" / "soxl"


def test_cognitive_alpha_signal_refresh_fails_closed_when_card_lags_data(tmp_path):
    from src.workflow import CognitiveAlphaSignalRefreshJob, CognitiveAlphaSignalRefreshSpec

    card = tmp_path / "card"
    card.mkdir()
    (card / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "stale_cogalpha",
                "instrument": "SOXL",
                "recorder_id": "recorder-stale",
                "provider_uri": PROVIDER_URI.as_posix(),
                "latest_signal": {
                    "action": "HOLD",
                    "as_of": "2026-01-05",
                    "target_position": 0.25,
                    "previous_target_position": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    job = CognitiveAlphaSignalRefreshJob(
        CognitiveAlphaSignalRefreshSpec(
            card_dir=card,
            provider_uri=PROVIDER_URI,
            output_dir=tmp_path / "signals",
            paper_cash=10000.0,
        )
    )

    with pytest.raises(ValueError, match="stale cognitive alpha signal"):
        job.run([{"source": "qlib.data.D.features", "max_datetime": "2026-01-06"}], [])


def test_cognitive_alpha_signal_refresh_builds_paper_bark_notification_from_latest_card(tmp_path):
    from src.schedule import BarkNotifier, BarkPushConfig
    from src.workflow import CognitiveAlphaSignalRefreshJob, CognitiveAlphaSignalRefreshSpec

    sent: list[tuple[str, dict, float]] = []
    job = CognitiveAlphaSignalRefreshJob(
        CognitiveAlphaSignalRefreshSpec(
            card_dir=POSITION_CARD,
            provider_uri=PROVIDER_URI,
            output_dir=tmp_path / "signals",
            paper_cash=10000.0,
        ),
        notifier=BarkNotifier(
            BarkPushConfig(server_url="https://bark.example", device_key="secret-device", dry_run=False),
            sender=lambda url, payload, timeout: sent.append((url, payload, timeout)) or {"status": 200},
        ),
    )
    latest_signal = json.loads(
        (POSITION_CARD / "latest_signal.json").read_text(
            encoding="utf-8"
        )
    )

    notifications = job.run(
        [
            {
                "source": "qlib.data.D.features",
                "max_datetime": latest_signal["as_of"],
            }
        ],
        [],
    )

    assert len(notifications) == 1
    notification = notifications[0]
    assert notification["source"] == "qlib_recorder.latest_signal"
    assert notification["symbol"] == "SOXL"
    assert notification["action"] == latest_signal["action"]
    assert notification["as_of"] == latest_signal["as_of"]
    assert notification["target_percent"] == pytest.approx(latest_signal["target_position"])
    assert notification["real_broker_order"] is False
    assert notification["push_result"]["status"] == "sent"
    assert sent[0][0] == "https://bark.example/push"
    assert sent[0][1]["title"] == f"SOXL paper/sim {latest_signal['action']}"
    assert "real_broker_order=False" in sent[0][1]["body"]
    written = json.loads((tmp_path / "signals" / "latest_signal_push.json").read_text(encoding="utf-8"))
    assert written["notification"]["recorder_id"] == notification["recorder_id"]
    assert written["data_gate"]["max_datetime"] == latest_signal["as_of"]


def test_cognitive_alpha_signal_refresh_updates_project_owned_paper_ledger(tmp_path):
    from src.workflow import CognitiveAlphaSignalRefreshJob, CognitiveAlphaSignalRefreshSpec

    card = tmp_path / "card"
    card.mkdir()
    (card / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "cogalpha-paper-ledger",
                "card_type": "research",
                "instrument": "SOXL",
                "recorder_id": "recorder-ledger",
                "provider_uri": PROVIDER_URI.as_posix(),
                "latest_signal": {
                    "action": "BUY",
                    "as_of": "2026-06-24",
                    "target_position": 0.25,
                    "previous_target_position": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "paper_ledger.json"
    job = CognitiveAlphaSignalRefreshJob(
        CognitiveAlphaSignalRefreshSpec(
            card_dir=card,
            provider_uri=PROVIDER_URI,
            output_dir=tmp_path / "signals",
            paper_cash=10_000.0,
            ledger_path=ledger_path,
            paper_account_id="cogalpha-goal",
        )
    )

    notifications = job.run([{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}], [])

    notification = notifications[0]
    written = json.loads((tmp_path / "signals" / "latest_signal_push.json").read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert notification["paper_simulation"]["execution_mode"] == "project_owned_paper_sim"
    assert notification["paper_simulation"]["account_id"] == "cogalpha-goal"
    assert notification["paper_simulation"]["ledger_path"] == ledger_path.as_posix()
    assert notification["real_broker_order"] is False
    assert notification["quantity"] == ledger["accounts"]["cogalpha-goal"]["positions"]["SOXL"]["quantity"]
    assert ledger["accounts"]["cogalpha-goal"]["positions"]["SOXL"]["target_percent"] == pytest.approx(0.25)
    assert ledger["fills"][0]["real_broker_order"] is False
    assert written["paper_simulation"] == notification["paper_simulation"]


def test_cognitive_alpha_signal_refresh_cli_updates_project_owned_paper_ledger(tmp_path):
    from src.workflow.cognitive_alpha_goal import main

    card = tmp_path / "card"
    card.mkdir()
    (card / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "cogalpha-paper-ledger-cli",
                "card_type": "research",
                "instrument": "SOXL",
                "recorder_id": "recorder-ledger-cli",
                "provider_uri": PROVIDER_URI.as_posix(),
                "latest_signal": {
                    "action": "BUY",
                    "as_of": "2026-06-24",
                    "target_position": 0.10,
                    "previous_target_position": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "signals"
    ledger_path = tmp_path / "paper_ledger.json"

    exit_code = main(
        [
            "--card-dir",
            card.as_posix(),
            "--provider-uri",
            PROVIDER_URI.as_posix(),
            "--output-dir",
            output_dir.as_posix(),
            "--paper-cash",
            "10000",
            "--ledger-path",
            ledger_path.as_posix(),
            "--paper-account-id",
            "cli-cogalpha",
            "--required-max-datetime",
            "2026-06-24",
        ]
    )

    written = json.loads((output_dir / "latest_signal_push.json").read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert written["paper_simulation"]["account_id"] == "cli-cogalpha"
    assert written["paper_simulation"]["ledger_path"] == ledger_path.as_posix()
    assert ledger["accounts"]["cli-cogalpha"]["positions"]["SOXL"]["target_percent"] == pytest.approx(0.10)


def test_cognitive_alpha_goal_codex_exec_attaches_direct_factor_base_to_model_frame():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        DIRECT_FACTOR_BASE_POSITION_COL,
        DIRECT_FACTOR_RAW_TARGET_COL,
        _attach_direct_factor_model_features,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "forward_return": [0.02, -0.01, 0.03],
        }
    )
    direct_result = SimpleNamespace(
        summary={"selected_candidate": {"candidate": "direct_winner"}},
        positions=pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                        "2026-01-02",
                        "2026-01-05",
                        "2026-01-06",
                    ]
                ),
                "candidate": ["direct_winner", "direct_winner", "direct_winner", "other", "other", "other"],
                "raw_target": [0.1, 0.8, 0.3, 1.0, 1.0, 1.0],
                "target_position": [0.0, 0.7, 0.7, 1.0, 1.0, 1.0],
            }
        ),
    )

    out, direct_feature_cols = _attach_direct_factor_model_features(frame, direct_result)

    assert direct_feature_cols == [DIRECT_FACTOR_BASE_POSITION_COL, DIRECT_FACTOR_RAW_TARGET_COL]
    assert out[DIRECT_FACTOR_BASE_POSITION_COL].tolist() == [0.0, 0.7, 0.7]
    assert out[DIRECT_FACTOR_RAW_TARGET_COL].tolist() == [0.1, 0.8, 0.3]


def test_cognitive_alpha_goal_codex_exec_uses_open_to_open_training_and_lgb_xgb():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        BASE_CONSTRAINED_REF_COL,
        COGALPHA_OPEN2OPEN_RISK_GATE_COL,
        LABEL_OPEN_TO_OPEN_5D_COL,
        OPEN_TO_OPEN_RETURN_COL,
        CognitiveAlphaGoalCodexExecCardSpec,
        _add_return_columns,
        _position_spec,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-02", periods=8, freq="B"),
            "open": [10.0, 11.0, 12.1, 10.89, 11.4345, 12.006225, 11.40591375, 12.546505125],
            "close": [10.5, 11.2, 12.0, 11.0, 11.6, 12.2, 11.4, 12.4],
        }
    )

    _add_return_columns(frame)

    assert frame.loc[0, OPEN_TO_OPEN_RETURN_COL] == pytest.approx(12.1 / 11.0 - 1.0)
    assert frame.loc[0, LABEL_OPEN_TO_OPEN_5D_COL] == pytest.approx(11.40591375 / 11.0 - 1.0)
    assert frame.loc[frame.index.max(), OPEN_TO_OPEN_RETURN_COL] == 0.0
    assert pd.isna(frame.loc[frame.index.max(), LABEL_OPEN_TO_OPEN_5D_COL])

    spec = CognitiveAlphaGoalCodexExecCardSpec(root=ROOT)
    position_spec = _position_spec(
        ["feature_a", BASE_CONSTRAINED_REF_COL, COGALPHA_OPEN2OPEN_RISK_GATE_COL],
        spec=spec,
        test=("2026-01-02", "2026-07-02"),
    )

    model_families = {item["model"] for item in position_spec.model_specs}
    assert {"lgbm_classifier", "lgbm_regressor", "xgb_classifier", "xgb_regressor"}.issubset(model_families)
    assert position_spec.return_col == OPEN_TO_OPEN_RETURN_COL
    assert position_spec.label_col == LABEL_OPEN_TO_OPEN_5D_COL
    assert position_spec.constrained_reference_name == "base_constrained_ref"
    assert position_spec.validation_max_daily_loss_floor is None
    assert position_spec.test_max_daily_loss_floor is None
    assert position_spec.walk_forward_folds == [
        {
            "fold": "wf_2023",
            "train": ("2019-01-02", "2021-12-31"),
            "validation": ("2022-01-03", "2022-12-30"),
            "test": ("2023-01-03", "2023-12-29"),
        },
        {
            "fold": "wf_2024",
            "train": ("2019-01-02", "2022-12-30"),
            "validation": ("2023-01-03", "2023-12-29"),
            "test": ("2024-01-02", "2024-12-31"),
        },
        {
            "fold": "wf_2025",
            "train": ("2019-01-02", "2023-12-29"),
            "validation": ("2024-01-02", "2024-12-31"),
            "test": ("2025-01-02", "2025-12-31"),
        },
        {
            "fold": "wf_2026",
            "train": ("2019-01-02", "2024-12-31"),
            "validation": ("2025-01-02", "2025-12-31"),
            "test": ("2026-01-02", "2026-07-02"),
        },
    ]
    assert any(
        item.get("position_mode") == "base_throttle" and item.get("base_position_col") == BASE_CONSTRAINED_REF_COL
        for item in position_spec.position_specs or []
    )
    assert all(
        item.get("selection_disabled") is not True
        for item in position_spec.position_specs or []
        if item.get("position_mode") == "base_throttle" and item.get("base_position_col") == BASE_CONSTRAINED_REF_COL
    )
    base_plus_specs = [
        item
        for item in position_spec.position_specs or []
        if item.get("position_mode") == "base_plus_probability"
        and item.get("base_position_col") == BASE_CONSTRAINED_REF_COL
    ]
    assert base_plus_specs
    assert all(item.get("selection_disabled") is not True for item in base_plus_specs)
    assert {item["boost_scale"] for item in base_plus_specs} >= {0.1, 0.2, 0.35, 0.5, 1.5}
    assert {item["risk_gate_max_position"] for item in base_plus_specs} >= {0.35, 0.75}
    assert all(item.get("risk_gate_col") == COGALPHA_OPEN2OPEN_RISK_GATE_COL for item in base_plus_specs)
    assert all(
        item.get("selection_disabled") is True
        for item in position_spec.position_specs or []
        if item.get("position_mode", "probability") == "probability"
    )


def test_base_plus_probability_position_mode_adds_model_boost_and_respects_risk_cap():
    from src.backtest.ml_position import _raw_target_from_probability

    data = pd.DataFrame(
        {
            "base_constrained_ref": [0.50, 0.20, 0.80],
            "risk_gate": [0.0, 0.0, 1.0],
        }
    )
    probability = pd.Series([0.55, 0.80, 0.90])

    raw = _raw_target_from_probability(
        data,
        probability,
        position_spec={
            "position_mode": "base_plus_probability",
            "base_position_col": "base_constrained_ref",
            "threshold": 0.50,
            "base_scale": 1.0,
            "boost_scale": 0.40,
            "max_position": 1.0,
            "risk_gate_col": "risk_gate",
            "risk_gate_threshold": 0.5,
            "risk_gate_direction": "above",
            "risk_gate_max_position": 0.35,
        },
    )

    assert raw.tolist() == pytest.approx([0.54, 0.44, 0.35])


def test_cognitive_alpha_goal_codex_exec_resume_returns_completed_card_without_rerun(tmp_path):
    from src.workflow.cognitive_alpha_goal_codex_exec import CognitiveAlphaGoalCodexExecCardSpec, _run_card

    card = tmp_path / "research" / "cognitive_alpha_goal_codex_exec_v1"
    card.mkdir(parents=True)
    (card / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "attempt_index": 12,
                "max_attempts": 12,
                "promotion_decision": "candidate_for_validated_research",
            }
        ),
        encoding="utf-8",
    )

    summary = _run_card(CognitiveAlphaGoalCodexExecCardSpec(root=tmp_path, resume=True, max_attempts=12))

    assert summary["status"] == "complete"
    assert summary["attempt_index"] == 12


def test_cognitive_alpha_goal_codex_exec_dynamic_open_to_open_benchmarks_use_current_window():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        BASE_CONSTRAINED_REF_COL,
        OPEN_TO_OPEN_RETURN_COL,
        _dynamic_open_to_open_benchmarks,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            OPEN_TO_OPEN_RETURN_COL: [0.10, -0.05, 0.20],
            BASE_CONSTRAINED_REF_COL: [1.0, 0.5, 0.0],
        }
    )

    benchmarks = _dynamic_open_to_open_benchmarks(frame, test=("2026-01-02", "2026-01-06"))
    rows = {item["name"]: item for item in benchmarks}

    assert rows["soxl_buyhold"]["return_pct"] == pytest.approx(((1.10 * 0.95 * 1.20) - 1.0) * 100.0)
    expected_base_returns = pd.Series([0.10 - 0.0003, -0.025 - 0.00015, -0.00015])
    assert rows["base_constrained_ref"]["return_pct"] == pytest.approx(
        ((1.0 + expected_base_returns).prod() - 1.0) * 100.0
    )
    assert rows["constrained_weight_reference"]["return_pct"] == pytest.approx(
        rows["base_constrained_ref"]["return_pct"]
    )


def test_cognitive_alpha_goal_codex_exec_normalizes_clipped_negative_direct_factor_raw_target():
    from src.workflow.cognitive_alpha_goal_codex_exec import (
        DIRECT_FACTOR_BASE_POSITION_COL,
        DIRECT_FACTOR_RAW_TARGET_COL,
        _attach_direct_factor_model_features,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "open_to_open_return": [0.02, -0.01, 0.0],
        }
    )
    direct_result = SimpleNamespace(
        summary={"selected_candidate": {"candidate": "negative_direct_winner"}},
        positions=pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
                "candidate": ["negative_direct_winner"] * 3,
                "raw_target": [-1.0, -0.5, 0.0],
                "target_position": [0.0, 0.0, 0.0],
            }
        ),
    )

    out, _ = _attach_direct_factor_model_features(frame, direct_result)

    assert out[DIRECT_FACTOR_RAW_TARGET_COL].tolist() == [-1.0, -0.5, 0.0]
    assert out[DIRECT_FACTOR_BASE_POSITION_COL].tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_cognitive_alpha_goal_runner_writes_resumable_full_chain_manifest(tmp_path):
    from src.workflow import CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    calls: list[str] = []

    def data_gate():
        calls.append("data_gate")
        return [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24", "provider_uri": "qlib://soxl"}]

    def evolution(manifests):
        calls.append("evolution")
        assert manifests[0]["source"] == "qlib.data.D.features"
        return {
            "candidate_provider": {"provider_id": "deterministic_code_agent_pool", "profile_agent_count": 21},
            "initial_candidate_count": 80,
            "completed_generations": 24,
            "checkpoints": {"candidate_pool": "candidate_pool/generation_*.jsonl"},
        }

    def position(manifests, evolution_result):
        calls.append("position")
        assert evolution_result["completed_generations"] == 24
        return {
            "selected_candidate": {"candidate": "lgbm_classifier_probability_p0.45_s1_h0.2_hold3"},
            "comparison": {"selected_return_pct": 90.18, "beats_constrained_weight_reference": False},
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.1133},
        }

    def research_card(manifests, evolution_result, position_result):
        calls.append("research_card")
        assert position_result["latest_signal"]["as_of"] == manifests[0]["max_datetime"]
        return {
            "card_dir": "research/cognitive_alpha_position_model_v1",
            "recorder_id": "recorder-position",
            "promotion_decision": "not_promoted_under_constrained_weight_reference",
        }

    def signal_push(manifests, model_results):
        calls.append("signal_push")
        assert model_results[0]["latest_signal"]["action"] == "HOLD"
        return [{"source": "qlib_recorder.latest_signal", "action": "HOLD", "as_of": "2026-06-24"}]

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-smoke",
            objective="CogAlpha goal runner smoke",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
        ),
        data_gate=data_gate,
        evolution=evolution,
        position=position,
        research_card=research_card,
        signal_push=signal_push,
    )

    manifest = runner.run()

    assert calls == ["data_gate", "evolution", "position", "research_card", "signal_push"]
    assert manifest["status"] == "needs_iteration"
    assert manifest["terminal_reason"] == "max_attempts_exhausted_needs_iteration"
    assert manifest["next_action"]["action"] == "continue_codex_exec_factor_generation"
    assert manifest["next_action"]["requires_budget_refresh"] is True
    assert manifest["stage_order"] == [
        "data_gate",
        "candidate_generation",
        "quality_checker",
        "fitness_evaluation",
        "evolution_loop",
        "model_and_position",
        "direct_factor_backtest",
        "research_card",
        "signal_push",
    ]
    assert manifest["data_gate"]["max_datetime"] == "2026-06-24"
    assert manifest["candidate_generation"]["candidate_provider"]["profile_agent_count"] == 21
    assert manifest["evolution_loop"]["completed_generations"] == 24
    assert manifest["model_and_position"]["latest_signal"]["target_position"] == pytest.approx(0.1133)
    assert manifest["direct_factor_backtest"]["status"] == "skipped"
    assert manifest["signal_push"]["notifications"][0]["action"] == "HOLD"
    written = json.loads((tmp_path / "goal_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert written == manifest
    assert state["status"] == "needs_iteration"
    assert state["terminal_reason"] == "max_attempts_exhausted_needs_iteration"
    assert state["next_action"] == manifest["next_action"]
    assert state["completed_stages"] == manifest["stage_order"]


def test_cognitive_alpha_goal_runner_strict_gates_fail_closed_for_incomplete_evolution(tmp_path):
    from src.workflow import CognitiveAlphaGoalGateSpec, CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-strict-fail",
            objective="strict CogAlpha full chain",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            gates=CognitiveAlphaGoalGateSpec(
                min_initial_candidate_count=80,
                min_completed_generations=24,
                min_selected_count=1,
                min_repair_attempt_count=1,
                required_candidate_provider_ids=("deterministic_code_agent_pool",),
                required_candidate_provider_kinds=("repo_owned_code",),
                require_candidate_provider_no_openai_api_key=True,
                required_metric_names=("ic", "icir", "rank_ic", "rank_icir", "mutual_information"),
                required_evolution_ops=("mutation", "crossover", "crossover_mutation"),
                required_position_model_families=("lgbm_classifier",),
                required_benchmark_names=(
                    "soxl_backtest",
                    "constrained_weight_reference",
                    "soxl_buyhold",
                    "previous_factor_enhanced",
                ),
            ),
        ),
        data_gate=lambda: [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}],
        evolution=lambda manifests: {
            "candidate_provider": {
                "provider_id": "deterministic_code_agent_pool",
                "provider_kind": "repo_owned_code",
                "profile_agent_count": 21,
                "codex_direct_compatible": True,
                "codex_execution_mode": "repo_owned_codex_generated_code",
                "openai_api_key_required": False,
            },
            "initial_candidate_count": 80,
            "completed_generations": 24,
            "selected_count": 2,
            "repair_attempt_count": 1,
            "evolution_ops": ["mutation", "crossover", "crossover_mutation"],
        },
        position=lambda manifests, evolution_result: {
            "workflow": "cognitive_alpha_position_model",
            "model_families": ["lgbm_classifier"],
            "comparison": {
                "beats_soxl_backtest": True,
                "beats_constrained_weight_reference": False,
                "beats_soxl_buyhold": False,
                "beats_previous_factor_enhanced": True,
            },
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.0},
        },
        research_card=lambda manifests, evolution_result, position_result: {"card_dir": "research/card"},
        signal_push=lambda manifests, model_results: [],
    )

    with pytest.raises(ValueError, match="missing required fitness metrics"):
        runner.run()


def test_cognitive_alpha_goal_runner_strict_gates_require_walk_forward(tmp_path):
    from src.workflow import CognitiveAlphaGoalGateSpec, CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-strict-walk-forward-fail",
            objective="strict CogAlpha full chain requires rolling validation",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            gates=CognitiveAlphaGoalGateSpec(min_walk_forward_fold_count=2),
        ),
        data_gate=lambda: [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}],
        evolution=lambda manifests: {
            "candidate_provider": {"provider_id": "deterministic_code_agent_pool"},
            "initial_candidate_count": 80,
            "completed_generations": 24,
        },
        position=lambda manifests, evolution_result: {
            "workflow": "cognitive_alpha_position_model",
            "comparison": {"beats_constrained_weight_reference": False},
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.25},
        },
        research_card=lambda manifests, evolution_result, position_result: {"card_dir": "research/card"},
        signal_push=lambda manifests, model_results: [],
    )

    with pytest.raises(ValueError, match="walk-forward fold count"):
        runner.run()


def test_cognitive_alpha_goal_runner_strict_gates_record_full_chain_validation(tmp_path):
    from src.workflow import CognitiveAlphaGoalGateSpec, CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-strict-pass",
            objective="strict CogAlpha full chain",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            gates=CognitiveAlphaGoalGateSpec(
                min_initial_candidate_count=80,
                min_completed_generations=24,
                min_selected_count=1,
                min_repair_attempt_count=1,
                required_candidate_provider_ids=("deterministic_code_agent_pool",),
                required_candidate_provider_kinds=("repo_owned_code",),
                require_candidate_provider_no_openai_api_key=True,
                required_metric_names=("ic", "icir", "rank_ic", "rank_icir", "mutual_information"),
                required_evolution_ops=("mutation", "crossover", "crossover_mutation"),
                required_position_model_families=("lgbm_classifier",),
                required_benchmark_names=(
                    "soxl_backtest",
                    "constrained_weight_reference",
                    "soxl_buyhold",
                    "previous_factor_enhanced",
                ),
            ),
        ),
        data_gate=lambda: [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}],
        evolution=lambda manifests: {
            "candidate_provider": {
                "provider_id": "deterministic_code_agent_pool",
                "provider_kind": "repo_owned_code",
                "profile_agent_count": 21,
                "codex_direct_compatible": True,
                "codex_execution_mode": "repo_owned_codex_generated_code",
                "openai_api_key_required": False,
            },
            "initial_candidate_count": 80,
            "completed_generations": 24,
            "selected_count": 8,
            "repair_attempt_count": 1,
            "metric_names": ["ic", "icir", "rank_ic", "rank_icir", "mutual_information"],
            "evolution_ops": ["mutation", "crossover", "crossover_mutation"],
            "checkpoints": {
                "candidate_pool": "candidate_pool/generation_*.jsonl",
                "quality": "quality/generation_*.csv",
                "fitness": "fitness/generation_*.csv",
                "rejected": "rejected/generation_*.jsonl",
                "lineage": "lineage/generation_*.jsonl",
                "state": "state.json",
            },
        },
        position=lambda manifests, evolution_result: {
            "workflow": "cognitive_alpha_position_model",
            "model_families": ["lgbm_classifier"],
            "selected_candidate": {"candidate": "lgbm_classifier_binary_p0.55", "test": {"max_position": 1.0}},
            "comparison": {
                "beats_soxl_backtest": True,
                "beats_constrained_weight_reference": False,
                "beats_soxl_buyhold": False,
                "beats_previous_factor_enhanced": True,
            },
            "latest_signal": {
                "as_of": "2026-06-24",
                "action": "HOLD",
                "target_position": 0.25,
                "previous_target_position": 0.50,
            },
        },
        research_card=lambda manifests, evolution_result, position_result: {
            "card_dir": "research/cognitive_alpha_position_model_v1",
            "recorder_id": "recorder-position",
            "promotion_decision": "not_promoted_under_constrained_weight_reference",
        },
        signal_push=lambda manifests, model_results: [
            {"source": "qlib_recorder.latest_signal", "action": "HOLD", "as_of": "2026-06-24"}
        ],
    )

    manifest = runner.run()

    assert manifest["validation_gates"]["min_initial_candidate_count"] == 80
    assert manifest["validation_gates"]["required_metric_names"] == [
        "ic",
        "icir",
        "rank_ic",
        "rank_icir",
        "mutual_information",
    ]
    assert manifest["validation_gates"]["require_candidate_provider_no_openai_api_key"] is True
    assert manifest["candidate_generation"]["candidate_provider"]["openai_api_key_required"] is False
    assert manifest["candidate_generation"]["candidate_provider"]["codex_direct_compatible"] is True
    assert manifest["fitness_evaluation"]["metric_names"] == [
        "ic",
        "icir",
        "rank_ic",
        "rank_icir",
        "mutual_information",
    ]
    assert manifest["model_and_position"]["latest_signal"]["target_position"] == pytest.approx(0.25)
    written = json.loads((tmp_path / "goal_manifest.json").read_text(encoding="utf-8"))
    assert written["validation_gates"] == manifest["validation_gates"]


def test_cognitive_alpha_goal_runner_records_direct_factor_backtest_goal_gate(tmp_path):
    from src.workflow import CognitiveAlphaGoalGateSpec, CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    calls: list[str] = []

    def data_gate():
        calls.append("data_gate")
        return [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}]

    def evolution(manifests):
        calls.append("evolution")
        return {
            "candidate_provider": {"provider_id": "deterministic_code_agent_pool"},
            "initial_candidate_count": 80,
            "completed_generations": 24,
        }

    def position(manifests, evolution_result):
        calls.append("position")
        return {
            "workflow": "cognitive_alpha_position_model",
            "comparison": {"beats_constrained_weight_reference": False},
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.25},
        }

    def direct_factor_backtest(manifests, evolution_result, position_result):
        calls.append("direct_factor_backtest")
        assert position_result["latest_signal"]["as_of"] == manifests[0]["max_datetime"]
        return {
            "workflow": "cognitive_alpha_weighted_factor_search",
            "selected_candidate": {
                "candidate": "canary_only_budget_h01_direct",
                "return_pct": 573.2281789190391,
                "max_drawdown_pct": -14.619712031891652,
                "max_position": 1.0,
                "min_position": 0.0,
                "days_above_100": 0,
            },
            "comparison": {
                "selected_return_pct": 573.2281789190391,
                "beats_constrained_weight_reference": True,
                "selected_vs_constrained_weight_reference_return_gap_pct": 2.0081211493893534,
                "beats_soxl_backtest": True,
            },
            "promotion_decision": "candidate_for_validated_research",
        }

    def research_card(manifests, evolution_result, position_result):
        calls.append("research_card")
        return {
            "card_dir": "research/cognitive_alpha_position_model_v1",
            "promotion_decision": "not_promoted_under_constrained_weight_reference",
        }

    def signal_push(manifests, model_results):
        calls.append("signal_push")
        assert model_results[0]["workflow"] == "cognitive_alpha_position_model"
        assert model_results[1]["workflow"] == "cognitive_alpha_weighted_factor_search"
        return [{"source": "qlib_recorder.latest_signal", "action": "HOLD", "as_of": "2026-06-24"}]

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-direct-factor-pass",
            objective="mine factors and beat constrained reference in direct backtest",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            gates=CognitiveAlphaGoalGateSpec(
                required_direct_factor_benchmark_names=("constrained_weight_reference", "soxl_backtest"),
                require_direct_factor_beats_constrained_weight_reference=True,
            ),
        ),
        data_gate=data_gate,
        evolution=evolution,
        position=position,
        direct_factor_backtest=direct_factor_backtest,
        research_card=research_card,
        signal_push=signal_push,
    )

    manifest = runner.run()

    assert calls == ["data_gate", "evolution", "position", "direct_factor_backtest", "research_card", "signal_push"]
    assert manifest["stage_order"] == [
        "data_gate",
        "candidate_generation",
        "quality_checker",
        "fitness_evaluation",
        "evolution_loop",
        "model_and_position",
        "direct_factor_backtest",
        "research_card",
        "signal_push",
    ]
    assert manifest["validation_gates"]["require_direct_factor_beats_constrained_weight_reference"] is True
    assert manifest["direct_factor_backtest"]["comparison"]["beats_constrained_weight_reference"] is True
    assert manifest["direct_factor_backtest"]["selected_candidate"]["max_position"] <= 1.0
    assert manifest["research_card"]["promotion_decision"] == "not_promoted_under_constrained_weight_reference"
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["completed_stages"] == manifest["stage_order"]


def test_cognitive_alpha_goal_runner_keeps_not_promoted_attempt_active_until_max_attempts(tmp_path):
    from src.workflow import CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-iterative-not-promoted",
            objective="continue CogAlpha generation until promotion or attempt budget",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            attempt_index=1,
            max_attempts=3,
        ),
        data_gate=lambda: [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}],
        evolution=lambda manifests: {
            "candidate_provider": {
                "provider_id": "codex_exec_candidate_provider",
                "provider_kind": "codex_exec_code",
                "openai_api_key_required": False,
            },
            "initial_candidate_count": 4,
            "completed_generations": 2,
            "selected_count": 1,
            "metric_names": ["ic", "icir", "rank_ic", "rank_icir", "mutual_information"],
            "evolution_ops": ["mutation", "crossover", "crossover_mutation"],
        },
        position=lambda manifests, evolution_result: {
            "workflow": "cognitive_alpha_position_model",
            "promotion_decision": "not_promoted_failed_walk_forward_risk_gates",
            "comparison": {
                "selected_return_pct": 7.6,
                "beats_constrained_weight_reference": False,
                "selected_vs_constrained_weight_reference_return_gap_pct": -563.5,
            },
            "walk_forward": {"fold_count": 2, "all_folds_pass_risk_gates": False},
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.0},
        },
        direct_factor_backtest=lambda manifests, evolution_result, position_result: {
            "workflow": "cognitive_alpha_weighted_factor_search",
            "promotion_decision": "diagnostic_direct_factor_candidate",
            "selected_candidate": {"max_position": 1.0, "min_position": 0.0, "days_above_100": 0},
            "comparison": {"beats_constrained_weight_reference": True},
        },
        research_card=lambda manifests, evolution_result, position_result: {
            "card_dir": "research/cognitive_alpha_goal_codex_exec_v1",
            "recorder_id": "recorder-attempt-001",
            "promotion_decision": "not_promoted_failed_walk_forward_risk_gates",
            "summary_path": "summary.json",
            "failure_modes_path": "failure_modes.md",
            "goal_manifest_path": "goal_manifest.json",
        },
        signal_push=lambda manifests, model_results: [
            {"source": "qlib_recorder.latest_signal", "action": "HOLD", "as_of": "2026-06-24"}
        ],
    )

    manifest = runner.run()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "needs_iteration"
    assert manifest["terminal_reason"] == "not_promoted_attempt_has_remaining_budget"
    assert manifest["attempt_index"] == 1
    assert manifest["max_attempts"] == 3
    assert manifest["remaining_attempts"] == 2
    assert manifest["next_action"]["action"] == "continue_codex_exec_factor_generation"
    assert manifest["next_action"]["read_artifacts"] == ["failure_modes.md", "goal_manifest.json", "summary.json"]
    assert manifest["next_action"]["candidate_provider"]["provider_id"] == "codex_exec_candidate_provider"
    assert "walk_forward_risk_gates" in manifest["next_action"]["feedback_focus"]
    assert state["status"] == "needs_iteration"
    assert state["current_attempt"] == 1
    assert state["next_action"] == manifest["next_action"]


def test_cognitive_alpha_goal_runner_keeps_not_promoted_active_when_attempt_budget_exhausted(tmp_path):
    from src.workflow import CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-iterative-exhausted",
            objective="keep CogAlpha goal active after local attempt budget",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            attempt_index=3,
            max_attempts=3,
        ),
        data_gate=lambda: [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}],
        evolution=lambda manifests: {
            "candidate_provider": {"provider_id": "codex_exec_candidate_provider", "provider_kind": "codex_exec_code"},
            "initial_candidate_count": 4,
            "completed_generations": 2,
        },
        position=lambda manifests, evolution_result: {
            "workflow": "cognitive_alpha_position_model",
            "promotion_decision": "not_promoted_failed_walk_forward_risk_gates",
            "comparison": {"beats_constrained_weight_reference": False},
            "walk_forward": {"fold_count": 2, "all_folds_pass_risk_gates": False},
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.0},
        },
        research_card=lambda manifests, evolution_result, position_result: {
            "card_dir": "research/cognitive_alpha_goal_codex_exec_v1",
            "promotion_decision": "not_promoted_failed_walk_forward_risk_gates",
        },
        signal_push=lambda manifests, model_results: [],
    )

    manifest = runner.run()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "needs_iteration"
    assert manifest["terminal_reason"] == "max_attempts_exhausted_needs_iteration"
    assert manifest["attempt_index"] == 3
    assert manifest["remaining_attempts"] == 0
    assert manifest["next_action"]["action"] == "continue_codex_exec_factor_generation"
    assert manifest["next_action"]["next_attempt"] == 4
    assert manifest["next_action"]["max_attempts"] == 3
    assert manifest["next_action"]["requires_budget_refresh"] is True
    assert manifest["next_action"]["required_max_attempts"] == 4
    assert "walk_forward_risk_gates" in manifest["next_action"]["feedback_focus"]
    assert state["status"] == "needs_iteration"
    assert state["terminal_reason"] == "max_attempts_exhausted_needs_iteration"
    assert state["next_action"] == manifest["next_action"]


def test_cognitive_alpha_goal_runner_rejects_direct_factor_promotion(tmp_path):
    from src.workflow import CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-direct-factor-promotion-rejected",
            objective="direct factor cannot become final promotion",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
        ),
        data_gate=lambda: [{"source": "qlib.data.D.features", "max_datetime": "2026-06-24"}],
        evolution=lambda manifests: {
            "candidate_provider": {"provider_id": "deterministic_code_agent_pool"},
            "initial_candidate_count": 80,
            "completed_generations": 24,
        },
        position=lambda manifests, evolution_result: {
            "workflow": "cognitive_alpha_position_model",
            "comparison": {"beats_constrained_weight_reference": False},
            "latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.25},
        },
        direct_factor_backtest=lambda manifests, evolution_result, position_result: {
            "workflow": "cognitive_alpha_weighted_factor_search",
            "selected_candidate": {
                "candidate": "canary_tsmom_direct",
                "max_position": 1.0,
                "min_position": 0.0,
                "days_above_100": 0,
            },
            "comparison": {"beats_constrained_weight_reference": True},
            "promotion_decision": "diagnostic_direct_factor_candidate",
        },
        research_card=lambda manifests, evolution_result, position_result: {
            "card_dir": "research/cognitive_alpha_position_model_v1",
            "promotion_decision": "candidate_for_validated_research_direct_factor",
        },
        signal_push=lambda manifests, model_results: [],
    )

    with pytest.raises(ValueError, match="direct factor backtest cannot be the final promotion gate"):
        runner.run()


def test_cognitive_alpha_goal_runner_resume_skips_completed_data_gate(tmp_path):
    from src.workflow import CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "running",
                "completed_stages": ["data_gate"],
                "stage_outputs": {
                    "data_gate": {
                        "manifests": [
                            {
                                "source": "qlib.data.D.features",
                                "max_datetime": "2026-06-24",
                                "provider_uri": "qlib://soxl",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def data_gate():
        raise AssertionError("resume should not re-run completed data gate")

    def evolution(manifests):
        calls.append("evolution")
        return {"candidate_provider": {"provider_id": "provider"}, "initial_candidate_count": 9, "completed_generations": 2}

    def position(manifests, evolution_result):
        calls.append("position")
        return {"latest_signal": {"as_of": "2026-06-24", "action": "HOLD", "target_position": 0.0}}

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-resume",
            objective="resume smoke",
            state_path=state_path,
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
            resume=True,
        ),
        data_gate=data_gate,
        evolution=evolution,
        position=position,
        research_card=lambda manifests, evolution_result, position_result: {"card_dir": "research/card"},
        signal_push=lambda manifests, model_results: [],
    )

    manifest = runner.run()

    assert calls == ["evolution", "position"]
    assert manifest["status"] == "complete"
    assert manifest["data_gate"]["max_datetime"] == "2026-06-24"
    assert manifest["evolution_loop"]["completed_generations"] == 2


def test_cognitive_alpha_goal_runner_fails_closed_for_non_qlib_or_stale_data(tmp_path):
    from src.workflow import CognitiveAlphaGoalRunner, CognitiveAlphaGoalRunnerSpec

    runner = CognitiveAlphaGoalRunner(
        CognitiveAlphaGoalRunnerSpec(
            run_id="goal-fail",
            objective="fail closed smoke",
            state_path=tmp_path / "state.json",
            manifest_path=tmp_path / "goal_manifest.json",
            required_max_datetime="2026-06-24",
        ),
        data_gate=lambda: [{"source": "csv", "max_datetime": "2026-06-23"}],
        evolution=lambda manifests: {},
        position=lambda manifests, evolution_result: {},
        research_card=lambda manifests, evolution_result, position_result: {},
        signal_push=lambda manifests, model_results: [],
    )

    with pytest.raises(ValueError, match="requires qlib-derived data manifests"):
        runner.run()
