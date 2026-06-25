from __future__ import annotations

import json
from pathlib import Path

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
    manifest = json.loads((PROVIDER_URI / "qlib_store_manifest.json").read_text(encoding="utf-8"))

    notifications = job.run([{"source": "qlib.data.D.features", "max_datetime": manifest["calendar_end"]}], [])

    assert len(notifications) == 1
    notification = notifications[0]
    latest_signal = json.loads((POSITION_CARD / "latest_signal.json").read_text(encoding="utf-8"))
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
    assert written["data_gate"]["max_datetime"] == manifest["calendar_end"]


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
    assert manifest["status"] == "complete"
    assert manifest["stage_order"] == [
        "data_gate",
        "candidate_generation",
        "quality_checker",
        "fitness_evaluation",
        "evolution_loop",
        "model_and_position",
        "research_card",
        "signal_push",
    ]
    assert manifest["data_gate"]["max_datetime"] == "2026-06-24"
    assert manifest["candidate_generation"]["candidate_provider"]["profile_agent_count"] == 21
    assert manifest["evolution_loop"]["completed_generations"] == 24
    assert manifest["model_and_position"]["latest_signal"]["target_position"] == pytest.approx(0.1133)
    assert manifest["signal_push"]["notifications"][0]["action"] == "HOLD"
    written = json.loads((tmp_path / "goal_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert written == manifest
    assert state["status"] == "complete"
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
