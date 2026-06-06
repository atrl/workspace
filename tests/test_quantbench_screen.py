from __future__ import annotations

from src.soxl_quant.app.quantbench_screen import (
    QuantBenchGate,
    build_quantbench_screen_report,
    quantbench_model_objective_coverage,
    load_summary_records,
    paper_requirement_matrix,
    screen_quantbench_candidates,
    summary_record_from_payload,
)


def _payload(
    profile: str,
    *,
    proxy_return: float,
    proxy_max_dd: float,
    proxy_bnh: float,
    stress_relative: float,
    stress_max_dd: float,
    stress_avg_pos: float,
    bull_return: float,
    bull_bnh: float,
) -> dict:
    return {
        "profile": profile,
        "test_proxy": {
            "return": proxy_return,
            "max_dd": proxy_max_dd,
            "buy_and_hold_return": proxy_bnh,
            "relative_return": proxy_return - proxy_bnh,
        },
        "slices": {
            "test_2022": {
                "return": -12.0,
                "relative_return": stress_relative,
                "max_dd": stress_max_dd,
                "avg_pos": stress_avg_pos,
                "buy_and_hold_return": -76.0,
            },
            "test_2023_2024": {
                "return": bull_return,
                "max_dd": 11.0,
                "avg_pos": 0.42,
                "buy_and_hold_return": bull_bnh,
                "relative_return": bull_return - bull_bnh,
            },
        },
    }


def test_paper_requirement_matrix_covers_quantbench_layers():
    matrix = paper_requirement_matrix()
    keys = {row["key"] for row in matrix}

    assert {
        "data_preparation",
        "factor_mining",
        "modeling_tree",
        "modeling_sequence",
        "modeling_relational",
        "training_objectives",
        "rolling_continual_learning",
        "ensemble_robustness",
        "evaluation_metrics",
    }.issubset(keys)
    assert all(row["paper_basis"] and row["soxl_mapping"] and row["next_artifact"] for row in matrix)


def test_summary_record_from_payload_normalizes_runtime_summary_schema():
    record = summary_record_from_payload(
        _payload(
            "candidate_a",
            proxy_return=60.0,
            proxy_max_dd=18.0,
            proxy_bnh=100.0,
            stress_relative=48.0,
            stress_max_dd=17.0,
            stress_avg_pos=0.24,
            bull_return=72.0,
            bull_bnh=120.0,
        ),
        source_path="runtime/model/card/candidate_a/summary.json",
    )

    assert record.profile == "candidate_a"
    assert record.source_path == "runtime/model/card/candidate_a/summary.json"
    assert record.test_proxy["max_dd"] == 18.0
    assert record.slices["test_2022"]["avg_pos"] == 0.24


def test_screen_quantbench_candidates_requires_drawdown_stress_retention_and_bull_capture():
    gate = QuantBenchGate(
        max_drawdown_pct=20.0,
        min_stress_relative_return=30.0,
        min_stress_avg_pos=0.10,
        max_stress_avg_pos=0.45,
        min_bull_capture=0.50,
        min_proxy_return=0.0,
    )
    records = [
        summary_record_from_payload(
            _payload(
                "eligible",
                proxy_return=80.0,
                proxy_max_dd=18.0,
                proxy_bnh=100.0,
                stress_relative=45.0,
                stress_max_dd=16.0,
                stress_avg_pos=0.22,
                bull_return=90.0,
                bull_bnh=120.0,
            ),
            source_path="eligible.json",
        ),
        summary_record_from_payload(
            _payload(
                "sells_all_chips",
                proxy_return=82.0,
                proxy_max_dd=17.0,
                proxy_bnh=100.0,
                stress_relative=50.0,
                stress_max_dd=13.0,
                stress_avg_pos=0.03,
                bull_return=91.0,
                bull_bnh=120.0,
            ),
            source_path="sells_all_chips.json",
        ),
        summary_record_from_payload(
            _payload(
                "too_much_drawdown",
                proxy_return=140.0,
                proxy_max_dd=24.0,
                proxy_bnh=100.0,
                stress_relative=55.0,
                stress_max_dd=21.0,
                stress_avg_pos=0.26,
                bull_return=110.0,
                bull_bnh=120.0,
            ),
            source_path="too_much_drawdown.json",
        ),
    ]

    screened = screen_quantbench_candidates(records, gate=gate)
    by_profile = {row["profile"]: row for row in screened}

    assert by_profile["eligible"]["eligible"] is True
    assert by_profile["eligible"]["bull_capture"] == 0.75
    assert by_profile["sells_all_chips"]["eligible"] is False
    assert "stress_avg_pos" in by_profile["sells_all_chips"]["failed_checks"]
    assert by_profile["too_much_drawdown"]["eligible"] is False
    assert "max_drawdown" in by_profile["too_much_drawdown"]["failed_checks"]
    assert screened[0]["profile"] == "eligible"


def test_load_summary_records_can_emit_repo_relative_paths(tmp_path):
    root = tmp_path
    summary_path = root / "runtime/model/card/candidate_a/soxl_candidate_a_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json_dumps(
            _payload(
                "candidate_a",
                proxy_return=80.0,
                proxy_max_dd=18.0,
                proxy_bnh=100.0,
                stress_relative=45.0,
                stress_max_dd=16.0,
                stress_avg_pos=0.22,
                bull_return=90.0,
                bull_bnh=120.0,
            )
        ),
        encoding="utf-8",
    )

    records = load_summary_records([summary_path], root=root)

    assert records[0].source_path == "runtime/model/card/candidate_a/soxl_candidate_a_summary.json"


def test_quantbench_model_objective_coverage_is_src_owned():
    coverage = quantbench_model_objective_coverage()
    by_family = {row["family"]: row for row in coverage}

    assert by_family["tree_xgboost_lightgbm"]["coverage"] == "implemented_partial"
    assert by_family["objectives_clf_ic_mse_ranking"]["coverage"] == "implemented"
    assert by_family["utility_rl_execution"]["coverage"] == "implemented_partial"
    assert by_family["rolling_continual"]["coverage"] == "implemented_partial"
    assert "rolling supervised refit" in by_family["rolling_continual"]["current_soxl_surface"]
    assert "TorchTCN" in by_family["attention_tide_mixer"]["current_soxl_surface"]
    assert by_family["seed_ensemble"]["coverage"] == "implemented_partial"
    assert "low-correlation signal blend" in by_family["seed_ensemble"]["current_soxl_surface"]
    assert all(row["next_step"] for row in coverage)


def test_build_quantbench_screen_report_writes_requirement_coverage_and_screen_artifacts(tmp_path):
    root = tmp_path
    card = root / "runtime/model_selection/card/quantbench_paper_soxl_v1"
    summary_path = root / "runtime/model/card/eligible/soxl_eligible_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json_dumps(
            _payload(
                "eligible",
                proxy_return=80.0,
                proxy_max_dd=18.0,
                proxy_bnh=100.0,
                stress_relative=45.0,
                stress_max_dd=16.0,
                stress_avg_pos=0.22,
                bull_return=90.0,
                bull_bnh=120.0,
            )
        ),
        encoding="utf-8",
    )

    report = build_quantbench_screen_report(root=root, card_dir=card)

    assert report["decision"] == "eligible_candidates_found"
    assert report["screen"]["eligible_profiles"] == ["eligible"]
    assert (card / "paper_requirement_matrix.csv").exists()
    assert (card / "paper_model_objective_coverage.csv").exists()
    assert (card / "paper_objective_families.csv").exists()
    assert (card / "candidate_screen.csv").exists()
    assert (card / "summary.json").exists()
    assert report["screen"]["best_rows"][0]["source_path"] == "runtime/model/card/eligible/soxl_eligible_summary.json"
    assert report["artifacts"]["paper_objective_families_csv"] == (
        "runtime/model_selection/card/quantbench_paper_soxl_v1/paper_objective_families.csv"
    )


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
