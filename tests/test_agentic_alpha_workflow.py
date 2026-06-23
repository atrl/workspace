from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class AgenticAlphaDataset:
    def __init__(self) -> None:
        dates = pd.date_range("2026-01-02", periods=8, freq="D")
        instruments = ["A", "B", "C", "D"]
        rows = []
        for day_index, date in enumerate(dates):
            for inst_index, instrument in enumerate(instruments):
                alpha = inst_index + 1 + (day_index % 3) * 0.2
                # Make daily rank IC strong but not constant so ICIR is meaningful.
                label_1d = alpha * (1.0 + (0.05 if day_index % 2 else -0.03))
                if day_index % 4 == 0 and instrument == "D":
                    label_1d = 0.5
                label_5d = label_1d * 0.9 + inst_index * 0.1
                noise = [0.2, -0.4, 0.1, -0.1][(inst_index + day_index) % 4]
                rows.append(
                    {
                        "instrument": instrument,
                        "datetime": date,
                        "alpha_good": alpha,
                        "alpha_duplicate": alpha * 2.0,
                        "alpha_noise": noise,
                        "LABEL1": label_1d,
                        "LABEL5": label_5d,
                    }
                )
        frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()
        self.frames = {"test": frame}

    def prepare(self, segment: str) -> pd.DataFrame:
        return self.frames[segment].copy()


def test_agentic_alpha_workflow_manifest_keeps_external_workflow_inside_src_boundary(tmp_path):
    from src.workflow import AgenticAlphaWorkflowSpec, build_agentic_alpha_workflow_manifest

    spec = AgenticAlphaWorkflowSpec(
        name="agentic_alpha_sfa_v1",
        route="sfa",
        market="all",
        provider_uri=Path("data/qlib/soxl"),
        recorder_uri=Path("research/agentic_alpha_sfa_v1/mlruns"),
        source_url="https://github.com/JacobDu/agentic-alpha",
    )

    manifest = build_agentic_alpha_workflow_manifest(spec)

    assert manifest["workflow"] == "agentic_alpha_sfa_v1"
    assert manifest["source_url"] == "https://github.com/JacobDu/agentic-alpha"
    assert manifest["stage_order"] == ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
    assert manifest["decision_labels"] == ["Promote", "Iterate", "Drop"]
    assert manifest["src_boundary"]["skill_scripts_imported"] is False
    assert manifest["src_boundary"]["business_logic_surface"] == "src"
    assert manifest["qlib_native_required"] is True
    assert manifest["artifact_policy"]["policy"] == "slim_card_recorder_first"

    output = tmp_path / "workflow_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["route"] == "sfa"


def test_agentic_alpha_rgevd_search_distills_external_knowledge_and_risk_gate():
    from src.workflow import (
        AgenticAlphaBacktestMetric,
        AgenticAlphaKnowledgeSource,
        AgenticAlphaSearchCandidate,
        build_agentic_alpha_hypotheses,
        distill_agentic_alpha_search,
    )

    sources = [
        AgenticAlphaKnowledgeSource(
            source_type="github",
            title="agentic-alpha R-G-E-V-D workflow",
            url="https://github.com/JacobDu/agentic-alpha",
            extracted_signal="research retrieve generate evaluate validate distill factor mining",
            tags=("workflow", "factor"),
        ),
        AgenticAlphaKnowledgeSource(
            source_type="paper",
            title="Conditional volatility targeting",
            url="https://alphaarchitect.com/conditional-volatility-targeting/",
            extracted_signal="conditional volatility targeting reduces drawdown and tail risk",
            tags=("volatility", "risk"),
        ),
        AgenticAlphaKnowledgeSource(
            source_type="twitter",
            title="Leveraged ETF regime thread",
            url="https://x.com/ces921/highlights",
            extracted_signal="levered ETF rebalancing loop and SOXL flow can amplify regime risk",
            tags=("trend", "regime"),
        ),
        AgenticAlphaKnowledgeSource(
            source_type="arxiv",
            title="Compounding effects in leveraged ETFs",
            url="https://arxiv.org/html/2504.20116v1",
            extracted_signal="return autocorrelation and volatility clustering drive LETF compounding",
            tags=("autocorrelation", "volatility"),
        ),
    ]

    hypotheses = build_agentic_alpha_hypotheses(sources)

    assert [item.stage for item in hypotheses] == ["generate", "generate"]
    assert hypotheses[0].theme == "volatility_targeted_trend"
    assert hypotheses[0].source_types == ("arxiv", "github", "paper", "twitter")
    assert "target volatility" in hypotheses[0].candidate_logic

    baseline = AgenticAlphaBacktestMetric(
        name="production_qlib_paper_sim_v1",
        return_pct=6570.0232821043,
        max_drawdown_pct=-85.23168365,
        benchmark_return_pct=3530.8157772578,
        recorder_id="e0b5ed66c6514873b2fc79f2aec5874c",
    )
    candidates = [
        AgenticAlphaSearchCandidate(
            name="vol_target_trend_candidate",
            hypothesis_id=hypotheses[0].hypothesis_id,
            source_urls=hypotheses[0].source_urls,
            metric=AgenticAlphaBacktestMetric(
                name="vol_target_trend_candidate",
                return_pct=7000.0,
                max_drawdown_pct=-18.5,
                benchmark_return_pct=3530.8157772578,
                recorder_id="candidate-recorder",
            ),
        ),
        AgenticAlphaSearchCandidate(
            name="return_only_candidate",
            hypothesis_id=hypotheses[1].hypothesis_id,
            source_urls=hypotheses[1].source_urls,
            metric=AgenticAlphaBacktestMetric(
                name="return_only_candidate",
                return_pct=8000.0,
                max_drawdown_pct=-42.0,
                benchmark_return_pct=3530.8157772578,
                recorder_id="bad-risk-recorder",
            ),
        ),
    ]

    report = distill_agentic_alpha_search(
        sources=sources,
        hypotheses=hypotheses,
        candidates=candidates,
        baseline=baseline,
        max_drawdown_floor=-20.0,
        require_return_above_baseline=True,
    )
    rows = report.candidate_metrics.set_index("candidate")

    assert report.workflow == "agentic_alpha_rgevd_search"
    assert report.summary["stage_order"] == ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
    assert report.summary["external_source_counts"] == {"arxiv": 1, "github": 1, "paper": 1, "twitter": 1}
    assert report.summary["goal_achieved"] is True
    assert report.summary["best_candidate"] == "vol_target_trend_candidate"
    assert rows.loc["vol_target_trend_candidate", "decision"] == "Promote"
    assert rows.loc["vol_target_trend_candidate", "beats_baseline_return"] is True
    assert rows.loc["vol_target_trend_candidate", "passes_drawdown_gate"] is True
    assert rows.loc["return_only_candidate", "decision"] == "Drop"
    assert rows.loc["return_only_candidate", "decision_basis"] == "drawdown_gate"


def test_agentic_alpha_loads_baseline_metric_from_qlib_card_sidecars(tmp_path):
    from src.workflow import load_agentic_alpha_backtest_metric

    card = tmp_path / "production_card"
    card.mkdir()
    (card / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "production_qlib_paper_sim_v1",
                "recorder_id": "recorder-123",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "window": "test",
                "strategy_return_pct": 6570.0232821043,
                "benchmark_return_pct": 3530.8157772578,
                "max_drawdown_pct": -85.23168365,
            },
            {
                "window": "stress",
                "strategy_return_pct": 10.0,
                "benchmark_return_pct": 8.0,
                "max_drawdown_pct": -5.0,
            },
        ]
    ).to_csv(card / "backtest_window_metrics.csv", index=False)

    metric = load_agentic_alpha_backtest_metric(card, window="test")

    assert metric.name == "production_qlib_paper_sim_v1"
    assert metric.recorder_id == "recorder-123"
    assert metric.return_pct == 6570.0232821043
    assert metric.benchmark_return_pct == 3530.8157772578
    assert metric.max_drawdown_pct == -85.23168365


def test_agentic_alpha_sfa_validation_uses_qlib_dataset_and_distills_decisions(tmp_path):
    from src.factor import validate_agentic_alpha_factors

    report = validate_agentic_alpha_factors(
        AgenticAlphaDataset(),
        segment="test",
        factor_cols=["alpha_good", "alpha_duplicate", "alpha_noise"],
        label_cols={"1d": "LABEL1", "5d": "LABEL5"},
        primary_horizon="1d",
        min_cross_section=3,
        min_abs_rank_icir=0.10,
        max_fdr_p=1.0,
        max_abs_correlation=0.80,
        min_decay_ratio=0.50,
        min_stability_score=0.30,
    )
    paths = report.write_sidecars(tmp_path)
    metrics = report.metrics.set_index("factor")

    assert report.source == "qlib_dataset"
    assert report.workflow == "agentic_alpha_sfa"
    assert set(paths) == {"metrics", "correlation", "summary"}
    assert paths["metrics"].name == "agentic_alpha_factor_validation.csv"
    assert metrics.loc["alpha_good", "decision"] == "Promote"
    assert metrics.loc["alpha_good", "weekly_suitable"] is True
    assert metrics.loc["alpha_duplicate", "redundancy_flag"] == "high_similarity"
    assert metrics.loc["alpha_duplicate", "decision"] == "Drop"
    assert metrics.loc["alpha_noise", "decision"] == "Drop"
    assert report.correlation.loc["alpha_good", "alpha_duplicate"] == 1.0

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["source"] == "qlib_dataset"
    assert summary["workflow"] == "agentic_alpha_sfa"
    assert summary["stage_order"] == ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
    assert summary["decision_counts"]["Promote"] == 1
    assert summary["recorder_first"] is True


def test_tail_risk_external_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_tail_risk_external_frame,
        evaluate_tail_risk_candidate,
        summarize_tail_risk_branch,
    )

    dates = pd.date_range("2026-01-02", periods=80, freq="B")
    panel = pd.DataFrame(
        {
            "date": dates,
            "soxl_close": [20.0 * (1.01**index) for index in range(len(dates))],
            "soxl_volume": 1_000_000,
            "soxx_close": [100.0 * (1.003**index) for index in range(len(dates))],
            "soxx_volume": 500_000,
            "vix": 16.0,
        }
    )
    vix9d = pd.DataFrame({"DATE": dates, "CLOSE": 14.0})
    vix3m = pd.DataFrame({"DATE": dates, "CLOSE": 18.0})
    vvix = pd.DataFrame({"DATE": dates, "VVIX": 85.0})
    skew = pd.DataFrame({"DATE": dates, "SKEW": 130.0})

    features = build_tail_risk_external_frame(
        panel=panel,
        vix9d=vix9d,
        vix3m=vix3m,
        vvix=vvix,
        skew=skew,
        start="2026-01-02",
        end="2026-04-30",
    )

    assert {
        "datetime",
        "next_return",
        "trend_20",
        "trend_60",
        "hv20",
        "vix9d_vix3m",
        "vix9d_vix3m_z",
        "vvix_vix_z",
        "skew_z",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing

    promoted = evaluate_tail_risk_candidate(
        features,
        branch="tail_risk_term_structure_ensemble",
        candidate="synthetic_promoted",
        exposure=[1.0] * len(features),
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
        params={"scale": 1.0},
    )
    failed_return_gate = evaluate_tail_risk_candidate(
        features,
        branch="tail_risk_term_structure_ensemble",
        candidate="synthetic_return_gate",
        exposure=[0.0] * len(features),
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
        params={"scale": 0.0},
    )
    summary = summarize_tail_risk_branch(
        pd.DataFrame([promoted, failed_return_gate]),
        branch="tail_risk_term_structure_ensemble",
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
    )

    assert promoted["decision"] == "Promote"
    assert failed_return_gate["decision"] == "Iterate"
    assert failed_return_gate["decision_basis"] == "return_gate"
    assert summary["passes_both_count"] == 1
    assert summary["best_drawdown_gate_max_drawdown_pct"] >= -20.0


def test_cross_asset_rotation_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_cross_asset_price_frame,
        evaluate_cross_asset_rotation_candidates,
        summarize_cross_asset_branch,
    )

    dates = pd.date_range("2026-01-02", periods=90, freq="B")
    raw_frames = {
        "SOXL": pd.DataFrame(
            {
                "date": dates,
                "close": [20.0 * (1.012**index) for index in range(len(dates))],
                "volume": 1_000_000,
            }
        ),
        "TQQQ": pd.DataFrame(
            {
                "date": dates,
                "close": [30.0 * (1.004**index) for index in range(len(dates))],
                "volume": 900_000,
            }
        ),
        "SPXL": pd.DataFrame(
            {
                "date": dates,
                "close": [40.0 * (1.002**index) for index in range(len(dates))],
                "volume": 800_000,
            }
        ),
    }

    price_frame = build_cross_asset_price_frame(raw_frames, start="2026-01-02", end="2026-05-15")
    metrics = evaluate_cross_asset_rotation_candidates(
        price_frame,
        baseline_return_pct=10.0,
        max_drawdown_floor=-20.0,
        symbols=["SOXL", "TQQQ", "SPXL"],
        rotation_lookbacks=[20],
        breadth_lookbacks=[20],
        caps=[1.0],
        breadth_scales=[1.0],
    )
    rotation_summary = summarize_cross_asset_branch(
        metrics,
        branch="relative_momentum_rotation",
        baseline_return_pct=10.0,
        max_drawdown_floor=-20.0,
    )

    assert price_frame["datetime"].is_monotonic_increasing
    assert {"SOXL_close", "TQQQ_close", "SPXL_close"}.issubset(price_frame.columns)
    assert set(metrics["branch"]) == {"cross_asset_breadth_soxl", "relative_momentum_rotation"}
    assert (metrics["decision"] == "Promote").any()
    assert rotation_summary["passes_both_count"] >= 1
    assert rotation_summary["best_drawdown_gate_max_drawdown_pct"] >= -20.0


def test_factor_library_reversal_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_factor_library_reversal_frame,
        compact_factor_library_search_summary,
        evaluate_factor_library_reversal_candidates,
    )

    dates = pd.date_range("2025-01-02", periods=280, freq="B")
    soxl_close = []
    price = 20.0
    for index, _ in enumerate(dates):
        if index in {80, 150, 220}:
            price *= 0.88
        elif index in {81, 82, 151, 152, 221, 222}:
            price *= 1.08
        else:
            price *= 1.003
        soxl_close.append(price)
    soxx_close = [100.0 * (1.0015**index) for index in range(len(dates))]
    rows = []
    for symbol, close_values, volume in [
        ("SOXL", soxl_close, 1_000_000),
        ("SOXX", soxx_close, 800_000),
    ]:
        for date, close in zip(dates, close_values):
            rows.append(
                {
                    "instrument": symbol,
                    "datetime": date,
                    "$close": close,
                    "$volume": volume,
                    "$vix": 18.0 if symbol == "SOXL" else None,
                }
            )
    qlib_frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()

    features = build_factor_library_reversal_frame(
        qlib_frame,
        start="2025-07-01",
        end="2025-12-31",
    )
    metrics = evaluate_factor_library_reversal_candidates(
        features,
        baseline_return_pct=2.0,
        max_drawdown_floor=-20.0,
        reversal_thresholds=[0.25],
        hold_days=[1],
        scales=[1.0],
        volcaps=[None],
        trend_sell_thresholds=[0.90],
        base_scales=[1.0],
        reentry_boosts=[0.0],
        target_vols=[0.20],
        score_caps=[1.0],
    )
    summary = compact_factor_library_search_summary(
        metrics,
        branches=sorted(metrics["branch"].unique()),
        baseline_return_pct=2.0,
        max_drawdown_floor=-20.0,
    )

    assert {
        "next_return",
        "buy_score",
        "panic_buy_score",
        "sell_score",
        "trend_score",
        "rel_volume_ratio",
        "soxx_bias10",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing
    assert set(metrics["branch"]) == {
        "factor_library_reversal_reentry",
        "factor_library_trend_sell_gate",
        "factor_library_vol_managed_score",
    }
    assert (metrics["passes_drawdown_gate"] == True).any()  # noqa: E712
    assert summary["passes_both_count"] >= 1
    assert summary["goal_achieved"] is True


def test_long_short_crisis_alpha_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_long_short_crisis_alpha_frame,
        compact_long_short_crisis_alpha_summary,
        evaluate_long_short_crisis_alpha_candidates,
    )

    dates = pd.date_range("2025-01-02", periods=320, freq="B")
    soxl_close = []
    price = 20.0
    for index, _ in enumerate(dates):
        if 80 <= index < 110:
            price *= 0.985
        elif 180 <= index < 210:
            price *= 0.975
        else:
            price *= 1.004
        soxl_close.append(price)
    soxx_close = [100.0 * (1.0015**index) for index in range(len(dates))]
    rows = []
    for symbol, close_values, volume in [
        ("SOXL", soxl_close, 1_000_000),
        ("SOXX", soxx_close, 800_000),
    ]:
        for date, close in zip(dates, close_values):
            rows.append(
                {
                    "instrument": symbol,
                    "datetime": date,
                    "$close": close,
                    "$volume": volume,
                    "$vix": 18.0 if symbol == "SOXL" else None,
                }
            )
    qlib_frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()

    features = build_long_short_crisis_alpha_frame(
        qlib_frame,
        start="2025-07-01",
        end="2026-03-31",
    )
    metrics = evaluate_long_short_crisis_alpha_candidates(
        features,
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
        trend_names=["dual60_120"],
        positive_thresholds=[0.0],
        negative_thresholds=[0.0],
        risk_modes=["none"],
        long_scales=[1.0],
        short_scales=[1.0],
        target_vols=[None],
        crisis_base_thresholds=[0.0],
        crisis_base_scales=[1.0],
        crisis_hedge_scales=[1.0],
    )
    summary = compact_long_short_crisis_alpha_summary(
        metrics,
        branches=sorted(metrics["branch"].unique()),
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
    )

    assert {
        "next_return",
        "trend60",
        "trend120",
        "dual60_120",
        "soxl_soxx_dual",
        "vix_z",
        "vol_z",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing
    assert set(metrics["branch"]) == {
        "crisis_alpha_overlay",
        "long_short_time_series_momentum",
    }
    assert {"drawdown_gate", "return_gate"} & set(metrics["decision_basis"])
    assert summary["passes_both_count"] >= 0
    assert set(summary["branch_results"]) == set(metrics["branch"])


def test_canary_sprint_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_canary_sprint_frame,
        compact_canary_sprint_summary,
        evaluate_canary_sprint_candidates,
    )

    dates = pd.date_range("2025-01-02", periods=90, freq="B")
    soxl_close = []
    price = 20.0
    for index, _ in enumerate(dates):
        if index in {20, 44, 68}:
            price *= 0.91
        elif index % 9 == 0:
            price *= 0.97
        else:
            price *= 1.012
        soxl_close.append(price)
    soxx_close = [100.0 * (1.001 + (0.004 if index % 12 < 7 else -0.001)) ** index for index in range(len(dates))]
    rows = []
    for symbol, close_values, vix in [
        ("SOXL", soxl_close, 19.0),
        ("SOXX", soxx_close, None),
    ]:
        for date, close in zip(dates, close_values):
            rows.append(
                {
                    "instrument": symbol,
                    "datetime": date,
                    "$close": close,
                    "$volume": 1_000_000,
                    "$vix": vix,
                }
            )
    qlib_frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()

    features = build_canary_sprint_frame(
        qlib_frame,
        start="2025-02-15",
        end="2025-05-15",
    )
    metrics = evaluate_canary_sprint_candidates(
        features,
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
        trend_lookbacks=[20],
        trend_thresholds=[-0.05, 0.0],
        vix_maxes=[26.0],
        autocorr_filters=["none", "positive"],
        hold_days=[1, 3],
        scales=[0.5, 1.0],
        dip_lookbacks=[2],
        dip_thresholds=[-0.08],
        guard_scales=[1.0],
        guard_stop_drawdowns=[-0.18],
        guard_max_daily_losses=[0.10],
        guard_vol_targets=[None],
    )
    summary = compact_canary_sprint_summary(
        metrics,
        branches=sorted(metrics["branch"].unique()),
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
    )

    assert {
        "next_return",
        "soxx_trend20",
        "soxl_dip2",
        "autocorr10",
        "hv20",
        "vix",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing
    assert set(metrics["branch"]) == {
        "canary_sprint_short_hold",
        "dip_sprint_short_hold",
        "guarded_sprint_budget",
    }
    assert {"drawdown_gate", "return_gate", "beats_baseline_and_drawdown_gate"} & set(metrics["decision_basis"])
    assert summary["passes_both_count"] >= 0
    assert set(summary["branch_results"]) == set(metrics["branch"])


def test_meta_label_probability_gate_flags_extreme_exposure_numeric_passes():
    from src.factor import (
        build_meta_label_probability_frame,
        compact_meta_label_probability_summary,
        evaluate_probability_gate_grid,
    )

    dates = pd.date_range("2025-01-02", periods=160, freq="B")
    soxl_close = []
    price = 20.0
    for index, _ in enumerate(dates):
        if index in {40, 75, 110}:
            price *= 0.92
        elif index in {41, 76, 111}:
            price *= 1.12
        else:
            price *= 1.002
        soxl_close.append(price)
    soxx_close = [100.0 * (1.0015**index) for index in range(len(dates))]
    rows = []
    for symbol, close_values, vix in [
        ("SOXL", soxl_close, 19.0),
        ("SOXX", soxx_close, None),
    ]:
        for date, close in zip(dates, close_values):
            rows.append(
                {
                    "instrument": symbol,
                    "datetime": date,
                    "$close": close,
                    "$volume": 1_000_000,
                    "$vix": vix,
                }
            )
    qlib_frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()

    features = build_meta_label_probability_frame(
        qlib_frame,
        start="2025-03-01",
        end="2025-08-15",
        min_history=20,
    )
    features["probability"] = 0.10
    event_dates = features.loc[features["next_return"] > 0.08, "datetime"]
    features.loc[features["datetime"].isin(event_dates), "probability"] = 0.95

    metrics = evaluate_probability_gate_grid(
        features,
        branch="meta_label_probability_gate",
        probability_col="probability",
        event_col="base_dip",
        baseline_return_pct=100.0,
        max_drawdown_floor=-20.0,
        thresholds=[0.9],
        scales=[2.0, 30.0],
        size_modes=["binary"],
        max_exposure_sanity=10.0,
        min_trade_count_sanity=5,
    )
    summary = compact_meta_label_probability_summary(
        metrics,
        branches=["meta_label_probability_gate"],
        baseline_return_pct=100.0,
        max_drawdown_floor=-20.0,
        max_exposure_sanity=10.0,
        min_trade_count_sanity=5,
    )

    assert {
        "next_return",
        "base_trend",
        "base_dip",
        "base_calendar",
        "base_any",
        "ret2",
        "soxx_ma150",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing
    assert (metrics["beats_baseline_return"] & metrics["passes_drawdown_gate"]).any()
    assert not (metrics["decision"] == "Promote").any()
    assert "exposure_sanity_gate" in set(metrics["decision_basis"])
    assert summary["raw_passes_both_count"] >= 1
    assert summary["passes_both_count"] == 0
    assert summary["goal_achieved"] is False


def test_session_effect_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_session_effect_frame,
        compact_session_effect_summary,
        evaluate_session_effect_candidates,
    )

    dates = pd.date_range("2025-01-02", periods=140, freq="B")
    soxl_close = []
    soxl_open = []
    price = 20.0
    for index, _ in enumerate(dates):
        overnight = 1.025 if index % 6 in {1, 2, 3} else 0.985
        intraday = 0.975 if index % 7 in {0, 1} else 1.006
        open_price = price * overnight
        close_price = open_price * intraday
        soxl_open.append(open_price)
        soxl_close.append(close_price)
        price = close_price
    soxx_close = [100.0 * (1.002**index) for index in range(len(dates))]
    rows = []
    for symbol, open_values, close_values, vix in [
        ("SOXL", soxl_open, soxl_close, 19.0),
        ("SOXX", [value * 0.99 for value in soxx_close], soxx_close, None),
    ]:
        for date, open_price, close_price in zip(dates, open_values, close_values):
            rows.append(
                {
                    "instrument": symbol,
                    "datetime": date,
                    "$open": open_price,
                    "$high": max(open_price, close_price) * 1.01,
                    "$low": min(open_price, close_price) * 0.99,
                    "$close": close_price,
                    "$volume": 1_000_000,
                    "$vix": vix,
                }
            )
    qlib_frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()

    features = build_session_effect_frame(
        qlib_frame,
        start="2025-03-01",
        end="2025-07-15",
        min_history=10,
    )
    metrics = evaluate_session_effect_candidates(
        features,
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
        overnight_windows=[3],
        overnight_thresholds=[0.0],
        trend_filters=["none"],
        vix_maxes=[100.0],
        gap_thresholds=[-0.03],
        gap_modes=["buy_gap_down"],
        tug_windows=[5],
        tug_thresholds=[0.0],
        tug_directions=["positive"],
        scales=[1.0],
    )
    summary = compact_session_effect_summary(
        metrics,
        branches=sorted(metrics["branch"].unique()),
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
    )

    assert {
        "next_overnight_return",
        "next_intraday_return",
        "next_close_return",
        "overnight_mom3",
        "tug5",
        "soxx_trend20",
        "vix",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing
    assert set(metrics["branch"]) == {
        "intraday_gap_reversal",
        "overnight_momentum",
        "tug_of_war_close_to_close",
    }
    assert {"drawdown_gate", "return_gate", "beats_baseline_and_drawdown_gate"} & set(metrics["decision_basis"])
    assert summary["passes_both_count"] >= 0
    assert set(summary["branch_results"]) == set(metrics["branch"])


def test_macro_credit_rate_features_and_candidate_gates_are_src_owned():
    from src.factor import (
        build_macro_credit_frame,
        compact_macro_credit_summary,
        evaluate_macro_credit_candidates,
    )

    dates = pd.date_range("2025-01-02", periods=140, freq="B")
    soxl_close = []
    soxx_close = []
    price = 20.0
    for index, _ in enumerate(dates):
        if index % 31 in {0, 1, 2, 3}:
            price *= 0.96
        else:
            price *= 1.008
        soxl_close.append(price)
        soxx_close.append(100.0 * (1.0015**index))

    rows = []
    for date, soxl_price, soxx_price, index in zip(dates, soxl_close, soxx_close, range(len(dates))):
        rows.append(
            {
                "instrument": "SOXL",
                "datetime": date,
                "$soxl_close": soxl_price,
                "$soxx_close": soxx_price,
                "$yc10y2y": -0.20 if index % 50 < 12 else 0.80,
                "$yc10y3m": -0.10 if index % 55 < 10 else 1.10,
                "$credit_baa": 2.0 + (0.7 if index % 40 < 8 else 0.0),
                "$baa_yield": 5.0 + (0.8 if index % 45 < 9 else 0.0),
                "$hy_spread": 4.0 + (1.0 if index % 37 < 7 else 0.0),
                "$dxy": 100.0 + (index % 23) * 0.2,
                "$oil": 70.0 + (index % 29) * 0.3,
            }
        )
    qlib_frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()

    features = build_macro_credit_frame(
        qlib_frame,
        start="2025-03-01",
        end="2025-07-15",
        min_history=10,
    )
    metrics = evaluate_macro_credit_candidates(
        features,
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
        risk_windows=[20],
        trend_windows=[20, 50],
        risk_thresholds=[0.0, 2.0],
        scales=[0.5, 1.0],
        composite_bases=[0.5],
        composite_trend_weights=[0.5],
        composite_risk_weights=[0.5],
        vol_targets=[0.4],
        stop_windows=[42],
        stop_levels=[-0.2],
        reentry_thresholds=[0.0],
    )
    summary = compact_macro_credit_summary(
        metrics,
        branches=sorted(metrics["branch"].unique()),
        baseline_return_pct=5.0,
        max_drawdown_floor=-20.0,
    )

    assert {
        "next_return",
        "soxx_trend20",
        "soxl_hv20",
        "macro_risk_score20",
        "credit_baa_z20",
        "rolling_drawdown42",
    }.issubset(features.columns)
    assert features["datetime"].is_monotonic_increasing
    assert set(metrics["branch"]) == {
        "macro_composite_throttle",
        "macro_credit_rate_gate",
        "trend_drawdown_stop",
    }
    assert {"drawdown_gate", "return_gate", "beats_baseline_and_drawdown_gate"} & set(metrics["decision_basis"])
    assert summary["passes_both_count"] >= 0
    assert set(summary["branch_results"]) == set(metrics["branch"])
