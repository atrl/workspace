from __future__ import annotations

import json

import src.core.knowledge as knowledge_api
from src.core.knowledge import (
    KnowledgeSource,
    build_hypothesis_card,
    build_knowledge_backlog,
    write_knowledge_backlog,
    write_knowledge_queue,
)


def test_build_hypothesis_card_structures_external_source_for_backlog():
    source = KnowledgeSource(
        source_id="arxiv:2504.18600",
        source_type="arxiv",
        title="QuantBench: Benchmarking AI Methods for Quantitative Investment",
        url="https://arxiv.org/abs/2504.18600",
        summary="Full-pipeline benchmark emphasizing continual learning and relational financial data.",
        license_note="arXiv abstract metadata; full text not vendored.",
    )

    card = build_hypothesis_card(
        source,
        target_subsystem="src/model",
        hypothesis="用统一模型矩阵比较不同 objective 和关系建模能力。",
        test_plan="先跑 smoke adapter，再替换严格论文实现并保留同一 manifest。",
    )

    assert card["source"]["source_id"] == "arxiv:2504.18600"
    assert card["target_subsystem"] == "src/model"
    assert card["promotion_status"] == "candidate"
    assert card["skill_boundary"] == "method_only"
    assert "业务功能" not in card["skill_usage"]


def test_build_hypothesis_card_blocks_external_sources_before_research_evidence():
    source = KnowledgeSource(
        source_id="x:regime-thread",
        source_type="x",
        title="Regime-aware model review thread",
        url="https://x.com/search?q=regime%20aware%20quant%20evaluation",
        summary="Social-source idea about volatility regime review.",
        license_note="Use as idea only; do not copy post text.",
        retrieved_at="2026-06-19",
        access_status="search_placeholder",
    )

    card = build_hypothesis_card(source, target_subsystem="src/analysis")

    assert card["implementation_allowed"] is False
    assert card["promotion_gate_status"] == "blocked_pending_research"
    assert card["blocked_surfaces_before_gate"] == ["src", "production", "skill"]
    assert card["evidence_requirements"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert card["required_evidence_count"] == 5
    assert card["usage_scope"] == "idea_only"
    assert card["source_reliability"] == "social_unverified"
    assert card["source_used_as"] == "hypothesis_only"
    assert card["allowed_next_surface"] == "research/regime_window_review_validation_v1/regime_window_review_validation_v1.ipynb"


def test_write_knowledge_queue_deduplicates_sources_and_writes_chinese_review_cards(tmp_path):
    sources = [
        KnowledgeSource(
            source_id="github:example/repo",
            source_type="github",
            title="Example model implementation",
            url="https://github.com/example/repo",
            summary="Implements a financial graph model.",
            license_note="Check repository license before vendoring code.",
        ),
        KnowledgeSource(
            source_id="github:example/repo",
            source_type="github",
            title="Duplicate",
            url="https://github.com/example/repo",
            summary="Duplicate entry.",
            license_note="Duplicate.",
        ),
        KnowledgeSource(
            source_id="twitter:thread-1",
            source_type="twitter",
            title="Regime analysis idea",
            url="https://twitter.com/example/status/1",
            summary="Suggests volatility-state evaluation.",
            license_note="Use as idea only; do not vendor text.",
        ),
    ]

    result = write_knowledge_queue(tmp_path, sources)

    assert result["written"] == 2
    assert result["deduped"] == 1
    queue = json.loads((tmp_path / "knowledge_queue.json").read_text(encoding="utf-8"))
    assert [item["source"]["source_id"] for item in queue["cards"]] == [
        "github:example/repo",
        "twitter:thread-1",
    ]
    review = (tmp_path / "knowledge_ingest_review.md").read_text(encoding="utf-8")
    assert "中文结论" in review
    assert "不要把抓取结果写入 skill" in review


def test_build_knowledge_backlog_maps_external_sources_to_testable_src_interfaces():
    sources = [
        KnowledgeSource(
            source_id="arxiv:2211.14730",
            source_type="arxiv",
            title="A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
            url="https://arxiv.org/abs/2211.14730",
            summary="PatchTST uses patches and channel-independence for long-history time-series forecasting.",
            license_note="arXiv metadata only; implementation license must be checked separately.",
        ),
        KnowledgeSource(
            source_id="github:microsoft/qlib",
            source_type="github",
            title="Qlib AI-oriented quantitative investment platform",
            url="https://github.com/microsoft/qlib",
            summary="Qlib provides recorder/analyzer style research-to-production workflow ideas.",
            license_note="Check upstream repository license before vendoring code.",
        ),
        KnowledgeSource(
            source_id="x:regime-review",
            source_type="x",
            title="Regime-aware model review thread",
            url="https://x.com/search?q=regime%20aware%20quant%20evaluation",
            summary="Social-source idea about reviewing models by volatility and drawdown regime.",
            license_note="Use as idea only; do not copy post text.",
        ),
    ]

    backlog = build_knowledge_backlog(sources)

    assert [item["priority_rank"] for item in backlog] == [1, 2, 3]
    patch = next(item for item in backlog if item["source_id"] == "arxiv:2211.14730")
    qlib = next(item for item in backlog if item["source_id"] == "github:microsoft/qlib")
    social = next(item for item in backlog if item["source_id"] == "x:regime-review")

    assert patch["candidate_name"] == "patch_tst_mse"
    assert patch["target_subsystem"] == "src/model"
    assert patch["target_interface"] == "src/model.registry.ModelSpec + src.model.training.fit_predict_model"
    assert patch["promotion_status"] == "backlog"
    assert "license" in patch["license_gate"].lower()
    assert "research notebook" in patch["acceptance_criteria"]

    assert qlib["target_subsystem"] == "src/analysis"
    assert "qlib_recorder" in qlib["target_interface"]
    assert social["candidate_type"] == "method_signal"
    assert social["target_subsystem"] == "src/analysis"
    assert social["source_used_as"] == "hypothesis_only"
    assert social["skill_boundary"] == "method_only"


def test_build_knowledge_backlog_adds_structured_promotion_gate_for_external_sources():
    source = KnowledgeSource(
        source_id="github:example/factor-lab",
        source_type="github",
        title="Factor lab with correlation diagnostics",
        url="https://github.com/example/factor-lab",
        summary="Discusses factor correlation, IC decay, and multi-factor review workflows.",
        license_note="Check repository license before using implementation details.",
    )

    item = build_knowledge_backlog([source])[0]

    assert item["candidate_name"] == "factor_diagnostics"
    assert item["candidate_type"] == "factor_pipeline_reference"
    assert item["target_subsystem"] == "src/factor"
    assert item["promotion_gate_status"] == "blocked_pending_research"
    assert item["allowed_next_surface"].startswith("research/")
    assert item["blocked_surfaces_before_gate"] == ["src", "production", "skill"]
    assert item["required_evidence_count"] == 5
    assert item["evidence_requirements"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert item["gate_notes"].startswith("未满足晋升门禁")


def test_build_knowledge_backlog_routes_rank_ic_sources_to_objective_interface():
    source = KnowledgeSource(
        source_id="arxiv:2605.00501",
        source_type="arxiv",
        title="LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction",
        url="https://arxiv.org/abs/2605.00501",
        summary=(
            "Rank IC objective reference for reducing mismatch between regression losses, "
            "generic ranking objectives, and finance evaluation metrics."
        ),
        license_note="arXiv metadata and self-written summary only.",
    )

    item = build_knowledge_backlog([source])[0]
    audit = knowledge_api.build_knowledge_promotion_audit(
        [item],
        evidence_index={
            "rank_ic_objective": [
                "license_review",
                "research_notebook",
                "focused_pytest",
                "manifest_or_export",
                "chinese_conclusion",
            ]
        },
    )

    assert item["candidate_name"] == "rank_ic_objective"
    assert item["candidate_type"] == "objective_reference"
    assert item["target_subsystem"] == "src/objective"
    assert "evaluate_prediction_objectives" in item["target_interface"]
    assert "build_promotion_scores" in item["target_interface"]
    assert item["allowed_next_surface"] == (
        "research/rank_ic_objective_validation_v1/rank_ic_objective_validation_v1.ipynb"
    )
    assert audit["items"][0]["allowed_surfaces_after_gate"] == [
        "src/objective",
        "research/rank_ic_objective_validation_v1/rank_ic_objective_validation_v1.ipynb",
    ]


def test_build_knowledge_backlog_keeps_factor_ic_sources_on_factor_interface():
    source = KnowledgeSource(
        source_id="github:quantopian/alphalens",
        source_type="github",
        title="Alphalens factor performance analysis",
        url="https://github.com/quantopian/alphalens",
        summary=(
            "Factor-analysis reference for returns analysis, information coefficient analysis, "
            "turnover analysis, and grouped factor diagnostics."
        ),
        license_note="Check upstream license before implementation.",
    )

    item = build_knowledge_backlog([source])[0]

    assert item["candidate_name"] == "factor_diagnostics"
    assert item["candidate_type"] == "factor_pipeline_reference"
    assert item["target_subsystem"] == "src/factor"
    assert item["target_interface"] == "src.factor.prepare_factor_feature_sets"


def test_build_knowledge_backlog_records_source_provenance_and_usage_scope():
    paper = KnowledgeSource(
        source_id="arxiv:2504.18600",
        source_type="arxiv",
        title="QuantBench: Benchmarking AI Methods for Quantitative Investment",
        url="https://arxiv.org/abs/2504.18600",
        summary="Benchmark reference for a standardized full-pipeline quant workflow.",
        license_note="arXiv metadata and self-written summary only.",
        retrieved_at="2026-06-19",
        access_status="metadata_accessible",
    )
    social = KnowledgeSource(
        source_id="x:regime-aware-quant-search",
        source_type="x",
        title="Regime-aware quant evaluation search",
        url="https://x.com/search?q=regime%20aware%20quant%20evaluation",
        summary="Social-source idea placeholder for regime review.",
        license_note="Do not copy post text.",
        retrieved_at="2026-06-19",
        access_status="search_placeholder",
    )

    backlog = build_knowledge_backlog([paper, social])
    paper_item = next(item for item in backlog if item["source_id"] == "arxiv:2504.18600")
    social_item = next(item for item in backlog if item["source_id"] == "x:regime-aware-quant-search")

    assert paper_item["retrieved_at"] == "2026-06-19"
    assert paper_item["access_status"] == "metadata_accessible"
    assert paper_item["usage_scope"] == "hypothesis_and_adapter_reference"
    assert paper_item["source_reliability"] == "primary_metadata"
    assert paper_item["implementation_allowed"] is False

    assert social_item["usage_scope"] == "idea_only"
    assert social_item["source_reliability"] == "social_unverified"
    assert social_item["implementation_allowed"] is False
    assert social_item["promotion_gate_status"] == "blocked_pending_research"


def test_write_knowledge_queue_uses_same_candidate_target_routing_as_backlog(tmp_path):
    source = KnowledgeSource(
        source_id="github:example/factor-lab",
        source_type="github",
        title="Factor lab with correlation diagnostics",
        url="https://github.com/example/factor-lab",
        summary="Discusses factor correlation, IC decay, and multi-factor review workflows.",
        license_note="Check repository license before using implementation details.",
    )

    write_knowledge_queue(tmp_path, [source])

    queue = json.loads((tmp_path / "knowledge_queue.json").read_text(encoding="utf-8"))
    assert queue["cards"][0]["target_subsystem"] == "src/factor"
    assert "research notebook" in queue["cards"][0]["test_plan"]


def test_write_knowledge_backlog_writes_json_csv_and_chinese_review(tmp_path):
    sources = [
        KnowledgeSource(
            source_id="arxiv:2205.13504",
            source_type="paper",
            title="Are Transformers Effective for Time Series Forecasting?",
            url="https://arxiv.org/abs/2205.13504",
            summary="DLinear is a simple decomposition-linear baseline for time-series forecasting.",
            license_note="arXiv metadata only.",
        ),
        KnowledgeSource(
            source_id="arxiv:2205.13504",
            source_type="paper",
            title="Duplicate DLinear",
            url="https://arxiv.org/abs/2205.13504",
            summary="Duplicate entry.",
            license_note="Duplicate.",
        ),
    ]

    result = write_knowledge_backlog(tmp_path, sources)

    assert result == {"written": 1, "deduped": 1}
    payload = json.loads((tmp_path / "knowledge_backlog.json").read_text(encoding="utf-8"))
    csv_bytes = (tmp_path / "knowledge_backlog.csv").read_bytes()
    csv_text = (tmp_path / "knowledge_backlog.csv").read_text(encoding="utf-8")
    review = (tmp_path / "knowledge_backlog_review.md").read_text(encoding="utf-8")

    assert payload["schema_version"] == "knowledge_backlog.v1"
    assert payload["summary"]["source_count"] == 1
    assert payload["summary"]["target_subsystems"] == ["src/model"]
    assert payload["items"][0]["candidate_name"] == "dlinear_mse"
    assert payload["items"][0]["promotion_gate_status"] == "blocked_pending_research"
    assert payload["items"][0]["required_evidence_count"] == 5
    assert b"\r" not in csv_bytes
    assert "source_id,candidate_name" in csv_text
    assert "promotion_gate_status" in csv_text
    assert "required_evidence_count" in csv_text
    assert "中文结论" in review
    assert "晋升门禁" in review
    assert "不进入 skill" in review


def test_build_knowledge_promotion_audit_blocks_items_without_required_evidence():
    source = KnowledgeSource(
        source_id="github:microsoft/qlib",
        source_type="github",
        title="Microsoft Qlib",
        url="https://github.com/microsoft/qlib",
        summary="Qlib recorder and analyzer workflow reference.",
        license_note="Check upstream license before implementation.",
    )
    backlog = build_knowledge_backlog([source])

    assert hasattr(knowledge_api, "build_knowledge_promotion_audit")
    audit = knowledge_api.build_knowledge_promotion_audit(backlog)

    assert audit["schema_version"] == "knowledge_promotion_audit.v1"
    assert audit["summary"]["item_count"] == 1
    assert audit["summary"]["blocked_count"] == 1
    assert audit["summary"]["implementation_allowed_count"] == 0
    item = audit["items"][0]
    assert item["candidate_name"] == "qlib_recorder_analysis"
    assert item["promotion_gate_status"] == "blocked_missing_evidence"
    assert item["implementation_allowed"] is False
    assert item["missing_evidence"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert item["blocked_surfaces"] == ["src", "production", "skill"]
    assert item["allowed_surfaces_after_gate"] == []


def test_build_knowledge_promotion_audit_allows_item_after_required_evidence():
    source = KnowledgeSource(
        source_id="arxiv:2211.14730",
        source_type="arxiv",
        title="A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
        url="https://arxiv.org/abs/2211.14730",
        summary="PatchTST patching and channel-independence reference.",
        license_note="arXiv metadata only; implementation license reviewed separately.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = {
        "patch_tst_mse": [
            "license_review",
            "research_notebook",
            "focused_pytest",
            "manifest_or_export",
            "chinese_conclusion",
        ]
    }

    assert hasattr(knowledge_api, "build_knowledge_promotion_audit")
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert audit["summary"]["ready_count"] == 1
    assert audit["summary"]["blocked_count"] == 0
    item = audit["items"][0]
    assert item["candidate_name"] == "patch_tst_mse"
    assert item["promotion_gate_status"] == "research_evidence_complete"
    assert item["implementation_allowed"] is True
    assert item["missing_evidence"] == []
    assert item["license_gate_status"] == "passed"
    assert item["allowed_surfaces_after_gate"] == [
        "src/model",
        "research/patch_tst_mse_validation_v1/patch_tst_mse_validation_v1.ipynb",
    ]


def test_build_knowledge_research_evidence_index_reads_real_research_card_files():
    sources = [
        KnowledgeSource(
            source_id="arxiv:2605.00501",
            source_type="arxiv",
            title="LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction",
            url="https://arxiv.org/abs/2605.00501",
            summary=(
                "Rank IC objective reference for reducing mismatch between regression losses, "
                "generic ranking objectives, and finance evaluation metrics."
            ),
            license_note="arXiv metadata and self-written summary only.",
        ),
        KnowledgeSource(
            source_id="arxiv:2106.13008",
            source_type="paper",
            title="Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
            url="https://arxiv.org/abs/2106.13008",
            summary="Autoformer decomposition and auto-correlation reference.",
            license_note="arXiv metadata only.",
        ),
        KnowledgeSource(
            source_id="arxiv:2201.12740",
            source_type="paper",
            title="FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
            url="https://arxiv.org/abs/2201.12740",
            summary="FEDformer frequency/decomposition reference.",
            license_note="arXiv metadata only.",
        ),
        KnowledgeSource(
            source_id="arxiv:2205.13504",
            source_type="paper",
            title="Are Transformers Effective for Time Series Forecasting?",
            url="https://arxiv.org/abs/2205.13504",
            summary="DLinear decomposition-linear reference.",
            license_note="arXiv metadata only.",
        ),
        KnowledgeSource(
            source_id="arxiv:2211.14730",
            source_type="arxiv",
            title="A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
            url="https://arxiv.org/abs/2211.14730",
            summary="PatchTST patching and channel-independence reference.",
            license_note="arXiv metadata only.",
        ),
        KnowledgeSource(
            source_id="github:thuml/Time-Series-Library",
            source_type="github",
            title="Time-Series-Library",
            url="https://github.com/thuml/Time-Series-Library",
            summary="Reference implementation hub for PatchTST and related long-series models.",
            license_note="GitHub metadata and repository license reviewed separately.",
        ),
    ]
    backlog = build_knowledge_backlog(sources)

    assert hasattr(knowledge_api, "build_knowledge_research_evidence_index")
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert evidence_index["rank_ic_objective"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert evidence_index["autoformer_mse"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert evidence_index["fedformer_mse"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert evidence_index["dlinear_mse"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert evidence_index["patch_tst_mse"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert evidence_index["arxiv:2211.14730"] == evidence_index["patch_tst_mse"]
    assert evidence_index["github:thuml/Time-Series-Library"] == evidence_index["patch_tst_mse"]
    rank_item = next(item for item in audit["items"] if item["candidate_name"] == "rank_ic_objective")
    autoformer_item = next(item for item in audit["items"] if item["candidate_name"] == "autoformer_mse")
    fedformer_item = next(item for item in audit["items"] if item["candidate_name"] == "fedformer_mse")
    dlinear_item = next(item for item in audit["items"] if item["candidate_name"] == "dlinear_mse")
    patch_items = [item for item in audit["items"] if item["candidate_name"] == "patch_tst_mse"]
    assert rank_item["promotion_gate_status"] == "research_evidence_complete"
    assert rank_item["implementation_allowed"] is True
    assert autoformer_item["promotion_gate_status"] == "research_evidence_complete"
    assert autoformer_item["implementation_allowed"] is True
    assert autoformer_item["missing_evidence"] == []
    assert fedformer_item["promotion_gate_status"] == "research_evidence_complete"
    assert fedformer_item["implementation_allowed"] is True
    assert fedformer_item["missing_evidence"] == []
    assert dlinear_item["promotion_gate_status"] == "research_evidence_complete"
    assert dlinear_item["implementation_allowed"] is True
    assert dlinear_item["missing_evidence"] == []
    assert len(patch_items) == 2
    assert {item["source_id"] for item in patch_items} == {
        "arxiv:2211.14730",
        "github:thuml/Time-Series-Library",
    }
    assert all(item["promotion_gate_status"] == "research_evidence_complete" for item in patch_items)
    assert all(item["implementation_allowed"] is True for item in patch_items)
    assert all(item["missing_evidence"] == [] for item in patch_items)


def test_write_knowledge_promotion_audit_writes_json_csv_and_chinese_review(tmp_path):
    source = KnowledgeSource(
        source_id="x:regime-aware-quant-search",
        source_type="x",
        title="Regime-aware quant evaluation search",
        url="https://x.com/search?q=regime%20aware%20quant%20evaluation",
        summary="Social-source idea placeholder for volatility/drawdown regime review.",
        license_note="Use as idea only; do not copy post text.",
    )
    backlog = build_knowledge_backlog([source])

    assert hasattr(knowledge_api, "write_knowledge_promotion_audit")
    result = knowledge_api.write_knowledge_promotion_audit(tmp_path, backlog)

    payload = json.loads((tmp_path / "knowledge_promotion_audit.json").read_text(encoding="utf-8"))
    csv_text = (tmp_path / "knowledge_promotion_audit.csv").read_text(encoding="utf-8")
    review = (tmp_path / "knowledge_promotion_audit_review.md").read_text(encoding="utf-8")

    assert result == {"written": 1, "ready": 0, "blocked": 1}
    assert payload["summary"]["blocked_count"] == 1
    assert "candidate_name,promotion_gate_status" in csv_text
    assert "中文结论" in review
    assert "不得进入 `src/production/skill`" in review


def test_rank_ic_objective_validation_card_supplies_backlog_evidence():
    card = "research/rank_ic_objective_validation_v1"
    notebook_path = f"{card}/rank_ic_objective_validation_v1.ipynb"
    source = KnowledgeSource(
        source_id="arxiv:2605.00501",
        source_type="arxiv",
        title="LambdaRankIC: Directly Optimizing Rank IC for Financial Prediction",
        url="https://arxiv.org/abs/2605.00501",
        summary=(
            "Rank IC objective reference for reducing mismatch between regression losses, "
            "generic ranking objectives, and finance evaluation metrics."
        ),
        license_note="arXiv metadata and self-written summary only.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = {
        "rank_ic_objective": [
            "license_review",
            "research_notebook",
            "focused_pytest",
            "manifest_or_export",
            "chinese_conclusion",
        ]
    }

    summary = json.loads(open(f"{card}/rank_ic_objective_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    objective_scores = json.loads(open(f"{card}/objective_scores.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert summary["schema_version"] == "research.rank_ic_objective_validation.v1"
    assert summary["source_id"] == "arxiv:2605.00501"
    assert summary["candidate_name"] == "rank_ic_objective"
    assert summary["target_subsystem"] == "src/objective"
    assert summary["objective_schema_version"] == "objective.evaluation.v1"
    assert summary["rank_ic_aligned_model"] == "rank_ic_aligned_signal"
    assert summary["rank_ic_aligned_score"] > summary["mse_aligned_score"]
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["rank_ic_objective"]
    assert source_provenance["source_id"] == summary["source_id"]
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert "arxiv.org/abs/2605.00501" in source_provenance["url"]
    assert objective_scores["rank_ic_aligned_signal"]["rank_ic"] == summary["rank_ic_aligned_score"]
    assert "未复制论文全文" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 1
    assert audit["items"][0]["promotion_gate_status"] == "research_evidence_complete"


def test_autoformer_mse_validation_card_supplies_backlog_evidence():
    card = "research/autoformer_mse_validation_v1"
    notebook_path = f"{card}/autoformer_mse_validation_v1.ipynb"
    source = KnowledgeSource(
        source_id="arxiv:2106.13008",
        source_type="paper",
        title="Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting",
        url="https://arxiv.org/abs/2106.13008",
        summary="Autoformer decomposition and auto-correlation reference for the autoformer_mse adapter fidelity backlog.",
        license_note="arXiv metadata and self-written summary only.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)

    summary = json.loads(open(f"{card}/autoformer_mse_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    adapter_evidence = json.loads(open(f"{card}/adapter_evidence.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert summary["schema_version"] == "research.autoformer_mse_validation.v1"
    assert summary["source_id"] == "arxiv:2106.13008"
    assert summary["candidate_name"] == "autoformer_mse"
    assert summary["target_subsystem"] == "src/model"
    assert summary["model_adapter"] == "autoformer_decomposition_attention_mse_v1"
    assert summary["adapter_status"] == "paper_strict_adapter"
    assert summary["paper_fidelity"] == "native_sequence"
    assert summary["is_proxy"] is False
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["autoformer_mse"]
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert "arxiv.org/abs/2106.13008" in source_provenance["url"]
    assert adapter_evidence["prediction_row_count"] > 0
    assert adapter_evidence["prediction_non_null"] is True
    assert "未复制论文全文" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 1
    assert audit["items"][0]["promotion_gate_status"] == "research_evidence_complete"


def test_fedformer_mse_validation_card_supplies_backlog_evidence():
    card = "research/fedformer_mse_validation_v1"
    notebook_path = f"{card}/fedformer_mse_validation_v1.ipynb"
    source = KnowledgeSource(
        source_id="arxiv:2201.12740",
        source_type="paper",
        title="FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
        url="https://arxiv.org/abs/2201.12740",
        summary="FEDformer frequency/decomposition reference for the fedformer_mse adapter fidelity backlog.",
        license_note="arXiv metadata and self-written summary only.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)

    summary = json.loads(open(f"{card}/fedformer_mse_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    adapter_evidence = json.loads(open(f"{card}/adapter_evidence.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert summary["schema_version"] == "research.fedformer_mse_validation.v1"
    assert summary["source_id"] == "arxiv:2201.12740"
    assert summary["candidate_name"] == "fedformer_mse"
    assert summary["target_subsystem"] == "src/model"
    assert summary["model_adapter"] == "fedformer_frequency_attention_mse_v1"
    assert summary["adapter_status"] == "paper_strict_adapter"
    assert summary["paper_fidelity"] == "native_sequence"
    assert summary["is_proxy"] is False
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["fedformer_mse"]
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert "arxiv.org/abs/2201.12740" in source_provenance["url"]
    assert adapter_evidence["prediction_row_count"] > 0
    assert adapter_evidence["prediction_non_null"] is True
    assert adapter_evidence["has_frequency_head"] is True
    assert "未复制论文全文" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 1
    assert audit["items"][0]["promotion_gate_status"] == "research_evidence_complete"


def test_dlinear_mse_validation_card_supplies_backlog_evidence():
    card = "research/dlinear_mse_validation_v1"
    notebook_path = f"{card}/dlinear_mse_validation_v1.ipynb"
    source = KnowledgeSource(
        source_id="arxiv:2205.13504",
        source_type="paper",
        title="Are Transformers Effective for Time Series Forecasting?",
        url="https://arxiv.org/abs/2205.13504",
        summary="DLinear decomposition-linear reference for the dlinear_mse adapter fidelity backlog.",
        license_note="arXiv metadata and self-written summary only.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)

    summary = json.loads(open(f"{card}/dlinear_mse_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    adapter_evidence = json.loads(open(f"{card}/adapter_evidence.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert summary["schema_version"] == "research.dlinear_mse_validation.v1"
    assert summary["source_id"] == "arxiv:2205.13504"
    assert summary["candidate_name"] == "dlinear_mse"
    assert summary["target_subsystem"] == "src/model"
    assert summary["model_adapter"] == "dlinear_decomposition_mse_v1"
    assert summary["adapter_status"] == "paper_strict_adapter"
    assert summary["paper_fidelity"] == "native_sequence"
    assert summary["is_proxy"] is False
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["dlinear_mse"]
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert "arxiv.org/abs/2205.13504" in source_provenance["url"]
    assert adapter_evidence["prediction_row_count"] > 0
    assert adapter_evidence["prediction_non_null"] is True
    assert adapter_evidence["has_trend_head"] is True
    assert adapter_evidence["has_residual_head"] is True
    assert "未复制论文全文" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 1
    assert audit["items"][0]["promotion_gate_status"] == "research_evidence_complete"


def test_patch_tst_mse_validation_card_supplies_dual_source_backlog_evidence():
    card = "research/patch_tst_mse_validation_v1"
    notebook_path = f"{card}/patch_tst_mse_validation_v1.ipynb"
    sources = [
        KnowledgeSource(
            source_id="arxiv:2211.14730",
            source_type="arxiv",
            title="A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
            url="https://arxiv.org/abs/2211.14730",
            summary="PatchTST patching and channel-independence reference for the patch_tst_mse adapter backlog.",
            license_note="arXiv metadata and self-written summary only.",
        ),
        KnowledgeSource(
            source_id="github:thuml/Time-Series-Library",
            source_type="github",
            title="Time-Series-Library",
            url="https://github.com/thuml/Time-Series-Library",
            summary="Reference implementation hub for PatchTST and related long-series models.",
            license_note="Repository metadata and license review only; no vendored code.",
        ),
    ]
    backlog = build_knowledge_backlog(sources)
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)

    summary = json.loads(open(f"{card}/patch_tst_mse_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    adapter_evidence = json.loads(open(f"{card}/adapter_evidence.json", encoding="utf-8").read())
    artifact_metadata = json.loads(open(f"{card}/artifact_metadata.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)
    patch_items = [item for item in audit["items"] if item["candidate_name"] == "patch_tst_mse"]

    assert summary["schema_version"] == "research.patch_tst_mse_validation.v1"
    assert summary["source_ids"] == ["arxiv:2211.14730", "github:thuml/Time-Series-Library"]
    assert summary["candidate_name"] == "patch_tst_mse"
    assert summary["target_subsystem"] == "src/model"
    assert summary["model_adapter"] == "patch_tst_transformer_mse_v1"
    assert summary["adapter_status"] == "paper_strict_adapter"
    assert summary["paper_fidelity"] == "native_sequence"
    assert summary["is_proxy"] is False
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["patch_tst_mse"]
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert [source["source_id"] for source in source_provenance["sources"]] == summary["source_ids"]
    assert "arxiv.org/abs/2211.14730" in source_provenance["sources"][0]["url"]
    assert "github.com/thuml/Time-Series-Library" in source_provenance["sources"][1]["url"]
    assert adapter_evidence["prediction_row_count"] > 0
    assert adapter_evidence["prediction_non_null"] is True
    assert adapter_evidence["has_transformer_encoder"] is True
    assert adapter_evidence["patch_size"] == 2
    assert adapter_evidence["patch_count"] > 0
    assert artifact_metadata["model_adapter"] == "patch_tst_transformer_mse_v1"
    assert "MIT" in license_review
    assert "未复制论文全文" in license_review
    assert "未 vendoring" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 2
    assert len(patch_items) == 2
    assert all(item["promotion_gate_status"] == "research_evidence_complete" for item in patch_items)


def test_qlib_recorder_analysis_validation_card_supplies_backlog_evidence():
    card = "research/qlib_recorder_analysis_validation_v1"
    notebook_path = f"{card}/qlib_recorder_analysis_validation_v1.ipynb"
    source = KnowledgeSource(
        source_id="github:microsoft/qlib",
        source_type="github",
        title="Microsoft Qlib",
        url="https://github.com/microsoft/qlib",
        summary="Qlib recorder and analyzer workflow reference for src.analysis recorder artifact ingestion.",
        license_note="GitHub metadata and MIT license review only; no vendored code.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)

    summary = json.loads(open(f"{card}/qlib_recorder_analysis_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    recorder_summary = json.loads(open(f"{card}/recorder_analysis_summary.json", encoding="utf-8").read())
    report_summary = json.loads(open(f"{card}/analysis_summary.json", encoding="utf-8").read())
    manifest = json.loads(open(f"{card}/analysis_manifest.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert summary["schema_version"] == "research.qlib_recorder_analysis_validation.v1"
    assert summary["source_id"] == "github:microsoft/qlib"
    assert summary["candidate_name"] == "qlib_recorder_analysis"
    assert summary["target_subsystem"] == "src/analysis"
    assert summary["analysis_report_status"] == "qlib_recorder_regime_window_analysis_completed"
    assert summary["source_model_count"] == 2
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["qlib_recorder_analysis"]
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert "github.com/microsoft/qlib" in source_provenance["url"]
    assert source_provenance["external_source_checks"]["repository_license"] == "MIT"
    assert recorder_summary["schema_version"] == "analysis.qlib_recorder.validation_summary.v1"
    assert set(recorder_summary["models"]) == {"ridge_mse", "gcn_mse"}
    assert recorder_summary["table_counts"]["qlib_portfolio_report"] == 16
    assert recorder_summary["table_counts"]["qlib_risk_analysis"] == 6
    assert report_summary["status"] == "qlib_recorder_regime_window_analysis_completed"
    assert "model_review_briefing" in report_summary["analysis_layers"]
    assert manifest["metadata"]["source_type"] == "qlib_recorder_regime_window"
    assert "MIT" in license_review
    assert "未 vendoring" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 1
    assert audit["items"][0]["promotion_gate_status"] == "research_evidence_complete"


def test_factor_diagnostics_validation_card_supplies_backlog_evidence():
    card = "research/factor_diagnostics_validation_v1"
    notebook_path = f"{card}/factor_diagnostics_validation_v1.ipynb"
    source = KnowledgeSource(
        source_id="github:quantopian/alphalens",
        source_type="github",
        title="Alphalens factor performance analysis",
        url="https://github.com/quantopian/alphalens",
        summary=(
            "GitHub factor-analysis reference for returns analysis, information coefficient analysis, "
            "turnover analysis, and grouped analysis; useful only as a src.factor diagnostics hypothesis."
        ),
        license_note="GitHub metadata and Apache-2.0 license review only; no vendored code.",
    )
    backlog = build_knowledge_backlog([source])
    evidence_index = knowledge_api.build_knowledge_research_evidence_index(".", backlog)

    summary = json.loads(open(f"{card}/factor_diagnostics_summary.json", encoding="utf-8").read())
    source_provenance = json.loads(open(f"{card}/source_provenance.json", encoding="utf-8").read())
    analysis_summary = json.loads(open(f"{card}/analysis_summary.json", encoding="utf-8").read())
    license_review = open(f"{card}/license_review.md", encoding="utf-8").read()
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    conclusion = "".join(notebook["cells"][-1]["source"])
    audit = knowledge_api.build_knowledge_promotion_audit(backlog, evidence_index=evidence_index)

    assert summary["schema_version"] == "research.factor_diagnostics_validation.v1"
    assert summary["source_id"] == "github:quantopian/alphalens"
    assert summary["candidate_name"] == "factor_diagnostics"
    assert summary["target_subsystem"] == "src/factor"
    assert summary["factor_report_schema_version"] == "factor_suite_report.v1"
    assert summary["diagnostic_interface"] == "src.factor.analyze_factor_suite"
    assert summary["implementation_allowed"] is False
    assert summary["promotion_target"] == "research_only"
    assert summary["evidence_requirements"] == evidence_index["factor_diagnostics"]
    assert {"factor_ic.csv", "factor_correlation.csv", "factor_quantile_returns.csv"}.issubset(
        set(summary["artifact_files"])
    )
    assert source_provenance["source_used_as"] == "hypothesis_only"
    assert "github.com/quantopian/alphalens" in source_provenance["url"]
    assert source_provenance["external_source_checks"]["repository_license"] == "Apache-2.0"
    assert analysis_summary["schema_version"] == "factor_suite_report.v1"
    assert analysis_summary["factor_count"] >= 3
    assert "Apache-2.0" in license_review
    assert "未 vendoring" in license_review
    assert "中文结论" in conclusion
    assert "是否晋升" in conclusion
    assert audit["summary"]["ready_count"] == 1
    assert audit["items"][0]["promotion_gate_status"] == "research_evidence_complete"


def test_knowledge_ingest_audit_marks_rank_ic_objective_research_evidence_complete():
    audit = json.loads(open("research/knowledge_ingest/knowledge_promotion_audit.json", encoding="utf-8").read())
    rank_item = next(item for item in audit["items"] if item["candidate_name"] == "rank_ic_objective")
    autoformer_item = next(item for item in audit["items"] if item["candidate_name"] == "autoformer_mse")
    fedformer_item = next(item for item in audit["items"] if item["candidate_name"] == "fedformer_mse")
    dlinear_item = next(item for item in audit["items"] if item["candidate_name"] == "dlinear_mse")
    patch_items = [item for item in audit["items"] if item["candidate_name"] == "patch_tst_mse"]
    qlib_item = next(item for item in audit["items"] if item["candidate_name"] == "qlib_recorder_analysis")
    factor_item = next(item for item in audit["items"] if item["candidate_name"] == "factor_diagnostics")

    assert audit["summary"]["ready_count"] == 8
    assert audit["summary"]["blocked_count"] == 2
    assert audit["summary"]["implementation_allowed_count"] == 8
    assert "rank_ic_objective" in audit["summary"]["ready_candidates"]
    assert "autoformer_mse" in audit["summary"]["ready_candidates"]
    assert "fedformer_mse" in audit["summary"]["ready_candidates"]
    assert "dlinear_mse" in audit["summary"]["ready_candidates"]
    assert audit["summary"]["ready_candidates"].count("patch_tst_mse") == 2
    assert "qlib_recorder_analysis" in audit["summary"]["ready_candidates"]
    assert "factor_diagnostics" in audit["summary"]["ready_candidates"]
    assert rank_item["promotion_gate_status"] == "research_evidence_complete"
    assert rank_item["implementation_allowed"] is True
    assert rank_item["missing_evidence"] == []
    assert rank_item["provided_evidence"] == [
        "license_review",
        "research_notebook",
        "focused_pytest",
        "manifest_or_export",
        "chinese_conclusion",
    ]
    assert rank_item["allowed_surfaces_after_gate"] == [
        "src/objective",
        "research/rank_ic_objective_validation_v1/rank_ic_objective_validation_v1.ipynb",
    ]
    assert autoformer_item["promotion_gate_status"] == "research_evidence_complete"
    assert autoformer_item["implementation_allowed"] is True
    assert autoformer_item["missing_evidence"] == []
    assert autoformer_item["allowed_surfaces_after_gate"] == [
        "src/model",
        "research/autoformer_mse_validation_v1/autoformer_mse_validation_v1.ipynb",
    ]
    assert fedformer_item["promotion_gate_status"] == "research_evidence_complete"
    assert fedformer_item["implementation_allowed"] is True
    assert fedformer_item["missing_evidence"] == []
    assert fedformer_item["allowed_surfaces_after_gate"] == [
        "src/model",
        "research/fedformer_mse_validation_v1/fedformer_mse_validation_v1.ipynb",
    ]
    assert dlinear_item["promotion_gate_status"] == "research_evidence_complete"
    assert dlinear_item["implementation_allowed"] is True
    assert dlinear_item["missing_evidence"] == []
    assert dlinear_item["allowed_surfaces_after_gate"] == [
        "src/model",
        "research/dlinear_mse_validation_v1/dlinear_mse_validation_v1.ipynb",
    ]
    assert {item["source_id"] for item in patch_items} == {
        "arxiv:2211.14730",
        "github:thuml/Time-Series-Library",
    }
    assert all(item["promotion_gate_status"] == "research_evidence_complete" for item in patch_items)
    assert all(item["implementation_allowed"] is True for item in patch_items)
    assert all(item["missing_evidence"] == [] for item in patch_items)
    assert all(
        item["allowed_surfaces_after_gate"]
        == [
            "src/model",
            "research/patch_tst_mse_validation_v1/patch_tst_mse_validation_v1.ipynb",
        ]
        for item in patch_items
    )
    assert qlib_item["promotion_gate_status"] == "research_evidence_complete"
    assert qlib_item["implementation_allowed"] is True
    assert qlib_item["missing_evidence"] == []
    assert qlib_item["allowed_surfaces_after_gate"] == [
        "src/analysis",
        "research/qlib_recorder_analysis_validation_v1/qlib_recorder_analysis_validation_v1.ipynb",
    ]
    assert factor_item["promotion_gate_status"] == "research_evidence_complete"
    assert factor_item["implementation_allowed"] is True
    assert factor_item["missing_evidence"] == []
    assert factor_item["allowed_surfaces_after_gate"] == [
        "src/factor",
        "research/factor_diagnostics_validation_v1/factor_diagnostics_validation_v1.ipynb",
    ]


def test_knowledge_ingest_notebook_derives_promotion_audit_from_research_cards():
    notebook = json.loads(open("research/knowledge_ingest/knowledge_ingest.ipynb", encoding="utf-8").read())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    summary = json.loads(open("research/knowledge_ingest/knowledge_ingest_summary.json", encoding="utf-8").read())

    assert "build_knowledge_research_evidence_index" in notebook_source
    assert "evidence_index = {" not in notebook_source
    assert summary["promotion_evidence_source"] == "research_card_scan"
    assert summary["promotion_evidence_ready_candidates"] == [
        "autoformer_mse",
        "fedformer_mse",
        "dlinear_mse",
        "patch_tst_mse",
        "rank_ic_objective",
        "patch_tst_mse",
        "qlib_recorder_analysis",
        "factor_diagnostics",
    ]
    assert summary["promotion_evidence_index_keys"] == [
        "arxiv:2106.13008",
        "arxiv:2201.12740",
        "arxiv:2205.13504",
        "arxiv:2211.14730",
        "arxiv:2605.00501",
        "autoformer_mse",
        "dlinear_mse",
        "factor_diagnostics",
        "fedformer_mse",
        "github:microsoft/qlib",
        "github:quantopian/alphalens",
        "github:thuml/Time-Series-Library",
        "patch_tst_mse",
        "qlib_recorder_analysis",
        "rank_ic_objective",
        "research/autoformer_mse_validation_v1/autoformer_mse_validation_v1.ipynb",
        "research/dlinear_mse_validation_v1/dlinear_mse_validation_v1.ipynb",
        "research/factor_diagnostics_validation_v1/factor_diagnostics_validation_v1.ipynb",
        "research/fedformer_mse_validation_v1/fedformer_mse_validation_v1.ipynb",
        "research/patch_tst_mse_validation_v1/patch_tst_mse_validation_v1.ipynb",
        "research/qlib_recorder_analysis_validation_v1/qlib_recorder_analysis_validation_v1.ipynb",
        "research/rank_ic_objective_validation_v1/rank_ic_objective_validation_v1.ipynb",
    ]
