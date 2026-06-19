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
