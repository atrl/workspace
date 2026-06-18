from __future__ import annotations

import json

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
    csv_text = (tmp_path / "knowledge_backlog.csv").read_text(encoding="utf-8")
    review = (tmp_path / "knowledge_backlog_review.md").read_text(encoding="utf-8")

    assert payload["schema_version"] == "knowledge_backlog.v1"
    assert payload["summary"]["source_count"] == 1
    assert payload["summary"]["target_subsystems"] == ["src/model"]
    assert payload["items"][0]["candidate_name"] == "dlinear_mse"
    assert "source_id,candidate_name" in csv_text
    assert "中文结论" in review
    assert "不进入 skill" in review
