from __future__ import annotations

import json

from src.core.knowledge import KnowledgeSource, build_hypothesis_card, write_knowledge_queue


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
