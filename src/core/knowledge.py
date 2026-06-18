from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    source_type: str
    title: str
    url: str
    summary: str
    license_note: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_hypothesis_card(
    source: KnowledgeSource,
    *,
    target_subsystem: str = "research",
    hypothesis: str = "待转化为可测试假设。",
    test_plan: str = "先进入 research notebook，再决定是否晋升到稳定框架。",
) -> dict[str, Any]:
    return {
        "schema_version": "knowledge_card.v1",
        "source": source.to_dict(),
        "target_subsystem": target_subsystem,
        "hypothesis": hypothesis,
        "test_plan": test_plan,
        "promotion_status": "candidate",
        "skill_boundary": "method_only",
        "skill_usage": "skill 只用于启发研究方法和发散思路，不承载项目业务状态。",
    }


def _default_target(source: KnowledgeSource) -> str:
    if source.source_type in {"github", "arxiv", "paper"}:
        return "src/model"
    if source.source_type in {"twitter", "x"}:
        return "src/analysis"
    return "research"


def write_knowledge_queue(output_dir: str | Path, sources: Iterable[KnowledgeSource]) -> dict[str, int]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    deduped = 0
    for source in sources:
        if source.source_id in seen:
            deduped += 1
            continue
        seen.add(source.source_id)
        cards.append(
            build_hypothesis_card(
                source,
                target_subsystem=_default_target(source),
                hypothesis=f"评估 {source.title} 是否能转化为 SOXL 的因子、模型、objective 或 regime analysis 假设。",
                test_plan="在 research notebook 中做可复现实验；通过后再进入 src 稳定接口或 production notebook。",
            )
        )
    payload = {"schema_version": "knowledge_queue.v1", "cards": cards}
    (output / "knowledge_queue.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review_lines = [
        "# 外部知识抓取队列 Review",
        "",
        "## 中文结论",
        "",
        f"- 本次写入 {len(cards)} 条结构化研究卡，去重 {deduped} 条。",
        "- 外部来源只能先作为假设候选，不直接进入生产逻辑。",
        "- 不要把抓取结果写入 skill；skill 只保留方法论和发散提示。",
        "- 后续应在 research notebook 中验证，再决定是否进入 `src/` 或 `production/`。",
    ]
    (output / "knowledge_ingest_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return {"written": len(cards), "deduped": deduped}
