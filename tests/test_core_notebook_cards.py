from __future__ import annotations

import json

from src.core import scan_notebook_cards, write_notebook_card_manifest


def _write_notebook(path, conclusion: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# test\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": [conclusion]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_scan_notebook_cards_reports_primary_notebook_shape_and_chinese_conclusion(tmp_path):
    _write_notebook(
        tmp_path / "research" / "alpha" / "alpha.ipynb",
        "## 中文结论\n- 做了什么：x\n- 核心指标：x\n- regime 差异：x\n- 是否晋升：否\n- 风险：x\n- 下一步：x\n",
    )
    _write_notebook(
        tmp_path / "production" / "paper" / "paper.ipynb",
        "## 中文结论\n- 做了什么：x\n- 核心指标：x\n- regime 差异：x\n- 是否晋升：否\n- 风险：x\n- 下一步：x\n",
    )
    _write_notebook(
        tmp_path / "research" / "knowledge_ingest" / "knowledge_ingest_reference.ipynb",
        "## 中文结论\n- 做了什么：x\n- 核心指标：x\n- regime 差异：x\n- 是否晋升：否\n- 风险：x\n- 下一步：x\n",
    )

    scan = scan_notebook_cards(tmp_path)

    assert scan["schema_version"] == "core.notebook_card_scan.v1"
    assert scan["summary"]["card_count"] == 3
    assert scan["summary"]["notebook_count"] == 3
    assert scan["summary"]["missing_notebook_count"] == 0
    assert scan["summary"]["multiple_notebook_count"] == 0
    assert scan["summary"]["canonical_notebook_count"] == 2
    assert scan["summary"]["noncanonical_notebook_count"] == 1
    assert scan["summary"]["chinese_conclusion_ok_count"] == 3
    knowledge = next(card for card in scan["cards"] if card["card_id"] == "knowledge_ingest")
    assert knowledge["primary_notebook"] == "knowledge_ingest_reference.ipynb"
    assert knowledge["canonical_notebook_name"] == "knowledge_ingest.ipynb"
    assert knowledge["status"] == "noncanonical_notebook_name"


def test_scan_notebook_cards_reports_missing_multiple_and_conclusion_gaps(tmp_path):
    (tmp_path / "research" / "missing").mkdir(parents=True)
    _write_notebook(tmp_path / "research" / "multi" / "multi.ipynb", "## 中文结论\n是否晋升：否\n")
    _write_notebook(tmp_path / "research" / "multi" / "extra.ipynb", "## 中文结论\n是否晋升：否\n")
    _write_notebook(tmp_path / "production" / "bad_conclusion" / "bad_conclusion.ipynb", "# no conclusion\n")

    scan = scan_notebook_cards(tmp_path)
    statuses = {card["card_id"]: card["status"] for card in scan["cards"]}

    assert statuses["missing"] == "missing_notebook"
    assert statuses["multi"] == "multiple_notebooks"
    assert statuses["bad_conclusion"] == "missing_chinese_conclusion"
    assert scan["summary"]["missing_notebook_count"] == 1
    assert scan["summary"]["multiple_notebook_count"] == 1
    assert scan["summary"]["missing_chinese_conclusion_count"] == 1


def test_write_notebook_card_manifest_writes_json_csv_and_chinese_review(tmp_path):
    _write_notebook(
        tmp_path / "research" / "alpha" / "alpha.ipynb",
        "## 中文结论\n- 做了什么：x\n- 核心指标：x\n- regime 差异：x\n- 是否晋升：否\n- 风险：x\n- 下一步：x\n",
    )
    scan = scan_notebook_cards(tmp_path)

    result = write_notebook_card_manifest(tmp_path / "card_manifest", scan)

    manifest = json.loads((tmp_path / result["manifest_path"]).read_text(encoding="utf-8"))
    csv_text = (tmp_path / result["csv_path"]).read_text(encoding="utf-8")
    review = (tmp_path / result["review_path"]).read_text(encoding="utf-8")
    assert manifest["schema_version"] == "core.notebook_card_manifest.v1"
    assert manifest["summary"]["card_count"] == 1
    assert len(manifest["manifest_hash"]) == 64
    assert "surface,card_id,status" in csv_text
    assert "中文结论" in review
    assert "主 notebook" in review


def test_scan_notebook_cards_accepts_regime_window_difference_wording(tmp_path):
    _write_notebook(
        tmp_path / "research" / "regime_alias" / "regime_alias.ipynb",
        "## 中文结论\n- 做了什么：x\n- 核心指标：x\n- Regime/区间差异：x\n- 是否晋升：否\n- 风险：x\n- 下一步：x\n",
    )

    scan = scan_notebook_cards(tmp_path)

    assert scan["summary"]["missing_chinese_conclusion_count"] == 0
    assert scan["cards"][0]["conclusion_missing_terms"] == []
