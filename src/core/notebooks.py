from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import stable_hash


NOTEBOOK_CARD_SCAN_SCHEMA_VERSION = "core.notebook_card_scan.v1"
NOTEBOOK_CARD_MANIFEST_SCHEMA_VERSION = "core.notebook_card_manifest.v1"
REQUIRED_CONCLUSION_TERMS = ("中文结论", "做了什么", "核心指标", "regime 差异", "是否晋升", "风险", "下一步")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _read_notebook_conclusion(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return ""
    cells = payload.get("cells", [])
    non_empty = [
        cell
        for cell in cells
        if isinstance(cell, dict) and "".join(cell.get("source", [])).strip()
    ]
    if not non_empty:
        return ""
    last = non_empty[-1]
    if last.get("cell_type") != "markdown":
        return ""
    return "".join(last.get("source", []))


def _conclusion_missing_terms(path: Path | None) -> list[str]:
    if path is None:
        return list(REQUIRED_CONCLUSION_TERMS)
    conclusion = _read_notebook_conclusion(path)
    conclusion_lower = conclusion.lower()
    missing: list[str] = []
    for term in REQUIRED_CONCLUSION_TERMS:
        if term == "regime 差异":
            if "regime" in conclusion_lower and "差异" in conclusion:
                continue
        if term not in conclusion:
            missing.append(term)
    return missing


def _card_status(
    *,
    notebook_count: int,
    canonical: bool,
    conclusion_ok: bool,
) -> str:
    if notebook_count == 0:
        return "missing_notebook"
    if notebook_count > 1:
        return "multiple_notebooks"
    if not conclusion_ok:
        return "missing_chinese_conclusion"
    if not canonical:
        return "noncanonical_notebook_name"
    return "ready"


def _scan_surface(root: Path, surface: str) -> list[dict[str, Any]]:
    surface_dir = root / surface
    if not surface_dir.exists():
        return []
    cards: list[dict[str, Any]] = []
    for card_dir in sorted(path for path in surface_dir.iterdir() if path.is_dir()):
        notebooks = sorted(card_dir.glob("*.ipynb"))
        primary = notebooks[0] if notebooks else None
        canonical_name = f"{card_dir.name}.ipynb"
        canonical = primary is not None and primary.name == canonical_name and len(notebooks) == 1
        missing_terms = _conclusion_missing_terms(primary)
        conclusion_ok = not missing_terms
        status = _card_status(
            notebook_count=len(notebooks),
            canonical=canonical,
            conclusion_ok=conclusion_ok,
        )
        sidecars = [
            path
            for path in card_dir.iterdir()
            if path.is_file() and path.suffix.lower() != ".ipynb"
        ]
        cards.append(
            {
                "schema_version": "core.notebook_card.v1",
                "surface": surface,
                "card_id": card_dir.name,
                "card_path": str(card_dir.relative_to(root)),
                "notebook_count": int(len(notebooks)),
                "notebooks": [path.name for path in notebooks],
                "primary_notebook": primary.name if primary is not None else "",
                "primary_notebook_path": str(primary.relative_to(root)) if primary is not None else "",
                "canonical_notebook_name": canonical_name,
                "is_canonical_notebook_name": bool(canonical),
                "chinese_conclusion_ok": bool(conclusion_ok),
                "conclusion_missing_terms": missing_terms,
                "sidecar_count": int(len(sidecars)),
                "status": status,
            }
        )
    return cards


def scan_notebook_cards(
    root: str | Path,
    *,
    surfaces: Iterable[str] = ("research", "production"),
) -> dict[str, Any]:
    repo_root = Path(root)
    cards: list[dict[str, Any]] = []
    for surface in surfaces:
        cards.extend(_scan_surface(repo_root, str(surface)))

    summary = {
        "schema_version": "core.notebook_card_scan.summary.v1",
        "surface_count": len(set(str(surface) for surface in surfaces)),
        "card_count": int(len(cards)),
        "notebook_count": int(sum(card["notebook_count"] for card in cards)),
        "ready_count": int(sum(card["status"] == "ready" for card in cards)),
        "missing_notebook_count": int(sum(card["status"] == "missing_notebook" for card in cards)),
        "multiple_notebook_count": int(sum(card["status"] == "multiple_notebooks" for card in cards)),
        "canonical_notebook_count": int(sum(card["is_canonical_notebook_name"] for card in cards)),
        "noncanonical_notebook_count": int(
            sum(card["notebook_count"] == 1 and not card["is_canonical_notebook_name"] for card in cards)
        ),
        "chinese_conclusion_ok_count": int(sum(card["chinese_conclusion_ok"] for card in cards)),
        "missing_chinese_conclusion_count": int(
            sum(card["status"] == "missing_chinese_conclusion" for card in cards)
        ),
        "status_counts": {
            status: int(sum(card["status"] == status for card in cards))
            for status in sorted({str(card["status"]) for card in cards})
        },
    }
    return _json_ready(
        {
            "schema_version": NOTEBOOK_CARD_SCAN_SCHEMA_VERSION,
            "summary": summary,
            "cards": cards,
        }
    )


def write_notebook_card_manifest(output_dir: str | Path, scan: dict[str, Any]) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": NOTEBOOK_CARD_MANIFEST_SCHEMA_VERSION,
        "summary": scan.get("summary", {}),
        "cards": scan.get("cards", []),
    }
    payload["manifest_hash"] = stable_hash(_json_ready(payload))

    manifest_path = output / "notebook_card_manifest.json"
    csv_path = output / "notebook_cards.csv"
    review_path = output / "notebook_card_review.md"

    manifest_path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "surface",
        "card_id",
        "status",
        "primary_notebook",
        "canonical_notebook_name",
        "is_canonical_notebook_name",
        "chinese_conclusion_ok",
        "sidecar_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for card in payload["cards"]:
            writer.writerow({field: card.get(field, "") for field in fields})

    summary = payload["summary"]
    review_lines = [
        "# Notebook Card Manifest Review",
        "",
        "## 中文结论",
        "",
        f"- 扫描 card 数：{summary.get('card_count', 0)}；主 notebook 数：{summary.get('notebook_count', 0)}。",
        f"- 缺失主 notebook：{summary.get('missing_notebook_count', 0)}；多 notebook card：{summary.get('multiple_notebook_count', 0)}。",
        f"- canonical notebook：{summary.get('canonical_notebook_count', 0)}；非 canonical 命名：{summary.get('noncanonical_notebook_count', 0)}。",
        f"- 中文结论完整：{summary.get('chinese_conclusion_ok_count', 0)}；缺中文结论：{summary.get('missing_chinese_conclusion_count', 0)}。",
        "- CSV/JSON 只是 sidecar；review 时仍以主 notebook 的中文结论为主要证据。",
    ]
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")

    return {
        "schema_version": "core.notebook_card_manifest_write_result.v1",
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path),
        "review_path": str(review_path),
        "manifest_hash": str(payload["manifest_hash"]),
    }
