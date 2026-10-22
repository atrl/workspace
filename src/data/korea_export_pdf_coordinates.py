"""Source-only coordinate extraction for original Korean export PDF tables."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


NUMBER = re.compile(r"^[+△\-−]?[\d,]+(?:\.\d+)?%?$")
AMOUNT_GROWTH = re.compile(r"^([\d,]+(?:\.\d+)?)\(([+△\-−]?[\d.]+)\)$")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str) -> float:
    value = value.strip().replace("%", "").replace(",", "").replace("−", "-")
    value = value.replace("△", "-")
    return float(value)


def _words(page) -> list[tuple]:
    # Some original HWP-to-PDF versions overlay identical Korean glyphs.
    unique = {}
    for item in page.get_text("words"):
        key = (round(item[0], 1), round(item[1], 1), item[4])
        unique[key] = item
    return list(unique.values())


def _month_header(word: str, target_month: str) -> bool:
    text = word.replace("‘", "").replace("’", "").replace("'", "").replace("월", "")
    match = re.search(r"(?<!\d)(\d{2})\.(\d{1,2})(?!\d)", text)
    return bool(match and int(match[1]) == int(target_month[2:4])
                and int(match[2]) == int(target_month[5:7]))


def _rank_row(words: list[tuple], anchor: tuple, target_month: str, page_number: int) -> dict | None:
    x0, y0, x1 = anchor[:3]
    same = [w for w in words if abs(w[1] - y0) <= 3]
    rank = [w for w in same if w[4] == "1" and x0 - 75 <= w[0] < x0]
    if not rank:
        return None
    above = [w for w in words if y0 - 75 <= w[1] <= y0 - 6]
    month_headers = [w for w in above if _month_header(w[4], target_month)]
    if not month_headers:
        return None
    nearby_unit = [w for w in words if y0 - 160 <= w[1] <= y0 + 15 and "백만" in w[4]]
    if not nearby_unit:
        return None
    growth_headers = [w for w in above if "증감률" in w[4]]
    boundary = max((w[0] for w in growth_headers), default=max(w[0] for w in month_headers)) + 72
    right = sorted([w for w in same if x1 + 2 < w[0] <= boundary], key=lambda w: w[0])
    amount_words = [w for w in right if re.fullmatch(r"\d{1,3},\d{3}(?:\([+△\-−]?[\d.]+\))?", w[4])]
    if not amount_words:
        return None
    amount = min(amount_words, key=lambda w: min(
        abs((h[0] + h[2]) / 2 - (w[0] + w[2]) / 2) for h in month_headers))
    if not any(abs((h[0] + h[2]) / 2 - (amount[0] + amount[2]) / 2) <= 45 for h in month_headers):
        return None
    integrated = AMOUNT_GROWTH.fullmatch(amount[4])
    if integrated:
        amount_million = _number(integrated[1])
        growth = _number(integrated[2])
        growth_x = amount[0]
    else:
        amount_million = _number(amount[4])
        following = [w for w in right if w[0] > amount[2] and w[0] - amount[2] < 70]
        growth_word = next((w for w in following if NUMBER.fullmatch(w[4]) and w[4] not in ("△", "+", "-")), None)
        if growth_word is None:
            return None
        growth = _number(growth_word[4])
        between = [w[4] for w in following if w[0] < growth_word[0]]
        if growth >= 0 and any(mark in ("△", "−", "-") for mark in between):
            growth = -growth
        growth_x = growth_word[0]
    return {"kind": "rank_row_million_USD", "page": page_number,
            "amount_100m_USD": amount_million / 100, "amount_original_million_USD": amount_million,
            "growth_pct": growth, "anchor_xy": [round(x0, 1), round(y0, 1)],
            "amount_x": round(amount[0], 1), "growth_x": round(growth_x, 1)}


def _column_table(words: list[tuple], anchor: tuple, page_number: int) -> dict | None:
    x0, y0, x1 = anchor[:3]
    center = (x0 + x1) / 2
    same = [w for w in words if abs(w[1] - y0) <= 3]
    if not any(w[4] in ("구", "구분", "분") and w[0] < x0 for w in same):
        return None
    caption_lines = defaultdict(list)
    for word in words:
        if y0 - 120 <= word[1] <= y0 - 5:
            caption_lines[round(word[1] / 3) * 3].append(word[4])
    captions = [(line_y, " ".join(parts)) for line_y, parts in caption_lines.items()
                if any("품목" in part for part in parts)]
    if captions:
        nearest_caption = max(captions)[1]
        if (any(mark in nearest_caption for mark in ("누계", "상반기", "분기", "연간", "上"))
                or re.search(r"1[~∼\-]\d{1,2}월", nearest_caption)):
            return None
    labels = [w for w in words if w[0] < x0 and y0 + 8 <= w[1] <= y0 + 55]
    amount_label = next((w for w in labels if "수출액" in w[4]), None)
    growth_label = next((w for w in labels if ("증감률" in w[4] or "증가율" in w[4])
                         and amount_label is not None and amount_label[1] + 7 <= w[1] <= amount_label[1] + 30), None)
    if amount_label is None or growth_label is None:
        return None
    nearby = [w for w in words if y0 - 55 <= w[1] <= growth_label[1] + 6]
    if not any("억" in w[4] for w in nearby):
        return None

    def aligned(row_y: float) -> list[tuple]:
        return [w for w in words if abs(w[1] - row_y) <= 3 and NUMBER.fullmatch(w[4])
                and abs((w[0] + w[2]) / 2 - center) <= 15]

    amounts, growths = aligned(amount_label[1]), aligned(growth_label[1])
    if len(amounts) != 1 or len(growths) != 1:
        return None
    amount, growth = _number(amounts[0][4]), _number(growths[0][4])
    return {"kind": "aligned_column_100m_USD", "page": page_number,
            "amount_100m_USD": amount, "growth_pct": growth,
            "anchor_xy": [round(x0, 1), round(y0, 1)],
            "amount_x": round(amounts[0][0], 1), "growth_x": round(growths[0][0], 1)}


def extract_version(path: Path, target_month: str) -> dict:
    import fitz

    candidates = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, 1):
            words = _words(page)
            for anchor in (w for w in words if w[4] == "반도체"):
                for method in (_rank_row, _column_table):
                    candidate = (method(words, anchor, target_month, page_number)
                                 if method is _rank_row else method(words, anchor, page_number))
                    if candidate and candidate["amount_100m_USD"] > 0:
                        candidates.append(candidate)
    by_identity = {(row["kind"], row["page"], *row["anchor_xy"]): row for row in candidates}
    candidates = list(by_identity.values())
    groups = defaultdict(list)
    for candidate in candidates:
        key = (round(candidate["amount_100m_USD"], 1), round(candidate["growth_pct"], 2))
        groups[key].append(candidate)
    status = "unique_current_document_table_pair" if len(groups) == 1 else (
        "no_qualified_table_pair" if not groups else "conflicting_qualified_table_pairs")
    selected_candidates = next(iter(groups.values())) if len(groups) == 1 else []
    selected = next((row for row in selected_candidates if row["kind"] == "rank_row_million_USD"),
                    selected_candidates[0] if selected_candidates else None)
    return {"status": status, "month": target_month, "qualified_table_candidates": candidates,
            "current_document_pair_100m_USD_growth_pct":
            [selected["amount_100m_USD"], selected["growth_pct"]] if selected else None}


def run_full(root: Path) -> dict:
    root = Path(root).resolve()
    card = root / "research/korea_semiconductor_exports_korean_vintage_20261022"
    plan_path = card / "coordinate_full_source_plan.json"
    plan = json.loads(plan_path.read_text())
    index_path = root / plan["source_index_path"]
    if _sha(index_path) != plan["source_index_sha256"]:
        raise ValueError("original Korean PDF source index changed")
    index = json.loads(index_path.read_text())
    items = index["items"]
    if len(items) != plan["expected_PDF_versions"]:
        raise ValueError("original Korean PDF version count changed")
    per_version = []
    for item in items:
        pdf = root / item["path"]
        if _sha(pdf) != item["sha256"]:
            raise ValueError("original Korean PDF bytes changed: " + item["path"])
        result = extract_version(pdf, item["month"])
        per_version.append({"month": item["month"], "version_index": item["version_index"],
                            "source_SHA256": item["sha256"], **result})
    per_month = {}
    for month in sorted({item["month"] for item in per_version}):
        versions = [item for item in per_version if item["month"] == month]
        pairs = {tuple(item["current_document_pair_100m_USD_growth_pct"])
                 for item in versions if item["current_document_pair_100m_USD_growth_pct"]}
        complete = all(item["status"] == "unique_current_document_table_pair" for item in versions)
        per_month[month] = {"status": "current_document_versions_consistent" if complete and len(pairs) == 1
                            else "unresolved_current_document_semantics",
                            "versions": len(versions), "source_pair": list(next(iter(pairs))) if complete and len(pairs) == 1 else None}
    result = {"schema_version": 1, "status": "source_coordinate_geometry_audit_complete_PIT_gap",
              "plan_sha256": _sha(plan_path), "PDF_versions": len(per_version), "months": len(per_month),
              "consistent_months": sum(x["status"] == "current_document_versions_consistent" for x in per_month.values()),
              "per_version": per_version, "per_month": per_month,
              "historical_first_public_vintage_qualified": False,
              "factor_values_qualified": False, "market_outcome_reads": 0, "model_fits": 0, "accounts": 0}
    (card / "coordinate_full_source_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps({key: value for key, value in run_full(Path.cwd()).items()
                      if key not in ("per_version", "per_month")}, indent=2))
