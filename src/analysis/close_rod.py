"""Fixed close-only permission comparison; passive references keep separate evidence roles."""
from src.analysis.stage1_domain_heads import analyze_paired_accounts

CANDIDATE = "C_close_ROD_positive"
REQUIRED = ("A_active_research_replay", "B_close_permission", "SOXX", "strong9097")
FAMILIES = (*REQUIRED, CANDIDATE, "additional27baa")


def analyze_accounts(books, daily_calendar_frame, plan):
    for (family, _), book in books.items():
        if family in ("B_close_permission", CANDIDATE):
            if "passive_expansion_audit" in book:
                raise ValueError("new candidate requires its own native validation")
        else:
            audit = book.get("passive_expansion_audit", {})
            if audit.get("status") != "passed" or audit.get("new_native_accounts") != 0:
                raise ValueError("frozen reference needs explicit passive-only evidence")
    return analyze_paired_accounts(books, daily_calendar_frame, plan,
        candidate_family=CANDIDATE, required_comparators=REQUIRED, expected_families=FAMILIES)
