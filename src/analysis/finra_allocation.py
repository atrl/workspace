"""Frozen seven-family FINRA information comparison, using native account reports."""
from src.analysis.stage1_domain_heads import analyze_paired_accounts


CANDIDATE = "C_FINRA_short_mix"
REQUIRED = ("A_active_research_replay", "B_shared", "B_FINRAcontext", "SOXX", "strong9097")
FAMILIES = (*REQUIRED, CANDIDATE, "additional27baa")


def analyze_accounts(books, daily_calendar_frame, plan):
    return analyze_paired_accounts(books, daily_calendar_frame, plan,
        candidate_family=CANDIDATE, required_comparators=REQUIRED,
        expected_families=FAMILIES)
