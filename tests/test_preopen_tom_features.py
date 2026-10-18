"""One synthetic contract; no real calendar, timing, ledger or return is loaded."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data.preopen_tom_features import build_preopen_tom_features


def test_preopen_proxy_is_causal_nullable_and_venue_separated():
    zone = "America/New_York"
    as_of = "2025-01-10"
    prefix_start = pd.Timestamp("2021-12-20")
    calendar = pd.bdate_range(prefix_start, as_of)
    calendar = calendar.difference(pd.DatetimeIndex(["2022-06-20", "2024-07-04", "2025-01-01"]))

    def statement(ident, document, kind, year, known_at, session=None, venue="NYSE"):
        event_date = pd.NaT if session is None else pd.Timestamp(session)
        return {"assertion_id": ident, "document_id": document, "venue": venue,
                "assertion_kind": kind, "session_date": event_date,
                "effective_from": pd.Timestamp(f"{year}-01-01") if session is None else event_date,
                "effective_until": pd.Timestamp(f"{year}-12-31") if session is None else event_date,
                "knowledge_at": pd.Timestamp(known_at, tz=zone),
                "version_status": "append_only_first_version", "conflict_status": "none"}

    ledger = pd.DataFrame([
        statement("base22", "old22", "weekday_baseline", 2022, "2021-01-01"),
        statement("closed22", "old22", "planned_closed", 2022, "2021-01-01", "2022-06-20"),
        statement("base24", "annual24", "weekday_baseline", 2024, "2023-12-01"),
        statement("closed24", "annual24", "planned_closed", 2024, "2023-12-01", "2024-07-04"),
        statement("half24", "annual24", "early_close", 2024, "2023-12-01", "2024-11-29"),
        statement("base25", "annual25", "weekday_baseline", 2025, "2024-12-20"),
        statement("closed25", "annual25", "planned_closed", 2025, "2024-12-20", "2025-01-01"),
    ])
    gaps = [{"id": "missing22_amendment", "venues": ["NYSE", "NYSE_ARCA"],
             "affected_calendar_years": [2022], "unknown_from": "2021-10-01T00:00:00-04:00"}]

    def timing_for(*days):
        signals = [calendar[calendar < pd.Timestamp(day)][-1] for day in days]
        return pd.DataFrame({"decision_at": pd.DatetimeIndex(
            [day + " 09:29" for day in days], tz=zone)}, index=pd.DatetimeIndex(signals))

    def run(timing, *, cal=calendar, book=ledger, prefix_gaps=()):
        return build_preopen_tom_features(calendar=cal, timing=timing, ledger=book,
                                         amendment_gaps=gaps, as_of=as_of,
                                         prefix_start=prefix_start, prefix_gaps=prefix_gaps)

    clocks = timing_for("2022-06-20", "2023-06-20", "2024-07-04", "2024-10-02",
                        "2024-11-01", "2024-11-15", "2024-11-29", "2024-12-02",
                        "2024-12-10", "2024-12-31")
    result = run(clocks, prefix_gaps=({"id": "known_prefix_gap", "start": "2024-10-01", "end": "2024-10-01"},))
    by_day = result.set_index("decision_date")
    missing22 = by_day.loc["2022-06-20"]
    assert not missing22.current_plan_known and not missing22.proxy_known
    assert missing22.amendment_gap_reason == "missing22_amendment"
    assert np.isnan(missing22.preopen_tom_proxy_nullable)  # even the old closed cell cannot certify false
    assert not by_day.loc["2023-06-20"].proxy_known
    closed = by_day.loc["2024-07-04"]
    assert closed.first3_branch_known and closed.expected_last_branch_known and closed.proxy_known
    assert closed.preopen_tom_proxy_nullable == 0
    prefix_gap = by_day.loc["2024-10-02"]
    assert not prefix_gap.prefix_available and not prefix_gap.first3_branch_known
    assert prefix_gap.expected_last_branch_known and not prefix_gap.proxy_known
    assert prefix_gap.prefix_gap_reason == "known_prefix_gap"

    first = by_day.loc["2024-12-02"]
    assert first.first3_branch_known and first.first3_branch_value_nullable == 1
    assert not first.expected_last_branch_known and first.proxy_known
    assert first.preopen_tom_proxy_nullable == 1  # true OR unknown
    unknown = by_day.loc["2024-12-10"]
    assert unknown.first3_branch_known and unknown.first3_branch_value_nullable == 0
    assert not unknown.expected_last_branch_known and not unknown.proxy_known
    last = by_day.loc["2024-12-31"]
    assert last.expected_last_branch_value_nullable == 1 and last.preopen_tom_proxy_nullable == 1
    assert last.next_planned_open_after_month_end == pd.Timestamp("2025-01-02")
    assert set(last.horizon_document_ids) == {"annual24", "annual25"}
    assert last.knowledge_max_at == pd.Timestamp("2024-12-20", tz=zone)
    assert by_day.loc["2024-11-29"].current_planned_state == "early_close"
    assert by_day.loc["2024-11-29"].expected_last_branch_value_nullable == 1
    mask_columns = ["first3_branch_known", "expected_last_branch_known", "proxy_known"]
    assert by_day.loc["2024-11-01", mask_columns].equals(by_day.loc["2024-11-15", mask_columns])
    assert by_day.loc["2024-11-01"].preopen_tom_proxy_nullable == 1
    assert by_day.loc["2024-11-15"].preopen_tom_proxy_nullable == 0

    one = timing_for("2024-12-02")
    # Neither membership of d, nor any later realized sessions, can affect this row.
    altered_future = calendar[calendar < "2024-12-02"].union(pd.DatetimeIndex(["2024-12-15", "2025-01-05"]))
    assert_frame_equal(run(one), run(one, cal=altered_future))
    future_close = statement("future_close", "later_notice", "temporary_closed", 2024,
                             "2024-12-02 09:30", "2024-12-02")
    assert_frame_equal(run(one), run(one, book=pd.concat([ledger, pd.DataFrame([future_close])], ignore_index=True)))
    nasdaq_close = dict(future_close, assertion_id="nasdaq_only", venue="NASDAQ",
                        knowledge_at=pd.Timestamp("2024-12-01", tz=zone))
    assert_frame_equal(run(one), run(one, book=pd.concat([ledger, pd.DataFrame([nasdaq_close])], ignore_index=True)))
    known_close = dict(future_close, knowledge_at=pd.Timestamp("2024-12-02 09:28", tz=zone))
    with_close = run(one, book=pd.concat([ledger, pd.DataFrame([known_close])], ignore_index=True)).iloc[0]
    assert with_close.current_plan_known and with_close.proxy_known and with_close.preopen_tom_proxy_nullable == 0
    assert "later_notice" in with_close.current_plan_document_ids
    assert with_close.knowledge_max_at == pd.Timestamp("2024-12-02 09:28", tz=zone)

    unknown_left = run(timing_for("2024-12-31"), prefix_gaps=(
        {"id": "past_missing", "start": "2024-12-01", "end": "2024-12-30"},)).iloc[0]
    assert not unknown_left.first3_branch_known and unknown_left.expected_last_branch_known
    assert unknown_left.proxy_known and unknown_left.preopen_tom_proxy_nullable == 1
    duplicate_version = statement("base24_other", "other24", "weekday_baseline", 2024, "2023-12-01")
    conflicting = run(timing_for("2024-07-04"), book=pd.concat([ledger, pd.DataFrame([duplicate_version])], ignore_index=True)).iloc[0]
    assert not conflicting.current_plan_known and not conflicting.proxy_known
    assert "conflicting_plan" in conflicting.conflict_reason
    bad_version = ledger.copy()
    bad_version.loc[bad_version.assertion_id.eq("base24"), "version_status"] = "unresolved_version_gap"
    version_gap = run(timing_for("2024-07-04"), book=bad_version).iloc[0]
    assert not version_gap.proxy_known and "unqualified_version" in version_gap.version_gap_reason

    # Preserve both kinds of non-executable tail without inventing an opening.
    nat_tail = pd.DataFrame({"decision_at": pd.DatetimeIndex([pd.NaT], tz=zone)},
                           index=pd.DatetimeIndex([as_of]))
    beyond_tail = pd.DataFrame({"decision_at": pd.DatetimeIndex(["2025-01-13 09:29"], tz=zone)},
                              index=pd.DatetimeIndex([as_of]))
    for tail in [nat_tail, beyond_tail]:
        out = run(tail)
        assert len(out) == 1 and not out[mask_columns].any().any()
        assert out.preopen_tom_proxy_nullable.isna().all() and not out.account_eligible.any()
        assert out.original_decision_at.dtype == tail.decision_at.dtype
    assert result.account_eligible.isna().all() and not result.historical_PIT_qualified.any()
    cited = result.knowledge_max_at.notna()
    assert (result.loc[cited, "knowledge_max_at"] <= result.loc[cited, "original_decision_at"]).all()
    assert "qualification_status" not in result and "current_date_in_observed_calendar" not in result

    with pytest.raises(ValueError, match="signal identity"):
        run(pd.concat([one, one]))
    duplicated_decision = pd.concat([one, one])
    duplicated_decision.index = pd.DatetimeIndex(["2024-11-28", "2024-11-29"])
    with pytest.raises(ValueError, match="duplicate nonempty decision"):
        run(duplicated_decision)
    wrong_clock = one.copy(); wrong_clock["decision_at"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="09:29"):
        run(wrong_clock)
    wrong_timezone = one.copy(); wrong_timezone["decision_at"] = wrong_timezone.decision_at.dt.tz_convert("UTC")
    with pytest.raises(ValueError, match="timezone"):
        run(wrong_timezone)
    wrong_identity = one.copy(); wrong_identity.index -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="last d-prior core session"):
        run(wrong_identity)
