"""One synthetic contract; no real announcement corpus or market data is read."""

import hashlib

import pandas as pd

from src.data.calendar_knowledge import build_knowledge_ledger, canonical_visible_text


def test_cached_announcement_ledger_keeps_knowledge_time_and_venue_boundaries():
    ice_scope = "Our cash equity markets: New York Stock Exchange and NYSE Arca."
    nasdaq_scope = "Nasdaq U.S. equities markets."
    sectioned = (f"<p>Nov 01, 2023</p><p>{ice_scope}</p><table>"
                 "<tr><td>2023 Holidays</td></tr>"
                 "<tr><td>Christmas Day</td><td>Monday, December 25</td></tr>"
                 "<tr><td>2024 Holidays</td></tr>"
                 "<tr><td>New Year's Day</td><td>Monday, January 1</td></tr>"
                 "<tr><td>Christmas Day</td><td>Wednesday, December 25</td></tr></table>"
                 "<p>Each market will close early at 1:00 p.m. on Wednesday, July 3, 2024.</p>").encode()
    columns = (f"<p>Nov 02, 2023</p><p>{ice_scope}</p><table>"
               "<tr><td>HOLIDAY</td><td></td><td>2024</td><td></td><td>2025</td></tr>"
               "<tr><td>New Year's Day</td><td></td><td>Monday, January 1</td>"
               "<td></td><td>Wednesday, January 1</td></tr>"
               "<tr><td>Christmas Day</td><td></td><td>Wednesday, December 25</td>"
               "<td></td><td>Thursday, December 25 (Christmas holiday observed)</td></tr></table>"
               "<p>Each market will close early at 1:00 p.m. (1:15 p.m. for eligible options) "
               "on Friday, November 29, 2024 and Friday, November 28, 2025.</p>").encode()
    close_sentence = "Nasdaq U.S. equities markets will be closed on Wednesday, July 3, 2024."
    open_sentence = "Nasdaq U.S. equities markets will be open on Wednesday, July 3, 2024."
    exact_clock = "2024-07-02T12:34:56-04:00"
    closed = f"<p>Jun 30, 2024</p><p>{close_sentence}</p>".encode()
    reopened = f"<p>{exact_clock}</p><p>{open_sentence}</p>".encode()
    tentative_sentence = "Nasdaq U.S. equities markets likely will be closed on Wednesday, July 3, 2024."
    tentative = f"<p>Jul 01, 2024</p><p>{tentative_sentence}</p>".encode()
    same_clock_closed = f"<p>{exact_clock}</p><p>{close_sentence}</p>".encode()
    bodies = {"section": sectioned, "columns": columns, "closed": closed,
              "reopen": reopened, "tentative": tentative, "wrong_venue": closed,
              "bad_hash": sectioned, "same_clock_closed": same_clock_closed}

    def publication(day, anchor):
        return {"precision": "day", "value": day, "anchor": anchor}

    def notice(pub, anchor, kind="temporary_closed"):
        return {"parser_type": "nasdaq_notice", "publication": pub,
                "scope_anchor": anchor, "venues": ["NASDAQ"],
                "events": [{"assertion_kind": kind, "date": "2024-07-03", "anchor": anchor}]}

    exact_pub = {"precision": "timestamp", "value": exact_clock, "anchor": exact_clock}
    specs = {
        "section": {"parser_type": "ice_calendar", "publication": publication("2023-11-01", "Nov 01, 2023"),
                    "scope_anchor": ice_scope, "venues": ["NYSE", "NYSE_ARCA"],
                    "calendar": {"table_index": 0, "years": [2023, 2024], "full_years": [2024]}},
        "columns": {"parser_type": "ice_calendar", "publication": publication("2023-11-02", "Nov 02, 2023"),
                    "scope_anchor": ice_scope, "venues": ["NYSE", "NYSE_ARCA"],
                    "calendar": {"table_index": 0, "years": [2024, 2025], "full_years": [2024, 2025]}},
        "closed": notice(publication("2024-06-30", "Jun 30, 2024"), close_sentence),
        "reopen": notice(exact_pub, open_sentence, "reopening"),
        "tentative": notice(publication("2024-07-01", "Jul 01, 2024"), tentative_sentence),
        "wrong_venue": notice(publication("2024-06-30", "Jun 30, 2024"), close_sentence),
        "same_clock_closed": notice(exact_pub, close_sentence),
    }
    specs["wrong_venue"]["venues"] = ["NYSE_ARCA"]
    specs["bad_hash"] = specs["section"]
    docs = [{"id": key, "native_capture": {"http_status": 200,
             "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "status": "captured"}}
            for key, raw in bodies.items()]
    next(d for d in docs if d["id"] == "bad_hash")["native_capture"]["sha256"] = "0" * 64
    docs.extend([
        {"id": "missing_raw", "native_capture": {"http_status": 200, "sha256": hashlib.sha256(sectioned).hexdigest()}},
        {"id": "sec403", "native_capture": {"http_status": 403, "sha256": None, "status": "not_captured"}},
        {"id": "wire_timeout", "native_capture": {"http_status": None, "sha256": None, "status": "not_captured"}},
    ])
    result = build_knowledge_ledger(docs, bodies, document_specs=specs)
    ledger, status = result["ledger"], result["document_status"].set_index("document_id")
    assert len(status) == len(docs)
    assert set(ledger.document_id) == {"section", "columns", "closed", "reopen", "same_clock_closed"}
    assert status.loc[["missing_raw", "sec403", "wire_timeout", "bad_hash", "tentative", "wrong_venue"], "assertion_count"].sum() == 0
    assert "raw_body_missing" == status.loc["missing_raw", "gap_reason"]
    assert "tentative" in status.loc["tentative", "gap_reason"]
    assert "another venue" in status.loc["wrong_venue", "gap_reason"]

    baselines = ledger[ledger.assertion_kind == "weekday_baseline"]
    assert set(baselines[baselines.document_id == "section"].rule_year) == {2024}
    assert set(baselines.venue) == {"NYSE", "NYSE_ARCA"}
    assert baselines.session_date.isna().all() and baselines.effective_from.dt.tz is None
    assert set(ledger[ledger.assertion_kind == "early_close"].cash_close_time) == {"13:00"}
    assert (ledger[ledger.assertion_kind == "early_close"].schedule_state == "open_short_session").all()

    day_only = ledger[ledger.document_id == "closed"].iloc[0]
    assert pd.isna(day_only.published_at)
    assert day_only.published_date == "2024-06-30"
    assert day_only.knowledge_at.isoformat() == "2024-07-01T00:00:00-04:00"
    exact = ledger[ledger.document_id == "reopen"].iloc[0]
    assert exact.knowledge_at == exact.published_at == pd.Timestamp(exact_clock)
    before = ledger[ledger.knowledge_at <= pd.Timestamp("2024-07-02T09:29:00-04:00")]
    assert set(before[before.venue == "NASDAQ"].assertion_kind) == {"temporary_closed"}
    assert before[before.venue == "NASDAQ"].iloc[0].conflict_status == "none"
    assert before[(before.venue == "NYSE_ARCA") & (before.session_date == "2024-07-03")].iloc[0].assertion_kind == "early_close"
    same_clock = ledger[(ledger.venue == "NASDAQ") & (ledger.knowledge_at == pd.Timestamp(exact_clock))]
    assert set(same_clock.conflict_status) == {"unresolved_same_knowledge_time"}

    for row in ledger.itertuples():
        text = canonical_visible_text(bodies[row.document_id])
        assert text[row.source_span["start"]:row.source_span["end"]] == row.source_text
        assert row.raw_sha256 == hashlib.sha256(bodies[row.document_id]).hexdigest()
    assert not any(name in ledger for name in ["tom", "first_three", "last_day", "feature", "return"])

    # The outer date is the closure; the parenthesis is a nominal holiday.
    observed = (f"<p>Nov 01, 2019</p><p>{ice_scope}</p><table>"
                "<tr><td>2020 Holidays</td></tr><tr><td>Independence Day</td>"
                "<td>Friday, July 3 (July 4 holiday observed)</td></tr>"
                "<tr><td>2021 Holidays</td></tr><tr><td>Independence Day</td>"
                "<td>Monday, July 5 (July 4 holiday observed)</td></tr>"
                "<tr><td>2022 Holidays</td></tr><tr><td>Christmas Day</td>"
                "<td>Sunday, December 25 (Observed Monday, December 26)</td></tr></table>"
                "<p>Each market will close early at 1:00 p.m. on Thursday, December 24, 2020.</p>").encode()
    regressions = {
        "observed": observed,
        "extra_nominal_date": observed.replace(b"July 4 holiday observed)",
                                               b"July 4 holiday observed and July 6)", 1),
        "missing_leading_weekday": observed.replace(b"Friday, July 3 (", b"July 3 (", 1),
    }
    observed_spec = {"parser_type": "ice_calendar", "scope_anchor": ice_scope,
                     "venues": ["NYSE", "NYSE_ARCA"],
                     "publication": publication("2019-11-01", "Nov 01, 2019"),
                     "calendar": {"table_index": 0, "years": [2020, 2021, 2022], "full_years": []}}
    regression_docs = [{"id": key, "native_capture": {"http_status": 200,
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
                       for key, raw in regressions.items()]
    repair = build_knowledge_ledger(regression_docs, regressions,
                                    document_specs={key: observed_spec for key in regressions})
    admitted = repair["ledger"]
    assert set(admitted.document_id) == {"observed"}
    closed_days = admitted[admitted.assertion_kind == "planned_closed"].session_date
    assert set(closed_days.dt.strftime("%Y-%m-%d")) == {"2020-07-03", "2021-07-05", "2022-12-26"}
    assert len(repair["source_gaps"]) == 2
    assert all("nominal observed-holiday annotation" in gap["reason"] for gap in repair["source_gaps"])
