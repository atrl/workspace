from src.data.fomc_schedule_source import release_links,next_meeting


def test_publication_date_is_distinct_and_next_meeting_can_cross_month():
    html=b'<p>Minutes (Released October 11, 2023): <a href="/monetarypolicy/fomcminutes20230920.htm">HTML</a></p>'
    row=release_links(html,'https://www.federalreserve.gov/')[0]
    assert row['minutes_meeting_end']=='2023-09-20' and row['release_date']=='2023-10-11'
    parsed=next_meeting('<p>It was agreed that the next meeting of the Committee would be held on Tuesday–Wednesday, October 31–\u200dNovember 1, 2023.</p>')
    assert parsed['next_meeting_start']=='2023-10-31' and parsed['next_meeting_end']=='2023-11-01'
    assert next_meeting('<p>An emergency meeting occurred today.</p>')['parse_status']=='no_next_meeting_sentence'
