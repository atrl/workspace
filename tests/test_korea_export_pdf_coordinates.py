"""The PDF table selector must not turn cumulative rows into monthly values."""
from src.data.korea_export_pdf_coordinates import _column_table, _rank_row


def _word(x, y, text, width=25):
    return (float(x), float(y), float(x + width), float(y + 10), text, 0, 0, 0)


def test_rank_row_uses_matching_month_not_rightmost_cumulative_column():
    anchor = _word(102, 175, "반도체", 36)
    words = [anchor, _word(62, 175, "1", 8), _word(730, 120, "백만", 24),
             _word(418, 136, "‘21.6", 35), _word(494, 136, "‘21.上", 35),
             _word(438, 151, "증감률", 36), _word(520, 151, "증감률", 36),
             _word(394, 175, "11,159", 42), _word(439, 175, "34.4", 27),
             _word(472, 175, "57,134", 42), _word(516, 175, "21.9", 27)]
    result = _rank_row(words, anchor, "2021-06", 17)
    assert result["amount_original_million_USD"] == 11159
    assert result["growth_pct"] == 34.4


def test_column_table_rejects_cumulative_caption():
    anchor = _word(135, 334, "반도체", 33)
    common = [anchor, _word(78, 334, "구", 10), _word(95, 334, "분", 10),
              _word(76, 355, "수출액", 35), _word(76, 376, "증감률", 35),
              _word(138, 355, "137.9", 28), _word(137, 376, "+21.2", 29)]
    monthly = common + [_word(76, 308, "품목별", 35), _word(115, 308, "수출액(억달러)", 85)]
    assert _column_table(monthly, anchor, 5)["amount_100m_USD"] == 137.9
    cumulative = common + [_word(76, 308, "6월", 18), _word(96, 308, "누계", 25),
                           _word(125, 308, "품목별", 35), _word(164, 308, "수출액(억달러)", 85)]
    assert _column_table(cumulative, anchor, 5) is None
