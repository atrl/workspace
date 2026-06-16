from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _css_rule(selector: str) -> str:
    css = (ROOT / "web/dashboard-v2/src/styles.css").read_text(encoding="utf-8")
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\}}", css, re.S)
    assert match is not None, f"missing CSS rule for {selector}"
    return match.group("body")


def test_model_delta_strip_renders_before_main_kline_chart():
    body = _css_rule(".model-delta-strip")

    assert re.search(r"\border\s*:\s*0\s*;", body), "model differences must be above the main K-line"


def test_model_delta_strip_stays_compact_for_first_viewport():
    body = _css_rule(".model-delta-strip")

    match = re.search(r"\bmax-height\s*:\s*(?P<height>\d+)px\s*;", body)
    assert match is not None, "model difference strip must have a fixed compact max-height"
    height = int(match.group("height"))
    assert 148 <= height <= 164, "model difference strip must show latest signals without crowding out the K-line"


def test_latest_signal_readouts_stay_visible_and_compact_in_first_viewport():
    css = (ROOT / "web/dashboard-v2/src/styles.css").read_text(encoding="utf-8")
    matrix = re.search(
        r"\.model-delta-strip\s+\.latest-decision-matrix\s*\{(?P<body>.*?)\bmax-height\s*:\s*(?P<height>\d+)px\s*;",
        css,
        re.S,
    )
    assert matrix is not None, "latest signal cards must be visible but height-limited in the compact first-viewport strip"
    assert int(matrix.group("height")) <= 80
    assert re.search(
        r"\.model-delta-strip\s+\.latest-tail-banner\s*\{(?P<body>.*?)\bdisplay\s*:\s*flex\s*;",
        css,
        re.S,
    ), "latest baseline and actor replay tail must stay visible above the K-line"
    assert ".latest-signal-difference-strip" in css


def test_first_viewport_includes_position_and_nav_difference_panes():
    main_ts = (ROOT / "web/dashboard-v2/src/main.ts").read_text(encoding="utf-8")
    lower_body = _css_rule(".lower-charts")

    assert 'id="position-delta-chart"' in main_ts
    assert "Position Delta vs Baseline" in main_ts
    assert 'id="delta-nav-chart"' in main_ts
    assert "Delta NAV vs Baseline" in main_ts
    assert "grid-template-columns" in lower_body


def test_main_kline_keeps_space_and_model_cards_use_first_viewport_grid():
    chart_body = _css_rule(".chart-main")
    css = (ROOT / "web/dashboard-v2/src/styles.css").read_text(encoding="utf-8")

    min_height = re.search(r"\bmin-height\s*:\s*(?P<height>\d+)px\s*;", chart_body)
    assert min_height is not None, "main K-line chart must have a stable minimum height"
    assert int(min_height.group("height")) >= 400, "main K-line must remain readable with first-viewport model differences"
    assert "46vh" in chart_body, "main K-line should leave room for position and NAV panes in the first viewport"
    cards = re.search(
        r"\.model-delta-strip\s+\.model-difference-cards\s*\{(?P<body>.*?)\n\}",
        css,
        re.S,
    )
    assert cards is not None, "first-viewport model cards need a strip-specific layout rule"
    assert re.search(r"\bdisplay\s*:\s*grid\s*;", cards.group("body"))
    assert "grid-template-columns" in cards.group("body")
    assert "overflow-x" not in cards.group("body")
