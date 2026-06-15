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

