import pytest
from src.data.amd_guidance_history_intake import report_facts


def test_signed_percent_keeps_printed_precision_and_qualitative_unknown():
    entry={'reported_quarter':'FY2011Q4','report_date':'2012-01-19','receipt':{'resolved_url':'https://ir.amd.com/example','sha256':'x'}}
    prefix='January 19, 2012 GAAP Financial Results Q4-11 Q3-11 Revenue $1.69B $1.69B Non-GAAP Financial Results Current Outlook '
    raw=(prefix+'AMD expects revenue to decrease eight percent, plus or minus three percent, sequentially for the first quarter of 2012. AMD Teleconference').encode()
    f=report_facts(raw,entry)
    assert (f['guide_lower'],f['guide_mid'],f['guide_upper'])==(-11,-8,-5)
    assert f['actual_rounding_half_million']==5 and f['target_quarter']=='FY2012Q1'
    qualitative=(prefix+'AMD expects first quarter of 2012 revenue to be flat to slightly down sequentially. AMD Teleconference').encode()
    assert report_facts(qualitative,entry)['guide_mid'] is None
    with pytest.raises(ValueError,match='nextquarter'):
        report_facts(raw.replace(b'first quarter of 2012',b'second quarter of 2012'),entry)
