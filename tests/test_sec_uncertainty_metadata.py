"""One synthetic-only regression for the frozen B0 metadata selection."""
import pickle

import pandas as pd

from src.data.sec_uncertainty_metadata import select_b0_metadata, canonical_cik
from src.workflow.sec_uncertainty_source import metadata_dataset


def test_fixed_anchor_distinct_ciks_and_unknown_identity_no_replacement():
    anchor = {'report_date': '2019-09-30', 'filing_date': '2019-11-25', 'accession': 'anchor-original'}
    late = {'report_date': '2019-06-30', 'filing_date': '2020-01-06', 'accession': 'late-old-amendment'}
    securities = [('A_CLASS1', '111111111', anchor), ('A_CLASS2', '222222222', anchor),
                  ('B', '333333333', anchor), ('LATER_ONLY', '444444444', late)]
    holdings = pd.DataFrame([{**key, 'series_id': 'S000004354', 'ticker': ticker, 'cusip': cusip,
        'isin': '', 'name': ticker, 'country': 'US', 'weight_pct': 25., 'market_value_usd': 1e6}
        for ticker, cusip, key in securities])
    manifest = {'cik': '0001100663', 'series_id': 'S000004354', 'contract_id': 'C000012084',
        'sha256': 'synthetic', 'row_count': 4,
        'filings': [{**anchor, 'form': 'NPORT-P', 'equity_count': 3}, {**late, 'form': 'NPORT-P/A', 'equity_count': 1}]}
    contract = {'fund_cik': '0001100663', 'series_id': 'S000004354', 'contract_id': 'C000012084',
        'declared_raw_holdings_sha256': 'synthetic', 'earliest_snapshot_expected': anchor,
        'metadata_declared_scope': {'holdings_rows': 4, 'filings': 2}}
    mapping = pd.DataFrame([{'ticker': ticker, 'cik': cik, 'identity_status': 'current_candidate_unknown_history',
        'source': 'synthetic', 'cusips': cusip, 'entity_name': ticker}
        for ticker, cik, cusip in [('A_CLASS1', '0000000100', '111111111'), ('A_CLASS2', 100, '222222222'),
                                  ('B', 200, '999999999'), ('LATER_ONLY', 300, '444444444')]])
    calendar = pd.bdate_range('2019-11-25', '2020-01-10')
    def select(frame):
        return select_b0_metadata(holdings, manifest, frame, calendar, contract=contract)
    result = select(mapping)
    assert result['anchor']['accession'] == anchor['accession']
    assert result['anchor']['available_session'] == '2019-11-26'
    assert result['selection'].candidate_cik.tolist() == ['0000000100', '0000000200']
    assert result['selection'].underlying_security_rows.tolist() == [2, 1]
    assert len(result['security_ledger']) == len(result['mapping_gaps']) == 3
    assert not result['selection'].historical_identity_qualified.any()
    assert result['selection'].original_10K_eligible.eq('unknown').all()
    # A conflicting historical CUSIP does not silently replace the locked B
    # candidate with later/current-universe CIK300, nor does it gain qualification.
    assert 'CUSIP_metadata_conflict' in result['mapping_gaps'].loc[result['mapping_gaps'].ticker.eq('B'), 'gap_reasons'].iloc[0]
    missing = select(mapping.loc[mapping.ticker.ne('B')])
    assert not missing['summary']['selection_complete']
    assert missing['selection'].candidate_cik.tolist() == ['0000000100']
    ambiguous = pd.concat([mapping, mapping.loc[mapping.ticker.eq('B')].assign(cik=201)], ignore_index=True)
    blocked = select(ambiguous)
    assert not blocked['summary']['selection_complete']
    assert blocked['security_ledger'].loc[blocked['security_ledger'].ticker.eq('B'), 'candidate_cik'].isna().all()
    wrong_fund = select(mapping.assign(cik=mapping.cik.where(mapping.ticker.ne('B'), '0001100663')))
    assert not wrong_fund['summary']['selection_complete']
    assert canonical_cik(True) is None and canonical_cik(1.25) is None and canonical_cik('0000000000') is None
    assert '0000000300' not in result['candidate_ledger'].candidate_cik.tolist()
    # Only source metadata is wrapped; timestamps are not encoded as float32.
    dataset, frame = metadata_dataset(result['selection'], result['anchor'])
    restored = pickle.loads(pickle.dumps(dataset))
    pd.testing.assert_frame_equal(restored.prepare('all', col_set='meta'), frame)
    assert isinstance(frame.conditional_decision_at.dtype, pd.DatetimeTZDtype)
