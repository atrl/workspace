from pathlib import Path
from decimal import Decimal
import hashlib,json,base64
from datetime import datetime,timezone
import pandas as pd,numpy as np,qlib
from qlib.workflow import R
from qlib.data import D
from src.core.recorder_sources import load_verified_recorder_objects
root=Path('/Users/rziiiii/dev/workspace');card=root/'research/etf_corporate_actions_20261009';out=Path('/tmp/etf-corporate-actions-validation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((card/'summary.json').read_text());p=json.loads((card/'plan.json').read_text());assert sha(card/'plan.json')==s['plan_sha256']=='2127d65862abb8c0b17178ba35a6694ead94af27d11099e4f0ddcc32806492eb'
assert all(sha(root/k)==v for k,v in s['source_sha256'].items())
qlib.init(provider_uri=s['provider_uri'],region='us',kernels=1)
names=['inputs/source_contract.pkl','inputs/parent_bindings.pkl','inputs/native_calendar.pkl','data/source_rows.pkl','data/cash_events.pkl','data/split_events.pkl','analysis/source_differences.pkl','analysis/qualification.pkl','validation/source_receipts.pkl']
objects,identity=load_verified_recorder_objects(card,expected_recorder_id=s['recorder_id'],artifacts=names)
assert objects['inputs/source_contract.pkl']==p
cash=objects['data/cash_events.pkl'];splits=objects['data/split_events.pkl'];rows=objects['data/source_rows.pkl'];cal=objects['inputs/native_calendar.pkl'];binding=objects['inputs/parent_bindings.pkl'];diff=objects['analysis/source_differences.pkl'];qualification=objects['analysis/qualification.pkl']
assert pd.DatetimeIndex(D.calendar(freq='day',future=False)).equals(cal) and len(cal)==5021
assert len(cash)==125 and len(splits)==3 and len(diff)==111 and cash.event_id.is_unique
counts=cash.groupby('instrument').size().to_dict();assert counts=={'SOXL':32,'SOXX':93}
assert cash.announcement_at.isna().all() and splits.announcement_at.isna().all()
assert cash.loc[cash.source_evidence_known_by_utc.notna(),'event_id'].tolist()==['SOXL:cash:2014-12-23']
for symbol,frame in rows.items():
 for number,row in enumerate(frame.to_dict('records'),1):
  ex=row['Ex-Date' if symbol=='SOXX' else 'Ex Date'];record=row['Record Date'];pay=row['Payable Date' if symbol=='SOXX' else 'Pay Date'];e=cash.loc[cash.event_id.eq(f'{symbol}:cash:{ex}')].iloc[0]
  assert e.source_row==number and e.ex_date==ex and e.record_date==record and e.pay_date==pay
  if symbol=='SOXX':
   mult=Decimal(3) if ex<'2024-03-07' else Decimal(1);total=Decimal(row['Total Distribution']);assert Decimal(e.original_total_decimal)==total*mult
  else:
   mult=Decimal(1);total=sum(Decimal(0) if row[k]=='—' else Decimal(row[k]) for k in ('Income Dividend','Short-Term Capital Gain','Long-Term Capital Gain'));assert Decimal(e.original_total_decimal)==total
  assert Decimal(e.original_share_multiplier_decimal)==mult
  if cal[0].date().isoformat()<=ex<=cal[-1].date().isoformat():assert pd.Timestamp(ex) in cal and e.reference_calendar_status=='native_session'
  else:assert e.reference_calendar_status=='outside_reference_range'
assert cash.loc[cash.reference_calendar_status.eq('outside_reference_range'),'instrument'].value_counts().to_dict()=={'SOXX':18}
assert sum(Decimal(v)==0 for v in cash.original_total_decimal)==14
pending=cash[cash.payable_status_at_market_as_of.eq('unpaid_receivable_not_spendable')];assert pending.event_id.tolist()==['SOXL:cash:2026-09-22'];assert pending.pay_date.iloc[0]=='2026-09-29' and Decimal(pending.original_total_decimal.iloc[0])==Decimal('.09290')
missing=diff[diff.status.eq('declared_vendor_missing_event')];assert missing[['instrument','ex_date']].values.tolist()==[['SOXX','2017-09-26']];assert Decimal(missing.issuer_original_total_decimal.iloc[0])==Decimal('.498528')
for e in diff.to_dict('records'):
 if e.futu_amount_decimal is not None:assert Decimal(e.difference_decimal)==Decimal(e.issuer_original_total_decimal)-Decimal(e.futu_amount_decimal)
assert diff.status.value_counts().to_dict()=={'within_declared_display_precision':70,'exact_amount_match':38,'declared_source_difference':2,'declared_vendor_missing_event':1}
archive=root/p['source_archive']['destination'];assert len([f for f in archive.rglob('*') if f.is_file()])==50
for f in p['source_archive']['files']:
 target=archive/f['archive_relative_path'];assert sha(target)==f['sha256'] and target.stat().st_size==f['bytes']
provider=Path(s['provider_uri']);assert {str(f.relative_to(provider)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in provider.rglob('*') if f.is_file()}==binding['provider_file_identity']
for name,audit in binding['parent_bindings'].items():
 contract=p['parent_references'][name];pc=root/contract['card'];assert sha(pc/'summary.json')==contract['summary_sha256'];assert sha(pc/contract['manifest_name'])==contract.get('manifest_sha256',contract.get('verification_manifest_sha256'))
 for f in audit['artifacts'].values():assert sha(f['local_path'])==f['sha256']
html=(archive/'soxl/independent_2014/issuer_distributions_20141224.html').read_bytes();assert base64.b32encode(hashlib.sha1(html).digest()).decode()==p['sources']['SOXL']['historical2014_source']['cdx_sha1_base32']
assert qualification==json.loads((card/'qualification.json').read_text())
nb=json.loads((card/'etf_corporate_actions_20261009.ipynb').read_text());code=[c for c in nb['cells'] if c['cell_type']=='code'];assert len(code)==3 and all(c['execution_count'] is not None for c in code);assert not any(o['output_type']=='error' for c in code for o in c.get('outputs',[]))
with R.uri_context(s['recorder_uri']):
 rec=R.get_recorder(recorder_id=s['recorder_id'],experiment_id=s['experiment_id']);assert rec.status=='FINISHED';assert (Path(rec.get_local_dir())/'artifacts/card/summary.json').read_bytes()==(card/'summary.json').read_bytes()
a={'status':'passed','audit_completed_at_utc':datetime.now(timezone.utc).isoformat(),'recorder_id':s['recorder_id'],'native_calendar_rows':len(cal),'cash_rows':len(cash),'cash_rows_by_instrument':counts,'split_rows':len(splits),'source_files_byte_exact':50,'all_decimal_cash_reconstructed_independently':True,'zero_rows_retained':14,'outside_calendar_rows_retained':18,'source_difference_counts':diff.status.value_counts().to_dict(),'2014_original_archive_sha1_matches_cdx':True,'only2014_knownby_after_ex':True,'unknown_announcement_at_all_null':True,'unpaid_terminal_receivable_nonspendable':True,'source_module_hashes_unchanged':True,'parent_sources_and_provider_unchanged':True,'executed_notebook_code_cells':len(code),'notebook_errors':0,'model_fits':0,'portfolio_replays':0,'new_broker_calls':0,'historical_vintage_or_investment_return_certified':False}
(out/'independent_audit.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');print(json.dumps(a,ensure_ascii=False,indent=2))
