"""Bounded actual SOXX/SOX structural-source qualification; no price inputs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess

NAME='soxx_sox_structure_source_20261016'

class StructuralFetcher:
    def __init__(self,root,staging='/tmp/soxx_sox_structure_source'):
        self.root=Path(root).resolve();self.card=self.root/'research'/NAME;self.staging=Path(staging);self.staging.mkdir(exist_ok=True,parents=True)
        self.plan=json.loads((self.card/'plan.json').read_text());review=json.loads((self.card/'design_review.json').read_text())
        if review['plan_sha256']!=hashlib.sha256((self.card/'plan.json').read_bytes()).hexdigest() or review['decision']!='approved_secondary_bounded_source_qualification':raise ValueError('unapproved source plan')
        self.path=self.staging/'receipts.json';self.receipts=json.loads(self.path.read_text()) if self.path.exists() else []
    def get(self,name,url):
        for e in self.receipts:
            if e['url']==url:return (self.staging/e['name']).read_bytes(),e
        if len(self.receipts)>=16:raise ValueError('16document/20totalaction budget exhausted')
        p=self.staging/name
        r=subprocess.run(['curl','--compressed','-L','--connect-timeout','10','--max-time','90','-sS','-H','User-Agent: Mozilla/5.0 SOXL research','-w','%{http_code}\n%{url_effective}',url,'-o',str(p)],capture_output=True,text=True)
        body=p.read_bytes() if p.exists() else b'';lines=r.stdout.splitlines();e={'name':name,'url':url,'effective_url':lines[-1] if len(lines)>1 else None,'status':int(lines[0]) if lines and lines[0].isdigit() else 0,'curl_returncode':r.returncode,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat()}
        self.receipts.append(e);self.path.write_text(json.dumps(self.receipts,indent=2));print(e,flush=True);return body,e


def parse_membership_export(payload):
    """Read the actual worksheet columns, never trust the 'Weightings' title."""
    import io,zipfile,xml.etree.ElementTree as ET
    import pandas as pd
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        rows=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row',ns):
            values=[]
            for cell in row.findall('s:c',ns):
                v=cell.find('s:v',ns)
                value=v.text if v is not None else ''
                values.append(strings[int(value)] if cell.get('t')=='s' and value else value)
            while values and values[-1]=='':values.pop()
            if values:rows.append(values)
    frame=pd.DataFrame(rows[1:],columns=rows[0])
    return frame,any('weight' in str(c).lower() for c in frame.columns)


def run(root,staging='/tmp/soxx_sox_structure_source'):
    import os
    import pandas as pd
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder,materialize_nas_evidence
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
    root=Path(root).resolve();staging=Path(staging);card=root/'research'/NAME
    qlib.init(provider_uri=str(root/'data/qlib/soxl'),region='us',kernels=1)
    calendar=D.calendar(start_time='2018-01-01',end_time='2026-10-16')
    previous=json.loads((card/'summary.json').read_text()) if (card/'summary.json').exists() else {}
    if previous.get('membership_parser_version')==2:
        summary=previous
        if not Path(summary['recorder_local_dir']).exists():materialize_nas_evidence(card/'nas_binding.json',card=card)
        R.set_uri(summary['recorder_uri']);rec=R.get_recorder(recorder_id=summary['recorder_id'],experiment_id=summary['experiment_id'])
        raw=rec.load_object('inputs/raw_documents')
        for e in rec.load_object('inputs/document_receipts'):assert hashlib.sha256(raw[e['name']]).hexdigest()==e['sha256']
        return summary
    plan=json.loads((card/'plan.json').read_text());review=json.loads((card/'design_review.json').read_text());ph=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if review['plan_sha256']!=ph or review['decision']!='approved_secondary_bounded_source_qualification':raise ValueError('unapproved source design')
    receipts=json.loads((staging/'receipts.json').read_text());queries=json.loads((card/'search_actions.json').read_text())
    if len(receipts)>16 or len(queries)>4 or len(receipts)+len(queries)>20:raise ValueError('public action budget exceeded')
    raw={e['name']:(staging/e['name']).read_bytes() for e in receipts}
    for e in receipts:
        if hashlib.sha256(raw[e['name']]).hexdigest()!=e['sha256']:raise ValueError('source response changed')
    memberships=[];matrix=[]
    for date in plan['target_rebalances']:
        name='sox_'+date.replace('-','')+'_export';frame,has_weights=parse_membership_export(raw[name])
        frame=frame.rename(columns={'Company Name':'company','Security Symbol':'symbol'});frame['requested_trade_date']=date;frame['body_asof_present']=False;frame['weight_field_present']=has_weights
        memberships.append(frame);matrix.append({'source':name,'target_date':date,'actual_rows':len(frame),'weight_rows':0,'availability':'retrieved_now_historical_query_no_body_asof','decision':'membership_only_not_announced_weight_history'})
    memberships=pd.concat(memberships,ignore_index=True)
    sets={date:set(memberships.loc[memberships.requested_trade_date.eq(date),'symbol']) for date in plan['target_rebalances']}
    a,b=plan['target_rebalances'];changes=pd.DataFrame([{'comparison':a+' to '+b,'only_earlier':';'.join(sorted(sets[a]-sets[b])),'only_later':';'.join(sorted(sets[b]-sets[a])),'shared_names':len(sets[a]&sets[b]),'interpretation':'one_year_membership_difference_not_isolated_rebalance_flow'}])
    holdings_path=root/'data/qlib/soxx_pit_leveraged_training_v1/sec_nport_holdings_history.csv';hm=json.loads(holdings_path.with_name('sec_nport_holdings_manifest.json').read_text())
    if hashlib.sha256(holdings_path.read_bytes()).hexdigest()!=hm['sha256']:raise ValueError('fund holdings source changed')
    holdings=pd.read_csv(holdings_path);parts=[]
    for date in ['2024-09-30','2025-09-30']:
        g=holdings.loc[holdings.report_date.eq(date)];g=g.loc[g.filing_date.eq(g.filing_date.min())].copy();g['source_type']='post_period_SEC_fund_holdings_not_index_announcement';parts.append(g)
        matrix.append({'source':'SOXX_SEC_NPORT','target_date':date,'actual_rows':len(g),'weight_rows':len(g),'availability':g.filing_date.iloc[0],'decision':'fund_weights_only_post_period_disclosure'})
    fund=pd.concat(parts,ignore_index=True)
    partial=pd.read_csv(card/'partial_historical_weights.csv')
    if abs(partial.weight_pct.sum()-100)>1e-8:raise ValueError('rounded chart transcription sum mismatch')
    matrix += [{'source':'Nasdaq_October_2025_research_page4','target_date':'2025-09-30','actual_rows':len(partial),'weight_rows':10,'availability':'publication_month_2025-10_exact_day_unknown','decision':'rounded_top10_plus_rest_post_rebalance_not_full_weights'}, {'source':'Nasdaq_live_FS_SOX','target_date':'2026-09-30','actual_rows':10,'weight_rows':10,'availability':'retrieved_2026-10-16','decision':'current_file_not_2025_search_cache_vintage'}, {'source':'Nasdaq_MonthEnd','target_date':'historical','actual_rows':0,'weight_rows':0,'availability':'login_and_subscription_prompt','decision':'unauthenticated_full_history_not_accessible'}, {'source':'ICE_official_announcements_and_rebalance_pages','target_date':'2024/2025','actual_rows':0,'weight_rows':0,'availability':'links_to_ICE_Index_Platform','decision':'exact_dated_full_weights_not_obtained_within_scope'}]
    parent=root/'research/etf_valuation_source_20261016';ps=json.loads((parent/'summary.json').read_text());R.set_uri(ps['recorder_uri']);pr=R.get_recorder(recorder_id=ps['recorder_id'],experiment_id=ps['experiment_id']);prior_notice=pr.load_object('inputs/raw_payloads')['change.html']
    policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['inputs/*','sources/*']};tracking=card/'qlib_recorder'
    with R.start(experiment_name=NAME,uri=str(tracking)):
        rec=R.get_recorder();rec.save_objects(artifact_path='inputs',plan=plan,search_actions=queries,document_receipts=receipts,raw_documents=raw,prior_SEC_2021_notice=prior_notice,prior_recorder_id=ps['recorder_id'],fund_holdings_manifest=hm)
        rec.save_objects(artifact_path='analysis',memberships=memberships,membership_changes=changes,fund_weights=fund,partial_historical_weights=partial,qualification=pd.DataFrame(matrix),index_identity=json.loads((card/'index_identity.json').read_text()))
        rec.save_objects(artifact_path='sources',module=Path(__file__).read_text())
        summary={'experiment':NAME,'membership_parser_version':2,'supersedes_recorder_id':previous.get('recorder_id'),'correction_reason':'ignore_empty_styled_spreadsheet_footer_rows' if previous else None,'decision':'needs_full_dated_index_weights_and_announcement_clock_secondary_only','public_source_actions':len(receipts)+len(queries),'search_queries':len(queries),'document_requests':len(receipts),'historical_membership_responses':2,'membership_rows':len(memberships),'historical_partial_weight_snapshots':1,'partial_snapshot_individual_weights':10,'post_period_fund_weight_rows':len(fund),'qualified_full_rebalance_weight_sets':0,'qualified_announcement_timestamps':0,'new_fits':0,'new_accounts':0,'new_return_series':0,'objective_achieved':False,'primary_annual_branch_blocked':False,'provider_uri':str(root/'data/qlib/soxl'),'region':'us','market':'SOXX/SOX/ICESEMI source metadata only','as_of':'2026-10-16','qlib_calendar_rows':len(calendar),'recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'plan_sha256':ph,'artifact_policy':policy,'next_information_need':'Full constituent weights for frozen two rebalances with explicitasof+announcement/effective/publication timestamps, through existingauthorizedaccess or independentlydatedofficialfiles. Publicmembershiphistory andmonthlytop10charts are not substitutes; no more fetches inthis20actionbranch.'}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(p.is_file() for p in artifacts.rglob('*')),membership_rows=len(memberships),qualified_full_weight_sets=0)
    memberships.to_csv(card/'historical_membership_responses.csv',index=False);changes.to_csv(card/'membership_changes.csv',index=False);fund.to_csv(card/'disclosed_fund_weights.csv',index=False);pd.DataFrame(matrix).to_csv(card/'source_qualification.csv',index=False)
    binding=json.loads((parent/'nas_binding.json').read_text());binding.pop('evidence',None);(card/'nas_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (card/'verification_manifest.json').write_text(json.dumps({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},indent=2)+'\n')
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));(card/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
