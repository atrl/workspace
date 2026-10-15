"""Validate saved official web-tool captures; never claim publisher HTML bytes."""
import hashlib,html,json,re
from pathlib import Path
import pandas as pd
from src.data.cpi_release_calendar import parse_header
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder


def parse_rendered_header(response,url):
    cleaned=re.sub(r'(?m)^L\d+:\s*','',response)
    return parse_header(('<meta charset="utf-8"><pre>'+html.escape(cleaned)+'</pre>').encode('utf-8'),url)


def validate_captures(directory,plan,calendar):
    files=sorted(Path(directory).glob('*.json'));raw={p.name:p.read_bytes() for p in files};parsed={name:json.loads(value) for name,value in raw.items()}
    if not files or len(files)>plan['budgets']['maximum_total_operations']:raise ValueError('capture operation budget exceeded/empty')
    indexes=[(name,value) for name,value in parsed.items() if value['operation']=='open_index']
    if not 1<=len(indexes)<=plan['budgets']['maximum_index_operations']:raise ValueError('index operation budget invalid')
    index_text='\n'.join(x['raw_tool_response'] for _,x in indexes)
    if 'https://www.bls.gov/bls/news-release/cpi.htm' not in index_text:raise ValueError('official index identity absent')
    captures=[];events=[];excerpts={};counts={};urls=set()
    for name,value in parsed.items():
        response=value['raw_tool_response']
        if not isinstance(response,str):raise ValueError('capture is not unmodified raw tool response text')
        reference=re.search(r'cite([^]+)',response)
        if not reference:raise ValueError('tool reference not exposed')
        stamp=pd.Timestamp(value['observed_at_utc'])
        if stamp.tzinfo is None:raise ValueError('capture observation UTC unknown')
        crawl=re.search(r'Crawled:\s*([^;\n]+)',response)
        date_label=crawl[1] if crawl else 'unknown'
        if value['operation']=='open_index':
            url=plan['index_url'];requested=url
        else:
            url_match=re.search(r'\(https://www\.bls\.gov/news\.release/archives/cpi_\d{8}\.htm\)',response.splitlines()[0])
            if not url_match:raise ValueError('resolved official archive URL absent')
            url=url_match[0][1:-1];requested=plan['index_url']+' [official link '+str(value['index_link_id'])+']'
            counts[url]=counts.get(url,0)+1
            if counts[url]>plan['budgets']['maximum_operations_per_release']:raise ValueError('release operation budget exceeded')
            context=[line for line in index_text.splitlines() if re.search(rf'cite{value["index_link_id"]}†',line)]
            if len(context)!=1:raise ValueError('missing/ambiguous official index context')
            heading=re.search(r'([A-Za-z]+ \d{4}) Consumer Price Index',context[0])
            if not heading:raise ValueError('index observation-month context absent')
            # Retain raw line-numbered capture; remove renderer line prefixes only for parsing.
            event=parse_rendered_header(response,url)
            if str(pd.Period(heading[1],freq='M'))!=event['observation_month']:raise ValueError('index/header observation month differs')
            release=pd.Timestamp(event['release_at']);day=release.tz_localize(None).normalize()
            if not pd.Timestamp('2019-01-01')<=day<=pd.Timestamp('2022-12-31'):raise ValueError('release outside actual release-year range')
            event['is_trading_session']=day in calendar
            event['decision_at']=str(day.tz_localize('America/New_York')+pd.Timedelta(hours=9,minutes=46)) if event['is_trading_session'] else None
            if event['is_trading_session'] and not release<=pd.Timestamp(event['decision_at']):raise ValueError('release later than actual trading decision')
            beginning=response.index('Transmission of material in this release is embargoed until')
            month_end=re.search(r'CONSUMER PRICE INDEX[^\n]*',response[beginning:],re.I)
            if not month_end:raise ValueError('complete raw embargo-to-month excerpt missing')
            verbatim=response[beginning:beginning+month_end.end()]
            excerpts[name]={'verbatim_embargo_to_month':verbatim,'verbatim_index_context':context[0]}
            event.pop('source_sha256',None)
            event.update(capture_name=name,response_capture_sha256=hashlib.sha256(raw[name]).hexdigest(),index_context_capture=indexes[0][0],tool_reference=reference[1],observation_utc=stamp.tz_convert('UTC').isoformat(),capture_type='web_tool_rendered_text_not_publisher_HTML')
            events.append(event)
        urls.add(url)
        captures.append({'capture_name':name,'requested_official_route':requested,'resolved_url_as_exposed':url,'operation':value['operation'],'tool_reference':reference[1],
            'observation_utc':stamp.tz_convert('UTC').isoformat(),'observation_time_role':'operation_start_UTC','exposed_crawl_or_cache_date':date_label,
            'response_capture_sha256':hashlib.sha256(raw[name]).hexdigest(),'raw_response_text_sha256':hashlib.sha256(response.encode('utf-8')).hexdigest(),'bytes':len(raw[name]),
            'hash_scope':'exact UTF8 serialized capture JSON bytes, not publisher HTML; raw-response hash separately named'})
    table=pd.DataFrame(events)
    if len(urls)>49 or len(table)!=48 or table.release_date.nunique()!=48:raise ValueError('required48 distinct events not obtained')
    if set(table.observation_month)!=set(str(x) for x in pd.period_range('2018-12','2022-11',freq='M')):raise ValueError('observation coverage/boundaries incomplete')
    if table.groupby(table.release_date.str[:4]).size().to_dict()!={str(y):12 for y in range(2019,2023)}:raise ValueError('actual release-year coverage incomplete')
    mapping=plan['calendar_mapping']
    trading=table.loc[table.is_trading_session]
    if len(trading)!=mapping['same_day_trading_events'] or trading.groupby(trading.release_date.str[:4]).size().to_dict()!=mapping['trading_events_by_release_year']:raise ValueError('same-day trading event counts differ')
    if sorted(table.loc[~table.is_trading_session,'release_date'])!=mapping['nontrading_release_dates']:raise ValueError('unexpected nontrading release mapping')
    gate={'passed':True,'same_day_trading_events':len(trading),'trading_events_by_release_year':mapping['trading_events_by_release_year'],'nontrading_release_dates':mapping['nontrading_release_dates'],'events':48,'distinct_urls':len(urls),'operations':len(files),'index_operations':len(indexes),'maximum_release_operations':max(counts.values()),
        'actual_release_year_counts':table.groupby(table.release_date.str[:4]).size().to_dict(),'capture_format':'official web-tool rendered text, NOT publisher HTML bytes','qualification':plan['qualification']}
    return raw,excerpts,pd.DataFrame(captures),table.sort_values('release_date').reset_index(drop=True),gate


def run(*,root,card,plan,capture_directory):
    from qlib.workflow import R
    from qlib.data import D
    import yaml
    digest=hashlib.sha256((card/'plan.json').read_bytes()).hexdigest()
    if (card/'summary.json').exists():return json.loads((card/'summary.json').read_text())
    calendar=pd.DatetimeIndex(D.calendar(start_time='2019-01-01',end_time='2022-12-31',freq='day'))
    raw,excerpts,captures,events,gate=validate_captures(capture_directory,plan,calendar)
    tracking=card/'qlib_recorder'
    with R.start(experiment_name=plan['experiment'],recorder_name='official_rendered_text_corpus',uri=str(tracking)):
        rec=R.get_recorder();rec.log_params(plan_sha256=digest,new_fits=0,new_accounts=0)
        rec.save_objects(artifact_path='inputs',plan=plan)
        rec.save_objects(artifact_path='sources',raw_tool_captures=raw,verbatim_excerpts=excerpts,capture_metadata=captures,event_manifest=events,gate=gate,execution_workflow=Path(__file__).read_text(),header_parser=(root/'src/data/cpi_release_calendar.py').read_text())
        policy={'policy':'slim_card_recorder_first','workflow_config':'workflow.yaml','recorder_only_artifacts':['sources/raw_tool_captures','sources/verbatim_excerpts']}
        summary={'experiment':plan['experiment'],'plan_sha256':digest,'decision':'official_rendered_text_calendar_qualified_for_separate_information_run','source_gate':gate,'objective_achieved':False,'new_statistical_fits':0,'new_allocator_fits':0,'new_native_accounts':0,'provider_uri':str(root/plan['provider_uri']),'region':'us','recorder_id':rec.id,'experiment_id':str(rec.experiment_id),'recorder_uri':str(tracking),'recorder_local_dir':str(tracking/str(rec.experiment_id)/rec.id),'artifact_policy':policy,'old_attempt_preserved':plan['old_attempt'],'next_stage':plan['next_stage']}
        rec.save_objects(artifact_path='analysis',summary=summary);artifacts=Path(summary['recorder_local_dir'])/'artifacts';rec.log_metrics(registered_artifact_count=sum(x.is_file() for x in artifacts.rglob('*')),qualified_events=len(events))
    captures.to_csv(card/'capture_metadata.csv',index=False);events.to_csv(card/'event_manifest.csv',index=False);atomic_json(gate,card/'source_gate.json')
    atomic_json({'recorder_id':rec.id,'entries':[{'path':str(p.relative_to(card)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(artifacts.rglob('*')) if p.is_file()]},card/'verification_manifest.json')
    binding=json.loads((root/plan['old_attempt']['card']/'nas_binding.json').read_text());binding.pop('evidence');atomic_json(binding,card/'nas_binding.json')
    (card/'workflow.yaml').write_text(yaml.safe_dump({'experiment':plan['experiment'],'provider_uri':summary['provider_uri'],'recorder_uri':str(tracking),'artifact_policy':policy},sort_keys=False))
    summary['nas_evidence']=archive_nas_recorder(card/'nas_binding.json',workspace_root=root,card=card,recorder_id=rec.id,experiment_id=str(rec.experiment_id));atomic_json(summary,card/'summary.json');return summary
