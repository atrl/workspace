from pathlib import Path
import json,hashlib,gzip,shutil
from datetime import datetime,timezone
import qlib
from qlib.workflow import R
ROOT=Path('/Users/rziiiii/dev/workspace');CARD=ROOT/'research/etf_corporate_actions_20261009';OUT=Path('/tmp/etf-corporate-actions-validation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
s=json.loads((CARD/'summary.json').read_text());p=json.loads((CARD/'plan.json').read_text());assert sha(CARD/'plan.json')==s['plan_sha256']=='2127d65862abb8c0b17178ba35a6694ead94af27d11099e4f0ddcc32806492eb';assert all(sha(ROOT/k)==v for k,v in s['source_sha256'].items())
qlib.init(provider_uri=s['provider_uri'],region='us',kernels=1)
with R.uri_context(s['recorder_uri']):
 rec=R.get_recorder(recorder_id=s['recorder_id'],experiment_id=s['experiment_id']);assert rec.status=='FINISHED';artifact=Path(rec.get_local_dir())/'artifacts'
 assert not (artifact/'execution/original_execution').exists()
 for name in ('etf_corporate_actions_20261009.ipynb','summary.json','conclusion.md','qualification.json','source_differences_summary.json','verification_manifest.json'):
  rec.log_artifact(str(CARD/name),artifact_path='execution/original_execution')
  assert (artifact/'execution/original_execution'/name).read_bytes()==(CARD/name).read_bytes()
 original_nb=sha(CARD/'etf_corporate_actions_20261009.ipynb')
 conclusion='''# ETF 公司行动数据资格\n\n离线资格通过，状态为 `usable_under_declared_source_contract`。保留 125 条现金分配（SOXX 93：79 正值、14 零值；SOXL 32）和 3 次拆股。50 个冻结源文件逐字节封存，原生 Qlib 参考日历为 2006-10-10～2026-09-25 的 5,021 日。SOXX 18 条更早事件完整保留并标明超出参考范围，没有补造日历或价格。\n\n111 个正现金事件的 Futu 对照保留 38 个精确同值、70 个声明显示精度内差异、2 个已登记金额差异和 1 个 SOXX 2017-09-26 缺失事件。SOXX 历史金额按已生效 2024 年 3:1 恢复原始份额；SOXL 逐行保留发行人原始份额金额。2014 年当时发行人原文直接支持 SOXL 每份额 0.00532，原文 SHA 与 CDX digest 相符，旧 Futu 0.00133 矛盾原样保留。\n\n2014-12-24 23:42:56 UTC 存档晚于 12-23 除息日；它只证明该时点之前已有原文，`announcement_at` 仍为空，不能倒填为除息日前已知。SOXL 2026-09-22 除息、09-29 付款在 09-25 截止日仍为不可支用应收。真实账户到账、预扣税及盘中可用时间没有认证。原始 HTTP、失败响应和浏览器 DOM 表格转存保持不同来源身份。\n\n97 项 focused 测试通过；独立 Recorder 审计复核全部 Decimal 金额、来源、事件/日历和未付应收。旧父证据与 provider 的 hash 未变。审计脚本首次有字典访问笔误，失败副本已保留，修正后审计通过；正式资格 notebook 只执行一次。\n\n不晋升。本卡零模型拟合、零策略回放、零收益评价、零新行情请求；这是事件源和份额时点资格，不是历史 vintage、真实含息投资回报或稳定超额认证。下一步须另卡登记原始份额、ex-date 应收和 pay-date 现金自融资账本。\n\nRecorder：582a0c7d39964e8b860ebe33941c5dd2。\n'''
 (CARD/'conclusion.md').write_text(conclusion)
 nb=json.loads((CARD/'etf_corporate_actions_20261009.ipynb').read_text());removed=0
 for c in nb['cells']:
  if c['cell_type']=='code':
   outputs=c.get('outputs',[]);kept=[o for o in outputs if not (o.get('output_type')=='stream' and o.get('name')=='stderr')];removed+=len(outputs)-len(kept);c['outputs']=kept
   if 'display(Markdown((CARD / "conclusion.md").read_text()))' in ''.join(c['source']):
    for o in c['outputs']:
     if o.get('output_type')=='display_data' and 'text/markdown' in o.get('data',{}):o['data']['text/markdown']=conclusion.splitlines(keepends=True)
 nb['cells'][-1]['source']=conclusion.splitlines(keepends=True)
 write(CARD/'etf_corporate_actions_20261009.ipynb',nb)
 audit=json.loads((OUT/'independent_audit.json').read_text())
 verification={'status':'passed','focused_tests':{'passed':97,'warnings':4,'log':'validation/logs/focused.log.gz'},'independent_audit':audit,'original_executed_notebook_sha256':original_nb,'qualification_notebook_execution_count':1,'notebook_stderr_outputs_removed_after_original_archive':removed,'original_execution_artifacts':'execution/original_execution','no_recorder_resume_during_finalization':True,'no_new_source_capture':True,'financial_results_or_source_amounts_changed':False,'implementation_sha256':s['source_sha256']}
 write(CARD/'verification.json',verification)
 s['verification']='verification.json';s['verification_manifest']='verification_manifest.json';s['sidecars']=list(dict.fromkeys([*s['sidecars'],'verification.json','verification_manifest.json']));s['verification_summary']={'focused_passed':97,'independent_recorder_audit_passed':True,'native_calendar_rows':5021,'source_files':50,'all_cash_rows_decimal_verified':125,'outside_calendar_retained':18,'source_receipts_retained':21,'exact2014_archived_primary_rate_verified':True,'historical_exante_timing_certified':False,'provider_and_parents_unchanged':True};s['finalization_note']='Original execution retained; synchronized Chinese conclusion andverification only. No model/strategy/market-source rerun.'
 write(CARD/'summary.json',s)
 for name in ('focused.log','notebook_execution.log','independent_audit.log','independent_audit_first_failed.log'):
  target=OUT/(name+'.gz');target.write_bytes(gzip.compress((OUT/name).read_bytes(),mtime=0));rec.log_artifact(str(target),artifact_path='validation/logs')
 for name in ('independent_audit.py','independent_audit_first_failed.py','finalize_card.py'):
  rec.log_artifact(str(OUT/name),artifact_path='validation/helpers')
 rec.log_artifact(str(OUT/'independent_audit.json'),artifact_path='validation')
 for name in ('tests/test_etf_corporate_actions.py','tests/test_etf_corporate_actions_workflow.py'):
  target=OUT/(Path(name).name+'.gz');target.write_bytes(gzip.compress((ROOT/name).read_bytes(),mtime=0));rec.log_artifact(str(target),artifact_path='validation/test_sources')
 for name in ('etf_corporate_actions_20261009.ipynb','summary.json','conclusion.md','verification.json'):
  rec.log_artifact(str(CARD/name),artifact_path='card');assert (artifact/'card'/name).read_bytes()==(CARD/name).read_bytes()
 # Final manifest is self-excluding; both copies are the same byte sequence.
 excluded={CARD/'verification_manifest.json',artifact/'validation/verification_manifest.json'}
 paths=set(f for f in CARD.rglob('*') if f.is_file() and f not in excluded)
 paths.update(f for f in (ROOT/p['source_archive']['destination']).rglob('*') if f.is_file())
 code=['src/data/etf_corporate_actions.py','src/workflow/etf_corporate_actions.py','tests/test_etf_corporate_actions.py','tests/test_etf_corporate_actions_workflow.py']
 paths.update(ROOT/name for name in code)
 entries=[{'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(paths)]
 manifest={'schema_version':'etf_event_verification_manifest_v1','experiment':s['experiment'],'recorder_id':rec.id,'recorder_status':'FINISHED','plan_sha256':s['plan_sha256'],'source_sha256':s['source_sha256'],'cash_event_rows':125,'split_event_rows':3,'native_calendar_rows':5021,'source_archive_files':50,'focused_passed':97,'independent_audit_passed':True,'entries':entries,'entry_count':len(entries),'self_reference_exclusions':[str(f.relative_to(ROOT)) for f in sorted(excluded)],'notebook_final_output_hash':sha(CARD/'etf_corporate_actions_20261009.ipynb'),'original_executed_notebook_sha256':original_nb,'all_recorded_artifact_hashes_read_back':True,'source_capture_kinds_preserved':True,'scope':'Eventsourcequalificationonly;notaccountledger,vintageorinvestmentreturncertification.'}
 write(CARD/'verification_manifest.json',manifest);rec.log_artifact(str(CARD/'verification_manifest.json'),artifact_path='validation');assert (artifact/'validation/verification_manifest.json').read_bytes()==(CARD/'verification_manifest.json').read_bytes()
 for e in entries:assert sha(ROOT/e['path'])==e['sha256'] and (ROOT/e['path']).stat().st_size==e['bytes']
 assert rec.client.get_run(rec.id).info.status=='FINISHED'
 inventory=sorted({f for f in CARD.rglob('*') if f.is_file()}|{f for f in (ROOT/p['source_archive']['destination']).rglob('*') if f.is_file()})
 Path('/tmp/etf-corporate-actions-artifact-paths').write_bytes(b''.join(str(f.relative_to(ROOT)).encode()+b'\0' for f in inventory))
 Path('/tmp/etf-corporate-actions-code-paths').write_bytes(b''.join(n.encode()+b'\0' for n in code))
 print(json.dumps({'status':'finalized','recorder_id':rec.id,'verification_manifest_sha256':sha(CARD/'verification_manifest.json'),'manifest_entries':len(entries),'artifact_inventory_paths':len(inventory),'code_paths':code,'original_notebook_sha256':original_nb,'final_notebook_sha256':sha(CARD/'etf_corporate_actions_20261009.ipynb'),'stderr_outputs_removed':removed},indent=2))
