"""Bounded offline annual financial-source qualification; no outcome reads."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.core.recorder_sources import load_verified_recorder_objects
from src.data.minute_intake import atomic_json

NAME = 'inventory_sales_source_qualification_20261021'
MODULE = 'src/workflow/inventory_sales_source_qualification.py'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract(payloads, manifest, plan):
    selected = {}
    for item in manifest:
        if item.get('status') == 200 and (item.get('parse_valid') is True
            or ('parse_valid' not in item and item.get('reused_parent') is True)):
            selected[int(item['cik'])] = item
    require(len(selected) == plan['source']['complete_ciks'], 'complete CIK inventory differs')
    rows = []; coverage = []
    for cik, item in sorted(selected.items()):
        raw = payloads[item['name']]
        require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'raw payload hash differs')
        doc = json.loads(raw)
        require(int(doc['cik']) == cik and isinstance(doc.get('facts'), dict), 'payload CIK/schema differs')
        count = {field: 0 for field in ('inventory', 'revenue', 'assets')}
        for taxonomy, fields in plan['tags'].items():
            for field, tags in fields.items():
                for tag in tags:
                    definition = doc['facts'].get(taxonomy, {}).get(tag, {})
                    for unit, facts in definition.get('units', {}).items():
                        for fact in facts:
                            row = {'cik': cik, 'taxonomy': taxonomy, 'field': field, 'tag': tag,
                                   'unit': unit, 'payload_name': item['name'], **fact}
                            rows.append(row); count[field] += 1
        coverage.append({'cik': cik, 'entity_name': doc.get('entityName'), **count})
    columns = ['cik', 'taxonomy', 'field', 'tag', 'unit', 'payload_name', 'start', 'end',
               'filed', 'form', 'accn', 'val', 'fy', 'fp']
    raw = pd.DataFrame(rows).reindex(columns=columns)
    for field in ('start', 'end', 'filed'):
        raw[field] = pd.to_datetime(raw[field], errors='coerce')
    raw['numeric'] = pd.to_numeric(raw.val, errors='coerce')
    raw['finite'] = raw.numeric.map(lambda v: bool(math.isfinite(v)))
    duration = (raw.end - raw.start).dt.days
    raw['mechanical_eligible'] = (raw.form.isin(plan['annual']['forms'])
        & raw.unit.isin(plan['annual']['currencies']) & raw.filed.ge(raw.end) & raw.finite
        & ((raw.field.eq('revenue') & duration.between(*plan['annual']['duration_days']))
           | (~raw.field.eq('revenue') & raw.start.isna())))
    raw['original_semantics_verified'] = False
    raw['historical_PIT_qualified'] = False
    eligible = raw.loc[raw.mechanical_eligible]
    keys = ['cik', 'taxonomy', 'field', 'unit', 'start', 'end']
    first = []; conflicts = []
    for key, group in eligible.groupby(keys, dropna=False, sort=True):
        earliest = group.loc[group.filed.eq(group.filed.min())]
        if any(earliest[name].nunique(dropna=False) != 1 for name in ('numeric', 'accn', 'form', 'tag')):
            conflicts.append({'economic_key': str(key), 'reason': 'earliest_value_alias_accession_or_form_conflict'})
            continue
        first.append(earliest.iloc[0].to_dict())
    return raw, pd.DataFrame(coverage), pd.DataFrame(first, columns=raw.columns), pd.DataFrame(conflicts)


def annual_pairs(first, calendar, plan):
    annual = []
    for row in first.loc[first.field.eq('revenue')].to_dict('records'):
        same = first.loc[first.cik.eq(row['cik']) & first.taxonomy.eq(row['taxonomy'])
            & first.unit.eq(row['unit']) & first.end.eq(row['end']) & first.accn.eq(row['accn'])]
        inv = same.loc[same.field.eq('inventory')]
        if len(inv) != 1 or row['numeric'] <= 0 or inv.numeric.iloc[0] < 0:
            continue
        asset = same.loc[same.field.eq('assets')]
        annual.append({**row, 'inventory': inv.numeric.iloc[0],
            'inventory_tag': inv.tag.iloc[0], 'assets': asset.numeric.iloc[0] if len(asset) == 1 else None,
            'joint_filed': max(row['filed'], inv.filed.iloc[0])})
    frame = pd.DataFrame(annual); pairs = []
    if frame.empty:
        return frame, pd.DataFrame()
    for _, group in frame.groupby(['cik', 'taxonomy', 'unit', 'tag', 'form']):
        records = group.sort_values(['end', 'start']).to_dict('records')
        for previous, current in zip(records, records[1:]):
            if ((current['start'] - previous['end']).days != 1
                or not 330 <= (current['end'] - previous['end']).days <= 380
                or previous['inventory'] <= 0 or current['inventory_tag'] != previous['inventory_tag']):
                continue
            filed = max(current['joint_filed'], previous['joint_filed'])
            position = calendar.searchsorted(filed, side='right')
            pairs.append({'cik': current['cik'], 'unit': current['unit'], 'taxonomy': current['taxonomy'],
                'revenue_tag': current['tag'], 'inventory_tag': current['inventory_tag'],
                'fiscal_end': current['end'], 'prior_fiscal_end': previous['end'],
                'accession': current['accn'], 'prior_accession': previous['accn'], 'latest_filed': filed,
                'conditional_available_session': calendar[position] if position < len(calendar) else pd.NaT,
                'calendar_lower_bound_clipped': filed < calendar.min(),
                'original_semantics_verified': False, 'historical_PIT_qualified': False})
    return frame, pd.DataFrame(pairs)


def monthly_coverage(holdings, mapping, pairs, calendar, plan):
    required = {'report_date', 'filing_date', 'accession', 'ticker', 'cusip', 'isin', 'weight_pct'}
    require(required <= set(holdings) and {'ticker', 'cik', 'identity_status'} <= set(mapping), 'basket/mapping schema differs')
    h = holdings.copy()
    for field in ('report_date', 'filing_date'):
        h[field] = pd.to_datetime(h[field], errors='raise')
    require(h.report_date.le(h.filing_date).all(), 'basket report occurs after filing')
    require(pd.to_numeric(h.weight_pct, errors='coerce').ge(0).all(), 'invalid disclosed weights')
    records = []; support = []
    for date in calendar.to_series().groupby(calendar.to_period('M')).first():
        known = h.loc[h.filing_date.lt(date)]
        if not known.empty:
            known = known.loc[known.report_date.eq(known.report_date.max())]
            known = known.loc[known.filing_date.eq(known.filing_date.max())]
        require(known.accession.nunique() <= 1 and not known.duplicated(['ticker', 'cusip', 'isin']).any(), 'ambiguous disclosed basket')
        covered = 0.; missing = 0
        for row in known.itertuples():
            matched = mapping.loc[mapping.ticker.eq(row.ticker)]
            ciks = set(pd.to_numeric(matched.cik, errors='coerce').dropna().astype(int))
            if row.ticker == 'MRVL' and row.report_date < pd.Timestamp('2021-04-20'):
                ciks = {1058057} if 1058057 in ciks else set()
            reason = ''; selected = None
            eligible = pairs.iloc[:0] if pairs.empty else pairs.loc[pairs.cik.isin(ciks)
                & pairs.conditional_available_session.le(date)]
            if len(ciks) != 1:
                reason = 'missing_or_ambiguous_candidate_CIK'
            elif row.ticker == 'WOLF' and row.report_date.year >= 2025:
                reason = 'post_bankruptcy_identity_unknown'
            elif eligible.empty:
                reason = 'no_conditionally_available_annual_pair'
            else:
                latest = eligible.loc[eligible.fiscal_end.eq(eligible.fiscal_end.max())]
                if len(latest) != 1:
                    reason = 'ambiguous_latest_fiscal_pair'
                elif not 0 <= (date - latest.fiscal_end.iloc[0]).days <= plan['calendar']['max_age_days']:
                    reason = 'latest_pair_stale_or_future_period'
                else:
                    selected = latest.iloc[0]
                    covered += float(row.weight_pct)
            missing += bool(reason)
            item = {'datetime': date, 'ticker': row.ticker, 'cusip': row.cusip, 'isin': row.isin,
                'candidate_ciks': json.dumps(sorted(ciks)), 'identity_status': json.dumps(sorted(set(matched.identity_status.astype(str)))),
                'weight_pct': float(row.weight_pct), 'basket_accession': row.accession, 'uncovered_reason': reason,
                'pair_accession': None if selected is None else selected.accession,
                'prior_pair_accession': None if selected is None else selected.prior_accession,
                'pair_fiscal_end': pd.NaT if selected is None else selected.fiscal_end,
                'conditional_available_session': pd.NaT if selected is None else selected.conditional_available_session,
                'historical_identity_qualified': False, 'original_semantics_verified': False, 'historical_PIT_qualified': False}
            support.append(item)
        records.append({'datetime': date, 'instrument': 'SOXX_DISCLOSED_PROXY',
            'basket_present': not known.empty, 'source_security_rows': len(known), 'uncovered_security_rows': missing,
            'conditional_mechanical_pair_weight_pct': covered, 'strict_PIT_qualified_weight_pct': 0.,
            'basket_accession': known.accession.iloc[0] if not known.empty else None,
            'historical_identity_qualified': False, 'original_semantics_verified': False, 'first_receipt_verified': False})
    return pd.DataFrame(records).set_index(['datetime', 'instrument']), pd.DataFrame(support)


def normalized_notebook(path):
    notebook = json.loads(Path(path).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
    return json.dumps(notebook, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def native_dataset(frame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({'meta': frame}, axis=1)), segments={'all': (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(dataset.prepare('all', col_set='meta'), frame)
    return dataset


def run(root, *, recovery=False):
    import qlib
    from qlib.data import D
    from qlib.workflow import R
    from src.data.nas_market_data import archive_nas_recorder
    root = Path(root).resolve(); card = root / 'research' / NAME
    plan = json.loads((card / 'plan.json').read_text())
    approval_path = card / ('recovery_authorization.json' if recovery else 'execution_approval.json')
    approval = json.loads(approval_path.read_text())
    require(plan['frozen'] and approval.get('approved') is True and approval['plan_sha256'] == sha(card / 'plan.json')
            and approval['module_sha256'] == sha(root / MODULE)
            and approval['notebook_sha256'] == hashlib.sha256(normalized_notebook(card / (NAME + '.ipynb'))).hexdigest(), 'root GO absent/stale')
    state_path = card / ('attempt2_run_state.json' if recovery else 'run_state.json')
    require(not state_path.exists(), 'attempt already exists; no automatic repeat')
    if recovery:
        require(approval.get('maximum_attempts') == 2 and approval.get('attempt') == 2, 'recovery attempt cap differs')
        for name in ('attempt1_evidence.json', 'schema_diagnostic.json', 'failed_run_state.json', 'recovery_review.json'):
            require(approval['recovery_bindings'][name] == sha(card / name), 'recovery evidence/review differs')
        prior = json.loads((card / 'failed_run_state.json').read_text())
        require(prior.get('status') == 'execution_failed' and prior.get('attempt') == 1, 'prior failure state differs')
        require(approval['source_artifacts'] == plan['source']['artifacts'], 'recovery source scope differs')
        require(approval['attempt1_module_sha256'] == plan['recovery']['attempt1_module_sha256'], 'attempt1 module binding differs')
    native = plan['calendar']; source = plan['source']; old = root / source['card']
    require(sha(root / native['path']) == native['sha256'] and sha(old / 'summary.json') == source['summary_sha256'], 'source metadata identity differs')
    require(sha(old / 'nas_binding.json') == source['nas_binding_sha256'], 'source transport binding differs')
    transport = json.loads((old / 'nas_binding.json').read_text())
    require(transport['evidence']['sha256'] == source['archive_sha256'], 'archive identity differs')
    atomic_json({'status': 'started', 'attempt': 2 if recovery else 1, 'plan_sha256': approval['plan_sha256'],
                 'cumulative_source_object_reads_reserved': 9 if recovery else 4}, state_path)
    qlib.init(provider_uri=str(root / native['provider_uri']), region='us', kernels=1, expression_cache=None, dataset_cache=None)
    objects, audit = load_verified_recorder_objects(old, expected_recorder_id=source['recorder_id'],
        artifacts=list(source['artifacts']), expected_artifact_sha256=source['artifacts'], expected_manifest_sha256=source['manifest_sha256'])
    manifest = objects['inputs/acquisition_manifest']
    schema = pd.DataFrame([{'status': x.get('status'), 'parse_valid': repr(x.get('parse_valid')) if 'parse_valid' in x else 'missing',
        'reused_parent': repr(x.get('reused_parent'))} for x in manifest]).value_counts(dropna=False).reset_index(name='rows')
    atomic_json({'breakdown': schema.to_dict('records'), 'additional_object_reads': 0}, card / 'attempt2_manifest_schema.json')
    audit['manifest_schema'] = schema
    raw, issuers, first, conflicts = extract(objects['inputs/raw_payloads'], manifest, plan)
    calendar = pd.DatetimeIndex(D.calendar(start_time=native['bounds'][0], end_time=native['bounds'][1], freq='day'))
    require(len(calendar) > 0 and sha(root / native['path']) == native['sha256'], 'calendar absent/changed')
    annual, pairs = annual_pairs(first, calendar, plan)
    monthly, support = monthly_coverage(objects['inputs/holdings'], objects['analysis/issuer_mapping'], pairs, calendar, plan)
    dataset = native_dataset(monthly)
    tracking = card / 'qlib_recorder'
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path='inputs', plan=plan, approval=approval, source_audit=audit,
            attempt1_evidence=json.loads((card / 'attempt1_evidence.json').read_text()) if recovery else None)
        rec.save_objects(artifact_path='analysis', extraction=raw, issuer_tag_coverage=issuers, earliest_non_amended_candidates=first,
                         conflicts=conflicts, annual_joint_candidates=annual, annual_pairs=pairs, monthly_coverage=monthly,
                         issuer_mapping=objects['analysis/issuer_mapping'], issuer_month_support=support)
        rec.save_objects(artifact_path='sources', module=(root / MODULE).read_bytes(),
            notebook=normalized_notebook(card / (NAME + '.ipynb')), identity=approval)
        rec.save_objects(artifact_path='metadata', qualification_dataset=dataset)
        pd.testing.assert_frame_equal(rec.load_object('metadata/qualification_dataset').prepare('all', col_set='meta'), monthly)
        summary = {'experiment': NAME, 'status': 'completed_pending_independent_review', 'decision': ('mechanical_source_candidates_exist_conditional_coverage_only' if len(pairs) else 'needs_data_fixed_tags_have_no_mechanical_annual_pairs'),
            'raw_fact_rows': len(raw), 'raw_CIKs': len(issuers),
            'raw_tag_issuer_availability': {field: int(issuers[field].gt(0).sum()) for field in ('inventory', 'revenue', 'assets')},
            'conditional_coverage_pct': {stat: float(getattr(monthly.conditional_mechanical_pair_weight_pct, stat)()) for stat in ('min', 'max', 'median')},
            'by_year': {str(year): {'months': len(group), 'months_with_basket': int(group.basket_present.sum()),
                'months_with_mechanical_support': int(group.conditional_mechanical_pair_weight_pct.gt(0).sum())}
                for year, group in monthly.groupby(monthly.index.get_level_values('datetime').year)}, 'mechanical_annual_joint_rows': len(annual), 'mechanical_prior_current_pairs': len(pairs),
            'issuer_month_uncovered_reasons': support.uncovered_reason.value_counts().to_dict() if len(support) else {},
            'issuer_month_identity_statuses': support.identity_status.value_counts().to_dict() if len(support) else {},
            'top_uncovered_tickers': support.loc[support.uncovered_reason.ne(''), 'ticker'].value_counts().head(10).to_dict() if len(support) else {},
            'conflicting_earliest_groups': len(conflicts), 'monthly_rows': len(monthly), 'strict_PIT_qualified_rows': 0,
            'model_input_qualified': False, 'original_semantics_verified': False, 'first_receipt_verified': False,
            'attempt': 2 if recovery else 1, 'cumulative_source_object_reads': 9 if recovery else 4,
            'new_fits': 0, 'new_accounts': 0, 'source_HTTP': 0, 'D_features_calls': 0, 'D_calendar_calls': 1,
            'plan_sha256': approval['plan_sha256'], 'module_sha256': approval['module_sha256'], 'recorder_id': rec.id,
            'experiment_id': str(rec.experiment_id), 'recorder_uri': str(tracking),
            'recorder_local_dir': str(tracking / str(rec.experiment_id) / rec.id), 'artifact_policy': plan['artifact_policy'],
            'qualification_boundary': 'Mechanical candidate coverage only; not original-document semantic, first receipt, historical PIT or factor effectiveness.',
            'minimum_missing_inputs': ['Original annual inventory/revenue context and units', 'Historical security/CIK continuity',
                                       'Authenticated publication/version clocks; current download is not PIT evidence'],
            'native_DatasetH_prepare_and_Recorder_readback': 'passed'}
        rec.save_objects(artifact_path='analysis', summary=summary)
        rec.log_metrics(raw_fact_rows=len(raw), annual_pairs=len(pairs), qualified_PIT_rows=0, new_fits=0)
    artifacts = Path(summary['recorder_local_dir']) / 'artifacts'
    atomic_json({'recorder_id': rec.id, 'entries': [{'path': str(p.relative_to(card)), 'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in sorted(artifacts.rglob('*')) if p.is_file()]}, card / 'verification_manifest.json')
    atomic_json({'schema_version': 'recorder_evidence_binding_v1', 'authority': transport['authority'], 'source_id': NAME,
                 'ssh': transport['ssh'], 'source_plan': 'plan.json', 'as_of': native['bounds'][1]}, card / 'nas_binding.json')
    summary['nas_evidence'] = archive_nas_recorder(card / 'nas_binding.json', workspace_root=root, card=card,
        recorder_id=rec.id, experiment_id=str(rec.experiment_id))
    atomic_json(summary, card / 'summary.json'); atomic_json({'status': 'completed', 'attempt': 2 if recovery else 1, 'recorder_id': rec.id,
            'cumulative_source_object_reads': 9 if recovery else 4}, state_path)
    (card / 'conclusion.md').write_text('库存/销售来源资格检查完成；原生Recorder：`' + rec.id + '`。\n\n'
        + f"原始字段{len(raw)}行，机械年度配对{len(pairs)}行；严格PIT合格0行。这里只检查覆盖和来源缺口，没有计算因子、收益、拟合或账户。\n\n"
        + '原始财报语义、历史证券身份和首次公开/版本时间仍需资格化；当前下载与最早已知filed日期不能认证历史PIT。目标未达成，不能声称库存因子有效。\n')
    return summary
