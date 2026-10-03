from pathlib import Path
import hashlib,json
from src.data.feasible_parent_sparse_search import load_feasible_parent_inputs
root=Path.cwd();source=root/'src/data/feasible_parent_sparse_search.py';before=hashlib.sha256(source.read_bytes()).hexdigest();plan=json.loads((root/'research/feasible_parent_sparse_search_20261003/selection_plan.json').read_text());b=load_feasible_parent_inputs(root,plan)
assert hashlib.sha256(source.read_bytes()).hexdigest()==before
report={'scope':'Read-only native source and original63development-parent cache verification; no policy replay, forecast/model fit, writableRecorder, or outer-parent loading.', 'loader_sha256':before,'parent_source_audit':b['parent_source_audit'],'source_audit':b['source_audit'],'parent_source_artifacts':b['parent_source_artifacts'],'parents':len(b['parent_seed_models']),'folds':len(b['parent_fold_bundles']),'failed':False}
Path('/tmp/feasible-parent-cache-readback-final.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',report['parents'],report['folds'],before)
