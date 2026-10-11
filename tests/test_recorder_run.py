import os
from pathlib import Path
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
from src.core.recorder_run import start_evidence_run,archive_evidence,publish_evidence_manifest
from src.core.recorder_artifacts import load_compressed_object


def test_native_recorder_resume_compression_and_published_identity(tmp_path):
    import qlib
    from qlib.workflow import R
    qlib.init(provider_uri=str(Path(__file__).resolve().parents[1]/'data/qlib/published_futu_affine_20260925_v1'),region='us',kernels=1)
    identity=start_evidence_run(experiment='recorder_helper_contract',uri=tmp_path/'recorder',name='zero_fit_contract',
        objects={'inputs.pkl':{'fit_calls':0}},params={'fit_calls':0})
    value={'source':'native_account_evidence','cash':3.25,'fit_calls':0}
    artifacts=archive_evidence(identity,{'case.pkl.gz':value},artifact_path='cases',compressed=True)
    archive_evidence(identity,{'summary.pkl':{'fit_calls':0}})
    with R.uri_context(identity['recorder_uri']):
        rec=R.get_recorder(experiment_name=identity['experiment'],recorder_id=identity['recorder_id'])
        assert rec.status=='FINISHED' and load_compressed_object(rec,artifacts['case.pkl.gz'])==value
    summary=dict(identity,fit_calls=0,recorder_status='FINISHED')
    publish_evidence_manifest(tmp_path,summary)
    import json
    manifest=json.loads((tmp_path/'verification_manifest.json').read_text())
    assert manifest['recorder_id']==identity['recorder_id']
    assert any(e['path'].endswith('/cases/case.pkl.gz') for e in manifest['entries'])
