"""Small native Recorder persistence helpers, without research choices or paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from src.core.recorder_artifacts import save_compressed_object


def start_evidence_run(*,experiment,uri,name,objects,params,artifact_path='inputs'):
    from qlib.workflow import R
    with R.start(experiment_name=experiment,recorder_name=name,uri=str(uri)):
        rec=R.get_recorder()
        rec.log_params(**params)
        rec.save_objects(artifact_path=artifact_path,**objects)
        identity=dict(experiment=experiment,recorder_id=rec.id,recorder_uri=str(Path(uri).resolve()),
                      recorder_local_dir=rec.get_local_dir())
    if rec.status!='FINISHED':
        raise ValueError('blocked: input Recorder did not finish')
    return identity


def archive_evidence(identity,objects,*,artifact_path='aggregate',compressed=False):
    from qlib.workflow import R
    artifacts={}
    with R.start(experiment_name=identity['experiment'],recorder_id=identity['recorder_id'],
                 uri=identity['recorder_uri'],resume=True):
        rec=R.get_recorder()
        if rec.id!=identity['recorder_id']:
            raise ValueError('blocked: resumed Recorder identity differs')
        if compressed:
            for name,value in objects.items():
                artifacts[name]=save_compressed_object(rec,name=name,artifact_path=artifact_path,value=value)
        else:
            rec.save_objects(artifact_path=artifact_path,**objects)
    if rec.status!='FINISHED':
        raise ValueError('blocked: archive Recorder did not finish')
    return artifacts


def publish_evidence_manifest(card,summary):
    card=Path(card);local=Path(summary['recorder_local_dir'])/'artifacts'
    entries=[dict(path=str(p.relative_to(card)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
             for p in sorted(local.rglob('*')) if p.is_file() and not p.name.startswith('code_')]
    (card/'verification_manifest.json').write_text(json.dumps(dict(recorder_id=summary['recorder_id'],entries=entries),indent=2)+'\n')
    (card/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
