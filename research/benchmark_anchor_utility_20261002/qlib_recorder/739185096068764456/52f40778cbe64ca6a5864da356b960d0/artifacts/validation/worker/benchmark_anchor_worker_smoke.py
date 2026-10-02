from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json, tempfile
import numpy as np
from src.model.benchmark_anchor_utility import AnchorUtilitySpec
from src.workflow.benchmark_anchor_utility import _fit_job
from qlib.workflow import R

def main():
 rng=np.random.default_rng(933)
 x=rng.normal(size=(252,27));ret=rng.normal(0,.003,size=(252,2))
 spec=AnchorUtilitySpec(updates=1)
 base=Path(tempfile.mkdtemp(prefix='soxl-anchor-synthetic-recorder-'))
 import qlib
 qlib.init(provider_uri=str(Path.cwd()/'data/qlib/published_futu_affine_20260925_v1'),region='us',kernels=1)
 R.set_uri(str(base))
 with R.start(experiment_name='synthetic_worker_contract'):
  recorder=R.get_recorder()
  with ProcessPoolExecutor(max_workers=3) as pool:
   results=list(pool.map(_fit_job,[('synthetic',2016,seed,x,ret,spec) for seed in (17,41,71)]))
  for name,year,seed,model,audit in results:
   assert audit['failure'] is None and len(audit['epochs'])==2
   recorder.save_objects(artifact_path=f'synthetic/{seed}',**{'model.pkl':model,'audit.pkl':audit})
   assert recorder.load_object(f'synthetic/{seed}/model.pkl')['tensor_sha256']==model['tensor_sha256']
  report={'market_inputs':False,'registered_market_fit_started':False,'synthetic_rows':252,'seeds':[17,41,71],
   'updates':1,'all_workers_and_recorder_roundtrips_pass':True,'recorder_id':recorder.id,'uri':str(base),
   'elapsed_seconds':[r[-1]['elapsed_seconds'] for r in results]}
  Path('/tmp/benchmark-anchor-validation/worker-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report))
if __name__=='__main__':main()
