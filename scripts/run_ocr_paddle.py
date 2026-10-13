"""Run a fixed image-only benchmark with resumable, per-page native outputs."""
import os
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE','HuggingFace')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('PADDLE_PDX_CACHE_HOME',str(ROOT/'models/ocr/paddlex'))
import argparse,json,time,traceback,importlib.metadata,resource
p=argparse.ArgumentParser();p.add_argument('model',choices=['pp_structure','paddle_vl']);p.add_argument('--limit',type=int);args=p.parse_args()
run=ROOT/'outputs/ocr_ablation/omnidoc100_20261013';out=run/args.model;out.mkdir(exist_ok=True)
common=dict(device='cpu',use_doc_orientation_classify=False,use_doc_unwarping=False)
config=common| (dict(use_formula_recognition=False,use_chart_recognition=False,use_seal_recognition=False) if args.model=='pp_structure' else dict(pipeline_version='v1.6',vl_rec_backend='mlx-vlm-server',vl_rec_server_url='http://127.0.0.1:8111/',vl_rec_api_model_name=json.loads((run/'paddle_vl_weights.json').read_text())['local_path'],vl_rec_max_concurrency=1))
(out/'config.json').write_text(json.dumps(dict(model=args.model,params=config,packages={k:importlib.metadata.version(k) for k in ['paddleocr','paddlepaddle','paddlex']}),indent=2))
start=time.perf_counter()
from paddleocr import PPStructureV3,PaddleOCRVL
pipeline=(PPStructureV3 if args.model=='pp_structure' else PaddleOCRVL)(**config)
(out/'initialization.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start)))
print('MODEL_READY',args.model,flush=True)
manifest=json.loads((run/'sample_manifest.json').read_text())['pages']
if args.limit:manifest=manifest[:args.limit]
for item in manifest:
 if (out/'STOP_AFTER_PAGE').exists():
  print('STOP_AFTER_PAGE',flush=True);break
 name=Path(item['image']).stem;record=out/f'{name}.status.json'
 if record.exists() and json.loads(record.read_text()).get('status')=='ok':continue
 t=time.perf_counter();status=dict(id=item['id'],image=item['image'],status='error')
 try:
  results=list(pipeline.predict(str(run/'images'/item['image'])))
  if len(results)!=1:raise RuntimeError(f'Expected one page, got {len(results)}')
  result=results[0];result.save_to_json(str(out/f'{name}.json'));result.save_to_markdown(str(out))
  if not (out/f'{name}.md').exists():raise RuntimeError('Markdown was not written')
  status['status']='ok'
 except Exception as exc:
  status['error']=f'{type(exc).__name__}: {exc}';status['traceback']=traceback.format_exc()
  (out/f'{name}.md').write_text('')
 status.update(seconds=time.perf_counter()-t,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 record.write_text(json.dumps(status,ensure_ascii=False,indent=2))
 print(args.model,item['id'],status['status'],round(status['seconds'],2),flush=True)
