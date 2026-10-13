"""Image-only MinerU benchmark. Local standard tier, resumable per-page outputs."""
from pathlib import Path
import argparse,json,time,traceback,resource,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'outputs/ocr_ablation/omnidoc100_20261013';out=run/'mineru';out.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args()
from mineru.parser.mineru_parser import MinerUParser
from mineru.config import VlmConfig,config as mineru_config
mineru_config.model.small_backend="torch"
config=dict(tier='standard',parse_mode='ocr',image_analysis=False)
pipeline=MinerUParser(**config,vlm_config=VlmConfig(engine='llama-cpp'))
(out/'config.json').write_text(json.dumps(config|dict(version=importlib.metadata.version('mineru'),vlm_engine='llama-cpp'),indent=2))
manifest=json.loads((run/'sample_manifest.json').read_text())['pages']
if args.limit:manifest=manifest[:args.limit]
for item in manifest:
 name=Path(item['image']).stem;record=out/f'{name}.status.json'
 if record.exists() and json.loads(record.read_text()).get('status')=='ok':continue
 start=time.perf_counter();status=dict(id=item['id'],image=item['image'],status='error')
 try:
  result=pipeline.parse(run/'images'/item['image'])
  (out/f'{name}.json').write_text(result.to_json(),encoding='utf-8')
  (out/f'{name}.md').write_text(result.markdown(),encoding='utf-8');status['status']='ok'
 except Exception as exc:
  status.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());(out/f'{name}.md').write_text('')
 status.update(seconds=time.perf_counter()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 record.write_text(json.dumps(status,ensure_ascii=False,indent=2));print('mineru',item['id'],status['status'],round(status['seconds'],2),flush=True)
