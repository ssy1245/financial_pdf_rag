"""Official OmniDocBench text/table/reading-order evaluation; formula CDM excluded."""
from pathlib import Path
import json,argparse,os,sys
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'outputs/ocr_ablation/omnidoc100_20261013'
def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=['pp_structure','paddle_vl','mineru']);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    sys.path.insert(0,str(run/'benchmark_source'))
    out=run/'evaluation'/('smoke_'+args.model if args.smoke else args.model);out.mkdir(parents=True,exist_ok=True)
    gt=run/'ground_truth.json'
    if args.smoke:
     gt=out/'ground_truth.json';gt.write_text(json.dumps(json.loads((run/'ground_truth.json').read_text())[:1],ensure_ascii=False))
    else:
     manifest=json.loads((run/'sample_manifest.json').read_text())['pages']
     missing=[x['id'] for x in manifest if not (run/args.model/(Path(x['image']).stem+'.status.json')).exists()]
     if missing:raise SystemExit(f'Inference is not complete: {len(missing)} missing pages')
    config={'end2end_eval':{'metrics':{'text_block':{'metric':['Edit_dist']},'table':{'metric':['TEDS','Edit_dist'],'teds_workers':2},'reading_order':{'metric':['Edit_dist']}},'dataset':{'dataset_name':'end2end_dataset','ground_truth':{'data_path':str(gt)},'prediction':{'data_path':str(run/args.model)},'match_method':'quick_match','match_workers':2,'quick_match_truncated_timeout_sec':300,'match_timeout_sec':420,'timeout_fallback_max_chunk_span':10,'timeout_fallback_order_penalty':0.10}}}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    os.chdir(out)
    from src.core.pipeline import run_config
    run_config(config)

if __name__ == "__main__":
    main()
