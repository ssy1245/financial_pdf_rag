"""FinanceBench 50: frozen 300-token chunks, two local encoders, retrieval only."""
import argparse, hashlib, json, time, platform
from pathlib import Path
from copy import deepcopy
import numpy as np
from financial_rag.evaluation.financebench import write_json, evaluation_question
from financial_rag.evaluation.chunk_ablation import token_chunks
from financial_rag.evaluation.embedding_ablation import MODELS, BGE_PROMPT
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/embedding_ablation/financebench50_300_0_20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 from transformers import AutoTokenizer
 from financial_rag.ingestion.parser import parse_pdf
 from financial_rag.ingestion.normalizer import normalize_pages
 OUT.mkdir(parents=True,exist_ok=True)
 data=ROOT/'data/raw/financebench50';manifest=json.loads((data/'manifest.json').read_text())
 old=ROOT/'outputs/chunk_ablation/financebench_20261013'
 pages_old=json.loads((old/'normalized_pages.json').read_text())
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/bge-small-en-v1.5'),local_files_only=True);tok.model_max_length=10**9
 pages={};chunks=[]
 for d in manifest['documents']:
  doc=d['document_id'];pdf=data/d['filename'];assert sha(pdf)==d['sha256']
  pages[doc]=pages_old.get(doc) or normalize_pages(parse_pdf(pdf))
  current=token_chunks(pages[doc],doc,tok,300,0)
  for c,ids in zip(current,tok([c['text'] for c in current],add_special_tokens=False)['input_ids']):
   c['content_tokens']=len(ids);c['truncated_tokens']=max(0,len(ids)-510);assert len(ids)<=300
  chunks+=current;print('PREPARE',doc,len(current),flush=True)
 oldchunks=json.loads((old/'tokens300_overlap0/chunks.json').read_text())
 mapped={c['chunk_id']:c for c in chunks}
 assert all(mapped[c['chunk_id']]==c for c in oldchunks),'Old chunks changed'
 write_json(OUT/'normalized_pages.json',pages);write_json(OUT/'chunks.json',chunks)
 cases=json.loads((data/'selected.json').read_text());write_json(OUT/'selected_cases.json',cases)
 write_json(OUT/'config.json',{'manifest':manifest,'questions':len(cases),'documents':len(pages),'pages':sum(map(len,pages.values())),'chunks':len(chunks),'chunk_size':300,'overlap':0,'tokenizer':'BAAI/bge-small-en-v1.5','old_chunks_exactly_preserved':len(oldchunks),'chunks_sha256':sha(OUT/'chunks.json'),'selected_sha256':sha(OUT/'selected_cases.json'),'query':'Original English question + target report filename; model recommended query instruction.','retrieval':'Global dense cosine; no doc filter, reranker, BM25, LLM generation.','budget':'Ranked prefix of whole chunks <=2000 BGE-small content tokens; no skipping.','source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/financial_rag/ingestion/parser.py',ROOT/'src/financial_rag/ingestion/normalizer.py',ROOT/'src/financial_rag/evaluation/chunk_ablation.py']}})
 criteria=json.loads((Path(__file__).parent/'data/financebench50_evidence.json').read_text())
 assert [c['id'] for c in criteria]==[c['financebench_id'] for c in cases]
 (OUT/'evidence_criteria.json').write_bytes((Path(__file__).parent/'data/financebench50_evidence.json').read_bytes())
 audit_evidence()
 print('PREPARED',len(cases),len(pages),len(chunks),flush=True)
def encode(name):
 import torch, importlib.metadata
 from sentence_transformers import SentenceTransformer
 chunks=json.loads((OUT/'chunks.json').read_text());cases=json.loads((OUT/'selected_cases.json').read_text())
 repo,dirname=MODELS[name];path=ROOT/'models'/dirname;folder=OUT/name;folder.mkdir(exist_ok=True)
 device='mps' if torch.backends.mps.is_available() else 'cpu';dtype=torch.float16 if device=='mps' else torch.float32
 start=time.perf_counter();model=SentenceTransformer(str(path),local_files_only=True,device=device,model_kwargs={'dtype':dtype},tokenizer_kwargs={'padding_side':'left'} if name.startswith('qwen') else {})
 prompt=BGE_PROMPT if name=='bge-small' else model.prompts['query']
 def sync():
  if device=='mps':torch.mps.synchronize()
 def enc(items):
  a=model.encode(items,prompt='',normalize_embeddings=True,batch_size=8,show_progress_bar=False,convert_to_numpy=True).astype(np.float32)
  assert np.isfinite(a).all();a/=np.linalg.norm(a,axis=1,keepdims=True);return a
 # Operational precision preflight against previously recorded FP32 on 64 fixed source chunks.
 previous=ROOT/'outputs/embedding_ablation/financebench_300_0_20261013'
 oldchunks=json.loads((previous/'chunks.json').read_text());indices=np.linspace(0,len(oldchunks)-1,64,dtype=int)
 oldvectors=np.load(previous/name/'vectors.npy',mmap_mode='r')[indices];small=enc([oldchunks[i]['text'] for i in indices])
 cos=np.sum(oldvectors*small,axis=1);audit={'reference':'Previous FP32 vectors, 64 equally spaced old corpus chunks.','cosine_min':float(cos.min()),'cosine_mean':float(cos.mean()),'dtype':str(dtype),'threshold_min_cosine':0.999}
 write_json(folder/'precision_audit.json',audit);assert cos.min()>=0.999,audit
 tok=deepcopy(model.tokenizer);tok.model_max_length=10**9
 texts=[c['text'] for c in chunks];queries=[prompt+evaluation_question(c) for c in cases]
 lengths=[len(x) for x in tok([t.strip() for t in texts],add_special_tokens=True,truncation=False)['input_ids']]
 qlens=[len(x) for x in tok(queries,add_special_tokens=True,truncation=False)['input_ids']]
 config={'repo':repo,'device':device,'dtype':str(dtype),'batch_size':8,'query_prompt':prompt,'document_prompt':'','pooling':str(model[1]),'padding_side':tok.padding_side,'max_seq_length':model.max_seq_length,'chunks_sha256':sha(OUT/'chunks.json'),'selected_sha256':sha(OUT/'selected_cases.json'),'model_sha256':{str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.parts and p.suffix in ('.json','.txt','.bin','.safetensors','.model')},'versions':{x:importlib.metadata.version(x) for x in ('torch','transformers','sentence-transformers','numpy')},'platform':platform.platform(),'source_sha256':sha(Path(__file__))}
 if (folder/'config.json').exists():assert json.loads((folder/'config.json').read_text())==config,'Cache configuration changed'
 write_json(folder/'config.json',config)
 write_json(folder/'token_audit.json',{'document_truncation_count':sum(n>model.max_seq_length for n in lengths),'query_truncation_count':sum(n>model.max_seq_length for n in qlens),'document_max':max(lengths),'document_p95':float(np.percentile(lengths,95)),'query_max':max(qlens),'max_seq_length':model.max_seq_length})
 cache=folder/'blocks';cache.mkdir(exist_ok=True);elapsed=[];arrays=[]
 for offset in range(0,len(texts),256):
  file=cache/f'{offset:06}.npy';timing=file.with_suffix('.json')
  if file.exists() and timing.exists():arr=np.load(file);seconds=json.loads(timing.read_text())['seconds']
  else:
   sync();begin=time.perf_counter();arr=enc(texts[offset:offset+256]);sync();seconds=time.perf_counter()-begin
   np.save(file,arr);write_json(timing,{'seconds':seconds,'rows':len(arr)})
  assert len(arr)==len(texts[offset:offset+256]);arrays.append(arr);elapsed.append(seconds)
  print('ENCODE',name,min(offset+256,len(texts)),len(texts),'block_seconds',round(seconds,2),flush=True)
 vectors=np.concatenate(arrays);np.save(folder/'vectors.npy',vectors)
 sync();begin=time.perf_counter();qv=enc(queries);sync();query_seconds=time.perf_counter()-begin;np.save(folder/'queries.npy',qv)
 write_json(folder/'encoding.json',{'encoding_seconds':sum(elapsed),'query_batch_seconds':query_seconds,'dimension':vectors.shape[1],'vector_bytes':vectors.nbytes,'vectors_sha256':sha(folder/'vectors.npy'),'queries_sha256':sha(folder/'queries.npy'),'wall_seconds_including_load_preflight_audit':time.perf_counter()-start})
 print('DONE',name,sum(elapsed),flush=True)

from financial_rag.evaluation.embedding_ablation import rank_questions
from financial_rag.evaluation.evidence_content import score_content, match_unit
MODES=('top5','top10','top20','budget2000')
CRITERIA_SHA='c4c74511b7869b178735fde4b5acc5506c740397c08b9e6eec1fac4c2b55210d'
def score():
 assert hashlib.sha256((OUT/'evidence_criteria.json').read_bytes()).hexdigest()==CRITERIA_SHA,'Frozen criteria changed'
 chunks=json.loads((OUT/'chunks.json').read_text());byid={c['chunk_id']:c for c in chunks}
 cases=json.loads((OUT/'selected_cases.json').read_text());criteria=json.loads((OUT/'evidence_criteria.json').read_text());common=json.loads((OUT/'config.json').read_text());old=set(common['manifest']['old_ids'])
 all_results={};summaries={}
 for name in ('bge-small','qwen3-0.6b'):
  folder=OUT/name
  if not (folder/'encoding.json').exists():print('PENDING',name);continue
  v=np.load(folder/'vectors.npy',mmap_mode='r');q=np.load(folder/'queries.npy')
  assert len(v)==len(chunks) and len(q)==50 and np.isfinite(v).all() and np.isfinite(q).all()
  assert np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-4)
  rows=rank_questions(v,q,chunks,cases)
  for r,rule in zip(rows,criteria):
   r['cohort']='old10' if r['id'] in old else 'new40'
   for mode in MODES:
    count=r[mode]['chunks'];retrieved=[byid[x['chunk_id']] for x in r['ranking'][:count]]
    r[mode]['content']=score_content(retrieved,rule)
    r[mode]['page_hit_no_content']=int(r[mode]['hit'] and not r[mode]['content']['hit']) if not rule['manual_required'] else None
   for item in r['ranking']:
    c=byid[item['chunk_id']];item['text']=c['text']
    item['individual_content_units']=score_content([c],rule)['matched_units']
  write_json(folder/'results.json',rows);summary={}
  for cohort in ('all50','old10','new40'):
   sample=rows if cohort=='all50' else [r for r in rows if r['cohort']==cohort]
   summary[cohort]={}
   for mode in MODES:
    auto=[r for r in sample if not r[mode]['content']['manual_required']]
    summary[cohort][mode]={'questions':len(sample),'content_evaluable':len(auto),
     'page_hit':float(np.mean([r[mode]['hit'] for r in sample])),
     'page_coverage':float(np.mean([r[mode]['coverage'] for r in sample])),
     'page_all':float(np.mean([r[mode]['all_pages'] for r in sample])),
     'page_mrr':float(np.mean([r[mode]['mrr'] for r in sample])),
     'content_hit':float(np.mean([r[mode]['content']['hit'] for r in auto])),
     'content_coverage':float(np.mean([r[mode]['content']['coverage'] for r in auto])),
     'content_all':float(np.mean([r[mode]['content']['all_units'] for r in auto])),
     'page_hit_no_content':sum(r[mode]['page_hit_no_content'] for r in auto),
     'used_tokens_mean':float(np.mean([sum(x['budget_tokens'] for x in r['ranking'][:r[mode]['chunks']]) for r in sample]))}
  summary['encoding']=json.loads((folder/'encoding.json').read_text());summary['token_audit']=json.loads((folder/'token_audit.json').read_text())
  write_json(folder/'summary.json',summary);all_results[name]=rows;summaries[name]=summary
 if len(all_results)==2:
  paired={}
  rng=np.random.default_rng(20261013)
  for cohort in ('all50','old10','new40'):
   paired[cohort]={}
   for mode in MODES:
    pairs=[(a,b) for a,b in zip(*all_results.values()) if (cohort=='all50' or a['cohort']==cohort) and not a[mode]['content']['manual_required']]
    delta=np.array([b[mode]['content']['coverage']-a[mode]['content']['coverage'] for a,b in pairs])
    boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
    paired[cohort][mode]={'n':len(delta),'qwen_better':int(sum(delta>1e-9)),'equal':int(sum(abs(delta)<=1e-9)),'bge_better':int(sum(delta< -1e-9)),'mean_delta':float(delta.mean()),'question_bootstrap_95ci':list(map(float,np.quantile(boot,[.025,.975])))}
  write_json(OUT/'paired_comparison.json',paired)
 write_json(OUT/'summary.json',summaries)
 write_json(OUT/'scoring_manifest.json',{'criteria_sha256':CRITERIA_SHA,'scorer_sha256':hashlib.sha256((ROOT/'src/financial_rag/evaluation/evidence_content.py').read_bytes()).hexdigest(),'score_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy':'Gold-document/page aligned local label + exact value or key claim matches. Only returned contiguous spans may be joined. One case (blank acquisitions table cells) is excluded from automatic content denominators, retained in 50-case page scores. All 50 are development cases; no generated answers or semantic judge.','bootstrap':'Paired question resampling, 10,000 draws, seed 20261013; exploratory, questions within a document may be correlated.'})
 print(json.dumps(summaries,ensure_ascii=False,indent=2))

def audit_evidence():
 criteria=json.loads((OUT/'evidence_criteria.json').read_text());pages=json.loads((OUT/'normalized_pages.json').read_text());chunks=json.loads((OUT/'chunks.json').read_text());audit=[]
 for c in criteria:
  for u in c['units']:
   matches=[x for x in chunks if x['document_id']==c['document_id'] and x['page'] in u['pages'] and match_unit(x['text'],u)]
   page_match=any(match_unit(x['text'],u) for x in pages[c['document_id']] if x['page'] in u['pages'])
   audit.append({'id':c['id'],'unit':u['id'],'label':u['label'],'in_parsed_page':page_match,'in_single_chunk':bool(matches),'chunk_ids':[x['chunk_id'] for x in matches]})
 write_json(OUT/'evidence_reachability.json',audit)


def select50(pool, previous):
 import random
 rows=list(pool);random.Random(20261014).shuffle(rows);selected=list(previous)
 for kind,n in [('metrics-generated',17),('domain-relevant',17),('novel-generated',16)]:
  for x in rows:
   if sum(c['question_type']==kind for c in selected)>=n:break
   if x['question_type']==kind and x['financebench_id'] not in {c['financebench_id'] for c in selected}:selected.append(x)
 if len(selected)!=50:raise ValueError('Insufficient question pool')
 return selected


def download():
 import httpx
 from concurrent.futures import ThreadPoolExecutor
 from financial_rag.evaluation.financebench import REVISION
 old=ROOT/'data/raw/financebench';data=ROOT/'data/raw/financebench50';data.mkdir(parents=True,exist_ok=True)
 pool=[json.loads(x) for x in (old/'financebench_open_source.jsonl').read_text().splitlines()]
 selected=select50(pool,json.loads((old/'selected.json').read_text()))
 criteria=json.loads((Path(__file__).parent/'data/financebench50_evidence.json').read_text())
 assert [c['id'] for c in criteria]==[c['financebench_id'] for c in selected],'Selected cases differ from frozen criteria'
 previous=json.loads((data/'manifest.json').read_text()) if (data/'manifest.json').exists() else {'documents':[]}
 hashes={d['filename']:d['sha256'] for d in previous['documents']}
 def fetch(name):
  filename=name+'.pdf';url=f'https://raw.githubusercontent.com/patronus-ai/financebench/{REVISION}/pdfs/{filename}';path=data/filename
  if path.exists() and filename in hashes:
   assert sha(path)==hashes[filename],'Cached PDF changed'
  else:
   response=httpx.get(url,timeout=120,follow_redirects=True);response.raise_for_status()
   assert response.content.startswith(b'%PDF');path.write_bytes(response.content)
  return {'document_id':name,'filename':filename,'url':url,'sha256':sha(path),'size_bytes':path.stat().st_size}
 names=sorted({c['doc_name'] for c in selected}|{e.get('doc_name',c['doc_name']) for c in selected for e in c['evidence']})
 with ThreadPoolExecutor(max_workers=4) as pool:docs=list(pool.map(fetch,names))
 write_json(data/'selected.json',selected)
 write_json(data/'manifest.json',{'upstream_commit':REVISION,'selection_seed':20261014,'selection':'Prior 10 development cases retained; add 40 without replacement by shuffled metadata until type quotas 17 metrics / 17 domain / 16 novel; no company-uniqueness constraint.','old_ids':[c['financebench_id'] for c in selected[:10]],'documents':docs})


def main():
 global OUT
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('action',choices=['download','prepare','bge-small','qwen3-0.6b','score','report'])
 p.add_argument('--output',type=Path,default=OUT)
 a=p.parse_args();OUT=a.output.resolve()
 if a.action=='download':download()
 elif a.action=='prepare':prepare()
 elif a.action in MODELS:encode(a.action)
 elif a.action=='score':score()
 else:
  from financial_rag.evaluation import evidence_report
  evidence_report.O=OUT;evidence_report.main()


if __name__=='__main__':main()
