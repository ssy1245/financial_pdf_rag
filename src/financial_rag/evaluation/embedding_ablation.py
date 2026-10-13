"""Frozen-chunk, dense-only embedding comparison; no generation or query rewriting."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from financial_rag.evaluation.financebench import evaluation_question, gold_pages, write_json
from financial_rag.evaluation.chunk_ablation import metric, select_budget

MODELS={
 'bge-small':('BAAI/bge-small-en-v1.5','bge-small-en-v1.5'),
 'bge-base':('BAAI/bge-base-en-v1.5','bge-base-en-v1.5'),
 'bge-m3':('BAAI/bge-m3','bge-m3'),
 'qwen3-0.6b':('Qwen/Qwen3-Embedding-0.6B','Qwen3-Embedding-0.6B'),
}
BGE_PROMPT='Represent this sentence for searching relevant passages: '


def rank_questions(vectors, query_vectors, chunks, cases):
    lengths=[c['content_tokens'] for c in chunks]  # frozen BGE-small tokenizer budget
    rows=[]
    for case,qv in zip(cases,query_vectors):
        start=time.perf_counter();scores=vectors@qv
        order=np.argsort(-scores,kind='stable')
        elapsed=(time.perf_counter()-start)*1000
        expected=gold_pages(case)
        row={'id':case['financebench_id'],'company':case['company'],'question':case['question'],
             'expected_pages':sorted(expected),'search_ms':elapsed}
        for k in (5,10,20):row[f'top{k}']=metric(order[:k],chunks,expected)
        indices,used=select_budget(order,lengths,2000)
        row['budget2000']=dict(metric(indices,chunks,expected),used_tokens=used)
        row['ranking']=[dict(rank=j,chunk_id=chunks[int(i)]['chunk_id'],document_id=chunks[int(i)]['document_id'],
            page=chunks[int(i)]['page'],score=float(scores[i]),budget_tokens=lengths[int(i)],
            gold_page=(chunks[int(i)]['document_id'],chunks[int(i)]['page']) in expected)
            for j,i in enumerate(order[:max(20,len(indices))],1)]
        rows.append(row)
    return rows


def summarize(rows):
    return {mode:{key:float(np.mean([r[mode][key] for r in rows]))
        for key in ('hit','coverage','all_pages','mrr','chunks','same_page_repeat_fraction')}
        for mode in ('top5','top10','top20','budget2000')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    source=root/'outputs/chunk_ablation/financebench_20261013'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=['tokens300_overlap0/chunks.json','selected_cases.json','config.json']
    fingerprints={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in frozen}
    chunks=json.loads((source/frozen[0]).read_text())
    cases=json.loads((source/'selected_cases.json').read_text())
    common={'frozen_source':str(source),'input_sha256':fingerprints,'chunks':len(chunks),'questions':len(cases),
        'chunk_policy':'Frozen BGE-small tokenizer 300 content tokens / overlap 0, not re-chunked by model.',
        'retrieval':'Normalized dense dot product only; no BM25, reranker, Agent, generation, document filter or rewriting.',
        'budget_policy':'Ranked whole-chunk prefix <=2000 content tokens counted by frozen BGE-small tokenizer for every model.',
        'question_policy':'Original English question + target report filename; model-specific query prompt added where recommended.',
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),root/'src/financial_rag/evaluation/chunk_ablation.py',root/'src/financial_rag/evaluation/financebench.py']},
        'limitations':'10 previously inspected development questions. Page labels may reward unrelated content; no answer accuracy.'}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=common:
        raise ValueError('Frozen inputs/source changed; choose a new output directory')
    write_json(out/'config.json',common)
    write_json(out/'chunks.json',chunks);write_json(out/'selected_cases.json',cases)
    folder=out/args.model;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        print('ALREADY COMPLETE',args.model,flush=True);return
    import torch
    from sentence_transformers import SentenceTransformer
    repo,dirname=MODELS[args.model];model_path=root/'models'/dirname
    model_files={str(p.relative_to(model_path)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(model_path.rglob('*')) if p.is_file() and '.cache' not in p.parts and p.suffix in ('.json','.txt','.model','.bin','.safetensors')}
    if not any(k.endswith(('.bin','.safetensors')) for k in model_files):raise ValueError('Missing model weights')
    revision=None
    manifest_path=root/'models/embedding_comparison_manifest.json'
    if manifest_path.exists():
        revision=next((r['revision'] for r in json.loads(manifest_path.read_text()) if r['repo']==repo),None)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    start=time.perf_counter()
    model=SentenceTransformer(str(model_path),local_files_only=True,device=device,model_kwargs={'dtype':torch.float32},
        tokenizer_kwargs={'padding_side':'left'} if args.model=='qwen3-0.6b' else {})
    load_seconds=time.perf_counter()-start
    if args.model.startswith('bge-') and args.model!='bge-m3': prompt=BGE_PROMPT
    elif args.model=='qwen3-0.6b':
        prompt=model.prompts.get('query')
        if not prompt:raise ValueError('Qwen official query prompt missing')
    else:prompt=''
    tokenizer=deepcopy(model.tokenizer);tokenizer.model_max_length=10**9
    texts=[c['text'] for c in chunks]
    plain_queries=[evaluation_question(c) for c in cases]
    queries=[prompt+q for q in plain_queries]
    # ST Transformer strips surrounding whitespace before tokenizer calls.
    lengths=[len(ids) for ids in tokenizer([t.strip() for t in texts],add_special_tokens=True,truncation=False)['input_ids']]
    query_lengths=[len(ids) for ids in tokenizer([q.strip() for q in queries],add_special_tokens=True,truncation=False)['input_ids']]
    maximum=model.max_seq_length
    audit={'max_seq_length':maximum,'document_truncation_count':sum(n>maximum for n in lengths),
        'document_input_token_p50':float(np.median(lengths)),'document_input_token_p95':float(np.percentile(lengths,95)),
        'document_input_token_max':max(lengths),'query_lengths':query_lengths,'query_truncation_count':sum(n>maximum for n in query_lengths),
        'document_lengths':lengths,'query_prompt':prompt}
    settings={'repo':repo,'revision':revision,'model_files_sha256':model_files,'query_prompt':prompt,'document_prompt':'',
        'pooling':str(model[1]),'max_seq_length':maximum,'padding_side':model.tokenizer.padding_side,
        'batch_size':8,'dtype':str(next(model.parameters()).dtype),'device':str(model.device),'load_seconds':load_seconds,
        'platform':platform.platform(),'versions':{x:importlib.metadata.version(x) for x in ('torch','transformers','sentence-transformers','numpy')}}
    write_json(folder/'config.json',settings);write_json(folder/'token_audit.json',audit)
    def sync():
        if device=='mps':torch.mps.synchronize()
    def encode(items):
        return model.encode(items,prompt='',batch_size=8,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    encode(['warmup']);sync()
    print('ENCODE',args.model,len(texts),'chunks','max_tokens',max(lengths),'truncated',audit['document_truncation_count'],flush=True)
    # Progress checkpoints every 256 chunks; fixed batch size shared by all models.
    arrays=[];sync();start=time.perf_counter()
    for offset in range(0,len(texts),256):
        arrays.append(encode(texts[offset:offset+256]));sync()
        print('PROGRESS',args.model,min(offset+256,len(texts)),'/',len(texts),flush=True)
    vectors=np.concatenate(arrays).astype(np.float32);encoding_seconds=time.perf_counter()-start
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
    np.save(folder/'vectors.npy',vectors)
    query_vectors=encode(queries);sync();np.save(folder/'queries.npy',query_vectors)
    rows=rank_questions(vectors,query_vectors,chunks,cases)
    times=[]
    for _ in range(3):
        for q in queries:
            sync();start=time.perf_counter();encode([q]);sync();times.append((time.perf_counter()-start)*1000)
    write_json(folder/'query_timing.json',{'repeats':3,'warm_encoding_ms_samples':times})
    write_json(folder/'results.json',rows)
    summary={'model':args.model,'repo':repo,'dimension':vectors.shape[1],'chunks':len(chunks),
        'encoding_seconds':encoding_seconds,'vector_bytes':vectors.nbytes,
        'query_encoding_ms_median':float(np.median(times)),'query_encoding_ms_p95':float(np.percentile(times,95)),
        'search_ms_median':float(np.median([r['search_ms'] for r in rows])),
        'document_truncation_count':audit['document_truncation_count'],'query_truncation_count':audit['query_truncation_count'],
        **summarize(rows)}
    # Old no-instruction query policy as a supplementary control on identical small vectors.
    if args.model=='bge-small':
        control=out/'bge-small-no-instruction';control.mkdir(exist_ok=True)
        raw=encode(plain_queries);np.save(control/'queries.npy',raw)
        old_rows=rank_questions(vectors,raw,chunks,cases)
        write_json(control/'results.json',old_rows)
        write_json(control/'summary.json',dict(summary,model='bge-small-no-instruction',query_policy='Legacy query without instruction; shared small document vectors.',**summarize(old_rows)))
    write_json(folder/'summary.json',summary)
    print('DONE',json.dumps(summary),flush=True)


if __name__=='__main__':main()
