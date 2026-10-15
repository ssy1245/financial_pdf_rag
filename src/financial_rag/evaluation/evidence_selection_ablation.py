"""Local-only matched10 evidence selection. Exactly ten retrieval/rerank calls.

Baseline Top7; thresholds .1/.3/.5; same thresholds x MMR lambda .6/.8.
No generator imports, no API keys loaded, outbound IP connections blocked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import numpy as np
from financial_rag.evaluation.financebench import evaluation_question, write_json
from financial_rag.evaluation.retrieval_k_ablation import evaluate_ids
from financial_rag.retrieval.selection import select_evidence

GRID=[dict(name='top7',threshold=None,weight=None)]+[
    dict(name=f'threshold_{t}',threshold=t,weight=None) for t in (.1,.3,.5)]+[
    dict(name=f'mmr_t{t}_w{w}',threshold=t,weight=w) for t in (.1,.3,.5) for w in (.6,.8)]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.project_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    network_attempts=[];connect=socket.socket.connect
    def local_only(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):
            network_attempts.append(str(address));raise RuntimeError('Network disabled in retrieval-only experiment')
        return connect(sock,address)
    socket.socket.connect=local_only
    data=root/'outputs/matched_query_ablation/revenue10_20261015';frozen=root/'outputs/retrieval_k_ablation/financebench50_20261015'
    cases=json.loads((data/'selected_cases.json').read_text());criteria=json.loads((data/'evidence_criteria.json').read_text());chunks=json.loads((frozen/'chunks.json').read_text());vectors=np.load(frozen/'vectors.npy')
    assert len(cases)==10 and vectors.shape==(18806,384)
    byid={c['chunk_id']:c for c in chunks};vec={c['chunk_id']:v for c,v in zip(chunks,vectors)}
    source_names=['retrieval/selection.py','retrieval/retriever.py','retrieval/reranker.py','retrieval/fusion.py','indexing/bm25_index.py','evaluation/evidence_selection_ablation.py','evaluation/evidence_content.py','evaluation/retrieval_k_ablation.py']
    config=dict(grid=GRID,top_k=7,candidate_k=20,rrf_k=60,query_policy='Original question plus public target report filename; no rewrite, no historical generated query, no metadata filtering.',
                cases_sha256=sha(data/'selected_cases.json'),criteria_sha256=sha(data/'evidence_criteria.json'),chunks_sha256=sha(frozen/'chunks.json'),vectors_sha256=sha(frozen/'vectors.npy'),
                source_sha256={n:sha(root/'src/financial_rag'/n) for n in source_names},
                policy='All strategies reuse exactly the same20 scored candidates per query. Raw CrossEncoder scores thresholded at0.1/0.3/0.5, not calibrated probabilities. MMR uses normalized cached document vectors, relevance weights0.6/0.8. At most7 chunks, no backfill below threshold. Gold and review annotations are scoring-only.',
                evaluation='Same10 cases:7 FinanceBench+3 custom; single local retrieval each, no loop or generation. User-requested small exploratory development experiment; review selected candidate union before aggregated strategy usefulness metrics.')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    write_json(out/'config.json',config)
    for n in ['selected_cases.json','evidence_criteria.json']:shutil.copyfile(data/n,out/n)
    for n in source_names:
        dst=out/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/'src/financial_rag'/n,dst)
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    from financial_rag.retrieval.retriever import Retriever
    embedding=EmbeddingModel(model_path=str(root/'models/bge-small-en-v1.5'),local_files_only=True)
    reranker=Reranker('BAAI/bge-reranker-v2-m3',2048,model_path=str(root/'models/bge-reranker-v2-m3'),local_files_only=True)
    class Store:
        def load(self,name):return chunks,vectors
    retriever=Retriever(Store(),embedding,'BAAI/bge-small-en-v1.5',reranker,candidate_k=20,top_k=7)
    embedding.encode_query('warmup');reranker.rerank('warmup',[chunks[0]],top_k=1)
    rows=[];review=[]
    for case,rubric in zip(cases,criteria):
        query=evaluation_question(case);start=time.perf_counter();trace=retriever.retrieve(query,top_k=20);elapsed=time.perf_counter()-start
        scored=trace['reranked'];assert len(scored)==20 and all(0<=c['rerank_score']<=1 for c in scored)
        choices={};union=set()
        for cfg in GRID:
            start=time.perf_counter();selected=select_evidence(scored,top_k=7,threshold=cfg['threshold'],diversity_weight=cfg['weight'],vectors=vec);seconds=time.perf_counter()-start
            ids=[c['chunk_id'] for c in selected];union.update(ids)
            choices[cfg['name']]=dict(ids=ids,scores=[c['rerank_score'] for c in selected],metrics=evaluate_ids(ids,byid,case,rubric),selection_seconds=seconds)
        row=dict(id=case['financebench_id'],company=case['company'],cohort=case['cohort'],query=query,retrieval_seconds=elapsed,
                 reranked=scored,trace=trace,selections=choices,candidate_coverage=evaluate_ids([c['chunk_id'] for c in scored],byid,case,rubric))
        write_json(out/'cases'/(row['id']+'.json'),row);rows.append(row)
        # No ranking, strategy assignment, or scores in review cards.
        cards=[dict(chunk_id=cid,document_id=byid[cid]['document_id'],page=byid[cid]['page'],text=byid[cid]['text'],content_tokens=byid[cid]['content_tokens']) for cid in sorted(union)]
        review.append(dict(id=row['id'],question=case['question'],target_report=case['doc_name'],cards=cards))
        print('DONE',case['company'],'review_chunks',len(cards),'seconds',round(elapsed,2),flush=True)
    write_json(out/'results.json',rows);write_json(out/'review_cards.json',review)
    for r in review:write_json(out/'review_cards'/(r['id']+'.json'),r)
    verify=dict(questions=10,strategies=len(GRID),evaluated_selections=10*len(GRID),generation_api_calls=0,network_connect_attempts=len(network_attempts),
                scored_candidates=200,unique_query_chunk_pairs_to_review=sum(len(r['cards']) for r in review),
                max_query_embedding_tokens=max(len(embedding.model.tokenizer.encode(evaluation_question(c))) for c in cases),
                reranker_truncated=sum(len(r['trace']['reranker_truncated_chunk_ids']) for r in rows),
                same_frozen_candidates_per_case=True,source_unchanged=all(sha(root/'src/financial_rag'/n)==h for n,h in config['source_sha256'].items()),
                timing='One local retrieval+rerank per query after warmup; all selections share that cost. Additional selection_seconds recorded separately.',
                application_defaults_unchanged=True)
    assert verify['source_unchanged'] and not network_attempts
    write_json(out/'verification.json',verify);print('FINISHED',json.dumps(verify),flush=True)


if __name__=='__main__':main()
