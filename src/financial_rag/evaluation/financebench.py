"""FinanceBench pilot: pinned PDFs, shared corpus, baseline/agent comparison.

Gold answers and evidence are scoring-only. Page metrics measure annotated-page
coverage, not semantic relevance, answer correctness, or exhaustive recall.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path
from datetime import datetime, timezone

REVISION = 'cc39aeb4afdf33909ee1412188bf89035950c2eb'
EMBEDDING = 'BAAI/bge-small-en-v1.5'
RERANKER = 'BAAI/bge-reranker-v2-m3'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def select_cases(rows, count=10, seed=20261013):
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    selected, companies = [], set()
    kinds = ['metrics-generated', 'domain-relevant', 'novel-generated']
    for index in range(count):
        row = next((r for r in candidates if r['question_type'] == kinds[index % 3]
                    and r['company'] not in companies), None)
        if row is None:
            raise ValueError('Not enough distinct companies for requested stratified sample')
        selected.append(row)
        companies.add(row['company'])
    return selected


def gold_pages(case):
    pages = set()
    for item in case['evidence']:
        page = item['evidence_page_num']
        if type(page) is not int or page < 0:
            raise ValueError('Evidence page must be a nonnegative zero-based integer')
        doc = item.get('doc_name') or item.get('evidence_doc_name') or case['doc_name']
        pages.add((doc, page + 1))
    if not pages:
        raise ValueError('Missing reference evidence')
    return pages


def page_metrics(results, chunks_by_id, expected):
    ranked = [(chunks_by_id[r['chunk_id']]['document_id'], chunks_by_id[r['chunk_id']]['page'])
              for r in results]
    matched = set(ranked) & expected
    first = next((rank for rank, item in enumerate(ranked, 1) if item in expected), None)
    return {'page_hit': bool(matched), 'annotated_page_coverage': len(matched) / len(expected),
            'all_annotated_pages_hit': matched == expected,
            'reciprocal_rank_by_page': 1 / first if first else 0,
            'matched_pages': sorted(matched), 'expected_pages': sorted(expected),
            'retrieved_chunks': len(ranked), 'retrieved_unique_pages': len(set(ranked))}


def evaluation_question(case):
    # Document identity is public question metadata; never include gold answer/pages.
    return case['question'] + '\nTarget report: ' + case['doc_name'] + '.pdf'


def prepare(folder, count, seed):
    import httpx
    from concurrent.futures import ThreadPoolExecutor
    folder.mkdir(parents=True, exist_ok=True)
    base = f'https://raw.githubusercontent.com/patronus-ai/financebench/{REVISION}'
    for name in ('financebench_open_source.jsonl', 'financebench_document_information.jsonl'):
        response = httpx.get(f'{base}/data/{name}', timeout=90, follow_redirects=True)
        response.raise_for_status()
        (folder / name).write_bytes(response.content)
    rows = [json.loads(line) for line in (folder / 'financebench_open_source.jsonl').read_text().splitlines()]
    selected = select_cases(rows, count, seed)
    names = sorted({r['doc_name'] for r in selected} | {d for r in selected for d, _ in gold_pages(r)})
    def download(name):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('Invalid document name')
        url = f'{base}/pdfs/{name}.pdf'
        response = httpx.get(url, timeout=120, follow_redirects=True)
        response.raise_for_status()
        if not response.content.startswith(b'%PDF'):
            raise ValueError('Source is not a PDF')
        path = folder / f'{name}.pdf'
        path.write_bytes(response.content)
        return dict(document_id=name, filename=path.name, url=url,
                    sha256=hashlib.sha256(response.content).hexdigest(), size_bytes=len(response.content))
    with ThreadPoolExecutor(max_workers=4) as pool:
        docs = list(pool.map(download, names))
    write_json(folder / 'selected.json', selected)
    write_json(folder / 'manifest.json', dict(upstream_commit=REVISION, selection_seed=seed,
               selection='round-robin question types; distinct companies; shared corpus', documents=docs))


def build_index(root, data, manifest):
    from financial_rag.ingestion.parser import parse_pdf
    from financial_rag.ingestion.normalizer import normalize_pages
    from financial_rag.ingestion.chunker import chunk_pages
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.indexing.vector_store import VectorStore
    sources = ['ingestion/parser.py', 'ingestion/normalizer.py', 'ingestion/chunker.py', 'indexing/embeddings.py']
    config = dict(documents=manifest['documents'], embedding=EMBEDDING, chunk_words=500, overlap_words=80,
                  code={name: hashlib.sha256((root/'src/financial_rag'/name).read_bytes()).hexdigest() for name in sources})
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    folder = root/'storage/financebench'/fingerprint[:16]
    store = VectorStore(folder/'vectors.sqlite3')
    embedding = EmbeddingModel(EMBEDDING, model_path=str(root/'models/bge-small-en-v1.5'), local_files_only=True)
    if store.db_path.exists() and (folder/'manifest.json').exists():
        print('Reusing index', str(store.db_path), flush=True)
        info = json.loads((folder/'manifest.json').read_text())
        if info['fingerprint'] != fingerprint:
            raise ValueError('Index fingerprint mismatch')
        return store, embedding, info
    started = time.perf_counter()
    chunks, docs = [], []
    for item in manifest['documents']:
        path = data/item['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('PDF checksum mismatch: ' + item['filename'])
        pages = parse_pdf(path)
        normalized = normalize_pages(pages)
        current = chunk_pages(normalized, item['document_id'], 500, 80)
        for chunk in current:
            chunk['title'] = item['filename']
        chunks.extend(current)
        docs.append({**item, 'pages':len(pages), 'chunks':len(current),
                     'empty_pages': [p['page'] for p in pages if not p['text'].strip()]})
        print('Parsed', item['filename'],len(pages),'pages',len(current),'chunks',flush=True)
    write_json(folder/'chunks.json', chunks)
    print('Encoding',len(chunks),'chunks on',str(embedding.model.device),flush=True)
    vectors = embedding.encode_documents([c['text'] for c in chunks])
    store.replace_all(chunks, vectors, EMBEDDING)
    info = dict(fingerprint=fingerprint, config=config, documents=docs, chunks=len(chunks),
                pages=sum(d['pages'] for d in docs), build_seconds=round(time.perf_counter()-started,3))
    write_json(folder/'manifest.json', info)
    return store, embedding, info


def summarize(output):
    records = [json.loads(p.read_text()) for p in sorted((output/'cases').glob('*.json'))]
    summary = {}
    for mode in ('baseline','agent'):
        subset = [r for r in records if r['mode']==mode]
        successful = [r for r in subset if 'metrics' in r]
        n = len(successful)
        summary[mode] = dict(attempted=len(subset), scored=n,
            errors=sum(r['result'].get('status')=='error' for r in subset),
            completed=sum(r['result'].get('status')=='completed' for r in subset),
            page_hit_count=sum(r['metrics']['page_hit'] for r in successful),
            all_annotated_pages_hit_count=sum(r['metrics']['all_annotated_pages_hit'] for r in successful),
            mean_annotated_page_coverage=sum(r['metrics']['annotated_page_coverage'] for r in successful)/n if n else None,
            total_model_calls=sum(r['model_calls'] for r in subset),
            total_seconds=round(sum(r['elapsed_seconds'] for r in subset),3))
    write_json(output/'summary.json', summary)
    fields=['id','mode','question','reference_answer','answer','status','page_hit','annotated_page_coverage',
            'all_annotated_pages_hit','retrieved_chunks','searches','model_calls','elapsed_seconds']
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in records:
            result=row['result']; metrics=row.get('metrics',{})
            writer.writerow(dict(id=row['id'],mode=row['mode'],question=row['question'],reference_answer=row['reference_answer'],
                answer=result.get('answer',''),status=result.get('status'),
                **{k:metrics.get(k) for k in fields[6:10]}, searches=result.get('search_attempts',1),
                model_calls=row['model_calls'],elapsed_seconds=row['elapsed_seconds']))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','index','run','all'],default='run')
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    data=root/'data/raw/financebench'
    if args.phase in ('prepare','all'):
        prepare(data,args.count,args.seed)
    if args.phase=='prepare': return
    manifest=json.loads((data/'manifest.json').read_text())
    cases=json.loads((data/'selected.json').read_text())
    store,embedding,info=build_index(root,data,manifest)
    if args.phase=='index': return
    from dotenv import load_dotenv
    from financial_rag.retrieval.reranker import Reranker
    from financial_rag.generation.generator import DeepSeekGenerator
    from financial_rag.pipeline import FinancialRAG
    load_dotenv(root/'.env')
    generator=DeepSeekGenerator()
    usage=[]
    original_create=generator.client.chat.completions.create
    def tracked_create(**kwargs):
        record={'model':kwargs['model']};usage.append(record)
        response=original_create(**kwargs)
        record.update(response.usage.model_dump() if response.usage else {})
        return response
    generator.client.chat.completions.create=tracked_create
    reranker=Reranker(RERANKER,2048,model_path=str(root/'models/bge-reranker-v2-m3'),local_files_only=True)
    rag=FinancialRAG(store,embedding,EMBEDDING,reranker,generator)
    output=args.output or root/'outputs/financebench'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=True)
    configuration=dict(dataset=manifest, index=info, generation_model=generator.model,
        selected_sha256=hashlib.sha256((data/'selected.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/financial_rag').rglob('*.py'))},
        reranker=RERANKER,reranker_max_length=2048,candidate_k=20,top_k=5,rrf_k=60,
        question_policy='Original question plus target report filename; shared-corpus retrieval without document filter; no gold evidence in prompts',
        agent_policy='Project defaults: query planning + max_steps=7, max_searches=8; independent questions without conversation history',
        metric_note='Page matches only; not answer accuracy or exhaustive recall. Agent union has a larger evidence budget than baseline top 5.')
    old=output/'config.json'
    if old.exists() and json.loads(old.read_text())!=configuration:
        raise ValueError('Output belongs to a different run configuration')
    write_json(old,configuration)
    write_json(output/'selected_cases.json',cases)
    for case in cases:
        expected=gold_pages(case)
        for doc,page in expected:
            if not any(c['document_id']==doc and c['page']==page for c in rag.chunks_by_id.values()):
                raise ValueError(f'Annotated page missing from parsed corpus: {doc} page {page}')
        for mode in ('baseline','agent'):
            path=output/'cases'/f"{case['financebench_id']}_{mode}.json"
            if path.exists():
                print('Resume: skip',path.name,flush=True);continue
            print('RUN',case['financebench_id'],mode,flush=True)
            usage.clear();started=time.perf_counter()
            try:
                question=evaluation_question(case)
                if mode=='baseline':
                    result=rag.ask(question);result['status']='completed';result['search_attempts']=1
                    traces=[result['retrieval']]; final=result['retrieval']['reranked']
                else:
                    result=rag.ask_agent(question)
                    traces=result['retrieval_rounds'];final=list(result['evidence'].values())
                stage_metrics=[{stage:page_metrics(trace[stage],rag.chunks_by_id,expected)
                                for stage in ('dense','bm25','hybrid','reranked')} for trace in traces]
                metrics=page_metrics(final,rag.chunks_by_id,expected)
                record=dict(id=case['financebench_id'],mode=mode,question=case['question'],
                    reference_answer=case['answer'],gold_evidence=case['evidence'],result=result,
                    metrics=metrics,stage_metrics=stage_metrics,model_calls=len(usage),usage=list(usage),
                    elapsed_seconds=round(time.perf_counter()-started,3))
            except Exception as error:
                record=dict(id=case['financebench_id'],mode=mode,question=case['question'],reference_answer=case['answer'],
                    result={'status':'error','error_type':type(error).__name__,'http_status':getattr(error,'status_code',None)},model_calls=len(usage),usage=list(usage),
                    elapsed_seconds=round(time.perf_counter()-started,3))
            write_json(path,record)
            summarize(output)
            print('DONE',mode,record['result']['status'],record.get('metrics',{}),record['elapsed_seconds'],'seconds',flush=True)
            if record['result'].get('status')=='error':
                raise RuntimeError('Run stopped after error; saved diagnostic type and completed cases for resume')
    print(json.dumps(summarize(output),indent=2),flush=True)
    print('REPORT',output,flush=True)


if __name__=='__main__':
    main()
