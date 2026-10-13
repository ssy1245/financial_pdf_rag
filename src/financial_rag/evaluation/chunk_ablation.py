"""Offline dense-only FinanceBench chunk ablation; no generation/API calls."""
import argparse
import bisect
import csv
import hashlib
import html
import json
import re
import time
from pathlib import Path

import numpy as np

from financial_rag.evaluation.financebench import gold_pages, evaluation_question, write_json
from financial_rag.ingestion.chunker import chunk_pages


def token_chunks(pages, document_id, tokenizer, size, overlap):
    """Page-local source spans. Prefer paragraph ends, then whole-word token windows.

    size counts content tokens only. overlap is a target maximum suffix length;
    whole-word alignment may reduce overlap. No cross-page overlap.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Invalid chunk size/overlap')
    chunks = []
    for page in pages:
        text = page['text']
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        offsets = encoded['offset_mapping']
        if not offsets:
            if text.strip():
                chunks.append({'document_id':document_id,'page':page['page'],
                    'chunk_id':f'{document_id}_p{page["page"]}_c0','text':text,
                    'word_count':len(text.split()),'char_start':0,'char_end':len(text)})
            continue
        starts = [a for a, b in offsets]
        ends = [b for a, b in offsets]
        boundaries = [m.start() for m in re.finditer(r'\n\s*\n', text)] + [len(text)]
        start = 0
        index = 0
        previous_stop = 0
        previous_char_end = 0
        while start < len(offsets):
            stop = min(start + size, len(offsets))
            # Prefer complete paragraph ends within the content-token window.
            lower = max(starts[start], ends[previous_stop-1] if previous_stop else 0)
            candidates = [b for b in boundaries if lower < b <= ends[stop-1]]
            if candidates:
                candidate = bisect.bisect_right(ends, candidates[-1])
                if candidate > start:
                    stop = candidate
            # Do not end in the middle of a source word, unless one word exceeds size.
            original_stop = stop
            while stop > start and stop < len(offsets) and not any(c.isspace() for c in text[ends[stop-1]:starts[stop]]):
                stop -= 1
            if stop == start:
                stop = original_stop
            begin = min(starts[start], previous_char_end)
            end = len(text) if stop == len(offsets) else ends[stop-1]
            current = text[begin:end]
            # WordPiece boundary tokenization can change when slicing; enforce actual length.
            while len(tokenizer(current, add_special_tokens=False)['input_ids']) > size:
                stop -= 1
                if stop <= start:
                    raise ValueError('Cannot fit token span')
                end = ends[stop-1]
                current = text[begin:end]
            chunks.append({'document_id':document_id, 'page':page['page'],
                'chunk_id':f'{document_id}_p{page["page"]}_c{index}', 'text':current,
                'word_count':len(current.split()), 'char_start':begin, 'char_end':end})
            index += 1
            if stop == len(offsets):
                break
            # Progress even when a short paragraph is smaller than the overlap target.
            next_start = max(start+1, stop-overlap)
            while next_start < stop and next_start > 0 and not any(c.isspace() for c in text[ends[next_start-1]:starts[next_start]]):
                next_start += 1
            previous_char_end = end
            previous_stop = stop
            start = next_start
    return chunks


def select_budget(order, lengths, budget):
    """Ranked prefix of whole chunks; no skipping or clipping to fit remaining budget."""
    selected, total = [], 0
    for i in order:
        cost = lengths[int(i)]
        if total + cost > budget:
            break
        selected.append(int(i)); total += cost
    return selected, total


def metric(indices, chunks, expected):
    pages = [(chunks[int(i)]['document_id'], chunks[int(i)]['page']) for i in indices]
    matched = set(pages) & expected
    first = next((i for i,p in enumerate(pages,1) if p in expected), None)
    return {'hit':int(bool(matched)), 'coverage':len(matched)/len(expected),
            'all_pages':int(matched==expected), 'mrr':1/first if first else 0,
            'chunks':len(indices), 'unique_pages':len(set(pages)),
            'same_page_repeat_fraction':1-len(set(pages))/len(pages) if pages else 0,
            'matched_pages':sorted(matched)}


def compact(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def evidence_audit(cases, pages_by_doc, chunks, tokenizer, content_limit):
    """Literal normalized-text containment only, not a semantic correctness metric."""
    by_page = {}
    for c in chunks:
        key=(c['document_id'],c['page'])
        by_page.setdefault(key,[]).append(c)
    rows=[]
    for case in cases:
        for ev in case['evidence']:
            doc=ev.get('doc_name') or ev.get('evidence_doc_name') or case['doc_name']
            page=ev['evidence_page_num']+1
            gold=compact(ev['evidence_text'])
            page_text=next(p['text'] for p in pages_by_doc[doc] if p['page']==page)
            group=by_page.get((doc,page),[])
            visible=[]
            for c in group:
                offsets=tokenizer(c['text'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                end=offsets[min(content_limit,len(offsets))-1][1] if offsets else 0
                visible.append(c['text'][:end])
            rows.append({'id':case['financebench_id'],'document_id':doc,'page':page,
                'gold_words':len(gold.split()),'gold_tokens':len(tokenizer(ev['evidence_text'],add_special_tokens=False)['input_ids']),
                'literal_in_parsed_page':bool(gold) and gold in compact(page_text),
                'literal_in_one_chunk':any(gold in compact(c['text']) for c in group) if gold else False,
                'literal_in_one_visible_chunk':any(gold in compact(t) for t in visible) if gold else False,
                'page_chunks':len(group)})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    from sentence_transformers import SentenceTransformer
    from financial_rag.ingestion.parser import parse_pdf
    from financial_rag.ingestion.normalizer import normalize_pages
    import torch
    import platform
    import importlib.metadata
    data=root/'data/raw/financebench'
    manifest=json.loads((data/'manifest.json').read_text())
    cases=json.loads((data/'selected.json').read_text())
    model_path=root/'models/bge-small-en-v1.5'
    model=SentenceTransformer(str(model_path),local_files_only=True)
    from copy import deepcopy
    tokenizer=deepcopy(model.tokenizer)
    tokenizer.model_max_length=10**9  # audit without warning/truncation; model.max_seq_length remains 512
    maximum=model.max_seq_length
    assert maximum == 512, f"Unexpected baseline model limit: {maximum}"
    special=tokenizer.num_special_tokens_to_add(pair=False)
    content_limit=maximum-special
    configs=[('words500_overlap80',500,80,'words')]+[(f'tokens{s}_overlap{o}',s,o,'tokens') for s in (200,300,400) for o in (0,int(s*.15))]
    source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),root/'src/financial_rag/ingestion/parser.py',root/'src/financial_rag/ingestion/normalizer.py',root/'src/financial_rag/ingestion/chunker.py']}
    model_hashes={str(p.relative_to(model_path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(model_path.rglob('*')) if p.is_file() and not p.name.startswith('.')}
    config={'model':'BAAI/bge-small-en-v1.5','model_files_sha256':model_hashes,'max_seq_length':maximum,
        'special_tokens':special,'content_limit':content_limit,'device':str(model.device),'batch_size':32,
        'torch_threads':torch.get_num_threads(),'dtype':str(next(model.parameters()).dtype),
        'platform':platform.platform(),'versions':{x:importlib.metadata.version(x) for x in ('torch','sentence-transformers','transformers','numpy','pymupdf')},
        'source_sha256':source_hashes,'dataset':manifest,'selected_sha256':hashlib.sha256((data/'selected.json').read_bytes()).hexdigest(),
        'configs':configs,'query_policy':'Original question + target filename, no instruction or rewrite; same as existing baseline',
        'retrieval':'Dense normalized dot product only. No BM25/reranker/Agent/generation.',
        'budget_policy':'2000 content tokens of full returned chunk text, ranked prefix; no clipping/skipping/deduplication.',
        'limitation':'Page-level labels; 10 previously inspected development cases; no held-out test or answer scoring.'}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Existing output configuration mismatch; choose another output folder')
    write_json(out/'config.json',config)
    write_json(out/'selected_cases.json',cases)
    pages_by_doc={}
    for doc in manifest['documents']:
        path=data/doc['filename']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=doc['sha256']:
            raise ValueError('PDF hash mismatch')
        pages_by_doc[doc['document_id']]=normalize_pages(parse_pdf(path))
    write_json(out/'normalized_pages.json',pages_by_doc)
    questions=[evaluation_question(c) for c in cases]
    query_lengths=[len(tokenizer(q,add_special_tokens=True)['input_ids']) for q in questions]
    write_json(out/'query_audit.json',{'lengths':query_lengths,'truncated':sum(n>maximum for n in query_lengths)})
    def sync():
        if model.device.type=='mps': torch.mps.synchronize()
        elif model.device.type=='cuda': torch.cuda.synchronize()
    model.encode(['warmup'],normalize_embeddings=True,show_progress_bar=False)
    query_vectors=model.encode(questions,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    query_times=[]
    for _ in range(3):
        for q in questions:
            sync();begin=time.perf_counter()
            model.encode(q,normalize_embeddings=True,show_progress_bar=False)
            sync();query_times.append((time.perf_counter()-begin)*1000)
    write_json(out/'query_timing.json',{'repeats':3,'warm_encoding_ms_median':float(np.median(query_times)),
        'warm_encoding_ms_p95':float(np.percentile(query_times,95)),'samples_ms':query_times})
    summaries=[]
    for name,size,overlap,unit in configs:
        folder=out/name;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():
            summaries.append(json.loads((folder/'summary.json').read_text()));print('RESUME',name,flush=True);continue
        begin=time.perf_counter();chunks=[]
        for doc,pages in pages_by_doc.items():
            current=chunk_pages(pages,doc,size,overlap) if unit=='words' else token_chunks(pages,doc,tokenizer,size,overlap)
            chunks.extend(current)
        chunk_seconds=time.perf_counter()-begin
        texts=[c['text'] for c in chunks]
        lengths=[len(ids) for ids in tokenizer(texts,add_special_tokens=False,truncation=False)['input_ids']]
        assert unit=='words' or max(lengths)<=size
        if unit=='tokens':
            # Verify source-span union covers every non-whitespace character.
            for doc,pages in pages_by_doc.items():
                for page in pages:
                    spans=sorted((c['char_start'],c['char_end']) for c in chunks if c['document_id']==doc and c['page']==page['page'])
                    covered=0
                    for a,b in spans:
                        assert not page['text'][covered:a].strip(), (doc,page['page'],covered,a)
                        covered=max(covered,b)
                    assert not page['text'][covered:].strip()
        for c,n in zip(chunks,lengths):
            c['content_tokens']=n;c['truncated_tokens']=max(0,n-content_limit)
        write_json(folder/'chunks.json',chunks)
        audit=evidence_audit(cases,pages_by_doc,chunks,tokenizer,content_limit)
        write_json(folder/'evidence_audit.json',audit)
        print('ENCODE',name,'chunks',len(chunks),'truncated',sum(n>content_limit for n in lengths),flush=True)
        sync();begin=time.perf_counter()
        vectors=model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
        sync();encoding_seconds=time.perf_counter()-begin
        np.save(folder/'vectors.npy',vectors)
        rows=[];search_times=[]
        for case,qv in zip(cases,query_vectors):
            begin=time.perf_counter();scores=vectors@qv
            order=np.argsort(-scores,kind='stable');search_times.append((time.perf_counter()-begin)*1000)
            expected=gold_pages(case)
            row={'id':case['financebench_id'],'company':case['company'],'question':case['question'], 'expected_pages':sorted(expected)}
            for k in (5,10,20): row[f'top{k}']=metric(order[:k],chunks,expected)
            budget_indices,used=select_budget(order,lengths,2000)
            row['budget2000']=dict(metric(budget_indices,chunks,expected),used_tokens=used)
            row['ranking']=[{'chunk_id':chunks[int(i)]['chunk_id'],'document_id':chunks[int(i)]['document_id'],
                'page':chunks[int(i)]['page'],'content_tokens':lengths[int(i)],'score':float(scores[i]),
                'gold_page':(chunks[int(i)]['document_id'],chunks[int(i)]['page']) in expected} for i in order[:max(20,len(budget_indices))]]
            rows.append(row)
        write_json(folder/'results.json',rows)
        stat={'name':name,'size':size,'overlap':overlap,'unit':unit,'chunks':len(chunks),
            'token_p50':float(np.median(lengths)),'token_p95':float(np.percentile(lengths,95)), 'token_max':max(lengths),
            'truncated_chunks':sum(n>content_limit for n in lengths),'truncated_fraction':sum(n>content_limit for n in lengths)/len(lengths),
            'total_content_tokens':sum(lengths),'discarded_tokens':sum(max(0,n-content_limit) for n in lengths),
            'chunk_seconds':chunk_seconds,'encoding_seconds':encoding_seconds,'vector_bytes':vectors.nbytes,
            'search_ms_median':float(np.median(search_times)),
            'gold_evidence_records':len(audit),'literal_page_matches':sum(r['literal_in_parsed_page'] for r in audit),
            'literal_chunk_matches':sum(r['literal_in_one_chunk'] for r in audit),
            'literal_visible_matches':sum(r['literal_in_one_visible_chunk'] for r in audit)}
        for mode in ('top5','top10','top20','budget2000'):
            stat[mode]={k:float(np.mean([r[mode][k] for r in rows])) for k in ('hit','coverage','all_pages','mrr','chunks','same_page_repeat_fraction')}
        write_json(folder/'summary.json',stat);summaries.append(stat)
        write_json(out/'summary.json',summaries)
        print('DONE',name, json.dumps({k:stat[k] for k in ('encoding_seconds','top5','top20','budget2000')}),flush=True)
    write_json(out/'summary.json',summaries)
    fields=['name','chunks','truncated_fraction','token_p50','token_p95','token_max','encoding_seconds','vector_bytes']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as file:
        cols=fields+[f'{m}_{k}' for m in ('top5','top10','top20','budget2000') for k in ('hit','coverage','all_pages','mrr')]
        writer=csv.DictWriter(file,fieldnames=cols);writer.writeheader()
        for s in summaries:writer.writerow({**{k:s[k] for k in fields},**{f'{m}_{k}':s[m][k] for m in ('top5','top10','top20','budget2000') for k in ('hit','coverage','all_pages','mrr')}})
    print('FINISHED',out,flush=True)


if __name__=='__main__': main()
