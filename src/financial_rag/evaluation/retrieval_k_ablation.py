"""FinanceBench50 candidate20/50 x reranked Top5/7/10; local retrieval only.

Use the frozen 300/0 corpus, fresh BGE-small FP32 document/query vectors with
no query instruction (application policy), and the application BGE reranker.
Each candidate pool is scored once per question; K is a ranking prefix.
"""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from financial_rag.evaluation.financebench import evaluation_question, gold_pages, write_json
from financial_rag.evaluation.chunk_ablation import metric, select_budget
from financial_rag.evaluation.evidence_content import score_content

CANDIDATES=(20,50)
TOP_K=(5,7,10)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_ids(ids, chunks_by_id, case, criteria):
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate returned chunk IDs')
    chunks=[chunks_by_id[i] for i in ids]
    content=score_content(chunks,criteria)
    supporting={cid for m in content['matches'] for cid in m['chunk_ids']}
    return dict(page=metric(range(len(chunks)),chunks,gold_pages(case)),content=content,
        content_tokens=sum(c['content_tokens'] for c in chunks),
        returned_chunks=len(chunks),unique_documents=len({c['document_id'] for c in chunks}),
        annotated_support_chunk_fraction=(len(supporting)/len(chunks) if chunks else 0) if content['coverage'] is not None else None,
        chunk_ids=ids)


def aggregate(scores):
    automatic=[s for s in scores if s['content']['coverage'] is not None]
    mean=lambda xs:float(np.mean(xs)) if xs else None
    return dict(questions=len(scores),content_questions=len(automatic),
        page_hit=mean([s['page']['hit'] for s in scores]),page_coverage=mean([s['page']['coverage'] for s in scores]),
        all_pages=mean([s['page']['all_pages'] for s in scores]),mrr=mean([s['page']['mrr'] for s in scores]),
        content_hit=mean([s['content']['hit'] for s in automatic]),
        content_coverage=mean([s['content']['coverage'] for s in automatic]),
        all_units=mean([s['content']['all_units'] for s in automatic]),
        annotated_support_chunk_fraction=mean([s['annotated_support_chunk_fraction'] for s in automatic]),
        content_tokens_mean=mean([s['content_tokens'] for s in scores]),
        content_tokens_p95=float(np.percentile([s['content_tokens'] for s in scores],95)),
        chunks_mean=mean([s['returned_chunks'] for s in scores]))


def summarize(rows):
    summaries=[]
    for n in CANDIDATES:
        for k in TOP_K:
            name=f'c{n}_top{k}'
            metrics=aggregate([r['pools'][str(n)]['selections'][f'top{k}'] for r in rows])
            budget=aggregate([r['pools'][str(n)]['selections'][f'top{k}_budget2000'] for r in rows])
            times=[r['timing']['retrieval_ms']+r['pools'][str(n)]['timing']['fusion_ms']+r['pools'][str(n)]['timing']['rerank_ms'] for r in rows]
            summaries.append(dict(name=name,candidate_k=n,top_k=k,**metrics,budget2000=budget,
                retrieval_rerank_ms_median=float(np.median(times)),retrieval_rerank_ms_p95=float(np.percentile(times,95)),
                timing_note='Dense+BM25 Top50 shared across candidate groups; excludes model loading/document encoding/query encoding. Same candidate pool is reranked once; K prefixes share its measured time.',
                original10=aggregate([r['pools'][str(n)]['selections'][f'top{k}'] for r in rows[:10]]),
                additional40=aggregate([r['pools'][str(n)]['selections'][f'top{k}'] for r in rows[10:]])))
    return summaries


def paired_comparisons(rows):
    rng=np.random.default_rng(20261015)
    comparisons={}
    for n in CANDIDATES:
        for k in TOP_K:
            if (n,k)==(20,5):continue
            pairs=[]
            for r in rows:
                a=r['pools']['20']['selections']['top5']['content']
                b=r['pools'][str(n)]['selections'][f'top{k}']['content']
                if a['coverage'] is None:continue
                pairs.append(dict(id=r['id'],company=r['company'],baseline=a['coverage'],new=b['coverage'],delta=b['coverage']-a['coverage'],baseline_all=a['all_units'],new_all=b['all_units']))
            deltas=np.array([p['delta'] for p in pairs])
            boot=np.mean(rng.choice(deltas,size=(10000,len(deltas)),replace=True),axis=1)
            comparisons[f'c{n}_top{k}']=dict(n=len(pairs),coverage_delta_mean=float(deltas.mean()),
                paired_bootstrap95=[float(x) for x in np.percentile(boot,[2.5,97.5])],
                improved=sum(deltas>1e-12).item(),worse=sum(deltas<-1e-12).item(),unchanged=sum(abs(deltas)<=1e-12).item(),cases=pairs,
                caveat='Exploratory paired bootstrap over previously used development questions; not an independent test or a causal estimate of LLM call savings.')
    return comparisons


def report(out,rows,summaries,chunks_by_id,config):
    e=lambda x:html.escape(str(x));pct=lambda x:f'{x:.1%}'
    table=lambda headers,values:'<div class="scroll"><table><tr>'+''.join('<th>'+e(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+e(v)+'</td>' for v in row)+'</tr>' for row in values)+'</table></div>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinanceBench50：Top5 / Top7 / Top10</title><style>body{font:16px/1.65 system-ui;color:#223047;background:#f4f6fa;margin:0}main{max-width:1300px;margin:auto;padding:32px}h1{font-size:30px}table{border-collapse:collapse;background:white;white-space:nowrap;width:100%}td,th{padding:10px;border:1px solid #dce1eb;text-align:left}th{background:#e9eef7}.scroll{overflow:auto}details{background:white;border:1px solid #dce1eb;padding:12px;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:14px}a{color:#175eb1}.note{padding:16px;background:#e6eefb;border-radius:8px}</style><main><h1>FinanceBench50：候选 20/50 × Top 5/7/10</h1>'''
    best=max(summaries,key=lambda s:s['content_coverage']);base=summaries[0]
    page+=f'<p class="note">50 题、41 份完整报告、18,806 个固定 300-token / overlap 0 片段。基线 c20/top5 平均证据覆盖率 {pct(base["content_coverage"])}；本轮最高为 {e(best["name"])} 的 {pct(best["content_coverage"])}。只评测检索，不调用生成 LLM，不能据此宣称回答准确率或调用次数改善。分数为对齐标注证据的规则指标。</p>'
    page+='<h2>六组主要结果</h2><p>页级 50 题；内容自动评分 49 题，Ulta 收购问题单独保留人工复核。证据覆盖为逐题覆盖比例的宏平均；全部找全率更接近“是否已具备所需标注证据”，仍不代表答案一定正确。</p>'
    page+=table(['候选','TopK','页命中','平均页覆盖','有效证据命中','平均证据覆盖','全部证据找全','支持片段比例*','平均正文tokens','P95正文tokens','检索+重排中位秒'],[[s['candidate_k'],s['top_k'],pct(s['page_hit']),pct(s['page_coverage']),pct(s['content_hit']),pct(s['content_coverage']),pct(s['all_units']),pct(s['annotated_support_chunk_fraction']),f"{s['content_tokens_mean']:.0f}",f"{s['content_tokens_p95']:.0f}",f"{s['retrieval_rerank_ms_median']/1000:.2f}"] for s in summaries])
    page+='<p>*支持片段比例仅指参与现有标注证据匹配的片段比例；未匹配片段可能仍有用，不能将其补数称为真实无关率。正文tokens使用固定BGE-small tokenizer，不含提示词、引用包装和历史，也不是DeepSeek实际计费tokens。相同候选池的三个K共用一次重排，表中耗时一致；最终LLM阅读成本尚未测量。</p>'
    page+='<h2>固定 2000-token 预算</h2><p>各组最多返回K块，再取不超过2000正文tokens的排名前缀；不跳过、不裁剪。这有助于识别TopK的收益中有多少来自增加输入长度。</p>'
    page+=table(['配置','证据覆盖','全部证据找全','平均片段数','平均正文tokens'],[[s['name'],pct(s['budget2000']['content_coverage']),pct(s['budget2000']['all_units']),f"{s['budget2000']['chunks_mean']:.1f}",f"{s['budget2000']['content_tokens_mean']:.0f}"] for s in summaries])
    page+='<h2>瓶颈定位：证据在哪里被丢掉</h2><p>并集和RRF候选的分数仅用于诊断：未出现在并集是召回问题；并集有而RRF池没有是融合截断；RRF池有而最终TopK没有是重排/截取问题。跨阶段覆盖按实际片段组合评分，不假设每个证据都能由单块覆盖。</p>'
    stages=[]
    for n in CANDIDATES:
        for stage in ('union','rrf'):
            agg=aggregate([r['pools'][str(n)]['stages'][stage] for r in rows])
            stages.append([n,stage,pct(agg['content_coverage']),pct(agg['all_units']),f"{agg['chunks_mean']:.1f}"])
    page+=table(['每路候选','阶段','证据覆盖','全部找全','平均块数'],stages)
    page+='<h2>原10题与新增40题</h2>'+table(['配置','原10证据覆盖','新增40证据覆盖（39自动）'],[[s['name'],pct(s['original10']['content_coverage']),pct(s['additional40']['content_coverage'])] for s in summaries])
    page+='<h2>逐题证据和变化</h2>'
    page+=table(['公司','问题ID']+[s['name'] for s in summaries],[[r['company'],r['id']]+[('人工' if r['pools'][str(s['candidate_k'])]['selections'][f'top{s["top_k"]}']['content']['coverage'] is None else pct(r['pools'][str(s['candidate_k'])]['selections'][f'top{s["top_k"]}']['content']['coverage'])) for s in summaries] for r in rows])
    for r in rows:
        page+='<details><summary>'+e(r['company']+' · '+r['id'])+'</summary><p>'+e(r['question'])+'</p>'
        for n in CANDIDATES:
            pool=r['pools'][str(n)]
            page+=f'<details><summary>候选 {n}：重排前10原文（分界：5 / 7 / 10）</summary>'
            for rank,hit in enumerate(pool['reranked'][:10],1):
                c=chunks_by_id[hit['chunk_id']]
                page+='<p><b>'+e(f'#{rank} {c["document_id"]} · 物理页 {c["page"]} · {c["content_tokens"]} tokens · rerank {hit["rerank_score"]:.4f}')+'</b></p><pre>'+e(c['text'])+'</pre>'
            page+='<p>各K实际匹配依据：</p><pre>'+e(json.dumps({f'top{k}':pool['selections'][f'top{k}']['content'] for k in TOP_K},ensure_ascii=False,indent=2))+'</pre></details>'
        page+='</details>'
    page+='<h2>实验方法与限制</h2><ul><li>沿用已经下载的50题、41份全报告和冻结分块，不重做PDF解析或结构化分块。原10题及新增40题均曾用于开发，不是未见测试集。</li><li>查询使用英文原题＋目标报告名，不加BGE查询指令，与应用编码策略一致。重新用BGE-small FP32编码文档和查询，不混用上一轮带指令的FP16查询向量。不给检索器标准答案、标注页码或证据规则。</li><li>每路Dense/BM25分别取20或50，RRF(k=60)再次保留20或50，再交给应用BGE-reranker-v2-m3（2048 token上限、batch8）。每个池重排一次，取Top5/7/10。</li><li>计时排除模型加载、文档编码和查询编码；包含共享的Dense/BM25 Top50获取、各池融合与各池实际重排。候选20的检索部分因此是保守的共享测量，不能当作独立完整请求延迟。额外记录查询编码和模型预热。</li><li>证据单元匹配来自既有标注，可合并实际返回的相邻源片段。年份列、单位和页外等价证据尚未全面语义核验。</li><li>这轮没有Agent、查询规划、答案生成或API调用；上下文增加是否降低LLM补查次数，需要下一轮端到端对照。未改应用默认参数。</li></ul>'
    page+='<p><a href="comparison.csv">指标CSV</a> · <a href="summary.json">汇总</a> · <a href="results.json">逐题结果</a> · <a href="paired_comparison.json">相对20/5变化与配对bootstrap</a> · <a href="config.json">配置和哈希</a> · <a href="verification.json">核验记录</a> · <a href="chunks.json">冻结片段</a></p></main></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.project_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    import torch
    from copy import deepcopy
    from sentence_transformers import SentenceTransformer
    from financial_rag.indexing.bm25_index import BM25Index
    from financial_rag.retrieval.dense import dense_search
    from financial_rag.retrieval.sparse import bm25_search
    from financial_rag.retrieval.fusion import reciprocal_rank_fusion
    from financial_rag.retrieval.reranker import Reranker
    frozen=root/'outputs/embedding_ablation/financebench50_300_0_20261013'
    chunks=json.loads((frozen/'chunks.json').read_text());cases=json.loads((frozen/'selected_cases.json').read_text())
    criteria=json.loads((frozen/'evidence_criteria.json').read_text())
    assert len(chunks)==18806 and len(cases)==50
    assert [r['id'] for r in criteria]==[c['financebench_id'] for c in cases]
    byid={c['chunk_id']:c for c in chunks};assert len(byid)==len(chunks)
    assert len({c['document_id'] for c in chunks})==41
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model_paths={'embedding':root/'models/bge-small-en-v1.5','reranker':root/'models/bge-reranker-v2-m3'}
    source_paths=[Path(__file__),root/'src/financial_rag/indexing/embeddings.py',root/'src/financial_rag/indexing/bm25_index.py',root/'src/financial_rag/retrieval/dense.py',root/'src/financial_rag/retrieval/sparse.py',root/'src/financial_rag/retrieval/fusion.py',root/'src/financial_rag/retrieval/reranker.py',root/'src/financial_rag/evaluation/evidence_content.py',root/'src/financial_rag/evaluation/chunk_ablation.py',root/'src/financial_rag/evaluation/financebench.py']
    config=dict(dataset=dict(questions=50,documents=41,chunks=18806,pages=6000,content_scored=49,source=str(frozen)),
        frozen_sha256={n:sha(frozen/n) for n in ('chunks.json','selected_cases.json','evidence_criteria.json','config.json')},
        model_sha256={name:{str(f.relative_to(path)):sha(f) for f in sorted(path.rglob('*')) if f.is_file() and f.suffix in ('.json','.txt','.model','.safetensors')} for name,path in model_paths.items()},
        source_sha256={str(f):sha(f) for f in source_paths},device=device,embedding_dtype='float32',embedding_batch_size=32,
        reranker='BAAI/bge-reranker-v2-m3',reranker_max_length=2048,reranker_batch_size=8,
        candidate_k=list(CANDIDATES),top_k=list(TOP_K),rrf_k=60,
        query='Original English question + target report filename; no instruction, rewrite, LLM planning, target-document filter or gold evidence in retrieval.',
        budget='BGE-small content tokens; each K-prefix additionally constrained to ranked whole-chunk prefix <=2000; no skipping or clipping.',
        timing='Dense/BM25 Top50 shared; per-pool RRF and actual reranker time; excludes model/document/query encoding; K-prefix derivations share rerank timing.',
        platform=platform.platform(),versions={x:importlib.metadata.version(x) for x in ('torch','transformers','sentence-transformers','numpy','pymupdf')},
        limitations='Previously used development corpus, heuristic evidence matching, no answer/call-count evaluation; frozen legacy PDF chunks, not new structured ingestion.')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'Output configuration changed; use a new directory'
    write_json(out/'config.json',config)
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    for f in source_paths:shutil.copyfile(f,snapshot/f.name)
    for n in ('chunks.json','selected_cases.json','evidence_criteria.json'):shutil.copyfile(frozen/n,out/n)
    model=SentenceTransformer(str(model_paths['embedding']),local_files_only=True,device=device,model_kwargs={'dtype':torch.float32})
    tok=deepcopy(model.tokenizer);tok.model_max_length=10**9
    texts=[c['text'] for c in chunks];queries=[evaluation_question(c) for c in cases]
    doclens=[len(ids) for ids in tok(texts,add_special_tokens=True)['input_ids']]
    qlens=[len(ids) for ids in tok(queries,add_special_tokens=True)['input_ids']]
    assert max(doclens)<=model.max_seq_length and max(qlens)<=model.max_seq_length
    assert [n-2 for n in doclens]==[c['content_tokens'] for c in chunks]
    write_json(out/'token_audit.json',dict(document_max=max(doclens),query_max=max(qlens),document_truncated=0,query_truncated=0,content_cap=300))
    def sync():
        if device=='mps':torch.mps.synchronize()
    model.encode(['warmup'],prompt='',normalize_embeddings=True,show_progress_bar=False)
    blockdir=out/'embedding_blocks';blockdir.mkdir(exist_ok=True);arrays=[];seconds=[]
    for offset in range(0,len(chunks),512):
        f=blockdir/f'{offset:06}.npy';tf=f.with_suffix('.json')
        if f.exists() and tf.exists():arr=np.load(f);elapsed=json.loads(tf.read_text())['seconds']
        else:
            sync();start=time.perf_counter()
            arr=model.encode(texts[offset:offset+512],prompt='',batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
            sync();elapsed=time.perf_counter()-start;np.save(f,arr);write_json(tf,dict(rows=len(arr),seconds=elapsed))
        assert len(arr)==len(texts[offset:offset+512]);arrays.append(arr);seconds.append(elapsed)
        print('ENCODE',min(offset+512,len(chunks)),'/',len(chunks),flush=True)
    vectors=np.concatenate(arrays);assert np.isfinite(vectors).all();assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
    np.save(out/'vectors.npy',vectors)
    sync();start=time.perf_counter()
    query_vectors=model.encode(queries,prompt='',batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    sync();query_encoding_seconds=time.perf_counter()-start;np.save(out/'query_vectors.npy',query_vectors)
    write_json(out/'encoding.json',dict(document_seconds=sum(seconds),query_batch_seconds=query_encoding_seconds,device=device,dtype=str(next(model.parameters()).dtype)))
    del model,arrays
    if device=='mps':torch.mps.empty_cache()
    print('BUILD_BM25',flush=True);bm25=BM25Index(chunks)
    print('LOAD_RERANKER',flush=True)
    reranker=Reranker('BAAI/bge-reranker-v2-m3',2048,model_path=str(model_paths['reranker']),local_files_only=True)
    # Match application dtype/device; record, rather than silently switching precision.
    config['actual_reranker_device']=str(reranker.model.device)
    config['actual_reranker_dtype']=str(next(reranker.model.parameters()).dtype)
    # Actual fields are stored separately to keep the resumable input manifest stable.
    write_json(out/'runtime.json',{k:config[k] for k in ('device','actual_reranker_device','actual_reranker_dtype')})
    print('RERANKER_RUNTIME',json.dumps(json.loads((out/'runtime.json').read_text())),flush=True)
    reranker.rerank(queries[0],[dict(chunks[0])],top_k=1)
    rows=[];casefolder=out/'cases';casefolder.mkdir(exist_ok=True)
    for i,(case,qv,crit) in enumerate(zip(cases,query_vectors,criteria)):
        f=casefolder/f'{i:02}_{case["financebench_id"]}.json'
        if f.exists():rows.append(json.loads(f.read_text()));print('RESUME_CASE',i+1,flush=True);continue
        start=time.perf_counter();dense=dense_search(qv,vectors,chunks,50);sparse=bm25_search(queries[i],bm25,50)
        retrieval_ms=(time.perf_counter()-start)*1000
        row=dict(id=case['financebench_id'],company=case['company'],question=case['question'],query=queries[i],manual_required=crit.get('manual_required',False),
            timing=dict(retrieval_ms=retrieval_ms),dense=[dict(chunk_id=r['chunk_id'],score=r['score']) for r in dense],
            bm25=[dict(chunk_id=r['chunk_id'],score=r['score']) for r in sparse],pools={})
        for n in CANDIDATES:
            start=time.perf_counter();rrf=reciprocal_rank_fusion(dense[:n],sparse[:n],k=60,top_k=n)
            fusion_ms=(time.perf_counter()-start)*1000
            union=list(dict.fromkeys([r['chunk_id'] for r in dense[:n]+sparse[:n]]))
            sync();start=time.perf_counter();ranked=reranker.rerank(queries[i],rrf,top_k=len(rrf));sync();rerank_ms=(time.perf_counter()-start)*1000
            pool=dict(timing=dict(fusion_ms=fusion_ms,rerank_ms=rerank_ms),rrf=[dict(chunk_id=r['chunk_id'],rrf_score=r['rrf_score'],dense_rank=r['dense_rank'],bm25_rank=r['bm25_rank']) for r in rrf],
                reranked=[dict(chunk_id=r['chunk_id'],rerank_score=r['rerank_score']) for r in ranked],truncated_chunk_ids=list(reranker.last_truncated_chunk_ids),
                stages=dict(union=evaluate_ids(union,byid,case,crit),rrf=evaluate_ids([r['chunk_id'] for r in rrf],byid,case,crit)),selections={})
            for k in TOP_K:
                ids=[r['chunk_id'] for r in ranked[:k]];pool['selections'][f'top{k}']=evaluate_ids(ids,byid,case,crit)
                lengths=[byid[cid]['content_tokens'] for cid in ids]
                selected,used=select_budget(range(len(ids)),lengths,2000)
                pool['selections'][f'top{k}_budget2000']=evaluate_ids([ids[j] for j in selected],byid,case,crit)
            # Diagnostic controls: same retrieval candidates before reranking.
            pool['unreranked']={f'top{k}':evaluate_ids([r['chunk_id'] for r in rrf[:k]],byid,case,crit) for k in TOP_K}
            row['pools'][str(n)]=pool
            print('CASE',i+1,'/',len(cases),case['company'],'candidate',n,'rerank_seconds',round(rerank_ms/1000,2),
                'coverage',[pool['selections'][f'top{k}']['content']['coverage'] for k in TOP_K],flush=True)
        write_json(f,row);rows.append(row)
        write_json(out/'progress.json',dict(completed=len(rows),total=len(cases),latest=case['financebench_id']))
    write_json(out/'results.json',rows)
    summaries=summarize(rows);write_json(out/'summary.json',summaries);write_json(out/'paired_comparison.json',paired_comparisons(rows))
    stages={str(n):{stage:aggregate([r['pools'][str(n)]['stages'][stage] for r in rows]) for stage in ('union','rrf')} for n in CANDIDATES}
    write_json(out/'stage_summary.json',stages)
    controls={str(n):{f'top{k}':aggregate([r['pools'][str(n)]['unreranked'][f'top{k}'] for r in rows]) for k in TOP_K} for n in CANDIDATES}
    write_json(out/'unreranked_controls.json',controls)
    cols=['name','candidate_k','top_k','page_hit','page_coverage','content_hit','content_coverage','all_units','annotated_support_chunk_fraction','content_tokens_mean','content_tokens_p95','retrieval_rerank_ms_median','retrieval_rerank_ms_p95']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols+['budget2000_content_coverage','budget2000_all_units']);w.writeheader()
        for s in summaries:w.writerow({**{k:s[k] for k in cols},'budget2000_content_coverage':s['budget2000']['content_coverage'],'budget2000_all_units':s['budget2000']['all_units']})
    for r in rows:
        for n in CANDIDATES:
            pools=r['pools'][str(n)]
            assert len(pools['rrf'])<=n and len(pools['reranked'])==len(pools['rrf'])
            assert not pools['truncated_chunk_ids']
            scores=[pools['selections'][f'top{k}']['content']['coverage'] for k in TOP_K]
            if scores[0] is not None:assert scores==sorted(scores)
            assert all(pools['selections'][f'top{k}_budget2000']['content_tokens']<=2000 for k in TOP_K)
    verify=dict(questions=len(rows),configurations=6,documents=41,chunks=len(chunks),automatic_content_cases=sum(not r['manual_required'] for r in rows),
        maxK_monotonicity_checked=True,rerank_truncation_count=0,embedding_truncation_count=0,budget_checks_passed=True,
        generation_calls=0,application_defaults_changed=False,source_mapping='All evaluation uses original frozen chunks by ID, retaining char_start/char_end for adjacent evidence.')
    write_json(out/'verification.json',verify);report(out,rows,summaries,byid,config)
    print('FINISHED',out,flush=True)
    for s in summaries:print(s['name'],json.dumps({k:s[k] for k in ('content_hit','content_coverage','all_units','content_tokens_mean')}),flush=True)


if __name__=='__main__':main()
