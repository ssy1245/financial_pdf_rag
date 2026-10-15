"""Staged real-LLM FinanceBench50 batch-query versus original-loop experiment.

9 configurations: max initial queries 1/2/3 x TOTAL initial chunks 5/7/10.
12 stratified screening cases; promote coverage and efficiency batch candidates,
plus all three single-query controls, to 50. Selection-only 12 and remaining38
are reported separately. No gold evidence/answers are supplied to either model.
"""
import argparse
import csv
import hashlib
import html
import json
import platform
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
from financial_rag.agent.loop import run_agent
from financial_rag.generation.generator import DeepSeekGenerator
from financial_rag.evaluation.financebench import write_json,evaluation_question
from financial_rag.evaluation.retrieval_k_ablation import evaluate_ids

GRID=[dict(name=f'q{q}_k{k}',queries=q,k=k) for q in (1,2,3) for k in (5,7,10)]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def screen_indices(cases):
    rng=random.Random(20261015);selected=[]
    for kind in sorted({c['question_type'] for c in cases}):
        ids=[i for i,c in enumerate(cases) if c['question_type']==kind]
        rng.shuffle(ids);selected+=ids[:4]
    return sorted(selected)


class TrackedGenerator(DeepSeekGenerator):
    def __init__(self):
        super().__init__('deepseek-flash')
        self.calls=[];self.http_attempts=0
        self.client._client.event_hooks.setdefault('request',[]).append(self.count_http)
        create=self.client.chat.completions.create
        def tracked(**kwargs):
            kwargs.update(temperature=0,max_tokens=8192)
            item=dict(sequence=len(self.calls)+1,kind='tool_turn' if 'tools' in kwargs else 'planning')
            self.calls.append(item);start=time.perf_counter()
            try:
                response=create(**kwargs)
                item.update(status='ok',usage=response.usage.model_dump() if response.usage else {},
                    returned_model=response.model,finish_reason=response.choices[0].finish_reason)
                return response
            except Exception as exc:
                item.update(status='error',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None))
                raise
            finally:item['seconds']=time.perf_counter()-start
        self.client.chat.completions.create=tracked
    def count_http(self,request):self.http_attempts+=1


class CaseRetriever:
    """Share immutable index and GPU models, serialize only local inference."""
    def __init__(self,base,lock,k):
        self.base=base;self.lock=lock;self.top_k=k;self.candidate_k=base.candidate_k
        self.chunks_by_id=base.chunks_by_id;self.wait_seconds=0.;self.active_seconds=0.
    def retrieve(self,query,*,top_k=None,exclude_chunk_ids=None):
        start=time.perf_counter();self.lock.acquire();self.wait_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        try:return self.base.retrieve(query,top_k=top_k or self.top_k,exclude_chunk_ids=exclude_chunk_ids)
        finally:
            self.active_seconds+=time.perf_counter()-start;self.lock.release()


def aggregate(records,case_ids=None):
    if case_ids is not None:records=[r for r in records if r['id'] in case_ids]
    auto=[r for r in records if r['final']['content']['coverage'] is not None]
    mean=lambda vals:float(np.mean(vals)) if vals else None
    return dict(n=len(records),content_n=len(auto),errors=sum(r['result']['status']=='error' for r in records),
        completed=sum(r['result']['status']=='completed' for r in records),
        initial_coverage=mean([r['initial']['content']['coverage'] for r in auto]),
        final_coverage=mean([r['final']['content']['coverage'] for r in auto]),
        initial_all=mean([r['initial']['content']['all_units'] for r in auto]),
        final_all=mean([r['final']['content']['all_units'] for r in auto]),
        final_hit=mean([r['final']['content']['hit'] for r in auto]),
        final_page_coverage=mean([r['final']['page']['coverage'] for r in records]),
        sdk_calls_mean=mean([r['sdk_calls'] for r in records]),http_attempts_mean=mean([r['http_attempts'] for r in records]),
        seconds_mean=mean([r['wall_seconds'] for r in records]),seconds_median=mean([]) if not records else float(np.median([r['wall_seconds'] for r in records])),
        seconds_without_local_queue_mean=mean([r['seconds_without_local_queue'] for r in records]),
        local_queue_seconds_mean=mean([r['local_queue_seconds'] for r in records]),
        local_active_seconds_mean=mean([r['local_active_seconds'] for r in records]),
        searches_mean=mean([r['result'].get('search_attempts',0) for r in records]),
        planned_queries_mean=mean([len(r['result'].get('query_plan',{}).get('queries',[])) for r in records]),
        initial_searches_mean=mean([r['result'].get('initial_searches',0) for r in records]),
        initial_chunks_mean=mean([r['initial']['returned_chunks'] for r in records]),
        final_chunks_mean=mean([r['final']['returned_chunks'] for r in records]),
        prompt_tokens_mean=mean([r['usage'].get('prompt_tokens',0) for r in records]),
        completion_tokens_mean=mean([r['usage'].get('completion_tokens',0) for r in records]),
        total_tokens=sum(r['usage'].get('total_tokens',0) for r in records),
        completed_with_incomplete_annotated_evidence=sum(r['result']['status']=='completed' and r['final']['content']['all_units']==0 for r in auto),
        planning_fallbacks=sum(bool(r['result'].get('query_plan',{}).get('warning')) for r in records))


def promote(screen):
    byname={r['name']:r for r in screen}
    batch=[r for r in screen if r['queries']>1 and not r['metrics']['errors']]
    if not batch:raise RuntimeError('No successful batch screening configurations')
    quality=max(batch,key=lambda r:(r['metrics']['final_coverage'],-r['metrics']['sdk_calls_mean'],-r['metrics']['seconds_without_local_queue_mean']))
    eligible=[r for r in batch if r['metrics']['final_coverage']>=byname[f'q1_k{r["k"]}']['metrics']['final_coverage']-1e-12]
    efficiency=min(eligible or batch,key=lambda r:(r['metrics']['sdk_calls_mean'],-r['metrics']['final_coverage'],r['metrics']['seconds_without_local_queue_mean']))
    names=list(dict.fromkeys(['q1_k5','q1_k7','q1_k10',quality['name'],efficiency['name']]))
    return dict(finalists=names,coverage_choice=quality['name'],efficiency_choice=efficiency['name'],
        rule='Promote highest screening final coverage (tie: fewer SDK calls, then less queue-adjusted time); also fewest-call batch with coverage >= same-K single-query control (ties: coverage, time). Always retain all q1 K controls. If no batch meets the constraint, efficiency choice is exploratory only.',
        efficiency_constraint_met=bool(eligible))


def build_report(out,config,screen,selection,full,records,cases,screen_ids):
    e=lambda x:html.escape(str(x));pct=lambda x:'—' if x is None else f'{x:.1%}'
    def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinanceBench50：批量初始查询与原Loop</title><style>body{font:16px/1.65 system-ui;color:#223047;background:#f4f6fa;margin:0}main{max-width:1350px;margin:auto;padding:32px}table{border-collapse:collapse;background:white;white-space:nowrap;width:100%}td,th{padding:10px;border:1px solid #dce1eb;text-align:left}th{background:#e9eef7}.scroll{overflow:auto}details{background:white;border:1px solid #dce1eb;padding:12px;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:14px}a{color:#175eb1}.note{padding:16px;background:#e6eefb;border-radius:8px}</style><main><h1>FinanceBench50：一次规划多个查询、批量检索</h1>'''
    page+='<p class="note">真实调用DeepSeek；候选20、固定300/0语料、相同原Loop后续补查。9组超参数先在12题筛选，按预先规则晋级，再运行全部50题。首批总证据K在查询之间均分，因此批量方案不会因每条查询各拿K块而获得额外初始上下文。批量指多条本地检索之间不调用LLM，不表示本机模型并行。</p>'
    headers=['配置','题数/内容分母','首次覆盖','最终覆盖','全部找全','平均SDK调用','平均首批查询','平均总检索','平均秒','去本地排队秒','平均输入tokens','错误']
    def row(name,m):return [name,f'{m["n"]}/{m["content_n"]}',pct(m['initial_coverage']),pct(m['final_coverage']),pct(m['final_all']),f"{m['sdk_calls_mean']:.2f}",f"{m['initial_searches_mean']:.2f}",f"{m['searches_mean']:.2f}",f"{m['seconds_mean']:.1f}",f"{m['seconds_without_local_queue_mean']:.1f}",f"{m['prompt_tokens_mean']:.0f}",m['errors']]
    page+='<h2>12题筛选：9组参数</h2>'+table(headers,[row(r['name'],r['metrics']) for r in screen])
    page+='<p>q表示允许规划的初始查询上限（简单题可少于上限），k表示首批证据总数，并作为后续检索默认TopK；模型仍可按工具协议改写后续top_k。q1使用原单查询规划提示词，q2/q3使用相同批量提示词，仅改变上限。</p><p>晋级选择：</p><pre>'+e(json.dumps(selection,ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>全部50题结果</h2>'+table(headers,[row(r['name'],r['all50']) for r in full])
    page+='<h2>未参与本轮参数筛选的38题</h2><p>这38题此前已用于其他开发实验，不能称为从未看过的测试集；只是未用于本轮超参数选择。主要比较此表，完整50题包含筛选数据，存在选择偏差。</p>'+table(headers,[row(r['name'],r['remaining38']) for r in full])
    page+='<h2>停止与证据完整性</h2><p>模型正常停止但未覆盖全部标注证据，并不直接等于回答错误；可能存在等价证据或合理拒答。此处用于定位停止判断风险，不报告语义答案准确率。</p>'+table(['配置','正常完成','错误','完成但标注未找全','规划回退','最终平均块数','实际HTTP尝试均值'],[[r['name'],r['all50']['completed'],r['all50']['errors'],r['all50']['completed_with_incomplete_annotated_evidence'],r['all50']['planning_fallbacks'],f"{r['all50']['final_chunks_mean']:.1f}",f"{r['all50']['http_attempts_mean']:.2f}"] for r in full])
    page+='<h2>逐题：查询、答案与标注证据覆盖</h2>'
    for case in cases:
        cid=case['financebench_id'];page+='<details><summary>'+e(case['company']+' · '+cid+(' · 筛选题' if cid in screen_ids else ' · 其余38题'))+'</summary><p>'+e(case['question'])+'</p><p>参考答案（仅复核）：'+e(case['answer'])+'</p>'
        for cfg in full:
            r=records[cfg['name']][cid];m=r['result']
            page+='<details><summary>'+e(f'{cfg["name"]} · {r["sdk_calls"]}次LLM · {r["wall_seconds"]:.1f}秒 · 证据覆盖 {pct(r["final"]["content"]["coverage"])}')+'</summary><p>初始规划：</p><pre>'+e(json.dumps(m.get('query_plan'),ensure_ascii=False,indent=2))+'</pre><p>最终回答：</p><pre>'+e(m.get('answer',''))+'</pre><p>补查过程：</p><pre>'+e(json.dumps(m.get('search_history'),ensure_ascii=False,indent=2))+'</pre><p>最终证据匹配：</p><pre>'+e(json.dumps(r['final']['content'],ensure_ascii=False,indent=2))+'</pre>'
            page+='<a href="cases/'+cfg['name']+'/'+cid+'.json">完整来源、每轮候选、调用usage与耗时</a></details>'
        page+='</details>'
    page+='<h2>实验口径</h2><ul><li>同一50题、41份全报告、18,806块，复用已核验FP32无指令文档向量；新查询实时编码。无文档筛选；不向规划/回答模型传标准答案、证据页或内容规则。</li><li>每配置独立请求记忆和模型会话；最多7个分析/回答步骤、8次检索、60,000证据字符、120,000消息字符。批量初始查询也消耗8次检索预算，初始片段按查询数做商余分配；后续工具调用和停止规则与原Loop一致。</li><li>全部配置使用deepseek-flash、temperature=0、max_tokens=8192、原SDK重试/超时。q1_k5是原Loop在上述统一生成设置和冻结语料下的复跑，不直接比较历史旧语料试跑。</li><li>SDK调用含规划、补查决策和最终回答；HTTP尝试另计重试。tokens来自实际API usage；全部保存调用返回模型标识、finish_reason和错误类型，不保存密钥。</li><li>最多4个独立问题并发，GPU检索/重排串行；记录端到端墙钟时间、本地锁等待及去锁等待时间。仍受API波动、服务端缓存和并发影响，不是严格单请求延迟基准。</li><li>每题首次与最终证据均按原始chunk来源评分，可拼接真实返回的相邻片段。49题自动内容评分，另1题人工；所有50题都统计调用与耗时。未做LLM判分或最终答案语义正确率评测。</li><li>9组筛选和晋级规则均保存；单次随机性及较小筛选集会影响晋级，不作普适最优或显著性承诺。网页仍默认单查询Loop。</li></ul>'
    page+='<p><a href="screen_summary.json">筛选结果</a> · <a href="selection.json">晋级记录</a> · <a href="summary.json">完整汇总</a> · <a href="comparison.csv">CSV</a> · <a href="config.json">设置与哈希</a> · <a href="paired_comparison.json">配对差异</a> · <a href="verification.json">核验记录</a></p></main></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    root=args.project_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    frozen=root/'outputs/retrieval_k_ablation/financebench50_20261015'
    chunks=json.loads((frozen/'chunks.json').read_text());cases=json.loads((frozen/'selected_cases.json').read_text())
    criteria=json.loads((frozen/'evidence_criteria.json').read_text());byid={c['chunk_id']:c for c in chunks}
    assert len(cases)==50 and len(chunks)==18806 and len(byid)==len(chunks)
    assert [c['id'] for c in criteria]==[c['financebench_id'] for c in cases]
    si=screen_indices(cases);screenids={cases[i]['financebench_id'] for i in si};remainingids={c['financebench_id'] for c in cases}-screenids
    assert len(si)==12 and len(remainingids)==38
    source_files=[root/'src/financial_rag'/n for n in ['agent/loop.py','agent/planner.py','generation/generator.py','generation/prompt.py','retrieval/retriever.py','retrieval/tool.py','retrieval/reranker.py','memory/evidence.py','evaluation/batch_query_ablation.py','evaluation/evidence_content.py','evaluation/retrieval_k_ablation.py']]
    config=dict(grid=GRID,screen_indices=si,screen_ids=sorted(screenids),remaining38_ids=sorted(remainingids),seed=20261015,
        dataset=dict(questions=50,documents=41,chunks=len(chunks),content_questions=49),frozen_source=str(frozen),
        input_sha256={n:sha(frozen/n) for n in ('chunks.json','selected_cases.json','evidence_criteria.json','vectors.npy','config.json')},
        source_sha256={str(p.relative_to(root)):sha(p) for p in source_files},model='deepseek-flash',temperature=0,max_tokens=8192,
        candidate_k=20,rrf_k=60,max_steps=7,max_searches=8,max_evidence_chars=60000,max_message_chars=120000,workers=args.workers,
        batch_policy='Up to q independent planned queries; total initial K split quotient/remainder; initial queries share ID exclusions; no intermediate LLM calls; local searches serialized. Subsequent searches use original loop.',
        selection_policy='Coverage and efficiency batch champions from stratified12; plus all q1 K controls; selected configurations expanded to50. Report remaining38 separately.',
        platform=platform.platform())
    cp=out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Configuration changed; choose new output'
    write_json(cp,config)
    for n in ('chunks.json','selected_cases.json','evidence_criteria.json'):shutil.copyfile(frozen/n,out/n)
    for p in source_files:
        dst=out/'source'/p.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    # Verify cached vectors are from the application's no-instruction FP32 policy.
    previous=json.loads((frozen/'config.json').read_text());assert previous['embedding_dtype']=='float32'
    vectors=np.load(frozen/'vectors.npy');assert vectors.shape==(len(chunks),384) and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    from financial_rag.retrieval.retriever import Retriever
    embedding=EmbeddingModel(model_path=str(root/'models/bge-small-en-v1.5'),local_files_only=True)
    reranker=Reranker('BAAI/bge-reranker-v2-m3',2048,model_path=str(root/'models/bge-reranker-v2-m3'),local_files_only=True)
    class Store:
        def load(self,name):return chunks,vectors
    base=Retriever(Store(),embedding,'BAAI/bge-small-en-v1.5',reranker,candidate_k=20,top_k=5)
    embedding.encode_query('warmup');reranker.rerank('warmup',[chunks[0]],top_k=1)
    lock=threading.Lock();print('READY',len(cases),'questions; screen',si,'workers',args.workers,flush=True)
    records={cfg['name']:{} for cfg in GRID}
    def path(cfg,i):return out/'cases'/cfg['name']/(cases[i]['financebench_id']+'.json')
    def run_one(cfg,i,phase):
        destination=path(cfg,i)
        if destination.exists():return json.loads(destination.read_text())
        case=cases[i];retriever=CaseRetriever(base,lock,cfg['k']);gen=TrackedGenerator();started=time.perf_counter()
        try:
            result=run_agent(evaluation_question(case),retriever,gen,use_query_planning=True,
                initial_query_count=cfg['queries'],max_steps=7,max_searches=8)
        except Exception as exc:
            result=dict(status='error',error_type=type(exc).__name__,evidence={},initial_chunk_ids=[],search_attempts=0)
        wall=time.perf_counter()-started
        finalids=list(result.get('evidence',{}));initialids=result.get('initial_chunk_ids',[])
        usage={k:sum(c.get('usage',{}).get(k,0) or 0 for c in gen.calls) for k in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens')}
        record=dict(id=case['financebench_id'],company=case['company'],configuration=cfg,phase=phase,result=result,
            initial=evaluate_ids(initialids,byid,case,criteria[i]),final=evaluate_ids(finalids,byid,case,criteria[i]),
            sdk_calls=len(gen.calls),http_attempts=gen.http_attempts,api_usage=gen.calls,usage=usage,
            wall_seconds=wall,local_queue_seconds=retriever.wait_seconds,local_active_seconds=retriever.active_seconds,
            seconds_without_local_queue=wall-retriever.wait_seconds)
        gen.client.close();write_json(destination,record)
        print('DONE',phase,cfg['name'],i+1,'calls',record['sdk_calls'],'searches',result.get('search_attempts'),
            'coverage',record['final']['content']['coverage'],'seconds',round(wall,1),'status',result['status'],flush=True)
        return record
    def execute(jobs,phase):
        # Bounded submission; halt on repeated API/model failures, preserving completed work.
        rng=random.Random(20261015+(0 if phase=='screen' else 1));rng.shuffle(jobs)
        pendingjobs=[]
        for cfg,i in jobs:
            p=path(cfg,i)
            if p.exists():records[cfg['name']][cases[i]['financebench_id']]=json.loads(p.read_text())
            else:pendingjobs.append((cfg,i))
        iterator=iter(pendingjobs);completed=0;consecutive_errors=0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active={}
            def submit_next():
                try:cfg,i=next(iterator)
                except StopIteration:return False
                active[pool.submit(run_one,cfg,i,phase)]=(cfg,i);return True
            for _ in range(args.workers):submit_next()
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    cfg,i=active.pop(future);r=future.result();records[cfg['name']][r['id']]=r;completed+=1
                    consecutive_errors=consecutive_errors+1 if r['result']['status']=='error' else 0
                    write_json(out/'progress.json',dict(phase=phase,completed_new=completed,total_new=len(pendingjobs),latest_config=cfg['name'],latest_id=r['id']))
                    if consecutive_errors>=3:raise RuntimeError('Repeated model/API failures; completed cases saved')
                    submit_next()
    # A single real smoke run before dispatching the full search.
    smoke=run_one(GRID[0],si[0],'screen');records[GRID[0]['name']][smoke['id']]=smoke
    if smoke['result']['status']=='error':raise RuntimeError('Real API smoke failed; no large run dispatched')
    execute([(cfg,i) for cfg in GRID for i in si],'screen')
    screen=[dict(cfg,metrics=aggregate(list(records[cfg['name']].values()),screenids)) for cfg in GRID]
    assert all(r['metrics']['n']==12 for r in screen)
    write_json(out/'screen_summary.json',screen);selection=promote(screen);write_json(out/'selection.json',selection)
    print('PROMOTED',json.dumps(selection),flush=True)
    finalists=[c for c in GRID if c['name'] in selection['finalists']]
    execute([(cfg,i) for cfg in finalists for i in range(len(cases))],'full')
    full=[dict(cfg,all50=aggregate(list(records[cfg['name']].values())),remaining38=aggregate(list(records[cfg['name']].values()),remainingids),screen12=aggregate(list(records[cfg['name']].values()),screenids)) for cfg in finalists]
    write_json(out/'summary.json',full)
    rng=np.random.default_rng(20261015);pairs={}
    for cfg in finalists:
        if cfg['queries']==1:continue
        for control in ('q1_k5',f'q1_k{cfg["k"]}'):
            key=cfg['name']+'_vs_'+control
            groups={}
            for cohort,ids in [('all50',{c['financebench_id'] for c in cases}),('remaining38',remainingids)]:
                diffs=[]
                for cid in sorted(ids):
                    a=records[control][cid];b=records[cfg['name']][cid]
                    diffs.append(dict(id=cid,coverage_delta=None if a['final']['content']['coverage'] is None else b['final']['content']['coverage']-a['final']['content']['coverage'],calls_delta=b['sdk_calls']-a['sdk_calls'],seconds_delta=b['seconds_without_local_queue']-a['seconds_without_local_queue']))
                d=np.array([r['coverage_delta'] for r in diffs if r['coverage_delta'] is not None]);boot=np.mean(rng.choice(d,size=(10000,len(d)),replace=True),axis=1)
                groups[cohort]=dict(coverage_delta_mean=float(d.mean()),coverage_bootstrap95=[float(x) for x in np.percentile(boot,[2.5,97.5])],calls_delta_mean=float(np.mean([r['calls_delta'] for r in diffs])),queue_adjusted_seconds_delta_mean=float(np.mean([r['seconds_delta'] for r in diffs])),cases=diffs)
            pairs[key]=groups
    write_json(out/'paired_comparison.json',pairs)
    fields=['n','content_n','initial_coverage','final_coverage','final_all','sdk_calls_mean','http_attempts_mean','seconds_mean','seconds_without_local_queue_mean','searches_mean','prompt_tokens_mean','completion_tokens_mean','errors']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['configuration','cohort']+fields);w.writeheader()
        for r in full:
            for cohort in ('all50','remaining38','screen12'):w.writerow(dict(configuration=r['name'],cohort=cohort,**{k:r[cohort][k] for k in fields}))
    allrecords=[r for values in records.values() for r in values.values()]
    assert all(len(r['result'].get('initial_chunk_ids',[]))<=r['configuration']['k'] for r in allrecords)
    assert all(r['result'].get('search_attempts',0)<=8 for r in allrecords)
    verify=dict(screen_configurations=9,screen_cases=12,final_configurations=len(finalists),final_cases=50,remaining_cases=38,
        unique_case_configuration_runs=len(allrecords),sdk_calls=sum(r['sdk_calls'] for r in allrecords),http_attempts=sum(r['http_attempts'] for r in allrecords),
        api_total_tokens=sum(r['usage']['total_tokens'] for r in allrecords),errors=sum(r['result']['status']=='error' for r in allrecords),
        total_initial_evidence_caps_checked=True,total_search_caps_checked=True,application_default_initial_queries=1,
        generation_model='deepseek-flash',answer_accuracy_evaluated=False)
    write_json(out/'verification.json',verify);build_report(out,config,screen,selection,full,records,cases,screenids)
    print('FINISHED',out,json.dumps(verify),flush=True)
    for r in full:print('SUMMARY',r['name'],json.dumps(r['all50']),flush=True)


if __name__=='__main__':main()
