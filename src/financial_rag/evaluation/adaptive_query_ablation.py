"""FinanceBench50: uncapped initial planning under a shared task budget.

Five predeclared configurations all run on all 50 development questions.
Only the adaptive prompt's count instruction differs from capped batch planning.
Initial total chunk budgets 8/12/16; all subsequent searches default to 7.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import html
import json
from pathlib import Path
import platform
import random
import shutil
import threading
import time

import numpy as np
from financial_rag.agent.loop import run_agent
from financial_rag.evaluation.batch_query_ablation import TrackedGenerator, CaseRetriever, aggregate, sha
from financial_rag.evaluation.financebench import write_json, evaluation_question
from financial_rag.evaluation.retrieval_k_ablation import evaluate_ids

GRID = [dict(name='single_b8', queries=1, initial_k=8),
        dict(name='capped2_b8', queries=2, initial_k=8),
        *[dict(name=f'adaptive_b{k}', queries=None, initial_k=k) for k in (8,12,16)]]


def summarize(records):
    m=aggregate(records)
    m.update(planned_count_distribution=dict(Counter(str(len(r['result'].get('query_plan',{}).get('queries',[]))) for r in records)),
             executed_count_distribution=dict(Counter(str(r['result'].get('initial_searches',0)) for r in records)),
             unexecuted_plan_cases=sum(bool(r['result'].get('initial_queries_not_executed')) for r in records),
             unexecuted_queries=sum(len(r['result'].get('initial_queries_not_executed',[])) for r in records),
             stop_reasons=dict(Counter(r['result'].get('stop_reason') or r['result']['status'] for r in records)),
             length_limited_calls=sum(call.get('finish_reason')=='length' for r in records for call in r['api_usage']))
    return m


def paired(a,b,seed=20261015):
    rng=np.random.default_rng(seed);result={}
    for group in ('all50','both_completed'):
        ids=sorted(set(a)&set(b))
        if group=='both_completed':ids=[i for i in ids if a[i]['result']['status']==b[i]['result']['status']=='completed']
        diffs=[dict(id=i,coverage=None if a[i]['final']['content']['coverage'] is None else b[i]['final']['content']['coverage']-a[i]['final']['content']['coverage'],
                    calls=b[i]['sdk_calls']-a[i]['sdk_calls'],seconds=b[i]['wall_seconds']-a[i]['wall_seconds'],
                    queue_adjusted_seconds=b[i]['seconds_without_local_queue']-a[i]['seconds_without_local_queue']) for i in ids]
        metrics={}
        for metric in ('coverage','calls','seconds','queue_adjusted_seconds'):
            values=np.array([r[metric] for r in diffs if r[metric] is not None])
            if len(values):
                boot=rng.choice(values,size=(10000,len(values)),replace=True).mean(axis=1)
                metrics[metric]=dict(n=len(values),mean=float(values.mean()),bootstrap95=[float(x) for x in np.percentile(boot,[2.5,97.5])])
        result[group]=dict(n=len(ids),metrics=metrics,cases=diffs)
    return result


def render(out,summary,records,cases,pairs,verification):
    esc=lambda x:html.escape(str(x));pct=lambda x:'—' if x is None else f'{x:.1%}'
    def table(headers,rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FinanceBench50 · 自由规划初始查询</title><style>body{font:16px/1.7 system-ui;color:#243047;background:#f4f6fa;margin:0}main{max-width:1320px;padding:32px;margin:auto}table{border-collapse:collapse;background:white;width:100%;white-space:nowrap}td,th{border:1px solid #dce1eb;padding:10px;text-align:left}th{background:#e9eef7}.scroll{overflow:auto}details{background:white;padding:14px;margin:12px 0;border:1px solid #dce1eb}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:14px}a{color:#175eb1}.note{padding:16px;background:#e6eefb}</style><main><h1>自由规划初始查询：FinanceBench50</h1>'''
    page+='<p class="note">规划提示词不设查询数量上限，模型按问题自行决定。执行仍遵守整个任务最多8次检索、7个模型步骤；因预算未执行的查询明确记录。五组均重新运行完整50题，没有本轮筛选晋级。初始证据总预算8/12/16与后续每次默认Top7分开设置。</p>'
    page+='<h2>完整50题结果（49题自动内容评分）</h2>'+table(['配置','首批总块数上限','首次覆盖','最终覆盖','全部找全','LLM调用','首批查询均值','总检索均值','墙钟秒','扣本地排队秒','输入tokens均值','错误'],[
        [s['name'],s['initial_k'],pct(s['metrics']['initial_coverage']),pct(s['metrics']['final_coverage']),pct(s['metrics']['final_all']),
         f"{s['metrics']['sdk_calls_mean']:.2f}",f"{s['metrics']['initial_searches_mean']:.2f}",f"{s['metrics']['searches_mean']:.2f}",
         f"{s['metrics']['seconds_mean']:.1f}",f"{s['metrics']['seconds_without_local_queue_mean']:.1f}",f"{s['metrics']['prompt_tokens_mean']:.0f}",s['metrics']['errors']] for s in summary])
    page+='<p>single：原单查询提示词；capped2：原最多两条提示词；adaptive：只去掉同一批量提示词中的数量上限。b为首批总证据预算，在实际执行查询间商余分配、共享chunk ID去重。所有组后续默认Top7，模型仍可指定1～20。先比较adaptive_b8与capped2_b8识别取消上限的影响，再比较adaptive的8/12/16识别证据预算的影响。</p>'
    page+='<h2>模型实际规划与执行</h2>'+table(['配置','规划查询数分布（数量:题数）','执行查询数分布','有计划未执行的题数','未执行查询总数','正常完成','长度截断调用','规划回退'],[
        [s['name'],json.dumps(s['metrics']['planned_count_distribution'],sort_keys=True),json.dumps(s['metrics']['executed_count_distribution'],sort_keys=True),s['metrics']['unexecuted_plan_cases'],s['metrics']['unexecuted_queries'],s['metrics']['completed'],s['metrics']['length_limited_calls'],s['metrics']['planning_fallbacks']] for s in summary])
    page+='<h2>配对差异（前者减后者）</h2><p>区间是逐题配对bootstrap 95%区间，未做多重比较校正；全部50题已用于开发，不能当独立测试集。双方完成子集仅用于失败敏感性分析，不能替代全样本。</p>'
    pairrows=[]
    for name,groups in pairs.items():
        for cohort,g in groups.items():
            m=g['metrics'];v=m['coverage']
            pairrows.append([name,cohort,g['n'],f"{v['mean']*100:+.1f} pp",f"[{v['bootstrap95'][0]*100:+.1f}, {v['bootstrap95'][1]*100:+.1f}]",f"{m['calls']['mean']:+.2f}",f"{m['queue_adjusted_seconds']['mean']:+.2f}"])
    page+=table(['比较','样本','题数','覆盖变化','覆盖95%区间','调用变化','扣排队秒变化'],pairrows)
    page+='<h2>口径与限制</h2><ul><li>相同公开FinanceBench50题、41份完整报告、6000页、18806个300-token/0-overlap片段；复用已核验FP32文档向量。Dense/BM25各取20，RRF k60保留20，BGE重排。标准答案和评分标签只用于本地评分，不传给模型。</li><li>全部五组独立重跑，随机交错顺序、最多4个独立问题并发；本地模型使用锁串行。扣本地排队时间仍受服务端排队/缓存、网络和本地并发影响，不是严格单请求速度。</li><li>DeepSeek Flash：请求temperature=0、max_tokens=8192，沿用上一轮生成设置。thinking模式不保证temperature生效，内部推理可能占满输出预算；截断与失败保留，不排除。每组单次运行，模型随机性未通过重复种子衡量。</li><li>内容覆盖是规则对齐标注证据，非答案语义准确率；49题自动评分，1题人工，调用和耗时统计全部50题。未证明适用于开放式公司分析。</li><li>不限制规划数量不等于无限执行：所有组共享整个任务8次检索、7步、60000证据字符与120000消息字符。初始预算至少8，避免另一个低于总检索预算的隐含查询数限制。查询越多，后续补查剩余次数越少。</li><li>默认应用仍保持单查询；自由规划仅支持独立问题。旧实验仅作历史参考，不与本轮变化的首批/后续预算混为同配置。</li></ul>'
    page+='<h2>逐题查询、回答与证据</h2>'
    for case in cases:
        cid=case['financebench_id'];page+='<details><summary>'+esc(case['company']+' · '+cid)+'</summary><p>'+esc(case['question'])+'</p><p>参考答案（仅本地核验）：'+esc(case['answer'])+'</p>'
        for s in summary:
            r=records[s['name']][cid];result=r['result']
            page+='<details><summary>'+esc(f"{s['name']} · 覆盖{pct(r['final']['content']['coverage'])} · {r['sdk_calls']}次LLM · {r['wall_seconds']:.1f}秒")+'</summary>'
            for label,value in [('初始规划',result.get('query_plan')),('未执行的规划查询',result.get('initial_queries_not_executed')),('补查过程',result.get('search_history')),('回答',result.get('answer')),('最终证据评分',r['final']['content'])]:
                page+='<p>'+label+'</p><pre>'+esc(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
            page+='<a href="cases/'+s['name']+'/'+cid+'.json">完整逐题记录</a></details>'
        page+='</details>'
    page+='<h2>核验</h2><pre>'+esc(json.dumps(verification,ensure_ascii=False,indent=2))+'</pre><p>'+ ' · '.join('<a href="'+f+'">'+f+'</a>' for f in ('summary.json','comparison.csv','paired_comparison.json','config.json','verification.json','public_data_verification.json'))+'</p></main></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('workers must be 1..4')
    root=args.project_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    frozen=root/'outputs/retrieval_k_ablation/financebench50_20261015'
    old=root/'outputs/batch_query_ablation/financebench50_20261015_8192'
    chunks=json.loads((frozen/'chunks.json').read_text());cases=json.loads((frozen/'selected_cases.json').read_text())
    criteria=json.loads((frozen/'evidence_criteria.json').read_text());byid={c['chunk_id']:c for c in chunks}
    assert len(cases)==50 and len(chunks)==len(byid)==18806
    assert [c['id'] for c in criteria]==[c['financebench_id'] for c in cases]
    provenance=json.loads((old/'public_data_verification.json').read_text())
    assert provenance['questions_exact_public_source_matches']==50 and provenance['chunk_source_spans_exactly_verified']==18806
    oldconfig=json.loads((old/'config.json').read_text())
    assert all(sha(frozen/n)==oldconfig['input_sha256'][n] for n in ('chunks.json','selected_cases.json','evidence_criteria.json','vectors.npy','config.json'))
    source_files=[root/'src/financial_rag'/n for n in ('agent/loop.py','agent/planner.py','generation/generator.py','generation/prompt.py','retrieval/retriever.py','retrieval/tool.py','retrieval/reranker.py','memory/evidence.py','evaluation/adaptive_query_ablation.py','evaluation/batch_query_ablation.py','evaluation/evidence_content.py','evaluation/retrieval_k_ablation.py')]
    config=dict(grid=GRID,seed=20261015,questions=50,documents=41,chunks=18806,content_questions=49,
                input_sha256={n:sha(frozen/n) for n in ('chunks.json','selected_cases.json','evidence_criteria.json','vectors.npy','config.json')},
                source_sha256={str(p.relative_to(root)):sha(p) for p in source_files},public_verification_sha256=sha(old/'public_data_verification.json'),
                candidate_k=20,rrf_k=60,followup_top_k=7,max_searches=8,max_steps=7,max_evidence_chars=60000,max_message_chars=120000,
                model='deepseek-flash',temperature=0,max_tokens=8192,workers=args.workers,platform=platform.platform(),
                execution_policy='Adaptive planner has no query-count cap in prompt or parser; execution limited by shared task search budget 8; total initial chunks split across executed queries; all unexecuted planned queries recorded.',
                design='All five configurations freshly run on all 50 development cases. No screening/promotion. Paired bootstrap descriptive, no multiplicity correction; no semantic answer accuracy evaluation.')
    cp=out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Config/source changed: use new output directory'
    write_json(cp,config)
    for n in ('chunks.json','selected_cases.json','evidence_criteria.json'):shutil.copyfile(frozen/n,out/n)
    shutil.copyfile(old/'public_data_verification.json',out/'public_data_verification.json')
    for p in source_files:
        dst=out/'source'/p.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    vectors=np.load(frozen/'vectors.npy');assert vectors.shape==(18806,384) and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    from financial_rag.retrieval.retriever import Retriever
    embedding=EmbeddingModel(model_path=str(root/'models/bge-small-en-v1.5'),local_files_only=True)
    reranker=Reranker('BAAI/bge-reranker-v2-m3',2048,model_path=str(root/'models/bge-reranker-v2-m3'),local_files_only=True)
    class Store:
        def load(self,name):return chunks,vectors
    base=Retriever(Store(),embedding,'BAAI/bge-small-en-v1.5',reranker,candidate_k=20,top_k=7)
    embedding.encode_query('warmup');reranker.rerank('warmup',[chunks[0]],top_k=1)
    lock=threading.Lock();records={cfg['name']:{} for cfg in GRID}
    def path(cfg,i):return out/'cases'/cfg['name']/(cases[i]['financebench_id']+'.json')
    def run_one(cfg,i):
        destination=path(cfg,i)
        if destination.exists():return json.loads(destination.read_text())
        case=cases[i];retriever=CaseRetriever(base,lock,7);gen=TrackedGenerator();start=time.perf_counter()
        try:
            result=run_agent(evaluation_question(case),retriever,gen,use_query_planning=True,
                             initial_query_count=cfg['queries'],initial_evidence_k=cfg['initial_k'],max_steps=7,max_searches=8)
        except Exception as exc:result=dict(status='error',error_type=type(exc).__name__,evidence={},initial_chunk_ids=[],search_attempts=0)
        finally:gen.client.close()
        wall=time.perf_counter()-start
        usage={k:sum(c.get('usage',{}).get(k,0) or 0 for c in gen.calls) for k in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens')}
        record=dict(id=case['financebench_id'],company=case['company'],configuration=cfg,result=result,
                    initial=evaluate_ids(result.get('initial_chunk_ids',[]),byid,case,criteria[i]),
                    final=evaluate_ids(list(result.get('evidence',{})),byid,case,criteria[i]),
                    sdk_calls=len(gen.calls),http_attempts=gen.http_attempts,api_usage=gen.calls,usage=usage,
                    wall_seconds=wall,local_queue_seconds=retriever.wait_seconds,local_active_seconds=retriever.active_seconds,
                    seconds_without_local_queue=wall-retriever.wait_seconds)
        write_json(destination,record)
        print('DONE',cfg['name'],i+1,'plan',len(result.get('query_plan',{}).get('queries',[])),'calls',record['sdk_calls'],
              'coverage',record['final']['content']['coverage'],'seconds',round(wall,1),'status',result['status'],flush=True)
        return record
    print('READY 5 x 50; adaptive planning uncapped; task budget 8; workers',args.workers,flush=True)
    # Smoke exercises the new path and is reused as one formal observation.
    smoke=run_one(GRID[2],0);records[GRID[2]['name']][smoke['id']]=smoke
    if smoke['result']['status']=='error':raise RuntimeError('Adaptive smoke failed; no large run dispatched')
    jobs=[(cfg,i) for cfg in GRID for i in range(50)];random.Random(20261015).shuffle(jobs)
    remaining=[]
    for cfg,i in jobs:
        if path(cfg,i).exists():records[cfg['name']][cases[i]['financebench_id']]=json.loads(path(cfg,i).read_text())
        else:remaining.append((cfg,i))
    iterator=iter(remaining);completed=0;consecutive_errors=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={}
        def submit_next():
            try:cfg,i=next(iterator)
            except StopIteration:return
            active[pool.submit(run_one,cfg,i)]=(cfg,i)
        for _ in range(args.workers):submit_next()
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                cfg,i=active.pop(future);r=future.result();records[cfg['name']][r['id']]=r;completed+=1
                consecutive_errors=consecutive_errors+1 if r['result']['status']=='error' else 0
                write_json(out/'progress.json',dict(completed=sum(len(v) for v in records.values()),total=250,latest=cfg['name']))
                if consecutive_errors>=3:raise RuntimeError('Repeated model failures; completed records preserved')
                submit_next()
    assert all(len(r)==50 for r in records.values())
    summary=[dict(cfg,metrics=summarize(list(records[cfg['name']].values()))) for cfg in GRID]
    write_json(out/'summary.json',summary)
    pairs={}
    for treatment,control in [('capped2_b8','single_b8'),('adaptive_b8','single_b8'),('adaptive_b8','capped2_b8'),('adaptive_b12','adaptive_b8'),('adaptive_b16','adaptive_b8'),('adaptive_b12','capped2_b8'),('adaptive_b16','capped2_b8')]:
        pairs[treatment+'_vs_'+control]=paired(records[control],records[treatment])
    write_json(out/'paired_comparison.json',pairs)
    fields=['n','content_n','initial_coverage','final_coverage','final_all','sdk_calls_mean','http_attempts_mean','seconds_mean','seconds_without_local_queue_mean','initial_searches_mean','searches_mean','prompt_tokens_mean','completion_tokens_mean','errors','planning_fallbacks','unexecuted_plan_cases']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['configuration','initial_k']+fields);w.writeheader()
        for s in summary:w.writerow(dict(configuration=s['name'],initial_k=s['initial_k'],**{k:s['metrics'][k] for k in fields}))
    allrecords=[r for group in records.values() for r in group.values()]
    assert all(len(r['result'].get('initial_chunk_ids',[]))<=r['configuration']['initial_k'] for r in allrecords)
    assert all(r['result'].get('search_attempts',0)<=8 for r in allrecords)
    assert all(sha(root/p)==digest for p,digest in config['source_sha256'].items())
    verification=dict(configurations=5,cases_per_configuration=50,unique_runs=len(allrecords),
                      sdk_calls=sum(r['sdk_calls'] for r in allrecords),http_attempts=sum(r['http_attempts'] for r in allrecords),
                      total_api_tokens=sum(r['usage']['total_tokens'] for r in allrecords),errors=sum(r['result']['status']=='error' for r in allrecords),
                      length_limited_calls=sum(c.get('finish_reason')=='length' for r in allrecords for c in r['api_usage']),
                      public_inputs_unchanged=True,source_unchanged=True,initial_caps_checked=True,search_caps_checked=True,
                      application_default_queries=1,semantic_answer_accuracy_evaluated=False)
    write_json(out/'verification.json',verification);render(out,summary,records,cases,pairs,verification)
    print('FINISHED',json.dumps(verification),flush=True)
    for s in summary:print('SUMMARY',s['name'],json.dumps(s['metrics']),flush=True)


if __name__=='__main__':main()
