"""Small matched experiment: seven FinanceBench cases + three custom revenue analyses.

Exactly three configurations x the same ten cases; no screening or expansion.
Custom question rubrics are frozen and corpus-validated before any model request.
"""
import argparse
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from financial_rag.agent.loop import run_agent
from financial_rag.evaluation.adaptive_query_ablation import summarize
from financial_rag.evaluation.batch_query_ablation import TrackedGenerator, CaseRetriever, sha
from financial_rag.evaluation.evidence_content import match_unit, score_content
from financial_rag.evaluation.financebench import evaluation_question, write_json
from financial_rag.evaluation.retrieval_k_ablation import evaluate_ids

GRID=[dict(name='capped2_b8',queries=2,initial_k=8),dict(name='adaptive_b8',queries=None,initial_k=8),dict(name='adaptive_b12',queries=None,initial_k=12)]
BENCHMARK_INDICES=[0,1,5,6,10,34,47]


def unit(label,phrases,numbers=(),span=240):
    return dict(label=label,anchors=[dict(phrases=[p]) for p in phrases]+[dict(number=str(n)) for n in numbers],max_span=span)


def build_cases(frozen,chunks):
    old=json.loads((frozen/'selected_cases.json').read_text());rubrics=json.loads((frozen/'evidence_criteria.json').read_text())
    cases=[dict(old[i],cohort='financebench7') for i in BENCHMARK_INDICES]
    criteria=[rubrics[i] for i in BENCHMARK_INDICES]
    specs=[
        dict(id='custom_revenue_amazon_2017',company='Amazon',doc='AMAZON_2017_10K',
             question="Analyze Amazon's FY2017 revenue performance using its annual report: quantify total net sales and year-over-year growth, compare the revenue and growth contributions of its main business segments, and explain the main drivers of retail and AWS sales growth. State the fiscal period and units, and cite supporting evidence.",
             answer='FY2017 net sales $177,866m vs $135,987m, +30.8% (~31%). North America $106,110m vs $79,785m; International $54,297m vs $43,983m; AWS $17,459m vs $12,219m. Retail: unit sales, third-party sellers, Whole Foods in North America. AWS: increased customer usage, partly offset by pricing changes. Amounts and causes in report physical pages25-26, statements38, segment note69.',
             units=[unit('总营收及同比计算输入',['Consolidated'],[135987,177866]),
                    unit('北美营收及同比计算输入',['North America'],[79785,106110]),
                    unit('国际营收及同比计算输入',['International'],[43983,54297]),
                    unit('AWS营收及同比计算输入',['AWS'],[12219,17459]),
                    unit('零售增长原因',['North America sales increased','increased unit sales','third-party sellers','acquisition of Whole Foods Market'],span=520),
                    unit('AWS增长原因',['AWS sales increased','increased customer usage','partially offset by pricing changes'],span=340)]),
        dict(id='custom_revenue_adobe_2022',company='Adobe',doc='ADOBE_2022_10K',
             question="Analyze Adobe's FY2022 revenue performance using its annual report: quantify total revenue and year-over-year growth, compare the revenue and changes of its main business segments, explain the main growth drivers, and discuss the impact of foreign exchange and hedging on reported revenue. State the fiscal period and units, and cite supporting evidence.",
             answer='FY2022 revenue $17,606m vs $15,785m, +11.5% (~12%). Digital Media $12,842m vs $11,520m, Digital Experience $4,422m vs $3,867m, Publishing/Advertising $342m vs $398m. Growth from Creative/Document Cloud subscriptions and customer acquisition/engagement; Digital Experience subscription additions. FX reduced revenue about $486m, partly offset by $176m net hedging gains. Report physical pages40-42 and65.',
             units=[unit('总营收及同比计算输入',['Total revenue'],[17606,15785]),
                    unit('Digital Media营收及同比输入',['Digital Media'],[12842,11520]),
                    unit('Digital Experience营收及同比输入',['Digital Experience'],[4422,3867]),
                    unit('Publishing and Advertising营收及同比输入',['Publishing and Advertising'],[342,398]),
                    unit('订阅及客户需求增长驱动',['Revenue from Digital Media increased','Creative and Document Cloud subscription offerings','customer acquisition and engagement'],span=500),
                    unit('汇率及套期保值影响',['decreased revenue','net hedging gains'],[486,176],span=450)]),
        dict(id='custom_revenue_3m_2022',company='3M',doc='3M_2022_10K',
             question="Analyze 3M's FY2022 revenue performance using its annual report: quantify total net sales and year-over-year change, compare the sales and changes of its four main business segments, and explain how organic sales, divestitures and foreign-currency translation contributed to the overall sales change. State the fiscal period and units, and cite supporting evidence.",
             answer='FY2022 net sales $34,229m vs $35,355m, -3.2%. Safety/Industrial $11,604m vs $11,981m; Transportation/Electronics $8,902m vs $9,262m; Health Care $8,421m vs $8,597m; Consumer $5,298m vs $5,513m. Company organic sales +1.2%, divestitures -0.5%, currency translation -3.9%, total -3.2%. Physical page25 tables, statement48.',
             units=[unit('总营收及同比计算输入',['Net sales'],[34229,35355]),
                    unit('Safety and Industrial营收及同比输入',['Safety and Industrial'],[11604,11981]),
                    unit('Transportation and Electronics营收及同比输入',['Transportation and Electronics'],[8902,9262]),
                    unit('Health Care营收及同比输入',['Health Care'],[8421,8597]),
                    unit('Consumer营收及同比输入',['Consumer'],[5298,5513]),
                    unit('整体营收变动拆解：有机/剥离/汇率',['Worldwide Sales Change By Business Segment','Organic sales Acquisitions Divestitures Translation','Total Company 1.2 — (0.5) (3.9) (3.2)'],span=650)])]
    pages=defaultdict(list)
    for c in chunks:pages[(c['document_id'],c['page'])].append(c)
    page_text={key:''.join(c['text'] for c in sorted(cs,key=lambda c:c['char_start'])) for key,cs in pages.items()}
    for spec in specs:
        units=[]
        for i,u in enumerate(spec['units']):
            matching=[page for (doc,page),txt in page_text.items() if doc==spec['doc'] and match_unit(txt,u)]
            if not matching:raise ValueError('Missing source evidence: '+spec['id']+' '+u['label'])
            units.append(dict(u,id=f'u{i+1}',pages=sorted(matching)))
        rubric=dict(id=spec['id'],document_id=spec['doc'],units=units,manual_required=False,
                    note='Custom local six-unit rubric, defined before execution. Numeric coverage checks operands, not arithmetic or semantic answer correctness. All matching pages in target report allowed; labels/year/unit associations require manual interpretation.')
        assert score_content(chunks,rubric)['coverage']==1
        assert score_content([],rubric)['coverage']==0
        criteria.append(rubric)
        evidence=[dict(doc_name=spec['doc'],evidence_page_num=page-1,evidence_text=page_text[(spec['doc'],page)]) for page in sorted({p for u in units for p in u['pages']})]
        cases.append(dict(financebench_id=spec['id'],company=spec['company'],doc_name=spec['doc'],question=spec['question'],answer=spec['answer'],
                          evidence=evidence,question_type='custom-revenue-analysis',cohort='custom3',dataset_subset_label='LOCAL_EXTENSION_PUBLIC_REPORTS'))
    assert len(cases)==10 and len({c['financebench_id'] for c in cases})==10
    return cases,criteria


def report(out,summary,records,cases,pairs,verification):
    e=lambda x:html.escape(str(x));pct=lambda x:f'{x:.1%}'
    def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10道相同题 · 自由规划查询对照</title><style>body{font:16px/1.7 system-ui;background:#f4f6fa;color:#243047;margin:0}main{max-width:1250px;margin:auto;padding:32px}.note{background:#e5eefb;padding:18px}table{border-collapse:collapse;background:white;white-space:nowrap;width:100%}td,th{border:1px solid #dce1eb;padding:10px;text-align:left}.scroll{overflow:auto}details{background:white;padding:14px;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#175eb1}</style><main><h1>相同10题：最多两条 vs 自由规划</h1><p class="note">7道原FinanceBench题＋3道本地扩展综合营收分析题（Amazon FY2017、Adobe FY2022、3M FY2022）。三组各10题、共30次，不做晋级扩跑。扩展题和评分规则在运行前固定，单列结果；不能称为官方FinanceBench10题得分。</p>'''
    for cohort,label in [('all10','全部10题'),('financebench7','原FinanceBench 7题'),('custom3','综合营收分析 3题（本地扩展）')]:
        page+='<h2>'+label+'</h2>'+table(['配置','首次覆盖','最终覆盖','全部找全','LLM调用','首批查询均值','总检索均值','墙钟秒','扣本地排队秒','错误'],[
            [s['name'],pct(s[cohort]['initial_coverage']),pct(s[cohort]['final_coverage']),f"{round(s[cohort]['final_all']*s[cohort]['n'])}/{s[cohort]['n']}",f"{s[cohort]['sdk_calls_mean']:.2f}",f"{s[cohort]['initial_searches_mean']:.2f}",f"{s[cohort]['searches_mean']:.2f}",f"{s[cohort]['seconds_mean']:.1f}",f"{s[cohort]['seconds_without_local_queue_mean']:.1f}",s[cohort]['errors']] for s in summary])
    page+='<p>capped2为最多两条；adaptive由模型自行决定数量；b8/b12是首批总证据数，按查询数均分，共享chunk ID去重。所有组后续补查默认Top7，模型可指定1～20。先对比相同预算8下的规划方式，再看自由规划预算8→12。</p><h2>逐题配对变化</h2>'+table(['题目']+[s['name']+'最终覆盖' for s in summary],[[c['company']+' · '+c['financebench_id']]+[pct(records[s['name']][c['financebench_id']]['final']['content']['coverage']) for s in summary] for c in cases])
    page+='<h2>实际规划数量与停止</h2>'+table(['配置','全部10题规划数量分布','综合3题规划数量分布','正常完成','停止原因','规划回退','未执行规划查询'],[[s['name'],json.dumps(s['all10']['planned_count_distribution']),json.dumps(s['custom3']['planned_count_distribution']),s['all10']['completed'],json.dumps(s['all10']['stop_reasons']),s['all10']['planning_fallbacks'],s['all10']['unexecuted_queries']] for s in summary])
    page+='<h2>方法与限制</h2><p>语料仍为41份公开完整报告、6000页、18806个300-token/0-overlap分块。BGE-small FP32、Dense/BM25各20、RRF k60保留20、BGE reranker 2048。最多8次检索、7个模型步骤、60000证据字符/120000消息字符。提示词不传答案、评分规则或证据页；目标报告名作为公开问题元数据传入，未按文档过滤。</p><p>所有组重新调用DeepSeek Flash，max_tokens8192、temperature请求值0；推理模式仍有随机性。三个配置的顺序交错随机化，2个问题并发，本地模型串行。时间包含网络/缓存/排队，不是严格单请求延迟。仅一次小样本试验，不做显著性或最优配置承诺。</p><p>综合题每题预设六个证据单元，查验全部单元能在冻结语料中匹配、空证据为零。扫描目标文档所有页寻找同样锚点，不只认可某一页；数值单元检验同比计算输入，不等于验证计算/年份列/单位正确。规则覆盖不等于语义答案准确率。七道原题与三道扩展题分开报告。每题等权宏平均，不把证据单元数量多的题加权。</p><h2>逐题规划、回答与证据</h2>'
    criteria={r['id']:r for r in json.loads((out/'evidence_criteria.json').read_text())}
    for c in cases:
        cid=c['financebench_id'];page+='<details><summary>'+e(c['company']+' · '+cid)+'</summary><p>'+e(c['question'])+'</p><p>本地参考（未传入模型）：'+e(c['answer'])+'</p><details><summary>预先固定的证据清单</summary><pre>'+e(json.dumps(criteria[cid],ensure_ascii=False,indent=2))+'</pre></details>'
        for s in summary:
            r=records[s['name']][cid];res=r['result']
            page+='<details><summary>'+e(s['name']+f" · {r['sdk_calls']}次LLM · 覆盖{pct(r['final']['content']['coverage'])}")+'</summary>'
            for label,value in [('初始规划',res.get('query_plan')),('补查记录',res.get('search_history')),('回答',res.get('answer')),('最终证据评分',r['final']['content'])]:page+='<p>'+label+'</p><pre>'+e(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
            page+='<a href="cases/'+s['name']+'/'+cid+'.json">完整轨迹</a></details>'
        page+='</details>'
    page+='<h2>核验</h2><pre>'+e(json.dumps(verification,ensure_ascii=False,indent=2))+'</pre><p>'+' · '.join('<a href="'+n+'">'+n+'</a>' for n in ('summary.json','comparison.csv','paired_comparison.json','selected_cases.json','evidence_criteria.json','config.json','verification.json','rubric_validation.json'))+'</p></main></html>'
    (out/'report.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    root=a.project_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=root/'outputs/retrieval_k_ablation/financebench50_20261015';old=root/'outputs/batch_query_ablation/financebench50_20261015_8192'
    oldconfig=json.loads((old/'config.json').read_text())
    assert all(sha(frozen/n)==h for n,h in oldconfig['input_sha256'].items())
    chunks=json.loads((frozen/'chunks.json').read_text());byid={c['chunk_id']:c for c in chunks};cases,criteria=build_cases(frozen,chunks)
    for name,data in [('selected_cases.json',cases),('evidence_criteria.json',criteria)]:
        if (out/name).exists():assert json.loads((out/name).read_text())==data
        write_json(out/name,data)
    validation=[dict(id=r['id'],full_corpus=score_content(chunks,r),empty_corpus=score_content([],r)) for r in criteria if r['id'].startswith('custom_')]
    write_json(out/'rubric_validation.json',validation)
    sources=[root/'src/financial_rag'/n for n in ['agent/loop.py','agent/planner.py','generation/generator.py','generation/prompt.py','retrieval/retriever.py','retrieval/tool.py','retrieval/reranker.py','memory/evidence.py','evaluation/matched_query_ablation.py','evaluation/adaptive_query_ablation.py','evaluation/batch_query_ablation.py','evaluation/retrieval_k_ablation.py','evaluation/evidence_content.py','evaluation/financebench.py']]
    config=dict(grid=GRID,seed=20261015,workers=2,questions=10,financebench_questions=7,custom_questions=3,benchmark_indices=BENCHMARK_INDICES,
                selection='Seven deliberately varied extraction, ratios, explanation and geography cases from existing set, plus three independently authored revenue analyses of existing public reports; no selection on new outcomes.',
                input_sha256=oldconfig['input_sha256'],cases_sha256=sha(out/'selected_cases.json'),criteria_sha256=sha(out/'evidence_criteria.json'),
                source_sha256={str(s.relative_to(root)):sha(s) for s in sources},candidate_k=20,rrf_k=60,followup_top_k=7,max_searches=8,max_steps=7,max_evidence_chars=60000,max_message_chars=120000,model='deepseek-flash',max_tokens=8192,temperature=0,
                note='Exactly30 fresh case/configuration runs. Custom annotations only for local evidence proxy, not official FinanceBench labels. No answer accuracy evaluation.')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config,'Config changed'
    write_json(out/'config.json',config);shutil.copyfile(old/'public_data_verification.json',out/'public_data_verification.json')
    for source in sources:
        dest=out/'source'/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    print('PREPARED 10 matched cases; 3 custom rubrics fully source-matched; no model requests yet',flush=True)
    if a.prepare_only:return
    from dotenv import load_dotenv
    load_dotenv(root/'.env')
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    from financial_rag.retrieval.retriever import Retriever
    vectors=np.load(frozen/'vectors.npy');assert vectors.shape==(18806,384)
    embedding=EmbeddingModel(model_path=str(root/'models/bge-small-en-v1.5'),local_files_only=True)
    reranker=Reranker('BAAI/bge-reranker-v2-m3',2048,model_path=str(root/'models/bge-reranker-v2-m3'),local_files_only=True)
    class Store:
        def load(self,name):return chunks,vectors
    base=Retriever(Store(),embedding,'BAAI/bge-small-en-v1.5',reranker,candidate_k=20,top_k=7)
    embedding.encode_query('warmup');reranker.rerank('warmup',[chunks[0]],top_k=1)
    lock=threading.Lock();records={cfg['name']:{} for cfg in GRID}
    def run_one(cfg,i):
        c=cases[i];dest=out/'cases'/cfg['name']/(c['financebench_id']+'.json')
        if dest.exists():return json.loads(dest.read_text())
        ret=CaseRetriever(base,lock,7);gen=TrackedGenerator();start=time.perf_counter()
        try:result=run_agent(evaluation_question(c),ret,gen,use_query_planning=True,initial_query_count=cfg['queries'],initial_evidence_k=cfg['initial_k'])
        except Exception as exc:result=dict(status='error',error_type=type(exc).__name__,evidence={},initial_chunk_ids=[],search_attempts=0)
        finally:gen.client.close()
        wall=time.perf_counter()-start
        usage={k:sum(x.get('usage',{}).get(k,0) or 0 for x in gen.calls) for k in ['prompt_tokens','completion_tokens','total_tokens']}
        record=dict(id=c['financebench_id'],company=c['company'],cohort=c['cohort'],configuration=cfg,result=result,
                    initial=evaluate_ids(result.get('initial_chunk_ids',[]),byid,c,criteria[i]),final=evaluate_ids(list(result.get('evidence',{})),byid,c,criteria[i]),
                    sdk_calls=len(gen.calls),http_attempts=gen.http_attempts,api_usage=gen.calls,usage=usage,wall_seconds=wall,
                    local_queue_seconds=ret.wait_seconds,local_active_seconds=ret.active_seconds,seconds_without_local_queue=wall-ret.wait_seconds)
        write_json(dest,record)
        print('DONE',cfg['name'],i+1,c['company'],'queries',result.get('initial_searches'),'calls',record['sdk_calls'],'coverage',record['final']['content']['coverage'],'seconds',round(wall,1),'status',result['status'],flush=True)
        return record
    # Submit only two jobs at a time, so an interrupted run cannot drain a queued full grid.
    jobs=[(cfg,i) for cfg in GRID for i in range(10)];random.Random(20261015).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0,len(jobs),2):
            futures={pool.submit(run_one,cfg,i):cfg for cfg,i in jobs[offset:offset+2]}
            for future in as_completed(futures):
                cfg=futures[future];r=future.result();records[cfg['name']][r['id']]=r
                write_json(out/'progress.json',dict(completed=sum(len(x) for x in records.values()),total=30))
            if all(f.result()['result']['status']=='error' for f in futures):raise RuntimeError('Both active cases failed; stopped with saved results')
    summary=[]
    for cfg in GRID:
        rs=list(records[cfg['name']].values());assert len(rs)==10
        summary.append(dict(cfg,all10=summarize(rs),financebench7=summarize([r for r in rs if r['cohort']=='financebench7']),custom3=summarize([r for r in rs if r['cohort']=='custom3'])))
    pairs={}
    for new,oldname in [('adaptive_b8','capped2_b8'),('adaptive_b12','adaptive_b8'),('adaptive_b12','capped2_b8')]:
        diffs=[]
        for c in cases:
            aa,bb=records[oldname][c['financebench_id']],records[new][c['financebench_id']]
            diffs.append(dict(id=c['financebench_id'],cohort=c['cohort'],coverage=bb['final']['content']['coverage']-aa['final']['content']['coverage'],calls=bb['sdk_calls']-aa['sdk_calls'],seconds=bb['wall_seconds']-aa['wall_seconds'],both_completed=aa['result']['status']==bb['result']['status']=='completed'))
        pairs[new+'_vs_'+oldname]=dict(cases=diffs,improved=sum(x['coverage']>1e-9 for x in diffs),worse=sum(x['coverage']< -1e-9 for x in diffs),unchanged=sum(abs(x['coverage'])<=1e-9 for x in diffs),mean_coverage_delta=float(np.mean([x['coverage'] for x in diffs])),mean_calls_delta=float(np.mean([x['calls'] for x in diffs])),mean_seconds_delta=float(np.mean([x['seconds'] for x in diffs])))
    write_json(out/'summary.json',summary);write_json(out/'paired_comparison.json',pairs)
    fields=['n','initial_coverage','final_coverage','final_all','sdk_calls_mean','seconds_mean','seconds_without_local_queue_mean','initial_searches_mean','searches_mean','errors']
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['configuration','cohort']+fields);w.writeheader()
        for s in summary:
            for cohort in ['all10','financebench7','custom3']:w.writerow(dict(configuration=s['name'],cohort=cohort,**{k:s[cohort][k] for k in fields}))
    rs=[r for group in records.values() for r in group.values()]
    assert len(rs)==30 and all(len(r['result'].get('initial_chunk_ids',[]))<=r['configuration']['initial_k'] and r['result'].get('search_attempts',0)<=8 for r in rs)
    assert all(sha(root/name)==h for name,h in config['source_sha256'].items())
    assert sha(out/'selected_cases.json')==config['cases_sha256'] and sha(out/'evidence_criteria.json')==config['criteria_sha256']
    verification=dict(runs=30,matched_cases=10,financebench_cases=7,custom_cases=3,custom_units=18,pre_run_rubrics_source_validated=True,source_and_rubric_hashes_unchanged=True,
                      sdk_calls=sum(r['sdk_calls'] for r in rs),http_attempts=sum(r['http_attempts'] for r in rs),api_total_tokens=sum(r['usage']['total_tokens'] for r in rs),
                      errors=sum(r['result']['status']=='error' for r in rs),length_limited_responses=sum(call.get('finish_reason')=='length' for r in rs for call in r['api_usage']),
                      unexecuted_planned_queries=sum(len(r['result'].get('initial_queries_not_executed',[])) for r in rs),budgets_checked=True,answer_accuracy_evaluated=False,application_default_queries=1)
    write_json(out/'verification.json',verification);report(out,summary,records,cases,pairs,verification)
    print('FINISHED',json.dumps(verification),flush=True)
    for s in summary:print('SUMMARY',s['name'],json.dumps(s['all10']),flush=True)


if __name__=='__main__':main()
