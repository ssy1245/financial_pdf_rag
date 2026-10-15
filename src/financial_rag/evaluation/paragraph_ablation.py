"""Controlled paragraph-vs-window experiment on the frozen 10-case chunk study.

Standalone entry point; imports the project's unchanged parser, metrics and
baseline splitter. No network, generation, query instructions or app changes.
"""
import argparse
import csv
import hashlib
import html
import json
import re
import time
from pathlib import Path


def paragraph_chunks(pages, document_id, tokenizer, size=300, pack=True):
    """Use blank-line/source-block paragraphs, preserve exact page source spans.

    Ordinary paragraphs are indivisible. Only paragraphs exceeding the content
    token cap are split, first at sentence ends and then with the existing
    source-preserving token splitter. No overlap or cross-page merging. Sentence
    boundaries are a heuristic, not an NLP sentence recognizer.
    """
    from financial_rag.evaluation.chunk_ablation import token_chunks
    if size < 1:
        raise ValueError('size must be positive')
    count = lambda t: len(tokenizer(t, add_special_tokens=False, truncation=False)['input_ids'])
    chunks = []
    for page in pages:
        text = page['text']
        starts = [0] + [m.end() for m in re.finditer(r'\n\s*\n', text)]
        units = []
        for i, start in enumerate(starts):
            end = starts[i+1] if i+1 < len(starts) else len(text)
            if not text[start:end].strip():
                continue
            # Include separators in source spans to make adjacent spans joinable.
            paragraph = text[start:end]
            if count(paragraph) <= size:
                units.append((start, end, i, False))
                continue
            sentence_ends = [m.end() for m in re.finditer(r'[.!?。！？](?:[\"\u201d\u2019)]*)\s+(?=[A-Z0-9\u4e00-\u9fff])', paragraph)] + [len(paragraph)]
            pending_start = 0
            last_end = 0
            for sentence_end in sentence_ends:
                if count(paragraph[pending_start:sentence_end]) <= size:
                    last_end = sentence_end
                    continue
                if last_end > pending_start:
                    units.append((start+pending_start, start+last_end, i, True))
                    pending_start = last_end
                if count(paragraph[pending_start:sentence_end]) > size:
                    fragment = paragraph[pending_start:sentence_end]
                    split = token_chunks([{'page':page['page'], 'text':fragment}], document_id, tokenizer, size, 0)
                    for c in split:
                        units.append((start+pending_start+c['char_start'], start+pending_start+c['char_end'], i, True))
                    pending_start = sentence_end
                last_end = sentence_end
            if pending_start < len(paragraph):
                units.append((start+pending_start, end, i, True))
        current = []
        page_index = 0
        def emit():
            nonlocal page_index
            if not current:
                return
            a, b = current[0][0], current[-1][1]
            part = text[a:b]
            touched = sorted({u[2] for u in current})
            chunks.append(dict(document_id=document_id, page=page['page'],
                chunk_id=f'{document_id}_p{page["page"]}_c{page_index}', text=part,
                char_start=a, char_end=b, word_count=len(part.split()),
                paragraph_ids=touched, split_long_paragraph=any(u[3] for u in current)))
            page_index += 1
            current.clear()
        for unit in units:
            # Long-paragraph fragments remain isolated, making the rule explicit.
            if current and (not pack or unit[3] or current[-1][3] or count(text[current[0][0]:unit[1]]) > size):
                emit()
            current.append(unit)
        emit()
    return chunks


def check_spans(pages_by_doc, chunks, tokenizer, cap):
    grouped = {}
    for c in chunks:
        grouped.setdefault((c['document_id'],c['page']), []).append(c)
        assert len(tokenizer(c['text'], add_special_tokens=False)['input_ids']) <= cap
    for doc, pages in pages_by_doc.items():
        for page in pages:
            end = 0
            for c in sorted(grouped.get((doc,page['page']), []), key=lambda c:c['char_start']):
                a,b = c['char_start'],c['char_end']
                assert not page['text'][end:a].strip(), (doc,page['page'],end,a)
                assert c['text'] == page['text'][a:b]
                end=max(end,b)
            assert not page['text'][end:].strip(), (doc,page['page'],end)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(out, summaries, results, cases, chunks_by_name, config):
    e=lambda x:html.escape(str(x))
    pct=lambda x:f'{x:.1%}'
    auto_count=summaries[0]['top5']['content_n']
    def table(headers, rows):
        return '<div class="scroll"><table><tr>'+''.join('<th>'+e(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>段落分块与窗口分块对照实验</title><style>body{font:16px/1.65 system-ui;color:#223047;background:#f4f6fa;margin:0}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:30px}h2{margin-top:32px}table{border-collapse:collapse;background:white;white-space:nowrap;width:100%}td,th{padding:10px;border:1px solid #dce1eb;text-align:left}th{background:#e9eef7}.scroll{overflow:auto}details{background:white;border:1px solid #dce1eb;padding:12px;margin:12px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:14px}a{color:#175eb1}.note{padding:16px;background:#e6eefb;border-radius:8px}</style><main><h1>段落分块与窗口分块对照实验</h1>'''
    report+='<p class="note">固定此前的 10 道开发题、10 份完整报告、原清洗页面、BGE-small FP32、英文原题＋报告名、无查询指令。11 组均在本次设备重新编码；仅 Dense 检索，无重排或生成。段落指原 PDF 文本块／空行边界，未经人工语义段落标注。</p>'
    report+='<h2>页级指标：与原窗口实验相同口径</h2>'
    report+=table(['方案','块数','token中位数','截断块','Top5页命中','Top5页覆盖','Top10页覆盖','Top20页覆盖','2000-token页覆盖','编码秒'],[[s['label'],s['chunks'],f"{s['token_p50']:.0f}",s['truncated_chunks'],pct(s['top5']['hit']),pct(s['top5']['coverage']),pct(s['top10']['coverage']),pct(s['top20']['coverage']),pct(s['budget2000']['coverage']),f"{s['encoding_seconds']:.1f}"] for s in summaries])
    report+=f'<h2>内容指标：有没有找到实际证据</h2><p>沿用现有 50 题标注中对应原 10 题的证据规则，本次 {auto_count} 题进入自动评分。数值与行名局部匹配，可拼接实际返回且原文相邻的片段。此项不是答案准确率，也尚未全面核实年度列与单位。</p>'
    report+=table(['方案','Top5有效证据命中','Top5平均证据覆盖','Top5全部单元找全','Top20证据覆盖','2000-token证据覆盖','2000-token全部找全'],[[s['label'],pct(s['top5']['content_hit']),pct(s['top5']['content_coverage']),pct(s['top5']['content_all_units']),pct(s['top20']['content_coverage']),pct(s['budget2000']['content_coverage']),pct(s['budget2000']['content_all_units'])] for s in summaries])
    report+='<h2>逐题指标和原文</h2><p>表内为 Top5 页覆盖 / 证据覆盖；“人工”表示不参与自动评分。</p>'
    report+=table(['公司']+[s['label'] for s in summaries],[[c['company']]+[pct(results[s['name']][i]['top5']['coverage'])+' / '+('人工' if results[s['name']][i]['top5']['content']['coverage'] is None else pct(results[s['name']][i]['top5']['content']['coverage'])) for s in summaries] for i,c in enumerate(cases)])
    for i,case in enumerate(cases):
        report+='<details><summary>'+e(case['company']+' · '+case['financebench_id'])+'</summary><p>'+e(case['question'])+'</p><p>参考答案（仅用于复核）：'+e(case.get('answer',''))+'</p>'
        for s in summaries:
            row=results[s['name']][i]
            report+='<details><summary>'+e(s['label'])+' · 页覆盖 '+pct(row['top5']['coverage'])+' · 证据覆盖 '+('人工' if row['top5']['content']['coverage'] is None else pct(row['top5']['content']['coverage']))+'</summary>'
            by_id={c['chunk_id']:c for c in chunks_by_name[s['name']]}
            for rank,hit in enumerate(row['ranking'][:5],1):
                c=by_id[hit['chunk_id']]
                report+='<p><b>'+e(f'#{rank} {c["document_id"]} · 物理页 {c["page"]} · {c["content_tokens"]} tokens')+'</b></p><pre>'+e(c['text'])+'</pre>'
            report+='<p>实际命中依据：</p><pre>'+e(json.dumps(row['top5']['content'],ensure_ascii=False,indent=2))+'</pre></details>'
        report+='</details>'
    report+='<h2>方法和限制</h2><ul><li>原窗口算法本已优先段落末尾；新策略区别在于普通段落不被拆开。完整段落合并只在加上下一段不超预算时进行；长段优先句界再退回 token 拆分，长段碎片独立成块。</li><li>所有长度上限按内容 tokens 计，不含两个特殊 tokens。新段落策略不重叠、不跨页。逐页验证所有非空白原文均被覆盖、字符来源一致及实际 token 上限。</li><li>原 500 词配置可能发生模型截断，保留原行为并报告截断数量。其他配置须通过无截断检查。</li><li>2000-token 预算使用完整片段的排名前缀，不跳过、不裁剪；不同块大小会改变能放入的片段数，因此同时看 TopK 和固定预算。</li><li>10 题已用于开发，单题即可改变命中率 10 个百分点。结果不能推广到新题、中文、Word/Excel、结构化表格或端到端问答。</li><li>与原实验的逐题页级指标差异保存在 historical_comparison.json；如设备数值精度造成排名变化，以本次同条件对照为主。</li></ul>'
    report+='<p><a href="comparison.csv">指标 CSV</a> · <a href="summary.json">汇总 JSON</a> · <a href="config.json">配置与哈希</a> · <a href="paired_comparison.json">相对 300/0 逐题变化</a> · <a href="historical_comparison.json">历史复现核对</a></p></main></html>'
    (out/'report.html').write_text(report,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.project_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    import platform
    import importlib.metadata
    from copy import deepcopy
    from sentence_transformers import SentenceTransformer
    from financial_rag.evaluation.chunk_ablation import token_chunks, metric, select_budget
    from financial_rag.ingestion.chunker import chunk_pages
    from financial_rag.evaluation.financebench import evaluation_question, gold_pages, write_json
    from financial_rag.evaluation.evidence_content import score_content
    old=root/'outputs/chunk_ablation/financebench_20261013'
    oldconfig=json.loads((old/'config.json').read_text())
    for rel,expected in oldconfig['source_sha256'].items():
        assert sha(root/rel)==expected, f'Historical source changed: {rel}'
    model_path=root/'models/bge-small-en-v1.5'
    for rel,expected in oldconfig['model_files_sha256'].items():
        assert sha(model_path/rel)==expected, f'Model changed: {rel}'
    pages=json.loads((old/'normalized_pages.json').read_text())
    cases=json.loads((old/'selected_cases.json').read_text())
    criteria_path=root/'src/financial_rag/evaluation/data/financebench50_evidence.json'
    all_criteria={c['id']:c for c in json.loads(criteria_path.read_text())}
    criteria=[all_criteria[c['financebench_id']] for c in cases]
    assert len(cases)==10 and len(pages)==10
    configs=[dict(name=name,size=size,overlap=overlap,unit=unit,label=('旧：500词 / 80重叠' if unit=='words' else f'旧：{size} tokens / {overlap}重叠')) for name,size,overlap,unit in oldconfig['configs']]
    configs += [dict(name='paragraph_single_cap300',size=300,overlap=0,unit='paragraph_single',label='一段一块（上限300）')]
    configs += [dict(name=f'paragraph_pack_cap{s}',size=s,overlap=0,unit='paragraph_pack',label=f'完整段落合并（上限{s}）') for s in (200,300,400)]
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=SentenceTransformer(str(model_path),local_files_only=True,device=device,model_kwargs={'dtype':torch.float32})
    assert model.max_seq_length==512
    tok=deepcopy(model.tokenizer);tok.model_max_length=10**9
    config=dict(frozen_baseline=str(old),frozen_pages_sha256=sha(old/'normalized_pages.json'),
        cases_sha256=sha(old/'selected_cases.json'),evidence_criteria_sha256=sha(criteria_path),
        script_sha256=sha(Path(__file__)),model_files_sha256=oldconfig['model_files_sha256'],
        historical_source_sha256=oldconfig['source_sha256'],configs=configs,device=device,dtype='float32',batch_size=32,
        model='BAAI/bge-small-en-v1.5',max_seq_length=512,content_limit=510,
        query_policy=oldconfig['query_policy'],retrieval=oldconfig['retrieval'],budget_policy=oldconfig['budget_policy'],
        paragraph_policy='Blank-line PDF source-block boundaries; normal paragraphs indivisible; greedy packing optional; oversized paragraphs sentence-first then source-preserving token split; no overlap/cross-page.',
        platform=platform.platform(),versions={x:importlib.metadata.version(x) for x in ('torch','sentence-transformers','transformers','numpy')},
        limitations=f'10 previously inspected development cases; paragraph boundaries heuristic; {sum(not c.get("manual_required",False) for c in criteria)} automatic content cases; no generation or answer accuracy.')
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config,'Existing output config changed'
    write_json(out/'config.json',config);write_json(out/'selected_cases.json',cases);write_json(out/'evidence_criteria.json',criteria)
    (out/'paragraph_ablation.py').write_bytes(Path(__file__).read_bytes())
    queries=[evaluation_question(c) for c in cases]
    qlengths=[len(t) for t in tok(queries,add_special_tokens=True)['input_ids']]
    assert max(qlengths)<=512
    model.encode(['warmup'],prompt='',normalize_embeddings=True,show_progress_bar=False)
    qv=model.encode(queries,prompt='',batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    np.save(out/'query_vectors.npy',qv)
    write_json(out/'query_audit.json',dict(lengths=qlengths,truncated=0))
    def sync():
        if device=='mps':torch.mps.synchronize()
    summaries=[];all_results={};all_chunks={}
    for cfg in configs:
        name=cfg['name'];folder=out/name;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():
            summaries.append(json.loads((folder/'summary.json').read_text()))
            all_results[name]=json.loads((folder/'results.json').read_text())
            all_chunks[name]=json.loads((folder/'chunks.json').read_text())
            print('RESUME',name,flush=True);continue
        start=time.perf_counter();chunks=[]
        for doc,pp in pages.items():
            if cfg['unit']=='words': current=chunk_pages(pp,doc,cfg['size'],cfg['overlap'])
            elif cfg['unit']=='tokens':current=token_chunks(pp,doc,tok,cfg['size'],cfg['overlap'])
            else:current=paragraph_chunks(pp,doc,tok,cfg['size'],cfg['unit']=='paragraph_pack')
            chunks+=current
        if cfg['unit']!='words':check_spans(pages,chunks,tok,cfg['size'])
        lengths=[len(t) for t in tok([c['text'] for c in chunks],add_special_tokens=False)['input_ids']]
        for c,n in zip(chunks,lengths):c.update(content_tokens=n,truncated_tokens=max(0,n-510))
        assert cfg['unit']=='words' or max(lengths)<=510
        if cfg['unit'] in ('words','tokens'):
            assert chunks==json.loads((old/name/'chunks.json').read_text()),f'Historical chunks differ: {name}'
        write_json(folder/'chunks.json',chunks)
        chunk_seconds=time.perf_counter()-start
        print('ENCODE',name,len(chunks),'device',device,flush=True)
        blockdir=folder/'blocks';blockdir.mkdir(exist_ok=True);arrays=[];seconds=[]
        for offset in range(0,len(chunks),512):
            f=blockdir/f'{offset:06}.npy';tf=f.with_suffix('.json')
            if f.exists() and tf.exists():a=np.load(f);elapsed=json.loads(tf.read_text())['seconds']
            else:
                sync();begin=time.perf_counter()
                a=model.encode([c['text'] for c in chunks[offset:offset+512]],prompt='',batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
                sync();elapsed=time.perf_counter()-begin;np.save(f,a);write_json(tf,dict(seconds=elapsed,rows=len(a)))
            assert len(a)==len(chunks[offset:offset+512]);arrays.append(a);seconds.append(elapsed)
            print('PROGRESS',name,min(offset+512,len(chunks)),'/',len(chunks),flush=True)
        vectors=np.concatenate(arrays);assert np.isfinite(vectors).all();np.save(folder/'vectors.npy',vectors)
        rows=[]
        for case,query,crit in zip(cases,qv,criteria):
            scores=vectors@query;order=np.argsort(-scores,kind='stable')
            expected=gold_pages(case)
            row=dict(id=case['financebench_id'],company=case['company'],question=case['question'],expected_pages=sorted(expected))
            selections={f'top{k}':list(map(int,order[:k])) for k in (5,10,20)}
            chosen,used=select_budget(order,lengths,2000);selections['budget2000']=chosen
            for mode,indices in selections.items():
                selected=[chunks[i] for i in indices]
                row[mode]=dict(metric(indices,chunks,expected),used_tokens=sum(lengths[i] for i in indices),content=score_content(selected,crit))
            row['ranking']=[dict(chunk_id=chunks[int(i)]['chunk_id'],document_id=chunks[int(i)]['document_id'],page=chunks[int(i)]['page'],score=float(scores[i]),content_tokens=lengths[int(i)]) for i in order[:max(20,len(chosen))]]
            rows.append(row)
        stat=dict(cfg,chunks=len(chunks),token_p50=float(np.median(lengths)),token_p95=float(np.percentile(lengths,95)),token_max=max(lengths),truncated_chunks=sum(n>510 for n in lengths),total_content_tokens=sum(lengths),encoding_seconds=sum(seconds),chunk_seconds=chunk_seconds,vector_bytes=vectors.nbytes)
        for mode in ('top5','top10','top20','budget2000'):
            auto=[r[mode]['content'] for r in rows if r[mode]['content']['coverage'] is not None]
            stat[mode]={k:float(np.mean([r[mode][k] for r in rows])) for k in ('hit','coverage','all_pages','mrr','chunks','same_page_repeat_fraction')}
            stat[mode].update(content_n=len(auto),content_hit=float(np.mean([r['hit'] for r in auto])),content_coverage=float(np.mean([r['coverage'] for r in auto])),content_all_units=float(np.mean([r['all_units'] for r in auto])))
        write_json(folder/'results.json',rows);write_json(folder/'summary.json',stat)
        summaries.append(stat);all_results[name]=rows;all_chunks[name]=chunks
        write_json(out/'summary.json',summaries)
        print('DONE',name,json.dumps(dict(top5=stat['top5'],budget=stat['budget2000'])),flush=True)
    reference=all_results['tokens300_overlap0']
    paired={}
    for cfg in configs[7:]:
        name=cfg['name'];paired[name]={}
        for mode in ('top5','top10','top20','budget2000'):
            paired[name][mode]=[dict(id=c['financebench_id'],company=c['company'],old_page_coverage=a[mode]['coverage'],new_page_coverage=b[mode]['coverage'],old_content=a[mode]['content'],new_content=b[mode]['content']) for c,a,b in zip(cases,reference,all_results[name])]
    write_json(out/'paired_comparison.json',paired)
    historical={}
    for cfg in configs[:7]:
        name=cfg['name'];previous=json.loads((old/name/'results.json').read_text())
        differences=[]
        for a,b in zip(previous,all_results[name]):
            assert a['id']==b['id']
            for mode in ('top5','top10','top20','budget2000'):
                for key in ('hit','coverage','all_pages','mrr'):
                    if abs(a[mode][key]-b[mode][key])>1e-10:differences.append(dict(id=a['id'],mode=mode,key=key,old=a[mode][key],new=b[mode][key]))
        historical[name]=dict(chunks_exactly_reproduced=True,page_metric_differences=differences)
    write_json(out/'historical_comparison.json',historical)
    fields=['name','label','chunks','token_p50','token_p95','token_max','truncated_chunks','encoding_seconds']
    metrics=['hit','coverage','all_pages','mrr','content_hit','content_coverage','content_all_units']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+[f'{m}_{k}' for m in ('top5','top10','top20','budget2000') for k in metrics]);w.writeheader()
        for s in summaries:w.writerow({**{k:s[k] for k in fields},**{f'{m}_{k}':s[m][k] for m in ('top5','top10','top20','budget2000') for k in metrics}})
    build_report(out,summaries,all_results,cases,all_chunks,config)
    write_json(out/'verification.json',dict(configurations=len(summaries),cases=len(cases),documents=len(pages),all_baseline_chunks_exact=True,paragraph_source_coverage_passed=True,paragraph_token_caps_passed=True,automatic_content_cases=sum(not c.get('manual_required',False) for c in criteria),generation_calls=0,historical_page_metric_differences=sum(len(x['page_metric_differences']) for x in historical.values())))
    print('FINISHED',out,flush=True)


if __name__=='__main__':main()
