"""Build a self-contained HTML comparison from actual inference and official metrics."""
from pathlib import Path
from collections import Counter
from statistics import mean
from urllib.parse import quote
import html,json,math
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/ocr_ablation/omnidoc100_20261013'
MODELS={'pp_structure':'PP-StructureV3','paddle_vl':'PaddleOCR-VL-1.6','mineru':'MinerU 4.0.12 standard'}
def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def aggregate(model,manifest):
    directory=RUN/model;result=RUN/'evaluation'/model/'result';prefix=model+'_quick_match'
    pages={x['image']:x for x in manifest};timings={};page_metrics={k:{} for k in pages}
    for item in manifest:
        status=read(directory/(Path(item['image']).stem+'.status.json'),{})
        timings[item['image']]=status
    for category in ['text_block','reading_order','table']:
        edits=read(result/(prefix+'_'+category+'_per_page_edit.json'),{})
        for name,value in edits.items():
            if name in page_metrics:page_metrics[name][category+'_edit']=value
    tables=read(result/(prefix+'_table_result.json'),[])
    table_scores={name:[] for name in pages};structure_scores={name:[] for name in pages}
    for item in tables:
        name=item.get('image_name',item.get('img_id'))
        if name not in pages:continue
        for metric,target in [('TEDS',table_scores),('TEDS_structure_only',structure_scores)]:
            score=item.get('metric',{}).get(metric)
            if isinstance(score,(float,int)):target[name].append(score)
    for name in pages:
        if table_scores[name]:page_metrics[name]['TEDS']=mean(table_scores[name]);page_metrics[name]['TEDS_structure_only']=mean(structure_scores[name])
    groups={}
    for group in ['all','dev','test']+sorted({x['stratum'] for x in manifest}):
        names=[x['image'] for x in manifest if group=='all' or x['split']==group or x['stratum']==group]
        records=[timings[n] for n in names];times=[x['seconds'] for x in records if 'seconds' in x]
        metrics={}
        for key in ['text_block_edit','reading_order_edit']:
            values=[page_metrics[n][key] for n in names if key in page_metrics[n]]
            metrics[key]=mean(values) if values else None
            metrics[key+'_pages']=len(values)
        for key,scores in [('TEDS',table_scores),('TEDS_structure_only',structure_scores)]:
            values=[v for n in names for v in scores[n]];metrics[key]=mean(values) if values else None;metrics[key+'_pairs']=len(values)
        groups[group]=dict(pages=len(names),completed=sum(bool(x) for x in records),ok=sum(x.get('status')=='ok' for x in records),mean_seconds=mean(times) if times else None,p95_seconds=sorted(times)[max(0,math.ceil(len(times)*.95)-1)] if times else None,**metrics)
    return dict(groups=groups,page_metrics=page_metrics,statuses=timings,official=read(result/(prefix+'_metric_result.json')),config=read(directory/'config.json',{}))

def fmt(v,percent=False):return '—' if v is None else f'{v*100 if percent else v:.2f}'+('%' if percent else '')
def esc(v):return html.escape(str(v))

def main():
    manifest=read(RUN/'sample_manifest.json')['pages'];results={model:aggregate(model,manifest) for model in MODELS}
    complete=all(r['groups']['all']['completed']==100 and r['official'] is not None for r in results.values())
    state=read(RUN/'experiment_state.json',{})
    paused=state.get('status')=='paused' and not complete
    summary=dict(status='complete' if complete else ('paused' if paused else 'in_progress'),experiment_state=state,models=results,sample=read(RUN/'sample_manifest.json'))
    (RUN/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    blocks=[]
    blocks.append('<h1>OCR 解析对比 · 公开 benchmark 100 页</h1><p class="lead">同一批图片 → 三套解析方案 → 官方文字、表格与阅读顺序评测。</p>')
    blocks.append('<p class="notice">'+('推理与评测已完成。' if complete else ('实验已暂停：已停止推理，保留样本、模型和已完成结果；目前不足以比较三种方案。未评测指标显示 —，不是 0 分。建议下一轮缩到 10 页，每个方案先试跑 2 页。' if paused else '实验进行中：未评测的指标显示 —，不是 0 分。'))+' 本轮不评估生成回答，不计算含公式 CDM 的官方 Overall。</p>')
    blocks.append('<nav><a href="../../../reports/index.html">全部实验报告</a> · <a href="sample_manifest.json">100 页清单与哈希</a> · <a href="summary.json">汇总 JSON</a></nav>')
    blocks.append('<h2>实验协议</h2><p>公开集：OmniDocBench，固定 revision '+esc(read(RUN/'dataset_revision.json')['revision'])+'。20 页用于适配调试，80 页用于比较；另报告全部 100 页结果。按文件名推断同源分组，UUID 文件无法确认父文档，划分不保证绝对文档隔离。</p>')
    blocks.append('<p>分层：'+esc(dict(Counter(x['stratum'] for x in manifest)))+'。research_report 是公开集的“报告类”，含技术资料，并非全为财报；独立模糊扫描来源不足，缺额按固定顺序补齐。输入使用原始公开图片；模型可执行各自内部缩放。所有推理均在本机执行。</p>')
    blocks.append('<p>PP-StructureV3 使用 CPU，关闭公式/图表/印章识别；PaddleOCR-VL 使用官方完整管线及本地视觉模型后端；MinerU 使用 standard 图像 OCR 模式。具体版本、配置及原始输出随报告保存。不同后端的耗时表示本机部署成本，不能直接当作模型算力效率排名。首张图耗时可能包含延迟初始化。</p>')
    blocks.append('<h2>核心指标</h2><p>文字与阅读顺序 Edit 越低越好，TEDS（结构和文字）、TEDS-S（仅结构）越高越好。表格按官方匹配对取均值；文字按页取均值。指标分母随有效标注类别变化，空白类别不强行赋分。金融事实元组准确率没有现成标注，本轮不报告。</p>')
    for group,title in [('test','固定测试集 · 80 页'),('dev','适配调试集 · 20 页'),('all','全部样本 · 100 页')]:
        blocks.append('<h2>'+title+'</h2><div class="scroll"><table><tr><th>方案</th><th>成功/计划</th><th>文字 Edit ↓</th><th>表格 TEDS ↑</th><th>TEDS-S ↑</th><th>顺序 Edit ↓</th><th>均值秒/页</th><th>P95 秒/页</th></tr>')
        for model,label in MODELS.items():
            r=results[model]['groups'][group]
            blocks.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [label,f"{r['ok']}/{r['pages']}",fmt(r['text_block_edit']),fmt(r['TEDS'],True),fmt(r['TEDS_structure_only'],True),fmt(r['reading_order_edit']),fmt(r['mean_seconds']),fmt(r['p95_seconds'])])+'</tr>')
        blocks.append('</table></div>')
    blocks.append('<h2>分组与逐页复习</h2><p>展开后对照原图和各方案的实际输出。表格 HTML 保留在原始 Markdown 中；引用链接可打开原始结构 JSON。</p><input id="filter" placeholder="筛选：页号、语言、类别、文件名" oninput="document.querySelectorAll(\'.page\').forEach(x=>x.hidden=!x.dataset.search.includes(this.value.toLowerCase()))">')
    for item in manifest:
        name=item['image'];stem=Path(name).stem
        blocks.append('<details class="page" data-search="'+esc((item['id']+' '+name+' '+str(item['attributes'])+' '+item['stratum']).lower())+'"><summary>'+esc(item['id']+' · '+item['split']+' · '+item['stratum']+' · '+item['attributes'].get('language',''))+'</summary><img loading="lazy" src="images/'+quote(name)+'" alt="'+esc(name)+'">')
        for model,label in MODELS.items():
            status=results[model]['statuses'][name];metrics=results[model]['page_metrics'][name]
            md=RUN/model/(stem+'.md');text=md.read_text() if md.exists() else '尚无输出'
            blocks.append('<details><summary>'+esc(label+' · '+status.get('status','pending')+' · '+fmt(status.get('seconds'))+' 秒 · '+str(metrics))+'</summary>')
            if status.get('error'):blocks.append('<p class="notice">'+esc(status['error'])+'</p>')
            blocks.append('<a href="'+model+'/'+quote(stem)+'.json">原始结构 JSON</a><pre>'+esc(text)+'</pre></details>')
        blocks.append('</details>')
    blocks.append('<h2>复现与限制</h2><p>官方 benchmark 代码、依赖版本、选样清单、模型配置、逐页运行记录和官方指标输出均保留在本实验目录。失败页面保留空预测参与评测，不能据部分完成结果宣布获胜模型。此为自选 100 页子集，不能与官方全量排行榜直接横比。</p>')
    css='body{font:16px/1.65 system-ui;color:#273d36;background:#f6f4ee;max-width:1200px;margin:45px auto;padding:20px}h1{font-size:38px}h2{margin-top:38px}a{color:#126b61}.lead{font-size:21px}.notice{background:#fff0d8;padding:18px;border-radius:10px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border:1px solid #d6ded7;text-align:left}details{background:white;border:1px solid #dde3dc;border-radius:10px;margin:12px 0;padding:15px}summary{cursor:pointer;font-weight:600}img{max-width:100%;max-height:1000px;display:block;margin:15px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace;max-height:650px;overflow:auto;background:#f1f4f2;padding:15px}input{padding:14px;width:90%;font:inherit}nav{margin:24px 0}'
    (RUN/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>OCR 100 页解析对比</title><style>'+css+'</style>'+''.join(blocks)+'</html>')
    print('report',summary['status'],{m:r['groups']['all']['completed'] for m,r in results.items()})

if __name__=='__main__':main()
