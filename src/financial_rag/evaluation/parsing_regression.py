"""Small offline source-structure regression; never calls generation APIs.

Inspect ten fixed public FinanceBench pages, preserving original physical page IDs.
Gold assertions focus on the four Verizon FY2022 quick-ratio inputs and both years.
Other pages are structural smoke tests, not claimed semantic ground truth.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter

from financial_rag.ingestion.adapters import _pdf
from financial_rag.ingestion.document import Document
from financial_rag.ingestion.structured_chunker import chunk_document

PAGES = {'VERIZON_2022_10K':[55,56,57], 'COSTCO_2021_10K':[38],
         'ADOBE_2022_10K':[41,42,54,65], 'AMAZON_2017_10K':[25,69]}
EXPECTED = {
    'Inventories':('2,388','3,055'),
    'Prepaid expenses and other':('8,358','6,906'),
    'Total current assets':('37,857','36,728'),
    'Total current liabilities':('50,171','47,160'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.project_root.resolve();out=args.output.resolve()
    if out.exists():raise SystemExit('Use a new output directory; previous evidence is immutable.')
    out.mkdir(parents=True)
    # Offline guarantees: local tokenizer only; fail any accidental network connect.
    import socket
    def blocked(*args,**kwargs):raise AssertionError('Network is disabled for parsing regression')
    socket.socket.connect=blocked
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root/'models/bge-small-en-v1.5',local_files_only=True)
    rows=[];checks=[];hashes={};started=perf_counter()
    for name,pages in PAGES.items():
        source=root/'data/raw/financebench50'/f'{name}.pdf'
        digest=hashlib.sha256(source.read_bytes()).hexdigest();hashes[name]=digest
        doc=Document(document_id=digest,filename=source.name,format='pdf')
        _pdf(source,doc,page_numbers=pages)
        doc=Document.model_validate(doc.model_dump())
        chunks=chunk_document(doc,tokenizer,300)
        (out/f'{name}.structured.json').write_text(doc.model_dump_json(indent=2))
        (out/f'{name}.chunks.json').write_text(json.dumps(chunks,ensure_ascii=False,indent=2))
        for page in pages:
            pc=[c for c in chunks if c['page']==page]
            tables=[t for t in doc.tables if t.source.page==page]
            assert pc and all(c['token_count']<=300 and c['source']['page']==page for c in pc)
            rows.append(dict(document=name,page=page,tables=len(tables),chunks=len(pc),
                             max_tokens=max(c['token_count'] for c in pc)))
        if name=='VERIZON_2022_10K':
            for label,(a,b) in EXPECTED.items():
                matches=[c for c in chunks if c['page']==56 and label in c['text']
                         and f'C2 [2022]: {a}' in c['text'] and f'C3 [2021]: {b}' in c['text']
                         and 'dollars in millions' in c['text'] and c.get('header_cell_refs')]
                checks.append(dict(metric=label,year_2022=a,year_2021=b,passed=bool(matches),
                                   text=matches[0]['text'] if matches else None))
    result=dict(pages=rows,verizon_checks=checks,source_sha256=hashes,
                generation_api_calls=0,parser_version='native-v2',elapsed_seconds=perf_counter()-started,
                limitations='Four source-reviewed Verizon rows; other nine pages test structure/provenance/token bounds only. Not answer accuracy or retrieval coverage. No cross-page merging.')
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    body='<h1>资料解析回归 · 10页</h1><p>生成式API调用：0。4份公开财报，固定10个物理页。其他9页仅检查结构、来源及300-token上限，不宣称语义正确率。</p>'
    body+='<h2>Verizon 第56页：年份、数值与单位</h2><p>原页面为无框表。旧逐行排序受微小纵向偏移影响，47,160可能排在50,171之前；新片段明确绑定两年，并保留单位。</p>'
    for c in checks:body+=f'<h3>{html.escape(c["metric"])} · {"通过" if c["passed"] else "失败"}</h3><pre>{html.escape(c["text"] or "未匹配")}</pre>'
    body+='<h2>页面检查</h2><table><tr><th>文档</th><th>物理页</th><th>表格</th><th>片段</th><th>最大tokens</th></tr>'
    for x in rows:body+='<tr>'+''.join(f'<td>{html.escape(str(x[k]))}</td>' for k in ['document','page','tables','chunks','max_tokens'])+'</tr>'
    body+='</table><p>尚未接入OCR、跨页自动续表及复杂表头语义推断。自动测试另覆盖PDF/Word/Excel的年份、负数、小数、零、百分比、合并单元格与公式缓存。</p>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>解析回归</title><style>body{max-width:1050px;margin:40px auto;font:16px/1.7 system-ui;color:#18332b}pre{white-space:pre-wrap;background:#f1f6f3;padding:20px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}</style>'+body)
    print(json.dumps(dict(pages=len(rows),checks_passed=sum(c['passed'] for c in checks),checks_total=len(checks),api_calls=0,output=str(out)),ensure_ascii=False))
    if not all(c['passed'] for c in checks):raise SystemExit(1)


if __name__=='__main__':main()
