"""创建请求绑定的离线检索评测包；不保存密钥、模型权重或模型推理内容。"""
import csv
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from financial_rag.ingestion.parser import parse_pdf
from financial_rag.ingestion.normalizer import normalize_pages


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item()) + "\n", encoding="utf-8")


def snapshot_run(state, result, root):
    """在资料库被替换前固化本次运行；仅在服务器临时目录缓存。"""
    folder = tempfile.TemporaryDirectory(prefix="rag_run_")
    target = Path(folder.name)
    try:
        retriever = state['rag'].retriever
        (target / 'documents').mkdir()
        documents = []
        for doc_id, name in state['names'].items():
            source = Path(state['folder'].name) / f'{doc_id}.pdf'
            destination = target / 'documents' / f'{doc_id}.pdf'
            shutil.copyfile(source, destination)
            documents.append({'document_id': doc_id, 'filename': name,
                              'file': f'documents/{doc_id}.pdf',
                              'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
        shutil.copyfile(Path(state['folder'].name) / 'vectors.sqlite3', target / 'vectors.sqlite3')
        write_json(target / 'chunks.json', retriever.chunks)
        write_json(target / 'run.json', result)
        write_json(target / 'documents.json', documents)
        (target / 'answer.md').write_text(result.get('answer', ''), encoding='utf-8')
        # 只收集明确的配置字段，绝不序列化模型客户端或环境变量。
        from inspect import signature
        from financial_rag.agent.loop import run_agent
        budgets = {k: signature(run_agent).parameters[k].default for k in
                   ('max_steps','max_searches','max_evidence_chars','max_message_chars')}
        revisions = {}
        for model in ('bge-small-en-v1.5','bge-reranker-v2-m3'):
            path = root / 'models' / model / 'local-model.json'
            if path.is_file(): revisions[model] = json.loads(path.read_text())
        config = {'embedding_model': 'BAAI/bge-small-en-v1.5',
                  'reranker_model': getattr(retriever.reranker, 'model_name', None),
                  'reranker_max_length': getattr(retriever.reranker, 'max_length', None),
                  'generation_model': getattr(state['rag'].generator, 'model', None),
                  'candidate_k': retriever.candidate_k, 'top_k': retriever.top_k, 'rrf_k': retriever.rrf_k,
                  'chunking': {'max_words':500,'overlap_words':80}, 'agent_budgets':budgets,
                  'use_query_planning':True, 'model_revisions':revisions}
        write_json(target / 'config.json', config)
        versions = {}
        for name in ('numpy','sentence-transformers','transformers','pymupdf','openai','flask'):
            try: versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: pass
        write_json(target / 'versions.json', versions)
        # 固化相关代码，便于后续修改后仍能解释当时的检索结果。
        source_dir = target / 'source'
        shutil.copytree(root / 'src/financial_rag', source_dir / 'financial_rag',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('app.py','pyproject.toml','uv.lock'):
            if (root / name).is_file(): shutil.copyfile(root / name, source_dir / name)
        return folder
    except Exception:
        folder.cleanup()
        raise


def save_bundle(snapshot, output_root, run_id, browser_elapsed_seconds=None):
    """只接受服务器生成的目录名；临时生成完整目录后原子发布。"""
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / run_id
    if final.is_dir(): return final
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output_root) as staging:
        destination = Path(staging) / 'bundle'
        shutil.copytree(snapshot, destination)
        result = json.loads((destination / 'run.json').read_text())
        documents = json.loads((destination / 'documents.json').read_text())
        (destination / 'parsed').mkdir()
        (destination / 'normalized').mkdir()
        for document in documents:
            pages = parse_pdf(destination / document['file'])
            write_json(destination / 'parsed' / (document['document_id']+'.json'), pages)
            write_json(destination / 'normalized' / (document['document_id']+'.json'), normalize_pages(pages))
        # 阶段排名一行一条；不同阶段的分数不能横向比较。
        with (destination / 'retrieval_stages.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round','query','missing_information','stage','rank','chunk_id','page','score','rrf_score','rerank_score','text'])
            writer.writeheader()
            for number, trace in enumerate(result.get('retrieval_rounds',[]),1):
                for stage in ('dense','bm25','hybrid','reranked'):
                    for rank,item in enumerate(trace.get(stage,[]),1):
                        row={k:item.get(k,'') for k in ('chunk_id','page','score','rrf_score','rerank_score','text')}
                        writer.writerow(dict(row,round=number,query=trace['query'],missing_information=trace.get('missing_information'),stage=stage,rank=rank))
        write_json(destination / 'ground_truth_template.json', {
            'question':result.get('question'), 'expected_evidence':[],
            'instructions':'人工填写 document_id、page、原文片段和必要信息点；不是已验证 ground truth。'})
        write_json(destination / 'timing.json', {'agent_elapsed_seconds':result.get('elapsed_seconds'),
                   'browser_elapsed_seconds':browser_elapsed_seconds})
        (destination / 'README.md').write_text("""# 检索评测资料包

- run.json：原问题、查询规划、补查理由、每阶段候选、工具结果、累计证据、引用及调用次数。
- retrieval_stages.csv：各轮 Dense/BM25/RRF/重排结果及排名、原始分数。
- documents/、documents.json：本次资料库全部原始 PDF、原文件名与哈希映射。
- parsed/、normalized/：点击保存时从本次固定 PDF 用当前解析函数重建的页面数据。
- chunks.json、vectors.sqlite3：问答时实际使用的分块与向量索引快照。
- config.json、versions.json、source/：运行配置、依赖版本、问答时源码快照。
- answer.md、timing.json：回答与计时；浏览器耗时由客户端提供，仅作诊断参考。
- ground_truth_template.json：人工证据标注模板，不自动宣称召回率或答案正确。

模型权重、API 密钥和 reasoning_content 不包含在包内。模型输入中的完整证据由
run.json 的初始及后续工具结果保留；不保存私有推理或原始模型请求。
Embedding 截断未记录，Reranker 超限候选 ID 在每轮 trace 中；片段全文不表示模型读过全部。
没有记录排序截断前全库分数。检索失败详情见工具结果；未成功执行的请求不一定有阶段排名。
""", encoding='utf-8')
        files = {str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in destination.rglob('*') if p.is_file()}
        write_json(destination / 'manifest.json', {'format_version':1,'run_id':run_id,
                   'saved_at':datetime.now(timezone.utc).isoformat(),'sha256':files})
        destination.rename(final)
    return final
