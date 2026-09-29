"""本地 HTML 多 PDF 问答服务。运行 uv run app.py，打开 127.0.0.1:8000。"""
import hashlib
import os
import secrets
import shutil
import numpy as np
import tempfile
import threading
import json
import queue
from time import perf_counter
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, session
from financial_rag.ingestion.parser import parse_pdf
from financial_rag.ingestion.normalizer import normalize_pages
from financial_rag.ingestion.chunker import chunk_pages
from financial_rag.indexing.vector_store import VectorStore
from financial_rag.pipeline import FinancialRAG

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
app = Flask(__name__)
app.config.update(SECRET_KEY=secrets.token_hex(32), MAX_CONTENT_LENGTH=101*1024*1024, SESSION_COOKIE_SAMESITE="Strict")
MODEL = "BAAI/bge-small-en-v1.5"
lock = threading.RLock()
workspaces = {}


@lru_cache(maxsize=1)
def models():
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    embedding_path = ROOT / "models/bge-small-en-v1.5"
    reranker_path = ROOT / "models/bge-reranker-v2-m3"
    if not all((path / "config.json").is_file() for path in (embedding_path, reranker_path)):
        raise ValueError("项目缺少本地模型，请先运行 uv run prepare_models.py；没有缓存时加 --download")
    return (
        EmbeddingModel(MODEL, model_path=str(embedding_path), local_files_only=True),
        Reranker("BAAI/bge-reranker-v2-m3", 2048,
                 model_path=str(reranker_path), local_files_only=True),
    )


def build_workspace(files, previous=None):
    """在临时工作区追加新文档，成功后才切换；复用旧文档向量。"""
    folder = tempfile.TemporaryDirectory(prefix="rag_web_")
    try:
        chunks = list(previous['rag'].retriever.chunks) if previous else []
        documents = list(previous['documents']) if previous else []
        names = dict(previous['names']) if previous else {}
        seen = set(names)
        old_count = len(chunks)
        total = 0
        library_size = sum(d['size_bytes'] for d in documents)
        if previous:
            for document_id in names:
                shutil.copyfile(Path(previous['folder'].name)/f"{document_id}.pdf",
                                Path(folder.name)/f"{document_id}.pdf")
        for file in files:
            name = Path(file.filename or "").name
            if not name.lower().endswith(".pdf"):
                raise ValueError("只支持 PDF 文件")
            data = file.read(20*1024*1024+1)
            total += len(data)
            if not data or len(data)>20*1024*1024 or total>100*1024*1024:
                raise ValueError("每份 PDF 须为 1 字节至 20 MB，合计不超过 100 MB")
            document_id = hashlib.sha256(data).hexdigest()
            if document_id in seen:
                continue
            if len(documents) >= 10 or library_size + len(data) > 100*1024*1024:
                raise ValueError("追加后资料库最多 10 份 PDF，合计不超过 100 MB；请先清空或减少新增文件")
            library_size += len(data)
            seen.add(document_id)
            path = Path(folder.name)/f"{document_id}.pdf"
            path.write_bytes(data)
            try:
                pages = parse_pdf(path)
            except Exception:
                raise ValueError(f"无法读取 {name}，请检查是否损坏或加密") from None
            new_chunks = chunk_pages(normalize_pages(pages), document_id, 500, 80)
            if not new_chunks:
                raise ValueError(f"{name} 没有可提取文字，扫描件暂不支持")
            # 将可读文件名交给生成模型，支持跨文件区分来源。
            for chunk in new_chunks:
                chunk["title"] = name
            chunks.extend(new_chunks)
            names[document_id] = name
            documents.append(dict(name=name, size_bytes=len(data), pages=len(pages), empty_pages=sum(not page['text'].strip() for page in pages)))
        if previous and len(chunks) == old_count:
            folder.cleanup()
            return previous
        embedding, reranker = models()
        store = VectorStore(Path(folder.name)/"vectors.sqlite3")
        new_vectors = embedding.encode_documents([c['text'] for c in chunks[old_count:]])
        vectors = np.concatenate((previous['rag'].retriever.vectors, new_vectors), axis=0) if previous else new_vectors
        store.replace_all(chunks, vectors, MODEL)
        from financial_rag.generation.generator import DeepSeekGenerator
        rag = FinancialRAG(store, embedding, MODEL, reranker, DeepSeekGenerator())
        return dict(folder=folder, rag=rag, documents=documents, names=names)
    except Exception:
        folder.cleanup()
        raise


@app.get("/")
def index():
    session.setdefault("id", secrets.token_urlsafe(24))
    return render_template("index.html")


@app.get("/api/status")
def status():
    with lock:
        state = workspaces.get(session.get("id"))
        return jsonify(documents=state['documents'] if state else [], key_ready=bool(os.getenv('DEEPSEEK_API_KEY')))


@app.post("/api/upload")
def upload():
    if not os.getenv("DEEPSEEK_API_KEY"):
        return jsonify(error="请先在 .env 配置 DEEPSEEK_API_KEY"), 400
    files = request.files.getlist("files")
    if not 1 <= len(files) <= 10:
        return jsonify(error="请选择 1–10 份 PDF"), 400
    session.setdefault("id", secrets.token_urlsafe(24))
    try:
        with lock:
            old = workspaces.get(session['id'])
            new = build_workspace(files, old)
            workspaces[session['id']] = new
            if old and old is not new:
                old['folder'].cleanup()
        return jsonify(documents=new['documents'])
    except ValueError as error:
        return jsonify(error=str(error)), 400
    except Exception:
        return jsonify(error="处理失败，请检查模型是否可用后重试；原文档仍保留"), 500


def evidence_history(result, names):
    """把每轮新增 ID 映射到正文；展示排名为当轮最终检索结果顺序。"""
    used = {source['chunk_id'] for source in result['sources']}
    history = []
    for record in result['search_history']:
        ranks = {key: rank for rank, key in enumerate(record.get('retrieved_chunk_ids', []), 1)}
        evidence = []
        for key in record.get('new_chunk_ids', []):
            item = result['evidence'][key]
            evidence.append({
                'chunk_id': key, 'citation_id': item['citation_id'],
                'filename': names[item['document_id']], 'page': item['page'],
                'text': item['text'], 'rank': ranks[key], 'used_in_answer': key in used,
            })
        history.append({**record, 'evidence': evidence})
    return history


@app.post("/api/ask")
def ask():
    started = perf_counter()
    body = request.get_json(silent=True) or {}
    question = body.get('question') if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip() or len(question)>4000:
        return jsonify(error="请输入不超过 4000 字的问题"), 400
    with lock:
        state = workspaces.get(session.get('id'))
        if not state:
            return jsonify(error="请先上传并阅读 PDF"), 400
        try:
            result = state['rag'].ask_agent(question)
            if result['status'] == 'error':
                return jsonify(error='模型调用失败，请检查网络或配置后重试', elapsed_seconds=round(perf_counter()-started, 3)), 502
            evidence = result['evidence']
            for source in result['sources']:
                source['filename'] = state['names'][source['document_id']]
                source['text'] = evidence[source['chunk_id']]['text']
            return jsonify(answer=result['answer'], sources=result['sources'], citation_status=result['citation_status'],
                           status=result['status'], stop_reason=result['stop_reason'],
                           query_plan=result['query_plan'], search_history=evidence_history(result, state['names']),
                           planning_calls=result.get('planning_calls'), model_steps=result.get('model_steps'),
                           model_calls=result['model_calls'], elapsed_seconds=round(perf_counter()-started, 3))
        except Exception:
            return jsonify(error="回答失败，请检查网络、DeepSeek 配置或账户额度后重试"), 502


@app.post("/api/ask/stream")
def ask_stream():
    body = request.get_json(silent=True) or {}
    question = body.get('question') if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        return jsonify(error="请输入不超过 4000 字的问题"), 400
    session_id = session.get('id')
    if session_id not in workspaces:
        return jsonify(error="请先上传并阅读 PDF"), 400
    events = queue.Queue(maxsize=128)
    cancelled = threading.Event()

    def emit(event):
        while not cancelled.is_set():
            try:
                events.put(event, timeout=0.2)
                return
            except queue.Full:
                continue
        raise RuntimeError("客户端已断开")

    def worker():
        acquired = False
        try:
            while not cancelled.is_set():
                acquired = lock.acquire(timeout=0.2)
                if acquired:
                    break
            if not acquired or cancelled.is_set():
                return
            state = workspaces.get(session_id)
            if not state:
                emit({"type": "error", "message": "资料库已清空，请重新上传"})
                return
            result = state['rag'].ask_agent(question, on_event=emit)
            if result['status'] == 'error':
                emit({"type": "error", "message": "生成未完成，请检查网络或模型配置后重试"})
                return
            for source in result['sources']:
                source['filename'] = state['names'][source['document_id']]
                source['text'] = result['evidence'][source['chunk_id']]['text']
            # 不发送完整候选和推理内容；UI 只需答案、来源与检索历史。
            fields = ('answer', 'sources', 'citation_status', 'status', 'stop_reason',
                      'query_plan', 'search_history', 'model_calls', 'search_attempts', 'elapsed_seconds')
            payload = {key: result[key] for key in fields}
            payload['search_history'] = evidence_history(result, state['names'])
            payload['planning_calls'] = result.get('planning_calls')
            payload['model_steps'] = result.get('model_steps')
            emit({"type": "done", "result": payload})
        except Exception:
            if not cancelled.is_set():
                emit({"type": "error", "message": "回答中断，请重试；已显示文字不是完整回答"})
        finally:
            if acquired:
                lock.release()

    def stream():
        try:
            yield json.dumps({"type": "progress", "message": "请求已收到，正在等待处理"}, ensure_ascii=False) + "\n"
            threading.Thread(target=worker, daemon=True).start()
            while not cancelled.is_set():
                try:
                    event = events.get(timeout=10)
                except queue.Empty:
                    yield '{"type":"heartbeat"}\n'
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event['type'] in ('done', 'error'):
                    break
        finally:
            cancelled.set()
    return Response(stream(), mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.delete("/api/documents")
def clear():
    with lock:
        old = workspaces.pop(session.get('id'), None)
        if old:
            old['folder'].cleanup()
    return jsonify(ok=True)


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="上传总大小不能超过 100 MB"), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
