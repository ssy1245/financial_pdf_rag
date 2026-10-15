"""本地 HTML 多格式文档问答服务。运行 uv run app.py，打开 127.0.0.1:8000。"""
import hashlib
import os
import secrets
import shutil
import numpy as np
import tempfile
import threading
import json
import queue
import asyncio
from time import perf_counter
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict
from datetime import datetime, timezone
from financial_rag.evaluation.export import snapshot_run, save_bundle

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from financial_rag.ingestion.adapters import parse_document
from financial_rag.ingestion.structured_chunker import chunk_document
from financial_rag.ingestion.document import provenance
from financial_rag.indexing.vector_store import VectorStore
from financial_rag.config import RETRIEVAL_DEFAULTS
from financial_rag.pipeline import FinancialRAG
from financial_rag.memory import ConversationMemory
from financial_rag.library import LibraryStore, LibraryError, session_secret

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
app = FastAPI(title="Financial PDF RAG", version="0.1.0")
LIBRARY_ROOT = Path(os.getenv("RAG_LIBRARY_ROOT", str(ROOT / "storage/libraries")))


class SavedSessionMiddleware(SessionMiddleware):
    def __init__(self, app):
        super().__init__(app, secret_key=session_secret(LIBRARY_ROOT), same_site="strict",
                         max_age=365 * 24 * 60 * 60)


app.add_middleware(SavedSessionMiddleware)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


MAX_UPLOAD_REQUEST_BYTES = 101 * 1024 * 1024  # 含 multipart 封装


class UploadSizeLimitMiddleware:
    """同时限制声明长度和实际接收长度，包括没有 Content-Length 的上传。"""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/upload":
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD_REQUEST_BYTES
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"error": "上传长度无效"}, status_code=400)(scope, receive, send)
        if declared > limit:
            return await JSONResponse({"error": "上传总大小不能超过 100 MB"}, status_code=413)(scope, receive, send)
        received = 0
        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(status_code=413, detail="上传总大小不能超过 100 MB")
            return message
        await self.app(scope, limited_receive, send)


app.add_middleware(UploadSizeLimitMiddleware)


class QuestionRequest(BaseModel):
    question: str = Field(strict=True, min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value


class SaveRunRequest(BaseModel):
    browser_elapsed_seconds: float | None = Field(default=None, strict=True, ge=0, allow_inf_nan=False)


def error_response(message, status_code=400, **extra):
    return JSONResponse({"error": message, **extra}, status_code=status_code)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return error_response("请求参数无效；问题须为非空字符串且不超过 4000 字，耗时须为非负有限数值")


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    return error_response(str(exc.detail), exc.status_code)


@app.exception_handler(LibraryError)
async def library_error(request, exc):
    return error_response(str(exc), 503)


MODEL = "BAAI/bge-small-en-v1.5"
lock = threading.RLock()
workspaces = {}
run_snapshots = {}
EXPORT_ROOT = ROOT / "outputs/retriever_evaluations"


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


@lru_cache(maxsize=1)
def document_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(ROOT / "models/bge-small-en-v1.5", local_files_only=True)


def get_workspace(session_id, *, hydrate=False):
    """Restore metadata without model loading; hydrate only for retrieval or mutation."""
    state = workspaces.get(session_id)
    if state is None:
        state = LibraryStore(LIBRARY_ROOT).read(session_id)
        if state:
            state.update(rag=None, conversation=ConversationMemory(secrets.token_hex(16)))
            workspaces[session_id] = state
    if state and hydrate and state['rag'] is None:
        try:
            embedding, reranker = models()
            store = VectorStore(Path(state['folder'].name) / 'vectors.sqlite3')
            rag = make_rag(store, embedding, reranker)
            if set(c['document_id'] for c in rag.retriever.chunks) != set(state['names']):
                raise ValueError('索引与文档清单不一致')
            state['rag'] = rag
        except Exception as exc:
            raise LibraryError('资料库索引加载失败；原文件保留，请检查本地模型和索引') from exc
    return state


def make_rag(store, embedding, reranker):
    from financial_rag.generation.generator import DeepSeekGenerator
    generator = DeepSeekGenerator() if os.getenv('DEEPSEEK_API_KEY') else None
    return FinancialRAG(store, embedding, MODEL, reranker, generator, **RETRIEVAL_DEFAULTS)


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
            for old_path in Path(previous['folder'].name).iterdir():
                if old_path.suffix in {'.pdf', '.docx', '.xlsx', '.json'}:
                    shutil.copyfile(old_path, Path(folder.name)/old_path.name)
        parsed = []
        for file in files:
            name = Path(file.filename or "").name
            if Path(name).suffix.lower() not in {".pdf", ".docx", ".xlsx"}:
                raise ValueError("只支持 PDF、Word (.docx)、Excel (.xlsx) 文件")
            data = file.file.read(20*1024*1024+1)
            total += len(data)
            if not data or len(data)>20*1024*1024 or total>100*1024*1024:
                raise ValueError("每份文档须为 1 字节至 20 MB，合计不超过 100 MB")
            document_id = hashlib.sha256(data).hexdigest()
            if document_id in seen:
                continue
            if len(documents) >= 10 or library_size + len(data) > 100*1024*1024:
                raise ValueError("追加后资料库最多 10 份文档，合计不超过 100 MB；请先清空或减少新增文件")
            library_size += len(data)
            seen.add(document_id)
            path = Path(folder.name)/f"{document_id}{Path(name).suffix.lower()}"
            path.write_bytes(data)
            try:
                document = parse_document(path, name)
            except Exception as exc:
                raise ValueError(f"无法读取 {name}，请检查是否损坏、加密或超出解析限制：{exc}") from None
            if not document.elements:
                reason = "需 OCR，尚未接入识别模型" if document.ocr_requests else "没有可检索的正文或表格"
                raise ValueError(f"{name}：{reason}")
            (Path(folder.name)/f"{document_id}.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")
            parsed.append(document)
            names[document_id] = name
            documents.append(dict(document_id=document_id,name=name,format=document.format,size_bytes=len(data),
                pages=document.page_count,sheets=document.sheets,tables=len(document.tables),
                empty_pages=len({r.page for r in document.ocr_requests}),
                ocr_status="needs_ocr" if document.ocr_requests else "not_requested",warnings=document.warnings))
        if previous and not parsed:
            folder.cleanup()
            return previous
        tokenizer = document_tokenizer()
        for document in parsed:
            new_chunks = chunk_document(document, tokenizer, 300)
            if not new_chunks:
                raise ValueError(f"{document.filename} 没有可索引的有效正文或表格")
            chunks.extend(new_chunks)
        if len(chunks) == old_count:
            raise ValueError("没有可索引的文字；OCR 尚未接入")
        embedding, reranker = models()
        store = VectorStore(Path(folder.name)/"vectors.sqlite3")
        new_vectors = embedding.encode_documents([c['text'] for c in chunks[old_count:]])
        vectors = np.concatenate((previous['rag'].retriever.vectors, new_vectors), axis=0) if previous else new_vectors
        store.replace_all(chunks, vectors, MODEL)
        rag = make_rag(store, embedding, reranker)
        return dict(folder=folder, rag=rag, documents=documents, names=names,
                    conversation=ConversationMemory(secrets.token_hex(16)))
    except Exception:
        folder.cleanup()
        raise


@app.get("/")
def index(request: Request):
    request.session.setdefault("id", secrets.token_urlsafe(24))
    return FileResponse(ROOT / "templates/index.html")


@app.get("/api/status")
def status(request: Request):
    with lock:
        state = get_workspace(request.session.get("id"))
        return dict(documents=state['documents'] if state else [], key_ready=bool(os.getenv('DEEPSEEK_API_KEY')))


@app.get("/api/documents/{document_id}/structure")
def document_structure(document_id: str, request: Request):
    with lock:
        state = get_workspace(request.session.get("id"))
        if not state or document_id not in state['names']:
            raise HTTPException(404, "文档不存在")
        return json.loads((Path(state['folder'].name)/f"{document_id}.json").read_text())


@app.get("/api/documents/{document_id}/original")
def document_original(document_id: str, request: Request):
    from fastapi.responses import Response
    from urllib.parse import quote
    with lock:
        state = get_workspace(request.session.get("id"))
        if not state or document_id not in state['names']:
            raise HTTPException(404, "文档不存在")
        name = state['names'][document_id]
        path = Path(state['folder'].name)/f"{document_id}{Path(name).suffix.lower()}"
        return Response(path.read_bytes(),media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename*=UTF-8''"+quote(name, safe='')})


@app.post("/api/upload")
async def upload(request: Request):
    request.session.setdefault("id", secrets.token_urlsafe(24))
    async with request.form(max_files=10, max_fields=0) as form:
        files = form.getlist("files")
        if not 1 <= len(files) <= 10 or any(not hasattr(file, "file") for file in files):
            return error_response("请选择 1–10 份 PDF、Word 或 Excel")
        return await run_in_threadpool(process_upload, files, request.session['id'])


def process_upload(files, session_id):
    try:
        with lock:
            old = get_workspace(session_id, hydrate=True)
            new = build_workspace(files, old)
            if new is not old:
                try:
                    LibraryStore(LIBRARY_ROOT).commit(session_id, new)
                except Exception:
                    new['folder'].cleanup()
                    raise
            workspaces[session_id] = new
            if old and old is not new:
                old['folder'].cleanup()
        return dict(documents=new['documents'])
    except ValueError as error:
        return error_response(str(error))
    except Exception:
        return error_response("处理失败，请检查模型是否可用后重试；原文档仍保留", 500)


def conversation_options(state):
    context = state['conversation'].context()
    return {'conversation_context': context} if context else {}


def remember_run(session_id, state, result):
    """每会话缓存最近五次快照，绑定当时资料库，不受追加/清空影响。"""
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + secrets.token_hex(8)
    snapshot = snapshot_run(state, result, ROOT)
    records = run_snapshots.setdefault(session_id, OrderedDict())
    records[run_id] = snapshot
    while len(records) > 5:
        _, old = records.popitem(last=False)
        old.cleanup()
    return run_id


@app.post('/api/runs/{run_id}/save')
def save_run(run_id: str, request: Request, body: SaveRunRequest):
    elapsed = body.browser_elapsed_seconds
    with lock:
        snapshot = run_snapshots.get(request.session.get('id'), {}).get(run_id)
        if snapshot is None:
            return error_response('该次记录已过期或服务已重启，请重新提问后保存', 404)
        try:
            path = save_bundle(Path(snapshot.name), EXPORT_ROOT, run_id, elapsed)
            return dict(path=str(path), run_id=run_id)
        except Exception:
            return error_response('保存失败，请检查磁盘空间和目录权限后重试', 500)


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
                'filename': names[item['document_id']], 'page': item['page'], **provenance(item),
                'text': item['text'], 'rank': ranks[key], 'used_in_answer': key in used,
            })
        history.append({**record, 'evidence': evidence})
    return history


@app.post("/api/ask")
def ask(body: QuestionRequest, request: Request):
    started = perf_counter()
    question = body.question
    if not os.getenv("DEEPSEEK_API_KEY"):
        return error_response("请先在 .env 配置 DEEPSEEK_API_KEY，再提问")
    with lock:
        state = get_workspace(request.session.get('id'), hydrate=True)
        if not state:
            return error_response("请先上传并阅读文档")
        try:
            result = state['rag'].ask_agent(question, **conversation_options(state))
            if result['status'] == 'error':
                return error_response('模型调用失败，请检查网络或配置后重试', 502, elapsed_seconds=round(perf_counter()-started, 3))
            try:
                run_id = remember_run(request.session.get('id'), state, result)
                export_warning = None
            except Exception:
                run_id, export_warning = None, '评测快照准备失败，本次无法保存；回答仍可查看'
            evidence = result['evidence']
            for source in result['sources']:
                source['filename'] = state['names'][source['document_id']]
                source['text'] = evidence[source['chunk_id']]['text']
            state['conversation'].record(question, result)
            return dict(resolved_question=result.get('resolved_question', question), run_id=run_id, export_warning=export_warning, answer=result['answer'], sources=result['sources'], citation_status=result['citation_status'],
                           status=result['status'], stop_reason=result['stop_reason'],
                           query_plan=result['query_plan'], search_history=evidence_history(result, state['names']),
                           planning_calls=result.get('planning_calls'), model_steps=result.get('model_steps'),
                           model_calls=result['model_calls'], elapsed_seconds=round(perf_counter()-started, 3))
        except Exception:
            return error_response("回答失败，请检查网络、DeepSeek 配置或账户额度后重试", 502)


@app.post("/api/ask/stream")
def ask_stream(body: QuestionRequest, request: Request):
    question = body.question
    session_id = request.session.get('id')
    if not os.getenv('DEEPSEEK_API_KEY'):
        return error_response('请先在 .env 配置 DEEPSEEK_API_KEY，再提问')
    with lock:
        if not get_workspace(session_id):
            return error_response('请先上传并阅读文档')
    events = queue.Queue(maxsize=128)
    cancelled = threading.Event()
    delivered = threading.Event()

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
            state = get_workspace(session_id, hydrate=True)
            if not state:
                emit({"type": "error", "message": "资料库已清空，请重新上传"})
                return
            result = state['rag'].ask_agent(question, on_event=emit, **conversation_options(state))
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
            payload['resolved_question'] = result.get('resolved_question', question)
            payload['search_history'] = evidence_history(result, state['names'])
            payload['planning_calls'] = result.get('planning_calls')
            payload['model_steps'] = result.get('model_steps')
            emit({"type": "progress", "message": "回答已生成，正在准备可保存的检索记录"})
            try:
                payload['run_id'] = remember_run(session_id, state, result)
                payload['export_warning'] = None
            except Exception:
                payload['run_id'] = None
                payload['export_warning'] = '评测快照准备失败，本次无法保存；回答仍可查看'
            emit({"type": "done", "result": payload})
            while not delivered.wait(timeout=0.2):
                if cancelled.is_set():
                    return
            state['conversation'].record(question, result)
        except Exception:
            if not cancelled.is_set():
                emit({"type": "error", "message": "回答中断，请重试；已显示文字不是完整回答"})
        finally:
            if acquired:
                lock.release()

    async def stream():
        try:
            yield json.dumps({"type": "progress", "message": "请求已收到，正在等待处理"}, ensure_ascii=False) + "\n"
            threading.Thread(target=worker, daemon=True).start()
            while not cancelled.is_set():
                try:
                    event = await asyncio.to_thread(events.get, timeout=1)
                except queue.Empty:
                    yield '{"type":"heartbeat"}\n'
                    continue
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event['type'] in ('done', 'error'):
                    if event['type'] == 'done':
                        delivered.set()
                    break
        finally:
            cancelled.set()
    return StreamingResponse(stream(), media_type='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str, request: Request):
    session_id = request.session.get('id')
    with lock:
        old = get_workspace(session_id)
        if not old or document_id not in old['names']:
            return error_response('文档不存在', 404)
        if len(old['documents']) == 1:
            LibraryStore(LIBRARY_ROOT).clear(session_id)
            workspaces.pop(session_id, None)
            old['folder'].cleanup()
            return dict(documents=[])
        folder = tempfile.TemporaryDirectory(prefix='rag_delete_')
        try:
            # Read saved vectors directly: deletion never re-encodes remaining documents.
            chunks, vectors = VectorStore(Path(old['folder'].name)/'vectors.sqlite3').load(MODEL)
            keep = [i for i,c in enumerate(chunks) if c['document_id'] != document_id]
            documents = [d for d in old['documents'] if d['document_id'] != document_id]
            names = {d['document_id']:d['name'] for d in documents}
            for key,name in names.items():
                for suffix in (Path(name).suffix.lower(), '.json'):
                    shutil.copyfile(Path(old['folder'].name)/(key+suffix), Path(folder.name)/(key+suffix))
            VectorStore(Path(folder.name)/'vectors.sqlite3').replace_all(
                [chunks[i] for i in keep], vectors[keep], MODEL)
            new = dict(folder=folder, rag=None, documents=documents, names=names,
                       conversation=ConversationMemory(secrets.token_hex(16)))
            LibraryStore(LIBRARY_ROOT).commit(session_id, new)
            workspaces[session_id] = new
            old['folder'].cleanup()
            return dict(documents=documents)
        except Exception:
            folder.cleanup()
            return error_response('删除失败，原资料库仍保留，请检查磁盘后重试', 500)


@app.delete("/api/documents")
def clear(request: Request):
    with lock:
        session_id = request.session.get('id')
        LibraryStore(LIBRARY_ROOT).clear(session_id)
        old = workspaces.pop(session_id, None)
        if old:
            old['folder'].cleanup()
    return dict(ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
