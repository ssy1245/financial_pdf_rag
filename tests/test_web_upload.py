"""追加上传的离线集成测试：真实 PDF 解析和 SQLite，模型用替身。"""
import importlib.util
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
import base64
import json

spec = importlib.util.spec_from_file_location('web_app_tests', Path(__file__).resolve().parents[1] / 'app.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


def pdf(text):
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), text)
        return doc.tobytes()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "LIBRARY_ROOT", tmp_path / "libraries")
    monkeypatch.setattr(web.app, "middleware_stack", None)
    class Embedding:
        def __init__(self):
            self.batches = []
        def encode_documents(self, texts):
            self.batches.append(texts)
            return np.tile([1., 0.], (len(texts), 1))
    embedding = Embedding()
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-test')
    monkeypatch.setattr(web, 'models', lambda: (embedding, object()))
    from test_structured_documents import Tokenizer
    monkeypatch.setattr(web, 'document_tokenizer', lambda: Tokenizer())
    monkeypatch.setattr('financial_rag.generation.generator.DeepSeekGenerator', lambda: object())
    with TestClient(web.app) as client:
        yield client, embedding
    for state in web.workspaces.values():
        state['folder'].cleanup()
    web.workspaces.clear()
    for records in web.run_snapshots.values():
        for snapshot in records.values():
            snapshot.cleanup()
    web.run_snapshots.clear()


def upload(client, data, name='report.pdf'):
    return client.post('/api/upload', files={'files': (name, data, 'application/pdf')})


def test_append_deduplicates_and_encodes_only_new_chunks(client):
    c, embedding = client
    first = pdf('first revenue 100')
    assert upload(c, first, 'first.pdf').status_code == 200
    old = next(iter(web.workspaces.values()))
    old_ids = set(old['rag'].chunks_by_id)
    response = upload(c, pdf('second revenue 200'), 'second.pdf')
    assert [d['name'] for d in response.json()['documents']] == ['first.pdf', 'second.pdf']
    current = next(iter(web.workspaces.values()))
    assert old_ids < set(current['rag'].chunks_by_id)
    assert len(embedding.batches) == 2
    assert all('first' not in text for text in embedding.batches[1])
    assert not Path(old['folder'].name).exists()
    assert len(list(Path(current['folder'].name).glob('*.pdf'))) == 2
    response = upload(c, first, 'renamed.pdf')
    assert len(response.json()['documents']) == 2
    assert next(iter(web.workspaces.values())) is current
    assert len(embedding.batches) == 2
    assert Path(current['folder'].name).exists()
    assert c.delete('/api/documents').status_code == 200
    assert c.get('/api/status').json()['documents'] == []


def test_bad_batch_leaves_original_workspace_intact(client):
    c, embedding = client
    upload(c, pdf('original'))
    old = next(iter(web.workspaces.values()))
    response = c.post('/api/upload', files=[('files', ('valid.pdf', pdf('valid addition'), 'application/pdf')),
               ('files', ('broken.pdf', b'broken', 'application/pdf'))])
    assert response.status_code == 400
    assert next(iter(web.workspaces.values())) is old
    assert len(c.get('/api/status').json()['documents']) == 1
    assert len(embedding.batches) == 1


def test_cumulative_count_limit_and_session_isolation(client):
    c, _ = client
    payloads = [pdf(f'document {i}') for i in range(10)]
    response = c.post('/api/upload', files=[('files', (f'{i}.pdf', data, 'application/pdf')) for i, data in enumerate(payloads)])
    assert len(response.json()['documents']) == 10
    assert upload(c, payloads[0]).status_code == 200  # 重复文件不占名额
    assert upload(c, pdf('eleventh')).status_code == 400
    with TestClient(web.app) as other:
        assert other.get('/api/status').json()['documents'] == []
        assert upload(other, pdf('another user')).status_code == 200
        assert len(other.get('/api/status').json()['documents']) == 1
    assert len(c.get('/api/status').json()['documents']) == 10


def test_cumulative_size_limit(client):
    c, _ = client
    upload(c, pdf('original'))
    old = next(iter(web.workspaces.values()))
    old['documents'][0]['size_bytes'] = 100 * 1024 * 1024
    assert upload(c, pdf('addition')).status_code == 400
    assert next(iter(web.workspaces.values())) is old


def test_web_uses_agent_and_maps_later_round_citations(client, monkeypatch):
    c, _ = client
    upload(c, pdf('original'), 'first.pdf')
    state = next(iter(web.workspaces.values()))
    document_id = next(iter(state['names']))
    def agent(question):
        assert question == 'compare'
        return dict(answer='answer [E6]', sources=[dict(label='E6', document_id=document_id, chunk_id='later', page=2)],
                    citation_status='labels_valid', evidence={'later': {'text':'later round evidence'}},
                    status='limited', stop_reason='search_limit', query_plan={'queries':['compare','revenue'], 'warning':None},
                    search_history=[{'query':'revenue','new_count':1}], model_calls=3)
    monkeypatch.setattr(state['rag'], 'ask_agent', agent)
    response = c.post('/api/ask', json={'question':'compare'})
    assert response.status_code == 200
    assert response.json()['sources'][0]['text'] == 'later round evidence'
    assert response.json()['sources'][0]['filename'] == 'first.pdf'
    assert response.json()['model_calls'] == 3
    assert response.json()['status'] == 'limited'
    assert response.json()['elapsed_seconds'] >= 0


def test_agent_failure_is_not_reported_as_success(client, monkeypatch):
    c, _ = client
    upload(c, pdf('original'))
    state = next(iter(web.workspaces.values()))
    monkeypatch.setattr(state['rag'], 'ask_agent', lambda q: {'status':'error'})
    response = c.post('/api/ask', json={'question':'q'})
    assert response.status_code == 502
    assert 'error' in response.json()


def test_stream_endpoint_sends_progress_text_and_final_sources(client, monkeypatch):
    import json
    c, _ = client
    upload(c, pdf('original'), 'first.pdf')
    state = next(iter(web.workspaces.values()))
    document_id = next(iter(state['names']))
    def agent(question, on_event):
        on_event({'type':'progress','message':'正在检索'})
        on_event({'type':'answer_delta','text':'answer'})
        return dict(answer='answer [E1]', sources=[dict(label='E1',document_id=document_id,chunk_id='x',page=1)],
                    citation_status='labels_valid', evidence={'x':{'text':'source'}}, status='completed', stop_reason=None,
                    query_plan={},search_history=[],model_calls=2,search_attempts=1,elapsed_seconds=1)
    monkeypatch.setattr(state['rag'], 'ask_agent', agent)
    response = c.post('/api/ask/stream', json={'question':'q'})
    events = [json.loads(line) for line in response.content.splitlines()]
    assert [e['type'] for e in events] == ['progress','progress','answer_delta','progress','done']
    assert events[-1]['result']['sources'][0]['text'] == 'source'
    assert 'evidence' not in events[-1]['result']


def test_stream_endpoint_returns_error_event(client, monkeypatch):
    import json
    c, _ = client
    upload(c, pdf('original'))
    state = next(iter(web.workspaces.values()))
    monkeypatch.setattr(state['rag'], 'ask_agent', lambda q, on_event: {'status':'error'})
    response = c.post('/api/ask/stream', json={'question':'q'})
    events = [json.loads(line) for line in response.content.splitlines()]
    assert events[-1]['type'] == 'error'
    assert not any(e['type']=='done' for e in events)


def test_evidence_history_contains_all_new_chunks_and_usage():
    result = {
        'sources': [{'chunk_id':'b'}],
        'search_history': [
            {'query':'first','retrieved_chunk_ids':['a','b'],'new_chunk_ids':['a','b'],'new_count':2},
            {'query':'second','retrieved_chunk_ids':['b','c'],'new_chunk_ids':['c'],'new_count':1},
            {'query':'empty','retrieved_chunk_ids':[],'new_chunk_ids':[],'new_count':0}],
        'evidence': {key:dict(citation_id=f'E{i}',document_id='doc',page=i,text=f'full text {key}')
                     for i,key in enumerate('abc',1)}}
    history = web.evidence_history(result, {'doc':'report.pdf'})
    assert [item['rank'] for item in history[0]['evidence']] == [1,2]
    assert [item['used_in_answer'] for item in history[0]['evidence']] == [False,True]
    assert history[1]['evidence'][0] == dict(chunk_id='c',citation_id='E3',filename='report.pdf',
                                           page=3,text='full text c',rank=2,used_in_answer=False)
    assert history[2]['evidence'] == []
    assert 'evidence' not in result['search_history'][0]


def test_export_preserves_run_after_clear_and_is_session_scoped(client, monkeypatch, tmp_path):
    import json
    import hashlib
    c, _ = client
    raw = pdf('revenue 100')
    upload(c, raw, 'original.pdf')
    state = next(iter(web.workspaces.values()))
    chunk = state['rag'].retriever.chunks[0]
    result = dict(question='revenue?', answer='100', sources=[], evidence={},
                  search_history=[], status='completed', query_plan={}, citation_status='no_citations', stop_reason=None, model_calls=2,
                  retrieval_rounds=[dict(query='revenue', missing_information='amount',
                      dense=[dict(chunk, score=0.9)], bm25=[], hybrid=[], reranked=[])])
    monkeypatch.setattr(state['rag'], 'ask_agent', lambda q: result)
    monkeypatch.setattr(web, 'EXPORT_ROOT', tmp_path / 'exports')
    response = c.post('/api/ask', json={'question':'revenue?'})
    run_id = response.json()['run_id']
    assert run_id
    assert not web.EXPORT_ROOT.exists()
    with TestClient(web.app) as other:
        assert other.post(f'/api/runs/{run_id}/save', json={}).status_code == 404
    c.delete('/api/documents')
    response = c.post(f'/api/runs/{run_id}/save', json={'browser_elapsed_seconds':14.8})
    assert response.status_code == 200
    target = Path(response.json()['path'])
    assert (target / 'run.json').is_file()
    assert json.loads((target / 'run.json').read_text())['question'] == 'revenue?'
    assert next((target / 'documents').glob('*.pdf')).read_bytes() == raw
    assert list((target / 'parsed').glob('*.json'))
    assert list((target / 'normalized').glob('*.json'))
    assert (target / 'vectors.sqlite3').is_file()
    assert 'revenue' in (target / 'retrieval_stages.csv').read_text()
    assert json.loads((target / 'timing.json').read_text())['browser_elapsed_seconds'] == 14.8
    manifest = json.loads((target / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    assert not list(target.rglob('.env'))
    assert c.post(f'/api/runs/{run_id}/save', json={}).json()['path'] == str(target)
    assert len(list(web.EXPORT_ROOT.iterdir())) == 1
    assert c.post(f'/api/runs/{run_id}/save', json={'browser_elapsed_seconds':-1}).status_code == 400


def completed_result(question):
    return dict(question=question, resolved_question='resolved '+question, answer='answer', sources=[],
                citation_status='no_citations', evidence={}, status='completed', stop_reason=None,
                query_plan={}, search_history=[], model_calls=2, search_attempts=1, elapsed_seconds=1,
                retrieval_rounds=[])


def test_conversation_is_passed_to_next_request_and_resets_only_on_content_change(client, monkeypatch):
    c, _ = client
    raw = pdf('report')
    upload(c, raw)
    state = next(iter(web.workspaces.values()))
    received = []
    def agent(question, **kwargs):
        received.append(kwargs.get('conversation_context', []))
        return completed_result(question)
    monkeypatch.setattr(state['rag'], 'ask_agent', agent)
    assert c.post('/api/ask', json={'question':'first'}).status_code == 200
    assert received == [[]]
    assert c.post('/api/ask', json={'question':'followup'}).status_code == 200
    assert received[1][0]['question'] == 'first'
    assert received[1][0]['resolved_question'] == 'resolved first'
    assert len(state['conversation'].turns) == 2
    version = state['conversation'].workspace_version
    upload(c, raw)
    assert next(iter(web.workspaces.values()))['conversation'].workspace_version == version
    assert upload(c, b'broken').status_code == 400
    assert len(state['conversation'].turns) == 2
    with TestClient(web.app) as other:
        upload(other, pdf('other'))
        assert not workspace_for(other)['conversation'].turns
    upload(c, pdf('addition'))
    current = workspace_for(c)
    assert not current['conversation'].turns
    assert current['conversation'].workspace_version != version
    c.delete('/api/documents')
    upload(c, raw)
    assert not workspace_for(c)['conversation'].turns


def test_stream_completed_history_available_to_next_request(client, monkeypatch):
    import json
    c, _ = client
    upload(c, pdf('report'))
    state = next(iter(web.workspaces.values()))
    def agent(question, **kwargs):
        if question == 'second':
            assert kwargs['conversation_context'][0]['question'] == 'first'
        return completed_result(question)
    monkeypatch.setattr(state['rag'], 'ask_agent', agent)
    response = c.post('/api/ask/stream', json={'question':'first'})
    events = [json.loads(line) for line in response.content.splitlines()]
    assert events[-1]['type'] == 'done'
    assert c.post('/api/ask', json={'question':'second'}).status_code == 200
    assert len(state['conversation'].turns) == 2


def test_stream_disconnect_does_not_commit_history(client, monkeypatch):
    """直接驱动 ASGI disconnect，避免 TestClient 缓冲流掩盖断线。"""
    import asyncio
    import threading
    c, _ = client
    upload(c, pdf('report'))
    state = workspace_for(c)
    release, finished = threading.Event(), threading.Event()
    def agent(question, on_event, **kwargs):
        on_event({'type':'progress','message':'started'})
        assert release.wait(5)
        finished.set()
        return completed_result(question)
    monkeypatch.setattr(state['rag'], 'ask_agent', agent)

    async def drive():
        disconnected = asyncio.Event()
        sent_body = False
        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {'type':'http.request', 'body':b'{"question":"first"}', 'more_body':False}
            await disconnected.wait()
            return {'type':'http.disconnect'}
        async def send(message):
            if message['type'] == 'http.response.body' and b'started' in message.get('body', b''):
                disconnected.set()
        scope = {'type':'http','asgi':{'version':'3.0','spec_version':'2.3'},
                 'http_version':'1.1','method':'POST','scheme':'http',
                 'path':'/api/ask/stream','raw_path':b'/api/ask/stream','query_string':b'',
                 'root_path':'','server':('testserver',80),'client':('testclient',123),
                 'headers':[(b'content-type',b'application/json'),
                            (b'cookie', ('session='+c.cookies.get('session')).encode())]}
        await asyncio.wait_for(web.app(scope, receive, send), timeout=5)
    try:
        asyncio.run(drive())
    finally:
        release.set()
    assert finished.wait(5)
    with web.lock:
        assert state['conversation'].turns == []


def workspace_for(client):
    encoded = client.cookies.get('session').split('.')[0]
    session = json.loads(base64.b64decode(encoded))
    return web.workspaces[session['id']]


def test_fastapi_page_static_files_and_schema(client):
    c, _ = client
    assert '阅财' in c.get('/').text
    assert c.get('/static/app.js').status_code == 200
    schema = c.get('/openapi.json').json()
    assert '/api/ask/stream' in schema['paths']
    assert c.get('/docs').status_code == 200


@pytest.mark.parametrize('body', [{}, {'question':''}, {'question':'  '}, {'question':4}, {'question':'x'*4001}, []])
def test_question_validation(client, body):
    c, _ = client
    for path in ('/api/ask', '/api/ask/stream'):
        response = c.post(path, json=body)
        assert response.status_code == 400
        assert 'error' in response.json()


def test_oversized_upload_and_multipart_errors(client):
    c, _ = client
    response = c.post('/api/upload', content=b'x', headers={'content-length':str(102*1024*1024)})
    assert response.status_code == 413
    assert 'error' in response.json()
    response = c.post('/api/upload', files=[('files',(f'{i}.pdf',b'x','application/pdf')) for i in range(11)])
    assert response.status_code == 400
    assert 'error' in response.json()


def test_upload_limit_without_content_length(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(web, 'MAX_UPLOAD_REQUEST_BYTES', 64)
    def body():
        yield b'--boundary\r\nContent-Disposition: form-data; name="files"; filename="x.pdf"\r\n\r\n'
        yield b'x' * 100
        yield b'\r\n--boundary--\r\n'
    response = c.post('/api/upload', content=body(),
                      headers={'content-type':'multipart/form-data; boundary=boundary'})
    assert 'content-length' not in response.request.headers
    assert response.status_code == 413
    assert 'error' in response.json()


def test_mixed_upload_structure_sources_export_and_isolation(client,monkeypatch,tmp_path):
    from test_structured_documents import word_file,excel_file
    c,embedding=client
    word=word_file(tmp_path/'report.docx').read_bytes()
    excel=excel_file(tmp_path/'report.xlsx').read_bytes()
    assert upload(c,pdf('Original'),'first.pdf').status_code==200
    response=c.post('/api/upload',files=[('files',('report.docx',word)),('files',('report.xlsx',excel))])
    assert response.status_code==200,response.text
    state=workspace_for(c)
    assert [d['format'] for d in response.json()['documents']]==['pdf','docx','xlsx']
    doc_id=next(d['document_id'] for d in state['documents'] if d['format']=='xlsx')
    structured=c.get(f'/api/documents/{doc_id}/structure')
    assert structured.status_code==200 and structured.json()['tables'][0]['cells']
    assert c.get(f'/api/documents/{doc_id}/original').content==excel
    with TestClient(web.app) as other:
        assert other.get(f'/api/documents/{doc_id}/structure').status_code==404
        assert other.get(f'/api/documents/{doc_id}/original').status_code==404
    chunks=[v for v in state['rag'].chunks_by_id.values() if v['document_id']==doc_id]
    from financial_rag.memory.evidence import EvidenceMemory
    mem=EvidenceMemory();mem.add_evidence(chunks,state['rag'].chunks_by_id)
    from financial_rag.generation.citations import check_citations
    sources=check_citations('answer [E1]',mem.citation_map)['sources']
    assert sources[0]['source']['format']=='xlsx'
    assert sources[0]['page'] is None
    result=completed_result('mixed');result['evidence']=mem.evidence;result['sources']=sources
    monkeypatch.setattr(state['rag'],'ask_agent',lambda q:result)
    monkeypatch.setattr(web,'EXPORT_ROOT',tmp_path/'exports')
    rid=c.post('/api/ask',json={'question':'mixed'}).json()['run_id']
    c.delete('/api/documents')
    saved=c.post(f'/api/runs/{rid}/save',json={})
    assert saved.status_code==200,saved.text
    destination=Path(saved.json()['path'])
    assert {p.suffix for p in (destination/'documents').iterdir()}=={'.pdf','.docx','.xlsx'}
    assert json.loads((destination/'structured'/f'{doc_id}.json').read_text())==structured.json()


def test_scanned_pdf_rejected_with_explicit_ocr_reason(client):
    c,_=client
    with pymupdf.open() as doc:
        doc.new_page();raw=doc.tobytes()
    response=upload(c,raw)
    assert response.status_code==400 and 'OCR' in response.json()['error']



def test_saved_library_restores_in_fresh_app_without_reencoding(client, monkeypatch):
    c, embedding = client
    raw = pdf('persisted revenue 321')
    upload(c, raw)
    state = workspace_for(c)
    ids = set(state['rag'].chunks_by_id)
    cookie = c.cookies.get('session')
    # A fresh app module rebuilds middleware and in-memory caches from disk.
    fresh_spec = importlib.util.spec_from_file_location('restarted_web', Path(web.__file__))
    fresh = importlib.util.module_from_spec(fresh_spec)
    fresh_spec.loader.exec_module(fresh)
    monkeypatch.setattr(fresh, 'LIBRARY_ROOT', web.LIBRARY_ROOT)
    monkeypatch.setattr(fresh, 'models', web.models)
    monkeypatch.setattr(fresh, 'document_tokenizer', lambda: (_ for _ in ()).throw(AssertionError('must not reparse')))
    with TestClient(fresh.app) as restarted:
        restarted.cookies.set('session', cookie)
        response = restarted.get('/api/status')
        assert response.status_code == 200
        assert len(response.json()['documents']) == 1
        restored = next(iter(fresh.workspaces.values()))
        assert restored['rag'] is None  # Listing does not load models.
        did = next(iter(restored['names']))
        assert restarted.get(f'/api/documents/{did}/original').content == raw
        session_id = next(iter(fresh.workspaces))
        fresh.get_workspace(session_id, hydrate=True)
        assert set(restored['rag'].chunks_by_id) == ids
        assert len(embedding.batches) == 1
        assert not restored['conversation'].turns
        monkeypatch.setattr(restored['rag'], 'ask_agent', lambda q: completed_result(q))
        assert restarted.post('/api/ask', json={'question':'recovered?'}).status_code == 200


def test_delete_document_removes_all_stages_and_survives_restart(client, monkeypatch):
    c, embedding = client
    upload(c, pdf('obsolete revenue 111'), 'old.pdf')
    upload(c, pdf('retained revenue 222'), 'keep.pdf')
    state = workspace_for(c)
    removed = state['documents'][0]['document_id']
    with TestClient(web.app) as other:
        assert other.delete(f'/api/documents/{removed}').status_code == 404
    response = c.delete(f'/api/documents/{removed}')
    assert [d['name'] for d in response.json()['documents']] == ['keep.pdf']
    assert c.get(f'/api/documents/{removed}/original').status_code == 404
    assert c.get(f'/api/documents/{removed}/structure').status_code == 404
    assert len(embedding.batches) == 2
    web.workspaces.clear()
    assert len(c.get('/api/status').json()['documents']) == 1
    sid = next(iter(web.workspaces))
    restored = web.get_workspace(sid, hydrate=True)
    assert removed not in {x['document_id'] for x in restored['rag'].retriever.chunks}
    monkeypatch.setattr(embedding, 'encode_query', lambda q: np.array([1.,0.]), raising=False)
    class Rank:
        last_truncated_chunk_ids = []
        def rerank(self, query, candidates, top_k):
            return [{**x, 'rerank_score':1.} for x in candidates[:top_k]]
    restored['rag'].retriever.reranker = Rank()
    trace = restored['rag'].retrieve('obsolete revenue 111')
    for stage in ('dense','bm25','hybrid','reranked'):
        assert all(restored['rag'].chunks_by_id[x['chunk_id']]['document_id'] != removed for x in trace[stage])
    assert not restored['conversation'].turns
    remaining = next(iter(restored['names']))
    assert c.delete(f'/api/documents/{remaining}').json()['documents'] == []
    web.workspaces.clear()
    assert c.get('/api/status').json()['documents'] == []


def test_publish_failure_keeps_saved_library_and_index(client, monkeypatch):
    c, embedding = client
    upload(c, pdf('original saved'))
    old = workspace_for(c)
    from financial_rag.library import LibraryStore
    def fail(*args):raise OSError('disk full')
    with monkeypatch.context() as m:
        m.setattr(LibraryStore, '_publish', fail)
        assert upload(c, pdf('new failed')).status_code == 500
    assert workspace_for(c) is old
    assert Path(old['folder'].name).exists()
    web.workspaces.clear()
    assert len(c.get('/api/status').json()['documents']) == 1
    assert 'original saved' in web.get_workspace(next(iter(web.workspaces)), hydrate=True)['rag'].retriever.chunks[0]['text']


def test_delete_failure_keeps_original_saved_state(client, monkeypatch):
    c, _ = client
    upload(c, pdf('first'))
    upload(c, pdf('second'))
    old = workspace_for(c)
    from financial_rag.library import LibraryStore
    def fail(*args):raise OSError('disk full')
    with monkeypatch.context() as m:
        m.setattr(LibraryStore, '_publish', fail)
        assert c.delete('/api/documents/'+old['documents'][0]['document_id']).status_code == 500
    assert workspace_for(c) is old
    web.workspaces.clear()
    assert len(c.get('/api/status').json()['documents']) == 2


def test_without_api_key_upload_restore_and_delete_work(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    assert upload(c, pdf('offline document')).status_code == 200
    web.workspaces.clear()
    status = c.get('/api/status').json()
    assert len(status['documents']) == 1 and not status['key_ready']
    assert c.post('/api/ask', json={'question':'q'}).status_code == 400
    assert c.post('/api/ask/stream', json={'question':'q'}).status_code == 400
    assert c.delete('/api/documents').status_code == 200
    web.workspaces.clear()
    assert c.get('/api/status').json()['documents'] == []


def test_corrupt_manifest_is_reported_not_overwritten(client):
    c, _ = client
    upload(c, pdf('preserve this'))
    old = workspace_for(c)
    manifest = Path(old['folder'].name)/'manifest.json'
    manifest.write_text('broken')
    web.workspaces.clear()
    assert c.get('/api/status').status_code == 503
    assert upload(c, pdf('do not overwrite')).status_code == 500
    assert manifest.read_text() == 'broken'
