"""追加上传的离线集成测试：真实 PDF 解析和 SQLite，模型用替身。"""
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import pymupdf
import pytest

spec = importlib.util.spec_from_file_location('web_app_tests', Path(__file__).resolve().parents[1] / 'app.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


def pdf(text):
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), text)
        return doc.tobytes()


@pytest.fixture
def client(monkeypatch):
    class Embedding:
        def __init__(self):
            self.batches = []
        def encode_documents(self, texts):
            self.batches.append(texts)
            return np.tile([1., 0.], (len(texts), 1))
    embedding = Embedding()
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-test')
    monkeypatch.setattr(web, 'models', lambda: (embedding, object()))
    monkeypatch.setattr('financial_rag.generation.generator.DeepSeekGenerator', lambda: object())
    with web.app.test_client() as client:
        yield client, embedding
    for state in web.workspaces.values():
        state['folder'].cleanup()
    web.workspaces.clear()
    for records in web.run_snapshots.values():
        for snapshot in records.values():
            snapshot.cleanup()
    web.run_snapshots.clear()


def upload(client, data, name='report.pdf'):
    return client.post('/api/upload', data={'files': (BytesIO(data), name)})


def test_append_deduplicates_and_encodes_only_new_chunks(client):
    c, embedding = client
    first = pdf('first revenue 100')
    assert upload(c, first, 'first.pdf').status_code == 200
    old = next(iter(web.workspaces.values()))
    old_ids = set(old['rag'].chunks_by_id)
    response = upload(c, pdf('second revenue 200'), 'second.pdf')
    assert [d['name'] for d in response.json['documents']] == ['first.pdf', 'second.pdf']
    current = next(iter(web.workspaces.values()))
    assert old_ids < set(current['rag'].chunks_by_id)
    assert len(embedding.batches) == 2
    assert all('first' not in text for text in embedding.batches[1])
    assert not Path(old['folder'].name).exists()
    assert len(list(Path(current['folder'].name).glob('*.pdf'))) == 2
    response = upload(c, first, 'renamed.pdf')
    assert len(response.json['documents']) == 2
    assert next(iter(web.workspaces.values())) is current
    assert len(embedding.batches) == 2
    assert Path(current['folder'].name).exists()
    assert c.delete('/api/documents').status_code == 200
    assert c.get('/api/status').json['documents'] == []


def test_bad_batch_leaves_original_workspace_intact(client):
    c, embedding = client
    upload(c, pdf('original'))
    old = next(iter(web.workspaces.values()))
    response = c.post('/api/upload', data={'files': [
        (BytesIO(pdf('valid addition')), 'valid.pdf'),
        (BytesIO(b'broken'), 'broken.pdf')]})
    assert response.status_code == 400
    assert next(iter(web.workspaces.values())) is old
    assert len(c.get('/api/status').json['documents']) == 1
    assert len(embedding.batches) == 1


def test_cumulative_count_limit_and_session_isolation(client):
    c, _ = client
    payloads = [pdf(f'document {i}') for i in range(10)]
    response = c.post('/api/upload', data={'files': [(BytesIO(data), f'{i}.pdf') for i, data in enumerate(payloads)]})
    assert len(response.json['documents']) == 10
    assert upload(c, payloads[0]).status_code == 200  # 重复文件不占名额
    assert upload(c, pdf('eleventh')).status_code == 400
    with web.app.test_client() as other:
        assert other.get('/api/status').json['documents'] == []
        assert upload(other, pdf('another user')).status_code == 200
        assert len(other.get('/api/status').json['documents']) == 1
    assert len(c.get('/api/status').json['documents']) == 10


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
    assert response.json['sources'][0]['text'] == 'later round evidence'
    assert response.json['sources'][0]['filename'] == 'first.pdf'
    assert response.json['model_calls'] == 3
    assert response.json['status'] == 'limited'
    assert response.json['elapsed_seconds'] >= 0


def test_agent_failure_is_not_reported_as_success(client, monkeypatch):
    c, _ = client
    upload(c, pdf('original'))
    state = next(iter(web.workspaces.values()))
    monkeypatch.setattr(state['rag'], 'ask_agent', lambda q: {'status':'error'})
    response = c.post('/api/ask', json={'question':'q'})
    assert response.status_code == 502
    assert 'error' in response.json


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
    events = [json.loads(line) for line in response.data.splitlines()]
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
    events = [json.loads(line) for line in response.data.splitlines()]
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
    run_id = response.json['run_id']
    assert run_id
    assert not web.EXPORT_ROOT.exists()
    with web.app.test_client() as other:
        assert other.post(f'/api/runs/{run_id}/save', json={}).status_code == 404
    c.delete('/api/documents')
    response = c.post(f'/api/runs/{run_id}/save', json={'browser_elapsed_seconds':14.8})
    assert response.status_code == 200
    target = Path(response.json['path'])
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
    assert c.post(f'/api/runs/{run_id}/save', json={}).json['path'] == str(target)
    assert len(list(web.EXPORT_ROOT.iterdir())) == 1
    assert c.post(f'/api/runs/{run_id}/save', json={'browser_elapsed_seconds':-1}).status_code == 400
