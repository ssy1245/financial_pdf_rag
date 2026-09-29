"""固定向量的三路对比回归测试：单次编码、同题查询、缺少 BM25 匹配及完整候选融合。
"""
import numpy as np
import pytest
from financial_rag.evaluation.compare_methods import compare_methods


class TinyEmbedding:
    def __init__(self):
        self.document_calls = 0
        self.queries = []

    def encode_documents(self, texts):
        self.document_calls += 1
        assert texts == ["china revenue", "operating cash"]
        return np.eye(2)

    def encode_query(self, query):
        self.queries.append(query)
        return np.array([1., 0.]) if query == "china" else np.array([0., 1.])


def test_both_methods_share_queries_and_documents():
    chunks = [{"chunk_id": "a", "page": 1, "text": "china revenue"},
              {"chunk_id": "b", "page": 2, "text": "operating cash"}]
    questions = [{"id": "china", "question": "china"},
                 {"id": "cash", "question": "cash"},
                 {"id": "unknown", "question": "unknown"}]
    model = TinyEmbedding()
    report = compare_methods(chunks, questions, model, 5)
    assert model.document_calls == 1
    assert model.queries == [q["question"] for q in questions]
    for case, expected in zip(report["queries"][:2], ["a", "b"]):
        assert case["dense"][0]["chunk_id"] == expected
        assert case["bm25"][0]["chunk_id"] == expected
        assert len(case["dense"]) == 2 and len(case["bm25"]) == 1
        assert case["bm25"][0]["rank"] == 1
        assert case["hybrid"][0]["chunk_id"] == expected
        assert case["hybrid"][0]["dense_rank"] == 1
        assert case["hybrid"][0]["bm25_rank"] == 1
        assert case["hybrid"][0]["rrf_score"] == pytest.approx(2 / 61)
    assert report["queries"][2]["bm25"] == []


def test_invalid_inputs_fail_before_encoding():
    model = TinyEmbedding()
    with pytest.raises(ValueError):
        compare_methods([], [], model)
    with pytest.raises(ValueError):
        compare_methods([], [], model, 0)
    assert model.document_calls == 0


def test_fusion_uses_candidates_beyond_display_top_k(monkeypatch):
    import financial_rag.evaluation.compare_methods as module
    a = {"chunk_id": "a", "page": 1, "text": "china revenue"}
    b = {"chunk_id": "b", "page": 2, "text": "operating cash"}
    calls = []
    def fake_dense(query, embeddings, chunks, top_k):
        calls.append(top_k)
        return [{**a, "score": .9}, {**b, "score": .8}]
    def fake_bm25(query, index, top_k):
        calls.append(top_k)
        return [{**b, "score": 5.}]
    monkeypatch.setattr(module, "dense_search", fake_dense)
    monkeypatch.setattr(module, "bm25_search", fake_bm25)
    report = compare_methods([a, b], [{"id": "q", "question": "china"}], TinyEmbedding(), top_k=1, candidate_k=2)
    case = report["queries"][0]
    assert calls == [2, 2]
    assert case["dense"][0]["chunk_id"] == "a"
    assert case["hybrid"][0]["chunk_id"] == "b"
    assert case["hybrid"][0]["dense_rank"] == 2
    assert case["hybrid"][0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)


def test_reranker_receives_rrf_candidates_before_top_k():
    class FakeReranker:
        last_truncated_chunk_ids = []
        def rerank(self, query, candidates, top_k):
            assert len(candidates) == 2 and top_k == 1
            return [{**candidates[-1], "rerank_score": 9.0}]
    chunks = [{"chunk_id": "a", "page": 1, "text": "china revenue"},
              {"chunk_id": "b", "page": 2, "text": "operating cash"}]
    report = compare_methods(chunks, [{"id": "q", "question": "china"}], TinyEmbedding(),
                             top_k=1, candidate_k=2, reranker=FakeReranker())
    case = report["queries"][0]
    assert case["hybrid"][0]["chunk_id"] == "a"
    assert case["reranked"][0]["chunk_id"] == "b"
    assert case["reranked"][0]["rank"] == 1
