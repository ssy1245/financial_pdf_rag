"""离线检查 SQLite 到生成的编排，以及引用异常报告。"""
import numpy as np
import pytest
from financial_rag.pipeline import FinancialRAG
from financial_rag.indexing.vector_store import VectorStore


class Embedding:
    def encode_query(self, question):
        return np.array([1., 0.])

    def encode_documents(self, texts):
        raise AssertionError("问答不应编码文档")


class Reranker:
    last_truncated_chunk_ids = []

    def rerank(self, question, candidates, top_k):
        assert len(candidates) == 2  # RRF 先保留 candidate_k，再重排取 top_k
        return [{**candidates[-1], "rerank_score": 1.0}][:top_k]


class Generator:
    answer = "现金流证据 [E1]"

    def generate(self, messages):
        assert "cash flow" in messages[1]["content"]
        assert "revenue" not in messages[1]["content"]
        return self.answer


@pytest.fixture
def rag(tmp_path):
    store = VectorStore(tmp_path / "vectors.sqlite3")
    chunks = [
        dict(chunk_id="a", document_id="doc", page=1, text="revenue"),
        dict(chunk_id="b", document_id="doc", page=2, text="cash flow"),
    ]
    store.replace_all(chunks, np.eye(2), "test-model")
    return FinancialRAG(store, Embedding(), "test-model", Reranker(), Generator(), candidate_k=2, top_k=1)


def test_pipeline_uses_loaded_vectors_and_reranked_evidence(rag):
    result = rag.ask("financial results")
    assert result["sources"][0]["chunk_id"] == "b"
    assert result["sources"][0]["page"] == 2
    assert result["citation_status"] == "labels_valid"


def test_pipeline_reports_unknown_citation(rag):
    rag.generator.answer = "回答 [E99]"
    result = rag.ask("financial results")
    assert result["unknown_citations"] == ["E99"]
    assert result["sources"] == []
    assert result["citation_status"] == "unknown_citations"


def test_pipeline_rejects_empty_question(rag):
    with pytest.raises(ValueError, match="问题不能为空"):
        rag.ask(" ")


def test_ask_memory_is_request_local(rag):
    first = rag.ask("first question")
    first["search_history"][0]["new_chunk_ids"].clear()
    second = rag.ask("second question")
    assert second["search_history"] == [{
        "query": "second question", "retrieved_chunk_ids": ["b"],
        "new_chunk_ids": ["b"], "new_count": 1,
    }]
    assert list(second["citation_map"]) == ["E1"]
    assert not hasattr(rag, "memory")
