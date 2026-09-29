"""离线验证多轮去重、候选补足及请求隔离；不下载模型。"""
import numpy as np
import pytest

from financial_rag.retrieval.retriever import Retriever
from financial_rag.pipeline import FinancialRAG


class Store:
    def __init__(self):
        self.loads = 0

    def load(self, name):
        self.loads += 1
        chunks = [dict(chunk_id=str(i), document_id="doc", page=i + 1,
                       text="revenue " * (6 - i)) for i in range(6)]
        angles = np.linspace(0, 1, 6)
        return chunks, np.column_stack((np.cos(angles), np.sin(angles)))


class Embedding:
    def __init__(self):
        self.calls = 0

    def encode_query(self, query):
        self.calls += 1
        return np.array([1., 0.])


class Reranker:
    def __init__(self):
        self.calls = 0
        self.last_truncated_chunk_ids = []

    def rerank(self, query, candidates, top_k):
        self.calls += 1
        self.last_truncated_chunk_ids = [candidates[0]["chunk_id"]]
        return [{**c, "rerank_score": 1.0} for c in candidates[:top_k]]


@pytest.fixture
def retriever():
    return Retriever(Store(), Embedding(), "test", Reranker(), candidate_k=2, top_k=2)


def ids(results):
    return [r["chunk_id"] for r in results]


def test_exclusion_refills_each_stage_and_does_not_leak_between_requests(retriever):
    first = retriever.retrieve("revenue")
    assert ids(first["reranked"]) == ["0", "1"]
    seen = set(ids(first["reranked"]))
    second = retriever.retrieve("revenue", exclude_chunk_ids=seen)
    for stage in ("dense", "bm25", "hybrid", "reranked"):
        assert ids(second[stage]) == ["2", "3"]
    assert seen == {"0", "1"}
    third = retriever.retrieve("revenue", exclude_chunk_ids=seen | {"2", "3"})
    assert ids(third["reranked"]) == ["4", "5"]
    assert retriever.retrieve("revenue") == first
    assert first["reranker_truncated_chunk_ids"] == ["0"]


def test_exhausted_corpus_skips_models_and_returns_clean_trace(retriever):
    retriever.retrieve("revenue")
    result = retriever.retrieve("revenue", exclude_chunk_ids=set(retriever.chunks_by_id))
    assert all(value == [] for value in result.values())
    assert retriever.embedding_model.calls == retriever.reranker.calls == 1


def test_override_unknown_ids_and_fewer_remaining_results(retriever):
    first = retriever.retrieve("revenue", top_k=1, exclude_chunk_ids=["unknown"])
    assert len(first["reranked"]) == 1
    assert retriever.top_k == 2
    last = retriever.retrieve("revenue", exclude_chunk_ids=["0", "1", "2", "3", "4", "4"])
    assert ids(last["reranked"]) == ["5"]


def test_no_bm25_matches_still_uses_dense(retriever):
    result = retriever.retrieve("unmatched")
    assert result["bm25"] == []
    assert len(result["reranked"]) == 2


@pytest.mark.parametrize("kwargs", [dict(top_k=0), dict(top_k=3), dict(top_k=True),
    dict(top_k=1.5), dict(exclude_chunk_ids="0"), dict(exclude_chunk_ids=[1])])
def test_invalid_options_fail_before_model_calls(retriever, kwargs):
    with pytest.raises(ValueError):
        retriever.retrieve("revenue", **kwargs)
    assert retriever.embedding_model.calls == 0


def test_pipeline_delegates_exclusions_and_loads_store_once():
    store = Store()
    rag = FinancialRAG(store, Embedding(), "test", Reranker(), None, candidate_k=2, top_k=2)
    first = rag.retrieve("revenue")
    second = rag.retrieve("revenue", exclude_chunk_ids=ids(first["reranked"]), top_k=1)
    assert ids(second["reranked"]) == ["2"]
    assert store.loads == 1


def test_memory_drives_retrieval_until_exhausted(retriever):
    from financial_rag.memory import EvidenceMemory
    memory = EvidenceMemory()
    for expected in (2, 2, 2, 0):
        trace = retriever.retrieve("revenue", exclude_chunk_ids=memory.chunk_ids)
        record = memory.record_search("revenue", trace["reranked"], retriever.chunks_by_id)
        assert record["new_count"] == expected
    assert len(memory.evidence) == 6
    assert list(memory.citation_map) == [f"E{i}" for i in range(1, 7)]
    assert "[E1]" in memory.build_context() and "[E6]" in memory.build_context()
