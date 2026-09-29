"""无模型下载的重排逻辑测试。"""
from copy import deepcopy
import pytest
from financial_rag.retrieval.reranker import Reranker


class FakeModel:
    def tokenizer(self, queries, texts, **kwargs):
        return {"length": [10, 600]}
    def predict(self, pairs, **kwargs):
        assert pairs == [("q", "first"), ("q", "second")]
        return [0.2, 0.9]


def test_scores_metadata_truncation_and_empty_input():
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeModel()
    reranker.max_length = 512
    candidates = [{"chunk_id": "a", "text": "first", "page": 1, "rrf_score": .03},
                  {"chunk_id": "b", "text": "second", "page": 2, "rrf_score": .02}]
    before = deepcopy(candidates)
    with pytest.warns(UserWarning, match="1 个"):
        results = reranker.rerank("q", candidates, 1)
    assert results == [{**candidates[1], "rerank_score": .9}]
    assert candidates == before
    assert reranker.last_truncated_chunk_ids == ["b"]
    assert reranker.rerank("q", [], 5) == []
    assert reranker.last_truncated_chunk_ids == []
