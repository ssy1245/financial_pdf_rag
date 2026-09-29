"""BM25 实例评分与搜索基础回归测试；不覆盖全部公式和边界情形。
"""
from financial_rag.indexing.bm25_index import BM25Index
from financial_rag.retrieval.sparse import bm25_search


def test_instance_scoring_and_search():
    chunks = [
        {"chunk_id": "a", "page": 1, "text": "china revenue"},
        {"chunk_id": "b", "page": 2, "text": "operating cash"},
    ]
    index = BM25Index(chunks)
    scores = index.get_scores("china")
    assert len(scores) == 2 and scores[0] > 0 and scores[1] == 0
    results = bm25_search("china", index, top_k=5)
    assert results == [{**chunks[0], "score": scores[0]}]
    assert bm25_search("unknown", index) == []
