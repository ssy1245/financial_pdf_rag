"""复用已加载索引执行 Dense/BM25 → RRF → Reranker 检索。

保留各阶段排名用于诊断；不调用回答模型、不管理证据记忆。
调用方可用 exclude_chunk_ids 在多轮检索中排除已经交给模型的片段。
"""
import numpy as np

from financial_rag.indexing.bm25_index import BM25Index
from financial_rag.retrieval.dense import dense_search
from financial_rag.retrieval.sparse import bm25_search
from financial_rag.retrieval.fusion import reciprocal_rank_fusion


class Retriever:
    def __init__(self, store, embedding_model, embedding_model_name,
                 reranker, candidate_k=20, top_k=5, rrf_k=60):
        if (any(type(value) is not int for value in (candidate_k, top_k, rrf_k))
                or top_k <= 0 or candidate_k < top_k or rrf_k < 0):
            raise ValueError("要求 candidate_k >= top_k > 0，rrf_k >= 0")
        self.chunks, self.vectors = store.load(embedding_model_name)
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        if len(self.chunks_by_id) != len(self.chunks):
            raise ValueError("chunk_id 必须唯一")
        self.bm25 = BM25Index(self.chunks)
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, question, *, top_k=None, exclude_chunk_ids=None):
        """返回各阶段候选；只排除调用方传入的已读 ID，不保存会话状态。

        top_k 可按次覆盖，范围为 1..candidate_k。未知 ID 忽略。
        排除后重新计算两路排名；BM25 仍使用完整语料的统计量。
        返回空列表表示没有剩余片段，不代表剩余片段一定能回答问题。
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        final_k = self.top_k if top_k is None else top_k
        if type(final_k) is not int or not 0 < final_k <= self.candidate_k:
            raise ValueError("要求 candidate_k >= top_k > 0，top_k 必须为整数")
        if isinstance(exclude_chunk_ids, (str, bytes)):
            raise ValueError("exclude_chunk_ids 必须是 chunk_id 字符串的集合或序列")
        excluded = set() if exclude_chunk_ids is None else set(exclude_chunk_ids)
        if any(not isinstance(chunk_id, str) for chunk_id in excluded):
            raise ValueError("exclude_chunk_ids 中的 ID 必须是字符串")
        excluded.intersection_update(self.chunks_by_id)
        if len(excluded) == len(self.chunks):
            return {"dense": [], "bm25": [], "hybrid": [], "reranked": [],
                    "reranker_truncated_chunk_ids": []}
        # 多取已排除 ID 的数量，保证过滤后仍有足够的候选；不重建索引。
        fetch_k = min(len(self.chunks), self.candidate_k + len(excluded))
        query_vector = np.asarray(self.embedding_model.encode_query(question))
        if query_vector.shape != (self.vectors.shape[1],):
            raise ValueError("查询向量维度与索引不一致")
        if not np.isfinite(query_vector).all() or not np.isclose(
            np.linalg.norm(query_vector), 1.0, atol=1e-4
        ):
            raise ValueError("查询向量必须有限且已经归一化")
        dense = dense_search(query_vector, self.vectors, self.chunks, fetch_k)
        bm25 = bm25_search(question, self.bm25, fetch_k)
        dense = [r for r in dense if r["chunk_id"] not in excluded][:self.candidate_k]
        bm25 = [r for r in bm25 if r["chunk_id"] not in excluded][:self.candidate_k]
        hybrid = reciprocal_rank_fusion(dense, bm25, k=self.rrf_k, top_k=self.candidate_k)
        reranked = self.reranker.rerank(question, hybrid, top_k=final_k)
        return {
            "dense": dense, "bm25": bm25, "hybrid": hybrid,
            "reranked": reranked,
            "reranker_truncated_chunk_ids": list(self.reranker.last_truncated_chunk_ids),
        }
