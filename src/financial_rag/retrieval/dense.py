"""向量检索。

状态：架构占位，尚未实现业务逻辑。

职责与输入输出：
输入：问题、top_k 和文档范围。输出：统一 SearchResult 列表。
调用查询编码和向量索引，恢复 chunk 正文及元数据，记录 dense_rank 与 dense_score。

边界与约束：
不建立索引、不调用生成模型；索引层提供搜索能力，本模块负责组合查询编码和结果装配。
"""
import numpy as np


def dense_search(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:

    scores = chunk_embeddings @ query_embedding #因为normalize_embeddings=True所以这里等价于cosine similarity
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in top_indices:
        results.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "page": chunks[index]["page"],
                "score": float(scores[index]),
                "text": chunks[index]["text"],
            }
        )

    return results
