"""NumPy 内存向量检索。

输入：查询向量、与 chunks 一一对应的文档向量矩阵和 top_k。
对归一化向量做点积（等价于余弦相似度），按分数降序返回
chunk_id/page/score/text；不在本模块编码文本或建立持久化索引。
当前不提供文档权限过滤，也没有完整的向量形状与参数校验。
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
