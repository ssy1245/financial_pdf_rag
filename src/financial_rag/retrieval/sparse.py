"""BM25 检索。

状态：架构占位，尚未实现业务逻辑。

职责与输入输出：
输入：问题、top_k 和文档范围。输出：统一 SearchResult 列表。
调用 BM25 索引，恢复 chunk 正文及元数据，记录 bm25_rank 与 bm25_score。

边界与约束：
不重新建立索引，不执行融合或生成。
"""
from financial_rag.indexing.bm25_index import BM25Index


def bm25_search(
    query: str,
    index: BM25Index,
    top_k: int = 5,
) -> list[dict]:
    if top_k <= 0:
        return []

    scores = index.get_scores(query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for i in ranked_indices:
        # 当前使用正值 IDF，无词匹配时得分为 0
        if scores[i] <= 0:
            continue

        chunk = index.chunks[i]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "score": scores[i],
            "text": chunk["text"],
        })

        if len(results) == top_k:
            break

    return results
