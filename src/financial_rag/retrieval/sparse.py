"""BM25 Top-K 检索。

输入：查询文本、已建立的 BM25Index 和 top_k。
调用实例 get_scores，按分数降序返回 chunk_id/page/score/text；
过滤零分，允许结果少于 K 条，top_k<=0 返回空列表。
不重建索引，不执行融合、文档权限过滤或回答生成。
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
