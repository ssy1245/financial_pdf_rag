"""按排名执行 Reciprocal Rank Fusion。

输入：两路已经排好序、各自 chunk_id 唯一的候选；排名从 1 开始。
按 chunk_id 合并并累加 1/(k+rank)，缺席一路贡献为零；
返回 rrf_score、dense_rank、bm25_rank 及来源正文与页码。
同分保持字典插入顺序（先 Dense，后仅 BM25 的项）。不使用原始分数相加。
限制：默认 chunk_id 全局唯一；本函数尚无独立的参数和重复 ID 防御校验。
"""
def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    rrf_results = {}
    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["chunk_id"]
        rrf_results[chunk_id] = {
            "chunk_id": chunk_id,
            "page": result["page"],
            "text": result["text"],
            "dense_rank": rank,
            "bm25_rank": None,
            "rrf_score": 1 / (k + rank),
        }
    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
        # 这个 chunk 可能没有被 Dense 召回
        if chunk_id not in rrf_results:
            rrf_results[chunk_id] = {
                "chunk_id": chunk_id,
                "page": result["page"],
                "text": result["text"],
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        rrf_results[chunk_id]["bm25_rank"] = rank
        rrf_results[chunk_id]["rrf_score"] += 1 / (k + rank)
    ranked_results = sorted(
        rrf_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )
    return ranked_results[:top_k]