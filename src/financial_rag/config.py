"""当前主应用开发基线；历史实验显式参数保持独立。"""

# 适用于 BAAI/bge-reranker-v2-m3；分数阈值不是正确概率。
RETRIEVAL_DEFAULTS = {
    "candidate_k": 20,
    "top_k": 7,
    "rrf_k": 60,
    "rerank_threshold": 0.1,
}
