"""当前字典/JSON 数据契约，使用 TypedDict 描述，不改变运行时存储格式。

覆盖页面、chunk、单路检索、RRF、重排与批量报告。page 为单个 PDF 物理页码；
模型分数分别保存为 score / rrf_score / rerank_score，不能混用。
TypedDict 仅提供静态提示，不执行运行时校验；现有模块尚未全部接入这些类型注解。
AnswerResult 和跨页 pages 留待实际实现，不将规划字段标为当前必需字段。
"""
from typing import NotRequired, TypedDict


class TextBlock(TypedDict):
    block_id: int
    bbox: list[float]  # [x0, y0, x1, y1]
    text: str


class ParsedPage(TypedDict):
    page: int
    blocks: list[TextBlock]
    text: str
    document_id: NotRequired[str]
    section: NotRequired[str | None]
    title: NotRequired[str | None]


class Chunk(TypedDict):
    chunk_id: str
    document_id: str
    page: int
    word_count: int
    text: str
    section: NotRequired[str | None]
    title: NotRequired[str | None]


class SearchResult(TypedDict):
    """Dense 或 BM25 输出；score 的含义由所属方法确定。"""
    chunk_id: str
    page: int
    text: str
    score: float


class RankedSearchResult(SearchResult):
    rank: int  # 当前方法展示排名，从 1 开始


class FusionResult(TypedDict):
    chunk_id: str
    page: int
    text: str
    dense_rank: int | None  # 两路完整候选中的排名，不是最终展示排名
    bm25_rank: int | None
    rrf_score: float


class RankedFusionResult(FusionResult):
    rank: int


class RerankedResult(FusionResult):
    """当前对比流程对 RRF 候选重排；原始 RRF 字段保留。"""
    rerank_score: float  # 可以为负数，不是概率或拒答阈值


class RankedRerankedResult(RerankedResult):
    rank: int  # 重排后的最终排名


class EvaluationQuestion(TypedDict):
    id: str
    question: str
    expected_pages: NotRequired[list[int]]  # 未来人工标注，当前问题集未提供
    evidence_note: NotRequired[str]


class CandidateResults(TypedDict):
    dense: list[SearchResult]
    bm25: list[SearchResult]
    rrf: list[FusionResult]  # 实际交给 reranker 的 RRF 候选


class QueryComparison(TypedDict):
    id: str
    question: str
    dense: list[RankedSearchResult]
    bm25: list[RankedSearchResult]
    hybrid: list[RankedFusionResult]
    candidates: CandidateResults
    reranked: NotRequired[list[RankedRerankedResult]]
    # 每个问题下超限的候选 ID，包含未进入最终 Top-K 的候选。
    # 不能据此断言答案证据被截掉，需检查实际 token 位置。
    reranker_truncated_chunk_ids: NotRequired[list[str]]


class BM25Config(TypedDict):
    k1: float
    b: float
    tokenizer: str
    idf: str


class RerankerConfig(TypedDict):
    model: str
    max_length: int  # 问题、正文和特殊 token 合计限制


class ComparisonReport(TypedDict):
    """compare_methods 返回的基础报告；CLI 额外补充可选运行元数据。"""
    top_k: int
    candidate_k: int
    rrf_k: int
    chunk_count: int
    bm25_config: BM25Config
    note: str
    queries: list[QueryComparison]
    reranker_config: NotRequired[RerankerConfig | None]
    created_at: NotRequired[str]
    model: NotRequired[str]
    normalize_embeddings: NotRequired[bool]
    chunks_sha256: NotRequired[str]
    questions_sha256: NotRequired[str]
