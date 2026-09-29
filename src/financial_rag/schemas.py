"""当前字典/JSON 数据契约，使用 TypedDict 描述，不改变运行时存储格式。

覆盖页面、chunk、单路检索、RRF、重排与批量报告。page 为单个 PDF 物理页码；
模型分数分别保存为 score / rrf_score / rerank_score，不能混用。
TypedDict 仅提供静态提示，不执行运行时校验；现有模块尚未全部接入这些类型注解。
包含当前 pipeline、Agent、网页和流式事件输出；Draft 前缀仅用于后续主动检索和对话记忆计划。
跨页 pages 未实现；下列类型不新增运行时行为。
"""
from typing import Literal, NotRequired, TypedDict


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


# 当前 pipeline 与网页已经返回的数据结构。
class CitationSource(TypedDict):
    label: str
    document_id: str
    chunk_id: str
    page: int


class WebCitationSource(CitationSource):
    filename: str
    text: str


class CitationLocation(TypedDict):
    document_id: str
    chunk_id: str
    page: int


class RetrievalTrace(TypedDict):
    dense: list[SearchResult]
    bm25: list[SearchResult]
    hybrid: list[FusionResult]
    reranked: list[RerankedResult]
    reranker_truncated_chunk_ids: list[str]


class SearchHistoryEntry(TypedDict):
    query: str
    retrieved_chunk_ids: list[str]
    new_chunk_ids: list[str]
    new_count: int


class AnswerResult(TypedDict):
    question: str
    answer: str
    sources: list[CitationSource]
    unknown_citations: list[str]
    has_citations: bool
    citation_status: Literal["unknown_citations", "labels_valid", "no_citations"]
    citation_map: dict[str, CitationLocation]
    retrieval: RetrievalTrace
    search_history: NotRequired[list[SearchHistoryEntry]]
    config: NotRequired[dict[str, str | int]]  # CLI 附加，不是 ask() 必需字段


# 已实现的请求内证据、检索工具、Agent 与网页结构。
class Evidence(TypedDict):
    citation_id: str
    document_id: str
    chunk_id: str
    page: int
    text: str
    title: NotRequired[str]
    filename: NotRequired[str]


class QueryPlan(TypedDict):
    queries: list[str]  # 原问题 + 最多两条扩展查询；当前先全部执行再交给模型判断
    warning: str | None


class SearchDocumentsRequest(TypedDict):
    query: str
    top_k: NotRequired[int]


class ToolError(TypedDict):
    code: Literal["unknown_tool", "invalid_arguments", "evidence_limit", "retrieval_failed", "search_limit"]
    message: str


class SearchDocumentsSuccess(TypedDict):
    status: Literal["ok", "no_results"]
    query: str
    evidence: list[Evidence]
    new_count: int
    warnings: list[str]


class SearchDocumentsFailure(TypedDict):
    status: Literal["error"]
    evidence: list[Evidence]  # 当前始终为空
    error: ToolError


SearchDocumentsResult = SearchDocumentsSuccess | SearchDocumentsFailure


class ToolFunction(TypedDict):
    name: str
    arguments: str  # 拼接完整后解析 JSON 并校验，不执行部分参数


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolFunction


class ModelTurn(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: NotRequired[list[ToolCall]]
    reasoning_content: NotRequired[str]  # 仅供协议回传，不展示或写入对话摘要


class ToolObservation(TypedDict):
    tool_call_id: str | None  # 程序执行的初始查询为 None
    result: SearchDocumentsResult


class RetrievalRound(RetrievalTrace):
    query: str


AgentStatus = Literal["completed", "limited", "step_limit", "error"]
AgentStopReason = Literal[
    "search_limit", "step_limit", "no_new_evidence", "evidence_limit", "message_limit",
    "retrieval_failed", "invalid_arguments", "model_error", "invalid_model_response", "invalid_tool_calls",
]


class AgentResult(TypedDict):
    question: str
    answer: str
    sources: list[CitationSource]
    unknown_citations: list[str]
    has_citations: bool
    citation_status: Literal["unknown_citations", "labels_valid", "no_citations"]
    citation_map: dict[str, CitationLocation]
    evidence: dict[str, Evidence]  # chunk_id 为键；一次用户问题内有效
    search_history: list[SearchHistoryEntry]
    retrieval_rounds: list[RetrievalRound]
    tool_observations: list[ToolObservation]
    status: AgentStatus
    stop_reason: AgentStopReason | None
    model_steps: int  # 回答/工具决策调用尝试数，不含查询规划；默认上限 7
    planning_calls: int  # 默认一次，关闭规划时为零
    model_calls: int  # planning_calls + model_steps；不统计本地模型和 SDK 内部重试
    search_attempts: int  # 包括初始查询和无效工具请求；默认上限 8
    query_plan: QueryPlan
    elapsed_seconds: float  # Agent 内部耗时，不等于浏览器端到端耗时


class WebEvidence(TypedDict):
    chunk_id: str
    citation_id: str
    filename: str
    page: int
    text: str
    rank: int  # 本轮最终结果排名，不是 Dense/BM25 排名
    used_in_answer: bool  # 未引用不等于不相关


class WebSearchHistoryEntry(SearchHistoryEntry):
    evidence: list[WebEvidence]  # 全部新增片段，而非仅回答引用的片段


class WebAnswerResult(TypedDict):
    answer: str
    sources: list[WebCitationSource]
    citation_status: Literal["unknown_citations", "labels_valid", "no_citations"]
    status: AgentStatus
    stop_reason: AgentStopReason | None
    query_plan: QueryPlan
    search_history: list[WebSearchHistoryEntry]
    model_calls: int
    model_steps: int | None
    planning_calls: int | None
    search_attempts: NotRequired[int]  # 流式 done 提供；普通 JSON 入口尚未提供
    elapsed_seconds: float


class ProgressEvent(TypedDict):
    type: Literal["progress"]
    message: str
    query: NotRequired[str]


class AnswerDeltaEvent(TypedDict):
    type: Literal["answer_delta"]
    text: str


class AnswerResetEvent(TypedDict):
    type: Literal["answer_reset"]  # 清除工具轮的临时正文


class HeartbeatEvent(TypedDict):
    type: Literal["heartbeat"]


class DoneEvent(TypedDict):
    type: Literal["done"]
    result: WebAnswerResult


class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


StreamEvent = ProgressEvent | AnswerDeltaEvent | AnswerResetEvent | HeartbeatEvent | DoneEvent | ErrorEvent


# 后续计划：以下仅为协议草案，本次不实现，不允许直接传给现有工具。
class DraftGapSearchRequest(SearchDocumentsRequest):
    missing_information: str  # 简短、可展示的信息缺口，与 query 在同一次模型调用中产生


class DraftEvidenceReference(CitationLocation):
    turn_id: str
    citation_id: str  # 必须与 turn_id 联合定位，不能混用上一轮的 E1
    workspace_version: str


class DraftConversationTurn(TypedDict):
    turn_id: str
    question: str
    answer: str  # 完成后的用户可见回答，不存 reasoning_content
    status: AgentStatus
    sources: list[DraftEvidenceReference]
    workspace_version: str


class DraftConversationMemory(TypedDict):
    """会话级 Python 内存；服务重启不保留，不与共享 Retriever 绑定。"""
    conversation_id: str  # 后端会话绑定，不由模型指定
    workspace_version: str
    recent_turns: list[DraftConversationTurn]  # 先按最近轮数和字符预算裁剪
    max_turns: int  # 首版建议 5 个已完成问答对，可配置
    max_context_chars: int  # 与证据预算分别控制


class DraftContextualQuestion(TypedDict):
    original_question: str
    resolved_question: str  # 结合历史解析“它”“上一年”等指代，不篡改用户意图
    referenced_turn_ids: list[str]
    needs_clarification: bool  # 指代不明确时询问，不盲猜公司、期间
