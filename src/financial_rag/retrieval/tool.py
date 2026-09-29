"""模型可见的检索工具；会话索引与已读 ID 由程序绑定。"""
import json
from copy import deepcopy


def tool_error(code, message):
    return {"status": "error", "evidence": [], "error": {"code": code, "message": message}}


class SearchDocumentsTool:
    def __init__(self, retriever, memory, max_evidence_chars=60000):
        self.retriever = retriever
        self.memory = memory
        self.max_evidence_chars = max_evidence_chars
        self.traces = []

    @property
    def definition(self):
        return {"type": "function", "function": {
            "name": "search_documents",
            "description": "检索当前资料库中的新证据。证据不足时，针对缺失信息提出具体查询。自动排除已获取片段。",
            "parameters": {"type": "object", "properties": {
                "missing_information": {"type": "string", "minLength": 1, "maxLength": 1000,
                    "description": "指出累计证据还缺少的必要信息，用一句话说明补查目的"},
                "query": {"type": "string", "minLength": 1, "maxLength": 4000},
                "top_k": {"type": "integer", "minimum": 1,
                          "maximum": self.retriever.candidate_k}},
                "required": ["query", "missing_information"], "additionalProperties": False}}}

    def execute(self, name, arguments_json, *, initial=False):
        if name != "search_documents":
            return tool_error("unknown_tool", "仅支持 search_documents")
        try:
            args = json.loads(arguments_json)
            if not isinstance(args, dict) or set(args) - {"query", "top_k", "missing_information"}:
                raise ValueError()
            gap = args.get("missing_information")
            if not initial and (not isinstance(gap, str) or not gap.strip() or len(gap) > 1000):
                raise ValueError()
            gap = None if initial else gap.strip()
            query = args.get("query")
            k = args.get("top_k", self.retriever.top_k)
            if (not isinstance(query, str) or not query.strip() or len(query) > 4000
                    or type(k) is not int or not 1 <= k <= self.retriever.candidate_k):
                raise ValueError()
        except (ValueError, TypeError):
            return tool_error("invalid_arguments", "需提供非空 missing_information（最多1000字）和 query（最多4000字），top_k 为候选上限内的正整数；不允许其他参数")
        try:
            trace = self.retriever.retrieve(query, top_k=k, exclude_chunk_ids=self.memory.chunk_ids)
            # 先试注册，超限时不把未交给模型的证据标记为已读。
            trial = deepcopy(self.memory)
            trial.record_search(query, trace["reranked"], self.retriever.chunks_by_id, gap)
            if len(trial.build_context()) > self.max_evidence_chars:
                return tool_error("evidence_limit", "累计证据超过字符预算，请根据已有证据回答并说明缺口")
            record = self.memory.record_search(query, trace["reranked"], self.retriever.chunks_by_id, gap)
            self.traces.append({"query": query, "missing_information": gap, **trace})
            evidence = self.memory.evidence
            return {"status": "ok" if record["new_count"] else "no_results", "query": query,
                    "evidence": [evidence[key] for key in record["new_chunk_ids"]],
                    "new_count": record["new_count"], "missing_information": gap,
                    "warnings": (["部分重排候选输入被截断"]
                                 if trace["reranker_truncated_chunk_ids"] else [])}
        except Exception:
            return tool_error("retrieval_failed", "检索失败，请稍后重试；不能据此判断文档没有答案")
