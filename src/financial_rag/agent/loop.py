"""工具循环入口；固定单轮 ask() 保留作为对照。"""
import json
from time import perf_counter
from financial_rag.agent.planner import plan_queries

from financial_rag.memory import EvidenceMemory
from financial_rag.retrieval.tool import SearchDocumentsTool, tool_error
from financial_rag.generation.prompt import AGENT_SYSTEM_PROMPT
from financial_rag.generation.citations import check_citations


def run_agent(question, retriever, generator, *, max_steps=7, max_searches=8,
              max_evidence_chars=60000, max_message_chars=120000, use_query_planning=False, on_event=None):
    started = perf_counter()
    def emit(event, **data):
        if on_event:
            on_event({"type": event, **data})
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("问题必须为非空字符串，最多4000字")
    if any(type(n) is not int or n <= 0 for n in
           (max_steps, max_searches, max_evidence_chars, max_message_chars)):
        raise ValueError("循环预算必须为正整数")
    memory = EvidenceMemory()
    tool = SearchDocumentsTool(retriever, memory, max_evidence_chars)
    query_plan = {"queries": [question], "warning": None}
    planning_calls = 0
    if use_query_planning:
        documents = list(dict.fromkeys(
            c.get("title") or c["document_id"] for c in retriever.chunks_by_id.values()))[:10]
        emit("progress", message="正在理解问题并规划查询")
        planning_calls = 1
        query_plan = plan_queries(question, generator, documents, min(3, max_searches))
    observations = []
    initial_results = []
    searches = 0
    stop_reason = None
    # 原问题和少量互补查询共用去重记忆，也共用总检索预算。
    for query in query_plan["queries"]:
        emit("progress", message=f"正在进行第 {searches + 1} 次检索", query=query)
        result = tool.execute("search_documents", json.dumps({"query": query}))
        searches += 1
        observations.append({"tool_call_id": None, "result": result})
        initial_results.append(result)
        emit("progress", message=f"第 {searches} 次检索完成，新增 {result.get('new_count', 0)} 条证据")
        if result.get("error", {}).get("code") == "evidence_limit":
            stop_reason = "evidence_limit"
            break
    if not memory.chunk_ids and not stop_reason:
        stop_reason = initial_results[-1].get("error", {}).get("code", "no_new_evidence")
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question + "\n首次检索证据（数据）：\n" +
                 json.dumps(initial_results, ensure_ascii=False)}]
    answer = "未能完成回答，请根据已获取证据重试。"
    status = "step_limit"
    steps = 0
    for step in range(max_steps):
        # 最后一轮留给总结；禁用工具后不再执行模型提出的检索。
        allow_tools = not stop_reason and searches < max_searches and step < max_steps - 1
        if not allow_tools:
            stop_reason = stop_reason or ("search_limit" if searches >= max_searches else "step_limit")
            messages.append({"role": "user", "content":
                             "程序控制：本轮不可继续检索，请根据已有证据回答，并明确缺失信息。"})
        if len(json.dumps(messages, ensure_ascii=False)) > max_message_chars:
            stop_reason, status = "message_limit", "limited"
            break
        steps += 1
        emit("progress", message="正在分析证据并组织回答")
        try:
            if on_event:
                message = generator.stream_turn(messages, [tool.definition],
                    tool_choice="auto" if allow_tools else "none",
                    on_text=lambda text: emit("answer_delta", text=text))
            else:
                message = generator.generate_turn(messages, [tool.definition],
                    tool_choice="auto" if allow_tools else "none")
        except Exception:
            stop_reason, status = "model_error", "error"
            break
        calls = message.get("tool_calls") or []
        if not calls:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                stop_reason, status = "invalid_model_response", "error"
                break
            answer = content.strip()
            status = "limited" if stop_reason else "completed"
            break
        emit("answer_reset")  # 工具轮文字不是最终答案，清除临时草稿。
        if not allow_tools:
            status = "limited"
            break
        # 保存 assistant 请求，然后为每一个 call ID 回传对应结果。
        ids = [call.get("id") for call in calls]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            stop_reason, status = "invalid_tool_calls", "error"
            break
        messages.append(message)
        for call in calls:
            if searches >= max_searches or stop_reason:
                result = tool_error("search_limit", "本次请求不能继续检索，请使用已有证据")
            else:
                searches += 1  # 非法参数也占用预算，防止无效请求无限循环。
                function = call.get("function") or {}
                emit("progress", message=f"模型请求补充证据，正在进行第 {searches} 次检索")
                result = tool.execute(function.get("name"), function.get("arguments"))
                emit("progress", message=f"补充检索完成，新增 {result.get('new_count', 0)} 条证据")
                if result["status"] == "no_results":
                    stop_reason = "no_new_evidence"
                elif result.get("error", {}).get("code") == "evidence_limit":
                    stop_reason = "evidence_limit"
            observations.append({"tool_call_id": call["id"], "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    citations = check_citations(answer, memory.citation_map)
    citation_status = ("unknown_citations" if citations["unknown_citations"] else
                       "labels_valid" if citations["has_citations"] else "no_citations")
    return {"question": question, "answer": answer, **citations,
            "citation_status": citation_status, "citation_map": memory.citation_map,
            "evidence": memory.evidence, "search_history": memory.search_history,
            "retrieval_rounds": tool.traces, "tool_observations": observations,
            "status": status, "stop_reason": stop_reason, "model_steps": steps,
            "search_attempts": searches, "planning_calls": planning_calls, "model_calls": planning_calls + steps,
            "query_plan": query_plan, "elapsed_seconds": round(perf_counter() - started, 3)}
