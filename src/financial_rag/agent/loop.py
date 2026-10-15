"""工具循环入口；固定单轮 ask() 保留作为对照。"""
import json
from time import perf_counter
from financial_rag.agent.planner import plan_queries, plan_batch_queries

from financial_rag.memory import EvidenceMemory
from financial_rag.retrieval.tool import SearchDocumentsTool, tool_error
from financial_rag.generation.prompt import AGENT_SYSTEM_PROMPT
from financial_rag.generation.citations import check_citations


def run_agent(question, retriever, generator, *, max_steps=7, max_searches=8,
              max_evidence_chars=60000, max_message_chars=120000, use_query_planning=False, on_event=None, conversation_context=None,
              initial_query_count=1, initial_evidence_k=None):
    started = perf_counter()
    def emit(event, **data):
        if on_event:
            on_event({"type": event, **data})
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("问题必须为非空字符串，最多4000字")
    if any(type(n) is not int or n <= 0 for n in
           (max_steps, max_searches, max_evidence_chars, max_message_chars)):
        raise ValueError("循环预算必须为正整数")
    if conversation_context and not use_query_planning:
        raise ValueError("会话上下文需要启用查询规划")
    if initial_query_count is not None and (type(initial_query_count) is not int or not 1 <= initial_query_count <= 4):
        raise ValueError('initial_query_count must be None (adaptive) or 1..4')
    if initial_query_count != 1 and (not use_query_planning or conversation_context):
        raise ValueError('Batch initial planning requires an independent question and query planning')
    if initial_evidence_k is not None and (type(initial_evidence_k) is not int or not 1 <= initial_evidence_k <= retriever.candidate_k):
        raise ValueError('Invalid initial evidence budget')
    memory = EvidenceMemory()
    tool = SearchDocumentsTool(retriever, memory, max_evidence_chars)
    query_plan = {"queries": [question], "warning": None}
    planning_calls = 0
    if use_query_planning:
        documents = list(dict.fromkeys(
            c.get("title") or c["document_id"] for c in retriever.chunks_by_id.values()))[:10]
        emit("progress", message="正在理解问题并规划查询")
        planning_calls = 1
        query_plan = (plan_queries(question, generator, documents, conversation_context) if initial_query_count == 1
                      else plan_batch_queries(question, generator, documents, initial_query_count))
    resolved_question = query_plan.get("resolved_question", question)
    if query_plan.get("clarification"):
        answer = query_plan["clarification"]
        emit("answer_delta", text=answer)
        return {"question": question, "resolved_question": question, "answer": answer,
                **check_citations(answer, {}), "citation_status": "no_citations",
                "citation_map": {}, "evidence": {}, "search_history": [],
                "retrieval_rounds": [], "tool_observations": [], "status": "clarification",
                "stop_reason": "needs_clarification", "model_steps": 0,
                "search_attempts": 0, "planning_calls": planning_calls, "model_calls": planning_calls,
                "query_plan": query_plan, "elapsed_seconds": round(perf_counter() - started, 3)}
    observations = []
    initial_results = []
    searches = 0
    stop_reason = None
    # Single-query defaults preserve the old loop. Batch mode shares a TOTAL
    # initial chunk budget, distributed by quotient/remainder across queries.
    total_k = initial_evidence_k or retriever.top_k
    initial_queries = query_plan['queries'][:min(max_searches,total_k)]
    no_progress = 0
    for index,query in enumerate(initial_queries):
        quota=total_k//len(initial_queries)+int(index<total_k%len(initial_queries))
        emit('progress',message=f'正在进行第 {index+1} 次初始检索',query=query)
        args={'query':query}
        if len(initial_queries)>1 or initial_evidence_k is not None:
            args['top_k']=quota
        first=tool.execute('search_documents',json.dumps(args),initial=True)
        searches+=1
        observations.append({'tool_call_id':None,'result':first})
        initial_results.append(first)
        emit('progress',message=f'初始检索完成，新增 {first.get("new_count",0)} 条证据')
        if first['status']=='ok':no_progress=0
        else:no_progress+=1
        if first.get('error',{}).get('code')=='evidence_limit':
            stop_reason='evidence_limit';break
    initial_searches=searches
    initial_chunk_ids=list(memory.evidence)
    # 一次空结果不代表文档无答案，允许模型换一种查询；连续无新增两次才停止。
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question + ("\n结合会话补全的问题：" + resolved_question if conversation_context else "") + "\n首次检索证据（数据）：\n" +
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
                if result["status"] == "ok":
                    no_progress = 0
                    emit("progress", message="补查依据：" + result["missing_information"],
                         query=result["query"])
                elif result["status"] == "no_results":
                    no_progress += 1
                elif result.get("error", {}).get("code") == "evidence_limit":
                    stop_reason = "evidence_limit"
            observations.append({"tool_call_id": call["id"], "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
        # 批内空结果不阻止后续查询；整批完成后再判断是否无进展。
        if no_progress >= 2 and not stop_reason:
            stop_reason = "no_new_evidence"
    citations = check_citations(answer, memory.citation_map)
    citation_status = ("unknown_citations" if citations["unknown_citations"] else
                       "labels_valid" if citations["has_citations"] else "no_citations")
    return {"question": question, "resolved_question": resolved_question, "answer": answer, **citations,
            "citation_status": citation_status, "citation_map": memory.citation_map,
            "evidence": memory.evidence, "search_history": memory.search_history,
            "retrieval_rounds": tool.traces, "tool_observations": observations,
            "status": status, "stop_reason": stop_reason, "model_steps": steps,
            "search_attempts": searches, "planning_calls": planning_calls, "model_calls": planning_calls + steps,
            "initial_searches":initial_searches,"initial_chunk_ids":initial_chunk_ids,
            "initial_queries_not_executed":query_plan['queries'][initial_searches:],
            "initial_query_execution_capacity":min(max_searches,total_k),
            "query_plan": query_plan, "elapsed_seconds": round(perf_counter() - started, 3)}
