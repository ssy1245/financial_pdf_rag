import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from financial_rag.agent.loop import run_agent
from financial_rag.memory import EvidenceMemory
from financial_rag.retrieval.tool import SearchDocumentsTool
from financial_rag.generation.generator import DeepSeekGenerator


class Retriever:
    top_k = 1
    candidate_k = 3

    def __init__(self):
        self.chunks_by_id = {key: dict(chunk_id=key, document_id="doc", page=i,
                                      text=f"evidence {key}")
                             for i, key in enumerate("abc", 1)}
        self.exclusions = []

    def retrieve(self, query, top_k, exclude_chunk_ids):
        self.exclusions.append(set(exclude_chunk_ids))
        rows = [c for key, c in self.chunks_by_id.items() if key not in exclude_chunk_ids][:top_k]
        return {"reranked": rows, "reranker_truncated_chunk_ids": []}


def call(id="call1", name="search_documents", args='{"query":"missing"}'):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


def turn(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls),
            "reasoning_content": "preserved provider field"}


class Model:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    def generate_turn(self, messages, tools, tool_choice):
        self.requests.append((deepcopy(messages), tool_choice))
        return self.turns.pop(0)


def answer(text="answer [E1]"):
    return {"role": "assistant", "content": text}


def test_two_rounds_preserve_evidence_and_pair_tool_id():
    retriever = Retriever()
    model = Model(turn(call()), answer("answer [E1] [E2]"))
    result = run_agent("question", retriever, model)
    assert result["status"] == "completed"
    assert retriever.exclusions == [set(), {"a"}]
    assert result["citation_map"]["E2"]["chunk_id"] == "b"
    assert len(result["sources"]) == 2
    messages = model.requests[1][0]
    assert messages[-1]["tool_call_id"] == "call1"
    assert messages[-2]["reasoning_content"] == "preserved provider field"
    assert "evidence a" in messages[1]["content"]
    assert "evidence b" in messages[-1]["content"]


def test_direct_answer_still_has_initial_retrieval():
    result = run_agent("q", Retriever(), Model(answer()))
    assert result["model_steps"] == 1
    assert len(result["search_history"]) == 1


@pytest.mark.parametrize("name,args,code", [
    ("shell", '{}', "unknown_tool"),
    ("search_documents", 'broken', "invalid_arguments"),
    ("search_documents", '{"query":"x", "top_k":true}', "invalid_arguments"),
    ("search_documents", '{"query":"x", "db":"secret"}', "invalid_arguments"),
])
def test_tool_validation(name, args, code):
    memory = EvidenceMemory()
    result = SearchDocumentsTool(Retriever(), memory).execute(name, args)
    assert result["error"]["code"] == code
    assert not memory.chunk_ids


def test_invalid_call_can_be_corrected_and_multiple_calls_are_paired():
    model = Model(turn(call("bad", args="{}")), turn(call("x"), call("y")), answer("[E3]"))
    result = run_agent("q", Retriever(), model)
    assert result["status"] == "completed"
    tools = [m for m in model.requests[-1][0] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["bad", "x", "y"]
    assert result["sources"][0]["chunk_id"] == "c"


def test_search_limit_disables_tools_and_reports_limited():
    model = Model(turn(call()), answer())
    result = run_agent("q", Retriever(), model, max_searches=2)
    assert model.requests[-1][1] == "none"
    assert result["stop_reason"] == "search_limit"
    assert result["status"] == "limited"


def test_step_limit_and_disobedient_model_do_not_execute_tool():
    retriever = Retriever()
    result = run_agent("q", retriever, Model(turn(call())), max_steps=1)
    assert len(retriever.exclusions) == 1
    assert result["status"] == "limited"


def test_no_new_results_stops_and_preserves_registered_evidence():
    model = Model(turn(call(args='{"query":"x","top_k":3}')), turn(call()), answer())
    result = run_agent("q", Retriever(), model)
    assert result["stop_reason"] == "no_new_evidence"
    assert len(result["evidence"]) == 3
    assert model.requests[-1][1] == "none"


def test_evidence_budget_does_not_mark_undelivered_chunks_as_read():
    memory = EvidenceMemory()
    tool = SearchDocumentsTool(Retriever(), memory, max_evidence_chars=1)
    assert tool.execute("search_documents", '{"query":"x"}')["error"]["code"] == "evidence_limit"
    assert memory.chunk_ids == set()


def test_retrieval_errors_are_safe():
    retriever = Retriever()
    retriever.retrieve = Mock(side_effect=RuntimeError("secret path"))
    result = SearchDocumentsTool(retriever, EvidenceMemory()).execute("search_documents", '{"query":"x"}')
    assert result["error"]["code"] == "retrieval_failed"
    assert "secret" not in json.dumps(result)


def test_message_budget_and_model_failure():
    model = Model(answer())
    result = run_agent("q", Retriever(), model, max_message_chars=1)
    assert result["stop_reason"] == "message_limit"
    assert not model.requests
    model.generate_turn = Mock(side_effect=RuntimeError("secret"))
    result = run_agent("q", Retriever(), model)
    assert result["status"] == "error"
    assert result["stop_reason"] == "model_error"


def test_provider_adapter_preserves_tools_and_reasoning():
    generator = DeepSeekGenerator.__new__(DeepSeekGenerator)
    generator.model = "test"
    message = Mock()
    message.model_dump.return_value = {**turn(call()), "refusal": None}
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    generator.client = Mock()
    generator.client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    result = generator.generate_turn([], [{"type": "function"}])
    assert result == turn(call())
    assert generator.client.chat.completions.create.call_args.kwargs["tool_choice"] == "auto"
    choice.finish_reason = "length"
    with pytest.raises(RuntimeError):
        generator.generate_turn([], [])


def test_query_planning_keeps_original_deduplicates_and_counts_calls():
    model = Model(answer('[E1] [E2] [E3]'))
    model.generate = Mock(return_value='{"queries":["apple net sales","APPLE NET SALES","nvidia revenue","extra"]}')
    result = run_agent('q', Retriever(), model, use_query_planning=True)
    assert result['query_plan']['queries'] == ['q', 'apple net sales', 'nvidia revenue']
    assert len(result['evidence']) == 3
    assert result['model_calls'] == 2
    assert result['elapsed_seconds'] >= 0
    assert result['search_history'][0]['query'] == 'q'


@pytest.mark.parametrize('raw', ['invalid', '{"queries":"bad"}', '{"queries":[3]}'])
def test_planning_failure_falls_back(raw):
    model = Model(answer())
    model.generate = Mock(return_value=raw)
    result = run_agent('q', Retriever(), model, use_query_planning=True)
    assert result['query_plan']['queries'] == ['q']
    assert result['query_plan']['warning']
    assert result['status'] == 'completed'


def test_query_plan_respects_total_search_budget():
    model = Model(answer())
    model.generate = Mock(return_value='{"queries":["a","b"]}')
    result = run_agent('q', Retriever(), model, use_query_planning=True, max_searches=1)
    assert len(result['search_history']) == 1
    assert model.requests[0][1] == 'none'


def test_stream_adapter_assembles_arguments_and_emits_only_content():
    generator = DeepSeekGenerator.__new__(DeepSeekGenerator)
    generator.model = 'test'
    def chunk(text=None, tools=None, reason=None, finish=None):
        delta = SimpleNamespace(content=text, tool_calls=tools, reasoning_content=reason)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])
    def part(index, id=None, name=None, args=None):
        return SimpleNamespace(index=index,id=id,function=SimpleNamespace(name=name,arguments=args))
    chunks = [chunk(reason='private', tools=[part(0,'c','search_documents','{"query":')]),
              chunk(tools=[part(0,args='"revenue"}')], finish='tool_calls')]
    class Stream:
        closed = False
        def __iter__(self): return iter(chunks)
        def close(self): self.closed = True
    stream = Stream()
    generator.client = Mock()
    generator.client.chat.completions.create.return_value = stream
    texts = []
    result = generator.stream_turn([], [], on_text=texts.append)
    assert result['tool_calls'][0]['function']['arguments'] == '{"query":"revenue"}'
    assert result['reasoning_content'] == 'private'
    assert not texts and stream.closed
    chunks[:] = [chunk(text='hello'),chunk(text=' world',finish='stop')]
    assert generator.stream_turn([], [], on_text=texts.append)['content'] == 'hello world'
    assert texts == ['hello',' world']
    chunks[:] = [chunk(text='partial')]
    with pytest.raises(RuntimeError): generator.stream_turn([], [])


def test_stream_loop_reports_progress_and_resets_tool_round_draft():
    model = Model(turn(call()), answer('[E1] [E2]'))
    def stream(messages, tools, tool_choice, on_text):
        result = model.generate_turn(messages, tools, tool_choice)
        on_text(result.get('content') or 'temporary')
        return result
    model.stream_turn = stream
    events = []
    result = run_agent('q', Retriever(), model, on_event=events.append)
    assert result['status'] == 'completed'
    assert any(e['type']=='answer_reset' for e in events)
    assert any('补充证据' in e.get('message','') for e in events)
    assert events[-1] == {'type':'answer_delta','text':'[E1] [E2]'}
