import json
from unittest.mock import Mock
import pytest
from financial_rag.agent.planner import plan_batch_queries
from financial_rag.agent.loop import run_agent


class Retriever:
    candidate_k=20
    top_k=7
    def __init__(self):
        self.chunks_by_id={str(i):dict(chunk_id=str(i),document_id='D',page=1,text=f'fact {i}') for i in range(40)}
        self.requests=[]
    def retrieve(self,query,top_k,exclude_chunk_ids):
        self.requests.append((query,top_k,set(exclude_chunk_ids)))
        rows=[v for k,v in self.chunks_by_id.items() if k not in exclude_chunk_ids][:top_k]
        return dict(reranked=rows,reranker_truncated_chunk_ids=[])


def model(queries):
    m=Mock()
    m.generate.return_value=json.dumps(dict(queries=queries))
    m.generate_turn.return_value=dict(role='assistant',content='Answer [E1]')
    return m


def test_initial_batch_shares_total_budget_and_calls_model_only_after_all_searches():
    r=Retriever();m=model(['assets','liabilities','inventory'])
    def answer(*a,**kw):
        assert len(r.requests)==3
        return dict(role='assistant',content='Answer [E1]')
    m.generate_turn.side_effect=answer
    result=run_agent('ratio?',r,m,use_query_planning=True,initial_query_count=3)
    assert [x[1] for x in r.requests]==[3,2,2]
    assert len(result['evidence'])==7 and result['model_calls']==2
    assert result['initial_searches']==3 and len(result['initial_chunk_ids'])==7
    assert r.requests[1][2]=={'0','1','2'}


def test_simple_question_can_plan_one_query():
    r=Retriever();m=model(['revenue'])
    result=run_agent('revenue?',r,m,use_query_planning=True,initial_query_count=3)
    assert len(r.requests)==1 and r.requests[0][1]==7
    assert result['initial_searches']==1


def test_initial_batch_respects_global_search_budget():
    r=Retriever();m=model(['a','b','c'])
    result=run_agent('q',r,m,use_query_planning=True,initial_query_count=3,max_searches=2)
    assert len(r.requests)==2 and sum(x[1] for x in r.requests)==7
    assert m.generate_turn.call_args.kwargs['tool_choice']=='none'
    assert result['stop_reason']=='search_limit'


def test_deduplicates_planned_queries_before_allocating_budget():
    m=model(['Revenue 2025',' revenue  2025 ','Profit'])
    result=plan_batch_queries('q',m,[],3)
    assert result['queries']==['Revenue 2025','Profit']


@pytest.mark.parametrize('raw',['broken','{"queries":[]}','{"queries":[3]}','{"queries":["a","b","c"]}'])
def test_invalid_batch_plan_falls_back_without_extra_model_call(raw):
    m=Mock();m.generate.return_value=raw
    result=plan_batch_queries('original',m,[],2)
    assert result['queries']==['original'] and result['warning']
    assert m.generate.call_count==1


def test_batch_mode_rejects_unsupported_history_instead_of_ignoring_it():
    with pytest.raises(ValueError):
        run_agent('q',Retriever(),model(['a']),use_query_planning=True,initial_query_count=2,conversation_context=[{'question':'history'}])


def test_original_single_query_policy_is_preserved():
    r=Retriever();m=model([]);m.generate.return_value='{"query":"single"}'
    result=run_agent('q',r,m,use_query_planning=True)
    assert result['query_plan']['queries']==['single'] and len(r.requests)==1
    assert r.requests[0][1]==7 and result['model_calls']==2
