import json
import pytest
from test_batch_planning import Retriever, model
from financial_rag.agent.planner import plan_batch_queries
from financial_rag.agent.loop import run_agent


def test_adaptive_plan_accepts_more_than_old_four_query_limit():
    queries=[f'fact {i}' for i in range(10)]
    m=model(queries)
    result=plan_batch_queries('q',m,[],None)
    assert result['queries']==queries and result['warning'] is None
    prompt=m.generate.call_args.args[0][0]['content']
    assert '自行决定' in prompt and '生成1到' not in prompt


def test_adaptive_execution_keeps_global_budget_and_reports_unexecuted_queries():
    r=Retriever();m=model([f'fact {i}' for i in range(10)])
    result=run_agent('q',r,m,use_query_planning=True,initial_query_count=None,initial_evidence_k=8)
    assert len(r.requests)==8 and all(req[1]==1 for req in r.requests)
    assert result['initial_queries_not_executed']==['fact 8','fact 9']
    assert result['initial_query_execution_capacity']==8
    assert result['stop_reason']=='search_limit'
    assert result['model_calls']==2


def test_adaptive_initial_and_followup_evidence_budgets_are_independent():
    r=Retriever();m=model(['revenue','growth','segment'])
    m.generate_turn.side_effect=[
        dict(role='assistant',content=None,tool_calls=[dict(id='call1',type='function',function=dict(name='search_documents',arguments=json.dumps(dict(query='missing',missing_information='unit'))))]),
        dict(role='assistant',content='Answer [E1]')]
    result=run_agent('q',r,m,use_query_planning=True,initial_query_count=None,initial_evidence_k=16)
    assert [req[1] for req in r.requests]==[6,5,5,7]
    assert len(result['initial_chunk_ids'])==16 and result['search_attempts']==4
    assert result['initial_queries_not_executed']==[]


def test_adaptive_deduplicates_and_invalid_plan_falls_back():
    assert plan_batch_queries('q',model(['a',' A ','b']),[],None)['queries']==['a','b']
    result=plan_batch_queries('q',model([]),[],None)
    assert result['queries']==['q'] and result['warning']


@pytest.mark.parametrize('kwargs',[dict(use_query_planning=False),dict(use_query_planning=True,conversation_context=[{'question':'prior'}])])
def test_adaptive_requires_independent_question_with_planning(kwargs):
    with pytest.raises(ValueError):
        run_agent('q',Retriever(),model(['a']),initial_query_count=None,**kwargs)
