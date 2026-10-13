import pytest
from financial_rag.evaluation.financebench import gold_pages, page_metrics, evaluation_question, select_cases


def test_gold_pages_zero_based_and_document_identity():
    case={'doc_name':'A','evidence':[{'doc_name':'B','evidence_page_num':0},
                                  {'evidence_doc_name':'B','evidence_page_num':0}]}
    assert gold_pages(case)=={('B',1)}
    case['evidence'][0]['evidence_page_num']=-1
    with pytest.raises(ValueError): gold_pages(case)


def test_page_metrics_do_not_confuse_documents_or_duplicate_chunks():
    chunks={'x':{'document_id':'A','page':1},'y':{'document_id':'B','page':1},
            'z':{'document_id':'B','page':1}}
    metrics=page_metrics([{'chunk_id':x} for x in ['x','y','z']],chunks,{('B',1),('B',2)})
    assert metrics['annotated_page_coverage']==0.5
    assert metrics['reciprocal_rank_by_page']==0.5
    assert metrics['page_hit'] and not metrics['all_annotated_pages_hit']
    assert page_metrics([],chunks,{('B',1)})['page_hit'] is False


def test_question_does_not_leak_reference_answer_or_page():
    case={'question':'Revenue?','doc_name':'A_2022','answer':'SECRET GOLD',
          'evidence':[{'evidence_page_num':123456,'evidence_text':'SECRET EVIDENCE'}]}
    assert evaluation_question(case)=='Revenue?\nTarget report: A_2022.pdf'


def test_selection_is_reproducible_and_balances_types():
    kinds=['metrics-generated','domain-relevant','novel-generated']
    rows=[{'company':str(i),'question_type':kind} for i in range(6) for kind in kinds]
    selected=select_cases(rows,6)
    assert selected==select_cases(rows,6)
    assert len({r['company'] for r in selected})==6
    assert [r['question_type'] for r in selected]==kinds*2
