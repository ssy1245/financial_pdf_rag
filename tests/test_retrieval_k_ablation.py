import pytest
from financial_rag.evaluation.retrieval_k_ablation import evaluate_ids, aggregate, paired_comparisons


def fixtures():
    text='Revenue 100. Profit 20.'
    chunks=[dict(chunk_id='a',document_id='D',page=1,text=text[:8],char_start=0,char_end=8,content_tokens=2),
            dict(chunk_id='b',document_id='D',page=1,text=text[8:],char_start=8,char_end=len(text),content_tokens=6),
            dict(chunk_id='c',document_id='D',page=1,text='Unrelated footer',char_start=100,char_end=116,content_tokens=3)]
    case=dict(financebench_id='q',doc_name='D',evidence=[dict(evidence_page_num=0)])
    criteria=dict(document_id='D',units=[dict(id='u1',label='Revenue',pages=[1],anchors=[dict(phrases=['Revenue']),dict(number='100')],max_span=40,values_after_label=True)],manual_required=False)
    return {c['chunk_id']:c for c in chunks},case,criteria


def test_source_spans_survive_fusion_lookup_and_adjacent_evidence():
    chunks,case,criteria=fixtures()
    # Evidence is split across two returned chunks; losing source offsets would undercount.
    first=evaluate_ids(['a'],chunks,case,criteria)
    complete=evaluate_ids(['a','b'],chunks,case,criteria)
    assert first['page']['hit']==1 and first['content']['coverage']==0
    assert complete['content']['coverage']==1 and complete['content_tokens']==8
    assert complete['annotated_support_chunk_fraction']==1
    # Gap-filled evidence must never be credited using unreturned text.
    assert evaluate_ids(['a','c'],chunks,case,criteria)['content']['coverage']==0


def test_manual_cases_do_not_dilute_content_denominator():
    chunks,case,criteria=fixtures()
    auto=evaluate_ids(['a','b'],chunks,case,criteria)
    manual=evaluate_ids(['c'],chunks,case,dict(criteria,manual_required=True))
    score=aggregate([auto,manual])
    assert score['questions']==2 and score['content_questions']==1
    assert score['content_coverage']==1 and score['page_hit']==1
    assert manual['annotated_support_chunk_fraction'] is None


def test_duplicates_rejected():
    chunks,case,criteria=fixtures()
    with pytest.raises(ValueError):evaluate_ids(['a','a'],chunks,case,criteria)


def test_support_fraction_is_not_page_precision():
    chunks,case,criteria=fixtures()
    score=evaluate_ids(['a','b','c'],chunks,case,criteria)
    assert score['page']['coverage']==1
    assert score['annotated_support_chunk_fraction']==pytest.approx(2/3)
