import numpy as np
import pytest
from financial_rag.retrieval.selection import select_evidence


def rows():return [dict(chunk_id='a',rerank_score=.9),dict(chunk_id='b',rerank_score=.89),dict(chunk_id='c',rerank_score=.8)]


def test_threshold_returns_fewer_and_never_fills_irrelevant_slots():
    assert [c['chunk_id'] for c in select_evidence(rows(),threshold=.9)]==['a']
    assert select_evidence(rows(),threshold=.95)==[]


def test_mmr_can_select_complementary_candidate_over_duplicate():
    vec={'a':np.array([1.,0.]),'b':np.array([1.,0.]),'c':np.array([0.,1.])}
    assert [c['chunk_id'] for c in select_evidence(rows(),top_k=2,threshold=.5,diversity_weight=.6,vectors=vec)]==['a','c']
    assert [c['chunk_id'] for c in select_evidence(rows(),top_k=2,diversity_weight=1.,vectors=vec)]==['a','b']


def test_mmr_cannot_resurrect_candidate_below_threshold():
    vec={'a':np.array([1.,0.]),'b':np.array([1.,0.]),'c':np.array([0.,1.])}
    assert [c['chunk_id'] for c in select_evidence(rows(),threshold=.85,diversity_weight=.6,vectors=vec)]==['a','b']


def test_distinct_year_evidence_is_not_hard_deleted_by_vector_similarity():
    vec={'a':np.array([1.,0.]),'b':np.array([1.,0.]),'c':np.array([0.,1.])}
    assert len(select_evidence(rows(),top_k=3,diversity_weight=.6,vectors=vec))==3


@pytest.mark.parametrize('kwargs',[dict(threshold=float('nan')),dict(top_k=0),dict(diversity_weight=1.1),dict(diversity_weight=.6,vectors={c:np.array([2.,0.]) for c in ['a','b','c']})])
def test_invalid_selection_parameters_fail(kwargs):
    with pytest.raises(ValueError):select_evidence(rows(),**kwargs)
