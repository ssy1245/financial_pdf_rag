from financial_rag.evaluation.batch_query_ablation import GRID, screen_indices, promote


def test_screening_balances_metadata_types_without_gold_labels():
    cases=[dict(question_type=t,answer='unused') for t in ('a','b','c') for _ in range(17)]
    selected=screen_indices(cases)
    assert len(selected)==12 and selected==screen_indices(cases)
    assert all(sum(cases[i]['question_type']==t for i in selected)==4 for t in ('a','b','c'))


def test_promotes_quality_and_call_efficiency_with_matched_k_quality_constraint():
    rows=[]
    for cfg in GRID:
        rows.append(dict(cfg,metrics=dict(errors=0,final_coverage=.5,sdk_calls_mean=3.,seconds_without_local_queue_mean=20.)))
    lookup={r['name']:r for r in rows}
    lookup['q3_k10']['metrics'].update(final_coverage=.8,sdk_calls_mean=4.)
    lookup['q2_k7']['metrics'].update(final_coverage=.6,sdk_calls_mean=2.)
    # This very cheap variant fails the same-K coverage constraint.
    lookup['q2_k5']['metrics'].update(final_coverage=.1,sdk_calls_mean=1.)
    result=promote(rows)
    assert result['coverage_choice']=='q3_k10'
    assert result['efficiency_choice']=='q2_k7'
    assert set(result['finalists'])=={'q1_k5','q1_k7','q1_k10','q3_k10','q2_k7'}
