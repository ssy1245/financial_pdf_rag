import json
from pathlib import Path
import pytest
from financial_rag.evaluation.evidence_content import match_unit, number_pattern, score_content


def rule():
 return {'id':'q','document_id':'report','units':[{'id':'u1','label':'Capex 2022','pages':[63],'anchors':[{'phrases':['Capital spending']},{'number':'4625'}],'max_span':100,'values_after_label':True}]}
def chunk(text,page=63,doc='report',start=0):
 return {'chunk_id':f'{doc}:{page}:{start}','document_id':doc,'page':page,'text':text,'char_start':start,'char_end':start+len(text)}

def test_footer_and_table_title_are_not_hits():
 for t in ['Verizon 2022 Annual Report on Form 10-K 56','Consolidated cash flow statements 2022 2021 2020']:
  assert score_content([chunk(t)],rule())['hit']==0

def test_exact_amount_row_and_scope_required():
 for c in [chunk('Capital spending 4,625',page=62),chunk('Capital spending 4,625',doc='other'),chunk('Capital spending 14,625'),chunk('Capital spending 4,625.7'),chunk('Capital spending 4,626'),chunk('Other expense 4,625')]:
  assert score_content([c],rule())['hit']==0
 assert score_content([chunk('Capital spending (4,625) (4,100)')],rule())['all_units']==1

def test_only_returned_adjacent_spans_can_be_joined():
 first=chunk('Capital spending ');last=chunk('(4,625)',start=first['char_end'])
 assert score_content([last,first],rule())['hit']==1
 assert score_content([first],rule())['hit']==0
 last['char_start']+=20;last['char_end']+=20
 assert score_content([first,last],rule())['hit']==0
 last=chunk('(4,625)',page=64,start=first['char_end'])
 assert score_content([first,last],rule())['hit']==0

def test_label_and_unrelated_distant_amount_rejected():
 assert not match_unit('Capital spending '+'unrelated '*30+'4,625',rule()['units'][0])

def test_partial_evidence_is_not_complete():
 r=rule();r['units'].append({'id':'u2','label':'Revenue','pages':[64],'anchors':[{'phrases':['Net sales']},{'number':'34229'}],'max_span':100,'values_after_label':True})
 result=score_content([chunk('Capital spending 4,625')],r)
 assert result['hit']==1 and result['coverage']==0.5 and result['all_units']==0

def test_missing_value_requires_manual_review():
 r=dict(rule(),manual_required=True)
 result=score_content([chunk('Acquisitions, net of cash acquired')],r)
 assert result['hit'] is None and result['all_units'] is None

def test_negative_sign_must_be_present():
 r=rule();r['units'][0].update(anchors=[{'regex':r'Consumer\s*\(0\.9\)'}])
 assert score_content([chunk('Consumer (0.9)')],r)['hit']==1
 assert score_content([chunk('Consumer 0.9')],r)['hit']==0

def test_actual_previous_footer_false_hit_rejected():
 root=Path(__file__).resolve().parents[1]
 criteria=root/'outputs/embedding_ablation/financebench50_300_0_20261013/evidence_criteria.json'
 if not criteria.exists():pytest.skip('Local experimental reference data not installed')
 rows=json.loads(criteria.read_text());r=next(c for c in rows if c['id']=='financebench_id_00216')
 assert score_content([chunk('Verizon 2022 Annual Report on Form 10-K 56',56,r['document_id'])],r)['hit']==0


def test_factual_phrase_preserves_decimal_and_number_boundaries():
 r=rule();r['units'][0]['anchors']=[{'phrases':['13.2 billion in cash proceeds']}]
 assert score_content([chunk('13.2 billion in cash proceeds')],r)['hit']==1
 for text in ['132 billion in cash proceeds','113.2 billion in cash proceeds']:
  assert score_content([chunk(text)],r)['hit']==0
 r['units'][0]['anchors']=[{'phrases':['decrease of 44']}]
 assert score_content([chunk('a decrease of 44%')],r)['hit']==1
 for text in ['a decrease of 440%','a decrease of 44.5%']:
  assert score_content([chunk(text)],r)['hit']==0
