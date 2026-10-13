from pathlib import Path
import pytest
from financial_rag.evaluation.chunk_ablation import token_chunks, select_budget, metric

@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    path=Path(__file__).resolve().parents[1]/'models/bge-small-en-v1.5'
    if not path.exists(): pytest.skip('Local tokenizer unavailable')
    return AutoTokenizer.from_pretrained(path,local_files_only=True)

@pytest.mark.parametrize('size,overlap',[(20,0),(20,3),(40,6)])
def test_source_preservation_length_and_page_isolation(tokenizer,size,overlap):
    text='\uf0b7 FY2022: $50,171 total current liabilities.\n\n\uf0b7 '+('Long financial paragraph with receivables and inventory. '*25)+'\n\nFinal short paragraph.'
    chunks=token_chunks([{'page':1,'text':text},{'page':2,'text':'Second page.'}],'A',tokenizer,size,overlap)
    covered=set()
    for c in chunks:
        assert len(tokenizer(c['text'],add_special_tokens=False)['input_ids'])<=size
        if c['page']==1:
            assert c['text']==text[c['char_start']:c['char_end']]
            covered.update(range(c['char_start'],c['char_end']))
    assert all(i in covered for i,ch in enumerate(text) if not ch.isspace())
    assert len({c['chunk_id'] for c in chunks})==len(chunks)
    assert chunks[-1]['text']=='Second page.'


def test_budget_ranked_prefix_and_metrics():
    assert select_budget([0,1,2],[1500,700,100],2000)==([0],1500)
    chunks=[{'document_id':'A','page':1},{'document_id':'A','page':1},{'document_id':'B','page':1}]
    result=metric([0,1,2],chunks,{('B',1),('B',2)})
    assert result['coverage']==.5 and result['mrr']==pytest.approx(1/3)
    assert result['same_page_repeat_fraction']==pytest.approx(1/3)
