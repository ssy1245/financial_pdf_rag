from pathlib import Path
import pytest
from financial_rag.evaluation.paragraph_ablation import paragraph_chunks, check_spans


@pytest.fixture(scope='module')
def tokenizer():
    import financial_rag
    from transformers import AutoTokenizer
    path=Path(financial_rag.__file__).resolve().parents[2]/'models/bge-small-en-v1.5'
    if not path.exists():
        pytest.skip('Local tokenizer unavailable')
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    tok.model_max_length=10**9
    return tok


@pytest.mark.parametrize('pack',[True,False])
@pytest.mark.parametrize('cap',[20,40,300])
def test_coverage_source_and_long_paragraphs(tokenizer,pack,cap):
    text='Revenue was $50,171 million.\n\n'+('Receivables and inventory grew substantially. '*70)+'\n\n'+'Unpunctuated '*100+'\n\nFinal short paragraph.'
    pages=[dict(page=1,text=text),dict(page=2,text='Second page. 零值为0。')]
    chunks=paragraph_chunks(pages,'A',tokenizer,cap,pack)
    check_spans({'A':pages},chunks,tokenizer,cap)
    assert len({c['chunk_id'] for c in chunks})==len(chunks)
    assert chunks[-1]['page']==2
    assert all(c['char_start']<c['char_end'] for c in chunks)


def test_whole_paragraph_packing_and_no_normal_paragraph_splits(tokenizer):
    paragraphs=['Revenue grew '+('strongly '*8)+'.','Costs declined '+('slightly '*8)+'.','Profit increased '+('significantly '*8)+'.']
    text='\n\n'.join(paragraphs)
    pages=[dict(page=1,text=text)]
    single=paragraph_chunks(pages,'A',tokenizer,25,False)
    packed=paragraph_chunks(pages,'A',tokenizer,25,True)
    assert len(single)==3 and len(packed)==2
    assert packed[0]['paragraph_ids']==[0,1]
    for group in (single,packed):
        for paragraph in paragraphs:
            assert sum(paragraph in c['text'] for c in group)==1
    for c in single+packed:
        assert not c['split_long_paragraph']


def test_long_paragraph_prefers_sentence_end(tokenizer):
    sentence='Revenue rose strongly during the fiscal year.'
    text=sentence+' '+sentence+' '+sentence
    cap=len(tokenizer(sentence,add_special_tokens=False)['input_ids'])+2
    chunks=paragraph_chunks([dict(page=1,text=text)],'A',tokenizer,cap,True)
    assert len(chunks)==3
    assert all(c['text'].strip()==sentence for c in chunks)
    check_spans({'A':[dict(page=1,text=text)]},chunks,tokenizer,cap)


def test_empty_pages_and_invalid_budget(tokenizer):
    assert paragraph_chunks([dict(page=1,text='\n\n  ')],'A',tokenizer)==[]
    with pytest.raises(ValueError):paragraph_chunks([], 'A',tokenizer,0)
