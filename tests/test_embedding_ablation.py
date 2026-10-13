import numpy as np
from financial_rag.evaluation.embedding_ablation import rank_questions, summarize


def test_dense_ranking_document_identity_and_shared_budget():
    chunks=[{'chunk_id':str(i),'document_id':doc,'page':1,'content_tokens':n}
            for i,(doc,n) in enumerate([('wrong',1500),('gold',700),('gold',100)])]
    vectors=np.array([[1.,0.],[.8,.6],[.6,.8]])
    cases=[{'financebench_id':'q','company':'C','question':'Q','doc_name':'gold',
            'evidence':[{'doc_name':'gold','evidence_page_num':0}]}]
    rows=rank_questions(vectors,np.array([[1.,0.]]),chunks,cases)
    assert [r['chunk_id'] for r in rows[0]['ranking']]==['0','1','2']
    assert rows[0]['top5']['mrr']==.5
    assert rows[0]['budget2000']['coverage']==0
    assert rows[0]['budget2000']['used_tokens']==1500
    assert summarize(rows)['top5']['coverage']==1
