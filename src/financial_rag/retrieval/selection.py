"""Optional reranked-evidence selection; no generation or query rewriting.

Reranker scores are ranking signals, not probabilities of factual correctness.
MMR uses cached normalized document vectors to discourage semantic redundancy.
"""
import math
import numpy as np


def select_evidence(candidates, *, top_k=7, threshold=None, diversity_weight=None, vectors=None):
    """Return a ranked subset without backfilling below the score threshold.

diversity_weight is relevance weight lambda in lambda*score-(1-lambda)*similarity.
Larger lambda prioritizes relevance. It does not guarantee distinct facts.
"""
    if type(top_k) is not int or top_k<1:raise ValueError('top_k must be positive integer')
    if threshold is not None and not math.isfinite(threshold):raise ValueError('threshold must be finite')
    if diversity_weight is not None and not 0<=diversity_weight<=1:raise ValueError('relevance weight must be in [0,1]')
    ids=[c['chunk_id'] for c in candidates]
    if len(set(ids))!=len(ids):raise ValueError('duplicate chunk IDs')
    if any(not math.isfinite(c['rerank_score']) for c in candidates):raise ValueError('nonfinite score')
    eligible=sorted((c for c in candidates if threshold is None or c['rerank_score']>=threshold),key=lambda c:c['rerank_score'],reverse=True)
    if diversity_weight is None:return eligible[:top_k]
    if vectors is None:raise ValueError('MMR requires normalized document vectors')
    for c in eligible:
        v=np.asarray(vectors[c['chunk_id']])
        if v.ndim!=1 or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v),1,atol=1e-4):raise ValueError('invalid normalized vector')
    selected=[]
    while eligible and len(selected)<top_k:
        def score(c):
            overlap=max((float(np.dot(vectors[c['chunk_id']],vectors[s['chunk_id']])) for s in selected),default=0.)
            return diversity_weight*c['rerank_score']-(1-diversity_weight)*max(0.,overlap)
        best=max(range(len(eligible)),key=lambda i:score(eligible[i]))
        selected.append(eligible.pop(best))
    return selected
