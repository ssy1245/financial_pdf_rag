"""Dense / BM25 / Hybrid / Hybrid+Reranker 批量对比。

文档编码、BM25 索引和重排模型各准备一次。两路召回后 RRF 保留 candidate_k 条，
再重排取 top_k；未重排的 Hybrid 基线取同一 RRF 候选前 top_k。
默认 CLI 执行四路；--skip-reranker 可仅跑三路。记录重排模型配置和超限候选 ID。
不计算相关性指标；重排分数不是正确概率，原始全文不代表全部进入模型。
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from financial_rag.indexing.bm25_index import BM25Index
from financial_rag.retrieval.dense import dense_search
from financial_rag.retrieval.sparse import bm25_search
from financial_rag.retrieval.fusion import reciprocal_rank_fusion


def compare_methods(chunks, questions, embedding_model, top_k=5, candidate_k=20, rrf_k=60, reranker=None):
    """模型从外部传入，便于复用和测试；文档编码与建索引均在循环外。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if candidate_k < top_k:
        raise ValueError("candidate_k 必须大于等于 top_k")
    if rrf_k < 0:
        raise ValueError("rrf_k 不能为负数")
    if not chunks:
        raise ValueError("chunks 不能为空，请先生成分块文件")
    if len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("chunk_id 必须唯一")
    if not questions:
        raise ValueError("问题列表不能为空")

    # 1. 仅准备一次文档向量和 BM25 索引。
    chunk_embeddings = embedding_model.encode_documents([c["text"] for c in chunks])
    bm25_index = BM25Index(chunks)
    comparisons = []

    # 2. 同一个 query 召回两路候选，再执行 RRF。
    for case in questions:
        query = case["question"]
        query_embedding = embedding_model.encode_query(query)
        dense_results = dense_search(query_embedding, chunk_embeddings, chunks, candidate_k)
        bm25_results = bm25_search(query, bm25_index, candidate_k)
        # 使用完整候选融合，展示时才截取各方法的 Top-K。
        hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results, k=rrf_k, top_k=candidate_k)
        comparisons.append({
            "id": case["id"],
            "question": query,
            "hybrid": [{"rank": rank, **item} for rank, item in enumerate(hybrid_results[:top_k], 1)],
            "candidates": {"dense": dense_results, "bm25": bm25_results, "rrf": hybrid_results},
            "dense": [{"rank": rank, **item} for rank, item in enumerate(dense_results[:top_k], 1)],
            "bm25": [{"rank": rank, **item} for rank, item in enumerate(bm25_results[:top_k], 1)],
        })

        if reranker is not None:
            reranked = reranker.rerank(query, hybrid_results, top_k=top_k)
            comparisons[-1]["reranked"] = [{**item, "rank": rank} for rank, item in enumerate(reranked, 1)]
            comparisons[-1]["reranker_truncated_chunk_ids"] = list(reranker.last_truncated_chunk_ids)

    return {
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "chunk_count": len(chunks),
        "bm25_config": {"k1": bm25_index.k1, "b": bm25_index.b,
                        "tokenizer": "lowercase + [a-z0-9]+", "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))"},
        "note": "人工对比证据与排名；两路分数不可直接比较。无标准证据标注，不计算 Recall/MRR。BM25 允许少于 K 条。",
        "queries": comparisons,
    }


def print_comparison(report):
    for case in report["queries"]:
        print("\n" + "=" * 80)
        print(f"Question [{case['id']}]: {case['question']}")
        for method in ("dense", "bm25", "hybrid", "reranked"):
            if method not in case:
                continue
            print(f"\n--- {method.upper()} Top-{report['top_k']} ---")
            if not case[method]:
                print("没有匹配结果")
            for result in case[method]:
                score_label = (
                    f"RRF: {result['rrf_score']:.6f} | Dense rank: {result['dense_rank']} | BM25 rank: {result['bm25_rank']}"
                    if method in ("hybrid", "reranked") else f"Score: {result['score']:.4f}"
                )
                if method == "reranked":
                    score_label = f"Rerank: {result['rerank_score']:.4f} | RRF: {result['rrf_score']:.6f}"
                print(f"\nRank: {result['rank']} | {score_label} | PDF Page: {result['page']} | Chunk: {result['chunk_id']}")
                print(result["text"])
                print("[END OF EVIDENCE]")


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="同题 Dense/BM25/Hybrid 批量检索对比")
    parser.add_argument("--chunks", type=Path, default=root / "data/chunks/apple-10k_chunks.json")
    parser.add_argument("--questions", type=Path, default=root / "data/eval/dense_questions.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20, help="每路召回数量")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑常数")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--reranker-max-length", type=int, default=512)
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args()
    if args.reranker_max_length <= 0:
        parser.error("--reranker-max-length 必须大于 0")
    if args.output is None:
        name = "dense_bm25_hybrid_comparison.json" if args.skip_reranker else "dense_bm25_hybrid_reranked_comparison.json"
        args.output = root / "outputs" / name
    if args.top_k <= 0:
        parser.error("--top-k 必须大于 0")

    if args.candidate_k < args.top_k or args.rrf_k < 0:
        parser.error("要求 candidate-k >= top-k 且 rrf-k >= 0")

    chunk_bytes = args.chunks.read_bytes()
    question_bytes = args.questions.read_bytes()
    chunks = json.loads(chunk_bytes)
    questions = json.loads(question_bytes)
    from financial_rag.indexing.embeddings import EmbeddingModel
    model = EmbeddingModel(model_name=args.model)
    reranker = None
    if not args.skip_reranker:
        from financial_rag.retrieval.reranker import Reranker
        reranker = Reranker(args.reranker_model, max_length=args.reranker_max_length)
    report = compare_methods(chunks, questions, model, args.top_k, args.candidate_k, args.rrf_k, reranker)
    report["reranker_config"] = None if reranker is None else {
        "model": reranker.model_name, "max_length": reranker.max_length,
    }
    report.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "normalize_embeddings": True,
        "chunks_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
        "questions_sha256": hashlib.sha256(question_bytes).hexdigest(),
    })
    # 3. 三路结果与完整候选保存到新报告，保留原来的双方法基线。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print_comparison(report)
    print(f"\n完成 {len(questions)} 题对比，结果保存到：{args.output}")


if __name__ == "__main__":
    main()
