"""实时单轮问答：SQLite → Dense/BM25 → RRF → 重排 → 生成 → 引用检查。

FinancialRAG 实例复用已加载的向量、BM25 和模型；不重新编码文档。
入库独立使用 index_demo.py build。引用检查仅验证编号，不验证事实支持。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from financial_rag.indexing.bm25_index import BM25Index
from financial_rag.indexing.vector_store import VectorStore
from financial_rag.retrieval.dense import dense_search
from financial_rag.retrieval.sparse import bm25_search
from financial_rag.retrieval.fusion import reciprocal_rank_fusion
from financial_rag.generation.citations import prepare_evidence, check_citations
from financial_rag.generation.prompt import build_messages


class FinancialRAG:
    def __init__(self, store, embedding_model, embedding_model_name,
                 reranker, generator, candidate_k=20, top_k=5, rrf_k=60):
        if top_k <= 0 or candidate_k < top_k or rrf_k < 0:
            raise ValueError("要求 candidate_k >= top_k > 0，rrf_k >= 0")
        self.chunks, self.vectors = store.load(embedding_model_name)
        self.chunks_by_id = {c["chunk_id"]: c for c in self.chunks}
        if len(self.chunks_by_id) != len(self.chunks):
            raise ValueError("chunk_id 必须唯一")
        self.bm25 = BM25Index(self.chunks)
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.generator = generator
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, question):
        """返回各阶段候选，方便区分召回和重排的问题。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        query_vector = np.asarray(self.embedding_model.encode_query(question))
        if query_vector.shape != (self.vectors.shape[1],):
            raise ValueError("查询向量维度与索引不一致")
        if not np.isfinite(query_vector).all() or not np.isclose(
            np.linalg.norm(query_vector), 1.0, atol=1e-4
        ):
            raise ValueError("查询向量必须有限且已经归一化")
        dense = dense_search(query_vector, self.vectors, self.chunks, self.candidate_k)
        bm25 = bm25_search(question, self.bm25, self.candidate_k)
        hybrid = reciprocal_rank_fusion(dense, bm25, k=self.rrf_k, top_k=self.candidate_k)
        reranked = self.reranker.rerank(question, hybrid, top_k=self.top_k)
        return {
            "dense": dense, "bm25": bm25, "hybrid": hybrid,
            "reranked": reranked,
            "reranker_truncated_chunk_ids": list(self.reranker.last_truncated_chunk_ids),
        }

    def ask(self, question):
        retrieval = self.retrieve(question)
        if not retrieval["reranked"]:
            raise ValueError("没有可用于生成的证据")
        context, citation_map = prepare_evidence(retrieval["reranked"], self.chunks_by_id)
        answer = self.generator.generate(build_messages(question, context))
        citations = check_citations(answer, citation_map)
        # 未知引用保留在报告中并明确标记，不静默修复，也不声称验证事实。
        status = ("unknown_citations" if citations["unknown_citations"] else
                  "labels_valid" if citations["has_citations"] else "no_citations")
        return {
            "question": question, "answer": answer,
            **citations, "citation_status": status,
            "citation_map": citation_map, "retrieval": retrieval,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[2] / "storage/vectors.sqlite3")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-max-length", type=int, default=2048)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--output", type=Path, help="保存完整检索过程与回答 JSON")
    args = parser.parse_args()
    if not args.question.strip() or not (args.candidate_k >= args.top_k > 0) or args.rrf_k < 0 or args.reranker_max_length <= 0:
        parser.error("检查非空问题、candidate_k >= top_k > 0、rrf_k >= 0 和正数 max_length")
    if not args.db.is_file():
        parser.error("索引不存在，请先运行 uv run index_demo.py build")
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    from financial_rag.generation.generator import DeepSeekGenerator
    generator = DeepSeekGenerator(args.model)
    rag = FinancialRAG(
        store=VectorStore(args.db),
        embedding_model=EmbeddingModel(args.embedding_model),
        embedding_model_name=args.embedding_model,
        reranker=Reranker(args.reranker_model, args.reranker_max_length),
        generator=generator, candidate_k=args.candidate_k,
        top_k=args.top_k, rrf_k=args.rrf_k,
    )
    result = rag.ask(args.question)
    result["config"] = {
        "embedding_model": args.embedding_model, "reranker_model": args.reranker_model,
        "reranker_max_length": args.reranker_max_length, "generation_model": args.model,
        "candidate_k": args.candidate_k, "top_k": args.top_k, "rrf_k": args.rrf_k,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["answer"])
    print("\n引用来源：")
    for source in result["sources"]:
        print(f"[{source['label']}] {source['document_id']} | PDF 第 {source['page']} 页 | {source['chunk_id']}")
    if result["unknown_citations"]:
        raise SystemExit("引用检查失败：" + ", ".join(result["unknown_citations"]))
    if not result["has_citations"]:
        print("提示：回答没有引用，请检查是否为合理拒答或遗漏引用。")


if __name__ == "__main__":
    main()
