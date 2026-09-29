"""实时单轮问答：SQLite → Dense/BM25 → RRF → 重排 → 生成 → 引用检查。

FinancialRAG 实例复用已加载的向量、BM25 和模型；不重新编码文档。
入库独立使用 index_demo.py build。引用检查仅验证编号，不验证事实支持。
"""
import argparse
import json
from pathlib import Path

from financial_rag.retrieval.retriever import Retriever
from financial_rag.indexing.vector_store import VectorStore
from financial_rag.generation.citations import check_citations
from financial_rag.memory import EvidenceMemory
from financial_rag.generation.prompt import build_messages


class FinancialRAG:
    def __init__(self, store, embedding_model, embedding_model_name,
                 reranker, generator, candidate_k=20, top_k=5, rrf_k=60):
        self.retriever = Retriever(
            store, embedding_model, embedding_model_name, reranker,
            candidate_k=candidate_k, top_k=top_k, rrf_k=rrf_k,
        )
        self.chunks_by_id = self.retriever.chunks_by_id
        self.generator = generator

    def retrieve(self, question, *, top_k=None, exclude_chunk_ids=None):
        """兼容原检索入口，并允许按次排除已经获取的证据。"""
        return self.retriever.retrieve(
            question, top_k=top_k, exclude_chunk_ids=exclude_chunk_ids,
        )

    def ask_agent(self, question, **budgets):
        """启用模型自主补充检索；原 ask() 仍为固定单轮基线。"""
        from financial_rag.agent.loop import run_agent
        budgets.setdefault("use_query_planning", True)
        return run_agent(question, self.retriever, self.generator, **budgets)

    def ask(self, question):
        memory = EvidenceMemory()
        retrieval = self.retrieve(question, exclude_chunk_ids=memory.chunk_ids)
        if not retrieval["reranked"]:
            raise ValueError("没有可用于生成的证据")
        memory.record_search(question, retrieval["reranked"], self.chunks_by_id)
        context, citation_map = memory.build_context(), memory.citation_map
        answer = self.generator.generate(build_messages(question, context))
        citations = check_citations(answer, citation_map)
        # 未知引用保留在报告中并明确标记，不静默修复，也不声称验证事实。
        status = ("unknown_citations" if citations["unknown_citations"] else
                  "labels_valid" if citations["has_citations"] else "no_citations")
        return {
            "question": question, "answer": answer,
            **citations, "citation_status": status,
            "citation_map": citation_map, "retrieval": retrieval,
            "search_history": memory.search_history,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--agent", action="store_true", help="启用自主补充检索")
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
    result = rag.ask_agent(args.question) if args.agent else rag.ask(args.question)
    result["config"] = {
        "embedding_model": args.embedding_model, "reranker_model": args.reranker_model,
        "reranker_max_length": args.reranker_max_length, "generation_model": args.model,
        "candidate_k": args.candidate_k, "top_k": args.top_k, "rrf_k": args.rrf_k,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["answer"])
    if args.agent and result["status"] != "completed":
        print(f"Agent 状态：{result['status']}；停止原因：{result['stop_reason']}")
    print("\n引用来源：")
    for source in result["sources"]:
        print(f"[{source['label']}] {source['document_id']} | PDF 第 {source['page']} 页 | {source['chunk_id']}")
    if result["unknown_citations"]:
        raise SystemExit("引用检查失败：" + ", ".join(result["unknown_citations"]))
    if not result["has_citations"]:
        print("提示：回答没有引用，请检查是否为合理拒答或遗漏引用。")


if __name__ == "__main__":
    main()
