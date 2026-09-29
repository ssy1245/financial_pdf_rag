"""SQLite 入库与检索入口：build 编码现有 chunks，search 加载索引。

每个数据库保存一套索引；build 整批替换。search 不解析 PDF、不编码文档，
仅编码问题，在内存中执行 Dense 和 BM25；尚未接 RRF、重排或生成。
"""
import argparse
import json
from pathlib import Path

import numpy as np

from financial_rag.indexing.vector_store import VectorStore
from financial_rag.indexing.bm25_index import BM25Index
from financial_rag.retrieval.dense import dense_search
from financial_rag.retrieval.sparse import bm25_search

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def build_index(store, chunks_path, model_name, model):
    """编码一次并保存；重新打开数据库验证往返一致性。"""
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks 文件必须包含非空列表")
    embeddings = model.encode_documents([chunk["text"] for chunk in chunks])
    store.replace_all(chunks, embeddings, model_name)
    restored_chunks, restored_vectors = VectorStore(store.db_path).load(model_name)
    if restored_chunks != chunks or not np.array_equal(
        restored_vectors, np.asarray(embeddings, dtype="<f4")
    ):
        raise RuntimeError("索引保存后的数据校验失败")
    return len(chunks), restored_vectors.shape[1]


def search_index(chunks, embeddings, question, top_k, model):
    """使用已加载的文档向量；只对本次问题执行编码。"""
    if not question.strip() or top_k <= 0:
        raise ValueError("问题不能为空，top_k 必须大于零")
    query_vector = model.encode_query(question)
    if query_vector.shape != (embeddings.shape[1],):
        raise ValueError("查询向量维度与保存的索引不一致")
    return {
        "question": question,
        "dense": dense_search(query_vector, embeddings, chunks, top_k),
        "bm25": bm25_search(question, BM25Index(chunks), top_k),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="从 chunks 建立 SQLite 索引")
    search = commands.add_parser("search", help="加载 SQLite 索引并检索")
    for command in (build, search):
        command.add_argument("--db", type=Path, default=ROOT / "storage/vectors.sqlite3")
        command.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--chunks", type=Path, default=ROOT / "data/chunks/apple-10k_chunks.json")
    build.add_argument("--replace", action="store_true", help="允许整批替换已有索引")
    search.add_argument("question")
    search.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "build":
        if args.db.exists() and not args.replace:
            parser.error("数据库已存在；如需整批重建，请添加 --replace")
        if not args.chunks.is_file():
            parser.error("chunks 文件不存在，请先解析并分块")
    elif not args.question.strip() or args.top_k <= 0:
        parser.error("问题不能为空，--top-k 必须大于零")

    store = VectorStore(args.db)
    # search 先检查索引及模型配置，缺少索引时不加载 embedding 模型。
    if args.command == "search":
        try:
            chunks, embeddings = store.load(args.model)
        except FileNotFoundError:
            parser.error("索引不存在，请先运行 index_demo.py build")

    from financial_rag.indexing.embeddings import EmbeddingModel
    model = EmbeddingModel(args.model)
    if args.command == "build":
        count, dimension = build_index(store, args.chunks, args.model, model)
        print(f"已保存并校验 {count} 个 chunks，向量维度 {dimension}。")
        print(f"数据库：{args.db}")
    else:
        result = search_index(chunks, embeddings, args.question, args.top_k, model)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
