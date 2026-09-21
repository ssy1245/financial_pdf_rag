"""单问题 Dense/BM25 演示入口。

解析 PDF、清洗、生成并保存 chunks，建立内存索引和文档向量。
两路共用 query，打印 Top-5；Dense 仅预览前 1000 字符。
此入口尚未展示 Hybrid；三路批量对比使用 evaluation.compare_methods。
"""
from pathlib import Path

from financial_rag.ingestion.normalizer import normalize_pages
from financial_rag.ingestion.parser import parse_pdf, save_parsed_document
from financial_rag.ingestion.chunker import chunk_pages
from financial_rag.indexing.embeddings import EmbeddingModel
from financial_rag.retrieval.dense import dense_search
from financial_rag.indexing.bm25_index import BM25Index
from financial_rag.retrieval.sparse import bm25_search

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "apple-10k.pdf"
OUTPUT_PATH = PROJECT_ROOT / "data" / "parsed" / f"{PDF_PATH.stem}_normalized.json"

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / f"{PDF_PATH.stem}_chunks.json"

def main():
    pages = parse_pdf(PDF_PATH)
    normalized_pages = normalize_pages(pages)
    save_parsed_document(normalized_pages, OUTPUT_PATH)
    chunks = chunk_pages(
        pages=normalized_pages,
        document_id="apple_2025_10k",
        chunk_size=500,
        overlap=80,
    )
    bm25_index = BM25Index(chunks)

    save_parsed_document(chunks, CHUNKS_PATH)
    print(f"Total chunks: {len(chunks)}")
    embedding_model = EmbeddingModel()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = (
        embedding_model.encode_documents(
            chunk_texts
        )
    )

    print(
        "Chunk embeddings shape:",
        chunk_embeddings.shape
    )
    query = (
        "What are Apple's major risks "
        "in Greater China?"
    )

    query_embedding = (
        embedding_model.encode_query(query)
    )
    print(f"Query: {query}")
    print("\n===== BM25 Top-5 =====")
    results = bm25_search(
        query=query,
        index=bm25_index,
        top_k=5,
    )

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank: {rank} | "
            f"Score: {result['score']:.4f} | "
            f"Page: {result['page']} | Chunk: {result['chunk_id']}"
        )
        print(result["text"])

    # --------------------
    # Dense Retrieval
    # --------------------
    print("\n===== Dense Top-5 =====")
    results = dense_search(
        query_embedding=query_embedding,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        top_k=5,
    )
    for rank, result in enumerate(
            results,
            start=1,
    ):
        print("=" * 80)

        print(f"Rank: {rank}")
        print(
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Page: {result['page']}"
        )
        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(result["text"][:1000])



if __name__ == "__main__":
    main()
