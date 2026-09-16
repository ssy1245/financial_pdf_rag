from pathlib import Path

from financial_rag.ingestion.normalizer import normalize_pages
from financial_rag.ingestion.parser import parse_pdf, save_parsed_document
from financial_rag.ingestion.chunker import chunk_pages
from financial_rag.indexing.embeddings import EmbeddingModel
from financial_rag.retrieval.dense import dense_search

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

    # --------------------
    # Dense Retrieval
    # --------------------

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
