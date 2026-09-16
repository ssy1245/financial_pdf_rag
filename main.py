from pathlib import Path

from financial_rag.ingestion.normalizer import normalize_pages
from financial_rag.ingestion.parser import parse_pdf, save_parsed_document
from financial_rag.ingestion.chunker import chunk_pages


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

    print(f"Total pages: {len(normalized_pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Covered pages: {len({chunk['page'] for chunk in chunks})}")
    print(f"Max words: {max((chunk['word_count'] for chunk in chunks), default=0)}")
    print(f"Chunks saved to: {CHUNKS_PATH}")

    for chunk in chunks[:5]:
        print("=" * 80)
        print("Chunk ID:", chunk["chunk_id"])
        print("Page:", chunk["page"])
        print("Words:", len(chunk["text"].split()))
        print(chunk["text"][:1000])
        print(chunk["text"])
        print("[END OF CHUNK]")



if __name__ == "__main__":
    main()
