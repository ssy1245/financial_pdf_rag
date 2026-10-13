"""Parse PDF/DOCX/XLSX to versioned JSON; optionally build retrieval views."""
import argparse
from pathlib import Path
import json
from .adapters import parse_document
from .structured_chunker import chunk_document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files",nargs="+",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tokenizer",type=Path,help="Optional local tokenizer folder; also writes 300-token chunks")
    args=parser.parse_args()
    tokenizer=None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for path in args.files:
        document=parse_document(path)
        (args.output/f"{document.document_id}.json").write_text(document.model_dump_json(indent=2),encoding="utf-8")
        if tokenizer:
            chunks=chunk_document(document,tokenizer)
            (args.output/f"{document.document_id}.chunks.json").write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"{path.name}: {len(document.elements)} elements, {len(document.tables)} tables, {len(document.ocr_requests)} OCR requests")


if __name__=="__main__":
    main()
