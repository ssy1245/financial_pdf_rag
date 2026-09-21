"""PDF 文本块解析与 JSON 保存。

已实现：PyMuPDF blocks + sort=True，返回 page/blocks/text；块包含 block_id/bbox/text。
页码从 1 开始，表示 PDF 物理页码；保存函数会创建父目录。
未实现：OCR、可靠的多栏顺序、表格结构和标题识别；无文字页可能返回空文本。
"""
from pathlib import Path
import json

import pymupdf


def parse_pdf(pdf_path: Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF file not supported: {pdf_path}")
    pages = []
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            blocks = []

            for block in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, text, block_number, block_type = block

                if block_type != 0 or not text.strip():
                    continue
                blocks.append({
                    "block_id": block_number,
                    "bbox": [x0, y0, x1, y1],
                    "text": text.strip(),
                })

            pages.append({
                "page": page_index + 1,
                "blocks": blocks,
                "text": "\n\n".join(
                    block["text"] for block in blocks
                ),
            })
    return pages

def save_parsed_document(pages:list[dict], output_path: Path)->None:
    """save parsed result as JSON file"""
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="提取 PDF 每页文字并保存为 JSON")
    parser.add_argument(
        "pdf_path", nargs="?", type=Path,
        default=project_root / "data/raw/apple-10k.pdf",
        help="PDF 文件路径，默认使用项目中的 Apple 10-K",
    )
    parser.add_argument(
        "--output", type=Path,
        help="JSON 输出路径，默认保存到项目 data/parsed/<PDF 文件名>.json",
    )
    args = parser.parse_args()
    output_path = args.output or project_root / "data/parsed" / f"{args.pdf_path.stem}.json"
    pages = parse_pdf(args.pdf_path)
    save_parsed_document(pages, output_path)
    empty_pages = [page["page"] for page in pages if not page["text"]]
    print(f"解析完成：{len(pages)} 页")
    print(f"JSON 已保存：{output_path}")
    if empty_pages:
        print(f"以下页面未提取到文字，需检查是否需要 OCR：{empty_pages}")
    if pages:
        print(f"第一页预览：\n{pages[0]['text'][:500]}")
