"""PDF 解析。

状态：已实现基础逐页文本提取与 JSON 保存；布局提取和扫描页识别待实现。

职责与输入输出：
当前输入：PDF 路径。输出：包含 page 和 text 的字典列表。
后续再接入 document_id 与 ParsedPage 数据结构。
提取逐页文字，保留页边界；可用时保留布局、标题和表格信息。

边界与约束：
不清洗、不分块、不调用模型。扫描页或无法解析的页面应报告，不把空内容当作成功解析。
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
