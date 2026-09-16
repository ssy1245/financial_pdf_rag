"""块内清洗并保留块边界；保守删除重复的 10-K 页脚和纯商标符号块。

页脚必须整块匹配“公司 | 年份 Form 10-K | 页码”，位于最下方，
且同一公司和年份在至少三个页面出现。缺少坐标时不删除页脚。
不识别复杂表格，不删除正文内的商标符号或金融标点。
"""
from collections import Counter
import re

def normalize_whitespace(text:str)->str:
    """clean unnecessary whitespace"""
    text=text.replace("\r\n","\n")#统一换行符
    text=text.replace("\r","\n")
    #多个空格tab压缩成一个空格
    text=re.sub(r"[ \t]+"," ",text)
    # 三个及以上连续换行压成两个
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def merge_broken_lines(text: str) -> str:
    """
    Merge line breaks that are likely caused by PDF layout,
    while preserving paragraph boundaries.
    """
    lines = text.split("\n")
    result = []
    current_paragraph = []
    for line in lines:
        line = line.strip()
        # 空行意味着 paragraph boundary
        if not line:
            if current_paragraph:
                result.append(" ".join(current_paragraph))
                current_paragraph = []
            continue
        current_paragraph.append(line)
    # 最后一个 paragraph
    if current_paragraph:
        result.append(" ".join(current_paragraph))

    return "\n\n".join(result)

def normalize_text(text: str) -> str:
    """
    Main text normalization pipeline.
    """

    text = normalize_whitespace(text)
    text = merge_broken_lines(text)
    return text.strip()



FOOTER_PATTERN = re.compile(
    r"(?P<label>[^|\n]+\|\s*\d{4}\s+Form\s+10-K)\s*\|\s*\d+",
    re.IGNORECASE,
)


def footer_key(block: dict, blocks: list[dict]) -> str | None:
    text = " ".join(block["text"].split())
    match = FOOTER_PATTERN.fullmatch(text)
    if not match or "bbox" not in block:
        return None
    # 使用文本块坐标判断是否位于本页最下方，不假定所有 PDF 纸张尺寸相同。
    bottom = max((b["bbox"][3] for b in blocks if "bbox" in b), default=0)
    if block["bbox"][3] < bottom - 2:
        return None
    return match["label"].casefold()


def normalize_pages(pages: list[dict]) -> list[dict]:
    frequencies = Counter()
    for page in pages:
        keys = {footer_key(block, page["blocks"]) for block in page["blocks"]}
        frequencies.update(key for key in keys if key is not None)

    normalized_pages = []
    for page in pages:
        normalized_blocks = []
        for block in page["blocks"]:
            text = block["text"]
            # 仅删整块纯商标符号；保留正文中的 ®、负号、百分号等。
            if re.fullmatch(r"[®™\s]+", text):
                continue
            key = footer_key(block, page["blocks"])
            if key is not None and frequencies[key] >= 3:
                continue
            cleaned_text = normalize_text(text)
            if cleaned_text:
                normalized_blocks.append({**block, "text": cleaned_text})
        normalized_pages.append({
            **page,
            "blocks": normalized_blocks,
            "text": "\n\n".join(block["text"] for block in normalized_blocks),
        })
    return normalized_pages
