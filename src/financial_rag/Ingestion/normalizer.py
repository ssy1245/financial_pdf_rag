"""正文清洗。

状态：架构占位，尚未实现业务逻辑。

职责与输入输出：
输入：同一文档的 ParsedPage 列表。输出：清洗后的页面列表。
利用多页重复特征识别明显页眉页脚，处理断行、空白与断词，记录必要的清洗信息。
把“PDF 提取器吐出来的文本”变成“后续 chunker 和 retriever 能稳定使用的文本“
边界与约束：
保留页码、金额、负号、百分号和表格语义；避免凭重复次数删除正文。
"""
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


def normalize_pages(pages: list[dict]) -> list[dict]:
    normalized_pages = []
    for page in pages:
        normalized_blocks = []
        for block in page["blocks"]:
            cleaned_text = normalize_text(block["text"])

            if cleaned_text:
                normalized_blocks.append({
                    **block,
                    "text": cleaned_text,
                })
        normalized_pages.append({
            **page,
            "blocks": normalized_blocks,
            "text": "\n\n".join(
                block["text"]
                for block in normalized_blocks
            ),
        })
    return normalized_pages


