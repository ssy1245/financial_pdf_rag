"""逐页按段落组合 chunks，保留段落空行。

已实现：长段落按空白词数拆分，校验 chunk_size/overlap，保留单页 page、
word_count、已有元数据及确定性页内 ID；不复制整页 blocks。
普通段落边界的 overlap 可减少以防超限，长段落内部使用固定 overlap；
独立长段落之间和页面之间不额外重叠。
限制：单位是词而非 token；不识别章节，不跨页，不保证模型输入不被截断。
"""
def validate_chunk_config(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")


def count_words(text: str) -> int:
    return len(text.split())

def split_long_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Hard split for a single paragraph that is too long.
    """

    validate_chunk_config(chunk_size, overlap)
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        chunks.append(
            " ".join(chunk_words)
        )

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def tail_paragraphs(paragraphs: list[str], word_limit: int) -> list[str]:
    """取末尾若干词作为 overlap，同时保留这些词之间原有的段落边界。"""
    result = []
    remaining = word_limit
    for paragraph in reversed(paragraphs):
        if remaining <= 0:
            break
        words = paragraph.split()
        result.append(" ".join(words[-remaining:]))
        remaining -= min(remaining, len(words))
    return list(reversed(result))


def chunk_pages(
    pages: list[dict],
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    validate_chunk_config(chunk_size, overlap)
    chunks = []

    for page in pages:
        metadata = {key: value for key, value in page.items() if key not in {"text", "blocks"}}
        page_chunk_index = 0
        current_paragraphs = []
        current_word_count = 0

        def append_chunk(text: str) -> None:
            nonlocal page_chunk_index
            chunks.append({
                **metadata,
                "chunk_id": f"{document_id}_p{page['page']}_c{page_chunk_index}",
                "document_id": document_id,
                "word_count": count_words(text),
                "text": text,
            })
            page_chunk_index += 1

        paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
        for paragraph in paragraphs:
            paragraph_word_count = count_words(paragraph)
            if paragraph_word_count > chunk_size:
                if current_paragraphs:
                    append_chunk("\n\n".join(current_paragraphs))
                    current_paragraphs = []
                    current_word_count = 0
                for text in split_long_text(paragraph, chunk_size, overlap):
                    append_chunk(text)
                continue

            if current_paragraphs and current_word_count + paragraph_word_count > chunk_size:
                append_chunk("\n\n".join(current_paragraphs))
                overlap_words = min(overlap, chunk_size - paragraph_word_count)
                current_paragraphs = tail_paragraphs(current_paragraphs, overlap_words)
                current_word_count = sum(count_words(p) for p in current_paragraphs)

            current_paragraphs.append(paragraph)
            current_word_count += paragraph_word_count

        if current_paragraphs:
            append_chunk("\n\n".join(current_paragraphs))

    return chunks
