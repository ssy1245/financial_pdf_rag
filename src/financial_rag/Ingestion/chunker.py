"""逐页分块：优先保留段落，长段落按空白分隔的词数拆分。

chunk_size 和 overlap 的单位是词，不是模型 token。
不跨页拼接；保留页面元数据，使用文档标识、页码与页内序号生成稳定 ID。
章节识别与基于 tokenizer 的分块留待后续实现。
普通段落边界优先保持新段落完整，因此 overlap 是上限；长段落内部使用固定 overlap。
独立长段落与相邻段落之间、页面之间不添加 overlap。
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
