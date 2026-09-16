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

def chunk_pages(
    pages: list[dict],
    document_id: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    validate_chunk_config(chunk_size, overlap)

    chunks = []

    for page in pages:
        page_number = page["page"]

        paragraphs = [
            paragraph.strip()
            for paragraph in page["text"].split("\n\n")
            if paragraph.strip()
        ]

        current_words = []
        page_chunk_index = 0

        for paragraph in paragraphs:
            paragraph_words = paragraph.split()

            # 情况 1：
            # paragraph 自己就超过 chunk_size
            if len(paragraph_words) > chunk_size:

                # 先保存前面已经积累的内容
                if current_words:
                    chunk_text = " ".join(current_words)

                    chunks.append(
                        {
                            **page,
                            "chunk_id": (
                                f"{document_id}_"
                                f"p{page_number}_"
                                f"c{page_chunk_index}"
                            ),
                            "document_id": document_id,
                            "page": page_number,
                            "word_count": len(current_words),
                            "text": chunk_text,
                        }
                    )

                    page_chunk_index += 1
                    current_words = []

                # 再单独处理这个超长 paragraph
                long_chunks = split_long_text(
                    paragraph,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                for chunk_text in long_chunks:
                    chunks.append(
                        {
                            **page,
                            "chunk_id": (
                                f"{document_id}_"
                                f"p{page_number}_"
                                f"c{page_chunk_index}"
                            ),
                            "document_id": document_id,
                            "page": page_number,
                            "word_count": len(
                                chunk_text.split()
                            ),
                            "text": chunk_text,
                        }
                    )

                    page_chunk_index += 1

                continue

            # 情况 2：
            # 当前 chunk + 新 paragraph 会超限
            if (
                current_words
                and
                len(current_words)
                + len(paragraph_words)
                > chunk_size
            ):
                chunk_text = " ".join(current_words)

                chunks.append(
                    {
                        **page,
                        "chunk_id": (
                            f"{document_id}_"
                            f"p{page_number}_"
                            f"c{page_chunk_index}"
                        ),
                        "document_id": document_id,
                        "page": page_number,
                        "word_count": len(current_words),
                        "text": chunk_text,
                    }
                )

                page_chunk_index += 1

                # 给完整的新段落留出空间；实际重叠不能使新块超限。
                overlap_words = min(overlap, chunk_size - len(paragraph_words))
                current_words = (
                    current_words[-overlap_words:]
                    if overlap_words > 0
                    else []
                )

            # 加入当前 paragraph
            current_words.extend(paragraph_words)

        # 当前页面最后剩余内容
        if current_words:
            chunks.append(
                {
                    **page,
                    "chunk_id": (
                        f"{document_id}_"
                        f"p{page_number}_"
                        f"c{page_chunk_index}"
                    ),
                    "document_id": document_id,
                    "page": page_number,
                    "word_count": len(current_words),
                    "text": " ".join(current_words),
                }
            )

    return chunks
