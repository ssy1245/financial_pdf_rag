"""分块回归测试：多页、长度、重叠、元数据、参数校验和段落边界。
"""
import pytest

from financial_rag.ingestion.chunker import chunk_pages


def test_all_pages_and_paragraphs_are_processed_without_duplicate_tail():
    pages = [
        {"page": 1, "text": "a b\n\nc d", "section": "Intro"},
        {"page": 2, "text": "e f"},
    ]
    chunks = chunk_pages(pages, "doc", chunk_size=5, overlap=1)
    assert [(c["page"], c["text"]) for c in chunks] == [(1, "a b\n\nc d"), (2, "e f")]
    assert chunks[0]["section"] == "Intro"
    assert chunks == chunk_pages(pages, "doc", chunk_size=5, overlap=1)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


@pytest.mark.parametrize("overlap", [0, 2])
def test_long_paragraph_preserves_words_and_overlap(overlap):
    words = [f"w{i}" for i in range(23)]
    chunks = chunk_pages([{"page": 3, "text": " ".join(words)}], "doc", 7, overlap)
    parts = [c["text"].split() for c in chunks]
    assert all(0 < len(part) <= 7 for part in parts)
    recovered = parts[0][:]
    for previous, part in zip(parts, parts[1:]):
        if overlap:
            assert previous[-overlap:] == part[:overlap]
        recovered.extend(part[overlap:])
    assert recovered == words


def test_paragraph_boundary_and_zero_overlap():
    chunks = chunk_pages([{"page": 1, "text": "a b c\n\nd e f"}], "doc", 5, 0)
    assert [c["text"] for c in chunks] == ["a b c", "d e f"]


def test_empty_pages():
    assert chunk_pages([], "doc") == []
    assert chunk_pages([{"page": 1, "text": "  "}], "doc") == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (5, -1), (5, 5), (5, 6)])
def test_invalid_configuration(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], "doc", size, overlap)


@pytest.mark.parametrize("paragraph, expected_tail", [("e f g h i", "e f g h i"), ("e f g h", "d\n\ne f g h")])
def test_overlap_does_not_exceed_chunk_size(paragraph, expected_tail):
    pages = [{"page": 1, "text": "a b c d\n\n" + paragraph, "title": "Title"}]
    chunks = chunk_pages(pages, "doc", 5, 2)
    assert [c["text"] for c in chunks] == ["a b c d", expected_tail]
    assert all(c["word_count"] <= 5 and c["title"] == "Title" for c in chunks)


@pytest.mark.parametrize("size, overlap", [(0, 0), (5, -1), (5, 5), (5, 6)])
def test_split_long_text_rejects_invalid_configuration(size, overlap):
    from financial_rag.ingestion.chunker import split_long_text
    with pytest.raises(ValueError):
        split_long_text("a b c d e f g", size, overlap)


def test_overlap_preserves_multiple_paragraph_boundaries_and_excludes_blocks():
    pages = [{"page": 1, "text": "a b\n\nc d\n\ne f g", "blocks": [{"text": "source"}], "section": "Risk"}]
    chunks = chunk_pages(pages, "doc", 6, 3)
    assert [c["text"] for c in chunks] == ["a b\n\nc d", "b\n\nc d\n\ne f g"]
    assert all("blocks" not in c and c["section"] == "Risk" for c in chunks)
    assert [c["word_count"] for c in chunks] == [4, 6]
