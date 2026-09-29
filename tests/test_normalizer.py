"""清洗回归测试：重复底部页脚、纯符号移除、正文数字符号和元数据保留。
"""
from copy import deepcopy
from financial_rag.ingestion.normalizer import normalize_pages


def block(text, y):
    return {"text": text, "bbox": [0, y, 100, y + 10]}


def test_only_repeated_bottom_footer_and_symbol_blocks_removed():
    pages = [{"page": i, "blocks": [
        block("Apple Inc. | 2025 Form 10-K | 9", 10),
        block("Revenue (321) -4% $100\nAppleCare®", 100),
        block("® ™ ®", 120),
        block(f"Apple Inc. | 2025 Form 10-K | {i}", 700),
    ]} for i in range(1, 4)]
    before = deepcopy(pages)
    result = normalize_pages(pages)
    assert pages == before
    for page in result:
        assert len(page["blocks"]) == 2
        assert "Revenue (321) -4% $100 AppleCare®" in page["text"]
        assert "\n\n" in page["text"]
        assert page["blocks"][0]["text"] == "Apple Inc. | 2025 Form 10-K | 9"
    assert [p["page"] for p in result] == [1, 2, 3]


def test_unique_footer_and_financial_symbols_preserved():
    pages = [{"page": 1, "blocks": [block("- % $ (321)", 100), block("Apple Inc. | 2025 Form 10-K | 1", 700)]}]
    assert len(normalize_pages(pages)[0]["blocks"]) == 2


def test_no_coordinates_does_not_remove_footer():
    pages = [{"page": i, "blocks": [{"text": f"Apple Inc. | 2025 Form 10-K | {i}"}]} for i in range(3)]
    assert all(p["text"] for p in normalize_pages(pages))
