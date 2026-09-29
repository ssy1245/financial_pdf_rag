from copy import deepcopy

import pytest

from financial_rag.memory import EvidenceMemory
from financial_rag.generation.citations import prepare_evidence, check_citations


@pytest.fixture
def chunks():
    return {key: dict(chunk_id=key, document_id="doc", page=i,
                      text=f"Original evidence {key}")
            for i, key in enumerate(("a", "b", "c"), 1)}


def results(*ids):
    return [{"chunk_id": key, "text": "not the authoritative text"} for key in ids]


def test_multiple_rounds_keep_old_evidence_and_stable_citations(chunks):
    memory = EvidenceMemory()
    memory.record_search("first", results("a", "b"), chunks)
    round_two = memory.record_search("second", results("b", "c", "c"), chunks)
    assert round_two["new_chunk_ids"] == ["c"]
    assert round_two["new_count"] == 1
    assert memory.chunk_ids == {"a", "b", "c"}
    assert {k: v["chunk_id"] for k, v in memory.citation_map.items()} == {
        "E1": "a", "E2": "b", "E3": "c"}
    context = memory.build_context()
    for key in chunks:
        assert context.count(chunks[key]["text"]) == 1
    checked = check_citations("answer [E1] [E3] [E99]", memory.citation_map)
    assert [item["chunk_id"] for item in checked["sources"]] == ["a", "c"]
    assert checked["unknown_citations"] == ["E99"]


def test_snapshots_and_source_mutations_do_not_change_memory(chunks):
    memory = EvidenceMemory()
    record = memory.record_search("first", results("a"), chunks)
    record["new_chunk_ids"].clear()
    memory.evidence["a"]["text"] = "changed"
    memory.chunk_ids.clear()
    memory.citation_map["E1"]["page"] = 99
    memory.search_history[0]["new_chunk_ids"].clear()
    chunks["a"]["text"] = "source changed"
    assert "Original evidence a" in memory.build_context()
    assert memory.citation_map["E1"]["page"] == 1
    assert memory.search_history[0]["new_chunk_ids"] == ["a"]
    with pytest.raises(ValueError, match="已变化"):
        memory.add_evidence(results("a"), chunks)


def test_invalid_batch_does_not_partially_register_or_consume_labels(chunks):
    memory = EvidenceMemory()
    with pytest.raises(KeyError):
        memory.record_search("query", results("a", "missing"), chunks)
    assert memory.chunk_ids == set()
    assert memory.search_history == []
    memory.add_evidence(results("b"), chunks)
    assert memory.citation_map["E1"]["chunk_id"] == "b"


def test_empty_and_duplicate_rounds_record_zero_new_evidence(chunks):
    memory = EvidenceMemory()
    assert memory.build_context() == ""
    memory.record_search("first", results("a"), chunks)
    assert memory.record_search("repeat", results("a"), chunks)["new_count"] == 0
    assert memory.record_search("exhausted", [], chunks)["new_count"] == 0
    assert len(memory.search_history) == 3
    assert EvidenceMemory().chunk_ids == set()


def test_prepare_evidence_compatibility(chunks):
    context, mapping = prepare_evidence(results("a", "a", "b"), chunks)
    assert list(mapping) == ["E1", "E2"]
    assert context.count("[E1]") == 1
    assert "Original evidence a" in context


@pytest.mark.parametrize("field,value", [("page", 0), ("text", " "), ("document_id", "")])
def test_invalid_source_rejected(chunks, field, value):
    bad = deepcopy(chunks)
    bad["a"][field] = value
    with pytest.raises(ValueError):
        EvidenceMemory().add_evidence(results("a"), bad)
