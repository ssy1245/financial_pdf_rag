import json
import pytest

from financial_rag.memory import ConversationMemory


def result(answer='answer [E1]', **kwargs):
    return dict(status='completed', answer=answer, sources=[{'text':'source', 'label':'E1'}], **kwargs)


def test_recent_turns_are_bounded_and_snapshots_are_independent():
    memory = ConversationMemory('v1', max_turns=2)
    source = result()
    memory.record('one', source)
    source['sources'][0]['text'] = 'changed'
    assert memory.turns[0]['sources'][0]['text'] == 'source'
    snapshot = memory.turns
    snapshot[0]['sources'].clear()
    assert memory.turns[0]['sources']
    memory.record('two', result())
    memory.record('three', result())
    assert [t['question'] for t in memory.turns] == ['two','three']
    assert len({t['turn_id'] for t in memory.turns}) == 2
    assert all(t['workspace_version'] == 'v1' for t in memory.turns)
    assert all('sources' not in t and '[E1]' not in t['answer'] for t in memory.context())


def test_context_budget_removes_oldest_whole_turn_and_storage_has_own_limit():
    memory = ConversationMemory('v1')
    memory.record('first', result())
    memory.record('second', result())
    latest = [memory.context()[-1]]
    memory.max_context_chars = len(json.dumps(latest, ensure_ascii=False))
    assert memory.context() == latest
    assert len(memory.turns) == 2
    memory.max_context_chars = 1
    assert memory.context() == []
    memory.max_storage_chars = 1
    memory.record('third', result())
    assert memory.turns == []


@pytest.mark.parametrize('status', ['limited','error','clarification','step_limit'])
def test_incomplete_results_are_not_history(status):
    memory = ConversationMemory('v1')
    memory.record('q', dict(status=status))
    assert not memory.turns


@pytest.mark.parametrize('value', [0, -1, True])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        ConversationMemory('v1', max_turns=value)
