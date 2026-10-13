"""有界的会话历史；来源快照留在后端，历史只帮助理解问题。"""
from copy import deepcopy
import json
import re
from uuid import uuid4


class ConversationMemory:
    def __init__(self, workspace_version, max_turns=5, max_context_chars=12000,
                 max_storage_chars=400000):
        for value in (max_turns, max_context_chars, max_storage_chars):
            if type(value) is not int or value <= 0:
                raise ValueError("会话预算必须为正整数")
        self.workspace_version = workspace_version
        self.max_turns = max_turns
        self.max_context_chars = max_context_chars
        self.max_storage_chars = max_storage_chars
        self._turns = []

    @property
    def turns(self):
        return deepcopy(self._turns)

    def context(self):
        # 不传旧来源和引用编号，避免旧 E1 被当成本轮证据。
        selected = []
        for turn in reversed(self._turns):
            item = {key: re.sub(r"\[E\d+\]", "", turn[key]) for key in
                    ("question", "resolved_question", "answer")}
            candidate = [item, *selected]
            if len(json.dumps(candidate, ensure_ascii=False)) > self.max_context_chars:
                break  # 保留连续的最近历史，不截断单轮问答
            selected = candidate
        return selected

    def record(self, question, result):
        if result.get("status") != "completed":
            return
        turn = {
            "turn_id": uuid4().hex,
            "workspace_version": self.workspace_version,
            "question": question,
            "resolved_question": result.get("resolved_question", question),
            "answer": result["answer"],
            "sources": deepcopy(result.get("sources", [])),
            "status": "completed",
        }
        turns = [*self._turns, turn][-self.max_turns:]
        while turns and len(json.dumps(turns, ensure_ascii=False)) > self.max_storage_chars:
            turns.pop(0)
        self._turns = turns
