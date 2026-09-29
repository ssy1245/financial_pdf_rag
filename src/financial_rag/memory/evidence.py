"""请求内证据注册表；不持久化、不调用模型、不共享跨问题状态。"""
from copy import deepcopy


class EvidenceMemory:
    def __init__(self):
        self._evidence = {}
        self._search_history = []

    @property
    def evidence(self):
        """返回快照，避免调用方修改正文或引用编号。"""
        return deepcopy(self._evidence)

    @property
    def chunk_ids(self):
        return set(self._evidence)

    @property
    def search_history(self):
        return deepcopy(self._search_history)

    @property
    def citation_map(self):
        return {
            item["citation_id"]: {
                key: item[key] for key in ("document_id", "chunk_id", "page")
            }
            for item in self._evidence.values()
        }

    def add_evidence(self, results, chunks_by_id):
        """注册最终检索结果，返回新增 ID；正文以原始 chunk 为准。

        整批校验后再写入；相同 ID 的来源或正文发生变化时拒绝混用索引。
        重排分数属于每次查询，不存为证据自身的固定属性。
        """
        pending = {}
        for result in results:
            chunk_id = result["chunk_id"]
            chunk = deepcopy(chunks_by_id[chunk_id])
            if chunk["chunk_id"] != chunk_id:
                raise ValueError("chunk_id 与来源映射不一致")
            for key in ("chunk_id", "document_id", "text"):
                if not isinstance(chunk[key], str) or not chunk[key].strip():
                    raise ValueError(f"证据 {key} 必须为非空字符串")
            if type(chunk["page"]) is not int or chunk["page"] <= 0:
                raise ValueError("证据 page 必须为正整数")
            # 只保留来源字段，不接受输入提供的 citation_id。
            item = {key: chunk[key] for key in ("chunk_id", "document_id", "page", "text")}
            for key in ("title", "filename"):
                if chunk.get(key):
                    item[key] = chunk[key]
            existing = self._evidence.get(chunk_id, pending.get(chunk_id))
            if existing is not None:
                if any(existing[key] != item[key] for key in ("document_id", "page", "text")):
                    raise ValueError("相同 chunk_id 的证据已变化，请为新索引创建新的记忆")
                continue
            item["citation_id"] = f"E{len(self._evidence) + len(pending) + 1}"
            pending[chunk_id] = item
        self._evidence.update(pending)
        return list(pending)

    def record_search(self, query, results, chunks_by_id, missing_information=None):
        """注册一轮最终证据并记录查询；空结果也记录，不代表检索失败。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("查询不能为空")
        results = list(results)
        new_ids = self.add_evidence(results, chunks_by_id)
        record = {
            "query": query,
            "retrieved_chunk_ids": list(dict.fromkeys(r["chunk_id"] for r in results)),
            "new_chunk_ids": new_ids,
            "new_count": len(new_ids),
        }
        if missing_information is not None:
            record["missing_information"] = missing_information
        self._search_history.append(record)
        return deepcopy(record)

    def build_context(self):
        """按首次加入顺序返回累计原文；不自动摘要或静默截断。"""
        return "\n\n".join(
            f"[{item['citation_id']}]\n"
            f"Document: {item.get('title', item['document_id'])}\n"
            f"PDF page: {item['page']}\n"
            f"Text:\n{item['text']}"
            for item in self._evidence.values()
        )
