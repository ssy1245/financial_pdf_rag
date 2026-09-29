"""为证据分配引用编号，并检查回答中的编号是否存在。

编号有效不代表证据支持结论；事实与引用的对应关系仍需评估。
"""
import re


def prepare_evidence(results, chunks_by_id):
    """保留单次调用接口；多轮调用应复用同一个 EvidenceMemory。"""
    from financial_rag.memory import EvidenceMemory

    memory = EvidenceMemory()
    memory.add_evidence(results, chunks_by_id)
    return memory.build_context(), memory.citation_map


def check_citations(answer, citation_map):
    """检查形如 [E1] 的引用；只返回回答实际使用的已知来源。"""
    labels = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", answer)))
    return {
        "sources": [
            {"label": label, **citation_map[label]}
            for label in labels if label in citation_map
        ],
        "unknown_citations": [
            label for label in labels if label not in citation_map
        ],
        "has_citations": bool(labels),
    }
