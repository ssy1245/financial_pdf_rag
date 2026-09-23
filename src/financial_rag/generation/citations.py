"""为证据分配引用编号，并检查回答中的编号是否存在。

编号有效不代表证据支持结论；事实与引用的对应关系仍需评估。
"""
import re


def prepare_evidence(results, chunks_by_id):
    evidence_blocks = []
    citation_map = {}

    for i, result in enumerate(results, start=1):
        label = f"E{i}"
        chunk = chunks_by_id[result["chunk_id"]]

        citation_map[label] = {
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
        }

        evidence_blocks.append(
            f"[{label}]\n"
            f"Document: {chunk['document_id']}\n"
            f"PDF page: {chunk['page']}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n".join(evidence_blocks)
    return context, citation_map


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
