"""8 题实际 Dense 模型测试，需 --run-dense；检查流程和证据映射，不评判相关性。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "data/eval/dense_questions.json").read_text())
MODEL_NAME = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def dense_context(request):
    if not request.config.getoption("--run-dense"):
        pytest.skip("使用 --run-dense 启用实际模型检索测试")

    # 延迟导入，普通单元测试不加载或下载模型。
    from financial_rag.indexing.embeddings import EmbeddingModel

    path = ROOT / "data/chunks/apple-10k_chunks.json"
    if not path.is_file():
        pytest.fail(f"找不到分块文件：{path}，请先运行 main.py 生成")
    content = path.read_bytes()
    chunks = json.loads(content)
    assert len(chunks) >= 5, "至少需要 5 个 chunks 才能检查 Top-5"
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)

    model = EmbeddingModel(model_name=MODEL_NAME)
    embeddings = model.encode_documents([chunk["text"] for chunk in chunks])
    assert embeddings.ndim == 2 and embeddings.shape[0] == len(chunks)
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4)
    report = {
        "model": MODEL_NAME,
        "chunks_sha256": hashlib.sha256(content).hexdigest(),
        "chunk_count": len(chunks),
        "top_k": 5,
        "note": "无标准证据标注；测试通过不代表检索相关性合格。未指定年份的问题原样保留，需人工检查。",
        "queries": [],
    }
    yield model, embeddings, chunks, report
    output = ROOT / "outputs/dense_retrieval_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n完整检索结果已保存：{output}")


@pytest.mark.parametrize("case", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_dense_top_five(case, dense_context):
    from financial_rag.retrieval.dense import dense_search

    model, embeddings, chunks, report = dense_context
    query_embedding = model.encode_query(case["question"])
    assert query_embedding.shape == (embeddings.shape[1],)
    assert np.isfinite(query_embedding).all()
    assert np.isclose(np.linalg.norm(query_embedding), 1, atol=1e-4)
    results = dense_search(query_embedding, embeddings, chunks, top_k=5)
    report["queries"].append({**case, "results": results})

    print("\n" + "=" * 80)
    print(f"Question [{case['id']}]: {case['question']}")
    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank} | Score: {result['score']:.4f} | PDF Page: {result['page']} | Chunk: {result['chunk_id']}")
        print(result["text"])
        print("[END OF EVIDENCE]")

    assert len(results) == 5
    assert len({result["chunk_id"] for result in results}) == 5
    scores = [result["score"] for result in results]
    assert all(np.isfinite(score) and -1.0001 <= score <= 1.0001 for score in scores)
    assert scores == sorted(scores, reverse=True)
    source = {chunk["chunk_id"]: chunk for chunk in chunks}
    for result in results:
        original = source[result["chunk_id"]]
        assert result["page"] == original["page"]
        assert result["text"] == original["text"]
