"""读取已保存的重排结果，运行单题生成；--dry-run 仅预览输入。

默认使用 BGE 报告中的 net_sales 问题，不重新执行检索。
环境变量由终端或 uv --env-file 加载，代码不读取或打印密钥。
"""
import argparse
import hashlib
import json
from pathlib import Path

from financial_rag.generation.citations import prepare_evidence, check_citations
from financial_rag.generation.prompt import build_messages

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/bge_m3_reranked_comparison.json")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks/apple-10k_chunks.json")
    parser.add_argument("--question-id", default="net_sales")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    chunk_bytes = args.chunks.read_bytes()
    expected_hash = report.get("chunks_sha256")
    if expected_hash and hashlib.sha256(chunk_bytes).hexdigest() != expected_hash:
        parser.error("chunks 与检索报告不匹配，请使用生成该报告时的 chunks")
    chunks = json.loads(chunk_bytes)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        parser.error("chunks 中存在重复 chunk_id")
    query = next((q for q in report["queries"] if q["id"] == args.question_id), None)
    if query is None:
        parser.error("找不到问题 ID，可选：" + ", ".join(q["id"] for q in report["queries"]))
    results = query.get("reranked", [])
    if not results:
        parser.error("该问题没有重排证据，请先生成重排报告")
    for result in results:
        chunk = chunks_by_id.get(result["chunk_id"])
        if chunk is None or chunk["text"] != result["text"] or chunk["page"] != result["page"]:
            parser.error("重排证据与 chunks 不一致，请重新生成报告")

    context, citation_map = prepare_evidence(results, chunks_by_id)
    messages = build_messages(query["question"], context)
    if args.dry_run:
        print(json.dumps({"messages": messages, "citation_map": citation_map}, ensure_ascii=False, indent=2))
        return

    # 仅在实际生成时初始化客户端；导入与预览不需要密钥。
    from financial_rag.generation.generator import DeepSeekGenerator

    generator = DeepSeekGenerator(model=args.model)
    answer = generator.generate(messages)
    result = {
        "question_id": query["id"],
        "question": query["question"],
        "model": args.model,
        "answer": answer,
        **check_citations(answer, citation_map),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["unknown_citations"]:
        raise SystemExit("引用检查失败：回答含有未知编号，不能作为已验证答案使用")
    if not result["has_citations"]:
        print("提示：回答没有引用，请检查是否为合理拒答或遗漏引用。")


if __name__ == "__main__":
    main()
