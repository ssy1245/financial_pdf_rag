"""共享数据结构。

状态：架构占位，尚未实现业务逻辑。

规划职责与输入输出（尚未实现）：
定义 ParsedPage、Chunk、SearchResult、AnswerResult 等数据契约。
ParsedPage 保留 document_id、PDF 物理页码（从 1 开始）、正文及可选标题/章节。
Chunk 保留稳定 chunk_id、document_id、pages、text、可选 section/title/token_count。
SearchResult 包含 Chunk、各路 score/rank、rrf_score 与 rerank_score。
AnswerResult 包含回答、证据与可追溯引用。

边界与约束：
跨页 chunk 使用 pages；印刷页码另存为可选标签。缺失标题或章节保持为空，不伪造元数据。
"""
