# Financial PDF RAG 项目规格说明

> 项目定位：一个面向金融报告（年报、10-K、财报、研究报告等）的可评估 RAG 系统。  
> 核心目标不是“做一个聊天机器人”，而是完整实现并理解：**PDF 解析 → Chunking → Indexing → Retrieval → Fusion → Rerank → Evidence → LLM Answer → Evaluation**。  
> 后续可以继续升级为用户上传 PDF 的 Web 应用，并最终接入钱包、链上数据和智能合约，演化为 AI + Web3 dApp。

---

## 1. 项目最终目标

用户上传一份金融 PDF，例如 Apple 10-K，然后可以提出：

- 2025 年总营收是多少？
- Greater China revenue 同比变化多少？
- 公司的主要 Risk Factors 是什么？
- R&D expense 为什么增长？
- 哪些内容提到了 AI / capital expenditure？
- 现金流是否出现明显恶化？
- 管理层如何解释某项指标变化？

系统需要：

1. 正确解析 PDF。
2. 保留页码、章节、标题等结构信息。
3. 对文档建立 Dense Retrieval 和 BM25 两套索引。
4. 支持 Hybrid Retrieval。
5. 对候选结果进行 Rerank。
6. 只基于检索到的证据回答。
7. 给出引用页码和原始证据。
8. 在证据不足时明确回答“无法从当前报告中确认”。
9. 有独立 Evaluation 模块，可以比较不同 Retrieval 方法。
10. 后期支持用户动态上传 PDF，而不是只使用固定知识库。

---

# 2. 项目核心原则

## 2.1 第一阶段不要过度依赖框架

优先自己实现核心逻辑。

尽量不要一开始直接使用：

```python
retriever = vectorstore.as_retriever()
retriever.invoke(query)
```

而应该显式实现：

```python
dense_results = dense_search(query, top_k=20)
bm25_results = bm25_search(query, top_k=20)

merged = reciprocal_rank_fusion(
    dense_results,
    bm25_results
)

reranked = rerank(
    query,
    merged[:20]
)

context = reranked[:5]
```

目标是理解 Retrieval Pipeline，而不是只会调用 LangChain。

---

## 2.2 Retrieval 和 Generation 分离

系统逻辑分为三部分：

### A. Ingestion / Indexing

```text
PDF
→ Parse
→ Normalize
→ Chunk
→ Metadata
→ Embedding
→ Vector Index
→ BM25 Index
```

### B. Retrieval

```text
User Query
→ Query Embedding
→ Dense Search
→ BM25 Search
→ Hybrid Fusion
→ Reranker
→ Top-K Evidence
```

### C. Generation

```text
Question
+ Top-K Evidence
→ Prompt Assembly
→ LLM
→ Answer
+ Citation
```

Retrieval 本身不依赖最终使用哪个 LLM。

---

# 3. 推荐技术栈

第一版尽量简单。

## Python

建议：

```text
Python 3.11+
```

## PDF Parsing

优先候选：

```text
PyMuPDF
Docling
MinerU
```

第一版可以先从 PyMuPDF 或 Docling 开始。

后续再尝试复杂 layout / table parsing。

---

## Embedding

可以使用任意成熟 Embedding API 或本地模型。

例如：

```text
OpenAI embedding
BGE
Qwen embedding
E5
```

要求：

- 文档 embedding 和 query embedding 必须使用兼容模型。
- embedding model 应可替换。

---

## Vector Store

第一版可以直接：

```text
FAISS
```

之后再替换：

```text
Qdrant
Milvus
Weaviate
pgvector
```

第一版不需要考虑分布式部署。

---

## BM25

建议：

```text
rank_bm25
```

或者自己封装 BM25 检索逻辑。

---

## Reranker

候选：

```text
BGE Reranker
Cross Encoder
Cohere Rerank
Qwen Reranker
```

要求 reranker 封装成独立模块，可以随时替换。

---

## LLM

LLM 只负责：

```text
Evidence → Answer
```

不要让 LLM 参与第一版 Retrieval 的主要逻辑。

---

# 4. 数据模型

每个 chunk 至少需要：

```python
{
    "chunk_id": "apple_2025_p18_03",
    "document_id": "apple_2025_10k",
    "company": "Apple",
    "year": 2025,
    "page": 18,
    "section": "Risk Factors",
    "title": "Business Risks",
    "text": "...",
}
```

后续可以扩展：

```python
{
    "content_type": "text",
    "table_id": None,
    "parent_chunk_id": None,
    "token_count": 420,
}
```

---

# 5. 项目目录结构

当前采用 `src/financial_rag/` 包布局。以下 `src` 架构已建立，Python 文件仅包含职责说明，**没有业务实现**。根目录配套文件和数据目录按阶段补齐，不表示相应功能已经完成。

```text
financial_pdf_rag/
├── README.md                       # uv 环境与运行说明
├── pyproject.toml                  # uv 依赖和构建配置（CLI 待实现）
├── uv.lock                         # 依赖锁文件
├── .python-version                 # 开发解释器：Python 3.11
├── .env.example                    # 配置示例，不含真实密钥
├── .gitignore
├── financial_pdf_rag_project_spec.md
├── data/
│   ├── raw/                        # 原始 PDF
│   ├── parsed/                     # 按页解析结果
│   ├── chunks/                     # 分块结果
│   └── eval/                       # 人工标注评估集
├── storage/                        # 索引和缓存，默认不提交 Git
├── outputs/                        # 实验结果
├── src/
│   └── financial_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── schemas.py
│       ├── cli.py
│       ├── pipeline.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── parser.py
│       │   ├── normalizer.py
│       │   └── chunker.py
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── vector_store.py
│       │   └── bm25_index.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── dense.py
│       │   ├── sparse.py
│       │   ├── fusion.py
│       │   ├── reranker.py
│       │   └── retriever.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── prompt.py
│       │   ├── generator.py
│       │   └── citations.py
│       └── evaluation/
│           ├── __init__.py
│           ├── dataset.py
│           ├── metrics.py
│           └── benchmark.py
└── tests/                          # 实现各阶段时补充
    └── fixtures/                   # 小型可提交测试数据
```

## 5.1 模块职责与依赖

- `ingestion`：PDF → 页面 → 清洗后的页面 → chunks，不依赖模型。
- `indexing`：提供编码与索引能力；索引保存 chunk 映射和模型/分词配置。
- `retrieval`：调用索引，统一结果，执行融合和重排；不依赖回答模型。
- `generation`：仅基于传入证据回答，并检查引用标签与来源的对应关系。
- `evaluation`：调用检索流程，按固定口径比较方法；不由检索模块反向依赖。
- `pipeline.py`：编排入库与问答，不承载底层算法。
- `cli.py`：接收命令与参数，调用对应流程。未来 API 同样调用流程层。
- `schemas.py` 与 `config.py`：提供公共数据契约与配置，不反向依赖业务模块。

每个 Python 文件中的模块文档字符串说明其职责、输入输出和边界。初始化文件不执行模型加载、网络请求或索引任务。

## 5.2 数据契约

```text
ParsedPage → Chunk → SearchResult → AnswerResult
```

页面保留从 1 开始的 PDF 物理页码。Chunk 使用 `pages` 保存来源页列表，以支持跨页内容；单页显示时可取唯一页码。印刷页码单独作为可选标签。章节和标题无法提取时留空。SearchResult 保留各路分数与排名，不用融合分数覆盖原始信息。

后文早期示例中的 `page` 表示单页情况，实际实现以本节 `pages` 契约为准。

## 5.3 开发顺序与当前状态

当前仅完成源码目录与职责占位说明，解析、检索、生成和评估均未实现。

1. 基础配置与数据结构，然后实现解析、清洗并检查结果。
2. 分块与元数据，再建立向量检索和 BM25。
3. RRF、重排、证据生成与引用。
4. 完成固定评估集与方法对比。指标定义可提前确定。
5. 动态上传、API、前端与后续扩展暂不创建实现。

CLI 是规划中的统一入口；后文 `python main.py` 为早期运行示意，正式实现时以已配置的 CLI 命令为准，不额外维护两套入口。

真实 `.env`、私密 PDF、生成索引和缓存不提交 Git。代码、配置示例和可公开的小型评估数据可以提交。依赖统一放在 `pyproject.toml`，使用 uv 管理并提交 `uv.lock` 与 `.python-version`。环境恢复运行 `uv sync --locked`，执行命令使用 `uv run`；详细步骤见 README.md。基础依赖为 PyMuPDF，开发依赖为 pytest，业务功能仍未实现。

---

# 6. Phase 0 — 项目骨架

## 目标

建立一个最小可运行 Python 项目。

## 任务

- 创建项目目录。
- 创建虚拟环境。
- 配置 `.env`。
- 准备一份金融 PDF。
- 创建统一配置文件。
- 建立基础数据结构。

## 完成标准

运行：

```bash
python main.py
```

能够成功加载配置和指定 PDF。

---

# 7. Phase 1 — PDF Parsing

## 目标

把 PDF 变成结构化内容。

第一版最少需要：

```text
page number
text
```

更理想：

```text
page
section
title
paragraph
table
```

## 输出示例

```json
[
  {
    "page": 18,
    "text": "The Company is exposed to..."
  }
]
```

## 必须考虑的问题

### Header / Footer

金融报告经常每页重复：

```text
Apple Inc.
2025 Form 10-K
Page 18
```

需要尽量移除重复内容。

### Broken Lines

PDF extraction 可能出现：

```text
The Company
expects revenue
to increase...
```

需要 normalizing。

### Page Boundaries

不能丢失页码。

---

## Phase 1 完成标准

输入：

```text
apple_10k.pdf
```

输出：

```text
data/parsed/apple_10k.json
```

并且：

- 可以看到每页正文。
- 页码正确。
- 大部分文本顺序正确。
- 没有明显大量重复 header/footer。

---

# 8. Phase 2 — Chunking

## 目标

把 parsed document 转成适合 Retrieval 的 chunk。

第一版不要只做：

```text
每 500 tokens 切一次
```

建议先支持：

```text
section-aware chunking
```

基本逻辑：

```text
Title
↓
Section
↓
Paragraph
↓
如果过长，再按 token 切
```

---

## Chunk 参数

初始可以测试：

```text
chunk_size = 400–700 tokens
overlap = 50–100 tokens
```

注意这些不是最终答案。

后续通过 evaluation 调整。

---

## 输出

```json
{
  "chunk_id": "apple_2025_p18_03",
  "page": 18,
  "section": "Risk Factors",
  "text": "..."
}
```

---

## Phase 2 完成标准

能够打印：

```text
Total chunks: 326

Chunk #105
Page: 18
Section: Risk Factors
Tokens: 512
Text: ...
```

人工抽查 20 个 chunk，基本具有完整语义。

---

# 9. Phase 3 — Dense Retrieval

## 目标

实现最基础的 Semantic Search。

流程：

```text
chunks
→ embedding
→ vector index

query
→ query embedding
→ cosine similarity
→ top-k
```

---

## API

希望最终形成：

```python
results = dense_search(
    query="What are Apple's major risks in China?",
    top_k=10
)
```

输出：

```python
[
    {
        "chunk_id": "...",
        "score": 0.82,
        "page": 18,
        "text": "..."
    }
]
```

---

## Phase 3 完成标准

至少准备 10 个问题。

人工检查：

```text
Top-5 中是否出现明显正确证据。
```

暂时不接 LLM。

---

# 10. Phase 4 — BM25 Retrieval

## 目标

建立 Sparse Retrieval。

目的：

Dense Search 擅长语义。

BM25 擅长：

- 财务指标名称
- 专有名词
- 股票代码
- 精确数字
- 产品名称
- 地区名称

例如：

```text
Greater China
R&D
10-K
Net sales
Operating cash flow
```

---

## API

```python
results = bm25_search(
    query="Greater China net sales 2025",
    top_k=10
)
```

---

## Phase 4 完成标准

同一个 query 同时打印：

```text
Dense Top-5

BM25 Top-5
```

人工比较结果差异。

---

# 11. Phase 5 — Hybrid Retrieval

## 目标

融合 Dense + BM25。

第一版推荐：

```text
Reciprocal Rank Fusion
```

RRF：

```text
score(d) =
Σ 1 / (k + rank(d))
```

不需要直接比较：

```text
cosine_score
BM25_score
```

因为两个 score 的尺度不同。

---

## API

```python
results = hybrid_search(
    query=query,
    dense_top_k=20,
    bm25_top_k=20,
    final_top_k=20
)
```

---

## Phase 5 完成标准

可以比较：

```text
Dense
BM25
Hybrid
```

并保存每个结果的来源：

```python
{
    "chunk_id": "...",
    "dense_rank": 4,
    "bm25_rank": 2,
    "rrf_score": ...
}
```

---

# 12. Phase 6 — Reranking

## 目标

Hybrid Search 先负责：

```text
Recall
```

Reranker 负责：

```text
Precision
```

流程：

```text
Dense Top 20
+
BM25 Top 20
↓
RRF
↓
Top 20 Candidates
↓
Cross Encoder Reranker
↓
Top 5
```

---

## API

```python
reranked = rerank(
    query=query,
    candidates=hybrid_results,
    top_k=5
)
```

---

## Phase 6 完成标准

输出：

```text
Before rerank:

1.
2.
3.
...

After rerank:

1.
2.
3.
...
```

至少人工观察 10 个问题，判断排序是否改善。

---

# 13. Phase 7 — LLM Generation

## 目标

基于 Retrieval Evidence 生成回答。

LLM 不允许自由发挥。

---

## Prompt 原则

Prompt 应明确：

```text
You must answer only using the provided evidence.

If the evidence is insufficient, say that the answer
cannot be confirmed from the document.

Every factual statement should be grounded in evidence.

Include page citations.
```

---

## Input

```text
Question:
...

Evidence 1:
Page 18
...

Evidence 2:
Page 42
...
```

---

## Output

例如：

```text
Apple identified several risks related to Greater China,
including regulatory uncertainty and competitive pressure.

Sources:
- p.18
- p.23
```

---

# 14. Phase 8 — Citation

## 目标

答案必须能追溯到 PDF。

最简单方案：

```text
[Page 18]
[Page 42]
```

更进一步：

```text
Document + Page + Section
```

例如：

```text
Apple 2025 10-K — Risk Factors — p.18
```

---

# 15. Phase 9 — Evaluation

这是整个项目最重要的部分之一。

不要只说：

```text
“感觉效果不错”
```

必须建立 evaluation dataset。

---

## Evaluation Dataset

建议第一版：

```text
30–50 questions
```

每个问题保存：

```json
{
  "question": "What was Apple's total revenue in 2025?",
  "expected_pages": [31],
  "expected_chunk_ids": [],
  "answer": "...",
  "category": "financial_metric"
}
```

---

## Question 类型

至少覆盖：

### Exact Retrieval

```text
Revenue是多少？
```

### Semantic Retrieval

```text
公司认为未来面临哪些主要风险？
```

### Keyword-heavy

```text
Greater China revenue
```

### Cross-section

```text
为什么利润增长但现金流下降？
```

### Negative Question

```text
报告有没有提到 Bitcoin treasury strategy？
```

正确行为可能是：

```text
没有足够证据。
```

---

# 16. Retrieval Evaluation Metrics

第一阶段重点：

## Recall@K

正确 chunk 是否出现在 Top-K。

例如：

```text
Recall@5
Recall@10
```

---

## MRR

正确结果排名越靠前越好。

```text
Mean Reciprocal Rank
```

---

## Hit Rate

例如：

```text
Top-5 是否至少命中一个正确证据。
```

---

# 17. Retrieval Benchmark

至少比较：

```text
1. BM25 only

2. Dense only

3. Hybrid

4. Hybrid + Reranker
```

输出表：

```text
Method                  Recall@5   Recall@10   MRR
-------------------------------------------------
BM25
Dense
Hybrid
Hybrid + Reranker
```

这张表是项目的重要成果。

---

# 18. Phase 10 — Dynamic PDF Upload

前面的版本可以固定：

```text
data/apple_10k.pdf
```

完成 Retrieval Benchmark 后再支持动态上传。

---

## 流程

```text
User uploads PDF
↓
Generate document_id
↓
Parse
↓
Chunk
↓
Embedding
↓
BM25 index
↓
Ready
```

---

## 需要增加

```text
document_id
session_id
user_id
```

避免不同用户数据混在一起。

---

# 19. Phase 11 — Backend API

将 RAG 封装成服务。

例如：

```text
POST /documents
POST /documents/{id}/index
POST /query
GET /documents/{id}
DELETE /documents/{id}
```

Query：

```json
{
  "document_id": "abc123",
  "query": "What are the major risks?"
}
```

Response：

```json
{
  "answer": "...",
  "sources": [
    {
      "page": 18,
      "chunk_id": "..."
    }
  ]
}
```

推荐：

```text
FastAPI
```

---

# 20. Phase 12 — Web Frontend

这一步再开始学习 Vue / React。

UI 第一版只需要：

```text
Upload PDF

[ Ask a question... ]

Answer

Sources

Retrieved Evidence
```

建议开发模式支持：

```text
Show Retrieval Debug Info
```

方便看到：

```text
BM25 Rank
Dense Rank
RRF Score
Rerank Score
```

---

# 21. Phase 13 — Enterprise RAG Extension

之后可以模拟企业内部 RAG。

给 chunk 增加：

```python
{
    "department": "finance",
    "permission": ["analyst", "admin"],
    "confidentiality": "internal"
}
```

Retrieval 前执行：

```text
Permission Filter
↓
Retrieval
```

不是：

```text
Retrieve Everything
↓
让 LLM 决定能不能看
```

权限必须在 Retrieval 层解决。

---

# 22. 数据安全方向

未来可以研究：

```text
Local parsing
Local vector database
Local retrieval
External LLM generation
```

只把：

```text
Top-K required evidence
```

发送给外部 LLM。

进一步可以尝试：

```text
Fully local LLM
```

用于敏感文档。

---

# 23. Phase 14 — Web3 / dApp Extension

最终可以把项目升级为：

# On-chain Research dApp

输入：

```text
Whitepaper
Audit Report
Financial Report
Protocol Documentation
+
On-chain Data
```

AI 综合分析：

```text
Documents
+
Blockchain Transactions
+
Smart Contract State
```

---

## dApp Architecture

```text
Wallet Login
      ↓
Frontend
      ↓
Backend API
      ↓
┌─────────────────────────────┐
│ Document Retrieval          │
│                             │
│ PDF → RAG                   │
└─────────────────────────────┘
      +
┌─────────────────────────────┐
│ On-chain Retrieval          │
│                             │
│ RPC / Indexer / Etherscan   │
└─────────────────────────────┘
      ↓
Analysis / Agent
      ↓
Answer
```

---

## 未来接口

可以提前设计：

```python
retrieve_from_documents(query)

retrieve_from_onchain_data(query)
```

未来增加：

```python
route_query(query)
```

判断：

```text
Document Question
On-chain Question
Both
```

---

# 24. 暂时不做的东西

为了避免项目失控，第一阶段明确不做：

```text
Agent
Multi-agent
Knowledge Graph
GraphRAG
Distributed Vector DB
Kubernetes
GPU Cluster
Fine-tuning
Complex Authentication
Blockchain
Smart Contract
Fancy Frontend
```

这些以后再加。

---

# 25. 第一阶段真正要解决的问题

整个项目最重要的问题是：

> 用户问一个自然语言问题后，系统如何从一份复杂金融 PDF 中稳定找到最相关的 5 段证据？

只要这个问题真正解决，RAG 核心就已经掌握。

---

# 26. Coding 原则

Codex 开发时需要遵循：

## Modular

每个组件独立：

```text
Parser
Chunker
Embedding
Dense Retriever
BM25 Retriever
Fusion
Reranker
Generator
Evaluator
```

不要全部写进一个 Python 文件。

---

## Replaceable

组件应该可替换。

例如：

```python
EmbeddingModel
```

之后可以：

```text
OpenAIEmbedding
BGEEmbedding
QwenEmbedding
```

Retriever 逻辑不应该因此全部重写。

---

## Observable

每一步能够 debug。

例如 query 时打印：

```text
Dense Candidates
BM25 Candidates
RRF Candidates
Reranked Candidates
Final Context
```

---

## Reproducible

实验参数放配置：

```python
DENSE_TOP_K = 20
BM25_TOP_K = 20
RERANK_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
```

不要散落在代码各处。

---

# 27. 第一版 Main Flow

最终希望：

```python
from financial_rag.pipeline import FinancialRAG

rag = FinancialRAG()

rag.index_document(
    "data/raw/apple_10k.pdf"
)

result = rag.query(
    "What are Apple's major risks in Greater China?"
)

print(result.answer)

for source in result.sources:
    print(
        source.page,
        source.section,
        source.text
    )
```

---

# 28. 开发顺序

严格按照：

```text
1. Parse PDF

2. Chunk

3. Dense Retrieval

4. BM25

5. Hybrid RRF

6. Reranker

7. LLM

8. Citation

9. Evaluation

10. Dynamic Upload

11. API

12. Frontend

13. Permission

14. Web3
```

不要反过来。

---

# 29. MVP 定义

第一版 MVP 必须满足：

```text
一份固定金融 PDF
+
Dense Retrieval
+
BM25
+
Hybrid Search
+
Reranker
+
LLM Answer
+
Page Citation
+
30+ Question Evaluation Dataset
+
Recall@5 / MRR Benchmark
```

达到这个程度后，项目才算完成第一阶段。

---

# 30. 最终简历价值

项目完成后应该可以描述成：

> Built a financial-document RAG system supporting PDF parsing, hybrid BM25 + dense retrieval, reciprocal rank fusion, cross-encoder reranking, evidence-grounded generation and page-level citations. Designed a retrieval evaluation benchmark comparing BM25, dense retrieval, hybrid retrieval and reranked retrieval using Recall@K and MRR.

未来升级后：

> Extended the system with dynamic document ingestion, user-level document isolation, permission-aware retrieval and on-chain data integration for AI-powered financial/Web3 research.

---

# 31. 给 Codex 的工作方式

不要让 Codex 一次生成整个系统。

建议每次只给一个阶段。

例如第一天：

```text
Read PROJECT_SPEC.md.

We are implementing Phase 1 only.

Do not implement embeddings, retrieval, LLM or frontend.

Create a clean PDF parsing module that:
1. accepts a PDF path,
2. extracts page-level text,
3. preserves page numbers,
4. removes obvious repeated headers/footers,
5. outputs structured JSON,
6. includes basic tests.

Explain architectural decisions before editing.
```

Phase 2：

```text
Read PROJECT_SPEC.md.

Phase 1 is complete.

Implement Phase 2 chunking only.

Requirements:
- preserve page metadata,
- preserve section/title metadata where available,
- configurable chunk size,
- configurable overlap,
- deterministic chunk IDs,
- unit tests.

Do not implement retrieval yet.
```

后续逐阶段推进。

---

# 32. 项目阶段 Checklist

## Foundation

- [ ] Project structure
- [ ] Config
- [ ] Sample financial PDF

## Document Pipeline

- [ ] PDF parsing
- [ ] Normalization
- [ ] Chunking
- [ ] Metadata

## Retrieval

- [ ] Embedding
- [ ] Vector index
- [ ] Dense search
- [ ] BM25 index
- [ ] BM25 search
- [ ] RRF
- [ ] Reranker

## Generation

- [ ] Prompt assembly
- [ ] LLM generation
- [ ] Evidence-only answering
- [ ] Page citations

## Evaluation

- [ ] 30–50 questions
- [ ] Ground-truth evidence
- [ ] Recall@5
- [ ] Recall@10
- [ ] MRR
- [ ] BM25 vs Dense vs Hybrid vs Rerank benchmark

## Productization

- [ ] Dynamic PDF upload
- [ ] Document isolation
- [ ] FastAPI
- [ ] Frontend
- [ ] Retrieval debug view

## Advanced

- [ ] Permission-aware retrieval
- [ ] Local/private retrieval
- [ ] On-chain data
- [ ] Wallet login
- [ ] Smart contract
- [ ] dApp

---

# 33. 项目成功标准

这个项目的成功标准不是：

> “网页能回答问题。”

而是你可以清楚解释：

1. PDF 为什么这样解析？
2. 为什么这样 chunk？
3. Dense Retrieval 为什么会漏掉某些结果？
4. BM25 为什么能补 Dense？
5. 为什么不能直接把 BM25 score 和 cosine score 相加？
6. RRF 在解决什么？
7. Reranker 为什么放在 retrieval 后面？
8. Top-K 为什么选择这个数字？
9. 如何证明 Retrieval 变好了？
10. LLM hallucination 如何降低？
11. Citation 如何和 chunk/page 对齐？
12. 如果用户上传私密文件，数据在哪里流动？
13. 如果数据量扩大 100 倍，哪个环节先成为瓶颈？
14. 将来如何把链上数据作为第二个 Retrieval Source？

如果这些问题都能回答，这个项目就已经实现了它真正的学习目标。
