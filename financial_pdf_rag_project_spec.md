# Financial PDF RAG 项目规格说明

> 项目定位：一个面向金融报告（年报、10-K、财报、研究报告等）的可评估 RAG 系统。  
> 核心目标不是“做一个聊天机器人”，而是完整实现并理解：**PDF 解析 → Chunking → Indexing → Retrieval → Fusion → Rerank → Evidence → LLM Answer → Evaluation**。  
> 后续可以继续升级为用户上传 PDF 的 Web 应用，并最终接入钱包、链上数据和智能合约，演化为 AI + Web3 dApp。

---

## 当前进度（2026-09-19）

已跑通：PDF → 文本块解析 → 清洗 → 分块 → 本地 Embedding → Dense Top-5。BM25 统计、评分与 Top-K 代码已编写，但 get_scores 方法归属错误尚未修复；融合、重排与 LLM 回答未实现。

| 模块 | 已完成 | 尚未完成或限制 |
|---|---|---|
| 环境与骨架 | uv、Python 3.11、源码包布局、依赖锁文件 | `config.py`、`schemas.py`、`pipeline.py`、`cli.py` 仍为职责占位 |
| PDF 解析 | PyMuPDF 文本块、bbox、块编号、从 1 开始的物理页码 | OCR、可靠的多栏阅读顺序、结构化表格 |
| 清洗 | 块内断行合并、块间空行、保守清除重复底部 10-K 页脚及纯 ®/™ 块 | 通用页眉页脚检测、标题识别；金融表格关系仍需抽查 |
| 分块 | 逐页按段落组合，长段落按词拆分；段落换行、元数据、word_count、稳定页内 ID | 当前 500/80 的单位是词而非 token；不跨页，不按章节识别边界 |
| Embedding | SentenceTransformers 本地 `BAAI/bge-small-en-v1.5`，文档/查询向量归一化 | tokenizer 长度与截断检查、编码策略验证、向量缓存 |
| Dense 检索 | NumPy 内存矩阵点积排序，返回 score/page/chunk_id/text | 持久化向量索引、索引配置兼容检查；`vector_store.py` 仍未实现 |
| BM25 | 已编写统计、评分与 Top-K | get_scores 在类外，调用未接通；尚无 BM25 测试 |
| 测试 | 清洗和分块 19 项；截图中的 8 题 Dense 流程测试 | 标准证据、相关性指标和正式对比实验 |

最近验证：Apple PDF 共 80 页，生成 132 个 chunks，最大 500 词；清理 58 个目标页脚块、16 个纯符号块。19 项单元测试通过，8 项实际模型测试通过，保存了 8 × Top-5 检索结果。上述数字是当前样本文档的验证记录，不是固定验收阈值。

**8 题测试通过只说明检索流程、排序与证据映射通过检查，不说明回答正确，也不代表 Recall/MRR 达标。**

> 下文 Phase 为长期目标；当前完成情况以本节和第 32 节为准，具体下一步见第 5.3 节。

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

当前采用 `src/financial_rag/` 包布局。文档处理、Embedding 和 Dense 已有实现；BM25 已编写但调用待修复，其余模块仍以职责占位为主。目录树是目标布局，不表示所有功能已经完成；当前执行入口为根目录 `main.py`。

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

模块职责说明是设计参考，少数已实现文件顶部仍保留早期占位描述；实际状态以本文件进度表及可运行代码为准。初始化文件不执行模型加载、网络请求或索引任务。

## 5.2 数据契约

```text
ParsedPage → Chunk → SearchResult → AnswerResult
```

当前实现使用字典。页面保存 `page`、`blocks` 和 `text`；Chunk 保存 `document_id`、`chunk_id`、单页 `page`、`word_count` 和 `text`。已有附加元数据会传递，但不会自动识别 section/title。检索结果包含 chunk_id/page/score/text。

当前不跨页，`page` 是从 1 开始的 PDF 物理页码，不能与印刷页码混淆。未来若支持跨页再引入 `pages`；`ParsedPage` 等类型、各路排名统一结构和 `AnswerResult` 尚未实现。

## 5.3 当前之后的执行顺序

### 1. 固定 Dense 的人工基线（当前最优先）

- 打开 `outputs/dense_retrieval_results.json`，逐题对照原 PDF，判断 Top-5 是否包含正确证据。
- 给 `data/eval/dense_questions.json` 增加经过核对的标准 PDF 物理页码和简短证据说明；不要根据当前检索结果反过来认定标准答案。
- 第 2、6 题未指定年份，标注前明确年份口径；“宣布的产品”不能直接用“当前产品清单”替代。
- 记录命中、漏检及原因，形成可比较的 Dense 基线；不要把当前 8 题称为完整评估集。
- 使用模型 tokenizer 检查各 chunk 的长度与实际模型输入上限，确认是否发生截断。500 词不等于 500 tokens；必要时调整分块并重新标注受影响的 chunk ID。

完成标准：8 题均有人工检查记录、核对后的证据页或明确的不确定项，并确认模型是否完整编码目标文本。

### 2. 修复 BM25 接口并与 Dense 对比

- 两个模块已编写基础逻辑；先将 get_scores 移入 BM25Index，再统一 main.py 的两路查询问题。补齐 BM25 单元测试，当前不具备已验证的可运行状态。
- 使用相同 chunks 和同一组问题，分别输出两种方法的 Top-5。
- 重点观察 Greater China、net sales、cash flow 等关键词与语义问题的区别。

完成标准：同一问题可直接比较两种排序，结果能追溯到相同来源；不直接相加 BM25 和余弦分数。

### 3. 实现 RRF，再接 Reranker

- `retrieval/fusion.py` 按排名融合候选，按文档与 chunk 去重，保留两路排名。
- `retrieval/retriever.py` 编排 Dense/BM25/Hybrid 模式。
- `retrieval/reranker.py` 重排候选并保留重排前后结果。

完成标准：相同问题能够比较 Dense、BM25、Hybrid、Hybrid + Reranker，并说明哪些证据排名改善或下降。

### 4. 补齐可复现评估与索引存储

- 逐步扩展到 30–50 道经过人工标注的问题，包括数值、语义、跨章节和负例。
- 实现 `evaluation/dataset.py`、`metrics.py`、`benchmark.py`；明确页级/块级评估粒度，计算 Hit Rate、Recall@K 和 MRR。
- 负例单独评估，不将空标准证据集合直接用于普通 Recall。
- 实现索引保存/加载，保存 chunks 摘要、模型标识、编码配置与维度；配置不一致时拒绝复用。
- 需要反复实验时，可提前做向量缓存，避免每次重新编码全部 chunks。

完成标准：同一数据和配置可重复运行，并输出四种检索方法的对比表。

### 5. 接入回答与引用，最后做产品化

检索证据稳定后实现 `generation/`：仅依据证据回答、证据不足时明确说明、引用与 PDF 页码对应。再逐步增加动态上传、API 和前端；权限、Web3 留待后续。不要将检索页码元数据等同于已经实现最终答案引用。

目前使用 `uv run main.py`，CLI 和统一 pipeline 待实现；依赖与运行方法以 README.md 为准。

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

## BM25 实现说明与当前状态（2026-09-19）

BM25 基础代码已编写，但尚未完成运行验证。`indexing/bm25_index.py` 负责文档分词、统计及评分；`retrieval/sparse.py` 负责排序、过滤零分并返回最多 K 条证据。不依赖 Embedding，也没有使用 rank_bm25 库。

### 统计和公式

每个 chunk 作为一篇文档，文档与查询统一使用小写化及 `[a-z0-9]+` 分词。

- TF(t,d)：词 t 在当前 chunk d 中的出现次数，由每个 chunk 的 Counter 统计。
- DF(t)：包含词 t 的 chunk 数；每个 chunk 先用 set 去重，最多贡献一次。
- N：chunk 总数；dl：当前 chunk 的分词数量；avgdl：平均分词数量。
- IDF 使用正值变体：`ln(1 + (N - DF + 0.5) / (DF + 0.5))`。
- 单词贡献：`IDF * TF * (k1 + 1) / (TF + k1 * (1 - b + b * dl / avgdl))`。
- 当前 `k1=1.5`、`b=0.75`，查询词去重后累加贡献；不匹配的词贡献为零。
- 按总分降序取 Top-K；过滤零分，所以不足 K 个匹配时返回更少结果，无匹配返回空列表。

当前分词为英文基础版：`R&D` 会拆成 r/d，`64,377` 会拆成 64/377，不支持中文分词。BM25 的分词长度与 chunker 的空白词数不同，不能混为一谈。当前索引仅保存在内存。

### 当前阻塞与修复顺序

1. **先把 `get_scores()` 放进 `BM25Index` 类中。** 当前它是模块级函数，但 sparse.py 调用 `index.get_scores(query)`，会触发 AttributeError。已有统计字段无需改成全局变量。
2. **统一 main.py 的 query。** 当前 Dense 查询大中华区风险，BM25 查询大中华区收入，不能据此比较两种算法。修复后向两种检索传入同一问题，分别打印方法名、排名、分数、chunk ID、页码与证据。
3. **补 BM25 单元测试。** 用小语料验证 TF/DF/IDF、重复查询词去重、词频饱和和长度调整、空语料、未知词、零分过滤、Top-K 排序与原文映射。
4. **用相同 8 题做双方法对比。** 复用 `data/eval/dense_questions.json` 和同一批 chunks，保存独立 BM25 报告；该批量测试目前还未实现。
5. 完成基础对比后再实现 RRF；两路原始分数不直接相加。

### 运行状态

`uv run main.py` 已加入 BM25 调用，但目前会在调用 get_scores 时失败，而且失败前仍会加载 Dense 模型并编码。修复上述方法归属后再用它验证双方法流程。

已有 Dense 测试独立于 main.py，仍可按以下命令单独运行（需已有 chunks 文件）：

```bash
uv run pytest tests/test_dense_retrieval.py --run-dense -s -v
```

现有 19 项清洗/分块测试与 8 题 Dense 测试不覆盖 BM25，不能作为 BM25 已完成的依据。


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

Top-K 中命中的标准相关证据数 / 全部标准相关证据数。须明确按页还是按 chunk 计数；“至少命中一个”属于 Hit Rate，不等于 Recall。

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

已勾选仅表示描述中的当前范围完成，不表示整个 Phase 已验收。

## Foundation

- [x] src 包布局、uv 环境与锁文件
- [x] 本地 Apple 样本 PDF
- [ ] 统一 Config、Schemas、Pipeline 和 CLI

## Document Pipeline

- [x] 逐页文本块解析与物理页码
- [x] 基础清洗、保守页脚/纯符号移除
- [x] 按段落组合及长段落词数分块，保留段落空行
- [x] ID、页码、word_count 与已有元数据传递
- [ ] tokenizer 预算与截断检查
- [ ] 自动章节/标题识别、结构化表格及 OCR

## Retrieval

- [x] 本地 BGE 文档与查询编码
- [x] NumPy Dense Top-K 检索
- [ ] 向量索引持久化与配置兼容检查
- [x] BM25 基础代码：TF/DF/IDF、评分公式与 Top-K（未完成运行验收）
- [ ] get_scores 方法归属修复、同题调用、BM25 单元测试与 8 题对比
- [ ] RRF 融合
- [ ] Reranker 与统一检索入口

## Tests and Evaluation

- [x] 19 项清洗/分块单元测试
- [x] 8 题真实 Dense 流程检查与 Top-5 报告
- [ ] 人工核对 8 题证据并固定基线
- [ ] 30–50 题与标准证据标注
- [ ] Recall@5、Recall@10、MRR、Hit Rate
- [ ] BM25 / Dense / Hybrid / Rerank 对比

## Generation and Productization

- [ ] Prompt、LLM、证据约束与拒答
- [ ] 最终答案引用映射与验证
- [ ] 动态上传、文档隔离、FastAPI、前端
- [ ] 权限控制、链上数据、钱包与 dApp

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
