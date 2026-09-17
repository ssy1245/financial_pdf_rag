# Financial PDF RAG

面向金融 PDF 的检索与问答学习项目。完整长期目标见 [项目规格说明](financial_pdf_rag_project_spec.md)。

## 当前进度（2026-09-17）

已跑通：PDF → 文本块解析 → 清洗 → 分块 → 本地 Embedding → Dense Top-5。尚未接入 BM25、融合、重排或 LLM 回答，第一阶段 MVP 尚未完成。

| 模块 | 已完成 | 尚未完成或限制 |
|---|---|---|
| 环境与骨架 | uv、Python 3.11、源码包布局、依赖锁文件 | `config.py`、`schemas.py`、`pipeline.py`、`cli.py` 仍为职责占位 |
| PDF 解析 | PyMuPDF 文本块、bbox、块编号、从 1 开始的物理页码 | OCR、可靠的多栏阅读顺序、结构化表格 |
| 清洗 | 块内断行合并、块间空行、保守清除重复底部 10-K 页脚及纯 ®/™ 块 | 通用页眉页脚检测、标题识别；金融表格关系仍需抽查 |
| 分块 | 逐页按段落组合，长段落按词拆分；段落换行、元数据、word_count、稳定页内 ID | 当前 500/80 的单位是词而非 token；不跨页，不按章节识别边界 |
| Embedding | SentenceTransformers 本地 `BAAI/bge-small-en-v1.5`，文档/查询向量归一化 | tokenizer 长度与截断检查、编码策略验证、向量缓存 |
| Dense 检索 | NumPy 内存矩阵点积排序，返回 score/page/chunk_id/text | 持久化向量索引、索引配置兼容检查；`vector_store.py` 仍未实现 |
| 测试 | 清洗和分块 19 项；8 题 Dense 流程测试及一轮结果人工检查 | 标准证据、相关性指标和正式对比实验 |

2026-09-16 全流程验证：Apple PDF 共 80 页，生成 132 个 chunks，最大 500 词；清理 58 个目标页脚块、16 个纯符号块。19 项单元测试通过，8 项实际模型测试通过，保存了 8 × Top-5 检索结果。上述数字是当前样本文档的验证记录，不是固定验收阈值。

2026-09-17 复查：普通测试运行结果为 **19 passed、8 skipped**；8 题模型测试需显式启用，本次文档更新没有重新编码或覆盖检索报告。

**8 题测试通过只说明检索流程、排序与证据映射通过检查，不说明回答正确，也不代表 Recall/MRR 达标。**

## Dense 基线的人工检查

已检查 `outputs/dense_retrieval_results.json` 中 8 个问题的返回证据。以下是 2026-09-16 那次运行的人工观察，页码均为 PDF 物理页码，并非正式评估指标。

| 问题 | 返回证据观察 |
|---|---|
| 2025 总营收 | Top-1，第 26 页包含总营收、年份及单位 |
| 大中华区营收 | Top-4，第 51 页包含直接数值；前三名偏向整体/产品营收 |
| 大中华区营收变化原因 | Top-3，第 25 页包含 iPhone 销售下降、Mac 增长部分抵消的解释 |
| 供应链风险 | Top-1，第 9 页有灾害和生产中断风险；Top-5 第 11 页补充外包依赖 |
| 网络安全风险 | Top-1，第 15 页有直接相关证据，但 Top-3 混入目录页 |
| 经营活动现金流 | Top-1，第 36 页现金流量表 |
| 2025 财年发布的产品 | Top-1，第 24 页按季度列出发布信息；后续结果混入财务报表 |
| 竞争风险 | Top-1，第 6 页有直接相关证据；第 10 页补充竞争风险 |

结论：Dense 能找到有用证据，地区限定和“为什么”类问题的排序仍有改善空间。此表仅根据返回文本判断相关性，未建立完整标准证据集合；风险类问题命中一段证据不等于覆盖全部风险。分数是相似度，不是答案正确概率。

当前可以保留这版 Dense 作为实验基线，下一开发任务是 BM25。人工标准证据标注和 tokenizer 截断检查仍需补齐。

## 环境与运行

```bash
uv sync --locked
```

开发解释器由 `.python-version` 固定为 Python 3.11。PyCharm 选择项目 `.venv/bin/python`。依赖由 `pyproject.toml` 和 `uv.lock` 管理；核心运行依赖为 PyMuPDF、NumPy、SentenceTransformers，pytest 为开发依赖。LangChain 已声明，但当前检索核心由显式代码实现。

将样本 PDF 放在 `data/raw/apple-10k.pdf`（本地数据默认不提交 Git），然后运行：

```bash
uv run main.py
```

当前 `main.py` 会解析和清洗 PDF、生成 chunks、重新编码文档，并打印一个固定问题的 Top-5 预览；不是交互式问答应用。首次加载 Embedding 模型可能下载模型文件，后续推理在本地进行。

输出文件：

- `data/parsed/apple-10k_normalized.json`：清洗后的逐页文本和文本块。
- `data/chunks/apple-10k_chunks.json`：分块正文与元数据。
- 当前没有保存文档向量，重新运行主流程会再次编码。

## 测试与人工检查

```bash
# 单元测试；实际模型测试默认跳过
uv run pytest -q

# 运行 8 题真实检索，逐题打印完整 Top-5
uv run pytest tests/test_dense_retrieval.py --run-dense -s -v
```

模型测试读取已生成的 chunks JSON，因此修改解析/清洗/分块后，应先重新运行 `main.py`。8 个问题共用一次文档编码。结果保存到 `outputs/dense_retrieval_results.json`，包含模型名称、chunks SHA-256 和完整证据。

自动检查向量归一化、返回数量、分数排序、ID 唯一性及证据映射，不检查人工相关性。当前报告是一次运行结果，不是正式评估成绩。

## 实现入口与模块

```text
main.py                                 # 当前主流程与单问题 Dense 演示
src/financial_rag/
  ingestion/parser.py                   # PDF 文本块提取
  ingestion/normalizer.py               # 块内清洗、目标噪声移除
  ingestion/chunker.py                  # 段落组合与长段落分块
  indexing/embeddings.py                # 本地模型编码
  retrieval/dense.py                    # NumPy 点积 Top-K
  indexing/bm25_index.py                # 下一步：BM25 索引（待实现）
  retrieval/sparse.py                   # 下一步：BM25 检索（待实现）
data/eval/dense_questions.json          # 8 个检索问题，尚无标准证据标注
tests/test_normalizer.py                # 清洗单元测试
tests/test_chunker.py                   # 分块单元测试
tests/test_dense_retrieval.py            # 可选的真实模型测试
```

其余索引存储、融合、重排、生成、正式评估与统一配置模块主要为架构占位。长期目标参考项目规格说明，当前可运行范围以本 README 为准。

## 当前数据约定

- 使用字典传递数据，尚未接入 `schemas.py` 的规划类型。
- `page` 是从 1 开始的 PDF 物理页码，与报告印刷页码可能不同。
- Chunk 不跨页，保留 `document_id`、`chunk_id`、`page`、`word_count`、`text`，传递已有元数据但不复制整页 `blocks`。
- `chunk_size=500`、`overlap=80` 按词数计量；普通段落边界可减少 overlap 以维持大小限制，独立长段落和页边界不额外重叠。
- `section/title` 不会自动识别；源码目录存在不表示模块已实现。

## 接下来做什么

### 1. 实现 BM25，与当前 Dense 基线对比（下一开发任务）

- 在 `indexing/bm25_index.py` 实现分词与索引，在 `retrieval/sparse.py` 提供搜索入口；目前两个文件仍为占位。
- 文档和查询使用相同分词规则，关注 Greater China、net sales、cash flow 等金融术语。
- 先使用现有 132 个 chunks 和 `data/eval/dense_questions.json` 中原有的 8 题，保存 BM25 Top-5，保留原 Dense 报告供对比。
- 对照分数、页码、chunk ID 和完整证据；重点查看大中华区收入/变化原因能否排得更靠前、目录等噪声是否减少。
- BM25 和余弦分数尺度不同，不直接相加；也不预先假定 BM25 一定优于 Dense。

完成标准：同一批 chunks、同一组问题可查看两种方法的 Top-5，并记录各自命中和漏检。暂不接 LLM、不重写整套解析流程。

### 2. 补齐评估口径与 Embedding 输入检查

- 对照原 PDF，为 8 个问题标注标准物理页码与证据说明；人工检查表只是初步观察，不能代替标准标注。
- 第 2、6 题未指定年份，正式评估前明确年份；“发布的产品”不能以“当前产品清单”替代。
- 若修改问题，Dense 与 BM25 都需重新运行；若调整分块，重新编码并检查相关 chunk ID 标注，不能与旧报告直接比较。
- 使用模型 tokenizer 检查 chunk 长度和模型输入上限。500 词不等于 500 tokens，是否截断尚未验证。
- 记录数据版本、模型和配置，避免把分块变化造成的影响归因于检索算法。

完成标准：问题口径一致、有经过核对的标准证据，并明确模型是否完整编码目标文本。

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

## Git 与依赖维护

提交源码、测试、文档、`pyproject.toml`、`uv.lock`、`.python-version` 和可公开的评估标注。不提交 `.venv`、真实 `.env`、本地 PDF、生成索引、缓存和结果目录。

```bash
uv add <package-name>
uv add --dev <package-name>
uv remove <package-name>
```

依赖变更后将 `pyproject.toml` 与 `uv.lock` 一起提交，不另外手工维护 requirements.txt。
