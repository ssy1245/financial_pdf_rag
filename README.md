# Financial PDF RAG

面向金融 PDF 的检索与问答学习项目。完整长期目标见 [项目规格说明](financial_pdf_rag_project_spec.md)。

## 当前进度（2026-09-16）

已跑通：PDF → 文本块解析 → 清洗 → 分块 → 本地 Embedding → Dense Top-5。尚未接入 BM25、融合、重排或 LLM 回答，第一阶段 MVP 尚未完成。

| 模块 | 已完成 | 尚未完成或限制 |
|---|---|---|
| 环境与骨架 | uv、Python 3.11、源码包布局、依赖锁文件 | `config.py`、`schemas.py`、`pipeline.py`、`cli.py` 仍为职责占位 |
| PDF 解析 | PyMuPDF 文本块、bbox、块编号、从 1 开始的物理页码 | OCR、可靠的多栏阅读顺序、结构化表格 |
| 清洗 | 块内断行合并、块间空行、保守清除重复底部 10-K 页脚及纯 ®/™ 块 | 通用页眉页脚检测、标题识别；金融表格关系仍需抽查 |
| 分块 | 逐页按段落组合，长段落按词拆分；段落换行、元数据、word_count、稳定页内 ID | 当前 500/80 的单位是词而非 token；不跨页，不按章节识别边界 |
| Embedding | SentenceTransformers 本地 `BAAI/bge-small-en-v1.5`，文档/查询向量归一化 | tokenizer 长度与截断检查、编码策略验证、向量缓存 |
| Dense 检索 | NumPy 内存矩阵点积排序，返回 score/page/chunk_id/text | 持久化向量索引、索引配置兼容检查；`vector_store.py` 仍未实现 |
| 测试 | 清洗和分块 19 项；截图中的 8 题 Dense 流程测试 | 标准证据、相关性指标和正式对比实验 |

最近验证：Apple PDF 共 80 页，生成 132 个 chunks，最大 500 词；清理 58 个目标页脚块、16 个纯符号块。19 项单元测试通过，8 项实际模型测试通过，保存了 8 × Top-5 检索结果。上述数字是当前样本文档的验证记录，不是固定验收阈值。

**8 题测试通过只说明检索流程、排序与证据映射通过检查，不说明回答正确，也不代表 Recall/MRR 达标。**

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

## 当前数据约定

- 使用字典传递数据，尚未接入 `schemas.py` 的规划类型。
- `page` 是从 1 开始的 PDF 物理页码，与报告印刷页码可能不同。
- Chunk 不跨页，保留 `document_id`、`chunk_id`、`page`、`word_count`、`text`，传递已有元数据但不复制整页 `blocks`。
- `chunk_size=500`、`overlap=80` 按词数计量；普通段落边界可减少 overlap 以维持大小限制，独立长段落和页边界不额外重叠。
- `section/title` 不会自动识别；源码目录存在不表示模块已实现。

## 接下来做什么

### 1. 固定 Dense 的人工基线（当前最优先）

- 打开 `outputs/dense_retrieval_results.json`，逐题对照原 PDF，判断 Top-5 是否包含正确证据。
- 给 `data/eval/dense_questions.json` 增加经过核对的标准 PDF 物理页码和简短证据说明；不要根据当前检索结果反过来认定标准答案。
- 第 2、6 题未指定年份，标注前明确年份口径；“宣布的产品”不能直接用“当前产品清单”替代。
- 记录命中、漏检及原因，形成可比较的 Dense 基线；不要把当前 8 题称为完整评估集。
- 使用模型 tokenizer 检查各 chunk 的长度与实际模型输入上限，确认是否发生截断。500 词不等于 500 tokens；必要时调整分块并重新标注受影响的 chunk ID。

完成标准：8 题均有人工检查记录、核对后的证据页或明确的不确定项，并确认模型是否完整编码目标文本。

### 2. 实现 BM25 并与 Dense 对比

- 在 `indexing/bm25_index.py` 实现分词和索引，在 `retrieval/sparse.py` 提供检索入口。
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

## Git 与依赖维护

提交源码、测试、文档、`pyproject.toml`、`uv.lock`、`.python-version` 和可公开的评估标注。不提交 `.venv`、真实 `.env`、本地 PDF、生成索引、缓存和结果目录。

```bash
uv add <package-name>
uv add --dev <package-name>
uv remove <package-name>
```

依赖变更后将 `pyproject.toml` 与 `uv.lock` 一起提交，不另外手工维护 requirements.txt。
