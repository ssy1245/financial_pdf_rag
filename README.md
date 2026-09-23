# Financial PDF RAG

面向金融 PDF 的可评估检索学习项目。长期目标见 [项目规格说明](financial_pdf_rag_project_spec.md)；当前实现状态以本 README 为准。

## 当前进度（2026-09-23）

已跑通 PDF → 文本块解析 → 清洗 → 分块 → Dense / BM25 → RRF Hybrid，并完成同一组 8 题的三路对比。已接入 Cross-Encoder Reranker；尚未实现 LLM 回答或正式相关性指标，MVP 尚未完成。

| 模块 | 当前实现 | 限制 |
|---|---|---|
| parser | PyMuPDF 文本块、bbox、块 ID、物理页码和 JSON 保存 | 无 OCR、结构化表格或可靠的复杂多栏处理 |
| normalizer | 块内清洗、块间空行、保守移除重复底部 10-K 页脚与纯商标符号块 | 不自动识别标题或恢复表格关系 |
| chunker | 段落组合、长段落切分、元数据与段落边界保留 | 按词数而非 token 计量；不跨页 |
| embeddings | 本地 BAAI/bge-small-en-v1.5，归一化 NumPy 向量 | 无缓存；tokenizer 截断尚未核对 |
| dense | 内存矩阵点积 Top-K | 无持久化向量索引 |
| bm25_index / sparse | TF/DF/IDF、长度调整、正分 Top-K | 基础英文分词，无持久化 |
| fusion | 按 chunk_id 累加 RRF，保留两路排名 | 假设各路 ID 唯一；同分按插入顺序；未做独立防御校验 |
| compare_methods | 8 题 Dense/BM25/Hybrid 对比、完整候选及 JSON 报告 | 无标准证据标注，不计算 Recall/MRR |

config、pipeline、统一 cli、vector_store、retriever、generation 和正式 evaluation 指标模块仍为规划占位。源码文件顶部已区分当前行为与规划职责。

## 环境和运行

使用 uv 管理环境，开发 Python 版本由 `.python-version` 固定为 3.11。

```bash
uv sync --locked
```

PyCharm 解释器选择项目 `.venv/bin/python`。核心依赖为 PyMuPDF、NumPy、SentenceTransformers；pytest 用于测试。LangChain 虽已声明，当前检索流程为显式实现。

将 PDF 放在 `data/raw/apple-10k.pdf`，运行：

```bash
uv run main.py
```

main.py 重新解析、清洗、分块、编码文档，然后用同一个 query 打印 BM25 和 Dense Top-5。它不展示 Hybrid，也不是交互式问答；BM25 打印完整正文，Dense 仅预览前 1000 字符。首次加载模型可能下载文件，后续推理在本地进行。

输出：

- `data/parsed/apple-10k_normalized.json`：清洗后的页面和文本块。
- `data/chunks/apple-10k_chunks.json`：供后续检索使用的 chunks。

## 8 题三路批量对比

```bash
uv run python -m financial_rag.evaluation.compare_methods --candidate-k 20 --top-k 5 --rrf-k 60
```

读取既有 chunks 和 `data/eval/dense_questions.json`。文档向量只编码一次，BM25Index 只建立一次；随后逐题召回、融合，打印三路完整证据。

| 参数 | 含义 |
|---|---|
| `--candidate-k 20` | Dense/BM25 各自最多召回 20 条，全部参与融合 |
| `--top-k 5` | 每种方法最终展示 5 条；BM25 正分匹配不足时允许更少 |
| `--rrf-k 60` | 排名平滑常数，不是返回数量 |
| `--output` | 自定义报告路径；默认报告重复运行时会更新 |

要求 candidate-k >= top-k > 0，rrf-k >= 0。可用 --chunks、--questions、--model 替换输入或模型。

新报告为 `outputs/dense_bm25_hybrid_comparison.json`，保留旧的 Dense 和双方法报告。每题包含：

- `dense`、`bm25`、`hybrid`：三路最终 Top-K。
- `candidates`：完整的两路候选，可追溯原始分数和排名。
- Hybrid 的 `rrf_score`、`dense_rank`、`bm25_rank`；缺席一路时排名为 null。

报告同时记录模型名称、归一化配置、BM25 参数、召回数量、RRF 参数、输入 SHA-256 和生成时间。没有标注标准答案，因此分数与排序不是准确率。

## 本次 Hybrid 报告检查

参数为两路各 20 候选、最终 Top-5、RRF k=60。检查 8 题，每题三路各返回 5 条。以下为返回文本与排名的人工观察，页码为 PDF 物理页码，不是正式指标。

| 问题 | Hybrid 结果与比较 |
|---|---|
| 总营收 | 第 39 页排第一，仍包含总营收、年份和单位；Dense 第 26 页降至第三 |
| 大中华区营收 | 第 51 页直接数值证据升至第二，但第一名第 39 页主要是产品收入；BM25 第一的第 25 页未进 Hybrid Top-5 |
| 大中华区收入变化原因 | 第 25 页解释升至第一；优于本次 Dense 的第三，与 BM25 第一一致 |
| 供应链风险 | 第 9 页灾害/生产中断证据第一，第 11 页外包依赖第二；候选内容具有互补性 |
| 网络安全风险 | 治理说明第 20 页第一，目录页第 3 页升至第二；Dense 第一的第 15 页具体风险降至第五 |
| 经营现金流 | 第 36 页在三种方法中均为第一 |
| 财年产品发布 | 第 24 页在三种方法中均为第一，但 Hybrid 后续仍混入目录等噪声 |
| 竞争风险 | 第 11 页产品推出风险第一，第 10 页竞争说明第二；Dense 第一的第 6 页竞争因素降至第五 |

RRF 会偏向两路共同排在前面的结果，不保证相关性更好。例如大中华区营收问题：

- 第 25 页：Dense 第 18、BM25 第 1，RRF = 1/78 + 1/61，约 0.029214。
- 第 39 页：Dense 第 2、BM25 第 4，RRF = 1/62 + 1/64，约 0.031754。

因此第 39 页排在前面，即使第 25 页更直接回答问题。网络安全的目录页也因两路都召回而被抬高。这是本次融合的局限，不能把 Hybrid 视为天然优于单路的方法。

## 测试与验证记录

```bash
# 默认不加载模型
uv run pytest -q

# 独立 Dense 8 题模型检查
uv run pytest tests/test_dense_retrieval.py --run-dense -s -v
```

最近普通测试为 **25 passed、8 skipped**；此前 Dense 8 题实际模型测试通过，最新三路批量运行也已成功。模型测试跳过不表示失败，需显式启用。已逐项核对 Hybrid 报告中两路排名与 RRF 分数；不等同于完成相关性评估。

- test_normalizer.py / test_chunker.py：清洗、分块及边界情况。
- test_bm25.py：实例评分与搜索基础回归。
- test_compare_methods.py：同题查询、一次编码、缺少 BM25 匹配、融合使用展示范围外的候选。
- test_dense_retrieval.py：真实模型的向量归一化、排序及来源映射。

当前样本为 80 页、132 chunks，最大 500 词。此前清洗验证删除 58 个目标页脚块和 16 个纯符号块；这些是样本观察，不是固定验收阈值。

## 数据和评分约定

- schemas.py 已用 TypedDict 描述当前字典结构（不做运行时校验）；现有模块尚未全部接入类型注解。page 从 1 开始；PDF 物理页码和报告印刷页码可能不同。
- chunk_id 在当前文档范围内唯一，包含文档标识、页码和页内序号；重分块后 ID 可能变化。
- chunk_size=500、overlap=80 按空白词数计量。普通段落 overlap 可减少以维持上限；独立长段落与页边界不额外重叠。
- BM25 文档与查询统一小写并使用 `[a-z0-9]+` 分词；R&D、64,377 会拆开，不支持中文分词。TF 是块内出现次数，DF 是包含该词的 chunk 数。
- BM25 IDF 为 `ln(1+(N-df+0.5)/(df+0.5))`，k1=1.5、b=0.75；重复查询词去重，零分结果过滤。
- RRF 只使用从 1 开始的排名，缺席一路贡献为零，不相加两路原始分数。
- section/title 尚未自动识别，表格关系与模型截断仍需检查。

## 下一步

1. **固定当前三路基线并补标准标注。** 对照原 PDF 核对 8 题的证据页，明确大中华区营收、经营现金流问题的年份；记录目录误召回和混合主题块。不要只凭检索结果自动生成标准答案。
2. **验证 Embedding 长度与 RRF 边界。** 用 tokenizer 检查实际输入上限；500 词不是 500 tokens。补 RRF 单路缺席、空列表、重复 ID、非法参数与同分处理测试。若调整分块或问题，三路全部重跑。
3. **改进并验证已接入的 Reranker。** 将 RRF 融合后的候选保留约 20 条，再以“问题 + 候选正文”重排并取 5 条，比较 Hybrid 与 Hybrid + Reranker；不要只给重排器现有最终 Top-5。重点验证目录页是否下降、直接证据是否上升。
4. **完善评估和存储。** 扩展到 30–50 道标注题，实现页级或块级 Hit Rate/Recall@K/MRR，负例单独处理；保存索引/向量缓存与兼容性信息。
5. **再接回答与引用。** 仅依据证据回答并支持证据不足说明，之后再做上传、API、前端与后续权限/Web3 扩展。

当前不根据这 8 题宣称某种算法整体更好，也不为改善单题排名盲调 RRF 参数。

## Git 与依赖

提交源码、测试、README、pyproject.toml、uv.lock、.python-version 和可公开评估标注。不提交 .venv、真实 .env、本地 PDF、生成索引、缓存和 outputs。

```bash
uv add <package-name>
uv add --dev <package-name>
uv remove <package-name>
```

依赖变化时一并提交 pyproject.toml 和 uv.lock。

## Reranker 接入（2026-09-23）

默认批量命令现已执行四路对比：

```bash
uv run python -m financial_rag.evaluation.compare_methods
```

每路召回 20 条，RRF 保留 20 条供 Cross-Encoder 重排，最终输出 5 条。模型为 cross-encoder/ms-marco-MiniLM-L6-v2，循环外加载一次。新报告为 outputs/dense_bm25_hybrid_reranked_comparison.json，不覆盖旧三路基线。

用 --skip-reranker 可回到三路；--reranker-model 可更换模型。当前默认文本对上限为 512 tokens，超限会警告并在每题 reranker_truncated_chunk_ids 中记录。报告正文为原始全文，模型输入可能被截断，重排效果需结合这一限制判断。尚未实现长文本分窗；分数不表示正确概率。

## 四路报告人工检查（2026-09-23）

已检查 `outputs/dense_bm25_hybrid_reranked_comparison.json`：8 题、每题 RRF 候选 20 条、重排后 5 条。此次模型为 cross-encoder/ms-marco-MiniLM-L6-v2，文本对上限 512 tokens。以下描述是当前样本的人工观察，不是正式准确率。

| 问题 | 重排后观察 |
|---|---|
| 总营收 | 第 32 页利润表升至第一，仍有直接数值证据 |
| 大中华区营收 | 第一名仍为第 39 页产品收入；第 51 页直接数值证据从 Hybrid 第二降至第五，未改善 |
| 大中华区收入变化原因 | 第 25 页仍为第一，但该候选超限；不能据返回全文断言模型看到了末尾原因说明 |
| 供应链风险 | 第 9 页灾害/生产中断仍第一，外包风险第 11 页从第二降至第三 |
| 网络安全风险 | 治理说明仍第一、目录仍第二；第 15 页具体风险从 Hybrid 第五掉出 Top-5，出现退步 |
| 经营现金流 | 第 36 页仍第一；该文本对超限，需要检查关键行是否处于模型保留范围 |
| 财年产品发布 | 第 24 页仍第一；后续混入利润表和目录，噪声未完全消除 |
| 竞争风险 | 第 6 页直接竞争因素从 Hybrid 第五升至第一，但它也被标为超限 |

每题超限候选数依次为 11、10、12、16、9、13、9、11，总计 **91/160 次问题—候选配对**（不是 91 个不同 chunks）。截断可能影响排序，但当前报告不能证明它是所有退步的原因。rerank_score 出现负数是允许的，不能套用 BM25 的“零分过滤”，也不能把分数直接视为回答置信度。

### 当前优先事项

1. 保留当前四路报告作为基线，不因个别题改善就判定重排整体有效。
2. 优先核对第 25 页和其他长候选的 token 截断位置，再选择按 token 分块或对长候选分窗评分；不要只把 max_length 调到超过模型支持范围。
3. 给 8 题补标准证据，并明确未指定年份的问题口径；单独标记目录/业务介绍等弱证据。
4. 修改长度策略后重新运行全部四路，再比较直接证据排名与噪声；之后才考虑模型替换和 LLM 回答。

### schemas.py 的当前契约

已定义 TextBlock、ParsedPage、Chunk、SearchResult、FusionResult、RerankedResult，以及带展示 rank 的结果类型、QueryComparison、ComparisonReport 和配置类型。

- 单路分数为 score，融合为 rrf_score，重排为 rerank_score。
- dense_rank/bm25_rank 是候选排名，缺席为 null；rank 是当前方法输出排名。
- candidates.rrf 是实际重排输入；reranked 是最终输出。
- reranker_truncated_chunk_ids 是每题超限候选列表，不只包含最终结果。JSON 中保存的是原始全文，不是截断后的模型输入。
- TypedDict 只描述字段，不做自动转换或运行时校验；AnswerResult、跨页 pages 尚未实现。


### SQLite 索引执行入口

`index_demo.py` 将现有 chunks 的向量持久化到 `storage/vectors.sqlite3`。
SQLite 负责保存，Dense 仍使用 NumPy 计算相似度；BM25 从同一批 chunks 建立内存索引。

```bash
# 首次入库：仅在这里编码全部文档，并验证保存/加载结果一致
uv run index_demo.py build

# 查询：加载已保存向量，仅编码当前问题
uv run index_demo.py search "What were Apple's total net sales in 2025?"

# chunks 或 embedding 模型改变后整批重建（替换该数据库全部旧记录）
uv run index_demo.py build --replace
```

两个命令均支持 `--db` 和 `--model`，查询模型必须与入库模型相同。
该入口目前输出 Dense/BM25 结果，未接 RRF、重排和生成；`generate_demo.py`
仍从保存的重排报告生成回答。下一步在 pipeline 中整合实时检索与生成。


### 实时完整问答 Pipeline

`FinancialRAG` 复用 SQLite 文档向量和 BM25 索引，每个问题执行 Dense/BM25
各 Top 20 → RRF Top 20 → BGE 重排 Top 5 → DeepSeek → 引用编号检查。
它直接检索当前问题，不读取之前的重排报告。文档入库仍使用 `index_demo.py build`。

```bash
uv run --env-file .env python -m financial_rag.pipeline \
  "What were Apple's total net sales in 2025?" \
  --output outputs/pipeline_answer.json
```

默认重排使用 `BAAI/bge-reranker-v2-m3`、2048 tokens，生成使用 `deepseek-flash`。
终端展示答案与引用，`--output` 保存所有阶段候选和配置。未知引用会标记并使命令失败；
无引用会提示人工检查。编号有效不代表事实获得证据支持，尚未实现语义事实核验。
该流程依赖已建立的 SQLite 索引；未实现增量入库、交互会话或问答标准指标评估。


### PDF 拖拽上传与网页问答

```bash
uv run --env-file .env streamlit run streamlit_app.py --server.address 127.0.0.1 --server.maxUploadSize 20
```

打开终端显示的本地地址，拖入 PDF → 点击「开始阅读」→ 输入问题。
网页复用 FinancialRAG，支持回答历史、引用页码和展开原文。每个浏览器会话使用
独立的临时 SQLite 索引；替换/移除文件时清除旧索引及问答，不覆盖 storage 中的索引。
模型共享加载并串行执行；首次启动可能需要下载模型。密钥只从服务端环境变量读取。
最多上传 20 MB，扫描件无 OCR；当前 embedding 更适合英文文档。
历史仅展示，每题独立检索；提问时相关文档片段会发送给 DeepSeek。
这是本地单机界面，尚未实现账户、公共部署或多用户并发优化。


### HTML 多 PDF 网页（推荐）

运行 `uv run app.py`，打开 http://127.0.0.1:8000；macOS 也可双击 `start.command`。
自动从项目 `.env` 读取 DeepSeek 密钥。前端为 `templates/index.html` 和
`static/style.css`、`static/app.js`，无需 Node 构建；HTML 必须通过 Python 服务打开。
一次选择或拖入 1–10 份 PDF，每份最多 20 MB，合计 100 MB，点击开始阅读后统一索引。
同内容自动去重，引用展示文件名与 PDF 页码。新一批成功入库才替换旧资料库；失败保留旧数据。
资料库按浏览器会话隔离，临时存储，服务重启后需重新上传；清空会删除当前临时索引。
当前跨文档使用全局 Top 5 证据，不保证每份文件都有证据；比较问题请明确文件和指标。
旧版 Streamlit 保留在 `streamlit_app.py`。当前是本地开发服务，不面向公开部署。
