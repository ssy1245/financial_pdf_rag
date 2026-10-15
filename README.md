# Financial PDF RAG

面向财报的 RAG 学习项目，当前已完成 PDF / Word (.docx) / Excel (.xlsx) 混合上传、结构化解析、检索、重排、DeepSeek 生成与来源展示。
检索层已拆为独立 Retriever，支持按 chunk_id 排除已获取片段。
已实现请求内证据记忆、search_documents 工具和显式 Agent loop；ask_agent() 可由模型选择补充检索。
默认 ask() 保留单轮基线；网页已使用带单条初始查询规划的 ask_agent()。

## 当前开发基线（2026-10-15 固定）

先固定核心参数，后续开发其他功能；暂不继续搜索超参数。主网页 `app.py` 与问答命令行共用 `src/financial_rag/config.py` 的 `RETRIEVAL_DEFAULTS`。

| 参数 | 当前选择 |
|---|---|
| 主网页分块 | 300 tokens，overlap 0，沿用现有结构化分块 |
| Dense / BM25 各路候选 | 各 20 |
| RRF | k=60，融合后保留 20 |
| 重排 | BGE-reranker-v2-m3，max_length=2048 |
| 返回证据 | 最多 7 条，rerank_score ≥ 0.1；不足不补满，可返回空集 |
| MMR | 不启用 |
| 查询与补查 | 保持单条初始查询规划 + 原有多轮补查；最多 8 次搜索、7 步 |

选择依据：50题检索实验中20/7比20/5覆盖更好，增加到20/10未增加覆盖；同10题选择实验中阈值0.1移除10块无关内容，保留全部原有直接/辅助证据，正文tokens减少约13.2%。更高阈值会漏证据，MMR暂无稳定收益。此为开发基线，不是独立测试集验证的全局最优；阈值仅沿用本轮BGE分数尺度，换模型需要重新评估。批量规划实验没有提供足够稳定的收益来要求本次切换查询流程。

参数固定阶段仅接入返回数量和轻量分数过滤，未重建索引、未新增metadata过滤、未调用生成API或重跑benchmark。重启主网页进程后新建资料库即可使用。导出评测包记录实际阈值；命令行可用 `--no-rerank-threshold --top-k 5` 恢复旧选择方式。低层 `Retriever` / `FinancialRAG` 构造器保留原默认值，历史实验继续显式指定自己的参数；旧 `streamlit_app.py` 演示不属于本次主应用基线。下文“应用默认未改”等表述是各实验完成时的历史记录，以本节为当前状态。

## 资料库持久化与解析回归（2026-10-15）

本轮沿用候选20／Top7／阈值0.1与原有Agent流程，完成资料库管理和解析修复；生成式API调用为0。

**资料库的使用方式**

- 上传后原文、Document JSON、SQLite向量索引和文档清单保存在 `storage/libraries/`；可用 `RAG_LIBRARY_ROOT` 指定其他目录。该目录已由Git忽略规则覆盖（自定义路径需自行纳入备份/忽略范围）。
- 使用同一浏览器、同一访问地址重启服务，自动恢复已保存文档。文档列表与原文查看不加载模型；首次检索/追加时载入已有索引，不重新解析或编码旧文档。
- 会话签名密钥 `session.key` 保存在资料库根目录，文件权限600；备份资料库时一并保存。Cookie最长一年；清除Cookie、更换浏览器或丢失密钥后，不会自动关联到旧库。当前仍是本机单worker服务，无账号或跨浏览器资料库选择功能。
- 文档列表新增“删除”。删除后更新Dense/BM25共同使用的索引，保留其余文档的原向量，重置对话上下文；删除最后一份等同清空。跨会话不能查看或删除对方文档。
- 上传/删除先构建完整新版本，再原子替换当前版本指针；失败保留旧库。损坏清单会报错并保留源文件，避免静默覆盖。成功后清理旧版本；异常退出留下的非当前版本在后续成功变更时清理。
- 资料库持久化不包括聊天历史；重启后最近5轮上下文仍清空。已单独保存的评测包，以及本进程内尚未过期的历史评测快照，保留当时原文，不随资料库删除而改写。
- 上传、查看、删除无需DeepSeek密钥；提问时才要求密钥。新版本不会自动迁移旧服务内存中的临时资料库，旧服务的文档需重新上传一次。既有持久索引不自动重分块；要应用新的解析版本，可删除对应文档后重新上传原件。

**解析修复与验证**

Verizon 2022财报物理第56页为无框表，原线框检测得到0张表；同一行数值有极小纵向偏移，逐行排序可能把2021值放在2022值之前。现在以明确年份列为锚，按坐标对齐至少3条完整数值行，恢复可追溯表格；不满足条件的内容保留原段落。PDF、Word、Excel检索视图均可将明确年份写在对应列值旁，保留单位、表头及单元格来源。解析版本更新为 `native-v2`；跨页内容保持各自物理页，不自动续表。

| 第56页项目（百万美元） | 2022 | 2021 | 新片段检查 |
|---|---:|---:|---|
| Inventories | 2,388 | 3,055 | 通过 |
| Prepaid expenses and other | 8,358 | 6,906 | 通过 |
| Total current assets | 37,857 | 36,728 | 通过 |
| Total current liabilities | 50,171 | 47,160 | 通过 |

本地回归固定4份真实财报共10页，检查结构有效、来源页码与300-token上限；Verizon的上述4行另外核验两年数值和单位。其他9页属于结构冒烟检查，不等于逐单元格正确性验证。Adobe第41/42/54页的复杂表头尚未恢复成结构化表格，原文仍保留。未宣称已提高端到端问答准确率，历史benchmark语料和结果不改写。

自动测试覆盖同浏览器新应用实例恢复、无重新编码、删除后的各检索阶段、会话隔离、发布失败回滚、损坏清单、无API密钥管理，以及三格式年份、负数/小数、百分比、空值、合并单元格与公式缓存。**项目测试195通过，8项实际Dense模型测试按默认配置跳过；前端脚本语法检查通过。** 默认pytest仅收集 `tests/`，避免把历史OCR基准源码副本当作本项目测试。

- [10页解析回归报告](outputs/parsing_regression/native_v2_final_20261015/report.html)
- [逐页结果、原件哈希与Verizon片段](outputs/parsing_regression/native_v2_final_20261015/summary.json)

```bash
# 项目离线测试（不调用生成API）
uv run pytest
# 使用本地财报和tokenizer，输出目录必须不存在
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m financial_rag.evaluation.parsing_regression --project-root . --output outputs/parsing_regression/new_run
```

## 项目现状（2026-10-15）

当前定位：可在本机演示和复测的金融文档 RAG 系统。主流程已贯通；部署扩展、OCR 接入和长期记忆仍需继续实现。
下表以代码和已保存的实验结果为准，后文保留各阶段的设计与实验记录。

| 能力 | 当前状态 | 实现范围 |
|---|---|---|
| 前后端 | 已实现 | HTML/CSS/JavaScript 前端；FastAPI + Uvicorn 后端，无 Node 构建步骤 |
| 多格式输入 | 首版已实现 | PDF、DOCX、XLSX；统一 Pydantic schema、表格结构、格式专属来源定位 |
| 检索 | 已实现 | BGE-small dense + BM25 → RRF → BGE reranker；网页分块 300 tokens / overlap 0 |
| 生成与补查 | 已实现 | DeepSeek API、查询规划、证据不足时补查、已读 chunk 排除、引用映射 |
| 短期记忆 | 已实现 | 最近 5 轮完整问答；请求内证据累积；进程内保存 |
| 长期记忆 | 未实现 | 尚无跨会话偏好、结论或历史恢复 |
| 异步交互 | 已实现 | 异步上传、阻塞任务移交线程池、后台线程及有界队列、NDJSON 回答流 |
| 存储与复用 | 已实现 | 本地持久资料库＋SQLite；同浏览器重启恢复，追加仅编码新文档，单文档删除同步更新索引 |
| 检索 benchmark | 已完成开发集实验 | 分块 7 组、embedding 4 模型初测、BGE/Qwen 50 题内容核验 |
| OCR | 接口已留，实验暂停 | 主应用未接入；PP-StructureV3 成功 4 页，MinerU 首页失败，PaddleOCR-VL 未运行 |
| 部署 | 本机单进程可运行 | 尚无 Dockerfile/Compose、外部任务队列、登录鉴权或多 worker 状态共享 |

### 架构、异步与部署边界

```text
浏览器：混合上传 / 追问 / 来源与表格查看
    ↓ FastAPI（会话隔离、请求校验）
格式适配器 → 统一 Document / Table / Cell + 来源 → 300-token 检索视图
    ↓ 本地 BGE-small 编码
SQLite（chunks / 向量 / 配置）+ 内存 Dense / BM25 索引
    ↓ 查询规划 → Dense + BM25 → RRF → CrossEncoder
请求内证据记忆 → DeepSeek → 按信息缺口补查或回答 → 引用映射
    ↓ 后台工作线程 → 有界事件队列 → 异步 NDJSON 流
浏览器：实时进度 / 回答 / 逐轮检索记录 / 评测导出
```

- 上传接口使用 `async/await` 和 `run_in_threadpool`，避免同步解析/编码直接阻塞事件循环。
- 问答由后台线程执行，通过容量 128 的 `queue.Queue` 传递事件；异步生成器使用 `asyncio.to_thread` 读取队列，并以 `StreamingResponse` 返回进度、回答增量和心跳。协议为 NDJSON。
- 异步交互不等于模型并行推理。当前全局锁串行使用共享资源；资料库、会话和任务状态都在单进程中，不能直接增加 Uvicorn workers 来扩容。
- SQLite 提供数据保存，不是向量搜索引擎；dense 相似度由 NumPy 在内存计算。网页资料库保存到 `storage/libraries/`，通过持久会话密钥在同一浏览器恢复；索引按需载入内存。
- 当前复用的是模型实例、解析结果、文档向量与索引；原文、结构与向量可跨重启复用；没有 Redis 或持久任务队列。生成模型通过 API 调用，项目不管理服务端 attention KV cache。
- Docker 容器化是后续部署工作，尚未落地。生产部署前需处理固定会话密钥、持久资料库、鉴权、工作进程与推理资源调度；OCR 可作为共享工作服务独立部署。

### 实验报告与可量化结果

**统一入口：[全部实验报告](reports/index.html)**。本机报告及原始结果位于 `outputs/`，该目录被 Git 忽略；新克隆仓库不会自带这些实验产物。`reports/` 保留导航和指向原实验目录的链接。

| 实验 | 状态与范围 | 报告 |
|---|---|---|
| FinanceBench 基线 | 10 题试跑与问题定位 | [查看](outputs/financebench/pilot_20261013/report.html) |
| 分块与截断 | 固定 BGE-small，7 组分块配置 | [查看](outputs/chunk_ablation/financebench_20261013/report.html) |
| Embedding 初测 | 固定 300/0，4 个模型、10 题 | [查看](outputs/embedding_ablation/financebench_300_0_20261013/report.html) |
| 证据内容复测 | BGE-small / Qwen3，50 题 | [查看](outputs/embedding_ablation/financebench50_300_0_20261013/report.html) |
| OCR 解析 | OmniDocBench 固定 100 页；已暂停，未形成排名 | [查看](outputs/ocr_ablation/omnidoc100_20261013/report.html) |
| 候选数量与 Top K | 50题；候选20/50 × 重排Top5/7/10，共6组 | [查看](outputs/retrieval_k_ablation/financebench50_20261015/report.html) |
| 批量初始查询与原Loop | 9组筛选；晋级方案50题真实LLM对照 | [查看](outputs/batch_query_ablation/financebench50_20261015_8192/report.html) |

50 题实验语料包括 **41 份完整报告、6,000 页、18,806 个分块**；固定 300 tokens / overlap 0，只比较全库 dense 检索。查询含原始英文问题与目标报告文件名，不按文档过滤，也不运行 BM25、reranker、Agent 或答案生成。

| Top-5 指标 | BGE-small | Qwen3-Embedding-0.6B | 变化 |
|---|---:|---:|---:|
| 证据页命中率（50 题） | 36.0% | 52.0% | +16.0 个百分点（相对 +44.4%） |
| 有效证据内容命中率（49 题） | 28.6% | 38.8% | 约 +10.2 个百分点 |

内容核验排除仅页脚/仅表头命中，数值题核对行名与金额；另有 1 题需版面人工复核，不计入自动内容指标分母。这些是开发集上的参考证据匹配指标，不是回答准确率、完整 RAG 增益或全量 FinanceBench 成绩。应用默认 embedding 仍为 BGE-small；Qwen 尚未替换默认模型。

分块实验发现旧 500 词 / 80 词配置有 **42.0%（1,214/2,891）** 的分块超过 embedding 正文预算；六组 token 分块均未截断。历史检索实验使用旧 PDF 解析入口，不能直接证明新统一表格解析的检索收益。

### 接下来按什么顺序做

1. **已完成本轮：资料库持久化与基础管理。** 同浏览器恢复、原文/结构查看、按内容去重、追加复用向量、单文档删除与清空；支持失败回滚。
2. **已完成本轮：解析回归与首项修复。** 三格式年份/数值/单位回归，固定10页真实财报检查；修复Verizon无框表的年份数值对应。复杂表头、跨页续表仍需专项用例。
3. **下一步：完善短期记忆。** 公司、期间、单位、约束与未解决问题的跟踪；覆盖连续追问、话题切换和上下文超限。沿用已固定的Retriever参数。
4. **随后按需求：按需OCR与长期记忆。** OCR继续暂停；先小样本验证再接入扫描页。长期记忆需独立的来源、更新和删除机制。
5. **部署基础。** Docker、登录鉴权、任务状态/重试、资源调度、运行监控和吞吐验证。

## 快速启动

```bash
uv sync --locked
uv run prepare_models.py  # 从已有缓存复制；首次无缓存时加 --download
uv run app.py
```

浏览器打开 http://127.0.0.1:8000，也可在 macOS 双击 `start.command`。
后端为 FastAPI + Uvicorn，接口文档位于 http://127.0.0.1:8000/docs；也可运行 `uv run uvicorn app:app --host 127.0.0.1 --port 8000`。
资料库已落盘，运行时索引和对话记忆仍在进程内，请使用单 worker；共享模型仍通过锁串行执行。
在项目 `.env` 设置 `DEEPSEEK_API_KEY`；FastAPI 入口自动加载，密钥只在后端使用。
拖入/多选 PDF、DOCX、XLSX（可分次添加）→ 开始阅读 → 输入问题 → 展开来源定位与原文；表格引用可查看完整结构。
前端是 HTML/CSS/JavaScript，不需要 Node 构建，但必须启动 Python 后端，不能仅双击 HTML。

- 资料库累计最多 10 份，每份 20 MB，总计 100 MB；跨批次同内容按哈希去重。重复上传不新增文档，清空按钮才会移除已有资料库。
- 多份文档共用当前会话的索引；新上传文档追加到旧资料库，复用已有向量；全部处理成功后切换索引，失败保留旧数据。
- 原文、结构与索引持久保存；同一浏览器重启服务后恢复。不同浏览器会话隔离，模型共享并串行执行。
- 支持最近 5 轮完整问答的上下文追问；每题重新检索原文。资料库变更或服务重启后重置上下文。
- 相关证据片段与问题会发送给 DeepSeek。扫描件无 OCR，当前检索更适合英文。
- 当前默认最多 Top7 且过滤低于0.1的结果，不保证每份文档都有证据；跨文档比较仍需专门评估。

## 当前实现与文件职责

| 文件/模块 | 已实现 | 主要限制 |
|---|---|---|
| ingestion/document.py、adapters.py | 统一运行时 schema、三种原生解析、表格单元格及格式专属来源 | PDF 无框表/跨页续表、Word 嵌套表等复杂结构尚待完善；OCR 仅接口 |
| ingestion/structured_chunker.py | 新网页入口：300 BGE tokens、0 overlap，表格行保留候选表头与来源 | 超长行分片；候选表头需核验 |
| ingestion/parser.py | 旧 PDF 实验入口，保留原文块、坐标、页码 | 保留原行为，未接新表格结构 |
| ingestion/normalizer.py | 块内清洗、保留段落、保守移除重复页脚 | 不恢复复杂版式 |
| ingestion/chunker.py | 旧实验入口：段落分块、长段切分，500 词/80 词重叠 | 按词而非 token，不跨页 |
| indexing/embeddings.py | BAAI/bge-small-en-v1.5，归一化向量 | 旧分块存在截断；token 分块已审计，新输入仍需控制预算 |
| indexing/vector_store.py | SQLite 保存/加载 chunks、float32 向量与模型配置 | 整批替换，无增量更新 |
| retrieval/dense.py | NumPy 点积检索 | SQLite 仅存储，查询仍在内存完成 |
| indexing/bm25_index.py、retrieval/sparse.py | TF/DF/IDF、BM25 和正分 Top K | 基础英文分词，内存索引 |
| retrieval/fusion.py | RRF，保留两路排名 | 不保证融合优于单路 |
| retrieval/reranker.py | CrossEncoder，记录输入截断 | 无分窗，分数不代表正确率 |
| memory/conversation.py | 最近 5 轮完整问答、来源快照与有界上下文 | 仅内存；无自动摘要或刷新后的聊天恢复 |
| memory/evidence.py | 请求内证据累积、去重、稳定编号、查询历史和上下文组装 | 仅内存，不持久化；无自动上下文预算 |
| generation/citations.py | 证据编号、来源映射、未知编号检查 | 不核验事实是否被证据支持 |
| generation/prompt.py、generator.py | 基于证据的提示词与 DeepSeek API | 支持 generate_turn 工具调用适配 |
| pipeline.py | FinancialRAG.retrieve/ask，实时完整单轮问答 | 检索委托给独立 Retriever；另有 ask_agent() 多轮入口 |
| retrieval/retriever.py | Dense/BM25 → RRF → 重排，支持 exclude_chunk_ids 和按次 top_k | 按 ID 去重，不做语义去重；不保存请求记忆 |
| app.py | FastAPI 多格式会话、统一入库、原文/结构及问答 API | 本地开发服务，无登录和公共部署 |
| templates/index.html、static/ | 多格式上传、表格预览、实时进度、回答与逐轮证据展示 | 近期历史传给查询规划器以补全追问 |
| schemas.py | 当前检索、问答与会话上下文数据契约 | TypedDict 不做运行时校验 |

当前网页与 pipeline 使用 BGE reranker v2-m3、2048 tokens；生成默认 deepseek-flash。
Reranker 类及批量对比入口仍默认 MiniLM、512 tokens，不要混淆入口的默认配置。
统一 CLI、config 等占位模块尚未全面实现；已有独立分块/embedding/证据内容评测入口，端到端回答评测尚未完成。

## 已跑通的两条流程

```text
入库（仅新增/替换文档时）：
PDF / DOCX / XLSX → 统一 Document（完整结构 JSON）→ 带来源的检索视图 → 文档向量 → SQLite

问答（每个问题）：
加载/复用索引 → 问题向量 → Dense 20 + BM25 20
→ RRF 20 → Reranker 5 → 证据编号 → 提示词 → DeepSeek → 引用检查
```

同一 FinancialRAG 实例复用文档向量、BM25 与模型，不重复编码全文。
SQLite 保存的是数据；相似度仍由 NumPy 算，尚未接向量扩展或独立向量数据库。
查询必须使用与索引一致的 embedding 模型，即使不同模型维度相同也不能混用。

## 运行入口

```bash
# 旧的解析/分块/Dense/BM25 学习演示，会重新编码文档
uv run main.py

# 从现有 chunks 首次入库；数据库已存在时用 --replace 明确整批替换
uv run index_demo.py build
uv run index_demo.py search "What were Apple's total net sales in 2025?"

# 实时完整问答，从 storage/vectors.sqlite3 加载
uv run --env-file .env python -m financial_rag.pipeline \
  "What were Apple's total net sales in 2025?" --output outputs/pipeline_answer.json

# 仅读取既有重排报告来测试生成
uv run --env-file .env generate_demo.py

# 8 题 BGE 四路对比：Dense/BM25/RRF/Reranked
uv run python -m financial_rag.evaluation.compare_methods \
  --reranker-model BAAI/bge-reranker-v2-m3 --reranker-max-length 2048 \
  --output outputs/bge_m3_reranked_comparison.json

# 三路对比使用 --skip-reranker；旧单 PDF 界面保留在 streamlit_app.py
uv run pytest -q
```

## 已完成：独立 Retriever 与去重检索

`FinancialRAG.retrieve()` 委托给 `retrieval.retriever.Retriever`，返回结构保持不变。
下面演示调用方保存已交付证据的 ID；证据正文及稳定引用编号现在也可交给 EvidenceMemory 管理。

```python
first = rag.retrieve(question)
seen = {item["chunk_id"] for item in first["reranked"]}
second = rag.retrieve("针对缺失信息的补充查询", exclude_chunk_ids=seen, top_k=5)
seen.update(item["chunk_id"] for item in second["reranked"])
```

`top_k` 默认使用初始化值，按次覆盖必须为正整数且不超过 `candidate_k`。
排除 ID 不会改变索引或影响其他请求；未知 ID 忽略。两路先扩大召回数量、过滤已读 ID、
再各取 `candidate_k`，融合后重排。BM25 的语料统计保持不变。
只有交给模型的 `reranked` 结果应标记为已读。全部片段均已排除时，各阶段返回空列表。
当前 `ask()` 每题创建独立 EvidenceMemory，仍只检索一次；自主工具调用和循环停止策略已在可选 ask_agent() 中实现。

离线回归测试覆盖以下行为（实际数量以当前 pytest 输出为准）：覆盖候选补足、
多轮 ID 排除、请求间隔离、片段耗尽、参数校验、原问答流程，以及记忆累积、
稳定引用、快照隔离、无新增结果记录、失败批次不写入和检索至耗尽；
另覆盖工具参数校验、调用编号配对、多轮上下文、预算停止及 DeepSeek 响应适配。
这些测试验证编排行为，不代表检索相关性已达标。

## 已完成：当前问题的证据记忆

`memory/evidence.py` 中的 `EvidenceMemory` 使用 Python 字典和列表，每次 `ask()` 创建新实例，
不保存到共享 Retriever。已接入单轮问答与 Agent，同一次请求的多轮检索复用同一实例。

- `add_evidence(results, chunks_by_id)`：注册最终证据，按 chunk_id 去重，返回新增 ID。
- `record_search(query, results, chunks_by_id)`：注册证据并记录返回 ID、新增 ID 和新增数量；空结果也记录。
- `chunk_ids`：返回已注册 ID 集合，作为下一轮的排除参数。
- `build_context()`：输出所有累计原文与稳定引用编号；`citation_map` 用于最终引用检查。
- `evidence`、`search_history`：返回数据快照；修改快照不会改动内部记忆。

正文从原始 chunks 读取，保留 document_id、物理页码及可用的标题、文件名。
同一片段重复加入保持编号，新片段继续分配 E6、E7 等编号；来源冲突会报错。
整批验证通过后才写入，失败不会留下部分证据。旧 `prepare_evidence()` 接口保留，内部复用此模块。
`ask()` 返回结果新增 `search_history`，原有回答、检索轨迹和引用字段保留。

```python
from financial_rag.memory import EvidenceMemory

memory = EvidenceMemory()  # 每个新问题重新创建
for query in [question, "针对缺失信息的补充查询"]:
    trace = rag.retrieve(query, exclude_chunk_ids=memory.chunk_ids)
    memory.record_search(query, trace["reranked"], rag.chunks_by_id)
context = memory.build_context()  # 后续模型调用必须实际传入该上下文
citation_map = memory.citation_map
```

上面是手动两轮示例，不是当前 ask() 的自动行为。记忆不做语义去重、自动摘要、长期持久化或
上下文裁剪；工具层和循环层设有字符预算保护。跨问题或更换索引时创建新实例。

ask_agent() 已实现的流程：

```text
新问题 → 规划一条初始查询 → 创建 EvidenceMemory → 检索一次并注册证据
→ 模型读取累计证据
   ├─ 足够：回答并检查引用
   └─ 不足：生成补充查询 → 排除 memory.chunk_ids 检索
           → 注册新证据、记录检索历史 → 再调用模型
```

记忆模块只负责保存和提供证据；是否继续检索由模型工具调用决定，
轮数上限、无新增证据和上下文预算由 Agent loop 控制。当前 `ask()` 尚未执行上述循环。

## 已完成：检索工具与 DeepSeek 循环

| 位置 | 职责 |
|---|---|
| `retrieval/tool.py` 的 `SearchDocumentsTool` | JSON Schema、参数校验、调用 Retriever、注册证据、结构化结果 |
| `generation/generator.py` 的 `generate_turn()` | 向 DeepSeek 传 tools，保留 tool_calls 和 reasoning_content |
| `agent/loop.py` 的 `run_agent()` | 模型 → 工具 → 模型循环；请求内记忆、预算、最终引用检查 |
| `generation/prompt.py` 的 `AGENT_SYSTEM_PROMPT` | 指导模型识别缺失证据、发起补充查询和停止检索 |
| `pipeline.py` 的 `ask_agent()` | 可选多轮入口；`ask()` 保持单轮基线 |

```python
result = rag.ask_agent("比较两年的收入变化", max_steps=5, max_searches=5)
print(result["answer"])
print(result["status"], result["stop_reason"])
```

```bash
uv run --env-file .env python -m financial_rag.pipeline \
  "What changed in revenue between the two years?" --agent --output outputs/agent_answer.json
```

网页已调用 `ask_agent()`，引用正文从累计 `evidence` 读取，支持跨轮来源。
发送后显示实时计时，完成后保留总耗时、模型调用次数和每轮查询。计时包含网络和服务等待时间。

默认先用一次模型调用规划一条初始查询，失败时以原问题回退；始终将原问题交给回答模型。
初始仅检索一次。模型读取证据后，如果不足，在同一次工具调用中给出 missing_information 和 query。
模型可在一轮提出多条对应独立缺口的查询，程序依次执行、共享去重记忆，再统一交回结果。
可用 `rag.ask_agent(question, use_query_planning=False)` 跳过规划、直接检索原问题。
补查工具要求 query、missing_information，可选 top_k；索引和已读 ID 由后端绑定。
每个工具调用都回传对应 tool_call_id；达到总次数或证据容量上限时，剩余调用返回限制提示。历史证据保留在消息中。
未知工具和非法参数返回可纠正错误；失败不暴露内部路径或密钥。

默认最多 1 次查询规划调用加 7 次回答/工具决策调用、8 次检索尝试（含首次检索和无效工具请求）。最后一次模型调用
禁用工具以便总结；连续两次无新增证据或证据预算耗尽也禁用后续检索。达到限制时返回 limited
和 stop_reason，不能当作已完整回答；模型失败返回 error。引用有效性不代表事实正确。
累计证据默认限 60000 字符、消息默认限 120000 字符；这是字符保护，不是精确 token 预算。
证据超限不写入记忆、不静默截断；消息超限直接返回 limited。API 使用已有的超时与重试配置，
尚无独立端到端墙钟超时。响应截断不会当成完整回答。

工具协议依据 [DeepSeek 官方 Chat Completions 文档](https://api-docs.deepseek.com/api/create-chat-completion/)。
自动回归以模拟模型离线验证；用户已通过网页完成真实问答并提供检索截图，这不等于完成系统化质量评测。

下一步优先评估结构化分块与检索过滤，已接入有界跨问答记忆；以检索相关性和证据覆盖评估效果。

## 评估观察与技术欠账

完整历史实验记录在 [retrieval_experiments.md](docs/retrieval_experiments.md)。
Apple 样本为 80 页、132 chunks。MiniLM 512 tokens 的 8 题有 91/160 输入对截断；
同候选 BGE 2048 报告为 0/160。大中华区销售额证据升至第一，但网络安全题仍将直接风险证据
排出 Top 5；竞争风险题也存在偏向监管内容的问题。换模型未证明整体更优。
真实 pipeline 已成功回答总销售额并返回引用；离线 pipeline 编排测试验证不重复编码、
引用异常与空问题。多 PDF 网页接口已做去重、会话隔离、上传失败保留和文件来源映射检查。
这些不等于完成标准相关性或生成质量评估；模型测试默认可能跳过。

待做：8 题人工证据标注、明确年份、Hit@K/Recall@K/MRR、生成事实与引用支持性检查、
embedding 截断评估、多文档证据覆盖、RRF 边界防御、token 分块/分窗、增量索引。
当前网站未实现公共部署、登录权限系统、并发优化和聊天历史持久化；文档资料库已支持本地持久化。

## Schema 与版本约定

`schemas.py` 保留检索/实验结构，并补充目前 pipeline 与网页实际输出类型。
类型描述当前证据、工具、Agent、网页、流事件和会话上下文。
TypedDict 只做静态描述；默认值、范围、互斥状态、文档访问范围都须另写运行时校验。
所有证据保留原始 chunk 正文；模型可能截断输入，不能据报告全文断言模型看过全部内容。
网页 document_id 是内容哈希，旧 Apple CLI 使用固定 ID；不能把两套索引的编号混为一谈。

## Git 与依赖

提交源码、测试、README、pyproject.toml、uv.lock；不提交 `.env`、`.venv`、PDF、索引与输出。
通过 `uv add`/`uv remove` 管理依赖。LangChain 虽在依赖中，当前链路及规划的 Agent loop 都显式实现。

## 网页模型本地存放

PDF 解析使用 PyMuPDF，不需要下载解析模型。网页检索使用的两个模型现存放在
`models/bge-small-en-v1.5/` 和 `models/bge-reranker-v2-m3/`，约占 2.2 GB。
`prepare_models.py` 固定模型版本，从已有缓存复制真实文件；无缓存时显式使用
`uv run prepare_models.py --download` 下载。现有目标目录不会被自动覆盖。
网页使用 `local_files_only=True`，不会联网获取这两个模型。重启后的第一次上传仍需
把本地权重加载到内存，之后复用；文档编码进度不表示模型下载。
模型目录不提交 Git。其他学习演示/CLI 仍保留原模型名称加载方式。
DeepSeek 回答仍需网络；本地模型存放不代表回答模型也已本地化。

查询改写可能改善召回，但会增加时延与模型用量，不能保证相关性提升；需用同一组标注题对照评估。

回答与查询规划提示词已收紧：按用户实际任务判断证据充分性；宽泛比较允许明确期间差异的有限结论，严格同期请求仍需相应证据。默认简洁回答，不因仍有检索预算而补查无关细节。当前已改为逐轮检索，实际质量与轮数收益仍需对照评估。

## 实时进度与回答流

网页使用 `/api/ask/stream`，逐行 JSON 传输规划、检索、补查进度和回答增量。
工具调用参数拼接完整后才执行；工具轮的临时正文会清除，不作为最终答案。
仅流出 content，不展示 reasoning_content。最终 done 事件附带引用和计时统计；
断线或截断会提示未完成，不当作成功。客户端断开后会停止后续轮次；已进行的阻塞请求
可能需要等返回或 API 超时。后端仍串行使用共享模型，等待期间发送心跳。
默认上限为 1 次规划 + 7 次决策/回答调用，以及 8 次检索尝试，并非固定执行次数。
26 秒不是硬性超时或时延保证；复杂问题可能更久，证据足够仍提前结束。

检索过程可按轮展开新增片段：本轮最终排名、稳定引用编号、文件名、PDF 页码、正文预览与全文，以及是否被最终回答引用。未引用不等于不相关。耗时下方显示模型调用拆分，查询规划计入一次模型调用。

## 已完成：按信息缺口逐轮驱动检索

已取消开头固定执行三条查询。每次补查前都由模型读取证据，在一次决策中给出简短
missing_information 与 query；理由保存在检索历史中并展示在网页，初始检索单独标记。
该理由是可核查的检索目的，不是模型思维链。

```text
原问题 → 模型规划一条查询 → 检索并注册证据 → 模型判断
  ├─ 足够：回答
  └─ 不足：缺少什么 + 新查询 → 检索一次 → 注册新证据 → 模型重新判断
```

同一轮的多条查询顺序执行。批内空结果不取消后续查询；整批结束后若连续两次无新增才停止。
保留调用上限、证据预算、按 ID 去重与稳定引用。不能为了找到符合预期的数字反复查询。
“DeepSeek 两次、检索一次”现在可对应一次初始规划和一次回答；任何额外检索必须经过工具决策。
现有测试验证首轮足够即停止、缺口必填、同轮多条查询依次执行、空结果改写和跨轮引用。
实际相关性改善仍需标注评测。意图分类与动态融合权重未包含在该轮实现中；对话记忆见下节。

## 已完成：跨问答的对话上下文记忆

`memory/conversation.py` 的 `ConversationMemory` 由后端每个浏览器会话的资料库持有，
不放在共享模型或 Retriever 中。保存最近 5 个 completed 问答对，包括原问题、补全后的问题、
最终回答、来源原文快照、turn_id 和 workspace_version；失败、limited、澄清和中断结果不进入历史。
这是一份有界的临时记忆，不是永久聊天归档；服务重启后丢失，也不恢复刷新后的聊天界面。

每次请求选取连续的最近完整问答，序列化后最多 12000 字符；超出时移除最旧轮次，
不截断单轮、不调用摘要模型。若最新一轮本身超限，本次不提供历史，请明确公司、期间和指标。
来源原文保存在后端但不发给规划器；会话总存储默认限 400000 字符，超限移除最旧记录。
以上预算可通过 ConversationMemory 构造参数调整。

现有一次查询规划同时完成追问补全和英文检索词生成，不新增一次模型调用。
有历史时返回 resolved_question、query、clarification；独立问题不沿用无关旧主题。
原问题和补全问题一起交给回答模型，每题仍创建新的 EvidenceMemory 并重新检索。
历史回答只帮助理解指代，不是事实证据；旧引用编号不传给规划器，本轮引用重新编号。
网页显示补全后的问题，便于发现误解。模型的语义判断仍可能出错，离线测试不代表真实模型质量保证。

指代不明确时返回 clarification 状态，不检索、不将澄清当成完整问答保存；用户需补充完整问题。
有历史时规划失败也要求重新明确问题，避免把未解析的追问直接用于检索；无历史时保留原有回退。
`ask_agent(..., conversation_context=memory.context())` 可显式提供上下文，必须启用查询规划；
单轮 `ask()` 保持独立基线。HTTP 普通和流式入口均接入；流式响应在服务端完成 done 事件交付后写入历史，
已观察到的断线不写入，服务端无法保证浏览器实际收到最后一个网络包。

成功追加新文档或清空资料库时重置会话记忆；重复上传不变更内容、上传失败时保留历史。
来源编号由所属 turn_id + workspace_version 定位，来源 text 是原文快照，不以摘要替代。
当前版本不持久化会话，不做自动对话压缩。

## 下一阶段的评估重点

固定问题和人工核实的原文证据，比较原问题单次检索与当前按信息缺口逐轮主动检索。
逐阶段记录 Dense/BM25 召回、RRF 候选与最终重排位置，检查新增片段贡献和冗余，
同时统计必要证据覆盖率、排名、DeepSeek 调用数、检索次数与耗时。
“最终答对”“新增五条”或“被回答引用”都不能单独证明 Retriever 质量。

## 一键保存检索评测资料

回答完成后点击“保存检索评测资料”，保存到项目 `outputs/retriever_evaluations/<run_id>/`。
每份包包含完整运行 JSON、各阶段排名 CSV、原始 PDF、文件名/哈希映射、实际分块与 SQLite
向量索引、重新解析的页面数据、配置/依赖/源码快照、回答、耗时和人工标注模板。
不复制模型权重、密钥或模型推理内容；outputs 已被 Git 忽略，不上传到 GitHub。

后端自动在临时目录保留当前会话最近五次问答快照，点击按钮才写入上述项目目录。
保存绑定问答时的资料库；之后追加/清空资料库不会把新文件混进旧记录。重复点击不重复建包。
服务重启或超过五次后的未保存快照失效；已保存文件不受影响。旧版本生成的回答无法补存完整轨迹。
本功能只保存本地文件，不调用 DeepSeek。页面解析在点击保存时重建，不应与原始入库日志混淆。


## 本次评测观察：服务业务增长问题（2026-09-29）

本地评测包：`outputs/retriever_evaluations/20260929T081616Z-508b57b9ecc52a8e/`（不提交 Git）。
问题：“苹果的服务业务增长比硬件快，这是不是意味着它以后赚钱会越来越轻松？财报里有哪些支持和反对这个判断的证据？”
浏览器耗时 28.1 秒，模型调用 3 次，检索 3 次，分别新增 5、8、8 个片段，共 21 个不同 chunk。
这些是检索次数，不意味着模型在每两次检索之间都重新判断；同轮可以提出多条查询。

- 首次命中 PDF 第 26、27、32 页，覆盖服务/产品收入与毛利率。
- 第二次补查风险时，第 3 页目录进入最终第 4；第三次补查成本时，第 80 页高管认证进入最终第 3，存在噪声。
- 第 13 页 `p13_c1` 包含内容授权续约、内容供应商提价及自制内容成本风险；第三次 BM25 第 5、RRF 第 11，但未进入重排 Top 8。片段已被召回，损失发生在最终重排选择，不能全部归因于 chunk 切分。
- 本次轨迹未报告 reranker 输入截断；不代表 embedding 没有截断，也不证明全文相关证据已穷尽。

这是单个案例的人工抽查，不是完整召回率评测。优先积累 10～20 个真实问题，以独立检查原文、人工核实的
“必要信息点 + 页码 + 原文片段”建立小型评测集；模型可辅助找证据，但不能直接当 ground truth。
标注未穷尽时报告“关键证据覆盖率”，不宣称严格 Recall。不要只从已有检索结果反向生成标准答案。
目录与认证文件可作为本题的负例，但对目录导航、签署人问题可能是正例。

## Chunk 策略修改路径（计划，尚未实现）

目标：保留全文可检索性，让系统知道片段属于什么内容，并按问题选择合适证据。
首阶段不单纯调整 500 词/80 词重叠，不全局删除目录、附件或认证页；结构标注与检索策略需要一起验证。

下表路径均相对 `src/financial_rag/`，除明确写出的项目根目录文件。

| 顺序 | 修改位置 | 计划内容与验收要点 |
|---|---|---|
| 1. 结构识别 | `ingestion/parser.py`；拟新增 `ingestion/structure.py` | 利用标题、文本块、坐标及可用 PDF 书签识别目录、财务报表、附注、风险、MD&A、附件、认证声明。按块标注，避免一页多章节被强行归为同类；不确定时保留 unknown，不按固定页码识别。 |
| 2. 数据契约 | `schemas.py` 的 ParsedPage/TextBlock/Chunk | 设计可选的 section_type、section_title、section_path、分类依据/置信标记和 chunk 策略版本。现有 section/title 可复用或明确迁移，避免同义字段冲突；这些新字段当前尚不存在。 |
| 3. 结构感知分块 | `ingestion/normalizer.py`、`ingestion/chunker.py` | 清洗保留结构标签；优先按章节/段落边界切分，重叠不跨不同内容类型；保留 document_id、物理页码和原文。首版维持单页来源，跨页表格、token 分块与分窗单独评估。 |
| 4. 入库与兼容 | `indexing/vector_store.py`、项目根 `app.py` 与入库入口 | chunk JSON 持久化结构字段，索引记录策略版本；旧索引缺字段回退为 unknown。切分或编码文本改变时重建向量和 BM25；仅增加元数据且文本、顺序与模型完全一致时才考虑复用向量。防止不同版本混入同一资料库。 |
| 5. 检索使用元数据 | `retrieval/retriever.py`、`retrieval/fusion.py`，必要时 `retrieval/tool.py` | 对一般财务分析软降权目录/认证；明确问签署人、附件或目录时允许正常召回。先实现可关闭的保守策略；如过滤候选须补足候选池，避免过滤后无证据。原始分数与策略调整分开记录。 |
| 6. 重排对照 | `retrieval/reranker.py`、`agent/planner.py`、`generation/prompt.py` | 单独比较宽泛长 query 与针对单一缺口的短 query；检查第 13 页证据为何被筛掉。保留所有进入重排候选的分数与最终选择，不能用结构降权掩盖相关正文丢失。 |
| 7. 可观测与回归 | `evaluation/export.py`、项目根 `app.py`、`static/app.js`、相关 tests | 导出/展示内容类型、过滤或降权原因及策略版本；检查标签贯穿解析→分块→入库→检索→证据。对固定问题比较关键证据覆盖、噪声、排名和耗时。 |

实施顺序：先保存当前基线 → 结构标签贯通但暂不改排名 → 启用可关闭的检索降权 → 单独比较 query/重排策略。
每次只改变一个因素；不同时改分块大小、融合权重和模型，否则难以定位收益来源。

首批验收案例：

- 服务业务问题：重点检查第 13 页内容成本证据是否进入最终证据，以及目录、认证文件是否减少。
- 认证签署人问题：认证文件仍可命中，不能因降噪永久失去这类信息。
- 混合章节页及旧索引：标签不串段，unknown 不被误删，引用页码仍准确。
- 切分变化后按必要信息点和原文位置比较，不以旧 chunk_id 作为唯一标准。

本节仅记录修改路线；当前仍是原有按段落/词数分块和无内容类型降权的检索流程。


## FinanceBench 10 题试跑

已接入 FinanceBench 官方公开数据的固定版本（`cc39aeb4afdf33909ee1412188bf89035950c2eb`）。
无需先自行标注；默认按三种题型轮流抽取 10 家不同公司的 10 道题，固定种子 20261013。

```bash
# 下载标注与样本 PDF，并构建独立的持久化索引
python -m financial_rag.evaluation.financebench --phase prepare
python -m financial_rag.evaluation.financebench --phase index
# 实际调用配置的 DeepSeek API，逐题比较单轮与多轮
python -m financial_rag.evaluation.financebench --phase run --output outputs/financebench/my_run
```

需已安装项目依赖、准备现有本地 BGE 模型，并配置 `.env` 中的 API 密钥。
下载文件在 `data/raw/financebench/`；索引在 `storage/financebench/<fingerprint>/`；
结果包含配置/源码哈希、标准答案、实际回答、检索各阶段轨迹、页级覆盖、耗时与 API token 用量。
同一输出目录会跳过已保存的案例；错误案例也保留，重试前需先检查并移除对应错误 JSON。
配置变化时使用新的输出目录。`--count` 受三类题型与公司去重约束，当前采样器不支持直接用 150 跑满全部题目。

2026-10-13 首跑报告：`outputs/financebench/pilot_20261013/report.html`，10 PDF / 1866 页 / 2891 chunks。
单轮至少命中一页为 5/10，覆盖全部标注页为 3/10；多轮两项均为 10/10，调用数为 10 vs 29。
这不是答案准确率。多轮累计证据预算更大，且命中表格页后仍可能读错年份（Verizon 案例）。
输入为原始问题加公开目标文件名，共享语料检索，不按文档过滤；答案与标注页不进入模型输入。
标注页并未穷尽所有有效证据，因此页级匹配仅用于诊断，不能等同于严格召回率。
本次试跑报告中的逐题文字为初步对照复核，未使用额外 LLM 裁判，也未全面核验扩展结论。

## 后续路线：金融文档理解、检索与可信回答（计划）

项目以金融文档阅读、检索证据和回答问题为核心，首阶段文档范围确定为 PDF、Word、Excel 三类，
优先沿用单 Agent / RAG 流程，逐步形成可管理、可追溯、可评测的文档问答系统。
以下是后续研究与实现清单，不代表功能已经完成；已有基础能力也需要按这些标准继续验证。
`src/financial_rag/schemas.py` 描述当前数据契约，待具体方案确定后再同步扩展字段与存储格式。

- [ ] **PDF 中的表格、跨页内容、扫描文字怎么解析？**
  区分带文字层的 PDF 与扫描页，评估按需 OCR；保留表格行列、表头、单位、脚注及原页位置，
  设计跨页表格续接与来源追溯。用原始页面核对解析结果，重点检查数字、负号、小数点和年份是否错位。
  首个回归案例采用 FinanceBench 中 Verizon 的流动负债年份错配问题。

- [ ] **用户问某个指标，怎样找到正确的年份、单位和原文？**
  将公司、指标、报告年份、指标所属期间、币种和数量级共同用于检索与核验；区分财年、自然年、
  季度和期末时点。计算题保留各输入值的来源、单位换算和公式，检查取数是否来自正确行列。
  验收包含同一指标多年度并列、千/百万/十亿单位差异及需要多页证据的题目。

- [ ] **多份报告里的口径不同，怎么避免混用？**
  区分合并与分部、GAAP 与调整后指标、原始与重述数据，以及公司披露公式与用户指定公式。
  记录每个数值所属报告、期间和定义；发现冲突时展示差异并解释，无法统一时要求澄清或分别回答。
  验收时专门加入同名指标不同定义、不同报告版本与跨公司比较的案例。

- [ ] **回答怎样附上证据？找不到信息时怎么处理？**
  在现有引用机制上核验关键结论是否被原文支持，能否定位到报告、物理页码和具体原文或表格区域。
  区分原文披露值、计算结果和分析判断；缺少证据时明确缺失项，按预算补查，仍不足则说明无法确定。
  验证证据不足、证据冲突、只找到部分计算输入时的行为，避免用无关引用支撑结论或凭空补数。

- [ ] **怎样验证检索和答案的质量？**
  以 FinanceBench 现成文档与标注为起点，固定语料、问题、配置和基线，再逐步扩大样本；
  对有歧义或失败的案例做少量原页核验，降低从零人工标注的成本。
  分开评估解析/取数、检索证据覆盖、答案数值与口径、引用支持性及证据不足时的处理，
  同时记录耗时、token 用量与模型调用数。单轮和多轮除默认配置对比外，再设计同等预算对比。
  标注页命中不等于答案正确，标注也可能未穷尽有效证据；自动裁判只能辅助，关键错误需原文复核。

建议推进顺序：保存现有 FinanceBench 基线 → 修复表格年份与数值对应 → 完善期间/单位/口径元数据
→ 加强引用与缺证据处理 → 扩大评测。扫描页 OCR、跨页表格各自建立小型专项用例，按需求分步接入。
每次尽量只改变一个因素，并复用同一批问题确认改进效果；结构分块的具体模块路线见上文“Chunk 策略修改路径”。


### Embedding 模型对比：已完成阶段与后续计划

截至 2026-10-14，已完成四模型初测及 BGE/Qwen 50 题 dense 对比、编码配置记录和实验报告。以下保留完整路线；生产配置切换、完整混合检索收益、稳定性能测量和回答质量验证仍待完成。

- [ ] **建立可切换的 embedding 模型配置。**
  以当前 `BAAI/bge-small-en-v1.5` 为基线，后续选择不同规模、中文/英文/多语言能力的模型进行比较；
  本地模型与 API embedding 可分别作为候选，具体型号和版本在实验前确定。
  记录模型版本、向量维度、输入长度、截断、归一化、相似度计算和 query/document 提示格式，遵循各模型的使用要求。
  每个模型独立生成文档向量与索引，查询使用匹配模型；缓存键包含模型及编码配置，避免复用不兼容的向量。

- [ ] **固定条件，分阶段比较检索效果。**
  先复用 FinanceBench 的同一批文档、解析结果、分块、问题和 Top-K，仅替换 embedding 模型；
  先比较纯向量召回，再固定 BM25、RRF 和 reranker 配置，观察模型差异在完整检索流程中是否仍有收益。
  按各阶段记录标注证据覆盖、Hit@K、首个相关结果的排名及失败案例；页级标注不等于片段级相关性，
  未穷尽证据时不把标注页覆盖称为严格召回率。中文提问、英文财报和跨语言检索另建分组用例，避免混在总分中。

- [ ] **同时评估效率与资源成本。**
  在相同硬件和明确的批大小条件下记录文档编码速度、索引构建时间、查询延迟、内存/显存及索引大小；
  API 模型另记录请求量和费用。区分冷启动、热运行与缓存命中，并重复测量延迟，避免只凭单次耗时下结论。

- [ ] **验证最终回答收益并保存实验记录。**
  对候选模型固定生成模型、提示词和 Agent 搜索预算，复核取数、年份/单位、答案及引用质量，
  区分 embedding 改善与后续重排、补查的贡献。保存配置、数据版本、逐题结果和对比表，
  用效果、速度与成本共同决定取舍。当前 10 题用于试跑与诊断，扩大样本并保留未参与调参的测试题后再确定默认模型。

### 多格式文档与记忆（部分完成）

- [x] **PDF、Word、Excel 统一输入与检索（首版；复杂结构及冲突评测继续推进）。**
  提供统一上传入口，按格式选择解析器，再接入共同的索引、检索和回答流程。
  首版已支持 `.pdf`、`.docx`、`.xlsx`；旧版 `.doc`、`.xls` 的转换支持后续单独评估。
  PDF 保留页码与区域；Word 保留标题层级、段落及表格位置；Excel 保留工作表名、单元格范围、
  表头、数值格式、公式与可用的计算结果，遇到结果缺失或可能过期时明确提示。
  不把所有格式强行压成无结构纯文本，也不为 Word、Excel 编造 PDF 页码。
  验收包含混合上传、跨格式取证、来源定位，以及同一指标在不同格式中数值或口径冲突的场景。

- [ ] **完善短期记忆。**
  当前已有会话内有界临时记忆，可辅助追问补全；后续继续完善实体、期间、单位、用户约束和未解决问题的跟踪。
  研究按需摘要与上下文预算，保留摘要到原对话的关联；切换公司、期间或资料库时检查旧上下文是否仍适用。
  历史回答帮助理解问题，财务结论仍需本轮证据支持。验收包含连续追问、话题切换和上下文超限。

- [ ] **加入长期记忆。**
  规划跨会话持久保存用户明确表达的偏好、项目术语、分析口径及经确认的结论，并按当前任务检索相关记忆。
  区分文档知识库、会话历史和长期记忆：上传文档进入知识库，不自动把模型回答当作可靠事实写入长期记忆。
  每条记忆记录来源、适用范围、时间与有效状态；提供查看、修正、删除与关闭入口，处理冲突和失效信息。
  文档更新或删除后，对依赖该文档的记忆重新核验或标记失效；按用户和项目隔离。
  验收包含重启后恢复、跨会话使用、错误记忆纠正、遗忘及项目间不串用。

### 系统工程配套（计划）

围绕上述能力补齐以下最小闭环，按实际需求推进：

- [ ] **文档与索引管理：** 文档列表、去重、版本、删除、增量入库与重建；更新文档后同步处理索引、缓存和相关记忆。
- [ ] **任务与错误恢复：** 展示上传、解析、索引的进度与失败原因；支持重试，避免重复任务造成重复入库。
- [ ] **缓存与可观测性：** 按文档内容、解析/模型版本和检索配置管理解析及向量缓存；记录检索轨迹、耗时、调用量与错误，便于定位瓶颈。
- [ ] **会话与资料隔离：** 明确用户、项目、会话和文档的归属，保证检索、记忆及缓存遵循同一访问范围。
- [ ] **运行与回归：** 整理配置、启动与部署说明，建立小型端到端回归集；覆盖三类文档、追问、证据不足、更新删除和失败恢复。

PDF/Word/Excel 与统一来源结构首版已接入；下一步完善解析回归、短期记忆，再引入长期记忆。
文档管理、任务状态和观测随各阶段补齐；每阶段形成能演示、能复测的小闭环。以上为路线记录，本次不新增运行时功能。

### 分块与截断实测（2026-10-13）

已按上述顺序完成首轮离线实验：固定 `bge-small-en-v1.5`，使用已有 10 份 FinanceBench PDF 和 10 道题，
比较原配置 500 词 / 80 词，以及 200/300/400 tokens × 0%/15% overlap 共 7 组。
只做归一化向量检索，无 BM25、reranker、查询改写、Agent 或生成模型调用。

报告：`outputs/chunk_ablation/financebench_20261013/report.html`；同目录保留 CSV、逐题排名、
完整 chunk、向量、固定解析页面、配置与源码/模型文件哈希。运行入口：

```bash
python -m financial_rag.evaluation.chunk_ablation --output outputs/chunk_ablation/another_run
```

- 原配置 2,891 个 chunk 中 1,214 个（42.0%）超过 510 个正文 tokens；模型总上限 512，另含 2 个特殊 tokens。
  中位长度 464、P95 为 698、最长 1,250 tokens；10 个查询均未截断。
- 六组 token 分块均无输入截断，并校验逐页非空白原文完整覆盖。边界优先保留段落，overlap 是可能因词边界缩小的目标值，不跨页。
- 原配置的 Top5 标注页覆盖为 35%，固定 2,000 tokens 返回文本预算下为 30%；300/0 对应 40% / 40%。
- 两组 200-token 配置的预算页覆盖均为 50%，但抽查发现：200/0 在 Pfizer 题只命中页脚，200/30 在 PepsiCo 题命中不含资本开支的同页文字。
  保留原始指标，不把这些页级命中解释成有效证据；报告列出原文供核对。
- 每个 size 下加入 15% overlap 均未提高本轮汇总页覆盖，且增加 chunk 数量；这不证明 overlap 对其他语料无效。

暂留 300 tokens / 0 overlap 为下一轮 embedding 对比候选，200 / 30 为补充对照；此建议结合原文抽查，
不是仅按最高页级分数选型，也未修改应用的默认分块。当前 10 题属于开发集，扩大样本并增加指标行/数值级证据核验后再定默认参数。
原配置与新配置还存在按词/按 token 计数和边界算法差异；六组新配置之间使用同一算法。


### 固定分块的 Embedding 实测（2026-10-13）

复用上轮 300 tokens / overlap 0 的 5,586 个完全相同片段，比较四个 embedding 模型；
仍使用 10 份报告、10 道开发题，仅做归一化 dense 检索，不调用生成模型、Agent、BM25 或 reranker。
报告：`outputs/embedding_ablation/financebench_300_0_20261013/report.html`。

| 模型 | Top5 标注页覆盖 | Top20 标注页覆盖 | 固定 2,000 tokens 预算覆盖 |
|---|---:|---:|---:|
| bge-small | 50% | 50% | 50% |
| bge-base | 45% | 70% | 55% |
| bge-m3 | 20% | 55% | 30% |
| qwen3-0.6b | 63% | 80% | 72% |

BGE-small/base 使用官方推荐查询前缀，BGE-M3 使用无指令 dense 编码，Qwen 使用官方内置 query 指令与 last-token pooling。
另保存 small 无指令旧查询对照。该实验比较模型及其接入方式，不能把收益全归因于模型参数。
统一文本预算按原 BGE-small tokenizer 计数，不随模型改变；所有模型的文档和查询均无截断。

报告包含逐题检索原文、三道题的关键字符串辅助诊断、模型权重/源码哈希、固定版本、索引大小和耗时。
页级覆盖不代表所需指标已找到；small 的 PepsiCo 新增命中不含资本开支行，Qwen 的 Verizon 首位命中仅为页脚。
Qwen 在 Corning 题的前两位则包含成本、库存和应付账款，属于可核对的改善。关键词核对也不等于完整正确性评分。
本机 Qwen 编码期间观察到系统较高 swap 使用，性能数字仅作为当次本机实测参考。
样本为已查看过的开发集，不代表中文检索或全量 FinanceBench 结果；应用默认模型与分块尚未更改。

```bash
python -m financial_rag.evaluation.embedding_ablation --model bge-small --output outputs/embedding_ablation/new_run
# 对同一输出目录依次运行 bge-base、bge-m3、qwen3-0.6b
```

运行依赖已有冻结分块和本地模型。新增模型固定版本见 `models/embedding_comparison_manifest.json`；
结果目录保存模型配置、token 审计、向量、查询向量、排名及源码快照。使用新模型前需独立建立匹配的向量索引。


### 50 题证据内容复测（2026-10-13）

保留 BGE-small 和 Qwen3-Embedding-0.6B，固定 300 tokens / overlap 0，扩展为 50 题、41 份完整报告、18,806 个分块。
原 10 题保留，按固定种子新增 40 题；旧文档的 5,586 个片段逐字段保持一致。
报告：`outputs/embedding_ablation/financebench50_300_0_20261013/report.html`。

| 模型 | Top5 页码命中 / 50 题 | 有效证据命中 / 49 题 | 平均证据单元覆盖 | 全部单元找全 / 49 题 |
|---|---:|---:|---:|---:|
| BGE-small | 36.0% | 28.6% | 24.1% | 18.4% |
| Qwen3-Embedding-0.6B | 52.0% | 38.8% | 29.9% | 22.4% |

逐题建立 116 个证据单元，数值题绑定行名与精确金额，文字题核对关键事实/否定词。
页脚和仅表头无内容的命中不给内容分；允许拼接真实返回且原文相邻的分块，不补入缺失文本。
116 个单元均可在原页解析文本中找到，111 个能在单块中找到，5 个需要相邻块。
Ulta 收购题涉及破折号、年度列和“没有收购”的语义，保留独立版面复核，不进入自动内容指标分母。
报告保留原 10 / 新增 40 分组、Top5/10/20、统一 2,000-token 预算、逐题返回原文与命中依据。

本轮两模型统一 FP16/MPS、batch 8；向量转 FP32 后归一化。文档/查询均无截断，保存实际源码与模型哈希、分块缓存、查询向量及精度抽查。
这些是参考证据对齐的内容规则指标，不是专家语义判断或回答准确率；年份列/单位尚未全面结构化，标注页外等价证据可能漏计。
50 题用于开发，不是未见测试集；新增文档带来额外干扰，因此旧题成绩不能直接与上一轮作因果比较。

```bash
# 使用现有原 10 题数据和本地模型；download 扩充至固定的 50 题 PDF
python -m financial_rag.evaluation.embedding_evidence_ablation download
python -m financial_rag.evaluation.embedding_evidence_ablation prepare --output outputs/embedding_ablation/new_50_run
python -m financial_rag.evaluation.embedding_evidence_ablation bge-small --output outputs/embedding_ablation/new_50_run
python -m financial_rag.evaluation.embedding_evidence_ablation qwen3-0.6b --output outputs/embedding_ablation/new_50_run
python -m financial_rag.evaluation.embedding_evidence_ablation score --output outputs/embedding_ablation/new_50_run
python -m financial_rag.evaluation.embedding_evidence_ablation report --output outputs/embedding_ablation/new_50_run
```

精度核验依赖上一轮 10 题 FP32 向量；已完成本次目录的实际运行脚本另存于其 `scripts/` 下。
证据规则纳入 `src/financial_rag/evaluation/data/financebench50_evidence.json`；结果 CSV、可达性和人工复核记录随报告保存。
应用默认检索与分块配置未改动。


## 统一文档输入首版（2026-10-13）

本轮将新 FastAPI 网页入口接到统一解析管线。旧 `parse_pdf`、`chunk_pages`、FinanceBench 及 embedding
实验入口保持原行为，历史报告不代表新结构化解析的召回效果；后续应单独进行解析/分块回归实验。

### 数据契约与来源

运行时契约在 `src/financial_rag/ingestion/document.py`，由 Pydantic 校验；可用
`Document.model_json_schema()` 取得 JSON Schema，仓库同时提供 `docs/document.schema.json`。层级为 `Document → Element / Table → Cell`：

- `document_id` 是原文件字节 SHA-256，绑定文件版本；`schema_version=1.0`，`parser_version=native-v1`。
- `elements` 保留正文顺序、段落/标题/表格类型及来源；`tables` 单独保存完整网格和合并单元格的锚点、跨度。
- PDF：1-based 物理页码，`bbox=[x0,y0,x1,y1]`，单位为 PDF point、未旋转页面左上角坐标系。表格行片段还附单元格 bbox 和表头 bbox。
- Word：正文段落序号、标题路径、表格序号和行列；序号从 1 开始。没有稳定页码，`page=null`。
- Excel：工作表名与 A1 范围，例如 `'Income statement'!B4:D4`；保留合并范围、原值类型、数值格式、公式及缓存结果。表内行列从有效区域左上角的 1 开始，真正来源以 A1 坐标为准。
- 空单元格 `value=null`、数字零和字符串横线分别保留。公式不执行；`cache_status=missing/unverified` 区分缺失与新鲜度未知，避免把缓存当成新计算结果。日期时间保存 ISO 字符串。
- Chunk 保留 `source/source_label`、`element_id/context_refs`、`table_id/cell_refs/header_cell_refs`。
  正文 `text_spans` 是原 element 内字符区间；表格 `text_span` 是该行检索文本（不含重复前缀）的字符区间。
  chunk ID 包含文档哈希、解析版本、预算、序号及文本/来源摘要，防止改文后沿用旧证据。

### 检索视图与保存

网页使用本地 BGE-small tokenizer，单片最多 **300 tokens（含特殊 token）、overlap 0**。
正文按来源分组；表格按行生成视图，重复候选表头，并在预算允许时附邻近标题/单位文字。
超长行继续分片、保留原单元格指针；过长表头不静默截掉，而是保留独立索引行和完整表格结构。
候选表头为启发式识别，明确标记 `header_origin=heuristic`；Word 显式重复表头可标 `explicit`。
完整表格不受 embedding 长度限制。原生 PDF 中已有的保守重复页脚过滤只影响检索视图，不删除结构化原文。

每个临时工作区保存原始文件、`<document_id>.json` 完整结构和 `vectors.sqlite3`。
同内容重复上传复用已有解析及向量；当前没有跨服务重启的持久解析缓存。
新格式的来源字段贯穿 Dense、BM25、RRF、证据注册、最终引用及每轮检索记录。
评测导出包含原始三类文件及**入库当时**的 `structured/` 快照，保存时不再重新解析新格式。

网页文档列表支持下载原文、查看结构；表格引用可打开完整表格预览。
预览最多显示前 100 行、50 列，完整内容在 JSON 中。接口均绑定当前浏览器会话：

- `GET /api/documents/{document_id}/original`：下载原始文件。
- `GET /api/documents/{document_id}/structure`：返回统一结构、单元格来源、解析警告及 OCR 待办。

独立解析，不调用模型或 API：

```bash
uv run python -m financial_rag.ingestion report.pdf report.docx report.xlsx --output outputs/parsed_documents
# 可选：同时输出 300-token 检索视图；只加载本地 tokenizer，不加载 embedding 权重。
uv run python -m financial_rag.ingestion report.xlsx --output outputs/parsed_documents --tokenizer models/bge-small-en-v1.5
```

### OCR 接口与已知边界

`OCRBackend.recognize(image_bytes, OCRRequest) -> OCRResult` 定义后续模型适配入口；结果要求包含
元素/表格、引擎和版本。当前**主应用未接入 OCR，也不自动执行该接口**；独立实验环境已准备模型并开始试跑，随后按用户要求暂停，详见下节。
无文字页以及含大幅图像的混合页生成页码/区域级 `needs_ocr` 请求；检测是启发式，空白页也可能进入待核验列表。
混合 PDF 的可读内容可以入库，界面及检索上下文会提示未识别页；纯扫描 PDF 因无可索引内容会拒绝入库并明确提示 OCR 未接入。

原生解析首版限制：

- PDF 使用 PyMuPDF 线框表检测，并对明确年份列、重复数值对齐的无框表增加保守恢复（native-v2）；复杂多栏和跨页续表仍需核验，暂不自动拼表。
- Word 解析正文段落和顶层表格，保留横纵合并。嵌套表、页眉页脚、文本框、批注、修订和图片内容尚未全面解析，记录相关警告；不虚构 Word 页码。
- Excel 每工作表保存一个有效区域，前置合并标题及下一行为候选表头。复杂多表布局、图表、外链数据刷新和公式重算未实现；隐藏工作表也会解析并提示。有效区域上限 200000 个单元格。
- `.doc/.xls/.xlsm` 暂不支持。Office 解压总量限 256 MiB；沿用上传每份 20 MiB、资料库 100 MiB/10 份限制。
- 本轮验证结构与来源贯通，不宣称提升金融问答或检索准确率；不同格式数值冲突的回答质量需要单独评测。

自动测试覆盖真实生成的 PDF 线框表、扫描待办、Word 横纵合并、Excel 多工作表/合并区域/空值/零/横线/
格式/公式缓存、token 上限、SQLite 往返、检索融合到引用、混合上传追加、会话隔离和清空后的评测导出。


真实文档冒烟检查：现有 `BESTBUY_2024Q2_10Q.pdf` 全部 30 页完成解析，原生检测返回 56 个候选表、
6763 个单元格和 543 个检索片段；实际 BGE tokenizer 的片段上限为 300 tokens。
这项检查验证管线与来源有效性，候选表数量不代表人工确认的表格识别准确率。未调用生成 API，也未重跑 embedding 对比。


## OCR 公开基准实验（截至 2026-10-14：已暂停）

从 OmniDocBench 固定抽取 100 页，保存数据版本、图片哈希、分层清单和原始标注。采用完整文档解析方案对比：

| 方案 | 本机尝试方式 | 当前结果 |
|---|---|---|
| PP-StructureV3 | PaddleOCR 完整管线，CPU；关闭公式/图表/印章识别 | 4 页成功，约 145–216 秒/页 |
| PaddleOCR-VL-1.6 | 已准备模型及完整管线 + 本地 MLX 服务适配脚本 | 尚未执行推理，兼容性未验证 |
| MinerU 4.0.12 | standard，图像 OCR，本地 llama.cpp 后端 | 首页尝试失败，需排查 |

当前机器为 16 GiB Apple Silicon Mac。上述 PP 耗时来自少量页面，包含完整解析与可能的首次初始化成本，不能推断其他硬件吞吐量，也不能得出三方案质量排名。
计划采用官方评测代码比较文字 Edit Distance、表格 TEDS/TEDS-S 和阅读顺序；已跑通 PP 单页评测冒烟，未完成 100 页正式比较，不报告官方 Overall。

样本含报告类 40 页、复杂表格 20 页、模糊扫描 5 页、其他版式 20 页及补足样本 15 页。报告类含技术资料，不全是财报；语言为简体中文 57、英文 36、繁体中文 4、中英混合 3 页。
20 页用于适配、80 页预留比较；文件名只能近似推断同源文档，不能保证严格文档隔离。公开标注也不提供本项目所需的金融事实准确率。

- 样本与状态：`outputs/ocr_ablation/omnidoc100_20261013/sample_manifest.json`、`experiment_state.json`。
- 准备与运行脚本：`scripts/prepare_ocr_sample.py`、`run_ocr_paddle.py`、`run_ocr_mineru.py`。
- 评测与报告：`scripts/evaluate_ocr.py`、`build_ocr_report.py`；报告保留失败状态，未评测项不记成零分。
- 独立环境：`.venv-ocr`、`.venv-mineru`、`.venv-ocr-eval`，依赖锁定快照在实验目录；与主应用 `.venv` 分开。

推理已停止，模型、样本和已完成结果保留。缩小样本和恢复执行仍是后续工作，本次文档更新不启动实验。


## 候选数量与 Top K 对照（2026-10-15）

使用已下载的 **50 题、41 份完整报告、6,000 页、18,806 个冻结片段**，对比每路候选数 20/50 与重排后 Top 5/7/10，共 6 组。分块固定为旧 PDF 实验的 300 content tokens / overlap 0；不改变新网页的结构化解析。

文档和查询重新用 BGE-small FP32 编码；查询为英文原题加目标报告名，不使用查询指令、改写或文档过滤。Dense 与 BM25 分别取 N 条，经 RRF（k=60）保留 N 条，再用应用的 BGE-reranker-v2-m3（2048 tokens）重排。同一候选池仅重排一次，Top 5/7/10 使用相同排名前缀。

| 每路候选 / RRF保留 | 最终Top K | 有效证据命中 | 平均证据单元覆盖 | 全部证据找全 | 平均正文tokens |
|---|---:|---:|---:|---:|---:|
| 20 | 5 | 38.8% | 30.3% | 20.4% | 1255 |
| 20 | 7 | 42.9% | 34.0% | 24.5% | 1735 |
| 20 | 10 | 42.9% | 34.0% | 24.5% | 2434 |
| 50 | 5 | 34.7% | 28.8% | 20.4% | 1269 |
| 50 | 7 | 38.8% | 31.5% | 22.4% | 1745 |
| 50 | 10 | 44.9% | 37.8% | 28.6% | 2476 |

基线 20/5 的平均证据覆盖为 **30.3%**，本轮最高为 **50/10 的 37.8%**。完整报告同时列出页级指标、固定 2,000-token 预算、候选并集/RRF阶段覆盖、原10题/新增40题分组、逐题原文与配对变化，避免只用更长输入解释为检索策略改善。

本轮取舍：**候选20＋Top7适合作为下一轮端到端验证的优先配置。** 相比20/5，平均证据覆盖由30.3%提高到34.0%，3题改善、0题下降，全部证据找全由10/49提高到12/49。20/10与20/7的内容指标相同，但平均正文由1735增加到2434 tokens（约增加40%）。

50/10的未限预算覆盖最高（37.8%、全部找全14/49），但检索＋重排中位耗时约3.05秒，20候选约1.25秒；固定2000正文tokens后，50/10覆盖为33.0%，略低于20/7的34.0%。这些是本轮开发集观察，尚不能证明Top7会降低LLM调用次数或提升最终回答正确率。

**口径与限制：** 页级分母50，内容规则分母49；Ulta收购题保留人工复核，不记零分。支持片段比例仅代表参与既有证据标注匹配的片段，不能等同真实相关率。正文tokens用BGE-small计数，不含提示词/来源包装/历史，也不是DeepSeek计费tokens。计时包含共享的Dense/BM25 Top50获取及各池实际融合、重排，排除模型加载和文档/查询编码；同池不同K共用重排耗时，未测生成阶段。

本轮**未运行查询规划、Agent或答案生成**，不报告回答准确率或LLM调用次数下降。题集已用于开发，结果不是未见测试集结论；年份列、单位和标注页外等价证据尚未全面语义评分。应用默认参数仍为候选20 / Top5，后续应对有希望的配置做端到端调用数、回答正确性与耗时对照。

- [完整报告及逐题证据](outputs/retrieval_k_ablation/financebench50_20261015/report.html)
- [结果 CSV](outputs/retrieval_k_ablation/financebench50_20261015/comparison.csv) · [配对变化](outputs/retrieval_k_ablation/financebench50_20261015/paired_comparison.json) · [验证记录](outputs/retrieval_k_ablation/financebench50_20261015/verification.json)

复现（使用本地已有数据和模型，选择新的输出目录）：

```bash
python -m financial_rag.evaluation.retrieval_k_ablation --project-root . --output outputs/retrieval_k_ablation/new_run
```


## 批量初始查询与原 Loop 对照（2026-10-15）

基于相同50题、41份报告和18,806个300-token/0-overlap片段，真实调用现有DeepSeek接口。先在固定种子、三类各4题的12题上筛选 **初始查询上限1/2/3 × 首批证据总数5/7/10，共9组**，再将覆盖和调用效率代表方案及全部单查询对照扩展到50题。

筛选晋级：覆盖优先 `q2_k7`，调用效率优先 `q3_k5`；实际晋级组在结果表列出。12题用于参数选择，另38题（37题可自动内容评分）单独报告，不把全部50题当作独立测试集。

**公平性：** 多条初始查询共享首批K个片段，按商余均分并共享ID去重；不是每条查询各取K。查询之间没有中间LLM调用，本机检索/重排仍串行。首批之后沿用原Loop的缺口补查、最多7个模型步骤/8次检索和相同字符预算；批量初始查询也计入8次检索预算。候选数固定20；后续默认TopK与该组K一致，模型仍可按工具协议指定top_k。

| 配置（查询上限/首批总K） | 首次证据覆盖 | 最终证据覆盖 | 全部证据找全 | 平均LLM调用 | 平均墙钟秒 | 去本地排队秒 |
|---|---:|---:|---:|---:|---:|---:|
| 1/5 | 42.1% | 76.8% | 31/49 | 3.24 | 18.4 | 16.5 |
| 1/7 | 47.4% | 76.9% | 33/49 | 3.28 | 18.3 | 16.0 |
| 1/10 | 47.4% | 70.0% | 28/49 | 2.98 | 15.5 | 13.7 |
| 2/7 | 50.0% | 82.0% | 36/49 | 3.08 | 19.3 | 16.8 |
| 3/5 | 45.9% | 64.7% | 26/49 | 3.12 | 20.0 | 17.0 |

未参与本轮参数筛选的38题：

| 配置 | 最终证据覆盖 | 全部找全 | 平均LLM调用 | 去本地排队秒 |
|---|---:|---:|---:|---:|
| 1/5 | 75.4% | 23/37 | 3.32 | 15.4 |
| 1/7 | 72.7% | 24/37 | 3.34 | 16.3 |
| 1/10 | 70.0% | 22/37 | 3.08 | 14.3 |
| 2/7 | 79.1% | 27/37 | 3.11 | 15.8 |
| 3/5 | 57.6% | 17/37 | 3.24 | 16.9 |

本轮结论：**最多2条初始查询、首批总7条证据有希望改善覆盖，但尚未证明提速。** 全50题中，相比同K单查询，最终证据覆盖由76.9%提高到82.0%，全部找全由33/49提高到36/49，平均LLM调用由3.28降到3.08；平均墙钟耗时由18.3秒增加到19.3秒。剩余38题覆盖由72.7%到79.1%，调用由3.34到3.11，但配对覆盖差异的bootstrap区间跨0，不能宣称稳定提升。双方正常完成的剩余题中，耗时差异接近零。

**不建议使用本轮3条查询/首批总5条的配置。** 它在筛选12题上的优势没有保持：全50题最终覆盖降至64.7%，剩余38题仅57.6%，虽略少调用却更慢。由此不能推断所有三查询方案都差；本轮只将3/5晋级到完整题集。网页默认继续保留原单查询Loop。

**口径：** LLM调用包含规划、补查决策和最终回答，HTTP重试另计，tokens来自API usage。全部配置temperature=0、max_tokens=8192；最多4个独立问题并发，GPU工作串行，故同时报告墙钟和扣除本地锁等待的时间，仍受网络/服务端缓存影响。完整报告保留规划回退、API错误、停止原因、逐题答案及首次/最终证据，不用“模型正常停止”替代证据完整性评分。

**限制：** 内容评分仍是对齐标注的规则覆盖率，不是最终答案准确率；49题自动、1题人工，50题均统计调用和耗时。该语料此前已用于开发，38题仅未参与本轮超参数筛选。网页默认保持原单查询Loop；批量规划目前仅支持独立问题，不用于带会话历史的追问。

- [完整报告、9组筛选及逐题结果](outputs/batch_query_ablation/financebench50_20261015_8192/report.html)
- [汇总CSV](outputs/batch_query_ablation/financebench50_20261015_8192/comparison.csv) · [配对差异](outputs/batch_query_ablation/financebench50_20261015_8192/paired_comparison.json) · [调用与验证记录](outputs/batch_query_ablation/financebench50_20261015_8192/verification.json)

复现（会调用已配置的DeepSeek API；使用新的输出目录）：

```bash
python -m financial_rag.evaluation.batch_query_ablation --project-root . --output outputs/batch_query_ablation/new_run --workers 4
```


### 批量查询结果的失败与耗时核验

DeepSeek默认启用thinking/high；请求虽传入temperature=0，官方说明该模式会忽略此参数，因此本实验不能视为确定性重复。8192输出上限包含内部推理，部分输出仍可能截断；这类失败单列，不改写成检索失败或正确答案。[DeepSeek说明](https://api-docs.deepseek.com/guides/thinking_mode/)

以下为剩余38题中，**双方都正常完成**的配对差异（批量减去同K单查询）。它用于检查输出失败是否误导调用/耗时结论；子集会变化，不能替代全样本结果。

| 批量 vs 同K基线 | 双方完成题数 | 最终覆盖变化（百分点） | LLM调用变化 | 去本地排队秒变化 |
|---|---:|---:|---:|---:|
| q2_k7_vs_q1_k7 | 31 | +3.9 | -0.16 | -0.0 |
| q3_k5_vs_q1_k5 | 29 | -13.4 | -0.03 | +2.0 |

[失败敏感性与内部推理tokens](outputs/batch_query_ablation/financebench50_20261015_8192/failure_sensitivity.json)。4096上限的早期试跑已停止并独立保留，不混入上述正式结果。

## 自由规划初始查询：粗略试跑（2026-10-15）

按用户要求停止完整实验，保留60次运行结果。五组分别完成10、17、13、9、11题，题目不完全重合；这是粗略试跑，不是完整50题对照。
自由规划共33次：1条查询20次、2条10次、3条3次。取消规划数量上限后，模型没有普遍生成大量查询，多数问题仍选择一条。
所有已保存案例均无模型错误，也没有规划查询因数量预算未执行；部分案例触及原有循环停止预算。自由规划路径可正常工作，但当前样本不能判断它比单查询/最多两条更好，也不能据此选择证据预算。
每组原始覆盖、调用和耗时已保留供检查。关键配对只有3道共同题，其中部分配对仅2道可自动评分；不给出优胜配置或显著性结论。应用默认保持原单查询。

**配置：** 单查询/首批8、最多2条/首批8、自由规划/首批8、12、16；后续默认Top7统一。候选20、RRF k60、原模型与300-token/0-overlap语料保持一致。自由规划使用`initial_query_count=None`，去掉规划提示词和解析器的数量上限；执行仍保留全任务8次检索、7步的原预算，未执行查询单列保存。

**状态：** 用户要求只做粗略实验后停止完整运行；250次原计划只保存60次，未补齐。原始各组均值、每条查询与回答、少量共同题配对、停止原因均在报告中，不能将不匹配的组均值当成公平对照。

核验：59项相关测试通过；本次实际184次SDK调用、184次HTTP尝试，1,223,592个API tokens；0个模型错误，0个长度截断响应。171条执行查询的embedding长度与重排截断检查已保存。

- [粗略试跑报告](outputs/adaptive_query_ablation/financebench50_20261015/report.html)
- [停止状态](outputs/adaptive_query_ablation/financebench50_20261015/run_status.json) · [原始汇总](outputs/adaptive_query_ablation/financebench50_20261015/summary.json) · [少量共同题](outputs/adaptive_query_ablation/financebench50_20261015/partial_pairs.json)

## 相同10题：自由规划与综合营收分析（2026-10-15）

**范围：** 7道原FinanceBench题＋3道本地扩展综合营收分析题，共10道相同题，每组完整重跑一遍，仅30次。原题覆盖直接提取、比率/多步计算、经营解释和地域信息；扩展题为Amazon FY2017、Adobe FY2022、3M FY2022，要求分析全年营收、业务构成及变化原因。扩展题不属于官方FinanceBench标签，结果分开汇总。

**设置：** 最多两条/首批8、自由规划/首批8、自由规划/首批12。后续默认Top7一致；候选20、RRF k60、原BGE模型与300-token/0-overlap语料保持不变。最多8次检索、7步。三组均新运行；2个问题并发、本地模型串行。

| 配置 | 首次覆盖 | 最终覆盖 | 全部找全 | 平均LLM调用 | 首批查询均值 | 墙钟秒 | 去本地排队秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 最多2条 / 首批8 | 52.5% | 75.8% | 6/10 | 3.20 | 1.60 | 19.2 | 18.1 |
| 自由规划 / 首批8 | 59.2% | 67.5% | 6/10 | 3.10 | 2.10 | 21.0 | 19.6 |
| 自由规划 / 首批12 | 60.8% | 78.3% | 7/10 | 3.10 | 2.40 | 21.3 | 20.1 |

| 配置 | 原7题最终覆盖 | 综合3题最终覆盖 | 综合3题全部找全 | 综合3题LLM调用 |
|---|---:|---:|---:|---:|
| 最多2条 / 首批8 | 67.9% | 94.4% | 2/3 | 3.33 |
| 自由规划 / 首批8 | 53.6% | 100.0% | 3/3 | 3.33 |
| 自由规划 / 首批12 | 71.4% | 94.4% | 2/3 | 3.67 |

这轮相同10题中，最多两条/首批8的最终规则证据覆盖为75.8%，平均3.20次LLM、19.2秒；自由规划/首批8为67.5%、3.10次、21.0秒；自由规划/首批12为78.3%、3.10次、21.3秒。自由规划没有表现出明确的提速，调用节省仅平均0.1次；不能据此认定自由规划全面优于最多两条。
三道综合营收题中，自由规划实际生成3～4条首批查询。自由规划/首批8的最终规则覆盖100%，另外两组均94.4%；但最多两条和自由规划/首批8的LLM调用均为3.33次，自由规划/首批12为3.67次，耗时分别24.6、27.6、30.0秒。自由规划能拆开信息需求，但本轮未证明能减少综合分析的模型调用。
逐题差异很集中：自由规划/首批8相对最多两条，8题覆盖不变、3M综合题多匹配1/6项、PepsiCo资本开支题从全匹配变为未匹配；后者在三组中都只规划了一条初始查询，不能把漏查直接归因于查询数量变多。自由规划/首批12相对最多两条，9题覆盖不变，只有Verizon多匹配1/4项，带来总体+2.5个百分点。单个案例和生成随机性对这个小样本影响很大。
自由规划/首批12的Adobe综合题达到证据字符预算，以limited状态结束；它仍匹配了全部预设证据，但不算正常完成。另29次均正常完成，30次没有API/模型错误或输出长度截断。全部结果保留，不排除受限案例。
当前建议：不把所有问题统一切成自由规划/首批12。最多两条/首批8继续作为比较基线，自由规划/首批8保留为综合任务的候选方案；尚未证明按任务类型切换会更优，也没有修改应用默认。候选20、RRF60、后续Top7暂时保持。规则证据覆盖不等于答案正确率：例如3M的未匹配单元要求特定表格的行及表头，回答中出现正确数字或引用另一张表并不会自动获得该项分数；此次未事后修改评分规则。

**评分：** 运行前固定三道综合题及每题六个证据单元，验证完整语料18项均可匹配、空证据得零，并锁定题目、规则和源代码哈希。评分标签与参考答案不传入模型。规则覆盖检查所需数字/文本是否被检索到，不能替代答案正确率；同义或不同表述的等价证据可能未被规则认可。

**核验与限制：** 94次SDK调用/94次HTTP尝试，620,788个API tokens；0个模型错误、0个长度截断响应。相关59项测试通过。本轮仅10题单次运行，服务端缓存/网络和模型推理随机性影响结果；不作统计显著或通用最优结论，不与前一次不同题目的部分均值混合。应用默认仍为原单查询。

- [完整报告与逐题查询/回答](outputs/matched_query_ablation/revenue10_20261015/report.html)
- [配对变化](outputs/matched_query_ablation/revenue10_20261015/paired_comparison.json) · [固定题目](outputs/matched_query_ablation/revenue10_20261015/selected_cases.json) · [预设评分依据](outputs/matched_query_ablation/revenue10_20261015/evidence_criteria.json) · [核验](outputs/matched_query_ablation/revenue10_20261015/verification.json)

复现相同小样本（三组各10题，真实调用API，使用新输出目录）：

```bash
python -m financial_rag.evaluation.matched_query_ablation --project-root . --output outputs/matched_query_ablation/new_run
```

## Retriever证据选择：有用性、重复与遗漏（2026-10-15）

**本轮生成式API调用0次，网络连接尝试0次。** 复用上一轮相同10题（7道FinanceBench＋3道综合营收分析），固定原题＋目标报告名，不做rewrite、不跑Loop。候选池保持20，BGE重排每题只做一次；对比原Top7、阈值0.1/0.3/0.5、各阈值配MMR相关性权重0.6/0.8，共10组。

| 策略 | 返回块数 | 直接证据 | 辅助信息 | 直接有用比例 | 含辅助比例 | 完全重复块 | 原规则覆盖 | 平均正文tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 Top7 | 70 | 14 | 9 | 20.0% | 32.9% | 4 | 44.3% | 1800 |
| 阈值 0.1 | 60 | 14 | 9 | 23.3% | 38.3% | 4 | 44.3% | 1562 |
| 阈值 0.3 | 58 | 13 | 9 | 22.4% | 37.9% | 4 | 37.7% | 1503 |
| 阈值 0.5 | 48 | 12 | 6 | 25.0% | 37.5% | 4 | 37.7% | 1249 |
| 阈值0.1 + MMR相关性权重0.6 | 60 | 10 | 9 | 16.7% | 31.7% | 0 | 42.7% | 1532 |
| 阈值0.1 + MMR相关性权重0.8 | 60 | 13 | 10 | 21.7% | 38.3% | 3 | 44.3% | 1541 |
| 阈值0.3 + MMR相关性权重0.6 | 58 | 9 | 9 | 15.5% | 31.0% | 0 | 36.0% | 1477 |
| 阈值0.3 + MMR相关性权重0.8 | 58 | 12 | 10 | 20.7% | 37.9% | 3 | 37.7% | 1483 |
| 阈值0.5 + MMR相关性权重0.6 | 48 | 8 | 6 | 16.7% | 29.2% | 0 | 36.0% | 1223 |
| 阈值0.5 + MMR相关性权重0.8 | 48 | 11 | 7 | 22.9% | 37.5% | 3 | 37.7% | 1228 |

温和阈值0.1在这组开发题上最值得保留为候选：原Top7共返回70块，其中14块直接证据、9块辅助信息、47块无关；阈值0.1返回60块，仍保留14块直接证据和9块辅助信息，只移除10块被复核为无关的内容。直接有用比例20.0%→23.3%，含辅助信息的比例32.9%→38.3%；原规则覆盖44.3%不变，平均正文tokens约1800→1562（减少13.2%）。这只是10题上的观察，不是已校准的通用阈值。

阈值不是越高越好：0.3和0.5均丢掉MGM全年Regional和China两项标注证据，原规则覆盖降至37.7%，并有1题返回空集。补充原文复核还发现0.5丢掉了American Express主要经营地区的等价证据，原规则本来就未识别该页，所以仅看标注覆盖会漏掉这一损失。reranker分数即使在0～1，也不代表正确概率。

MMR去重存在取舍：阈值0.1、相关性权重0.8时，完全不新增相关事实的重复片段由4块减到3块，直接证据由14块变13块；权重0.6时重复减到0，但直接证据降至10块、含辅助信息的比例为31.7%，有些空出的名额被其他无关片段占用。补充原文复核显示这两组的必要信息覆盖没有低于基线，因此不能把所有被删直接片段都视为不可替代的必要证据；但本轮也没有显示去重带来更多必要信息。

复核揭示了两个不同问题：候选中存在高分但公司/期间错误的内容，例如PepsiCo题里Costco资本开支分数约0.879；同时标注规则会漏认等价来源，例如American Express重要海外市场在第5页、Adobe完整分部营收在第65页。原Top7已覆盖此次20候选池中全部能被现有规则识别的证据，原规则均值均为44.3%；仅调整这20条的选择方式，无法恢复根本没进入池中的所需事实。

建议先把阈值0.1作为后续小样本候选，暂不启用强MMR或高阈值；应用默认未改。下一步更值得检验明确公司/目标报告约束和期间核对，减少高分错对象内容，而不是继续增加LLM调用。未开展额外实验，也没有调用答案生成验证最终正确率。

**评估口径：** 82个问题/片段组合由助手逐条读原文标注，非独立人工评审；评审无API。2=直接支持答案事实，1=辅助/部分或财年有歧义，0=无关。有用比例为所有返回片段的微平均，不把未命中标注直接判为无用。重复指按返回顺序不再增加回答相关事实标签，不因年份不同但文字相似而删证据。原始内容覆盖仍按冻结规则；另附9题等价来源补充复核（Ulta财年歧义排除），该补充映射在查看规则汇总后添加，非预注册，不覆盖原始评分。

**验证：** 25项相关测试通过；同候选/同分数、阈值无回填、相同10题、零网络尝试、输入未截断均已核验。10次本地检索与重排共14.0秒（不含加载/复核），选择附加计算不足1毫秒/题。未生成答案，不将此单次检索结果与旧Loop最终覆盖直接比较。应用默认保持原TopK选择；新选择器为可选函数，尚未接入默认网页流程。

- [完整报告、10组参数、逐片段复核](outputs/evidence_selection_ablation/matched10_20261015/report.html)
- [复核标注](outputs/evidence_selection_ablation/matched10_20261015/review_annotations.json) · [逐题遗漏](outputs/evidence_selection_ablation/matched10_20261015/reviewed_results.json) · [核验](outputs/evidence_selection_ablation/matched10_20261015/verification.json)

本地复现候选与选择（不调用API，使用新目录；有用性复核需另行执行离线分析）：

```bash
python -m financial_rag.evaluation.evidence_selection_ablation --project-root . --output outputs/evidence_selection_ablation/new_run
```
