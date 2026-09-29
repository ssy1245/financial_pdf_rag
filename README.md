# Financial PDF RAG

面向财报的 RAG 学习项目，当前已完成多 PDF 上传、检索、重排、DeepSeek 生成与引用展示。
检索层已拆为独立 Retriever，支持按 chunk_id 排除已获取片段。
已实现请求内证据记忆、search_documents 工具和显式 Agent loop；ask_agent() 可由模型选择补充检索。
默认 ask() 保留单轮基线；网页已使用带单条初始查询规划的 ask_agent()。

## 快速启动

```bash
uv sync --locked
uv run prepare_models.py  # 从已有缓存复制；首次无缓存时加 --download
uv run app.py
```

浏览器打开 http://127.0.0.1:8000，也可在 macOS 双击 `start.command`。
在项目 `.env` 设置 `DEEPSEEK_API_KEY`；Flask 入口自动加载，密钥只在后端使用。
拖入/多选 PDF（可分次添加，待上传列表支持移除）→ 开始阅读 → 输入问题 → 展开文件名、PDF 页码和引用原文。
前端是 HTML/CSS/JavaScript，不需要 Node 构建，但必须启动 Python 后端，不能仅双击 HTML。

- 资料库累计最多 10 份，每份 20 MB，总计 100 MB；跨批次同内容按哈希去重。重复上传不新增文档，清空按钮才会移除已有资料库。
- 多份文档共用当前会话的索引；新上传文档追加到旧资料库，复用已有向量；全部处理成功后切换索引，失败保留旧数据。
- 索引临时保存，服务重启需重新上传；不同浏览器会话隔离，模型共享并串行执行。
- 历史问答仅展示，每题独立检索；不会自动理解“它”“刚才那个”等历史指代。
- 相关证据片段与问题会发送给 DeepSeek。扫描件无 OCR，当前检索更适合英文。
- 当前全局 Top 5 不保证每份 PDF 都有证据，跨文档比较需要后续专门评估。

## 当前实现与文件职责

| 文件/模块 | 已实现 | 主要限制 |
|---|---|---|
| ingestion/parser.py | PyMuPDF 文本块、坐标、物理页码 | 无 OCR、可靠表格结构恢复 |
| ingestion/normalizer.py | 块内清洗、保留段落、保守移除重复页脚 | 不恢复复杂版式 |
| ingestion/chunker.py | 段落分块、长段切分，500 词/80 词重叠 | 按词而非 token，不跨页 |
| indexing/embeddings.py | BAAI/bge-small-en-v1.5，归一化向量 | 长输入截断尚待评估 |
| indexing/vector_store.py | SQLite 保存/加载 chunks、float32 向量与模型配置 | 整批替换，无增量更新 |
| retrieval/dense.py | NumPy 点积检索 | SQLite 仅存储，查询仍在内存完成 |
| indexing/bm25_index.py、retrieval/sparse.py | TF/DF/IDF、BM25 和正分 Top K | 基础英文分词，内存索引 |
| retrieval/fusion.py | RRF，保留两路排名 | 不保证融合优于单路 |
| retrieval/reranker.py | CrossEncoder，记录输入截断 | 无分窗，分数不代表正确率 |
| memory/evidence.py | 请求内证据累积、去重、稳定编号、查询历史和上下文组装 | 仅内存，不持久化；无自动上下文预算 |
| generation/citations.py | 证据编号、来源映射、未知编号检查 | 不核验事实是否被证据支持 |
| generation/prompt.py、generator.py | 基于证据的提示词与 DeepSeek API | 支持 generate_turn 工具调用适配 |
| pipeline.py | FinancialRAG.retrieve/ask，实时完整单轮问答 | 检索委托给独立 Retriever；另有 ask_agent() 多轮入口 |
| retrieval/retriever.py | Dense/BM25 → RRF → 重排，支持 exclude_chunk_ids 和按次 top_k | 按 ID 去重，不做语义去重；不保存请求记忆 |
| app.py | Flask 多 PDF 会话、统一入库和问答 API | 本地开发服务，无登录和公共部署 |
| templates/index.html、static/ | 多 PDF、实时进度、流式回答、耗时与逐轮证据展示 | 历史问答仅展示，未传回模型 |
| schemas.py | 当前数据契约与明确标为 Draft 的未来契约 | TypedDict 不做运行时校验 |

当前网页与 pipeline 使用 BGE reranker v2-m3、2048 tokens；生成默认 deepseek-flash。
Reranker 类及批量对比入口仍默认 MiniLM、512 tokens，不要混淆入口的默认配置。
统一 CLI、config、正式评估指标等占位模块尚未全面实现。

## 已跑通的两条流程

```text
入库（仅新增/替换文档时）：
PDF → 解析 → 清洗 → 分块 → 文档向量 → SQLite

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

下一步优先评估结构化分块与检索过滤，另保留跨问答记忆计划；以检索相关性和证据覆盖评估效果。

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
当前网站未实现公共部署、权限系统、并发优化和会话资料库长期持久化。

## Schema 与版本约定

`schemas.py` 保留检索/实验结构，并补充目前 pipeline 与网页实际输出类型。
无 Draft 前缀的类型描述当前证据、工具、Agent、网页与流事件；Draft 前缀仅描述尚未实现的跨问答记忆。
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
实际相关性改善仍需标注评测。对话记忆、意图分类与动态融合权重未包含在本次实现中。

## 后续计划二：跨问答的对话上下文记忆（尚未实现）

已有 EvidenceMemory 只服务一次问题内的多轮检索。网页虽然保留聊天记录，当前后端和模型
不会读取上一轮提问与回答；不能据此声称支持“它呢”“上一年呢”等追问。

首版建议新增 `memory/conversation.py`，在后端当前浏览器会话内用 Python 内存保存最近
5 个已完成问答对（数量可配置），包括用户问题、最终回答、来源引用和资料库版本。
每次新问题先读取有界历史，辅助理解公司、指标、期间和指代，再创建独立 EvidenceMemory。
历史回答是上下文线索，不是新的事实证据；财报事实仍需可回查的原文支持，不能反复引用
先前模型生成的数字作为 ground truth。指代不明确时应澄清，不擅自猜测。

- 对话记忆按会话隔离，不放进共享模型或 Retriever；失败/中断的输出不当作已完成回答。
- 分别限制最近轮数与历史字符数，首版先裁剪最旧问答，不引入额外摘要模型或数据库。
- 引用按 turn_id + citation_id + workspace_version 定位，上一轮 E1 不能直接当成本轮 E1。
- 首版清空或成功变更资料库后重置对话上下文；重复上传无内容变化不必重置。
- 服务重启丢失内存；长期持久化暂不在首版范围内。

`DraftConversationTurn`、`DraftConversationMemory`、`DraftContextualQuestion` 描述以上草案，
不改变当前运行行为。验收：能承接“那上一年呢”；新会话无旧历史；资料库变更后不串旧引用；
超预算正确裁剪；历史中的错误答案不能覆盖文档事实。

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
