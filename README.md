# Financial PDF RAG

面向金融 PDF 的检索与问答学习项目。完整规划见 [项目规格说明](financial_pdf_rag_project_spec.md)。

当前源码只有架构和职责说明，尚未实现 PDF 处理、检索、问答或命令行入口。

## 环境管理

使用 uv 管理 Python、依赖和虚拟环境。开发解释器固定为 Python 3.11，项目声明支持 Python 3.11 及以上。

首次使用或拉取项目后，在项目根目录运行：

```bash
uv sync --locked
```

uv 会根据 `.python-version` 选择解释器，并按 `uv.lock` 创建或同步 `.venv`。缺少 Python 时，默认允许 uv 下载对应解释器。

检查源码包是否可导入：

```bash
uv run python -c "import financial_rag; print(financial_rag.__file__)"
```

运行命令通常不需要手动激活环境，使用 `uv run` 即可。PyCharm 等编辑器请选择项目下 `.venv/bin/python` 作为解释器。

## 依赖维护

当前仅包含 PDF 解析的基础依赖 PyMuPDF，以及开发测试工具 pytest；源码尚未使用它们。

```bash
# 添加运行依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 删除依赖
uv remove <package-name>

# 按现有声明同步环境
uv sync
```

添加或删除依赖时，将 `pyproject.toml` 和 `uv.lock` 一起提交。不要另外手动维护 `requirements.txt`。

后续添加实际测试后，可以运行 `uv run pytest`；当前尚无测试用例。

## Git 文件约定

提交 `pyproject.toml`、`uv.lock` 和 `.python-version`。不提交 `.venv`、真实 `.env`、本地 PDF、生成的索引与缓存。`data/eval/` 中可公开的人工标注评估集可以提交。

## 源码结构

- `ingestion/`：解析、清洗、分块。
- `indexing/`：向量编码、向量索引与 BM25 索引。
- `retrieval/`：召回、RRF 融合与重排。
- `generation/`：证据提示词、回答生成与引用。
- `evaluation/`：评估数据、指标和方法对比。
- `schemas.py`、`config.py`：共享数据契约与配置。
- `pipeline.py`、`cli.py`：流程编排与未来命令行入口。
