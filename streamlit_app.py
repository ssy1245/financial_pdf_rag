"""本地 PDF 问答网页：会话独立索引，复用模型，调用现有 FinancialRAG。"""
import hashlib
import os
import tempfile
import threading
from pathlib import Path

import streamlit as st

from financial_rag.ingestion.parser import parse_pdf
from financial_rag.ingestion.normalizer import normalize_pages
from financial_rag.ingestion.chunker import chunk_pages
from financial_rag.indexing.vector_store import VectorStore
from financial_rag.pipeline import FinancialRAG

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@st.cache_resource
def get_models():
    from financial_rag.indexing.embeddings import EmbeddingModel
    from financial_rag.retrieval.reranker import Reranker
    # 重排实例有上次截断信息；锁避免不同会话并发覆盖。
    return EmbeddingModel(EMBEDDING_MODEL), Reranker("BAAI/bge-reranker-v2-m3", 2048), threading.Lock()


def prepare_pdf(data, workspace, embedding):
    """上传文件名不作为磁盘路径；索引只属于当前会话。"""
    pdf_path = Path(workspace) / "document.pdf"
    pdf_path.write_bytes(data)
    pages = parse_pdf(pdf_path)
    if not pages or not any(page["text"].strip() for page in pages):
        raise ValueError("未提取到文字。请上传可选中文字的 PDF；扫描件暂不支持。")
    chunks = chunk_pages(normalize_pages(pages), document_id=hashlib.sha256(data).hexdigest(), chunk_size=500, overlap=80)
    if not chunks:
        raise ValueError("文档没有可用于问答的文字。")
    store = VectorStore(Path(workspace) / "vectors.sqlite3")
    store.replace_all(chunks, embedding.encode_documents([c["text"] for c in chunks]), EMBEDDING_MODEL)
    return store, len(pages), sum(not page["text"].strip() for page in pages)


def show_answer(result):
    st.markdown(result["answer"])
    if result["unknown_citations"]:
        st.warning("回答包含无法对应的引用，请核对原文：" + ", ".join(result["unknown_citations"]))
    elif not result["has_citations"]:
        st.caption("这条回答没有引用，可能是证据不足；请核对原文。")
    evidence = {c["chunk_id"]: c for c in result["retrieval"]["reranked"]}
    for source in result["sources"]:
        with st.expander(f"[{source['label']}] PDF 第 {source['page']} 页 · 查看原文"):
            st.text(evidence[source["chunk_id"]]["text"])


def main():
    st.set_page_config(page_title="阅财 · PDF 问答", page_icon="📄", layout="wide")
    st.markdown("""<style>
    .stApp {background: #f7f8fa;}
    .block-container {max-width: 1120px; padding-top: 3rem;}
    h1 {letter-spacing: -0.04em;}
    [data-testid="stChatMessage"] {background: white; border: 1px solid #e8eaef; border-radius: 14px;}
    </style>""", unsafe_allow_html=True)
    st.caption("阅财  /  DOCUMENT ASSISTANT")
    st.title("读懂你的 PDF")
    st.write("上传文档，用一个问题开始。回答附带原文与页码，方便你随时核对。")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("document_hash", None)
    st.session_state.setdefault("rag", None)
    with st.sidebar:
        st.subheader("你的文档")
        uploaded = st.file_uploader("拖拽 PDF 到这里，或点击选择", type=["pdf"])
        st.caption("一次阅读一份 PDF，最大 20 MB。暂不支持扫描件识别。")
        st.caption("提问时，问题和检索到的文档片段会发送给 DeepSeek。")
        if st.button("清空问答", use_container_width=True):
            st.session_state.history = []

    data = uploaded.getvalue() if uploaded else None
    digest = hashlib.sha256(data).hexdigest() if data else None
    if digest != st.session_state.document_hash:
        # 换文件或移除文件时，立即丢弃旧问答，避免答错文档。
        st.session_state.rag = None
        st.session_state.history = []
        previous = st.session_state.pop("workspace", None)
        if previous:
            previous.cleanup()
        st.session_state.document_hash = digest

    valid_upload = data is not None and len(data) <= 20 * 1024 * 1024
    if data is not None and not valid_upload:
        st.error("文件超过 20 MB，请上传较小的 PDF。")
    key_ready = bool(os.getenv("DEEPSEEK_API_KEY"))
    if not key_ready:
        st.info("服务尚未配置 DeepSeek 密钥，请使用项目 .env 启动应用。")

    if valid_upload and st.session_state.rag is None:
        if st.button("开始阅读", type="primary", disabled=not key_ready):
            workspace = tempfile.TemporaryDirectory(prefix="financial_rag_")
            try:
                with st.spinner("正在阅读文档，首次使用需要加载模型…"):
                    embedding, reranker, lock = get_models()
                    from financial_rag.generation.generator import DeepSeekGenerator
                    with lock:
                        store, page_count, empty_pages = prepare_pdf(data, workspace.name, embedding)
                        rag = FinancialRAG(store, embedding, EMBEDDING_MODEL, reranker, DeepSeekGenerator())
                    st.session_state.rag = rag
                    st.session_state.workspace = workspace
                    st.session_state.page_count = page_count
                    st.session_state.empty_pages = empty_pages
            except ValueError as error:
                workspace.cleanup()
                st.error(str(error))
            except Exception:
                workspace.cleanup()
                st.error("文档处理失败。请检查 PDF 是否损坏或加密，以及模型是否可用，然后重试。")

    rag = st.session_state.rag
    if rag:
        st.success(f"已准备好 · {uploaded.name} · {st.session_state.page_count} 页")
        if st.session_state.empty_pages:
            st.warning(f"有 {st.session_state.empty_pages} 页未提取到文字，回答可能不覆盖这些页面。")
    elif not uploaded:
        st.info("← 上传一份 PDF，开始阅读。")
    st.caption("每个问题独立检索，请写明年份、对象和指标；历史问答仅用于展示。当前检索模型更适合英文文档。")
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            show_answer(item)
    question = st.chat_input("例如：2025 财年的总销售额是多少？", disabled=rag is None or not key_ready)
    if question and question.strip():
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("正在查找证据并整理回答…"):
                    _, _, lock = get_models()
                    with lock:
                        result = rag.ask(question)
                st.session_state.history.append(result)
                show_answer(result)
            except Exception:
                st.error("本次回答失败，请检查网络、DeepSeek 配置或账户额度，然后重新提问。")


if __name__ == "__main__":
    main()
