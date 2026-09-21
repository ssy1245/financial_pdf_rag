"""本地 SentenceTransformers 编码适配。

已实现：默认 BAAI/bge-small-en-v1.5；文档批次和单条查询返回归一化 NumPy 向量。
首次加载可能下载模型。文档与查询均使用同一模型的 encode。
未实现：向量缓存、tokenizer 超长输入检查、模型版本固定和额外查询指令策略。
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model = SentenceTransformer(model_name)

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return np.asarray(embeddings)

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return np.asarray(embedding)
