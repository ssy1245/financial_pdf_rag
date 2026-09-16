"""向量编码适配。

状态：架构占位，尚未实现业务逻辑。

职责与输入输出：
输入：文档文本批次或查询。输出：对应向量。
封装可替换的 API 或本地模型，明确文档编码与查询编码的差异、模型版本和维度。

边界与约束：
不管理索引或排序；查询与文档必须使用兼容的编码配置，不能静默混用模型。
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
