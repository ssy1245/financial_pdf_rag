"""Cross-Encoder 重排：问题与候选正文配对评分，降序取 Top-K。

已实现：本地 MiniLM 模型、原始元数据保留、空输入处理和分数校验。
模型每个实例只加载一次；不检索、不生成答案，不将分数解释为概率。
默认问题和正文合计最多 512 tokens，记录超限候选 ID 并警告；未实现长文本分窗。
"""
import warnings
import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        max_length: int = 512,
    ):
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")
        self.model_name = model_name
        self.max_length = max_length
        self.last_truncated_chunk_ids = []
        # 初始化时加载一次模型，不要每个问题都重新加载
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        self.last_truncated_chunk_ids = []
        if top_k <= 0 or not candidates:
            return []
        # 1. 每个候选都与同一个问题配对
        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]
        # 对完整文本对计数；超限只报告，不伪称模型读取了完整正文。
        lengths = self.model.tokenizer(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            truncation=False, padding=False, return_length=True,
        )["length"]
        self.last_truncated_chunk_ids = [
            candidate["chunk_id"] for candidate, length in zip(candidates, lengths)
            if length > self.max_length
        ]
        if self.last_truncated_chunk_ids:
            warnings.warn(
                f"{len(self.last_truncated_chunk_ids)} 个重排候选超过 {self.max_length} tokens，模型输入将截断；报告中保留原始全文。",
                UserWarning, stacklevel=2,
            )
        # 2. 模型为每一对文本计算相关性分数
        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )
        scores = np.asarray(scores).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError("重排模型必须为每个候选返回一个有限分数")
        # 3. 保留原始信息，添加新分数
        results = [
            {
                **candidate,
                "rerank_score": float(score),
            }
            for candidate, score in zip(candidates, scores)
        ]
        # 4. 按重排分数排序，取前 K 条
        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )
        return results[:top_k]
