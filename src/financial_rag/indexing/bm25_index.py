"""关键词索引。

状态：已实现内存 BM25 统计与评分，持久化待实现。
职责与输入输出：
输入：chunks 或查询文本。输出：索引及候选 chunk_id 与 BM25 分数。
负责一致的分词、索引建立、持久化、加载与搜索，保存分词策略和 chunk 映射。

边界与约束：
保留金融术语与数字的可检索性；不与向量分数直接相加。
TF:这个词在某个 chunk 中出现几次
DF:有几个 chunk 包含这个词
"""
import re
from collections import Counter
import math

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())



class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]
        # 每个 chunk 内部的词频
        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_chunks
        ]

        # 每个 chunk 的长度，单位是分词后的词数
        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_chunks
        ]

        # chunk 总数
        self.document_count = len(self.tokenized_chunks)

        # 平均 chunk 长度
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0.0
        )
        # 每个词出现在多少个 chunk 中
        self.document_frequencies = Counter()
        for tokens in self.tokenized_chunks:
            self.document_frequencies.update(set(tokens))
        self.idf = {}
        for word, df in self.document_frequencies.items():
            self.idf[word] = math.log(
                1 + (self.document_count - df + 0.5) / (df + 0.5)
            )
        self.k1 = 1.5
        self.b = 0.75

    def get_scores(self, query: str) -> list[float]:
        # 第一版对重复查询词去重，并保留顺序
        query_tokens = list(dict.fromkeys(tokenize(query)))
        scores = []
        for index, term_frequency in enumerate(self.term_frequencies):
            score = 0.0
            document_length = self.document_lengths[index]
            for word in query_tokens:
                tf = term_frequency.get(word, 0)
                # 当前 chunk 没有这个词，不贡献分数
                if tf == 0:
                    continue
                idf = self.idf[word]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                        1 - self.b
                        + self.b
                        * document_length
                        / self.average_document_length
                )
                score += idf * numerator / denominator
            scores.append(score)
        return scores
