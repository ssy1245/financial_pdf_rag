"""手写内存 BM25 统计与评分。

已实现：小写化及 [a-z0-9]+ 分词，Counter 统计 TF 和 DF，记录文档长度与平均长度。
IDF=ln(1+(N-df+0.5)/(df+0.5))，k1=1.5、b=0.75；查询词去重。
get_scores 返回与输入 chunks 顺序一致的分数列表；排序由 sparse.py 负责。
限制：无持久化，R&D 和含逗号数字会被拆分，不支持中文分词。
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
