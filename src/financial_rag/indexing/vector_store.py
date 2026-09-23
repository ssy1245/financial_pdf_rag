"""使用 SQLite 持久化 chunks 和向量；相似度计算仍由 dense.py 完成。"""

import json
import sqlite3
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def replace_all(self, chunks, embeddings, model_name):
        # 固定保存格式：小端序、32 位浮点数。
        vectors = np.asarray(embeddings, dtype="<f4")

        if not chunks:
            raise ValueError("不能保存空 chunks")

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("向量必须是二维矩阵，且行数与 chunks 数量一致")

        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("向量维度或数值无效")

        ids = [chunk["chunk_id"] for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id 不能重复")

        # 当前 dense_search 使用点积，需要单位向量。
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("当前索引要求向量已经归一化")

        metadata = {
            "model_name": model_name,
            "dimension": vectors.shape[1],
            "dtype": "<f4",
            "normalized": True,
        }

        rows = [
            (
                position,
                chunk["chunk_id"],
                json.dumps(chunk, ensure_ascii=False),
                vector.tobytes(),
            )
            for position, (chunk, vector) in enumerate(
                zip(chunks, vectors)
            )
        ]

        connection = sqlite3.connect(self.db_path)
        try:
            # 同一事务内替换：插入失败会回滚，不留下半份数据。
            with connection:
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        position INTEGER NOT NULL UNIQUE,
                        chunk_id TEXT PRIMARY KEY,
                        chunk_json TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    )
                """)
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS index_metadata (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        metadata_json TEXT NOT NULL
                    )
                """)

                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM index_metadata")

                connection.executemany(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    rows,
                )
                connection.execute(
                    "INSERT INTO index_metadata VALUES (1, ?)",
                    (json.dumps(metadata),),
                )
        finally:
            connection.close()

    def load(self, expected_model_name):
        if not self.db_path.is_file():
            raise FileNotFoundError("索引不存在，请先执行入库")

        connection = sqlite3.connect(self.db_path)
        try:
            metadata_row = connection.execute(
                "SELECT metadata_json FROM index_metadata WHERE id = 1"
            ).fetchone()

            if metadata_row is None:
                raise ValueError("索引缺少模型配置")

            metadata = json.loads(metadata_row[0])

            if metadata["model_name"] != expected_model_name:
                raise ValueError("查询模型与索引模型不同，请重建索引")

            if metadata["dtype"] != "<f4" or not metadata["normalized"]:
                raise ValueError("索引格式或归一化配置不兼容")

            rows = connection.execute("""
                SELECT chunk_json, embedding
                FROM chunks
                ORDER BY position
            """).fetchall()
        finally:
            connection.close()

        if not rows:
            raise ValueError("索引为空")

        chunks = []
        vectors = []

        for chunk_json, blob in rows:
            vector = np.frombuffer(blob, dtype="<f4").copy()

            if vector.size != metadata["dimension"]:
                raise ValueError("保存的向量维度与索引配置不一致")

            chunks.append(json.loads(chunk_json))
            vectors.append(vector)

        return chunks, np.stack(vectors)