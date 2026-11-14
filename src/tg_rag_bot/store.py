"""Векторное хранилище на SQLite + numpy.

Для личного бота (тысячи-десятки тысяч чанков) брute-force cosine поверх
loaded-в-памяти-вектора более чем достаточен. Меньше зависимостей,
кросс-платформенно, без C++-сборки на Windows.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path
from typing import Protocol

import numpy as np


class EmbeddingFn(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        ...


class VectorStore:
    """Per-user изоляция через колонку `user_id`. Векторы хранятся как
    BLOB (float32, little-endian). Поиск — brute force cosine в памяти.
    """

    def __init__(self, base_dir: Path, embedding_fn: EmbeddingFn):
        base_dir.mkdir(parents=True, exist_ok=True)
        self._path = base_dir / "store.sqlite3"
        self._embed = embedding_fn
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                user_id      INTEGER NOT NULL,
                doc_id       TEXT NOT NULL,
                doc_title    TEXT NOT NULL,
                text         TEXT NOT NULL,
                embedding    BLOB NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user ON chunks(user_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_doc ON chunks(user_id, doc_id)"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_hash "
            "ON chunks(user_id, content_hash)"
        )
        self._conn.commit()

    def add_document(
        self,
        *,
        user_id: int,
        doc_id: str,
        doc_title: str,
        chunks: list[str],
    ) -> int:
        if not chunks:
            return 0
        unique_chunks: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            chash = _hash(chunk)
            if chash in seen:
                continue
            seen.add(chash)
            unique_chunks.append(chunk)
        if not unique_chunks:
            return 0
        vectors = self._embed(unique_chunks)
        with self._conn:
            inserted = 0
            for idx, (chunk, vec) in enumerate(zip(unique_chunks, vectors, strict=True)):
                chash = _hash(chunk)
                chunk_id = f"{doc_id}:{idx}:{chash[:10]}"
                blob = _encode_vector(vec)
                try:
                    self._conn.execute(
                        "INSERT INTO chunks "
                        "(chunk_id, user_id, doc_id, doc_title, text, embedding, content_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (chunk_id, user_id, doc_id, doc_title, chunk, blob, chash),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
        return inserted

    def search(
        self,
        *,
        user_id: int,
        query: str,
        top_k: int,
    ) -> list[tuple[str, str, float]]:
        rows = self._conn.execute(
            "SELECT doc_title, text, embedding FROM chunks WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        if not rows:
            return []
        query_vec = np.asarray(self._embed([query])[0], dtype=np.float32)
        query_vec = _normalize(query_vec)
        matrix = np.stack([_decode_vector(row["embedding"]) for row in rows])
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        similarities = matrix @ query_vec
        distances = 1.0 - similarities
        order = np.argsort(distances)[:top_k]
        out: list[tuple[str, str, float]] = []
        for idx in order:
            row = rows[int(idx)]
            out.append((row["text"], row["doc_title"], float(distances[int(idx)])))
        return out

    def list_documents(self, user_id: int) -> list[tuple[str, str, int]]:
        rows = self._conn.execute(
            "SELECT doc_id, doc_title, COUNT(*) AS n "
            "FROM chunks WHERE user_id = ? "
            "GROUP BY doc_id, doc_title ORDER BY doc_id",
            (user_id,),
        ).fetchall()
        return [(row["doc_id"], row["doc_title"], row["n"]) for row in rows]

    def forget_document(self, *, user_id: int, doc_id: str) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM chunks WHERE user_id = ? AND doc_id = ?",
                (user_id, doc_id),
            )
            return cursor.rowcount or 0

    def close(self) -> None:
        self._conn.close()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _encode_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / float(norm)
