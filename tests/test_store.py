"""Тесты на векторное хранилище с детерминированным эмбеддером."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tg_rag_bot.answerer import Source, assemble_extractive
from tg_rag_bot.store import VectorStore


class FakeEmbedder:
    """Детерминированный эмбеддер для тестов: хеш текста разбит на 32 dim."""

    dim = 32

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
            vector = [b / 255.0 for b in digest[: self.dim]]
            vectors.append(vector)
        return vectors


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedding_fn=FakeEmbedder())


def test_add_and_list(store: VectorStore) -> None:
    added = store.add_document(
        user_id=1,
        doc_id="doc-a",
        doc_title="Заметка А",
        chunks=["Первый чанк", "Второй чанк"],
    )
    assert added == 2
    docs = store.list_documents(1)
    assert docs == [("doc-a", "Заметка А", 2)]


def test_add_dedupes_repeated_chunks(store: VectorStore) -> None:
    added = store.add_document(
        user_id=1,
        doc_id="doc-a",
        doc_title="Дубликат",
        chunks=["Один и тот же чанк", "Один и тот же чанк"],
    )
    assert added == 1
    docs = store.list_documents(1)
    assert docs == [("doc-a", "Дубликат", 1)]


def test_users_isolated(store: VectorStore) -> None:
    store.add_document(user_id=1, doc_id="d1", doc_title="A", chunks=["alpha"])
    store.add_document(user_id=2, doc_id="d2", doc_title="B", chunks=["beta"])
    assert store.list_documents(1) == [("d1", "A", 1)]
    assert store.list_documents(2) == [("d2", "B", 1)]


def test_search_returns_added_chunk(store: VectorStore) -> None:
    store.add_document(
        user_id=1,
        doc_id="doc-a",
        doc_title="Заметка",
        chunks=["единственный текст"],
    )
    hits = store.search(user_id=1, query="единственный текст", top_k=3)
    assert len(hits) == 1
    chunk, title, distance = hits[0]
    assert chunk == "единственный текст"
    assert title == "Заметка"
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_search_empty_collection_returns_nothing(store: VectorStore) -> None:
    assert store.search(user_id=999, query="что-нибудь", top_k=3) == []


def test_forget_deletes_chunks(store: VectorStore) -> None:
    store.add_document(user_id=1, doc_id="doc-a", doc_title="A", chunks=["x", "y"])
    store.add_document(user_id=1, doc_id="doc-b", doc_title="B", chunks=["z"])
    removed = store.forget_document(user_id=1, doc_id="doc-a")
    assert removed == 2
    assert store.list_documents(1) == [("doc-b", "B", 1)]


def test_forget_missing_doc_returns_zero(store: VectorStore) -> None:
    assert store.forget_document(user_id=1, doc_id="nope") == 0


def test_extractive_empty_message_when_no_sources() -> None:
    answer = assemble_extractive([], "вопрос")
    assert "ничего не нашлось" in answer.text
    assert answer.used_llm is False


def test_extractive_contains_titles_and_snippets() -> None:
    sources = [Source(doc_title="A", snippet="Кусок А", distance=0.1)]
    answer = assemble_extractive(sources, "что в А?")
    assert "A" in answer.text
    assert "Кусок А" in answer.text
    assert answer.used_llm is False
