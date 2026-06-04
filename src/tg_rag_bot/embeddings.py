"""Локальные эмбеддинги через sentence-transformers."""
from __future__ import annotations

import threading
from typing import Any


class LocalEmbedder:
    """Тонкая обёртка над sentence-transformers с ленивой инициализацией.

    Модель грузится при первом вызове — это убирает 5-10 секунд из старта
    бота, если пользователь сейчас ничего не индексирует.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
        return self._model

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        result = model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in result]
