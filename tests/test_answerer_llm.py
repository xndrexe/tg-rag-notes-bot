"""Тесты на ``assemble_llm`` с моком httpx — без сетевых вызовов.

Главная цель: убедиться, что extractive-fallback срабатывает на любой
ошибке транспортного уровня и невалидном payload. Кейс «модель ответила
успешно» проверяется отдельно.
"""
from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from tg_rag_bot.answerer import Source, assemble_llm

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _make_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    """Создать ASGI-подобный мок-транспорт из обработчика-функции."""

    return httpx.MockTransport(handler)


@pytest.fixture
def sources() -> list[Source]:
    return [
        Source(doc_title="A.md", snippet="Кусок про А", distance=0.1),
        Source(doc_title="B.md", snippet="Кусок про Б", distance=0.2),
    ]


@pytest.mark.asyncio
async def test_assemble_llm_success(
    monkeypatch: pytest.MonkeyPatch, sources: list[Source]
) -> None:
    """Модель вернула валидный ответ — текст и used_llm=True."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == OPENAI_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [
                    {"message": {"role": "assistant", "content": "Ответ от модели."}}
                ],
            },
        )

    _patch_async_client(monkeypatch, _make_transport(handler))

    answer = await assemble_llm(
        sources, "вопрос", api_key="test-key", model="gpt-test"
    )
    assert answer.used_llm is True
    assert answer.text == "Ответ от модели."
    assert answer.sources == sources


@pytest.mark.asyncio
async def test_assemble_llm_falls_back_on_5xx(
    monkeypatch: pytest.MonkeyPatch, sources: list[Source]
) -> None:
    """500 от API — отвечаем extractive-цитатами."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server down"}})

    _patch_async_client(monkeypatch, _make_transport(handler))

    answer = await assemble_llm(
        sources, "вопрос", api_key="test-key", model="gpt-test"
    )
    assert answer.used_llm is False
    assert "A.md" in answer.text and "B.md" in answer.text


@pytest.mark.asyncio
async def test_assemble_llm_falls_back_on_network_error(
    monkeypatch: pytest.MonkeyPatch, sources: list[Source]
) -> None:
    """Транспорт упал (DNS / timeout / reset) — extractive."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    _patch_async_client(monkeypatch, _make_transport(handler))

    answer = await assemble_llm(
        sources, "вопрос", api_key="test-key", model="gpt-test"
    )
    assert answer.used_llm is False


@pytest.mark.asyncio
async def test_assemble_llm_falls_back_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch, sources: list[Source]
) -> None:
    """200, но в JSON нет choices — extractive, не падаем в KeyError наружу."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "weird", "no_choices": "here"})

    _patch_async_client(monkeypatch, _make_transport(handler))

    answer = await assemble_llm(
        sources, "вопрос", api_key="test-key", model="gpt-test"
    )
    assert answer.used_llm is False


@pytest.mark.asyncio
async def test_assemble_llm_empty_sources_short_circuits() -> None:
    """Нет источников — не дёргаем сеть вообще, возвращаем «ничего не нашлось»."""

    answer = await assemble_llm(
        [], "вопрос", api_key="test-key", model="gpt-test"
    )
    assert answer.used_llm is False
    assert "ничего не нашлось" in answer.text


def _patch_async_client(
    monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport
) -> None:
    """Подменить AsyncClient так, чтобы он использовал мок-транспорт.

    Дёргаем за конструктор: единственный аргумент ``timeout`` — пробрасываем,
    остальное игнорируем.
    """

    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
