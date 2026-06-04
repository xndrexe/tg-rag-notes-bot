"""Сборка ответа на вопрос: подбор контекста + опциональный LLM."""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(slots=True)
class Source:
    doc_title: str
    snippet: str
    distance: float


@dataclass(slots=True)
class Answer:
    text: str
    sources: list[Source]
    used_llm: bool


def assemble_extractive(sources: list[Source], query: str) -> Answer:
    """Без LLM: возвращает релевантные куски как есть.

    Используется когда `OPENAI_API_KEY` не задан или модель временно
    недоступна — пользователь хотя бы получит цитаты для самостоятельного
    разбора.
    """
    if not sources:
        return Answer(
            text="По вашему запросу ничего не нашлось. Загрузите документы командой /upload.",
            sources=[],
            used_llm=False,
        )
    head = "Релевантные фрагменты:\n\n"
    body = "\n\n".join(
        f"— {src.doc_title}:\n{src.snippet[:400].strip()}" for src in sources
    )
    note = f"\n\n(Вопрос: «{query}». LLM не подключена — показываю цитаты.)"
    return Answer(text=head + body + note, sources=sources, used_llm=False)


async def assemble_llm(
    sources: list[Source],
    query: str,
    *,
    api_key: str,
    model: str,
    timeout: float = 30.0,
) -> Answer:
    """С LLM (OpenAI-совместимый chat completions). При ошибке падает в
    extractive-режим — пользователь не остаётся без ответа.
    """
    if not sources:
        return Answer(
            text="По вашему запросу ничего не нашлось. Загрузите документы командой /upload.",
            sources=[],
            used_llm=False,
        )

    context = "\n\n".join(f"[{src.doc_title}]\n{src.snippet}" for src in sources)
    system = (
        "Ты ассистент по личным заметкам пользователя. "
        "Отвечай только на основе предоставленных фрагментов. "
        "Если ответа нет в контексте — честно скажи об этом. "
        "Кратко цитируй название источника в скобках."
    )
    user = f"Контекст:\n{context}\n\nВопрос: {query}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            return Answer(text=text, sources=sources, used_llm=True)
    except (httpx.HTTPError, KeyError, IndexError):
        return assemble_extractive(sources, query)
