"""Хендлеры Telegram-бота на aiogram 3."""
from __future__ import annotations

import html
import uuid
from pathlib import Path

import structlog
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Document, Message

from .answerer import Source, assemble_extractive, assemble_llm
from .config import Settings
from .extractors import ExtractError, chunk_text, extract_text
from .store import VectorStore

LOG = structlog.get_logger("tg_rag_bot")

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".md", ".markdown", ".txt")


def indexed_message(added: int, file_name: str, doc_id: str) -> str:
    """Ответ после индексации. Уходит с parse_mode=HTML, поэтому имя файла экранируется."""
    if added % 10 == 1 and added % 100 != 11:
        word = "фрагмент"
    elif added % 10 in (2, 3, 4) and added % 100 not in (12, 13, 14):
        word = "фрагмента"
    else:
        word = "фрагментов"
    return (
        f"Добавил {added} {word} из {html.escape(file_name)}. "
        f"ID документа: <code>{doc_id}</code>"
    )


def build_dispatcher(settings: Settings, store: VectorStore) -> Dispatcher:
    dispatcher = Dispatcher()
    router = Router(name="rag")

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(
            "Я — RAG-бот по вашим документам.\n\n"
            "Пришлите PDF / DOCX / Markdown — я их индексирую. "
            "Дальше задайте вопрос текстом — отвечу со ссылкой на источник.\n\n"
            "/list — что у меня уже есть\n"
            "/forget <id> — забыть один документ\n"
            "/help — справка."
        )

    @router.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(
            "Команды:\n"
            "/list — список загруженных документов\n"
            "/forget <id> — удалить документ\n\n"
            "Просто пришлите файл — я добавлю его в индекс.\n"
            f"Поддерживаемые форматы: {', '.join(SUPPORTED_EXTENSIONS)}.\n"
            f"Максимальный размер: {settings.max_file_mb} МБ."
        )

    @router.message(Command("list"))
    async def on_list(message: Message) -> None:
        assert message.from_user is not None
        documents = store.list_documents(message.from_user.id)
        if not documents:
            await message.answer("У вас пока нет документов. Пришлите файл.")
            return
        lines = ["Ваши документы:\n"]
        for doc_id, title, chunks in documents:
            lines.append(f"• <code>{doc_id}</code> — {title} ({chunks} чанков)")
        await message.answer("\n".join(lines), parse_mode="HTML")

    @router.message(Command("forget"))
    async def on_forget(message: Message) -> None:
        assert message.from_user is not None
        if not message.text:
            return
        parts = message.text.strip().split(maxsplit=1)
        if len(parts) < 2:
            await message.answer("Использование: /forget <id>. ID из /list.")
            return
        doc_id = parts[1].strip()
        removed = store.forget_document(user_id=message.from_user.id, doc_id=doc_id)
        if removed:
            await message.answer(f"Удалено {removed} чанков документа {doc_id}.")
        else:
            await message.answer(f"Документа с id {doc_id} нет.")

    @router.message(F.document)
    async def on_document(message: Message, bot: Bot) -> None:
        assert message.from_user is not None
        assert message.document is not None
        document: Document = message.document
        await _handle_upload(message, bot, document, settings, store)

    @router.message(F.text)
    async def on_question(message: Message) -> None:
        assert message.from_user is not None
        if not message.text:
            return
        query = message.text.strip()
        if not query:
            return
        hits = store.search(user_id=message.from_user.id, query=query, top_k=settings.top_k)
        sources = [Source(doc_title=title, snippet=chunk, distance=dist) for chunk, title, dist in hits]
        if settings.openai_api_key:
            answer = await assemble_llm(
                sources,
                query,
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
        else:
            answer = assemble_extractive(sources, query)
        await message.answer(answer.text)

    dispatcher.include_router(router)
    return dispatcher


async def _handle_upload(
    message: Message,
    bot: Bot,
    document: Document,
    settings: Settings,
    store: VectorStore,
) -> None:
    file_name = document.file_name or "document"
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        await message.answer(
            f"Формат {suffix} не поддерживается. Доступно: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
        return
    if document.file_size and document.file_size > settings.max_file_mb * 1024 * 1024:
        await message.answer(f"Файл больше {settings.max_file_mb} МБ.")
        return
    settings.ensure_dirs()
    doc_id = uuid.uuid4().hex[:10]
    storage_path = settings.storage_dir / f"{doc_id}{suffix}"
    file = await bot.get_file(document.file_id)
    if file.file_path is None:
        await message.answer("Не удалось получить файл — попробуйте ещё раз.")
        return
    await bot.download_file(file.file_path, destination=storage_path)

    try:
        text = extract_text(storage_path)
        chunks = chunk_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except ExtractError as exc:
        storage_path.unlink(missing_ok=True)
        await message.answer(f"Не получилось извлечь текст: {exc}")
        return
    assert message.from_user is not None
    added = store.add_document(
        user_id=message.from_user.id,
        doc_id=doc_id,
        doc_title=file_name,
        chunks=chunks,
    )
    LOG.info("indexed", user_id=message.from_user.id, doc_id=doc_id, chunks=added)
    await message.answer(
        indexed_message(added, file_name, doc_id),
        parse_mode="HTML",
    )
