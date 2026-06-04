"""Точка входа: запуск бота через polling."""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties

from .bot import build_dispatcher
from .config import Settings
from .embeddings import LocalEmbedder
from .store import VectorStore


async def _run() -> None:
    settings = Settings()
    settings.ensure_dirs()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    embedder = LocalEmbedder(settings.embedding_model)
    store = VectorStore(settings.store_dir, embedding_fn=embedder)
    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    dispatcher = build_dispatcher(settings, store)
    logging.info("starting polling")
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:  # pragma: no cover
    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
