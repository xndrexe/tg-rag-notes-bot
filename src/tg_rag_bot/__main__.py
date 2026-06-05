"""Точка входа: запуск бота через polling с graceful shutdown."""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal

import structlog
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties

from .bot import build_dispatcher
from .config import Settings
from .embeddings import LocalEmbedder
from .logging_setup import configure_logging
from .store import VectorStore

LOG = structlog.get_logger("tg_rag_bot.main")


async def _run() -> None:
    settings = Settings()
    settings.ensure_dirs()

    log_fmt = os.getenv("LOG_FORMAT", "console")
    if log_fmt not in {"console", "json"}:
        log_fmt = "console"
    configure_logging(level=settings.log_level, fmt=log_fmt)  # type: ignore[arg-type]

    LOG.info("starting", embedding_model=settings.embedding_model, top_k=settings.top_k)
    embedder = LocalEmbedder(settings.embedding_model)
    store = VectorStore(settings.store_dir, embedding_fn=embedder)
    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    dispatcher = build_dispatcher(settings, store)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        LOG.info("stop signal received")
        stop_event.set()

    # signal.SIGTERM есть только на POSIX. На Windows polling завершится по
    # Ctrl+C через KeyboardInterrupt; здесь регистрируем там, где можем.
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        # Windows ProactorEventLoop не умеет add_signal_handler — это нормально.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_stop)

    polling_task = asyncio.create_task(dispatcher.start_polling(bot))

    try:
        done, _ = await asyncio.wait(
            [polling_task, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if polling_task in done:
            # polling вернулся сам — посмотрим, не было ли исключения.
            polling_task.result()
    finally:
        LOG.info("shutting down")
        polling_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await polling_task
        await bot.session.close()
        LOG.info("stopped")


def main() -> None:  # pragma: no cover
    # Windows fallback — Ctrl+C во время polling прилетает как KeyboardInterrupt.
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
