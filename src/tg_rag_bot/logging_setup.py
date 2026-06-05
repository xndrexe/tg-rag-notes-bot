"""Настройка структурированного логирования.

structlog — структурированные ключевые поля вместо текстовых сообщений.
В разработке выводит human-readable формат с цветами, в продакшене —
JSON в stdout (легко скрэпить в Loki/CloudWatch/ELK).
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(
    *,
    level: str = "INFO",
    fmt: Literal["console", "json"] = "console",
) -> None:
    """Настроить и stdlib logging, и structlog в один поток.

    Зачем оба: aiogram, sqlalchemy и прочие сторонние пишут через stdlib —
    их мы тоже хотим видеть в едином формате. Bridge переводит stdlib
    LogRecord в structlog-events.
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Шумные библиотеки приглушаем — нам не нужны HTTP-debug-логи каждого
    # запроса к Telegram API.
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("aiogram.dispatcher").setLevel(logging.WARNING)
