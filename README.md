# tg-rag-notes-bot

[![CI](https://github.com/xndrexe/tg-rag-notes-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/xndrexe/tg-rag-notes-bot/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Что это и кому.** Telegram-бот для тех, кто держит инструкции, регламенты и
заметки в файлах: загружаете PDF, DOCX или Markdown и спрашиваете обычными
словами, бот отвечает цитатами из ваших документов. Поиск по смыслу идёт на
локальных эмбеддингах, документы каждого пользователя хранятся отдельно.

Telegram-бот, который индексирует ваши PDF / DOCX / Markdown-документы и
отвечает на вопросы по их содержимому через retrieval-augmented generation.
Локальные эмбеддинги на CPU через sentence-transformers + локальное
векторное хранилище (SQLite + numpy, без C++-зависимостей) для поиска.
Опциональный LLM-фолбэк на OpenAI-совместимое API для генеративного
ответа — без ключа бот работает в extractive-режиме (цитаты).

Минимальный жизнеспособный продукт: один процесс, одна команда `docker compose up`,
никаких внешних сервисов кроме Telegram. Чистая архитектура, 22 теста,
структурированное логирование через structlog, graceful shutdown по
SIGTERM, открытая лицензия.

## Что умеет

- Принимает PDF, DOCX, Markdown, txt
- Извлекает текст и режет на перекрывающиеся чанки
- Сохраняет векторы в локальное хранилище на SQLite + numpy, у каждого пользователя свои документы
- Отвечает на текстовые вопросы релевантными цитатами (extractive) или
  через LLM (с `OPENAI_API_KEY`)
- `/list` — посмотреть загруженные документы
- `/forget <id>` — забыть документ

## Запуск

### Через Docker

```bash
cp .env.example .env
# отредактировать .env: вставить BOT_TOKEN
docker compose up -d
docker compose logs -f
```

### Локально

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -e ".[dev]"
cp .env.example .env
# отредактировать .env
tg-rag-bot
```

## Стек

- Python 3.11+
- aiogram 3 (Telegram Bot API)
- SQLite + numpy для векторного хранилища (brute-force cosine, без внешних
  векторных БД — ноль C++-зависимостей, кросс-платформенно)
- sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2` — работает на
  русском и английском, ~120 МБ, ~30 МБ RAM)
- pypdf + python-docx (извлечение текста)
- httpx (OpenAI клиент, опционально)
- pydantic + pydantic-settings (конфиг)
- pytest (тесты)

## Тесты

```bash
pytest -q
```

22 теста, ~3 секунды на полный прогон, никаких сетевых вызовов.
Покрывают:
- Извлечение текста из всех поддерживаемых форматов
- Чанкование (короткие/длинные/перекрытия)
- Векторное хранилище (add / search / forget / изоляция между пользователями)
- Extractive-ассемблер ответа
- LLM-ассемблер через `httpx.MockTransport`: успех, 5xx-ошибка, network
  error, malformed payload, пустые источники

## Качество кода

- **CI:** GitHub Actions matrix Linux/Windows × Python 3.11/3.12 +
  отдельный mypy-проход (см. `.github/workflows/ci.yml`).
- **Pre-commit:** хуки на ruff + базовые проверки в
  `.pre-commit-config.yaml`. Активация: `pre-commit install`.
- **Логирование:** structlog с двумя выводами — `LOG_FORMAT=console`
  (dev) и `LOG_FORMAT=json` (prod, легко скрэпится в Loki/CloudWatch).
- **Graceful shutdown:** SIGINT/SIGTERM прерывают polling через
  `asyncio.Event`, бот аккуратно закрывает сессию HTTP-клиента и
  завершается без зависших background-task'ов.

## Архитектура

См. [docs/architecture.md](docs/architecture.md).

## Лицензия

MIT.

## Контекст

Это публичная демонстрация моего стека для работы с RAG / Telegram-ботами /
embeddings. Боевые системы (8+ месяцев в проде, мультиканальная аналитика
с собственным OAuth, мониторинг лимитов с алертами в Telegram) идут под NDA —
подробнее расскажу по запросу.
