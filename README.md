# tg-rag-notes-bot

Telegram-бот, который индексирует ваши PDF / DOCX / Markdown-документы и
отвечает на вопросы по их содержимому через retrieval-augmented generation.
Локальные эмбеддинги на CPU через sentence-transformers + persistent Chroma
для поиска. Опциональный LLM-фолбэк на OpenAI-совместимое API для
генеративного ответа.

Минимальный жизнеспособный продукт: один процесс, одна команда `docker compose up`,
никаких внешних сервисов кроме Telegram. Чистая архитектура, тесты,
открытая лицензия.

## Что умеет

- Принимает PDF, DOCX, Markdown, txt
- Извлекает текст и режет на перекрывающиеся чанки
- Сохраняет векторы в локальную Chroma (per-user collection)
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

Покрывают:
- Извлечение текста из всех поддерживаемых форматов
- Чанкование (короткие/длинные/перекрытия)
- Векторное хранилище (add / search / forget / изоляция между пользователями)
- Extractive-ассемблер ответа

## Архитектура

См. [docs/architecture.md](docs/architecture.md).

## Лицензия

MIT.

## Контекст

Это публичная демонстрация моего стека для работы с RAG / Telegram-ботами /
embeddings. Боевые системы (8+ месяцев в проде, мультиканальная аналитика
с собственным OAuth, мониторинг лимитов с алертами в Telegram) идут под NDA —
архитектуру разберу на созвоне по запросу.

Контакт: [skostuhov@gmail.com](mailto:skostuhov@gmail.com) /
[GitHub @xndrexe](https://github.com/xndrexe).
