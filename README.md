# graphrag-lab

Стенд для сравнения разных видов **гибридного RAG** на одном корпусе своих документов.
Стек: **Python + LlamaIndex + Neo4j** (граф и векторный индекс живут в одной базе).
LLM — **Claude (Anthropic)**, эмбеддинги — **локальная модель** (без ключа, офлайн).

Идея: собрать общий слой ingestion (чанкинг + эмбеддинги + извлечение графа),
а поверх него подключить несколько ретриверов и сравнить их ответы на одном наборе вопросов.

## Что внутри

| Файл | Назначение |
|------|-----------|
| `config.py` | Настройки, инициализация LLM/эмбеддингов и подключений к Neo4j |
| `ingest.py` | Загрузка документов из `data/`, чанкинг, векторный индекс + property graph |
| `retrievers.py` | Четыре ретривера: `vector`, `graph`, `graph_vector`, `fusion` |
| `query.py` | CLI: задать вопрос выбранным ретривером |
| `evaluate.py` | Прогнать `questions.yaml` по всем ретриверам, вывести сравнение |
| `questions.yaml` | Набор вопросов разных типов (факт / multi-hop / глобальные) |
| `data/` | Сюда кладёшь свои PDF / txt / md |

## Сравниваемые подходы

1. **`vector`** — baseline. Обычный векторный поиск по чанкам.
2. **`graph`** — обход property graph от найденных сущностей (multi-hop по связям).
3. **`graph_vector`** — граф + векторный контекст на узлах (гибрид внутри PropertyGraphIndex).
4. **`fusion`** — параллельно vector и graph, результаты объединяются через Reciprocal Rank Fusion.

## LLM: Claude вместо OpenAI

Проект переведён с OpenAI на Claude. Два важных момента:

- **Эмбеддинги на Claude невозможны.** У Anthropic нет embeddings API — Claude умеет
  только генерацию текста. Поэтому вектора (нужны для `vector`, `graph_vector`,
  `fusion`) считает **локальная** модель sentence-transformers: бесплатно, офлайн,
  без ключа. Хочешь обратно облачные эмбеддинги — можно вернуть OpenAI
  (`text-embedding-3-small`) или взять [Voyage AI](https://www.voyageai.com/)
  (партнёр Anthropic по эмбеддингам); оба требуют свой ключ.
- **LLM ходит по `ANTHROPIC_API_KEY`.** Это ключ из [Anthropic Console](https://console.anthropic.com/),
  тарифицируется по токенам.

### А «токены Claude Code» (подписка Pro/Max)?

Коротко: **напрямую так делать не стоит.** Подписка Claude Pro/Max лицензирована под
сам Claude Code и приложения Claude, а не как универсальный API для сторонних
библиотек. Технически Anthropic SDK умеет брать OAuth-токен из `ANTHROPIC_AUTH_TOKEN`
или из профиля `ant auth login` (плюс заголовок `anthropic-beta: oauth-2025-04-20`),
но гонять через подписочный токен RAG-пайплайн с тысячами вызовов — это не
предусмотренный сценарий: легко упереться в лимиты, поймать блокировку или нарушить
условия. Поэтому проект завязан на обычный `ANTHROPIC_API_KEY` — это надёжный и
поддерживаемый путь. Если хочется именно подписку — используй её в Claude Code, а не
как бэкенд LlamaIndex.

### Выбор модели

По умолчанию `LLM_MODEL=claude-opus-5` (самая мощная). Но `ingest.py` зовёт LLM
**на каждый чанк** для извлечения графа — на opus это дорого и медленно. Практично:

```bash
LLM_MODEL=claude-haiku-4-5 python ingest.py    # дёшево/быстро строим граф
LLM_MODEL=claude-opus-5   python query.py ...  # качество на ответах
```

> **Если словишь `400` про `temperature`:** новые модели Claude (Opus 5, Sonnet 5,
> Opus 4.7/4.8, Fable 5) не принимают параметр `temperature`, который старые версии
> LlamaIndex шлют по умолчанию. Лечится `pip install -U llama-index-llms-anthropic`
> либо моделью, которая его принимает (`claude-haiku-4-5`, `claude-sonnet-4-5`).

## Быстрый старт

### 1. Neo4j

Самый простой вариант — Docker (нужен плагин APOC для извлечения графа):

```bash
docker run -d --name neo4j-graphrag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.24
```

UI будет на http://localhost:7474 (логин `neo4j` / `password123`).

Альтернатива — бесплатный [Neo4j Aura](https://neo4j.com/product/auradb/) в облаке.

### 2. Окружение

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # впиши свой ANTHROPIC_API_KEY и данные Neo4j
```

При первом запуске скачается локальная модель эмбеддингов (~120 МБ) — дальше она
работает офлайн, без интернета и ключей. LLM (Claude) ходит в сеть по токенам.

### 3. Данные

Положи свои документы (PDF, txt, md, docx) в папку `data/`.

### 4. Построить индексы

```bash
python ingest.py
```

Это разобьёт документы на чанки, посчитает эмбеддинги и попросит LLM извлечь
сущности и связи — построится property graph. Можно открыть Neo4j Browser и посмотреть:

> Если раньше индекс строился на эмбеддингах OpenAI (1536 измерений), у новой
> локальной модели другая размерность (384) — старый векторный индекс несовместим.
> Очисти базу перед повторным ingest: в Neo4j Browser `MATCH (n) DETACH DELETE n`
> и удали индекс `DROP INDEX chunk_vector_index IF EXISTS`.

```cypher
MATCH (n) RETURN n LIMIT 100
```

### 5. Спросить

```bash
python query.py --retriever fusion "Твой вопрос по документам?"
```

### 6. Сравнить подходы

```bash
python evaluate.py                    # все ретриверы, параллельно, + метрики RAGAS
python evaluate.py --no-ragas         # быстро: только latency/источники
python evaluate.py --concurrency 8    # больше параллелизма (осторожно с rate limit)
```

`evaluate.py` гоняет сетку «вопрос × ретривер» асинхронно и считает
[RAGAS](https://docs.ragas.io/)-метрики (`faithfulness`, `answer_relevancy`).
Судья RAGAS — та же модель Claude и те же локальные эмбеддинги, что и в пайплайне,
так что OpenAI-ключ не нужен. Отчёт — в `results.md` / `results.json`.

## Заметки по исследованию

- Начни с маленького **связного** корпуса (10-50 документов с перекрёстными ссылками между сущностями) — так разница между подходами виднее.
- Граф выигрывает на **multi-hop** и **глобальных** вопросах; на простых фактических baseline vector часто не хуже и дешевле.
- Извлечение графа делает LLM — это самый дорогой шаг. Для больших корпусов ограничивай схему сущностей/связей (см. `SchemaLLMPathExtractor` в `ingest.py`).
- Numeric-метрики качества ([RAGAS](https://docs.ragas.io/): `faithfulness`, `answer_relevancy`) считаются прямо в `evaluate.py`. Учти: сами метрики — это доп. вызовы Claude (небыстро и не бесплатно), а RAGAS-промпты заточены под OpenAI, так что с Claude-судьёй отдельные оценки иногда приходят как `n/a` — это норма, не баг.
