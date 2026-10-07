"""Общие настройки: LLM, эмбеддинги, подключения к Neo4j.

Импортируй из этого модуля во всех остальных скриптах, чтобы держать
конфигурацию в одном месте.

Что изменилось при переезде с OpenAI:
  - LLM теперь Claude (Anthropic), ключ из ANTHROPIC_API_KEY.
  - Эмбеддинги — локальная модель (у Claude нет embeddings API, поэтому вектора
    считаем офлайн, без всякого ключа; модель скачается один раз при первом запуске).
"""
import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

load_dotenv()

# --- Модели ---
# LLM — Claude. По умолчанию самая мощная модель. ВНИМАНИЕ: шаг ingest.py зовёт
# LLM на каждый чанк (извлечение графа) — это дорого и медленно на opus. Для
# ingest практично поставить LLM_MODEL=claude-haiku-4-5 (дешёвый быстрый воркхорс,
# аналог прежнего gpt-4o-mini), а для качества ответов вернуть opus/sonnet.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5")

# Эмбеддинги — локальная модель sentence-transformers (без ключа, офлайн).
# Мультиязычная, 384 измерения — норм для русских/смешанных корпусов.
# Для чисто английского корпуса чуть точнее BAAI/bge-small-en-v1.5 (тоже 384).
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 -> 384

# Чанки-вектора храним в локальном SimpleVectorStore на диске (встроен в
# llama-index-core). Пакет llama-index-vector-stores-neo4j не поддерживает
# Python 3.13, поэтому baseline/fusion берут вектора отсюда, а граф — из Neo4j.
VECTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "vector")

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Общие для всего проекта LLM и эмбеддинги (LlamaIndex Settings — глобальные).
# Ключ Claude берётся из окружения (ANTHROPIC_API_KEY) автоматически.
Settings.llm = Anthropic(model=LLM_MODEL, max_tokens=4096)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.chunk_size = 512
Settings.chunk_overlap = 64


def get_graph_store() -> Neo4jPropertyGraphStore:
    """Property graph store — узлы-сущности и связи между ними."""
    return Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database=NEO4J_DATABASE,
    )


# Векторный индекс по чанкам теперь живёт на диске (SimpleVectorStore): строится
# в ingest.py и грузится в retrievers.py через config.VECTOR_DIR.
