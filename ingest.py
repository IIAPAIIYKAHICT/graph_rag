"""Ingestion: документы из data/ -> чанки -> векторный индекс + property graph.

Строит два индекса поверх одной базы Neo4j:
  1. VectorStoreIndex   — чанки с эмбеддингами (для baseline vector и fusion)
  2. PropertyGraphIndex — сущности и связи, извлечённые LLM (для graph-ретриверов)

Запуск:  python ingest.py
"""
import argparse
from pathlib import Path

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)

import config

DATA_DIR = Path(__file__).parent / "data"


def load_documents():
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        raise SystemExit(
            f"Папка {DATA_DIR} пустая. Положи туда свои PDF/txt/md и запусти снова."
        )
    docs = SimpleDirectoryReader(
        str(DATA_DIR), recursive=True, exclude=["README.md", "**/README.md"]
    ).load_data()
    print(f"Загружено документов/страниц: {len(docs)}")
    return docs


def build_vector_index(docs):
    """Baseline: чанки -> эмбеддинги -> локальный векторный индекс на диске."""
    index = VectorStoreIndex.from_documents(docs)  # дефолтный SimpleVectorStore
    index.storage_context.persist(persist_dir=config.VECTOR_DIR)
    print("Векторный индекс построен и сохранён:", config.VECTOR_DIR)
    return index


def build_property_graph(docs):
    """LLM извлекает сущности и связи -> property graph в Neo4j.

    SimpleLLMPathExtractor не требует фиксированной схемы — LLM сам решает,
    какие сущности и связи выделить. Для более строгого графа замени его на
    SchemaLLMPathExtractor с явным списком entities/relations.
    """
    graph_store = config.get_graph_store()
    index = PropertyGraphIndex.from_documents(
        docs,
        property_graph_store=graph_store,
        kg_extractors=[
            SimpleLLMPathExtractor(
                llm=config.Settings.llm,
                max_paths_per_chunk=10,
            ),
            ImplicitPathExtractor(),  # связи из структуры документов
        ],
        embed_kg_nodes=True,  # эмбеддинги на узлах графа -> нужно для graph_vector
        show_progress=True,
    )
    print("Property graph построен (сущности + связи).")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["vector", "graph"],
        help="Построить только один индекс (по умолчанию оба).",
    )
    args = parser.parse_args()

    docs = load_documents()
    if args.only != "graph":
        build_vector_index(docs)
    if args.only != "vector":
        build_property_graph(docs)
    print("\nГотово. Дальше: python query.py --retriever fusion \"вопрос\"")


if __name__ == "__main__":
    main()
