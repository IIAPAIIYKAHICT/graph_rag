"""Инкрементально заинджестить папку документов в УЖЕ построенные индексы.

Не перестраивает всё заново: добавляет новые документы в существующий
property graph (Neo4j) и в векторный индекс на диске. Удобно для live-демо —
показать, как добавление знаний сразу появляется в графе.

Запуск:  python ingest_docs.py            # по умолчанию ./demo_docs
         python ingest_docs.py path/to/dir
"""
import sys
from pathlib import Path

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)

import config


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "demo_docs")
    docs = SimpleDirectoryReader(
        path, recursive=True, exclude=["README.md", "**/README.md"]
    ).load_data()
    print(f"New documents: {len(docs)}  (from {path})")

    # 1) Векторный индекс (SimpleVectorStore на диске) — догружаем и дописываем.
    sc = StorageContext.from_defaults(persist_dir=config.VECTOR_DIR)
    vindex = load_index_from_storage(sc)
    for d in docs:
        vindex.insert(d)
    vindex.storage_context.persist(persist_dir=config.VECTOR_DIR)
    print("Vector index updated.")

    # 2) Property graph (Neo4j) — извлекаем сущности/связи из новых документов.
    pg = PropertyGraphIndex.from_existing(
        property_graph_store=config.get_graph_store(),
        embed_kg_nodes=True,
        kg_extractors=[
            SimpleLLMPathExtractor(llm=config.Settings.llm, max_paths_per_chunk=10),
            ImplicitPathExtractor(),
        ],
    )
    for d in docs:
        pg.insert(d)
    print("Property graph updated (new entities and relations).")
    print('\nCheck:  python query.py -r graph "Which projects does project P99 depend on?"')


if __name__ == "__main__":
    main()
