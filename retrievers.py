"""Четыре ретривера для сравнения подходов гибридного RAG.

  vector        — baseline, чистый векторный поиск по чанкам
  graph         — обход property graph от узлов, найденных по эмбеддингам (multi-hop)
  graph_vector  — граф (вектор-вход) + keyword-линкинг сущностей (гибрид в PG-индексе)
  fusion        — vector + graph параллельно, результаты объединяются через RRF

Все ретриверы подключаются к уже построенным в Neo4j индексам (см. ingest.py).
"""
from llama_index.core import (
    PropertyGraphIndex,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.core.retrievers import QueryFusionRetriever

import config

TOP_K = 5


def _vector_index() -> VectorStoreIndex:
    sc = StorageContext.from_defaults(persist_dir=config.VECTOR_DIR)
    return load_index_from_storage(sc)


def _pg_index() -> PropertyGraphIndex:
    return PropertyGraphIndex.from_existing(
        property_graph_store=config.get_graph_store(),
        embed_kg_nodes=True,
    )


def build_vector_retriever():
    return _vector_index().as_retriever(similarity_top_k=TOP_K)


# Keyword entity-linking does an exact id match (after .capitalize()), which is
# reliable for code-named entities like P07 / T12. For natural-language names,
# graph_vector adds embedding-based node entry (VectorContextRetriever) as a fallback.
SYNONYM_PROMPT = (
    "From the query, extract the key named entities (people, teams, projects, "
    "technologies) exactly as they appear. Copy code identifiers like P07 or T12 "
    "verbatim, as separate variants. Give up to {max_keywords} variants on one "
    "line separated by '^': 'entity1^entity2^...'.\n"
    "QUERY: {query_str}\nKEYWORDS: "
)


def _vector_ctx(pg, path_depth=2):
    """Вход в граф по эмбеддингам узлов + обход на path_depth хопов."""
    return VectorContextRetriever(
        pg.property_graph_store,
        embed_model=config.Settings.embed_model,
        include_text=True,
        similarity_top_k=TOP_K,
        path_depth=path_depth,
    )


def _synonym(pg, path_depth=3):
    return LLMSynonymRetriever(
        pg.property_graph_store,
        llm=config.Settings.llm,
        include_text=True,
        max_keywords=10,
        path_depth=path_depth,  # глубина обхода от найденной сущности (multi-hop)
        synonym_prompt=SYNONYM_PROMPT,
    )


def build_graph_retriever():
    """Граф: keyword-линкинг сущностей (точный матч id — надёжен для коротких
    кодов типа P19/T04, где эмбеддинги узла слабые) + multi-hop обход по связям."""
    pg = _pg_index()
    return pg.as_retriever(sub_retrievers=[_synonym(pg, path_depth=3)])


def build_graph_vector_retriever():
    """Гибрид в графе: keyword-линкинг + вход по эмбеддингам узлов (помогает для
    сущностей с естественными именами и нечётких совпадений)."""
    pg = _pg_index()
    return pg.as_retriever(sub_retrievers=[_synonym(pg, path_depth=3), _vector_ctx(pg, path_depth=3)])


def build_fusion_retriever():
    """Vector + graph_vector параллельно, объединение через Reciprocal Rank Fusion."""
    return QueryFusionRetriever(
        [build_vector_retriever(), build_graph_vector_retriever()],
        similarity_top_k=TOP_K,
        num_queries=1,  # >1 включит query expansion (генерацию доп. запросов)
        mode="reciprocal_rerank",
        use_async=False,
        llm=config.Settings.llm,
    )


BUILDERS = {
    "vector": build_vector_retriever,
    "graph": build_graph_retriever,
    "graph_vector": build_graph_vector_retriever,
    "fusion": build_fusion_retriever,
}


def get_retriever(name: str):
    if name not in BUILDERS:
        raise ValueError(f"Неизвестный ретривер '{name}'. Доступно: {list(BUILDERS)}")
    return BUILDERS[name]()
