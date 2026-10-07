"""CLI: задать один вопрос выбранным ретривером и увидеть ответ + источники.

Примеры:
  python query.py --retriever fusion "Кто отвечает за проект X?"
  python query.py -r graph "Как связаны A и B?" --show-context
"""
import argparse

from llama_index.core.query_engine import RetrieverQueryEngine

import config  # noqa: F401  — инициализирует Settings (LLM/эмбеддинги)
from retrievers import BUILDERS, get_retriever


def run(retriever_name: str, question: str, show_context: bool = False):
    retriever = get_retriever(retriever_name)
    engine = RetrieverQueryEngine.from_args(retriever, llm=config.Settings.llm)
    response = engine.query(question)

    print(f"\n=== [{retriever_name}] {question} ===\n")
    print(response.response)

    if show_context:
        print("\n--- Источники ---")
        for i, node in enumerate(response.source_nodes, 1):
            score = f"{node.score:.3f}" if node.score is not None else "n/a"
            text = node.node.get_content()[:200].replace("\n", " ")
            print(f"[{i}] score={score}  {text}...")
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Вопрос по твоим документам")
    parser.add_argument(
        "-r", "--retriever", default="fusion", choices=list(BUILDERS),
        help="Какой ретривер использовать (по умолчанию fusion)",
    )
    parser.add_argument(
        "--show-context", action="store_true", help="Показать найденные источники",
    )
    args = parser.parse_args()
    run(args.retriever, args.question, args.show_context)


if __name__ == "__main__":
    main()
