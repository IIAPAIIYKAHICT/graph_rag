"""Прогон набора вопросов по всем ретриверам + сравнительный отчёт.

Для каждого вопроса из questions.yaml собирает ответ каждым ретривером
(асинхронно, параллельно), замеряет latency и число источников, и опционально
считает RAGAS-метрики качества (faithfulness / answer relevancy).

Судья RAGAS — та же модель Claude и та же локальная модель эмбеддингов, что и в
пайплайне (см. config.py), поэтому OpenAI-ключ не нужен.

Запуск:  python evaluate.py
         python evaluate.py --retrievers vector fusion    # только эти
         python evaluate.py --concurrency 8               # больше параллелизма
         python evaluate.py --no-ragas                    # только скорость/источники
"""
import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import yaml
from llama_index.core.query_engine import RetrieverQueryEngine

import config
from retrievers import BUILDERS, get_retriever

HERE = Path(__file__).parent

# input-колонки RAGAS-датасета — всё остальное в to_pandas() считаем метриками
RAGAS_INPUT_COLS = {"user_input", "response", "retrieved_contexts", "reference"}


def load_questions():
    with open(HERE / "questions.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def _is_nan(x) -> bool:
    try:
        return math.isnan(x)
    except TypeError:
        return x is None


# --- Сбор ответов (async) ---------------------------------------------------

async def _answer(engine, name, q, sem):
    """Один вопрос одним ретривером; семафор ограничивает параллелизм."""
    async with sem:
        t0 = time.perf_counter()
        resp = await engine.aquery(q["text"])
        dt = time.perf_counter() - t0
    return {
        "question": q["text"],
        "type": q.get("type", "n/a"),
        "reference": q.get("reference"),
        "retriever": name,
        "latency_s": round(dt, 2),
        "n_sources": len(resp.source_nodes),
        "answer_len": len(resp.response or ""),
        "answer": resp.response,
        # контексты нужны RAGAS; в markdown их не печатаем
        "contexts": [n.node.get_content() for n in resp.source_nodes],
    }


async def collect_rows(retriever_names, concurrency):
    """Гоняет всю сетку (вопрос × ретривер) параллельно, порядок строк сохраняет."""
    # строим каждый ретривер один раз и переиспользуем
    engines = {
        name: RetrieverQueryEngine.from_args(get_retriever(name), llm=config.Settings.llm)
        for name in retriever_names
    }
    questions = load_questions()
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _answer(engine, name, q, sem)
        for q in questions
        for name, engine in engines.items()
    ]
    rows = await asyncio.gather(*tasks)
    for r in rows:  # печатаем по порядку уже после сбора
        print(f"[{r['retriever']:12}] ({r['latency_s']:4.1f}s) {r['question'][:60]}")
    return rows


# --- RAGAS ------------------------------------------------------------------

def add_ragas_scores(rows):
    """Досчитывает метрики качества RAGAS, дописывает их в rows.

    Возвращает (rows, metric_cols). Если RAGAS не установлен или упал — мягко
    пропускает и возвращает пустой список метрик, чтобы не потерять отчёт по
    скорости.
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError:
        print(
            "\nRAGAS не установлен — пропускаю метрики качества "
            "(pip install 'ragas>=0.2,<0.3'). Останутся latency/источники."
        )
        return rows, []

    # RAGAS передаёт judge-модели параметр temperature, а новые модели Claude
    # (sonnet-5/opus-5/…) его отклоняют (400) -> все метрики становятся n/a.
    # Поэтому судью держим на модели, принимающей temperature; пайплайн при этом
    # может работать на любой модели. Переопределяется через RAGAS_JUDGE_MODEL.
    from llama_index.llms.anthropic import Anthropic as _JudgeLLM
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "claude-haiku-4-5")
    judge = LlamaIndexLLMWrapper(_JudgeLLM(model=judge_model, max_tokens=1024))
    emb = LlamaIndexEmbeddingsWrapper(config.Settings.embed_model)

    # reference есть → добавляем context_recall: «достал ли ретривер контекст,
    # которого хватает на эталонный ответ» — самая наглядная метрика для сравнения
    # ретриверов (особенно на multi-hop).
    has_reference = any((r.get("reference") or "").strip() for r in rows)

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"] or "",
                "retrieved_contexts": r["contexts"] or [""],
                "reference": r.get("reference") or "",
            }
            for r in rows
        ]
    )

    metrics = [faithfulness, answer_relevancy]
    if has_reference:
        metrics.append(context_recall)

    print(
        f"\nRAGAS: считаю {len(metrics)} метрик по {len(rows)} ответам "
        "(это доп. вызовы Claude — небыстро)..."
    )
    try:
        # llm/embeddings прокидываются во все метрики, которым они нужны
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge,
            embeddings=emb,
        )
        df = result.to_pandas()
    except Exception as e:  # noqa: BLE001 — не роняем отчёт из-за судьи
        print(f"RAGAS упал ({type(e).__name__}: {e}) — отчёт без метрик качества.")
        return rows, []

    metric_cols = [c for c in df.columns if c not in RAGAS_INPUT_COLS]
    for i, r in enumerate(rows):
        if i >= len(df):
            break
        for c in metric_cols:
            val = df.iloc[i][c]
            r[c] = None if _is_nan(val) else round(float(val), 3)
    return rows, metric_cols


# --- Отчёты -----------------------------------------------------------------

def write_reports(rows, metric_cols):
    (HERE / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = ["# Сравнение ретриверов\n"]

    # сводка качества по ретриверу (среднее метрик RAGAS)
    if metric_cols:
        lines.append("## Качество (RAGAS, судья — Claude)\n")
        lines.append("Среднее по ретриверу, шкала 0..1 (выше — лучше):\n")
        lines.append("| Ретривер | " + " | ".join(metric_cols) + " |")
        lines.append("|----------|" + "|".join(["-----:"] * len(metric_cols)) + "|")
        for name in sorted({r["retriever"] for r in rows}):
            cells = []
            for c in metric_cols:
                vals = [
                    r[c] for r in rows
                    if r["retriever"] == name and r.get(c) is not None
                ]
                cells.append(f"{sum(vals) / len(vals):.3f}" if vals else "n/a")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    # сводная таблица по скорости/источникам
    lines.append("## Скорость и покрытие\n")
    lines.append("| Вопрос | Тип | Ретривер | Latency, с | Источников | Длина ответа |")
    lines.append("|--------|-----|----------|-----------:|-----------:|-------------:|")
    for r in rows:
        q = r["question"][:40]
        lines.append(
            f"| {q} | {r['type']} | {r['retriever']} | "
            f"{r['latency_s']} | {r['n_sources']} | {r['answer_len']} |"
        )

    # полные ответы, сгруппированные по вопросу
    lines.append("\n## Ответы\n")
    seen = []
    for r in rows:
        if r["question"] not in seen:
            seen.append(r["question"])
            lines.append(f"\n### {r['question']}  _({r['type']})_\n")
        meta = f"{r['latency_s']}s, {r['n_sources']} ист."
        for c in metric_cols:
            v = r.get(c)
            meta += f", {c}={'n/a' if v is None else v}"
        lines.append(f"**{r['retriever']}** — {meta}\n")
        lines.append(f"> {r['answer']}\n")

    (HERE / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nОтчёты: {HERE / 'results.md'}  и  results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrievers", nargs="+", default=list(BUILDERS), choices=list(BUILDERS),
        help="Какие ретриверы сравнивать (по умолчанию все).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Сколько запросов гнать параллельно (осторожно с rate limit Claude).",
    )
    parser.add_argument(
        "--no-ragas", action="store_true",
        help="Не считать RAGAS-метрики — только скорость/источники.",
    )
    args = parser.parse_args()

    rows = asyncio.run(collect_rows(args.retrievers, args.concurrency))
    metric_cols = []
    if not args.no_ragas:
        rows, metric_cols = add_ragas_scores(rows)
    write_reports(rows, metric_cols)


if __name__ == "__main__":
    main()
