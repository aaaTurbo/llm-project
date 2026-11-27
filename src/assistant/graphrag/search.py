"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations
from pathlib import Path

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Answer, KnowledgeBase, MAP_SCHEMA

REFUSED_TEXT = "В базе знаний не найдено."

PROMPTS_DIR = Path(__file__).with_name("prompts")


def _load_prompt(filename: str) -> str:
    """Загружает промпт из файла."""
    path = PROMPTS_DIR / filename

    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Файл промпта не найден: {path}"
        ) from exc


def _is_refusal(text: str) -> bool:
    """Проверяет, вернула ли модель стандартный отказ."""
    normalized = text.strip().rstrip(".!").casefold()
    expected = REFUSED_TEXT.rstrip(".").casefold()
    return normalized == expected


def _refused() -> Answer:
    return Answer(text=REFUSED_TEXT, refused=True)


def _document_sources(chunk_sources: list[str]) -> list[str]:
    """Преобразует идентификаторы чанков в идентификаторы документов."""
    return sorted({
        source.split("#", 1)[0]
        for source in chunk_sources
    })


def _collect_local_edges(
    question: str,
    kb: KnowledgeBase,
) -> set[tuple[str, str]]:
    """Выбирает рёбра, относящиеся к опорным сущностям вопроса."""

    graph = kb.graph
    codes = find_codes(question)
    names = find_names(question, kb.vocabulary)

    for code in codes:
        node_id = f"{code_type(code)}:{code}"
        if node_id not in graph:
            return set()

    anchors = []

    for code in codes:
        node_id = f"{code_type(code)}:{code}"
        if node_id not in anchors:
            anchors.append(node_id)

    for entity_type, name in names:
        node_id = f"{entity_type}:{name}"
        if node_id in graph and node_id not in anchors:
            anchors.append(node_id)

    if not anchors:
        return set()

    undirected = graph.to_undirected()
    selected_edges: set[tuple[str, str]] = set()
    selected_nodes: set[str] = set()

    if len(anchors) >= 2:
        for start, end in combinations(anchors, 2):
            if not nx.has_path(undirected, start, end):
                continue

            for path in nx.all_simple_paths(
                undirected, start, end, cutoff=3
            ):
                selected_nodes.update(path)

                for left, right in zip(path, path[1:]):
                    if graph.has_edge(left, right):
                        selected_edges.add((left, right))
                    if graph.has_edge(right, left):
                        selected_edges.add((right, left))

    else:
        anchor = anchors[0]

        distances = nx.single_source_shortest_path_length(
            undirected, anchor, cutoff=2
        )

        selected_nodes.update(distances)

        for source, target in graph.edges:
            if source in distances and target in distances:
                selected_edges.add((source, target))

    queue = [
        node_id
        for node_id in selected_nodes
        if graph.nodes[node_id].get("type") == "Part"
    ]
    visited_parts = set(queue)

    while queue:
        part_id = queue.pop(0)

        for _, target, attributes in graph.out_edges(
            part_id, data=True
        ):
            if attributes.get("type") != "REPLACED_BY":
                continue

            selected_edges.add((part_id, target))

            if target not in visited_parts:
                visited_parts.add(target)
                queue.append(target)

    return selected_edges

def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    selected_edges = _collect_local_edges(question, kb)

    if not selected_edges:
        return _refused()

    facts = []
    sources = set()

    for source, target in sorted(selected_edges):
        attributes = graph.edges[source, target]

        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        relation_type = attributes.get("type", "UNKNOWN")

        fact = f"{source_name} -{relation_type}-> {target_name}"

        candidate = f"{len(facts) + 1}. {fact}"
        current_length = sum(len(item) + 1 for item in facts)

        if current_length + len(candidate) > 8000:
            break

        facts.append(candidate)
        sources.update(attributes.get("sources", []))

    if not facts:
        return _refused()

    facts_text = "\n".join(facts)

    messages = [
        {
            "role": "system",
            "content": _load_prompt("local_search.txt"),
        },
        {
            "role": "user",
            "content": (
                f"Вопрос: {question}\n\n"
                f"Факты графа знаний:\n{facts_text}\n\n"
                "Ответь на вопрос по этим фактам."
            ),
        },
    ]

    response = llm.chat(messages, max_tokens=1024)
    answer_text = response.content.strip()

    if not answer_text or _is_refusal(answer_text):
        return Answer(
            text=REFUSED_TEXT,
            facts=[item.split(". ", 1)[1] for item in facts],
            sources=_document_sources(list(sources)),
            refused=True,
        )

    return Answer(
        text=answer_text,
        facts=[item.split(". ", 1)[1] for item in facts],
        sources=_document_sources(list(sources)),
        refused=False,
    )


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partial_answers = []
    sources = set()

    map_prompt = _load_prompt("global_map.txt")
    reduce_prompt = _load_prompt("global_reduce.txt")

    for community in kb.communities:
        if not community.summary.strip():
            continue

        messages = [
            {
                "role": "system",
                "content": map_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    f"Сводка сообщества {community.community_id}:\n"
                    f"{community.summary}"
                ),
            },
        ]

        response = llm.chat(
            messages,
            json_schema=MAP_SCHEMA,
            max_tokens=512,
        )

        try:
            result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            continue

        answer = result.get("answer")
        score = result.get("score")

        if (
                not isinstance(answer, str)
                or type(score) is not int
                or not 0 <= score <= 10
                or score == 0
                or not answer.strip()
        ):
            continue

        partial_answers.append(
            f"Сообщество {community.community_id} "
            f"(оценка {score}/10): {answer.strip()}"
        )

        community_nodes = set(community.nodes)

        for node_id in community.nodes:
            if node_id in kb.graph:
                sources.update(
                    kb.graph.nodes[node_id].get("sources", [])
                )

        for source, target, attributes in kb.graph.edges(data=True):
            if source in community_nodes and target in community_nodes:
                sources.update(attributes.get("sources", []))

    if not partial_answers:
        return _refused()

    messages = [
        {
            "role": "system",
            "content": reduce_prompt,
        },
        {
            "role": "user",
            "content": (
                    f"Вопрос: {question}\n\n"
                    "Частичные ответы:\n"
                    + "\n".join(partial_answers)
            ),
        },
    ]

    response = llm.chat(messages, max_tokens=1024)
    answer_text = response.content.strip()

    if not answer_text or _is_refusal(answer_text):
        return Answer(
            text=REFUSED_TEXT,
            sources=_document_sources(list(sources)),
            refused=True,
        )

    return Answer(
        text=answer_text,
        sources=_document_sources(list(sources)),
        refused=False,
    )
