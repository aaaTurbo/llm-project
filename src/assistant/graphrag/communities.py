"""Сообщества графа и их сводки. Требование FR-3."""

from collections import defaultdict

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)

    for source, target, attributes in graph.edges(data=True):
        weight = attributes.get("weight", 1)

        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    groups = nx.community.louvain_communities(
        undirected,
        weight="weight",
        seed = 42
    )

    ordered_groups = sorted(
        (sorted(group) for group in groups),
        key=lambda nodes: tuple(nodes),
    )

    return [
        Community(community_id, nodes)
        for community_id, nodes in enumerate(ordered_groups)
    ]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    nodes = set(community.nodes)

    node_lines = []
    for node_id in sorted(nodes):
        if node_id not in graph:
            continue

        attributes = graph.nodes[node_id]
        node_lines.append(
            f"- {attributes.get('type', '')}: "
            f"{attributes.get('name', node_id)}"
        )

    edge_lines = []
    totals: dict[tuple[str, str], int | float] = defaultdict(int)

    edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (edge[0], edge[1], edge[2].get("type", "")),
    )

    for source, target, attributes in edges:
        if source not in nodes or target not in nodes:
            continue

        relation_type = attributes.get("type", "UNKNOWN")
        weight = attributes.get("weight", 1)

        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        source_type = graph.nodes[source].get("type", "")
        target_type = graph.nodes[target].get("type", "")

        edge_lines.append(
            f"- {source_type} {source_name} "
            f"-[{relation_type}, вес {weight}]-> "
            f"{target_type} {target_name}"
        )

        totals[(relation_type, target)] += weight

    frequency_lines = []
    for (relation_type, target), total_weight in sorted(
            totals.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        target_name = graph.nodes[target].get("name", target)
        frequency_lines.append(
            f"- {relation_type} → {target_name}: "
            f"суммарный вес {total_weight}"
        )

    facts = (
            "Сущности:\n"
            + ("\n".join(node_lines) or "- нет")
            + "\n\nСвязи внутри сообщества и вес каждой связи:\n"
            + ("\n".join(edge_lines) or "- нет")
            + "\n\nЧастоты по типу связи и целевой сущности "
              "(суммы весов посчитаны кодом):\n"
            + ("\n".join(frequency_lines) or "- нет")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты составляешь краткую сводку сообщества графа знаний "
                "для сервисной службы. Опиши главные сущности, связи между "
                "ними и наиболее часто встречающиеся явления. Используй "
                "только переданные факты. Не придумывай сущности, связи "
                "и числа. Все числа уже подсчитаны кодом. Ответ должен "
                "быть не длиннее 150 слов."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Сообщество {community.community_id}.\n\n"
                f"{facts}\n\n"
                "Составь связную сводку на русском языке."
            ),
        },
    ]

    response = llm.chat(messages, max_tokens=512)
    return " ".join(response.content.split()[:150])
