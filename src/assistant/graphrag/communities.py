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
        seed=42,
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
    model_failures: dict[str, int] = defaultdict(int)
    model_failure_details: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    component_failures: dict[str, int] = defaultdict(int)
    model_lines: dict[str, set[str]] = defaultdict(set)
    model_parts: dict[str, set[str]] = defaultdict(set)
    model_part_chains: dict[str, set[str]] = defaultdict(set)
    replacements: set[tuple[str, str]] = set()
    suppliers: set[tuple[str, str]] = set()
    totals: dict[tuple[str, str], int | float] = defaultdict(int)

    edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (edge[0], edge[1], edge[2].get("type", "")),
    )

    for source, target, attributes in edges:
        if source not in nodes and target not in nodes:
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

        if source in nodes:
            totals[(relation_type, target)] += weight

        if source in nodes and relation_type == "FAILED" and source_type == "Model":
            model_failures[source_name] += weight
            model_failure_details[source_name][target_name] += weight
            component_failures[target_name] += weight

        if source in nodes and relation_type == "BELONGS_TO" and source_type == "Model":
            model_lines[source_name].add(target_name)

        if source in nodes and relation_type == "HAS_PART" and source_type == "Model":
            model_parts[source_name].add(target_name)
            chain = [target]
            visited = {target}
            current = target
            while True:
                next_parts = sorted(
                    replacement_target
                    for _, replacement_target, replacement_attributes in graph.out_edges(current, data=True)
                    if replacement_attributes.get("type") == "REPLACED_BY"
                    and replacement_target not in visited
                )
                if not next_parts:
                    break
                current = next_parts[0]
                chain.append(current)
                visited.add(current)
            chain_names = [str(graph.nodes[item].get("name", item)) for item in chain]
            model_part_chains[source_name].add(" -> ".join(chain_names))
            for old_part, new_part in zip(chain, chain[1:]):
                replacements.add((
                    str(graph.nodes[old_part].get("name", old_part)),
                    str(graph.nodes[new_part].get("name", new_part)),
                ))
            for chain_part in chain:
                for _, supplier_id, supplier_attributes in graph.out_edges(chain_part, data=True):
                    if supplier_attributes.get("type") == "SUPPLIED_BY":
                        suppliers.add((
                            str(graph.nodes[chain_part].get("name", chain_part)),
                            str(graph.nodes[supplier_id].get("name", supplier_id)),
                        ))

        if source in nodes and relation_type == "REPLACED_BY" and source_type == "Part":
            replacements.add((source_name, target_name))

        if source in nodes and relation_type == "SUPPLIED_BY" and source_type == "Part":
            suppliers.add((source_name, target_name))

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

    model_failure_lines = []
    for model_name, total in sorted(
        model_failures.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lines = ", ".join(sorted(model_lines.get(model_name, set()))) or "линейка не указана"
        details = ", ".join(
            f"{component}: {count}"
            for component, count in sorted(
                model_failure_details[model_name].items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        model_failure_lines.append(
            f"- {model_name} ({lines}): {total} отказов; {details}"
        )

    component_failure_lines = [
        f"- {component}: {count}"
        for component, count in sorted(
            component_failures.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    line_model_lines = [
        f"- {model}: {', '.join(sorted(lines))}"
        for model, lines in sorted(model_lines.items())
    ]
    part_lines = [
        f"- {model}: {', '.join(sorted(parts))}"
        for model, parts in sorted(model_parts.items())
    ]
    part_chain_lines = [
        f"- {model}: {chain}"
        for model, chains in sorted(model_part_chains.items())
        for chain in sorted(chains)
    ]
    replacement_lines = [
        f"- {old_part} -> {new_part}"
        for old_part, new_part in sorted(replacements)
    ]
    supplier_lines = [
        f"- {part}: {supplier}"
        for part, supplier in sorted(suppliers)
    ]

    facts = (
        "Сущности сообщества:\n"
        + ("\n".join(node_lines) or "- нет")
        + "\n\nПодсчитанные отказы по моделям. Числа равны сумме весов исходящих связей FAILED:\n"
        + ("\n".join(model_failure_lines) or "- нет")
        + "\n\nПодсчитанные отказы по компонентам:\n"
        + ("\n".join(component_failure_lines) or "- нет")
        + "\n\nМодели и их линейки:\n"
        + ("\n".join(line_model_lines) or "- нет")
        + "\n\nЦепочки деталей каждой модели:\n"
        + ("\n".join(part_chain_lines) or "- нет")
        + "\n\nПоставщики деталей:\n"
        + ("\n".join(supplier_lines) or "- нет")
        + "\n\nДетали моделей:\n"
        + ("\n".join(part_lines) or "- нет")
        + "\n\nЦепочки замен; левая деталь заменена правой:\n"
        + ("\n".join(replacement_lines) or "- нет")
        + "\n\nСвязи сообщества и соседние связи:\n"
        + ("\n".join(edge_lines[:100]) or "- нет")
        + "\n\nСуммарные веса остальных связей:\n"
        + ("\n".join(frequency_lines) or "- нет")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты составляешь краткую сводку сообщества графа знаний "
                "для сервисной службы. Сохраняй коды моделей, деталей, "
                "названия линеек, компонентов и поставщиков. Обязательно "
                "сохраняй строки с подсчитанными отказами по моделям и "
                "компонентам, включая все числа. Не объединяй разные модели "
                "в одну. Связь REPLACED_BY означает, что левая деталь "
                "заменена правой. Используй только переданные факты. "
                "Все числа уже подсчитаны кодом. Ответ должен быть "
                "не длиннее 150 слов."
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
