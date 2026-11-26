"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0

    for result in results:
        source = result.chunk_id

        for entity in result.entities:
            node_id = entity.node_id

            if not graph.has_node(node_id):
                graph.add_node(
                    node_id,
                    type=entity.type,
                    name=entity.name,
                    sources=set(),
                )

            graph.nodes[node_id]["sources"].add(source)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            for entity in (relation.subject, relation.object):
                node_id = entity.node_id

                if not graph.has_node(node_id):
                    graph.add_node(
                        node_id,
                        type=entity.type,
                        name=entity.name,
                        sources=set(),
                    )

                graph.nodes[node_id]["sources"].add(source)

            source_id = relation.subject.node_id
            target_id = relation.object.node_id

            if not graph.has_edge(source_id, target_id):
                graph.add_edge(
                    source_id,
                    target_id,
                    type=relation.type,
                    weight=0,
                    sources=set(),
                )

            edge = graph.edges[source_id, target_id]
            edge["weight"] += 1 if source not in edge["sources"] else 0
            edge["sources"].add(source)

    checked_pairs = set()

    for source, target, data in list(graph.edges(data=True)):
        if data["type"] != "REPLACED_BY":
            continue

        pair = frozenset((source, target))
        if pair in checked_pairs:
            continue
        checked_pairs.add(pair)

        if not graph.has_edge(target, source):
            continue

        reverse = graph.edges[target, source]
        graph.graph["conflicts"] += 1

        if data["weight"] > reverse["weight"]:
            graph.remove_edge(target, source)
        elif data["weight"] < reverse["weight"]:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)

    for _, data in graph.nodes(data=True):
        data["sources"] = sorted(data["sources"])

    for _, _, data in graph.edges(data=True):
        data["sources"] = sorted(data["sources"])

    return graph
